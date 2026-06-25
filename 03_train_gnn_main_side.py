#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torch_geometric is required. Install PyTorch Geometric first.\n"
        "See: https://pytorch-geometric.readthedocs.io/\n"
        f"Original import error: {exc}"
    )


LABEL_TO_ID = {"side_branch": 0, "main_stem": 1}
ID_TO_LABEL = {0: "side_branch", 1: "main_stem"}


@dataclass
class GraphBuildConfig:
    edge_radius_norm: float = 0.12
    knn_k: int = 3
    include_leaf_context: bool = True


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def axis_or_bbox_bottom_to_top(rec: dict[str, Any]) -> list[float]:
    axis = rec.get("axis_xyxy_bottom_to_top")
    if axis is not None:
        return [float(v) for v in axis]
    x1, y1, x2, y2 = [float(v) for v in rec["bbox_xyxy"]]
    cx = (x1 + x2) * 0.5
    return [cx, y2, cx, y1]


def normalize_axis(axis: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = axis
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def normalize_point(point: tuple[float, float] | list[float], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) / width, float(point[1]) / height


def point_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def farthest_polygon_points(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if poly is None or len(poly) < 2:
        return None
    points = poly.astype(np.float32)
    deltas = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(deltas * deltas, axis=2)
    flat_idx = int(np.argmax(dist2))
    i, j = np.unravel_index(flat_idx, dist2.shape)
    return points[i].copy(), points[j].copy()


def line_polygon_intersection_parameters(
    polygon: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> list[float]:
    params: list[float] = []
    eps = 1e-6
    for idx in range(len(polygon)):
        point_a = polygon[idx].astype(np.float32)
        point_b = polygon[(idx + 1) % len(polygon)].astype(np.float32)
        edge = point_b - point_a
        denom = float(direction[0] * edge[1] - direction[1] * edge[0])
        if abs(denom) <= eps:
            continue
        diff = point_a - origin
        t = float((diff[0] * edge[1] - diff[1] * edge[0]) / denom)
        u = float((diff[0] * direction[1] - diff[1] * direction[0]) / denom)
        if -eps <= u <= 1.0 + eps:
            params.append(t)
    return sorted(params)


def leaf_scalar_point(rec: dict[str, Any]) -> tuple[float, float]:
    poly = np.asarray(rec.get("polygon_xy", []), dtype=np.float32)
    if len(poly) < 3:
        return tuple(float(v) for v in rec.get("centroid_xy", [0.0, 0.0]))

    endpoints = farthest_polygon_points(poly)
    if endpoints is None:
        return tuple(float(v) for v in rec.get("centroid_xy", [0.0, 0.0]))

    start, end = endpoints
    axis = end - start
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 1e-6:
        return tuple(float(v) for v in rec.get("centroid_xy", [0.0, 0.0]))

    axis_dir = axis / axis_len
    perp_dir = np.asarray([-axis_dir[1], axis_dir[0]], dtype=np.float32)
    best_point = (start + end) * 0.5
    best_width = -1.0

    # 잎은 줄기처럼 방향이 중요하지 않아서, 가장 두꺼운 지점을 대표점으로 씁니다.
    for ratio in np.linspace(0.05, 0.95, 25):
        point = start + axis * float(ratio)
        if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) < -0.5:
            continue
        params = line_polygon_intersection_parameters(poly, point, perp_dir)
        if len(params) < 2:
            continue
        for left_t, right_t in zip(params[0::2], params[1::2]):
            width = abs(right_t - left_t)
            if width > best_width:
                best_width = width
                best_point = point

    return float(best_point[0]), float(best_point[1])


def point_to_segment_distance_norm(point: tuple[float, float], axis_norm: list[float]) -> float:
    px, py = point
    ax, ay, bx, by = axis_norm
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return point_dist(point, (ax, ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    projection = (ax + t * abx, ay + t * aby)
    return point_dist(point, projection)


def stem_endpoint_distance(node_a: dict[str, Any], node_b: dict[str, Any]) -> float:
    axis_a = node_a["stem_axis_norm"]
    axis_b = node_b["stem_axis_norm"]
    points_a = [(axis_a[0], axis_a[1]), (axis_a[2], axis_a[3])]
    points_b = [(axis_b[0], axis_b[1]), (axis_b[2], axis_b[3])]
    return min(point_dist(pa, pb) for pa in points_a for pb in points_b)


def centroid_distance(node_a: dict[str, Any], node_b: dict[str, Any]) -> float:
    ca = node_a["centroid_norm"]
    cb = node_b["centroid_norm"]
    return point_dist(ca, cb)


def node_distance(node_a: dict[str, Any], node_b: dict[str, Any]) -> float:
    kind_a = node_a["kind"]
    kind_b = node_b["kind"]
    if kind_a == "leaf" and kind_b == "leaf":
        return float("inf")
    if kind_a == "stem" and kind_b == "stem":
        return min(stem_endpoint_distance(node_a, node_b), centroid_distance(node_a, node_b))

    stem_node = node_a if kind_a == "stem" else node_b
    leaf_node = node_b if kind_a == "stem" else node_a
    return min(
        point_to_segment_distance_norm(leaf_node["leaf_point_norm"], stem_node["stem_axis_norm"]),
        centroid_distance(stem_node, leaf_node),
    )


def build_edges(nodes: list[dict[str, Any]], edge_radius_norm: float, knn_k: int) -> torch.Tensor:
    n = len(nodes)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    undirected: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if node_distance(nodes[i], nodes[j]) <= edge_radius_norm:
                undirected.add((i, j))

    # 가까운 애들만 잇되, 그래프가 끊기면 주변 k개를 더 붙여 줍니다.
    if knn_k > 0:
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                d = node_distance(nodes[i], nodes[j])
                if not math.isfinite(d):
                    continue
                dists.append((d, j))
            for _, j in sorted(dists)[:knn_k]:
                a, b = sorted((i, j))
                undirected.add((a, b))

    directed = []
    for i, j in sorted(undirected):
        directed.append((i, j))
        directed.append((j, i))

    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def load_label_map(label_path: Path) -> dict[int, int]:
    payload = load_json(label_path)
    label_map: dict[int, int] = {}
    for item in payload.get("labels", []):
        stem_id = int(item["stem_id"])
        label = str(item["label"]).strip()
        if label not in LABEL_TO_ID:
            continue
        label_map[stem_id] = LABEL_TO_ID[label]
    return label_map


def build_graph(segment_path: Path, label_path: Path, cfg: GraphBuildConfig) -> Data | None:
    seg = load_json(segment_path)
    labels = load_label_map(label_path)
    width = int(seg["width"])
    height = int(seg["height"])

    nodes: list[dict[str, Any]] = []
    for rec in seg["segments"]:
        kind = rec["kind"]
        if kind == "leaf" and not cfg.include_leaf_context:
            continue
        if kind not in {"stem", "leaf"}:
            continue

        axis = axis_or_bbox_bottom_to_top(rec)
        cx, cy = rec.get("centroid_xy", [(axis[0] + axis[2]) * 0.5, (axis[1] + axis[3]) * 0.5])
        centroid_norm = (float(cx) / width, float(cy) / height)
        is_stem = 1.0 if kind == "stem" else 0.0
        is_leaf = 1.0 if kind == "leaf" else 0.0
        if kind == "stem":
            stem_axis_norm = normalize_axis(axis, width, height)
            leaf_point_norm = (0.0, 0.0)
        else:
            stem_axis_norm = [0.0, 0.0, 0.0, 0.0]
            leaf_point_norm = normalize_point(leaf_scalar_point(rec), width, height)
        # stem은 긴 축 벡터, leaf는 대표점 하나만 넣어서 서로 다른 모양 정보를 나눠 줍니다.
        x = stem_axis_norm + [leaf_point_norm[0], leaf_point_norm[1], is_stem, is_leaf]

        y = -100
        stem_id = rec.get("stem_id")
        if kind == "stem" and stem_id is not None and int(stem_id) in labels:
            y = labels[int(stem_id)]

        nodes.append({
            "x": x,
            "y": y,
            "kind": kind,
            "segment_id": int(rec["segment_id"]),
            "stem_id": None if stem_id is None else int(stem_id),
            "stem_axis_norm": stem_axis_norm,
            "leaf_point_norm": leaf_point_norm,
            "centroid_norm": centroid_norm,
        })

    if not nodes:
        return None

    if not any(n["y"] >= 0 for n in nodes):
        return None

    x = torch.tensor([n["x"] for n in nodes], dtype=torch.float32)
    y = torch.tensor([n["y"] for n in nodes], dtype=torch.long)
    train_mask = y >= 0
    edge_index = build_edges(nodes, cfg.edge_radius_norm, cfg.knn_k)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
    data.image_id = seg["image_id"]
    data.segment_path = str(segment_path)
    data.label_path = str(label_path)
    data.node_meta = [
        {
            "kind": n["kind"],
            "segment_id": n["segment_id"],
            "stem_id": n["stem_id"],
            "stem_axis_norm": n["stem_axis_norm"],
            "leaf_point_norm": n["leaf_point_norm"],
        }
        for n in nodes
    ]
    return data


def collect_graphs(dataset_root: Path, cfg: GraphBuildConfig, progress_every: int = 25) -> list[Data]:
    seg_dir = dataset_root / "segments"
    label_dir = dataset_root / "llm_labels"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Missing segments dir: {seg_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing llm_labels dir: {label_dir}")

    graphs = []
    seg_paths = sorted(seg_dir.glob("*.json"))
    start_time = time.perf_counter()
    print(f"Building GNN graphs on CPU: segment_jsons={len(seg_paths)}")
    for idx, seg_path in enumerate(seg_paths, start=1):
        label_path = label_dir / f"{seg_path.stem}.json"
        if not label_path.exists():
            continue
        g = build_graph(seg_path, label_path, cfg)
        if g is not None:
            graphs.append(g)
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(seg_paths)):
            elapsed = time.perf_counter() - start_time
            print(
                f"  graph_build {idx}/{len(seg_paths)} "
                f"usable={len(graphs)} elapsed={elapsed:.1f}s"
            )
    print(f"Built {len(graphs)} graph(s) in {time.perf_counter() - start_time:.1f}s")
    return graphs


class StemBranchGNN(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual
        return self.head(h)


def compute_class_weights(graphs: list[Data], device: torch.device) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.float32)
    for g in graphs:
        mask = g.train_mask
        yy = g.y[mask]
        for c in [0, 1]:
            counts[c] += float((yy == c).sum())
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    tp = torch.zeros(2, dtype=torch.float64)
    fp = torch.zeros(2, dtype=torch.float64)
    fn = torch.zeros(2, dtype=torch.float64)
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            mask = batch.train_mask
            if int(mask.sum()) == 0:
                continue
            loss = F.cross_entropy(logits[mask], batch.y[mask])
            loss_sum += float(loss.item()) * int(mask.sum())
            pred = logits[mask].argmax(dim=-1)
            target = batch.y[mask]
            total += int(target.numel())
            correct += int((pred == target).sum())
            for c in [0, 1]:
                tp[c] += ((pred == c) & (target == c)).sum().cpu()
                fp[c] += ((pred == c) & (target != c)).sum().cpu()
                fn[c] += ((pred != c) & (target == c)).sum().cpu()

    acc = correct / max(total, 1)
    f1s = []
    for c in [0, 1]:
        precision = tp[c] / max(tp[c] + fp[c], torch.tensor(1.0))
        recall = tp[c] / max(tp[c] + fn[c], torch.tensor(1.0))
        f1 = 2 * precision * recall / max(precision + recall, torch.tensor(1.0e-12))
        f1s.append(float(f1))
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": acc,
        "macro_f1": sum(f1s) / len(f1s),
        "side_branch_f1": f1s[0],
        "main_stem_f1": f1s[1],
        "num_labeled_nodes": float(total),
    }


def split_graphs(graphs: list[Data], val_ratio: float, seed: int) -> tuple[list[Data], list[Data]]:
    graphs = list(graphs)
    rng = random.Random(seed)
    rng.shuffle(graphs)
    if len(graphs) == 1:
        return graphs, graphs
    n_val = max(1, int(round(len(graphs) * val_ratio)))
    n_val = min(n_val, len(graphs) - 1)
    return graphs[n_val:], graphs[:n_val]


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    build_cfg = GraphBuildConfig(
        edge_radius_norm=args.edge_radius,
        knn_k=args.knn_k,
        include_leaf_context=not args.no_leaf_context,
    )
    graphs = collect_graphs(args.dataset, build_cfg, progress_every=args.progress_every)
    if not graphs:
        raise ValueError("No trainable graphs found. Run 01 and 02 first.")

    train_graphs, val_graphs = split_graphs(graphs, args.val_ratio, args.seed)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training device: {device}")
    model = StemBranchGNN(in_dim=8, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = compute_class_weights(train_graphs, device)

    best_macro_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)
            mask = batch.train_mask
            if int(mask.sum()) == 0:
                continue
            loss = F.cross_entropy(logits[mask], batch.y[mask], weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(mask.sum())
            total_nodes += int(mask.sum())

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
                f"val_acc={val_metrics['accuracy']:.3f} val_macro_f1={val_metrics['macro_f1']:.3f} "
                f"val_main_f1={val_metrics['main_stem_f1']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_dim": 8,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "edge_radius_norm": args.edge_radius,
                "knn_k": args.knn_k,
                "include_leaf_context": not args.no_leaf_context,
            },
            "label_map": ID_TO_LABEL,
        },
        args.checkpoint,
    )
    save_json(args.dataset / "gnn_training_history.json", {"history": history})
    print(f"Saved checkpoint: {args.checkpoint}")

    write_predictions(model, graphs, device, args.dataset / "gnn_predictions.json")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> StemBranchGNN:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    checkpoint_in_dim = int(cfg.get("in_dim", 8))
    if checkpoint_in_dim != 8:
        raise ValueError(
            f"Checkpoint input dimension is {checkpoint_in_dim}, but this script now uses 8D "
            "stem-vector/leaf-point features. Re-train the GNN checkpoint with this updated script."
        )
    model = StemBranchGNN(
        in_dim=checkpoint_in_dim,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 3)),
        dropout=float(cfg.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def write_predictions(model: nn.Module, graphs: list[Data], device: torch.device, out_path: Path) -> None:
    model.eval()
    results = []
    with torch.no_grad():
        for g in graphs:
            gg = g.to(device)
            logits = model(gg.x, gg.edge_index)
            prob = logits.softmax(dim=-1).cpu().numpy()
            pred = prob.argmax(axis=-1)
            items = []
            for idx, meta in enumerate(g.node_meta):
                if meta["kind"] != "stem":
                    continue
                items.append({
                    "segment_id": meta["segment_id"],
                    "stem_id": meta["stem_id"],
                    "pred_label": ID_TO_LABEL[int(pred[idx])],
                    "prob_side_branch": float(prob[idx, 0]),
                    "prob_main_stem": float(prob[idx, 1]),
                    "target_label": None if int(g.y[idx]) < 0 else ID_TO_LABEL[int(g.y[idx])],
                })
            results.append({"image_id": g.image_id, "stems": items})
    save_json(out_path, {"schema_version": "plant-gnn-predictions-v1", "results": results})
    print(f"Saved predictions: {out_path}")


def predict_only(args) -> None:
    build_cfg = GraphBuildConfig(
        edge_radius_norm=args.edge_radius,
        knn_k=args.knn_k,
        include_leaf_context=not args.no_leaf_context,
    )
    graphs = collect_graphs(args.dataset, build_cfg, progress_every=args.progress_every)
    if not graphs:
        raise ValueError("No graphs found.")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Prediction device: {device}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    write_predictions(model, graphs, device, args.dataset / "gnn_predictions.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("plant_dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("plant_dataset/gnn_model.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-radius", type=float, default=0.12, help="Normalized radius for graph edges")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--no-leaf-context", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25, help="Print CPU graph-building progress every N segment JSON files. Use 0 to disable.")
    parser.add_argument("--predict-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_only:
        predict_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
