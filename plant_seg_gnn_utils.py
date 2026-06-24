from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except Exception as exc:  # pragma: no cover
    SAGEConv = None
    PYG_IMPORT_ERROR = exc
else:
    PYG_IMPORT_ERROR = None


SAM3_PROMPTS = ["stem", "leaf"]
RAW_STEM = 0
RAW_LEAF = 1

DEMO_LEAF = 0
DEMO_STEM = 1
DEMO_PATH = 2
DEMO_CLASS_NAMES = ["leaf", "stem", "path"]
YOLO_PSEUDO_CLASS_NAMES = ["leaf", "stem"]

LABEL_TO_ID = {"side_branch": 0, "main_stem": 1}
ID_TO_LABEL = {0: "side_branch", 1: "main_stem"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_device(device: str = "") -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def load_sam3_predictor(
    sam3_model_path: str | Path,
    conf: float,
    iou: float,
    imgsz: int,
    half: bool,
    device: str,
):
    from ultralytics.models.sam import SAM3SemanticPredictor

    resolved_device = resolve_device(device)
    use_half = bool(half and resolved_device == "cuda")
    overrides = {
        "conf": conf,
        "iou": iou,
        "task": "segment",
        "mode": "predict",
        "model": str(sam3_model_path),
        "imgsz": imgsz,
        "half": use_half,
        "device": resolved_device,
        "save": False,
        "verbose": False,
    }
    return SAM3SemanticPredictor(overrides=overrides)


def resample_closed_polygon(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0 or len(points) == target_count:
        return points.astype(np.float32)

    closed = np.vstack([points, points[0]])
    edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(edge_lengths.sum())
    if perimeter <= 1e-6:
        return points.astype(np.float32)

    distances = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    sample_distances = np.linspace(0.0, perimeter, target_count, endpoint=False)
    sampled = []
    edge_idx = 0
    for distance in sample_distances:
        while edge_idx < len(edge_lengths) - 1 and distances[edge_idx + 1] < distance:
            edge_idx += 1
        start = closed[edge_idx]
        end = closed[edge_idx + 1]
        edge_length = max(float(edge_lengths[edge_idx]), 1e-6)
        ratio = (distance - distances[edge_idx]) / edge_length
        sampled.append(start + ratio * (end - start))
    return np.asarray(sampled, dtype=np.float32)


def mask_to_polygon(
    mask: np.ndarray,
    min_area_px: int,
    approx_epsilon_ratio: float,
    num_vertices: int,
) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area_px]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 3:
        return None

    perimeter = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    epsilon = max(1.0, perimeter * approx_epsilon_ratio)
    polygon = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2).astype(np.float32)
    if len(polygon) < 3:
        polygon = contour
    if num_vertices and num_vertices >= 3:
        polygon = resample_closed_polygon(polygon, num_vertices)
    return polygon.astype(np.float32)


def farthest_polygon_points(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = polygon.astype(np.float32)
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sum(deltas * deltas, axis=2)
    flat_idx = int(np.argmax(distances))
    start_idx, end_idx = np.unravel_index(flat_idx, distances.shape)
    return points[start_idx].copy(), points[end_idx].copy()


def bottom_to_top_axis(polygon: np.ndarray) -> list[float] | None:
    if polygon is None or len(polygon) < 2:
        return None
    point_a, point_b = farthest_polygon_points(polygon)
    # GNN 학습 스크립트와 동일하게 y가 큰 점을 bottom, 작은 점을 top으로 둡니다.
    if point_a[1] >= point_b[1]:
        bottom, top = point_a, point_b
    else:
        bottom, top = point_b, point_a
    return [float(bottom[0]), float(bottom[1]), float(top[0]), float(top[1])]


def polygon_centroid(polygon: np.ndarray, mask: np.ndarray) -> list[float]:
    moments = cv2.moments(polygon.astype(np.float32))
    if moments["m00"]:
        return [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def assign_segment_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stem_id = 1
    for segment_id, record in enumerate(records):
        record["segment_id"] = segment_id
        if record["kind"] == "stem":
            record["stem_id"] = stem_id
            stem_id += 1
        else:
            record["stem_id"] = None
    return records


def records_from_result(
    result,
    image_shape_hw: tuple[int, int],
    min_area_px: int,
    max_area_ratio: float,
    approx_epsilon_ratio: float,
    num_vertices: int,
) -> list[dict[str, Any]]:
    if result.masks is None or result.boxes is None:
        return []

    image_h, image_w = image_shape_hw
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    records = []
    for idx, mask_data in enumerate(masks):
        raw_class_id = int(class_ids[idx])
        if raw_class_id not in {RAW_STEM, RAW_LEAF}:
            continue

        mask = mask_data > 0.5
        if mask.shape[:2] != (image_h, image_w):
            mask = cv2.resize(mask.astype(np.uint8), (image_w, image_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        area_px = float(mask.sum())
        if area_px < min_area_px:
            continue
        if area_px / float(image_h * image_w) > max_area_ratio:
            continue

        polygon = mask_to_polygon(mask, min_area_px, approx_epsilon_ratio, num_vertices)
        if polygon is None:
            continue

        x1, y1, x2, y2 = [float(value) for value in boxes[idx].tolist()]
        kind = "stem" if raw_class_id == RAW_STEM else "leaf"
        records.append(
            {
                "segment_id": -1,
                "kind": kind,
                "raw_class_id": raw_class_id,
                "raw_class_name": SAM3_PROMPTS[raw_class_id],
                "confidence": float(confidences[idx]),
                "area_px": area_px,
                "bbox_xyxy": [x1, y1, x2, y2],
                "centroid_xy": polygon_centroid(polygon, mask),
                "polygon_xy": [[float(x), float(y)] for x, y in polygon.tolist()],
                "axis_xyxy_bottom_to_top": bottom_to_top_axis(polygon),
                "stem_id": None,
            }
        )
    return records


def bbox_intersection_xyxy(
    left: list[float],
    right: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, int(math.floor(max(left[0], right[0]))))
    y1 = max(0, int(math.floor(max(left[1], right[1]))))
    x2 = min(width, int(math.ceil(min(left[2], right[2]))))
    y2 = min(height, int(math.ceil(min(left[3], right[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def polygon_overlap_area_ratio(
    left: dict[str, Any],
    right: dict[str, Any],
    width: int,
    height: int,
) -> float:
    crop = bbox_intersection_xyxy(left["bbox_xyxy"], right["bbox_xyxy"], width, height)
    if crop is None:
        return 0.0

    x1, y1, x2, y2 = crop
    crop_w = x2 - x1
    crop_h = y2 - y1
    left_poly = np.asarray(left["polygon_xy"], dtype=np.float32).copy()
    right_poly = np.asarray(right["polygon_xy"], dtype=np.float32).copy()
    left_poly[:, 0] -= x1
    left_poly[:, 1] -= y1
    right_poly[:, 0] -= x1
    right_poly[:, 1] -= y1

    left_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    right_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_poly.astype(np.int32)], 1)
    cv2.fillPoly(right_mask, [right_poly.astype(np.int32)], 1)

    intersection = int(np.bitwise_and(left_mask, right_mask).sum())
    if intersection <= 0:
        return 0.0

    # 작은 객체가 큰 객체 안에 중복 검출된 경우도 제거하기 위해 작은 영역 기준 비율을 사용합니다.
    reference_area = max(1.0, min(float(left.get("area_px", 0.0)), float(right.get("area_px", 0.0))))
    return float(intersection / reference_area)


def suppress_overlapping_records_by_confidence(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    same_overlap_area: float,
) -> list[dict[str, Any]]:
    if same_overlap_area <= 0 or len(records) <= 1:
        return records

    kept: list[dict[str, Any]] = []
    candidates = sorted(
        records,
        key=lambda record: (
            -float(record.get("confidence", 0.0)),
            -float(record.get("area_px", 0.0)),
        ),
    )
    for candidate in candidates:
        should_suppress = False
        for accepted in kept:
            if polygon_overlap_area_ratio(candidate, accepted, width, height) >= same_overlap_area:
                should_suppress = True
                break
        if not should_suppress:
            kept.append(candidate)
    return kept


def predict_sam3_records(
    image_path: Path,
    predictor,
    min_area_px: int,
    max_area_ratio: float,
    approx_epsilon_ratio: float,
    num_vertices: int,
    same_overlap_area: float = 0.0,
) -> tuple[np.ndarray, int, int, list[dict[str, Any]]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    predictor.set_image(str(image_path))
    results = predictor(text=SAM3_PROMPTS)
    if not isinstance(results, (list, tuple)):
        results = [results]

    records: list[dict[str, Any]] = []
    for result in results:
        records.extend(
            records_from_result(
                result=result,
                image_shape_hw=(image_h, image_w),
                min_area_px=min_area_px,
                max_area_ratio=max_area_ratio,
                approx_epsilon_ratio=approx_epsilon_ratio,
                num_vertices=num_vertices,
            )
        )
    records = suppress_overlapping_records_by_confidence(
        records=records,
        width=image_w,
        height=image_h,
        same_overlap_area=same_overlap_area,
    )
    return image_bgr, image_w, image_h, assign_segment_ids(records)


def classify_yolo_kind(
    class_id: int,
    names: dict[int, str] | list[str] | tuple[str, ...] | None,
    leaf_class_ids: set[int],
    stem_class_ids: set[int],
) -> str | None:
    if class_id in leaf_class_ids:
        return "leaf"
    if class_id in stem_class_ids:
        return "stem"

    class_name = ""
    if isinstance(names, dict):
        class_name = str(names.get(class_id, ""))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        class_name = str(names[class_id])
    normalized = class_name.strip().lower().replace("_", " ")

    if "leaf" in normalized:
        return "leaf"
    if "stem" in normalized or "branch" in normalized or "path" in normalized:
        return "stem"

    if not leaf_class_ids and not stem_class_ids:
        if class_id == 0:
            return "leaf"
        if class_id == 1:
            return "stem"
    return None


def predict_yolo_records(
    image_path: Path,
    yolo_model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    min_area_px: int,
    max_area_ratio: float,
    approx_epsilon_ratio: float,
    num_vertices: int,
    leaf_class_ids: Iterable[int] = (),
    stem_class_ids: Iterable[int] = (),
    yolo_model=None,
) -> tuple[np.ndarray, int, int, list[dict[str, Any]]]:
    from ultralytics import YOLO

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    model = yolo_model if yolo_model is not None else YOLO(str(yolo_model_path))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=resolve_device(device),
        verbose=False,
    )

    records: list[dict[str, Any]] = []
    leaf_ids = {int(v) for v in leaf_class_ids}
    stem_ids = {int(v) for v in stem_class_ids}
    names = getattr(model, "names", None)

    for result in results:
        if result.masks is None or result.boxes is None:
            continue
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for idx, mask_data in enumerate(masks):
            class_id = int(class_ids[idx])
            kind = classify_yolo_kind(class_id, names, leaf_ids, stem_ids)
            if kind is None:
                continue

            mask = mask_data > 0.5
            if mask.shape[:2] != (image_h, image_w):
                mask = cv2.resize(mask.astype(np.uint8), (image_w, image_h), interpolation=cv2.INTER_NEAREST).astype(bool)

            area_px = float(mask.sum())
            if area_px < min_area_px:
                continue
            if area_px / float(image_h * image_w) > max_area_ratio:
                continue

            polygon = mask_to_polygon(mask, min_area_px, approx_epsilon_ratio, num_vertices)
            if polygon is None:
                continue

            x1, y1, x2, y2 = [float(value) for value in boxes[idx].tolist()]
            records.append(
                {
                    "segment_id": -1,
                    "kind": kind,
                    "raw_class_id": class_id,
                    "raw_class_name": str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id),
                    "confidence": float(confidences[idx]),
                    "area_px": area_px,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "centroid_xy": polygon_centroid(polygon, mask),
                    "polygon_xy": [[float(x), float(y)] for x, y in polygon.tolist()],
                    "axis_xyxy_bottom_to_top": bottom_to_top_axis(polygon),
                    "stem_id": None,
                }
            )

    return image_bgr, image_w, image_h, assign_segment_ids(records)


def axis_or_bbox_bottom_to_top(record: dict[str, Any]) -> list[float]:
    axis = record.get("axis_xyxy_bottom_to_top")
    if axis is not None:
        return [float(value) for value in axis]
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    cx = (x1 + x2) * 0.5
    return [cx, y2, cx, y1]


def normalize_axis(axis: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = axis
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def normalize_point(point: tuple[float, float] | list[float], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) / width, float(point[1]) / height


def point_dist(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


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


def leaf_scalar_point(record: dict[str, Any]) -> tuple[float, float]:
    polygon = np.asarray(record.get("polygon_xy", []), dtype=np.float32)
    if len(polygon) < 3:
        return tuple(float(value) for value in record.get("centroid_xy", [0.0, 0.0]))

    start, end = farthest_polygon_points(polygon)
    axis = end - start
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 1e-6:
        return tuple(float(value) for value in record.get("centroid_xy", [0.0, 0.0]))

    axis_dir = axis / axis_len
    perp_dir = np.asarray([-axis_dir[1], axis_dir[0]], dtype=np.float32)
    best_point = (start + end) * 0.5
    best_width = -1.0

    # 잎은 방향 벡터 대신, 최장축과 수직 chord가 가장 긴 위치의 교점을 대표점으로 사용합니다.
    for ratio in np.linspace(0.05, 0.95, 25):
        point = start + axis * float(ratio)
        if cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) < -0.5:
            continue
        params = line_polygon_intersection_parameters(polygon, point, perp_dir)
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
    return point_dist(node_a["centroid_norm"], node_b["centroid_norm"])


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
    node_count = len(nodes)
    if node_count <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    undirected: set[tuple[int, int]] = set()
    for left_idx in range(node_count):
        for right_idx in range(left_idx + 1, node_count):
            if node_distance(nodes[left_idx], nodes[right_idx]) <= edge_radius_norm:
                undirected.add((left_idx, right_idx))

    # 그래프가 지나치게 희소해지는 것을 막기 위해 각 노드의 가까운 k개 이웃을 추가합니다.
    if knn_k > 0:
        for left_idx in range(node_count):
            distances = []
            for right_idx in range(node_count):
                if left_idx == right_idx:
                    continue
                distance = node_distance(nodes[left_idx], nodes[right_idx])
                if not math.isfinite(distance):
                    continue
                distances.append((distance, right_idx))
            for _, right_idx in sorted(distances)[:knn_k]:
                a, b = sorted((left_idx, right_idx))
                undirected.add((a, b))

    directed = []
    for left_idx, right_idx in sorted(undirected):
        directed.append((left_idx, right_idx))
        directed.append((right_idx, left_idx))

    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def build_graph_tensors(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    edge_radius_norm: float,
    knn_k: int,
    include_leaf_context: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        kind = record["kind"]
        if kind == "leaf" and not include_leaf_context:
            continue
        if kind not in {"stem", "leaf"}:
            continue

        axis = axis_or_bbox_bottom_to_top(record)
        cx, cy = record.get("centroid_xy", [(axis[0] + axis[2]) * 0.5, (axis[1] + axis[3]) * 0.5])
        centroid_norm = (float(cx) / width, float(cy) / height)
        is_stem = 1.0 if kind == "stem" else 0.0
        is_leaf = 1.0 if kind == "leaf" else 0.0
        if kind == "stem":
            stem_axis_norm = normalize_axis(axis, width, height)
            leaf_point_norm = (0.0, 0.0)
        else:
            stem_axis_norm = [0.0, 0.0, 0.0, 0.0]
            leaf_point_norm = normalize_point(leaf_scalar_point(record), width, height)
        nodes.append(
            {
                "record_idx": record_idx,
                "kind": kind,
                "segment_id": int(record["segment_id"]),
                "stem_id": record.get("stem_id"),
                "stem_axis_norm": stem_axis_norm,
                "leaf_point_norm": leaf_point_norm,
                "centroid_norm": centroid_norm,
                "x": stem_axis_norm + [leaf_point_norm[0], leaf_point_norm[1], is_stem, is_leaf],
            }
        )

    if not nodes:
        return torch.empty((0, 8), dtype=torch.float32), torch.empty((2, 0), dtype=torch.long), []

    x = torch.tensor([node["x"] for node in nodes], dtype=torch.float32)
    edge_index = build_edges(nodes, edge_radius_norm=edge_radius_norm, knn_k=knn_k)
    return x, edge_index, nodes


class StemBranchGNN(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        if SAGEConv is None:
            raise RuntimeError(
                "torch_geometric is required for GNN inference. "
                f"Original import error: {PYG_IMPORT_ERROR}"
            )
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.input(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = hidden
            hidden = conv(hidden, edge_index)
            hidden = norm(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = hidden + residual
        return self.head(hidden)


def torch_load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_gnn_model(gnn_model_path: Path, device: str = "") -> tuple[StemBranchGNN, torch.device, dict[str, Any]]:
    resolved_device = torch.device(resolve_device(device))
    checkpoint = torch_load_checkpoint(gnn_model_path, resolved_device)
    config = checkpoint.get("config", {})
    model = StemBranchGNN(
        in_dim=int(config.get("in_dim", 8)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_layers=int(config.get("num_layers", 3)),
        dropout=float(config.get("dropout", 0.15)),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, resolved_device, config


def apply_gnn_predictions(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    model: nn.Module,
    device: torch.device,
    edge_radius_norm: float,
    knn_k: int,
    include_leaf_context: bool,
) -> list[dict[str, Any]]:
    for record in records:
        if record["kind"] == "leaf":
            record["gnn_label"] = "leaf"
            record["output_class_id"] = DEMO_LEAF
        elif record["kind"] == "stem":
            record["gnn_label"] = "side_branch"
            record["output_class_id"] = DEMO_STEM

    x, edge_index, nodes = build_graph_tensors(
        records=records,
        width=width,
        height=height,
        edge_radius_norm=edge_radius_norm,
        knn_k=knn_k,
        include_leaf_context=include_leaf_context,
    )
    if len(nodes) == 0:
        return records

    expected_in_dim = getattr(getattr(model, "input", None), "in_features", None)
    if expected_in_dim is not None and int(expected_in_dim) != int(x.shape[1]):
        raise ValueError(
            f"GNN checkpoint input dimension is {expected_in_dim}, but current graph feature dimension is {x.shape[1]}. "
            "Re-train 03_train_gnn_main_side.py with the updated 8D stem-vector/leaf-point features."
        )

    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device))
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=-1)

    for node_idx, node in enumerate(nodes):
        record = records[node["record_idx"]]
        if record["kind"] != "stem":
            continue
        pred_id = int(predictions[node_idx])
        pred_label = ID_TO_LABEL[pred_id]
        record["gnn_label"] = pred_label
        record["prob_side_branch"] = float(probabilities[node_idx, 0])
        record["prob_main_stem"] = float(probabilities[node_idx, 1])
        record["output_class_id"] = DEMO_PATH if pred_label == "main_stem" else DEMO_STEM

    return records


def normalized_polygon_values(record: dict[str, Any], width: int, height: int) -> list[float] | None:
    polygon = np.asarray(record["polygon_xy"], dtype=np.float32)
    if len(polygon) < 3:
        return None
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width) / width
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height) / height
    return polygon.reshape(-1).tolist()


def records_to_yolo_lines(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    class_key: str = "output_class_id",
) -> list[str]:
    lines = []
    for record in records:
        class_id = record.get(class_key)
        if class_id is None:
            continue
        values = normalized_polygon_values(record, width, height)
        if values is None:
            continue
        coords = " ".join(f"{value:.6f}" for value in values)
        lines.append(f"{int(class_id)} {coords}")
    return lines


def write_data_yaml(root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {json.dumps(root.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {json.dumps(class_name)}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_segment_overlay(
    image_bgr: np.ndarray,
    records: list[dict[str, Any]],
    out_path: Path,
    vector_display: bool = True,
) -> None:
    colors = {
        DEMO_LEAF: (60, 220, 60),
        DEMO_STEM: (255, 180, 40),
        DEMO_PATH: (40, 40, 255),
    }
    overlay = image_bgr.copy()
    fill = image_bgr.copy()

    for record in records:
        class_id = int(record.get("output_class_id", DEMO_LEAF))
        polygon = np.asarray(record["polygon_xy"], dtype=np.int32)
        cv2.fillPoly(fill, [polygon], colors.get(class_id, (255, 255, 255)))
    overlay = cv2.addWeighted(fill, 0.28, overlay, 0.72, 0)

    for record in records:
        class_id = int(record.get("output_class_id", DEMO_LEAF))
        polygon = np.asarray(record["polygon_xy"], dtype=np.int32)
        color = colors.get(class_id, (255, 255, 255))
        cv2.polylines(overlay, [polygon], True, color, 2)
        axis = record.get("axis_xyxy_bottom_to_top")
        if vector_display and axis is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in axis]
            cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def draw_edge_overlay(
    image_bgr: np.ndarray,
    records: list[dict[str, Any]],
    width: int,
    height: int,
    edge_radius_norm: float,
    knn_k: int,
    include_leaf_context: bool,
    out_path: Path,
) -> None:
    overlay = image_bgr.copy()
    _, edge_index, nodes = build_graph_tensors(
        records=records,
        width=width,
        height=height,
        edge_radius_norm=edge_radius_norm,
        knn_k=knn_k,
        include_leaf_context=include_leaf_context,
    )
    if not nodes:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        return

    points = []
    for node in nodes:
        if node["kind"] == "stem":
            axis = node["stem_axis_norm"]
            px = (axis[0] + axis[2]) * 0.5 * width
            py = (axis[1] + axis[3]) * 0.5 * height
        else:
            leaf_point = node["leaf_point_norm"]
            px = leaf_point[0] * width
            py = leaf_point[1] * height
        points.append((int(round(px)), int(round(py))))

    undirected_edges = set()
    if edge_index.numel() > 0:
        for left_idx, right_idx in edge_index.t().cpu().numpy().tolist():
            if left_idx == right_idx:
                continue
            undirected_edges.add(tuple(sorted((int(left_idx), int(right_idx)))))

    for left_idx, right_idx in sorted(undirected_edges):
        cv2.line(overlay, points[left_idx], points[right_idx], (0, 220, 255), 2, cv2.LINE_AA)

    for node_idx, node in enumerate(nodes):
        record = records[node["record_idx"]]
        class_id = int(record.get("output_class_id", DEMO_LEAF))
        if class_id == DEMO_PATH:
            color = (40, 40, 255)
        elif class_id == DEMO_STEM:
            color = (255, 180, 40)
        else:
            color = (60, 220, 60)
        cv2.circle(overlay, points[node_idx], 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, points[node_idx], 4, color, -1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def save_demo_outputs(
    output_dir: Path,
    image_path: Path,
    image_bgr: np.ndarray,
    width: int,
    height: int,
    records: list[dict[str, Any]],
    source: str,
    segment_only: bool = True,
    edge_only: bool = True,
    edge_radius_norm: float = 0.12,
    knn_k: int = 3,
    include_leaf_context: bool = True,
    vector_display: bool = True,
) -> None:
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (output_dir / "debug").mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, output_dir / "images" / image_path.name)
    label_lines = records_to_yolo_lines(records, width, height, class_key="output_class_id")
    (output_dir / "labels" / f"{image_path.stem}.txt").write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )
    if segment_only:
        draw_segment_overlay(
            image_bgr=image_bgr,
            records=records,
            out_path=output_dir / "overlays" / f"{image_path.stem}_segments.png",
            vector_display=vector_display,
        )
    if edge_only:
        draw_edge_overlay(
            image_bgr=image_bgr,
            records=records,
            width=width,
            height=height,
            edge_radius_norm=edge_radius_norm,
            knn_k=knn_k,
            include_leaf_context=include_leaf_context,
            out_path=output_dir / "overlays" / f"{image_path.stem}_edges.png",
        )
    if not segment_only and not edge_only:
        draw_segment_overlay(
            image_bgr=image_bgr,
            records=records,
            out_path=output_dir / "overlays" / f"{image_path.stem}_overlay.png",
            vector_display=vector_display,
        )
    write_data_yaml(output_dir, DEMO_CLASS_NAMES)

    payload = {
        "schema_version": "plant-demo-seg-gnn-v1",
        "source": source,
        "image": image_path.name,
        "width": width,
        "height": height,
        "class_names": DEMO_CLASS_NAMES,
        "segment_only_overlay": segment_only,
        "edge_only_overlay": edge_only,
        "vector_display": vector_display,
        "segments": records,
    }
    (output_dir / "debug" / f"{image_path.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
