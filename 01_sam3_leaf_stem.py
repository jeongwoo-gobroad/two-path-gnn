#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROMPTS = ["stem", "leaf"]
RAW_STEM = 0
RAW_LEAF = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    input_path: Path
    out_dir: Path
    sam3_model: Path
    conf: float = 0.25
    iou: float = 0.70
    imgsz: int = 1024
    half: bool = True
    device: str = ""
    min_area_px: int = 80
    max_area_ratio: float = 0.75
    approx_epsilon_ratio: float = 0.002
    num_vertices: int = 32
    same_overlap_area: float = 0.6
    save_masks: bool = True
    overwrite: bool = False


def resolve_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
        return
    raise FileNotFoundError(f"Input path not found: {path}")


def prepare_dirs(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["images", "segments", "overlays", "masks"]:
        (cfg.out_dir / name).mkdir(parents=True, exist_ok=True)


def load_sam3_predictor(cfg: Config):
    from ultralytics.models.sam import SAM3SemanticPredictor

    device = resolve_device(cfg.device)
    use_half = bool(cfg.half and device == "cuda")
    overrides = {
        "conf": cfg.conf,
        "iou": cfg.iou,
        "task": "segment",
        "mode": "predict",
        "model": str(cfg.sam3_model),
        "imgsz": cfg.imgsz,
        "half": use_half,
        "device": device,
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
    for dist in sample_distances:
        while edge_idx < len(edge_lengths) - 1 and distances[edge_idx + 1] < dist:
            edge_idx += 1
        start = closed[edge_idx]
        end = closed[edge_idx + 1]
        edge_length = max(float(edge_lengths[edge_idx]), 1e-6)
        t = (dist - distances[edge_idx]) / edge_length
        sampled.append(start + t * (end - start))
    return np.asarray(sampled, dtype=np.float32)


def mask_to_polygon(mask: np.ndarray, cfg: Config) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= cfg.min_area_px]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 3:
        return None

    perimeter = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    eps = max(1.0, perimeter * cfg.approx_epsilon_ratio)
    poly = cv2.approxPolyDP(contour.reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float32)
    if len(poly) < 3:
        poly = contour
    if cfg.num_vertices and cfg.num_vertices >= 3:
        poly = resample_closed_polygon(poly, cfg.num_vertices)
    return poly.astype(np.float32)


def farthest_polygon_points(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = poly.astype(np.float32)
    deltas = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(deltas * deltas, axis=2)
    flat_idx = int(np.argmax(dist2))
    i, j = np.unravel_index(flat_idx, dist2.shape)
    return pts[i].copy(), pts[j].copy()


def bottom_to_top_axis(poly: np.ndarray) -> list[float] | None:
    if poly is None or len(poly) < 2:
        return None
    p1, p2 = farthest_polygon_points(poly)
    # 이미지에서는 y가 클수록 아래라서, 축 방향을 아래->위로 맞춰 둡니다.
    if p1[1] >= p2[1]:
        bottom, top = p1, p2
    else:
        bottom, top = p2, p1
    return [float(bottom[0]), float(bottom[1]), float(top[0]), float(top[1])]


def polygon_centroid(poly: np.ndarray, fallback_mask: np.ndarray) -> list[float]:
    m = cv2.moments(poly.astype(np.float32))
    if m["m00"]:
        return [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])]
    ys, xs = np.where(fallback_mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def result_to_records(result, image_shape_hw: tuple[int, int], cfg: Config) -> list[dict]:
    if result.masks is None or result.boxes is None:
        return []

    h, w = image_shape_hw
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    records: list[dict] = []
    for idx, mask_data in enumerate(masks):
        raw_class = int(classes[idx])
        if raw_class not in {RAW_STEM, RAW_LEAF}:
            continue

        mask = mask_data > 0.5
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        area = float(mask.sum())
        if area < cfg.min_area_px:
            continue
        if area / float(h * w) > cfg.max_area_ratio:
            continue

        poly = mask_to_polygon(mask, cfg)
        if poly is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[idx].tolist()]
        kind = "stem" if raw_class == RAW_STEM else "leaf"
        record = {
            "segment_id": -1,
            "kind": kind,
            "raw_class_id": raw_class,
            "raw_class_name": PROMPTS[raw_class],
            "confidence": float(confs[idx]),
            "area_px": area,
            "bbox_xyxy": [x1, y1, x2, y2],
            "centroid_xy": polygon_centroid(poly, mask),
            "polygon_xy": [[float(x), float(y)] for x, y in poly.tolist()],
            "axis_xyxy_bottom_to_top": bottom_to_top_axis(poly),
            "stem_id": None,
        }
        records.append(record)
    return records


def bbox_intersection_xyxy(left: list[float], right: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, int(math.floor(max(left[0], right[0]))))
    y1 = max(0, int(math.floor(max(left[1], right[1]))))
    x2 = min(width, int(math.ceil(min(left[2], right[2]))))
    y2 = min(height, int(math.ceil(min(left[3], right[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def polygon_overlap_area_ratio(left: dict, right: dict, width: int, height: int) -> float:
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

    # 작은 마스크가 큰 마스크 안에 겹쳐 잡히는 경우가 있어서, 더 작은 쪽 면적 기준으로 봅니다.
    reference_area = max(1.0, min(float(left.get("area_px", 0.0)), float(right.get("area_px", 0.0))))
    return float(intersection / reference_area)


def suppress_overlapping_records_by_confidence(records: list[dict], width: int, height: int, same_overlap_area: float) -> list[dict]:
    if same_overlap_area <= 0 or len(records) <= 1:
        return records

    kept: list[dict] = []
    candidates = sorted(
        records,
        key=lambda rec: (-float(rec.get("confidence", 0.0)), -float(rec.get("area_px", 0.0))),
    )
    for candidate in candidates:
        if any(
            polygon_overlap_area_ratio(candidate, accepted, width, height) >= same_overlap_area
            for accepted in kept
        ):
            continue
        kept.append(candidate)
    return kept


def draw_overlay(image_bgr: np.ndarray, records: list[dict], out_path: Path) -> None:
    overlay = image_bgr.copy()
    alpha = 0.35
    fill = image_bgr.copy()

    for rec in records:
        poly = np.asarray(rec["polygon_xy"], dtype=np.int32)
        if rec["kind"] == "stem":
            color = (50, 80, 255)
        else:
            color = (60, 220, 60)
        cv2.fillPoly(fill, [poly], color)

    overlay = cv2.addWeighted(fill, alpha, overlay, 1 - alpha, 0)

    for rec in records:
        poly = np.asarray(rec["polygon_xy"], dtype=np.int32)
        color = (50, 80, 255) if rec["kind"] == "stem" else (60, 220, 60)
        cv2.polylines(overlay, [poly], True, color, 2)
        axis = rec.get("axis_xyxy_bottom_to_top")
        if axis is not None and rec["kind"] == "stem":
            x1, y1, x2, y2 = axis
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            if rec.get("stem_id") is not None:
                cx, cy = rec["centroid_xy"]
                cv2.putText(overlay, str(rec["stem_id"]), (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(overlay, str(rec["stem_id"]), (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), overlay)


def save_masks(records: list[dict], image_shape_hw: tuple[int, int], out_dir: Path, image_id: str) -> None:
    h, w = image_shape_hw
    for rec in records:
        poly = np.asarray(rec["polygon_xy"], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cv2.imwrite(str(out_dir / f"{image_id}_{rec['segment_id']:04d}_{rec['kind']}.png"), mask)


def process_one(image_path: Path, predictor, cfg: Config) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = image_bgr.shape[:2]
    image_id = image_path.stem

    predictor.set_image(str(image_path))
    results = predictor(text=PROMPTS)
    if not isinstance(results, (list, tuple)):
        results = [results]

    records: list[dict] = []
    for result in results:
        records.extend(result_to_records(result, (h, w), cfg))
    records = suppress_overlapping_records_by_confidence(records, w, h, cfg.same_overlap_area)

    for sid, rec in enumerate(records):
        rec["segment_id"] = sid

    stem_no = 1
    for rec in records:
        if rec["kind"] == "stem":
            rec["stem_id"] = stem_no
            stem_no += 1

    shutil.copy2(image_path, cfg.out_dir / "images" / image_path.name)
    if cfg.save_masks:
        save_masks(records, (h, w), cfg.out_dir / "masks", image_id)
    draw_overlay(image_bgr, records, cfg.out_dir / "overlays" / f"{image_id}_sam3_overlay.png")

    payload = {
        "schema_version": "plant-sam3-segments-v1",
        "image_id": image_id,
        "image_path": str((cfg.out_dir / "images" / image_path.name).resolve()),
        "original_image_path": str(image_path.resolve()),
        "width": w,
        "height": h,
        "prompts": PROMPTS,
        "segments": records,
    }
    with (cfg.out_dir / "segments" / f"{image_id}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[{image_id}] segments={len(records)} stems={stem_no - 1} -> {cfg.out_dir / 'segments' / (image_id + '.json')}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or image directory")
    parser.add_argument("--out", default="plant_dataset", help="Output dataset directory")
    parser.add_argument("--sam3-model", default=str(Path(__file__).with_name("sam3.pt")))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--same-overlap-area", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()
    return Config(
        input_path=Path(args.input),
        out_dir=Path(args.out),
        sam3_model=Path(args.sam3_model),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=not args.no_half,
        device=args.device,
        min_area_px=args.min_area,
        same_overlap_area=args.same_overlap_area,
        save_masks=not args.no_masks,
        overwrite=args.overwrite,
    )


def main() -> None:
    cfg = parse_args()
    if not cfg.sam3_model.exists():
        raise FileNotFoundError(f"sam3.pt not found: {cfg.sam3_model}")
    prepare_dirs(cfg)
    predictor = load_sam3_predictor(cfg)
    image_paths = list(iter_images(cfg.input_path))
    if not image_paths:
        raise ValueError(f"No images found: {cfg.input_path}")
    for image_path in image_paths:
        out_json = cfg.out_dir / "segments" / f"{image_path.stem}.json"
        if out_json.exists() and not cfg.overwrite:
            print(f"Skip existing: {out_json}")
            continue
        process_one(image_path, predictor, cfg)


if __name__ == "__main__":
    main()
