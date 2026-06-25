from __future__ import annotations

import json
import shutil
from pathlib import Path

from plant_seg_gnn_utils import (
    YOLO_PSEUDO_CLASS_NAMES,
    draw_segment_overlay,
    predict_yolo_records,
    records_to_yolo_lines,
    write_data_yaml,
)


img_path = ""
img_paths = [""]
yolo_model_path = ""
output_dir = "yolo_only_demo_output"
segment_only = True
vector_display = True

yolo_conf = 0.25
yolo_iou = 0.70
yolo_imgsz = 1024
device = ""

yolo_leaf_class_ids: list[int] = []
yolo_stem_class_ids: list[int] = []

min_area_px = 80
max_area_ratio = 0.75
polygon_approx_epsilon_ratio = 0.002
num_vertices = 32


def resolve_image_paths() -> list[Path]:
    paths = [Path(path) for path in img_paths if str(path).strip()]
    if not paths and img_path:
        paths = [Path(img_path)]
    if not paths:
        raise ValueError('Set img_paths = ["img1.png", "img2.png"] or img_path = "img1.png".')
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
    return paths


def mark_yolo_only_classes(records: list[dict]) -> list[dict]:
    for record in records:
        if record["kind"] == "leaf":
            record["output_class_id"] = 0
            record["output_class_name"] = "leaf"
        elif record["kind"] == "stem":
            record["output_class_id"] = 1
            record["output_class_name"] = "stem"
        else:
            record["output_class_id"] = None
            record["output_class_name"] = None
    return records


def save_yolo_only_outputs(
    output_root: Path,
    image_path: Path,
    image_bgr,
    width: int,
    height: int,
    records: list[dict],
) -> None:
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)
    (output_root / "debug").mkdir(parents=True, exist_ok=True)
    if segment_only:
        (output_root / "overlays").mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, output_root / "images" / image_path.name)

    label_lines = records_to_yolo_lines(records, width, height, class_key="output_class_id")
    (output_root / "labels" / f"{image_path.stem}.txt").write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )

    if segment_only:
        draw_segment_overlay(
            image_bgr=image_bgr,
            records=records,
            out_path=output_root / "overlays" / f"{image_path.stem}_segments.png",
            vector_display=vector_display,
        )

    write_data_yaml(output_root, YOLO_PSEUDO_CLASS_NAMES)

    payload = {
        "schema_version": "plant-yolo-only-demo-v1",
        "source": "YOLO-seg leaf/stem only",
        "image": image_path.name,
        "width": width,
        "height": height,
        "class_names": YOLO_PSEUDO_CLASS_NAMES,
        "segment_only_overlay": segment_only,
        "vector_display": vector_display,
        "segments": records,
    }
    (output_root / "debug" / f"{image_path.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_one_image(image_path: Path, yolo_model) -> tuple[int, int]:
    image_bgr, width, height, records = predict_yolo_records(
        image_path=image_path,
        yolo_model_path=Path(yolo_model_path),
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        device=device,
        min_area_px=min_area_px,
        max_area_ratio=max_area_ratio,
        approx_epsilon_ratio=polygon_approx_epsilon_ratio,
        num_vertices=num_vertices,
        leaf_class_ids=yolo_leaf_class_ids,
        stem_class_ids=yolo_stem_class_ids,
        yolo_model=yolo_model,
    )
    records = mark_yolo_only_classes(records)

    save_yolo_only_outputs(
        output_root=Path(output_dir),
        image_path=image_path,
        image_bgr=image_bgr,
        width=width,
        height=height,
        records=records,
    )

    num_leaf = sum(1 for record in records if record.get("output_class_id") == 0)
    num_stem = sum(1 for record in records if record.get("output_class_id") == 1)
    print(f"[{image_path.name}] leaf={num_leaf}, stem={num_stem}")
    return num_leaf, num_stem


def main() -> None:
    image_paths_to_run = resolve_image_paths()
    yolo_path = Path(yolo_model_path)
    if not yolo_model_path:
        raise ValueError('Set yolo_model_path, for example yolo_model_path = "runs/segment/train/weights/best.pt".')
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

    from ultralytics import YOLO
    yolo_model = YOLO(str(yolo_path))
    totals = [0, 0]
    for image_path in image_paths_to_run:
        counts = process_one_image(image_path, yolo_model)
        totals = [left + right for left, right in zip(totals, counts)]

    print(f"Saved output: {Path(output_dir).resolve()}")
    print(f"Images: {len(image_paths_to_run)}")
    print(f"Segments total: leaf={totals[0]}, stem={totals[1]}")


if __name__ == "__main__":
    main()
