from __future__ import annotations

from pathlib import Path

from plant_seg_gnn_utils import (
    apply_gnn_predictions,
    load_gnn_model,
    predict_yolo_records,
    save_demo_outputs,
)


# =========================
# User parameters
# =========================

yolo_model_path = "best.pt"
img_path = "./img1.png"
img_paths = [""]
gnn_model_path = "./plant_dataset/gnn_model.pt"
output_dir = "yolo_to_gnn_demo_output"
segment_only = True
edge_only = True
vector_display = True

yolo_conf = 0.25
yolo_iou = 0.70
yolo_imgsz = 1024
device = ""

# 비워두면 model.names에서 leaf/stem을 자동 추론합니다.
# 이름이 불명확한 모델이면 예: yolo_leaf_class_ids = [0], yolo_stem_class_ids = [1]
yolo_leaf_class_ids: list[int] = []
yolo_stem_class_ids: list[int] = []

min_area_px = 80
max_area_ratio = 0.75
polygon_approx_epsilon_ratio = 0.002
num_vertices = 16

edge_radius_norm = 0.12
knn_k = 3
include_leaf_context = True


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


def process_one_image(image_path: Path, yolo_model, model, torch_device) -> tuple[int, int, int]:
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

    records = apply_gnn_predictions(
        records=records,
        width=width,
        height=height,
        model=model,
        device=torch_device,
        edge_radius_norm=edge_radius_norm,
        knn_k=knn_k,
        include_leaf_context=include_leaf_context,
    )

    save_demo_outputs(
        output_dir=Path(output_dir),
        image_path=image_path,
        image_bgr=image_bgr,
        width=width,
        height=height,
        records=records,
        source="YOLO-seg leaf/stem -> GNN main/side -> leaf/stem/path",
        segment_only=segment_only,
        edge_only=edge_only,
        edge_radius_norm=edge_radius_norm,
        knn_k=knn_k,
        include_leaf_context=include_leaf_context,
        vector_display=vector_display,
    )

    num_leaf = sum(1 for record in records if record.get("output_class_id") == 0)
    num_stem = sum(1 for record in records if record.get("output_class_id") == 1)
    num_path = sum(1 for record in records if record.get("output_class_id") == 2)
    print(f"[{image_path.name}] leaf={num_leaf}, stem={num_stem}, path={num_path}")
    return num_leaf, num_stem, num_path


def main() -> None:
    image_paths_to_run = resolve_image_paths()
    yolo_path = Path(yolo_model_path)
    checkpoint_path = Path(gnn_model_path)
    if not yolo_model_path:
        raise ValueError('Set yolo_model_path, for example yolo_model_path = "runs/segment/train/weights/best.pt".')
    if not gnn_model_path:
        raise ValueError('Set gnn_model_path, for example gnn_model_path = "plant_dataset/gnn_model.pt".')
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint_path}")

    from ultralytics import YOLO
    yolo_model = YOLO(str(yolo_path))
    model, torch_device, _ = load_gnn_model(checkpoint_path, device=device)
    totals = [0, 0, 0]
    for image_path in image_paths_to_run:
        counts = process_one_image(image_path, yolo_model, model, torch_device)
        totals = [left + right for left, right in zip(totals, counts)]

    print(f"Saved output: {Path(output_dir).resolve()}")
    print(f"Images: {len(image_paths_to_run)}")
    print(f"Segments total: leaf={totals[0]}, stem={totals[1]}, path={totals[2]}")


if __name__ == "__main__":
    main()
