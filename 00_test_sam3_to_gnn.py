from __future__ import annotations

from pathlib import Path

from plant_seg_gnn_utils import (
    apply_gnn_predictions,
    load_gnn_model,
    load_sam3_predictor,
    predict_sam3_records,
    save_demo_outputs,
)


img_path = "./img1.png"
img_paths = [""]
gnn_model_path = "./plant_dataset/gnn_model.pt"
output_dir = "sam3_to_gnn_demo_output"
segment_only = True
edge_only = False
vector_display = False

sam3_model_path = "sam3.pt"
sam3_conf = 0.25
sam3_iou = 0.70
sam3_imgsz = 1024
sam3_half = True
device = ""

min_area_px = 80
max_area_ratio = 0.75
polygon_approx_epsilon_ratio = 0.002
num_vertices = 32
SAME_OVERLAP_AREA = 0.6

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


def process_one_image(image_path: Path, predictor, model, torch_device) -> tuple[int, int, int]:
    image_bgr, width, height, records = predict_sam3_records(
        image_path=image_path,
        predictor=predictor,
        min_area_px=min_area_px,
        max_area_ratio=max_area_ratio,
        approx_epsilon_ratio=polygon_approx_epsilon_ratio,
        num_vertices=num_vertices,
        same_overlap_area=SAME_OVERLAP_AREA,
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
        source="SAM3 leaf/stem -> GNN main/side -> leaf/stem/path",
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
    checkpoint_path = Path(gnn_model_path)
    if not gnn_model_path:
        raise ValueError('Set gnn_model_path, for example gnn_model_path = "plant_dataset/gnn_model.pt".')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint_path}")

    predictor = load_sam3_predictor(
        sam3_model_path=sam3_model_path,
        conf=sam3_conf,
        iou=sam3_iou,
        imgsz=sam3_imgsz,
        half=sam3_half,
        device=device,
    )
    model, torch_device, _ = load_gnn_model(checkpoint_path, device=device)
    totals = [0, 0, 0]
    for image_path in image_paths_to_run:
        counts = process_one_image(image_path, predictor, model, torch_device)
        totals = [left + right for left, right in zip(totals, counts)]

    print(f"Saved output: {Path(output_dir).resolve()}")
    print(f"Images: {len(image_paths_to_run)}")
    print(f"Segments total: leaf={totals[0]}, stem={totals[1]}, path={totals[2]}")


if __name__ == "__main__":
    main()
