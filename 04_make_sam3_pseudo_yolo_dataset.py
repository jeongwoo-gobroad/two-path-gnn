from __future__ import annotations

import random
import shutil
from pathlib import Path

from plant_seg_gnn_utils import (
    YOLO_PSEUDO_CLASS_NAMES,
    iter_images,
    load_sam3_predictor,
    predict_sam3_records,
    records_to_yolo_lines,
    write_data_yaml,
)


# =========================
# User parameters
# =========================

source_img_dir = "./tomato_greenhouse_dataset/images"
target_path = "./tomato4yolo"
t_v_ratio = (0.7, 0.3)

sam3_model_path = "sam3.pt"
sam3_conf = 0.25
sam3_iou = 0.70
sam3_imgsz = 1024
sam3_half = True
device = ""

min_area_px = 80
max_area_ratio = 0.75
polygon_approx_epsilon_ratio = 0.002
num_vertices = 16
SAME_OVERLAP_AREA = 0.6

seed = 42
max_images = 0
overwrite_target_path = False


def make_unique_name(image_path: Path, source_root: Path, used_names: set[str]) -> str:
    try:
        relative = image_path.relative_to(source_root)
    except ValueError:
        relative = Path(image_path.name)

    stem = "__".join(relative.with_suffix("").parts)
    candidate = f"{stem}{image_path.suffix.lower()}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    counter = 1
    while True:
        candidate = f"{stem}__{counter}{image_path.suffix.lower()}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def split_images(image_paths: list[Path], train_val_ratio: tuple[float, float], split_seed: int) -> tuple[list[Path], list[Path]]:
    train_ratio, val_ratio = train_val_ratio
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Both train and val ratios must be positive.")

    paths = list(image_paths)
    random.Random(split_seed).shuffle(paths)
    train_count = int(round(len(paths) * train_ratio / (train_ratio + val_ratio)))
    train_count = max(1, min(train_count, len(paths) - 1)) if len(paths) > 1 else len(paths)
    return paths[:train_count], paths[train_count:]


def prepare_target(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite_target_path:
            raise FileExistsError(f"Target path already exists and is not empty: {root}")
        shutil.rmtree(root)

    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def mark_pseudo_yolo_classes(records: list[dict]) -> list[dict]:
    for record in records:
        if record["kind"] == "leaf":
            record["pseudo_class_id"] = 0
        elif record["kind"] == "stem":
            record["pseudo_class_id"] = 1
        else:
            record["pseudo_class_id"] = None
    return records


def process_split(
    split_name: str,
    image_paths: list[Path],
    source_root: Path,
    target_root: Path,
    predictor,
    used_names: set[str],
) -> tuple[int, int]:
    image_count = 0
    label_count = 0

    try:
        from tqdm import tqdm
    except Exception:
        iterator = image_paths
    else:
        iterator = tqdm(image_paths, desc=f"SAM3 {split_name}")

    for image_path in iterator:
        output_name = make_unique_name(image_path, source_root, used_names)
        image_bgr, width, height, records = predict_sam3_records(
            image_path=image_path,
            predictor=predictor,
            min_area_px=min_area_px,
            max_area_ratio=max_area_ratio,
            approx_epsilon_ratio=polygon_approx_epsilon_ratio,
            num_vertices=num_vertices,
            same_overlap_area=SAME_OVERLAP_AREA,
        )
        _ = image_bgr
        records = mark_pseudo_yolo_classes(records)

        shutil.copy2(image_path, target_root / "images" / split_name / output_name)
        label_lines = records_to_yolo_lines(records, width, height, class_key="pseudo_class_id")
        (target_root / "labels" / split_name / f"{Path(output_name).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

        image_count += 1
        label_count += len(label_lines)

    return image_count, label_count


def main() -> None:
    source_root = Path(source_img_dir)
    target_root = Path(target_path)
    if not target_path:
        raise ValueError('Set target_path, for example target_path = "tomato_greenhouse_sam3_pseudo_yolo".')
    if not source_root.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_root}")
    if not Path(sam3_model_path).exists():
        raise FileNotFoundError(f"SAM3 model not found: {sam3_model_path}")

    image_paths = list(iter_images(source_root))
    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No images found: {source_root}")

    train_paths, val_paths = split_images(image_paths, t_v_ratio, seed)
    prepare_target(target_root)
    write_data_yaml(target_root, YOLO_PSEUDO_CLASS_NAMES)

    predictor = load_sam3_predictor(
        sam3_model_path=sam3_model_path,
        conf=sam3_conf,
        iou=sam3_iou,
        imgsz=sam3_imgsz,
        half=sam3_half,
        device=device,
    )

    used_names: set[str] = set()
    train_images, train_labels = process_split("train", train_paths, source_root, target_root, predictor, used_names)
    val_images, val_labels = process_split("val", val_paths, source_root, target_root, predictor, used_names)

    print(f"Saved dataset: {target_root.resolve()}")
    print(f"train: images={train_images}, labels={train_labels}")
    print(f"val: images={val_images}, labels={val_labels}")


if __name__ == "__main__":
    main()
