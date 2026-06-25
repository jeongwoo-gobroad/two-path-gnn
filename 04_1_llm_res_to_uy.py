from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


dataset_root = "../plant_dataset"
target_dir = "../tomato"
train_test_ratio = (1.0, 0.0)

include_leaf = True
include_unlabeled_stem_as_side_branch = True
overwrite_target_dir = False

class_names = ["leaf", "side-branch", "main-stem"]
leaf_class_id = 0
side_branch_class_id = 1
main_stem_class_id = 2


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {json.dumps(root.resolve().as_posix())}",
        "train: images/train",
        "val: images/train",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {json.dumps(class_name)}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_target(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite_target_dir:
            raise FileExistsError(f"Target directory already exists and is not empty: {root}")
        shutil.rmtree(root)

    for split in ["train", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def label_map_from_llm(llm_payload: dict[str, Any]) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for item in llm_payload.get("labels", []):
        try:
            stem_id = int(item["stem_id"])
        except (KeyError, TypeError, ValueError):
            continue
        label_map[stem_id] = str(item.get("label", "")).strip()
    return label_map


def yolo_class_for_segment(segment: dict[str, Any], stem_label_map: dict[int, str]) -> int | None:
    kind = str(segment.get("kind", "")).strip()
    if kind == "leaf":
        return leaf_class_id if include_leaf else None

    if kind != "stem":
        return None

    stem_id = segment.get("stem_id")
    if stem_id is None:
        return side_branch_class_id if include_unlabeled_stem_as_side_branch else None

    label = stem_label_map.get(int(stem_id))
    if label == "main_stem":
        return main_stem_class_id
    if label == "side_branch":
        return side_branch_class_id

    return side_branch_class_id if include_unlabeled_stem_as_side_branch else None


def yolo_line_from_polygon(class_id: int, polygon_xy: list[list[float]], width: int, height: int) -> str | None:
    polygon = np.asarray(polygon_xy, dtype=np.float32)
    if len(polygon) < 3:
        return None

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width) / width
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height) / height
    coords = " ".join(f"{value:.6f}" for value in polygon.reshape(-1).tolist())
    return f"{class_id} {coords}"


def resolve_segment_path(root: Path, llm_path: Path, llm_payload: dict[str, Any]) -> Path:
    source = llm_payload.get("source_segments_json")
    if source:
        source_path = Path(source)
        if source_path.exists():
            return source_path

    fallback = root / "segments" / f"{llm_path.stem}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find segment JSON for {llm_path.name}")


def image_path_from_segment_payload(segment_payload: dict[str, Any]) -> Path:
    image_path = Path(segment_payload.get("image_path", ""))
    if image_path.exists():
        return image_path

    original_image_path = Path(segment_payload.get("original_image_path", ""))
    if original_image_path.exists():
        return original_image_path

    raise FileNotFoundError(f"Could not find image for image_id={segment_payload.get('image_id')}")


def convert_one(llm_path: Path, source_root: Path, target_root: Path) -> tuple[int, int]:
    llm_payload = load_json(llm_path)
    segment_path = resolve_segment_path(source_root, llm_path, llm_payload)
    segment_payload = load_json(segment_path)
    stem_label_map = label_map_from_llm(llm_payload)

    width = int(segment_payload["width"])
    height = int(segment_payload["height"])
    image_path = image_path_from_segment_payload(segment_payload)
    image_name = image_path.name

    label_lines = []
    for segment in segment_payload.get("segments", []):
        class_id = yolo_class_for_segment(segment, stem_label_map)
        if class_id is None:
            continue
        line = yolo_line_from_polygon(class_id, segment.get("polygon_xy", []), width, height)
        if line is not None:
            label_lines.append(line)

    shutil.copy2(image_path, target_root / "images" / "train" / image_name)
    label_path = target_root / "labels" / "train" / f"{Path(image_name).stem}.txt"
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return 1, len(label_lines)


def validate_ratio() -> None:
    train_ratio, test_ratio = train_test_ratio
    if float(train_ratio) != 1.0 or float(test_ratio) != 0.0:
        raise ValueError("This script intentionally supports only train:test = 1.0:0.0.")


def main() -> None:
    validate_ratio()
    source_root = Path(dataset_root)
    output_root = Path(target_dir)
    if not target_dir:
        raise ValueError('Set target_dir, for example target_dir = "llm_yolo_dataset".')

    llm_dir = source_root / "llm_labels"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing llm_labels directory: {llm_dir}")

    llm_paths = sorted(path for path in llm_dir.glob("*.json") if not path.name.endswith(".error.json"))
    if not llm_paths:
        raise ValueError(f"No LLM label JSON files found: {llm_dir}")

    prepare_target(output_root)
    write_data_yaml(output_root)

    total_images = 0
    total_labels = 0
    for llm_path in llm_paths:
        image_count, label_count = convert_one(llm_path, source_root, output_root)
        total_images += image_count
        total_labels += label_count

    print(f"Saved Ultralytics YOLO dataset: {output_root.resolve()}")
    print(f"train images: {total_images}")
    print(f"train labels: {total_labels}")
    print("test images: 0")


if __name__ == "__main__":
    main()
