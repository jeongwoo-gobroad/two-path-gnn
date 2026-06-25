#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


OPENAI_API_KEY = ""


MAIN_ALIASES = {"main", "main_stem", "main stem", "stem", "primary", "primary_stem", "primary stem"}
SIDE_ALIASES = {"side", "side_branch", "side branch", "branch", "lateral", "lateral_branch", "lateral branch", "sub"}
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled", "canceled"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def image_to_data_url(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def resize_for_api(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image_bgr
    scale = max_side / float(side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_text_with_outline(img, text: str, org: tuple[int, int], scale: float, color, thickness: int) -> None:
    x, y = org
    outline = (255, 255, 255)
    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_numbered_stem_overlay(segment_json: dict[str, Any], overlay_path: Path, api_image_path: Path, max_side: int = 1800) -> None:
    image_path = Path(segment_json["image_path"])
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    overlay = image_bgr.copy()
    translucent = image_bgr.copy()

    # 모델이 번호를 헷갈리지 않도록 stem만 확실히 튀게 그립니다.
    for rec in segment_json["segments"]:
        poly = np.asarray(rec["polygon_xy"], dtype=np.int32)
        if rec["kind"] == "leaf":
            cv2.fillPoly(translucent, [poly], (80, 180, 80))
        elif rec["kind"] == "stem":
            cv2.fillPoly(translucent, [poly], (40, 80, 255))
    overlay = cv2.addWeighted(translucent, 0.25, overlay, 0.75, 0)

    h, w = image_bgr.shape[:2]
    font_scale = max(1.4, min(h, w) / 650.0)
    text_thickness = max(3, int(round(font_scale * 2)))
    circle_radius = int(round(font_scale * 18))

    for rec in segment_json["segments"]:
        if rec["kind"] != "stem":
            continue
        poly = np.asarray(rec["polygon_xy"], dtype=np.int32)
        cv2.polylines(overlay, [poly], True, (0, 0, 255), max(3, text_thickness // 2))

        axis = rec.get("axis_xyxy_bottom_to_top")
        if axis is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in axis]
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), max(3, text_thickness // 2))
            cv2.circle(overlay, (x1, y1), max(5, circle_radius // 4), (255, 255, 255), -1)
            cv2.circle(overlay, (x2, y2), max(5, circle_radius // 4), (0, 255, 255), -1)

        cx, cy = rec["centroid_xy"]
        cx, cy = int(round(cx)), int(round(cy))
        stem_id = int(rec["stem_id"])
        cv2.circle(overlay, (cx, cy), circle_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), circle_radius, (0, 0, 255), max(3, text_thickness // 2))

        text = str(stem_id)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        tx = cx - tw // 2
        ty = cy + th // 2
        draw_text_with_outline(overlay, text, (tx, ty), font_scale, (0, 0, 255), text_thickness)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)

    api_img = resize_for_api(overlay, max_side)
    api_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(api_image_path), api_img)


def build_prompt(segment_json: dict[str, Any]) -> str:
    stems = [s for s in segment_json["segments"] if s["kind"] == "stem"]
    lines = []
    for s in stems:
        axis = s.get("axis_xyxy_bottom_to_top")
        if axis is None:
            axis_str = "unknown"
        else:
            axis_str = ", ".join(f"{v:.1f}" for v in axis)
        lines.append(
            f"- stem_id={s['stem_id']}, segment_id={s['segment_id']}, "
            f"axis_bottom_to_top=[{axis_str}], bbox={['%.1f' % v for v in s['bbox_xyxy']]}, "
            f"confidence={s['confidence']:.3f}"
        )

    return f"""
You are labeling plant topology for robot pruning data.
The attached image is the original plant image with numbered red stem candidates.
Only label the numbered stem candidates. Leaves are only context.

Definitions:
- main_stem: the primary continuous stem/trunk axis of the plant, usually the root-to-top backbone.
- side_branch: a lateral/offshoot/sub-branch that departs from the main stem.
- If a segment is ambiguous or too occluded, choose side_branch unless it is clearly part of the main backbone.
- The white/yellow line drawn inside a red mask is the bottom-to-top axis estimate for that stem candidate.

Stem candidates:
{chr(10).join(lines)}

Return strict JSON only. Do not include markdown.
Required schema:
{{
  "image_id": "{segment_json['image_id']}",
  "labels": [
    {{"stem_id": 1, "label": "main_stem", "confidence": 0.0, "reason": "short reason"}},
    {{"stem_id": 2, "label": "side_branch", "confidence": 0.0, "reason": "short reason"}}
  ]
}}

The label value must be exactly one of: "main_stem", "side_branch".
Include every stem_id exactly once.
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def response_body_to_text(body: dict[str, Any]) -> str:
    # Batch 결과는 raw body로 돌아오니, 여기서 실제 답변 text만 꺼냅니다.
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    texts: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype in {"output_text", "text"}:
                text = content.get("text") or content.get("content")
                if isinstance(text, str):
                    texts.append(text)
    if texts:
        return "\n".join(texts)

    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content

    raise ValueError(f"Could not find output text in response body keys={list(body.keys())}")


def canonical_label(value: str) -> str:
    v = str(value).strip().lower().replace("-", "_")
    if v in MAIN_ALIASES:
        return "main_stem"
    if v in SIDE_ALIASES:
        return "side_branch"
    if "main" in v or "primary" in v:
        return "main_stem"
    if "side" in v or "branch" in v or "lateral" in v:
        return "side_branch"
    raise ValueError(f"Unknown label: {value}")


def normalize_response(payload: dict[str, Any], segment_json: dict[str, Any], raw_text: str) -> dict[str, Any]:
    expected_stem_ids = sorted(int(s["stem_id"]) for s in segment_json["segments"] if s["kind"] == "stem")
    seen: dict[int, dict[str, Any]] = {}
    for item in payload.get("labels", []):
        stem_id = int(item["stem_id"])
        seen[stem_id] = {
            "stem_id": stem_id,
            "label": canonical_label(item["label"]),
            "confidence": float(item.get("confidence", 0.0)),
            "reason": str(item.get("reason", ""))[:300],
        }

    labels = []
    for stem_id in expected_stem_ids:
        if stem_id in seen:
            labels.append(seen[stem_id])
        else:
            labels.append({
                "stem_id": stem_id,
                "label": "side_branch",
                "confidence": 0.0,
                "reason": "missing from LLM output; defaulted to side_branch",
            })

    return {
        "schema_version": "plant-llm-stem-labels-v1",
        "image_id": segment_json["image_id"],
        "source_segments_json": str(segment_json.get("segments_json_path", "")),
        "labels": labels,
        "raw_model_output": raw_text,
    }


def openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def openai_obj_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "model_dump_json"):
        return json.loads(obj.model_dump_json())
    return json.loads(json.dumps(obj, default=str))


def file_response_to_text(file_response: Any) -> str:
    text_attr = getattr(file_response, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        text = text_attr()
        if isinstance(text, str):
            return text
    content_attr = getattr(file_response, "content", None)
    if isinstance(content_attr, bytes):
        return content_attr.decode("utf-8")
    if callable(content_attr):
        content = content_attr()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        if isinstance(content, str):
            return content
    if hasattr(file_response, "read"):
        data = file_response.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
    raise TypeError(f"Unsupported file content response type: {type(file_response)}")


def call_openai_vision(api_image_path: Path, prompt: str, model: str, temperature: float, max_output_tokens: int) -> tuple[dict[str, Any], str]:
    client = openai_client()
    data_url = image_to_data_url(api_image_path)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    raw_text = response.output_text
    return extract_json_object(raw_text), raw_text


def iter_segment_paths(args) -> list[Path]:
    seg_dir = args.dataset / "segments"
    if not seg_dir.exists():
        raise FileNotFoundError(f"segments directory not found: {seg_dir}")

    paths = sorted(seg_dir.glob("*.json"))
    if args.max_images and args.max_images > 0:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError(f"No segment JSON files found: {seg_dir}")
    return paths


def make_overlay_and_request(json_path: Path, args) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    segment_json = load_json(json_path)
    segment_json["segments_json_path"] = str(json_path.resolve())
    image_id = segment_json["image_id"]

    overlay_path = args.dataset / "llm_overlays" / f"{image_id}_numbered.png"
    api_image_path = args.dataset / "llm_overlays" / f"{image_id}_numbered_api.png"
    out_label_path = args.dataset / "llm_labels" / f"{image_id}.json"

    if out_label_path.exists() and not args.overwrite:
        print(f"Skip existing label: {out_label_path}")
        return None, None

    stems = [s for s in segment_json["segments"] if s["kind"] == "stem"]
    if not stems:
        print(f"[{image_id}] no stems; skip")
        return None, None

    draw_numbered_stem_overlay(segment_json, overlay_path, api_image_path, args.max_side)
    prompt = build_prompt(segment_json)
    data_url = image_to_data_url(api_image_path)

    custom_id = image_id
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": args.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
        },
    }
    index_record = {
        "custom_id": custom_id,
        "image_id": image_id,
        "segment_json_path": str(json_path.resolve()),
        "numbered_overlay_path": str(overlay_path.resolve()),
        "api_overlay_path": str(api_image_path.resolve()),
        "output_label_path": str(out_label_path.resolve()),
        "model": args.model,
    }
    return request, index_record


def process_one_sequential(json_path: Path, args) -> None:
    segment_json = load_json(json_path)
    segment_json["segments_json_path"] = str(json_path.resolve())
    image_id = segment_json["image_id"]

    overlay_path = args.dataset / "llm_overlays" / f"{image_id}_numbered.png"
    api_image_path = args.dataset / "llm_overlays" / f"{image_id}_numbered_api.png"
    out_label_path = args.dataset / "llm_labels" / f"{image_id}.json"

    if out_label_path.exists() and not args.overwrite:
        print(f"Skip existing label: {out_label_path}")
        return

    stems = [s for s in segment_json["segments"] if s["kind"] == "stem"]
    if not stems:
        print(f"[{image_id}] no stems; skip")
        return

    draw_numbered_stem_overlay(segment_json, overlay_path, api_image_path, args.max_side)
    prompt = build_prompt(segment_json)

    if args.dry_run:
        print(f"[{image_id}] dry-run overlay={overlay_path}")
        return

    parsed, raw_text = call_openai_vision(
        api_image_path=api_image_path,
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    normalized = normalize_response(parsed, segment_json, raw_text)
    normalized["numbered_overlay_path"] = str(overlay_path.resolve())
    normalized["api_overlay_path"] = str(api_image_path.resolve())
    normalized["model"] = args.model
    normalized["labeling_mode"] = "sequential"
    save_json(out_label_path, normalized)
    print(f"[{image_id}] saved {out_label_path}")
    if args.sleep > 0:
        time.sleep(args.sleep)


def batch_dir(args) -> Path:
    return args.dataset / "llm_batch"


def batch_manifest_path(args) -> Path:
    return batch_dir(args) / "batch_manifest.json"


def batch_input_path(args, batch_no: int | None = None) -> Path:
    if batch_no is None:
        return batch_dir(args) / "batch_input.jsonl"
    return batch_dir(args) / f"batch_input_{batch_no:06d}.jsonl"


def batch_index_path(args, batch_no: int | None = None) -> Path:
    if batch_no is None:
        return batch_dir(args) / "batch_index.json"
    return batch_dir(args) / f"batch_index_{batch_no:06d}.json"


def batch_state_path(args, batch_no: int | None = None) -> Path:
    if batch_no is None:
        return batch_dir(args) / "batch_state.json"
    return batch_dir(args) / f"batch_state_{batch_no:06d}.json"


def batch_output_path(args, batch_no: int | None = None) -> Path:
    if batch_no is None:
        return batch_dir(args) / "batch_output.jsonl"
    return batch_dir(args) / f"batch_output_{batch_no:06d}.jsonl"


def batch_error_path(args, batch_no: int | None = None) -> Path:
    if batch_no is None:
        return batch_dir(args) / "batch_errors.jsonl"
    return batch_dir(args) / f"batch_errors_{batch_no:06d}.jsonl"


def jsonl_row_bytes(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_jsonl_bytes(path: Path, rows: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            f.write(row)


def prepare_batch_inputs(args) -> list[dict[str, Any]]:
    # overlay 이미지를 base64로 넣으면 금방 커져서, 실제 JSONL byte 기준으로 잘라 냅니다.
    max_bytes = int(float(args.max_batch_file_mb) * 1024 * 1024)
    if max_bytes <= 0:
        raise ValueError("--max-batch-file-mb must be positive.")

    max_requests = int(args.max_requests_per_batch)
    if max_requests <= 0:
        raise ValueError("--max-requests-per-batch must be positive.")

    batch_dir(args).mkdir(parents=True, exist_ok=True)

    batches: list[dict[str, Any]] = []
    current_rows: list[bytes] = []
    current_items: dict[str, Any] = {}
    current_size = 0
    seen_custom_ids: set[str] = set()
    total_requests = 0

    def flush_current() -> None:
        nonlocal current_rows, current_items, current_size, total_requests
        if not current_rows:
            return

        batch_no = len(batches) + 1
        input_path = batch_input_path(args, batch_no)
        index_path = batch_index_path(args, batch_no)
        state_path = batch_state_path(args, batch_no)
        output_path = batch_output_path(args, batch_no)
        error_path = batch_error_path(args, batch_no)

        write_jsonl_bytes(input_path, current_rows)
        size_bytes = input_path.stat().st_size

        index = {
            "schema_version": "plant-llm-batch-index-v2",
            "endpoint": "/v1/responses",
            "model": args.model,
            "batch_no": batch_no,
            "created_at_unix": int(time.time()),
            "num_requests": len(current_rows),
            "size_bytes": size_bytes,
            "items": current_items,
        }
        save_json(index_path, index)

        batches.append({
            "batch_no": batch_no,
            "input_path": str(input_path.resolve()),
            "index_path": str(index_path.resolve()),
            "state_path": str(state_path.resolve()),
            "output_path": str(output_path.resolve()),
            "error_path": str(error_path.resolve()),
            "num_requests": len(current_rows),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 4),
            "status": "prepared",
        })
        total_requests += len(current_rows)

        current_rows = []
        current_items = {}
        current_size = 0

    for json_path in iter_segment_paths(args):
        try:
            request, index_record = make_overlay_and_request(json_path, args)
        except Exception as exc:
            print(f"ERROR while preparing [{json_path.name}]: {exc}")
            continue

        if request is None or index_record is None:
            continue

        custom_id = str(request["custom_id"])
        if custom_id in seen_custom_ids:
            raise ValueError(
                f"Duplicate custom_id found: {custom_id}. "
                "Image stems may have duplicate file stems. Rename images or adjust image_id generation."
            )
        seen_custom_ids.add(custom_id)

        row = jsonl_row_bytes(request)
        row_size = len(row)

        if row_size > max_bytes:
            raise ValueError(
                f"One request is larger than --max-batch-file-mb: custom_id={custom_id}, "
                f"row_size={row_size / (1024 * 1024):.2f} MB, "
                f"limit={args.max_batch_file_mb:.2f} MB. "
                "Reduce --max-side, or change the script to use externally hosted image_url instead of base64 data URLs."
            )

        if current_rows and (
            current_size + row_size > max_bytes
            or len(current_rows) >= max_requests
        ):
            flush_current()

        current_rows.append(row)
        current_items[custom_id] = index_record
        current_size += row_size

    flush_current()

    if not batches:
        raise ValueError("No batch requests were created. Check existing labels, stem detections, or --overwrite.")

    manifest = {
        "schema_version": "plant-llm-batch-manifest-v2",
        "endpoint": "/v1/responses",
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "created_at_unix": int(time.time()),
        "max_batch_file_mb": float(args.max_batch_file_mb),
        "max_requests_per_batch": int(args.max_requests_per_batch),
        "num_batches": len(batches),
        "total_requests": total_requests,
        "batches": batches,
    }
    save_json(batch_manifest_path(args), manifest)

    print(f"Prepared {len(batches)} batch input file(s), total_requests={total_requests}")
    for entry in batches:
        print(
            f"  batch_no={entry['batch_no']:06d} "
            f"requests={entry['num_requests']} "
            f"size={entry['size_mb']:.2f} MB "
            f"path={entry['input_path']}"
        )
    print(f"Prepared manifest: {batch_manifest_path(args)}")

    return batches


def load_batch_manifest(args) -> dict[str, Any]:
    manifest_path = batch_manifest_path(args)
    if not manifest_path.exists():
        old_state = batch_state_path(args)
        old_index = batch_index_path(args)
        old_input = batch_input_path(args)
        if old_state.exists():
            state = load_json(old_state)
            batch = state.get("batch", {})
            return {
                "schema_version": "plant-llm-batch-manifest-compat-v1",
                "endpoint": "/v1/responses",
                "model": state.get("batch", {}).get("model", args.model),
                "dataset": str(args.dataset.resolve()),
                "num_batches": 1,
                "total_requests": int(state.get("num_requests", 0)),
                "batches": [{
                    "batch_no": 1,
                    "input_path": str(old_input.resolve()),
                    "index_path": str(old_index.resolve()),
                    "state_path": str(old_state.resolve()),
                    "output_path": str(batch_output_path(args).resolve()),
                    "error_path": str(batch_error_path(args).resolve()),
                    "num_requests": int(state.get("num_requests", 0)),
                    "size_bytes": old_input.stat().st_size if old_input.exists() else 0,
                    "batch_id": batch.get("id") or state.get("batch_id"),
                    "status": batch.get("status"),
                }],
            }
        raise FileNotFoundError(
            f"Batch manifest not found: {manifest_path}. Run --mode batch-submit first."
        )
    return load_json(manifest_path)


def save_batch_manifest(args, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") == "plant-llm-batch-manifest-compat-v1":
        return
    manifest["last_updated_at_unix"] = int(time.time())
    save_json(batch_manifest_path(args), manifest)


def submit_batch(args) -> None:
    batches = prepare_batch_inputs(args)

    if args.dry_run:
        print("dry-run: not uploading batch files and not creating batches.")
        return

    client = openai_client()
    manifest = load_batch_manifest(args)

    for entry in manifest["batches"]:
        batch_no = int(entry["batch_no"])
        input_path = Path(entry["input_path"])
        index_path = Path(entry["index_path"])
        state_path = Path(entry["state_path"])
        num_requests = int(entry["num_requests"])

        print(f"Uploading batch_no={batch_no:06d} JSONL with purpose='batch'...")
        with input_path.open("rb") as f:
            input_file = client.files.create(file=f, purpose="batch")

        input_file_dict = openai_obj_to_dict(input_file)
        input_file_id = input_file_dict.get("id")
        if not input_file_id:
            raise RuntimeError(f"Could not find uploaded file id in: {input_file_dict}")

        print(f"Creating Batch job for batch_no={batch_no:06d}, requests={num_requests}...")
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": "plant stem main/side labeling",
                "dataset": str(args.dataset.resolve()),
                "model": args.model,
                "batch_no": str(batch_no),
            },
        )
        batch_dict = openai_obj_to_dict(batch)

        state = {
            "schema_version": "plant-llm-batch-state-v2",
            "mode": "batch-submit",
            "dataset": str(args.dataset.resolve()),
            "batch_no": batch_no,
            "batch": batch_dict,
            "input_file": input_file_dict,
            "batch_input_path": str(input_path.resolve()),
            "batch_index_path": str(index_path.resolve()),
            "num_requests": num_requests,
        }
        save_json(state_path, state)

        entry.update({
            "batch_id": batch_dict.get("id"),
            "status": batch_dict.get("status"),
            "input_file_id": input_file_id,
            "output_file_id": batch_dict.get("output_file_id"),
            "error_file_id": batch_dict.get("error_file_id"),
        })
        save_batch_manifest(args, manifest)

        print(f"  submitted batch_id={batch_dict.get('id')} status={batch_dict.get('status')}")

    print(f"Submitted {len(manifest['batches'])} batch job(s).")
    print(f"Saved manifest: {batch_manifest_path(args)}")


def get_saved_batch_id(args) -> str:
    if args.batch_id:
        return args.batch_id
    state_path = batch_state_path(args)
    if not state_path.exists():
        raise FileNotFoundError(f"Batch state not found: {state_path}. Pass --batch-id or run --mode batch-submit first.")
    state = load_json(state_path)
    batch_id = state.get("batch", {}).get("id") or state.get("batch_id")
    if not batch_id:
        raise ValueError(f"Could not find batch id in {state_path}")
    return str(batch_id)


def refresh_batch_entry(args, entry: dict[str, Any], client=None) -> dict[str, Any] | None:
    if client is None:
        client = openai_client()

    batch_id = entry.get("batch_id")
    state_path = Path(entry["state_path"])
    if not batch_id and state_path.exists():
        state = load_json(state_path)
        batch_id = state.get("batch", {}).get("id") or state.get("batch_id")

    if not batch_id:
        return None

    batch = client.batches.retrieve(str(batch_id))
    batch_dict = openai_obj_to_dict(batch)

    old_state = load_json(state_path) if state_path.exists() else {}
    old_state.update({
        "schema_version": "plant-llm-batch-state-v2",
        "dataset": str(args.dataset.resolve()),
        "batch_no": int(entry.get("batch_no", 1)),
        "batch": batch_dict,
        "last_refreshed_at_unix": int(time.time()),
    })
    save_json(state_path, old_state)

    entry.update({
        "batch_id": batch_dict.get("id"),
        "status": batch_dict.get("status"),
        "output_file_id": batch_dict.get("output_file_id"),
        "error_file_id": batch_dict.get("error_file_id"),
        "request_counts": batch_dict.get("request_counts"),
    })
    return batch_dict


def refresh_all_batch_states(args) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
    manifest = load_batch_manifest(args)
    client = openai_client()
    batch_dicts: list[dict[str, Any] | None] = []
    for entry in manifest.get("batches", []):
        batch_dicts.append(refresh_batch_entry(args, entry, client=client))
    save_batch_manifest(args, manifest)
    return manifest, batch_dicts


def print_batch_status(args) -> list[dict[str, Any] | None]:
    if args.batch_id:
        client = openai_client()
        batch = client.batches.retrieve(args.batch_id)
        batch_dict = openai_obj_to_dict(batch)
        counts = batch_dict.get("request_counts") or {}
        print(f"Batch id: {batch_dict.get('id')}")
        print(f"Status: {batch_dict.get('status')}")
        print(f"Requests: total={counts.get('total')} completed={counts.get('completed')} failed={counts.get('failed')}")
        print(f"output_file_id: {batch_dict.get('output_file_id')}")
        print(f"error_file_id: {batch_dict.get('error_file_id')}")
        return [batch_dict]

    manifest, batch_dicts = refresh_all_batch_states(args)
    total_batches = len(manifest.get("batches", []))
    total_requests = 0
    total_completed = 0
    total_failed = 0

    print(f"Manifest: {batch_manifest_path(args)}")
    print(f"Batches: {total_batches}")
    for entry, batch_dict in zip(manifest.get("batches", []), batch_dicts):
        batch_no = int(entry.get("batch_no", 1))
        status = entry.get("status") or "not_submitted"
        batch_id = entry.get("batch_id")
        counts = (batch_dict or {}).get("request_counts") or entry.get("request_counts") or {}
        total = int(counts.get("total") or entry.get("num_requests") or 0)
        completed = int(counts.get("completed") or 0)
        failed = int(counts.get("failed") or 0)
        total_requests += total
        total_completed += completed
        total_failed += failed
        print(
            f"  [{batch_no:06d}] status={status} "
            f"requests={total} completed={completed} failed={failed} "
            f"batch_id={batch_id}"
        )
        if entry.get("output_file_id") or entry.get("error_file_id"):
            print(
                f"           output_file_id={entry.get('output_file_id')} "
                f"error_file_id={entry.get('error_file_id')}"
            )

    print(
        f"Total requests: {total_requests}, completed={total_completed}, failed={total_failed}"
    )
    return batch_dicts


def download_batch_file(client, file_id: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_response = client.files.content(file_id)
    text = file_response_to_text(file_response)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def fetch_batch_results(args) -> None:
    if args.batch_id:
        raise ValueError(
            "--batch-id fetch is not supported without local index metadata. "
            "Use the dataset manifest created by --mode batch-submit."
        )

    manifest, _ = refresh_all_batch_states(args)
    client = openai_client()

    not_ready: list[str] = []
    parsed_total = 0

    for entry in manifest.get("batches", []):
        batch_no = int(entry.get("batch_no", 1))
        status = str(entry.get("status") or "")
        batch_id = entry.get("batch_id")

        if not batch_id:
            print(f"[{batch_no:06d}] not submitted; skip")
            continue

        if status != "completed" and not args.force_fetch:
            not_ready.append(f"{batch_no:06d}:{status}")
            print(f"[{batch_no:06d}] not completed yet: {status}")
            continue

        output_file_id = entry.get("output_file_id")
        error_file_id = entry.get("error_file_id")
        output_path = Path(entry.get("output_path") or batch_output_path(args, batch_no))
        error_path = Path(entry.get("error_path") or batch_error_path(args, batch_no))

        if output_file_id:
            out_path = download_batch_file(client, str(output_file_id), output_path)
            print(f"[{batch_no:06d}] downloaded output: {out_path}")
        else:
            print(f"[{batch_no:06d}] no output_file_id found.")

        if error_file_id:
            err_path = download_batch_file(client, str(error_file_id), error_path)
            print(f"[{batch_no:06d}] downloaded errors: {err_path}")

        if output_path.exists():
            saved, errors = parse_batch_output_file(
                args=args,
                index_path=Path(entry["index_path"]),
                output_path=output_path,
                batch_no=batch_no,
            )
            parsed_total += saved
        else:
            print(f"[{batch_no:06d}] output file does not exist yet: {output_path}")

    if not_ready and not args.force_fetch:
        raise RuntimeError(
            "Some batches are not completed yet: " + ", ".join(not_ready) +
            ". Use --mode batch-status, --mode batch-wait, or --force-fetch."
        )

    print(f"Fetch complete. parsed_saved_total={parsed_total}")


def parse_batch_output(args) -> None:
    manifest = load_batch_manifest(args)
    total_saved = 0
    total_errors = 0
    for entry in manifest.get("batches", []):
        batch_no = int(entry.get("batch_no", 1))
        output_path = Path(entry.get("output_path") or batch_output_path(args, batch_no))
        if not output_path.exists():
            print(f"[{batch_no:06d}] skip; output not found: {output_path}")
            continue
        saved, errors = parse_batch_output_file(
            args=args,
            index_path=Path(entry["index_path"]),
            output_path=output_path,
            batch_no=batch_no,
        )
        total_saved += saved
        total_errors += errors
    print(f"Batch output parsed. saved={total_saved}, errors={total_errors}")


def parse_batch_output_file(args, index_path: Path, output_path: Path, batch_no: int) -> tuple[int, int]:
    if not index_path.exists():
        raise FileNotFoundError(f"Batch index not found: {index_path}")
    index = load_json(index_path)
    items = index.get("items", {})

    rows = read_jsonl(output_path)
    saved = 0
    errors = 0

    for row in rows:
        custom_id = str(row.get("custom_id"))
        index_record = items.get(custom_id)
        if index_record is None:
            print(f"WARN: custom_id not found in index: {custom_id}")
            continue

        image_id = index_record["image_id"]
        out_label_path = Path(index_record["output_label_path"])

        error = row.get("error")
        response = row.get("response")
        if error:
            errors += 1
            err_path = args.dataset / "llm_labels" / f"{image_id}.error.json"
            save_json(err_path, {"custom_id": custom_id, "batch_no": batch_no, "error": error, "row": row})
            print(f"[{image_id}] batch row error -> {err_path}")
            continue

        if not response or int(response.get("status_code", 0)) >= 400:
            errors += 1
            err_path = args.dataset / "llm_labels" / f"{image_id}.error.json"
            save_json(err_path, {"custom_id": custom_id, "batch_no": batch_no, "response": response, "row": row})
            print(f"[{image_id}] bad response -> {err_path}")
            continue

        body = response.get("body", {})
        try:
            raw_text = response_body_to_text(body)
            parsed = extract_json_object(raw_text)
            segment_json = load_json(Path(index_record["segment_json_path"]))
            segment_json["segments_json_path"] = index_record["segment_json_path"]
            normalized = normalize_response(parsed, segment_json, raw_text)
            normalized["numbered_overlay_path"] = index_record["numbered_overlay_path"]
            normalized["api_overlay_path"] = index_record["api_overlay_path"]
            normalized["model"] = index_record.get("model", index.get("model"))
            normalized["labeling_mode"] = "batch"
            normalized["batch_no"] = batch_no
            normalized["batch_custom_id"] = custom_id
            normalized["batch_response_id"] = body.get("id")
            save_json(out_label_path, normalized)
            saved += 1
            print(f"[{image_id}] saved {out_label_path}")
        except Exception as exc:
            errors += 1
            err_path = args.dataset / "llm_labels" / f"{image_id}.error.json"
            save_json(err_path, {"custom_id": custom_id, "batch_no": batch_no, "exception": str(exc), "row": row})
            print(f"[{image_id}] parse error -> {err_path}: {exc}")

    print(f"[{batch_no:06d}] parsed. saved={saved}, errors={errors}")
    return saved, errors


def wait_for_batch(args) -> None:
    deadline = time.time() + max(1, args.timeout_minutes) * 60
    last_batch_dicts: list[dict[str, Any] | None] = []

    while True:
        last_batch_dicts = print_batch_status(args)
        statuses = [str((bd or {}).get("status") or "not_submitted") for bd in last_batch_dicts]
        if statuses and all(status in TERMINAL_BATCH_STATES for status in statuses):
            break
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for batch after {args.timeout_minutes} minutes.")
        time.sleep(max(5, args.poll_interval))

    if last_batch_dicts and all(str((bd or {}).get("status")) == "completed" for bd in last_batch_dicts):
        fetch_batch_results(args)
    else:
        print("At least one batch ended with non-completed status. Run --mode batch-status for details.")


def run_sequential(args) -> None:
    for p in iter_segment_paths(args):
        try:
            process_one_sequential(p, args)
        except Exception as exc:
            print(f"ERROR [{p.name}]: {exc}")
            if not args.dry_run:
                err_path = args.dataset / "llm_labels" / f"{p.stem}.error.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(str(exc), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("plant_dataset"), help="Dataset root made by 01_sam3_leaf_stem.py")
    parser.add_argument("--mode", choices=["batch-submit", "batch-status", "batch-fetch", "batch-wait", "sequential"], default="batch-submit")
    parser.add_argument("--batch-id", default="", help="For batch-status only: inspect one remote Batch id directly. Normal fetch/wait uses dataset/llm_batch/batch_manifest.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-side", type=int, default=720, help="Resize overlay for API if longer side exceeds this")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-batch-file-mb", type=float, default=180.0, help="Maximum JSONL file size per Batch input. The script actively splits into multiple batch_input_*.jsonl files before this limit.")
    parser.add_argument("--max-requests-per-batch", type=int, default=50000, help="Maximum number of requests per generated Batch input file.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sequential mode only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="For batch-submit, creates overlay and JSONL only. For sequential, no API call.")
    parser.add_argument("--poll-interval", type=int, default=60, help="batch-wait polling interval in seconds")
    parser.add_argument("--timeout-minutes", type=int, default=180, help="batch-wait timeout")
    parser.add_argument("--force-fetch", action="store_true", help="Download/parse available output even if batch is not completed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "batch-submit":
        submit_batch(args)
    elif args.mode == "batch-status":
        print_batch_status(args)
    elif args.mode == "batch-fetch":
        fetch_batch_results(args)
    elif args.mode == "batch-wait":
        wait_for_batch(args)
    elif args.mode == "sequential":
        run_sequential(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
