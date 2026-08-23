#!/usr/bin/env python3
"""把公开数据集的检测标注/推理结果导入为 wm_kit replay JSONL。

该脚本只做格式转换，不把第三方格式耦合进 WorldModel。
输出字段白名单：
  帧级: frame_id, timestamp, image_width, image_height, robot_pose, detections
  检测级: class_name, confidence, bbox, camera_id
不得透传 radius_cm / size_source / size_trusted / calibration_trusted /
frame_quality / coordinate_frame。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from world_model.calibration import load_camera_calibration

FRAME_WHITELIST = {"frame_id", "timestamp", "image_width", "image_height", "robot_pose", "detections"}
DETECTION_WHITELIST = {"class_name", "confidence", "bbox", "camera_id"}


def convert_generic(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            print(f"WARNING: 跳过非对象 frame: {type(rec).__name__}")
            continue
        frame = {k: rec[k] for k in FRAME_WHITELIST if k in rec}
        if "detections" in rec:
            dets = []
            for d in rec["detections"]:
                if not isinstance(d, dict):
                    print("WARNING: 跳过非对象 detection")
                    continue
                dets.append({k: d[k] for k in DETECTION_WHITELIST if k in d})
            frame["detections"] = dets
        if "robot_pose" not in frame:
            frame["robot_pose"] = None
        frames.append(frame)
    return frames


def convert_coco(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    categories = {c["id"]: c.get("name") for c in data.get("categories", [])}
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    clamp_count = 0
    reject_count = 0
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        image = next((img for img in images if img.get("id") == ann.get("image_id")), None)
        if image is None:
            reject_count += 1
            continue
        width = int(image["width"])
        height = int(image["height"])
        x1, y1, x2, y2 = x, y, x + w, y + h
        tol = 1.0
        if x1 < -tol or y1 < -tol or x2 > width + tol or y2 > height + tol:
            reject_count += 1
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            clamp_count += 1
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y2 = max(0.0, min(float(height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            reject_count += 1
            continue
        category_id = ann.get("category_id")
        category_name = categories.get(category_id)
        if category_name is None:
            reject_count += 1
            print(f"WARNING: 缺少 category_id={category_id!r} 的名称，已拒绝")
            continue
        by_image.setdefault(ann["image_id"], []).append({
            "class_name": category_name,
            "confidence": float(ann.get("score", 1.0)),
            "bbox": [x1, y1, x2, y2],
            "camera_id": "overhead",
        })
    frames: List[Dict[str, Any]] = []
    for idx, img in enumerate(images):
        detections = by_image.get(img.get("id"), [])
        if not detections:
            continue
        frames.append({
            "frame_id": str(img.get("file_name", img.get("id", idx))),
            "timestamp": float(idx),
            "image_width": int(img["width"]),
            "image_height": int(img["height"]),
            "robot_pose": None,
            "detections": detections,
        })
    print(f"coco_clamp_count={clamp_count} coco_reject_count={reject_count}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("generic", "coco"), default="generic")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-origin", type=float, default=1786417200.0)
    args = parser.parse_args()

    calibration = load_camera_calibration(args.calibration)
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if args.format == "coco":
        frames = convert_coco(raw)
    else:
        frames = convert_generic(raw if isinstance(raw, list) else raw.get("frames", []))

    out_lines = [json.dumps({"time_origin": args.time_origin})]
    for frame in frames:
        out_lines.append(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"calibration={calibration.name}")
    print(f"frames={len(frames)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
