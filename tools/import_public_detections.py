#!/usr/bin/env python3
"""把公开数据集的检测标注/推理结果导入为 wm_kit replay JSONL。

说明：
  - 该脚本只做格式转换，不把第三方格式耦合进 WorldModel。
  - 支持两种输入：
      1. generic: 已经是逐帧检测 JSONL（每行 frame_id/timestamp/image_width/
         image_height/robot_pose/detections，bbox 为原始图像 xyxy）。
      2. coco:    COCO 风格检测 JSON（images + annotations，bbox 为
         [x, y, width, height]，转换到原始图像 xyxy）。
  - 输出 JSONL 第一行为 time_origin meta，后续每行一个 replay frame。
  - 只转换，不下载、不标注、不伪造“动作成功”标签。

用法：
    python3 tools/import_public_detections.py \
      --input detections.jsonl --format generic \
      --calibration configs/overhead_camera.example.json \
      --output scenes/imported_replay.jsonl \
      --time-origin 1786417200.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from world_model.calibration import CameraCalibration, load_camera_calibration
from world_model.types import RobotPose


def convert_generic(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return records


def convert_coco(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        by_image.setdefault(ann["image_id"], []).append({
            "class_name": ann.get("category_name") or str(ann.get("category_id", "object")),
            "confidence": float(ann.get("score", 1.0)),
            "bbox": [x, y, x + w, y + h],
        })
    frames: List[Dict[str, Any]] = []
    for idx, img in enumerate(images):
        frames.append({
            "frame_id": str(img.get("file_name", img.get("id", idx))),
            "timestamp": float(idx),
            "image_width": int(img["width"]),
            "image_height": int(img["height"]),
            "robot_pose": {"x": 0.0, "z": 0.0, "yaw_rad": 0.0, "pose_uncertainty_cm": 0.0},
            "detections": by_image.get(img["id"], []),
        })
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
        frame = dict(frame)
        frame.setdefault("robot_pose", {})
        frame.setdefault("timestamp", float(frame.get("timestamp", 0.0)))
        # 只做格式转换，不在此处重新投影；投影由 CameraDetectorProvider 完成。
        out_lines.append(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"calibration={calibration.name}")
    print(f"frames={len(frames)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
