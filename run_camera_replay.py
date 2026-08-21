#!/usr/bin/env python3
"""离线回放：外部检测结果 -> CameraDetectorProvider -> WorldModel -> scene_observations。

用法：
    python run_camera_replay.py \
      --detections scenes/tennis_detection_replay.jsonl \
      --calibration configs/overhead_camera.example.json

输出每帧的 frame_id、原始 bbox/confidence、转换后的 x/z，以及更新后的
obj_id / state / last_seen / scene_observations。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from world_model import WorldModel, to_scene_observation
from world_model.providers import CameraDetectorProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="JSON/JSONL 检测帧回放文件",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="相机标定 JSON（测试用途示例见 configs/overhead_camera.example.json）",
    )
    parser.add_argument(
        "--time-origin",
        type=float,
        default=0.0,
        help="回放相对时刻 0.0 对应的 Unix 时间；默认 0.0 表示只使用相对秒",
    )
    args = parser.parse_args()

    provider = CameraDetectorProvider(
        calibration_path=args.calibration,
        replay_path=args.detections,
        time_origin=args.time_origin,
    )

    stream = provider.stream()
    try:
        first = next(stream)
    except StopIteration:
        print("回放文件没有帧。")
        return

    # stream() 启动后才从文件里解析出 time_origin（JSONL 首行可带 meta）。
    wm = WorldModel(time_origin=provider.time_origin)
    print(f"calibration: {args.calibration}")
    print(f"replay: {args.detections}")
    print(f"time_origin: {provider.time_origin}（相对秒 -> Unix 由 scene_observations 负责）")
    print("=" * 100)

    def process_one(ts: float, pose, dets) -> None:
        print(f"\n[frame {dets[0].frame_id if dets else '<none>'}] "
              f"t={ts:.2f}s  robot=({pose.x:.2f},{pose.z:.2f},{pose.yaw_rad:.2f})  "
              f"detections={len(dets)}")

        for d in dets:
            x1, y1, x2, y2 = d.bbox
            print(
                f"  raw  class={d.class_name:<14s} conf={d.confidence:.2f} "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                f"-> x={d.x:+.3f} z={d.z:.3f}  frame_id={d.frame_id}"
            )

        try:
            wm.update(dets, pose, now=ts)
        except Exception as exc:  # provider/关联异常不允许丢 WorldModel 状态
            print(f"  !!! update failed: {exc}（WorldModel 保持上一状态）")

        scene = wm.get_scene()
        print(f"  scene_observations: {wm.to_contract()}")
        for obj in scene:
            obs = to_scene_observation(obj, time_origin=provider.time_origin)
            print(
                f"  obj  id={obj.obj_id:<12s} name={obj.name:<8s} state={obj.state.value:<9s} "
                f"last_seen={obj.last_seen:.2f}s  x={obj.x:+.3f} z={obj.z:.3f} "
                f"conf={obj.confidence:.3f}  obs={obs}"
            )

    process_one(*first)
    for item in stream:
        process_one(*item)

    print("\n" + "=" * 100)
    print("回放完成。注意：以上 x/z 来自测试用途标定，不是真实相机标定结果。")


if __name__ == "__main__":
    main()
