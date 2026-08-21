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
import logging
from pathlib import Path

from world_model import WorldModel, to_scene_observation
from world_model.types import ObjectState
from world_model.providers import CameraDetectorProvider

logger = logging.getLogger("camera_replay")


def configure_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


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
        default=None,
        help="回放相对时刻 0.0 对应的 Unix 时间；不传则必须由 JSON/JSONL meta 提供",
    )
    parser.add_argument(
        "--relative-time-only",
        action="store_true",
        help="显式启用仅相对时间模式；scene_observations 的 timestamp 将保持相对秒",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="可选日志文件；控制台仍会同步输出",
    )
    args = parser.parse_args()
    configure_logging(args.log_level, args.log_file)

    provider = CameraDetectorProvider(
        calibration_path=args.calibration,
        replay_path=args.detections,
        time_origin=args.time_origin,
        relative_time_only=args.relative_time_only,
    )

    stream = provider.stream()
    try:
        first = next(stream)
    except StopIteration:
        logger.warning(
            "回放没有可处理帧 loaded=%d skipped_frames=%d",
            provider.frames_loaded,
            provider.frames_skipped,
        )
        return

    # stream() 启动后才从文件里解析出 time_origin（JSONL 首行可带 meta）。
    wm = WorldModel(time_origin=provider.time_origin, visibility=provider.calibration)
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
            logger.exception("WorldModel 更新失败 frame_id=%s；保持上一状态", dets[0].frame_id if dets else "<none>")

        print(f"  scene_observations: {wm.to_contract()}")
        confirmed = [o for o in wm.get_scene() if o.state == ObjectState.CONFIRMED]
        for obj in confirmed:
            obs = to_scene_observation(obj, time_origin=provider.time_origin)
            print(
                f"  obj  id={obj.obj_id:<12s} name={obj.name:<8s} state={obj.state.value:<9s} "
                f"last_seen={obj.last_seen:.2f}s  x={obj.x:+.3f} z={obj.z:.3f} "
                f"conf={obj.confidence:.3f}  size={obj.radius_cm:.1f}cm "
                f"size_trusted={obj.size_trusted}  obs={obs}"
            )

    process_one(*first)
    for item in stream:
        process_one(*item)

    print("\n" + "=" * 100)
    logger.info(
        "回放统计 loaded=%d yielded=%d skipped_frames=%d skipped_detections=%d objects=%d",
        provider.frames_loaded,
        provider.frames_yielded,
        provider.frames_skipped,
        provider.detections_skipped,
        len(wm.get_scene()),
    )
    print("回放完成。注意：以上 x/z 来自测试用途标定，不是真实相机标定结果。")


if __name__ == "__main__":
    main()
