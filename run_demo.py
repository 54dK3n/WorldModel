"""端到端演示：mock 感知源 -> World Model -> scene_observations -> Judge evidence

    python run_demo.py
"""
from __future__ import annotations

import json

from judge.providers import JudgeRequest, WorldModelDiffProvider
from world_model import WorldModel, to_scene_observations
from world_model.decay import is_in_fov
from world_model.providers import MockProvider


def main() -> None:
    wm = WorldModel()
    provider = MockProvider("scenes/demo_scene.json")

    before_snapshot = None
    last_ts = 0.0

    print("=" * 74)
    for ts, pose, dets in provider.stream():
        wm.update(dets, pose, now=ts)
        last_ts = ts

        print(f"\n[t={ts:>5.1f}s] 检测 {len(dets)} 个 | yaw={pose.yaw_rad:.2f}rad")
        for o in wm.get_scene():
            in_fov = is_in_fov(o.x, o.z, pose, wm.fov_cfg)
            print(
                f"   {o.obj_id:<12} conf={o.confidence:.3f} "
                f"state={o.state.value:<9} "
                f"pos=({o.x:+.2f},{o.z:+.2f}) "
                f"age={o.age(ts):.1f}s "
                f"{'视野内' if in_fov else '视野外'} "
                f"hit={o.hit_count} miss={o.miss_count}"
            )

        if abs(ts - 7.0) < 1e-6:          # 放球动作之前
            before_snapshot = wm.snapshot()

    print("\n" + "=" * 74)
    print("对外契约 scene_observations:")
    print(json.dumps(to_scene_observations(wm.get_scene()), ensure_ascii=False, indent=2))

    print("\n" + "=" * 74)
    print("Judge 判定（evidence 不再是空字典）:")
    resp = WorldModelDiffProvider().judge(
        JudgeRequest(task="put_ball_in_basket", target="ball",
                     container="basket", now=last_ts),
        before=before_snapshot or [],
        after=wm.snapshot(),
    )
    print(f"success = {resp.success}")
    print(f"detail  = {resp.detail}")
    print(json.dumps(resp.evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
