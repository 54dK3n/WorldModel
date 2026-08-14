# wm_kit — World Model + 视觉判定内核骨架

嘲风项目 · 杨铮 · 2026-07-31

设计说明与讨论结论见 **[DESIGN.md](DESIGN.md)**。

## 跑起来

```bash
python run_demo.py                   # 端到端：mock感知 -> World Model -> 契约 -> Judge evidence
python -m pytest tests/ -q           # 49 passed
python tools/false_verdict_probe.py  # 判定可靠性探针：16 PASS / 0 FAIL / 1 已知边界
```

零外部依赖（仅 pytest 用于测试）。

## 目录

```
wm_kit/
├── DESIGN.md                     方案总结、边界、待确认清单
├── world_model/
│   ├── types.py                  Detection / TrackedObject / RobotPose / 状态机
│   ├── aliases.py                #5 别名表
│   ├── association.py            #3 关联：代价矩阵 + 门控 + 贪心
│   ├── decay.py                  #4 时效：FOV 判断 + 双模衰减
│   ├── core.py                   WorldModel 主体（update/get_scene/get_object/snapshot）
│   ├── adapters.py               导出 scene_observations 契约
│   └── providers/
│       ├── base.py               provider 抽象 + 相机provider桩（#1检测 #2地平面求交）
│       └── mock.py               读 JSON 场景序列，零硬件
├── judge/
│   ├── evidence.py               填 evidence 字段：身份锁定 + 前置条件 + reason code
│   └── providers.py              WorldModelDiff(已实现) / RewardClassifier(待填) / YoloOverlap(备选)
├── scenes/demo_scene.json        覆盖四类验收场景的 mock 观测序列
├── tools/false_verdict_probe.py  判定可靠性探针：对抗场景，查虚假判定
├── logs/                         各次运行的输出记录
└── tests/
    ├── test_world_model.py       20 条：信念维护的不变式
    └── test_judge.py             29 条：每条对应一个曾经的虚假判定
```

判定内核的设计与修复记录见 **[DESIGN.md 第十节](DESIGN.md)**，
判定可靠性复核报告见 **[2026-08-11_判定可靠性review.md](2026-08-11_判定可靠性review.md)**。

## 我的任务

两块：**World Model**（维护世界状态信念，回答"现在场景里有什么、分别在哪"）和**视觉判定内核**（回答"这件事成没成"，填 `judge_service` 里的 `RewardClassifierProvider` 与 `evidence` 字段）。

## 换数据源不改核心

```
MockProvider -> 曹志伟 SO101 双摄 -> 真机 RGBD
```
实现 `PerceptionProvider.stream()` 即可，`WorldModel` 一行不动。
