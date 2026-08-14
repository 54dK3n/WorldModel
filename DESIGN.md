# World Model + 视觉判定内核 — 方案总结与骨架说明

> 杨铮 · 2026-07-31 · 嘲风项目
> 本文是前期讨论结论的收敛版，附带可运行骨架 `wm_kit/`。

---

## 一、我负责什么

两块：**World Model**（世界现在是什么样）和**视觉判定内核**（这件事成没成）。

World Model 的一句话定义：

> 把**离散、带噪声、会丢帧**的感知结果，维护成一份**连续、可查询、带置信度和时效**的世界状态信念。

典型场景（也是它与"检测器"的分界）：机械臂挡住了球，这一帧检测器输出"无球"。World Model **不能**说没有球，它要说："球上次在 (0.48, 0.25)，2 秒前看到，当前可能被遮挡，置信度 0.91 → 0.6"。

我提供的是**只读事实**（`get_object("ball")` 返回"在地上、置信度 0.9"），上层拿事实自己决定下一步——数据来自我，调度决策不归我。

---

## 二、两个任务包

### 任务包一：World Model（从零做，无现成外壳）
- 状态结构定义、`get_scene()` / `get_object(id)` 只读查询
- 更新逻辑与 id 稳定性
- 置信度衰减（静止物慢衰减、滚动球快衰减）
- 遮挡时**不能删除物体**，只能降置信度 + 记 `last_seen`
- 抓取确认需"夹爪反馈 + 视觉确认"双条件，不能因为发了命令就断言抓住
- 验收：≥4 实体、状态流转正确、衰减生效、单测覆盖四类场景

### 任务包二：视觉事实服务（外壳已冻结，只填内核）
- 已有：`judge_service` + 健康检查 + HTTP + `JudgeRequest/JudgeResponse` + mock 版
- 我填：`RewardClassifierProvider` 占位
- `evidence` 字段目前恒为 `{}`，需要填检测框 / 置信度 / 帧引用 / 前后差分
- 路线：主线 RewardClassifier（LeRobot 自带，与 SmolVLA 同生态），备选 YOLO 重叠判定，VLM 仅做早期原型

### 两个任务包耦合度高于文档描述
判定内核输出的检测框 → 正是 World Model 的输入；World Model 的 `last_seen` 与置信度 → 正是 `evidence` 里该带的东西。**合并规划、共用一套感知抽象层**，而不是当两个项目做。

---

## 三、五个子问题分解

| # | 问题 | 本质 | 我的工作 |
|---|---|---|---|
| 1 | 检测 | 现成模型 | 选型 + 别名表，不自己训 |
| 2 | 定位 | 标定 + 几何 | **地平面求交**，绕开深度失效 |
| 2b | 位姿 | 里程计 / SLAM | 留字段接入，记录不确定度 |
| **3** | **关联** | **指派问题** | 代价矩阵 + 门控 + 贪心 |
| **4** | **时效** | **衰减模型** | 区分"不在视野"vs"视野内未检测到" |
| 5 | 别名 | 映射表 | 一张 JSON |

重点在 **#3 关联**和 **#4 时效**——没有这两块，写出来的东西只是每帧检测结果转发，那叫管道不叫模型。而且这两块都不需要硬件，喂假观测就能开发、就能测。

### #2 为什么用地平面求交
宋红的 schema 只有 `x` / `z` 没有 `y` —— 这是俯视 2D 世界，物体都在地上。因此：

```
检测框底边中点 → 从相机射出的一条光线 → 与地平面求交 → (x, z)
```

全程不需要深度值，透明/反光物体照样能定位。代价是物体必须接地（放桌面的物体需已知桌面高度）。

### #4 为什么必须区分两种"没看见"
- **不在视野里** → 只是转过头了，**不构成**"东西不在了"的证据，confidence 几乎不该掉
- **在视野里但没检测到** → 这才是证据，confidence 应快速衰减

两者混在一起处理，机器人转个身整个世界就忘光了。判断"某坐标是否在当前视野内"只需要位姿 + 相机 FOV，不需要任何模型。

### #3 为什么不是置信区间问题
当前 N 个已知对象、这一帧 M 个检测，要决定谁配谁 —— 这是**指派问题**，标准做法是 MOT（SORT / ByteTrack / DeepSORT）。我的场景物体少、移动慢，**SORT 级别（距离 + 类别）大概率够用，ReID 先不上**。

---

## 四、分级路线

```
L0  快照转发   每帧检测直接输出，物体出视野即消失
L1  带记忆     维护对象表，新观测融合，旧观测按时效衰减    ← 本期目标
L2  语义关系   "球在桌子上"、"桶是空的"                   ← 以后
```

---

## 五、数据源现状与绕过方案

**硬阻塞：我既没有相机也没有场景。**

| 候选源 | 状态 | 能验证什么 |
|---|---|---|
| 蒋玉月 SG2002 | 唯一跑通的真实视觉链路，202 张待标注帧 + 原始 YUYV，网球检出 0.95–0.97 | 真实检测 |
| 曹志伟 SO101 桌面 | overhead + wrist 双摄，`so101_ball_pick_v1` 采集中，带 episode 标注 | 判定内核训练/评测 |
| 宋红 SimCar | 有真值状态，**明确无图像**（绕障来源是 `known_scene_geometry`） | 状态流转与衰减，验证不了视觉 |
| 蒋丰泽 Isaac Gym | 原型机未组装，暂不产数据 | — |

**绕过办法：感知源做成 provider。**

```
MockProvider（读 JSON 场景序列）   ← 今天就能开工，零硬件依赖
      ↓ 换
曹志伟桌面臂 overhead/wrist 相机
      ↓ 换
真机 RGBD
```

与叶志阳那条原则一致：**契约不改，在外围加 provider**。

---

## 六、待确认清单

### 问宋红
1. `radius_cm` 是外接圆半径还是包围盒半宽？决定绕障余量。
2. 坐标系原点是世界固定点还是机器人本体？**如果是本体系，位姿漂移不影响我，#2b 整块可以砍掉。**
3. 多个观测冲突时以谁为准（最新 / 最高置信度 / 加权融合）？
4. World Model 是进程内库还是独立 HTTP 服务？（判定服务写死了 HTTP，World Model 文档没说）

### 问判定服务外壳 owner（2026-08-11 复核新增）
A. `JudgeRequest` 能否加 `target_id` / `container_id`？没有稳定 id，判定主语只能靠名字猜，
   场景里出现第二个同类物体（干扰球、断轨残留）就是歧义。当前走 `JudgeContext` 旁路。
B. `JudgeRequest` 能否加 `gripper_pose`（末端 x/z）？没有它，"视觉确认"退化成"球还在世界上"，
   抓取双条件不成立。
C. `now` 的时钟源是墙钟 / ROS 时间 / 回放时间？三者混用时陈旧度护栏会被绕开。
D. `scene_observations` 能否加 `id` 与 `state`？下游现在无法区分实时观测与残留信念。

### 问带教
5. 输出契约以哪套为准：任务规划文档的 `pose{frame: base_link, x,y,z} + class`（机械臂桌面三维），还是 SimCar 的 `name/aliases/x/z/radius_cm/source`（小车地面二维）？还是内部一套 + 两个适配器？
6. 观测数据落点锁定哪个？—— 这与宋红第 1 周就挂着的遗留问题 #2「判定图像从哪来？」是同一个问题。

**骨架当前的选择：内部状态结构独立，对外通过 adapter 导出 `scene_observations` 格式。上面 1/2/5 无论怎么定，改的都只是 adapter，核心逻辑不动。**

---

## 七、对外契约（当前对齐目标）

```json
{
  "name": "coke_bottle",
  "aliases": ["可乐瓶"],
  "x": 0.0,
  "z": -5.4,
  "radius_cm": 3.3,
  "source": "rgbd_detector",
  "timestamp": "2026-07-31T10:00:00Z",
  "confidence": 0.93
}
```

内部状态额外维护（不对外暴露，但 `evidence` 会用）：
`obj_id / first_seen / last_seen / hit_count / miss_count / state / pose_uncertainty_cm / last_bbox / last_frame_id`

---

## 八、Judge `evidence` 填什么

`evidence` 当前恒为 `{}`。判定"球真的进桶了吗"不能靠技能自报成功，要靠"我看见球现在在桶里"。因此 evidence 装的是**动作前后两份世界快照的差分**：

```json
{
  "verdict_basis": "world_model_diff",
  "target": {"name": "ball", "before": {...}, "after": {...}},
  "container": {"name": "basket", "after": {...}},
  "relation": {"type": "inside", "distance_cm": 4.2, "threshold_cm": 8.0},
  "observations": [{"frame_id": "...", "bbox": [...], "confidence": 0.94}],
  "staleness_s": 0.3,
  "warnings": []
}
```

---

## 九、复用与作废

**复用（编排层阶段留下的）**：executor 状态机骨架的 `world_state` 空位、judge 分层设计、技能契约里的 `evidence` 字段。

**作废**：planner、prompt、20 条规划基准、计划级修复轮。

---

## 十、骨架跑通结果与已知局限

`python run_demo.py` + `pytest tests/ -q`（49 passed）+ `python tools/false_verdict_probe.py`（16 PASS / 0 FAIL）已跑通。

**验证到的关键不变式**
- 遮挡（t=1.5s，视野内漏检）：`ball_001` 置信度 0.955 → 0.758，对象未被删除，`last_seen` 停在真实观测时刻
- 转身出视野（t=2.5→6.0s，3.5 秒）：置信度仅 0.749 → 0.720，`miss_count` 不增加 —— **转个身没把世界忘光**
- 重新看到（t=7.0s）：置信度恢复到 0.867，`obj_id` 保持不变
- 放球（t=9.0s，位移 0.89m）：`obj_id` 仍为 `ball_001`，位置落到 (-0.56, 1.60)
- Judge `evidence` 不再是 `{}`，含前后快照、距离/阈值、帧引用、bbox、陈旧度、原因码

### 10.1 第一版骨架的判定缺陷与修复（2026-08-11）

第一版跑通了，但专项复核发现**判定内核会给出虚假判定**：7 条假阳性、2 条假阴性。
根因不在信念怎么维护，在**信念怎么被判定层消费**。已全部修复，逐条固化成回归单测。

| 缺陷 | 修复 |
|---|---|
| 判定对象按 `name` 取 `max(confidence)` —— 干扰物 / 断轨残留 / 单帧误检都可能被抽中 | 用 `before` 锁定 `obj_id`，在 `after` 里按同一 id 找；断轨报 `identity_broken`，同名多实例报 `ambiguous_target`，都判失败而不是猜 |
| `satisfied` 只看 `after` —— 空动作、球本来就在桶里照样报成功 | 要求 `before` 不满足且 `after` 满足，否则 `no_state_change` |
| 护栏是事后 warning，两条判定路径口径不一致，状态机从没被查过 | 统一前置条件 `check_quality()`（置信度 / CONFIRMED / 陈旧度 / 时钟），输出机器可读 reason code |
| `age()` 把负数夹成 0，`now` 漏传时陈旧度护栏静默失效 | `age()` 如实返回负值，判定层显式检出 `clock_skew` |
| 抓取"双条件"退化成一个自报条件 —— 球在三米外也算视觉确认 | 视觉确认要求目标在夹爪可及范围内；同时给自遮挡 1.5s 宽限，不再误杀真抓取 |
| `success` 与 `relation.satisfied` 互相矛盾，`detail` 只报距离 | `relation.satisfied` 明确只表示几何成立；判定结论唯一来源是 `verdict.success`；`detail` 必须解释原因 |
| 固定门控 0.5m 在长帧间隔下断轨（原文标 P1） | 门控随 dt 缩放 `gate = min(base + v_max·dt, 上限)`；**优先级提到 P0 —— 断轨会一路传导成判定翻转** |
| 固定 EMA 平滑在大位移后把位置留在半路 | 平滑改成时间常数固定的指数滤波，名义帧间隔下与原行为一致 |
| 契约 `timestamp` 把相对秒当 Unix 纪元，导出 1970 年 | 引入 `time_origin`，场景 JSON 声明原点，`wm.to_contract()` 走这条路 |
| 未实现 provider 从 `judge()` 抛异常（对冻结的 HTTP 外壳是 500） | 返回 `JudgeResponse(success=False, ...)`；`get_provider` 的错配 kwargs 报清楚的 TypeError |

**仍未解决（需契约层决策，不是代码能修的）**

1. **契约只有 x/z 没有 y** —— "球被举在桶正上方"与"球掉进桶里"俯视投影完全相同。
   当前对策：夹爪仍闭合时判 `still_grasped`；每条 containment 判定都带 `no_height_evidence` 提示。
   真正解决要么加 y，要么加接触/支撑判定。
2. **`scene_observations` 没有 `id` / `state` 字段** —— 真实场景同名多实例（多个球）下游无法区分。
   判定层已能用 id 自保，但对外契约还不行。加字段要契约 owner 签字。
3. **`JudgeRequest` 缺 `target_id` / `gripper_pose`** —— 外壳已冻结，当前走 `JudgeContext` 旁路传。
   不传时判定会保守失败并给出原因码，不会假装确认。
4. **`confidence` 是启发式分数不是概率** —— 融合公式带下限保护，所有基于它的阈值都是经验值，
   真机标定后要重新定。evidence 里以 `heuristic_confidence` 提示。
