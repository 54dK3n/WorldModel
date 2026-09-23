# 已验证动作证据：撤销原位置

`WorldModel.mark_removed(obj_id, now) -> bool` 用于调用方已通过公开
`holding()` 确认抓持**所选目标**之后。`grab()` 的返回值、计划动作、任务完成计数
或类别名称本身都不足以调用此接口；调用方必须锁定此前选中的精确轨迹 ID。

这不是感知衰减：接口不伪造观测，不调用 `update([])`，不改融合、测距、关联、
确认次数、漏检计数或视野外对象的状态。它以已验证动作结果直接撤销该轨迹的
**原位置占用信念**，不是声称物体已经送达。

- 只接受精确 `obj_id`，不解析类别/别名。未知 ID 返回 `False`，零副作用。
- 时间必须是与观测相同时间轴的有限数；非有限值抛 `ValueError`，零副作用。
  调用方应提供当前动作验证时刻，不得回填早于已有证据的时刻。
- 已知轨迹返回 `True`：`confidence=0`、`state=LOST`，通过现有归档机制退出
  `get_scene()` / `snapshot()` / `to_contract()`；显式 `get_archived(exact_id)` 或
  `get_object(exact_id, include_lost=True)` 仍可查。通用模型的默认查询不返回
  LOST；`guangyang_static_world_model()` 显式启用旧广阳岛的归档查询后备。
- 保留坐标、尺寸、`first_seen`、`last_seen`、命中次数、帧信息和命中位姿历史。
  第一次撤销只更新该对象 `last_updated` 和全局最新写入时刻；其它对象不变。
- 重复撤销已为 `LOST/confidence=0` 的 ID 仍返回 `True`，包括时间在内均不再改写。
  已因感知衰减归档但尚未归零的轨迹也可接受一次动作证据。
- 后续新观测按原关联规则处理；归档 ID 不复活，原位置的新检测可生成新轨迹。
  调用方仍需自己的已抓取/已送达集合防止业务重复选择。

最小集成次序：抓取 → 公开 `holding()` 验证目标 →
`mark_removed(selected_track_id, public_time)` → 记录前后状态 → 运输/释放验证。
不得为了触发归档额外调用视觉，也不得把这个接口解释为相机在 3 秒内自然衰减的证据。

`tests/test_action_evidence.py` 覆盖精确身份、历史保留、幂等、坏时间、类别拒绝、
其它对象及其后续融合不变、新观测不复活旧 ID、公开持球结果的调用边界。
