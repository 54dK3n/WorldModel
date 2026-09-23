# 显式广阳岛静态配置

通用 `WorldModel()`、`FovConfig()` 和 `DecayConfig()` 保持迁移基线
`0ea6539` 的默认行为。广阳岛参数不再通过全局默认值影响其它数据源。
本次隔离只调整配置与查询入口；不改融合、关联、确认、衰减算法、Judge，
也不改 `mark_removed` 的动作证据算法。相机输入和严格安全迁移已经属于该基线，
本改动不撤销这些前置迁移。

```python
from world_model import WorldModel
from world_model.providers import guangyang_static_world_model

wm_generic = WorldModel()
wm_guangyang = guangyang_static_world_model()  # 完整静态场景配置
wm_confirmation = guangyang_static_world_model(max_range_m=0.9)
```

| 配置 | 通用默认 | 显式广阳岛静态工厂 |
| --- | --- | --- |
| 水平视角 | 70° | 75.2° |
| 最远距离 | 4 m | 8 m；确认程序显式传 0.9 m |
| 最近距离 | 0.15 m | 0.15 m |
| 视野内漏检/视野外半衰期 | 1.5 / 60 s | 1.5 / 60 s |
| 类别半衰期系数 | basket=4、table=8、ball=1、bottle=2 | 左列及 target=1、distractor=1.5、obstacle/storage-zone/cleanup-zone=8 |
| 静态等权平均 | 关闭，沿用时间相关指数权重 | 开启 |
| 命中位姿间距去重 | 关闭（0 m） | 0.15 m |
| 关联基础/最大门控 | 沿用 `AssociationConfig()` | 0.30 / 0.30 m，速度扩展为 0 |
| 默认对象查询 | 仅活跃轨迹 | 活跃轨迹及归档后备 |

其它参数沿用原广阳岛实例：`position_smoothing=0.6`、`nominal_dt_s=0.5`、
`time_origin=None`、STALE/LOST 门槛 0.50/0.15、确认命中数 3。
工厂每次生成独立配置，修改某个实例不会改变通用默认或其它实例。
`GuangyangProvider` 只做公开观测/里程计转换，导入或使用它不自动启用静态融合。
原 `guangyang_static_association_config()` 仍可显式选取关联参数，但它不是完整配置。

## 归档查询

```python
wm.get_object(track_id)                      # 通用默认只查活跃轨迹
wm.get_object(track_id, include_lost=True)   # 显式允许归档后备
wm.get_archived(track_id)                    # 只查归档，严格匹配 ID
```

`get_object` 继续支持规范名/别名，优先返回活跃同名对象；显式允许归档时，
若没有活跃同名对象则返回置信度最高的归档同名对象。精确 ID 优先于名称。
`get_archived` 不解析类别或别名，未知 ID 或活跃 ID 返回 `None`。

`get_object(..., include_lost=None)` 使用实例的
`include_lost_in_get_object` 选项（通用默认为 `False`，工厂显式置 `True`）。
工厂实例也可传 `include_lost=False` 只查活跃对象。所有查询均不改变状态；
LOST 仍不进入 `get_scene()`、`snapshot()` 或正式 `to_contract()` 输出。

`mark_removed` 不会自动调用。调用方必须在公开持球接口已验证所选目标后，
显式传入锁定的精确轨迹 ID；见 [动作证据接口](ACTION_EVIDENCE.md)。
归档访问不等于恢复原位置，也不改变后续新观测产生新轨迹的规则。

## 从旧广阳岛集成迁移

原构造依赖全局广阳岛默认值：

```python
wm = WorldModel(
    assoc_cfg=guangyang_static_association_config(),
    decay_cfg=DecayConfig(),
    fov_cfg=FovConfig(max_range_m=0.9),
)
```

改为 `wm = guangyang_static_world_model(max_range_m=0.9)`，即可保留相同有效
视野、衰减、关联、静态融合、去重与 LOST 查询后备。不要只保留原三参数构造，
否则它现在会正确采用通用的 70° 视角和通用类别系数。

验证：`python3 -m pytest tests/ -q`，168 项通过。迁移基线 `0ea6539` 的
130 项原测试另行原样运行在新实现上，全部通过；两条 LOST 查询断言恢复为
原始 `is None`，动作 API 测试改为显式归档访问并保留全部原断言。
