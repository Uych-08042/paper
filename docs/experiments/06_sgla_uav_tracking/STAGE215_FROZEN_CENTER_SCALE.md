# Stage-215：固定权重的中心×尺度完整闭环消融

依据214：forecast/coverage AUC53.9480/45.3236，coverage覆盖95.58%但已覆盖无好网格38.62%，父框几何平均侧长626.08。不能由这些关联直接归因尺度；本阶段用分量干预区分中心、尺度及交互，不重训、不调阈值、不改变物理边界。

## 来源及范围

Stage-214原包SHA256 `e9fde8f36b8da411d31dd91fe7bafb95853f9089953567b57ff728bdd59907b6`，decision `ed45014f53c42067f3ff9381263c36a9cd9736c02051bc4719ce05bd5dab9340`。要求完整214物理缓存/八个更新器与receipt、208四个适配器、原生checkpoint、七个旧源目录。全部代码/模型/协议/元数据核验，原GPU/软件身份严格保持；不自动换卡。38个已消费源组、原两折、2132/2133种子固定，确认集不开放。

## 四个因素格

原214输出 $d=(d_x,d_y,d_s)$，raw框为残差基点；仍使用原 `parent` 映射和图像物理边界。

| 模式 | 应用的修正 | 含义 |
| --- | --- | --- |
| none | (0,0,0) | 保留raw搜索父框 |
| center | (dx,dy,0) | 只修正中心 |
| scale | (0,0,ds) | 只修正尺度，中心仍为raw中心 |
| joint | (dx,dy,ds) | 原214完整更新 |

两种来源目标forecast/coverage分别做四格，权重、标准化缓冲始终不变。每格每帧都使用原网络推理一次，随后将禁用维精确置零；none也不省掉网络调用，joint直接返回原输出以保留原位模式。不是重新拟合中心/尺度子网络，也不是评估删掉某一训练损失。每格在自身前继状态上重新计算当前输入并闭环，因此后续输出可不同；因素效应包含这种反馈，不是同一时刻固定状态上的直接效应。

SGLA、空间适配、选择器/CENTER、固定搜索倍数不变，对外总取适配框；每帧一次crop/selector、共享六前缀块、同选中块两次读出、两次CENTER、一次更新器。没有额外候选层遍历、PPO、未来图像或GT决策。GT只在动作确定后用于诊断。

## 评价、端点及预算

8格×2seed×38源=608条**新完整轨迹**。重跑none和joint用于端点验收：none须精确复现214的nn，joint须精确复现对应forecast/coverage，排除pipeline计时；none新增的一次updater调用单独记账。其余非计时框/状态/诊断数组全部相同，不能用宏AUC近似代替。

另绑定复用214的native38条、mm76条，共114条参照，显式标记复用而不是新推理或预算匹配组。合计722条报告质量序列。所有学习格的原生/mm恢复和好帧保护均从这些参照计算。端点新轨迹全量复现验证同一运行身份下的可比性；旧参照计时不与新轨迹作速度结论。

每折首个固定源用于smoke：32条串行、32条独立双spawn短轨迹，排除计时后逐数组一致才进行全评价。完整评价默认两个worker按视频分区。一个总体进度覆盖两种smoke及608新轨迹，旧参照不计新工作；实际38源40517跟踪帧，608新轨迹共648272帧，smoke为512帧，总工作计数648785含最终封存一步。0训练更新。

端点复现304行、新旧绑定参照114行，共418行source_replay。8行model_scope、32行smoke、10行policy、132行paired、36行factor、722行seed_video/calls、684行search_geometry、3行execution，合计10张表。NPZ与receipt、模型及来源哈希封存，已存在产物不符则拒绝覆盖。

## 统计与判断

唯一主要比较：`coverage__center − coverage__joint` AUC，检验关闭coverage模型的尺度修正是否减轻退化。这个优先级来自214的大尺度/低分辨率线索，不是215事后选择。forecast同对照、只尺度对照、各格对none/native/mm均为支持分析。

每种来源目标另算中心平均效应、尺度平均效应和中心×尺度交互。6指标包含AUC/低IoU及原生/mm匹配恢复/好帧保护。先源内平均两seed，再按38源bootstrap（既有20000次/固定随机种子），名义区间用于历史开发诊断，无自动采用门槛。

覆盖、目标名义网络侧长、覆盖无好网格、有好网格漏选、父框面积平方根均按完整轨迹记录。父框侧长不是实际裁剪边长（还需乘固定倍数再取整）。不挑0176/0204作唯一证据。

若中心单独恢复而尺度单独/联合受损，支持重新建模尺度目标；若中心单独也明显退化，需核一步预测误差和自身状态分布偏移；若相同分量随另一分量而改变作用，报告闭环交互。不能只从某格最好就宣布部署成功，也不在本阶段增加尺度阈值或扫权重。

## 运行

```bash
bash tools/run_stage215_components.sh
```

默认 `output/stage215/frozen_center_scale_v1`，`STAGE215_GPU=0`，`STAGE215_WORKERS=2`。完整执行包括重跑已有端点，可能较214更耗时，coverage大裁剪仍会保留，以免改变被检验机制。后台：

```bash
mkdir -p output/stage215
nohup bash tools/run_stage215_components.sh > output/stage215/run.log 2>&1 < /dev/null &
echo $! > output/stage215/run.pid
tail -f output/stage215/run.log
```

自动生成总体进度与合并10表、manifest/decision的时间戳TXT；不执行Git提交或推送。旧路径沿用STAGE208_ROOT/STAGE214_ROOT等变量，CHECKPOINT为原始权重。完成后仅校验：

```bash
python -B tools/stage215_component_experiment.py --check-results \
  --output-dir output/stage215/frozen_center_scale_v1
```

本地合成CPU检查不构成真实GPU/视频结果，真实执行须在有完整服务器缓存的环境进行。
