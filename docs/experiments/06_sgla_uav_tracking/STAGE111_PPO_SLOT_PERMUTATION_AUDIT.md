# Stage-111：同一候选集合的 PPO 槽位排列诊断

状态：已实现并通过本机合成缓存测试，正式服务器结果待运行。固定设计日期：2026-11-15。

Stage-110 的校准生成集合最佳 IoU 为 0.670099，高于固定集合的 0.650362；旧 PPO 实际影子选框却从 0.632668 降到 0.479921。有 331 个状态 q0 的 IoU≥0.5，PPO 却选了 IoU<0.2 的框。先检验旧选择器与新输入的兼容性，再决定怎样训练新选择器。

本轮问题是：**候选框和历史完全相同，只交换三个非 q0 输入的位置，旧 PPO 是否会选择不同的实际框？不使用 GT 的统一排列或概率汇总能否减小损失？** 这可以量化排列依赖，不能预先断言它是全部退化的原因。

## 固定来源与预算

- 只读完整 Stage-110，decision 指纹 `dd804631d678a6e112c45af5df33d3ebbbf88bd3113dfdd7af0be7f7fb31b254`。沿用其 28 个已消耗开发源组、40 个 sequence/seed 单元、80 个历史上下文、每上下文原定 24 个评价帧，共 1,920 个状态。
- 六个原候选集合：fixed、motion、learned_constant、learned_uncalibrated、learned_calibrated、calibrated_random。每个仍为四个已解码候选。保持父框、输出框、图像观测及原历史；不重生成、不改变预算，不按 GT 重新选帧。
- q0 固定于槽位0，枚举其余三项的全部六种排列：0123、0132、0213、0231、0312、0321。四个对象的父框、输出和24维基础特征同步重排；根据重排后的框重新计算关系特征。
- 使用同一冻结 PPO、原256维逐槽位归一化、原截断和重复框掩码。每次以概率 argmax 选槽位，再映射回原物理候选。共 `1,920×6×6=69,120` 次 NumPy actor 前向；原 gate 同时计算 critic。
- 新训练、新 SGLA 搜索、图像读取、新完整轨迹、确认集推理均为0。只读取 JSON/NPZ 和既有标签，无需 GPU 或 checkpoint；保留旧 GPU 信息仅用于来源记录。检查来源可能需要读取较多旧缓存字节。
- 原排列必须逐状态、逐集合复现 Stage-110 的 `shadow_selected`；q0、候选最佳 IoU 和原选择 IoU 误差≤1e-12。来源、代码、输出缓存和收据都有哈希绑定；禁止覆盖冻结来源。

## 比较对象

| 策略 | 固定定义 | 是否可使用 GT |
|---|---|---|
| identity | 原排列和原 PPO 推理 | 否 |
| canonical_geometry | 在六种排列中，将三个非 q0 父框对应到该状态原 fixed 的 q1/q2/q3；最小化中心距离平方及 log 宽高比平方之和 | 否 |
| probability_symmetrized | 六种排列的 PPO 概率映射回原物理候选后等权平均，再选择最大者 | 否 |
| uniform_order_expected | 对六种排列的硬选择作等概率随机排列的解析期望 | 不用于决策；只在评价时算期望 IoU/事件率 |
| q0 | 始终保留原 q0 输出 | 否 |
| hann | 选择基础特征中的最大 Hann 峰值 | 否 |
| best_order_offline / worst_order_offline | 六种排列产生的实际选择中，按当前 GT 取最好/最差 | 是，仅离线界限 |
| candidate_oracle | 原四框中的最佳当前 IoU | 是，仅供给上界 |

几何对应的中心距离除以上一预测框面积平方根，尺寸距离使用 log(w/w_ref)、log(h/h_ref)，两项无可调权重。几何成本在1e-12内并列时按对应物理父框及输出框字典序确定。概率汇总按**精确相同输出框**合并质量，避免同一框的概率分散在重复索引；分数在1e-12内并列优先q0，否则按物理输出框字典序。原 identity 保留旧 argmax 的低槽位并列规则，不修改旧策略。

概率汇总是一个无需新拟合的诊断控制，增加六次选择器计算；其是否适合部署仍需后续评价。不能把 GT 事后挑出的最好排列当作在线算法，也不在结果中挑一个最有利固定排列作为采用方案。

## 评价与解释

主要比较预先固定为 learned_calibrated 集合上的 `probability_symmetrized_minus_identity` 当前帧 IoU。同时报告几何对应、随机排列期望、最好排列上界，以及与 q0/Hann 的比较；其余五个集合完整保留，特别检查 fixed 本身的排列敏感性。

全部指标先在 source-group/seed/source-arm 上下文内平均，再平均同组模式、种子及源组；同时分 historical/expanded、four_matched/four_pairwise，不将长视频帧直接混池。报告源组胜/平/负、中位数和最小留一均差。本轮没有采用门槛或独立显著性结论。

- 当前 IoU、好框选中率、`best≥0.5 且 selected<0.5` 的漏选率、`q0≥0.5 且 selected<0.2` 的有害替换率、候选最佳与选择差距。
- 同一集合在六种排列下是否选择不同**物理输出框**，及 IoU 最大最小差。重复索引但相同输出不算变化。
- 各排列进入旧归一化后、截断前的坐标越界比例和最大绝对z值；它们反映接口支持压力，不能单独证明整体分布外或唯一失效原因。
- 所有无GT选择与概率先写入带收据的预测NPZ，再附评价标签和离线上界。恢复时重算CPU预测并核对；已完成任务验证后直接返回。

解释顺序：若同一框集仅改变排列就大幅变动，则旧接口存在可量化排列依赖；若不使用GT的两项控制改善，还需看是否超过 q0/Hann、是否在扩展源组保留。若只有最好排列上界改善，说明存在潜力而非已经得到可用规则。若所有排列仍差，则需要检查候选几何、特征支持及匹配训练，不能据此认为算法生成无价值。任一结果都不能替代完整自身轨迹、稳定性/恢复保护及封存源确认。

## 服务器运行

同步包含本轮文件的仓库后，在原 conda 环境、仓库根目录执行：

```bash
cd ~/workspace/paper
bash tools/run_stage111_slot_audit.sh
```

希望关闭 SSH 后继续运行时：

```bash
cd ~/workspace/paper
mkdir -p output/stage111
nohup bash tools/run_stage111_slot_audit.sh > output/stage111/run.log 2>&1 < /dev/null &
echo $!
```

默认输出 `output/stage111/cached_ppo_slot_audit_v1`；可用 `STAGE111_ROOT` 指定新目录。来源变量为 `STAGE76_ROOT`、`STAGE89_ROOT`、`STAGE93_ROOT`、`STAGE103_ROOT`、`STAGE110_ROOT`。不需要重新设置 CHECKPOINT 或选择 L20/L40。需要完整服务器产物，单独上传的结果txt不能代替源缓存。

完成后自动核验并生成 `stage111_results_<UTC>_<PID>.txt`。如只需重新打包：

```bash
bash tools/run_stage111_slot_audit.sh --pack-only
```

包中包含decision及六张表：scope 1,920行、state_summary 11,520行、policy_summary 486行、paired 972行、permutation_summary 324行、sensitivity_summary 54行。原始预测保存在 predictions/ 下的1,920个NPZ及收据中，保留用于复核。

## 本机验证

`python -B tools/test_stage111_slot_audit.py -v` 的5项测试通过，无跳过。覆盖原 gate 的排列动作/概率精确重放、只偏好槽位的反例、全部六种重编号下概率汇总的物理不变性、重复框和并列、几何对应恢复、GT与未来历史隔离、旧归一化截断统计、完整合成缓存流程、保存预测先于附标签、来源只读、续跑一致及缓存/来源篡改拒绝。测试明确禁止调用拟合、GPU初始化和图像读取。Python CLI、Bash语法和LF检查通过。本机无正式缓存，未得到正式Stage-111成绩。
