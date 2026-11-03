# Stage-69：冻结预测器的同状态门控与真实成对目标诊断

状态：2026-11-04新增方案／实现，服务器尚未运行；不是方法采用实验。最新已完成成绩仍为[Stage-68](STAGE68_RESULTS_REVIEW.md)。

Stage-68的future改善了退化的immediate控制，但对Q仅2胜10负，0304仍出现新增可恢复事件，0307仍未恢复。本轮先回答：**同一状态下future是否选错、fit到自身历史的特征支持怎样变化，以及它在已有真实后继分支上是否预测对了增益符号？** 继续以原始SGLA V1为基线，不修改网络、候选结构、模板、Q、预测器、阈值或接管率。

## 固定范围与三种证据

入口为`tools/stage69_gate_audit.py`，默认新根`output/stage69/cached_paired_gate_audit_v1`，来源为完整`output/stage68/fit_paired_takeover_advantage_v1`及其封存的Stage-67／66上游链。只拿到合并TXT不能运行本诊断；服务器已有缓存必须保留。

1. **384个fit共同起点：** 读取共享候选、allow／veto真实32帧Q后继，重算已封存标签和等视频权重，重构并逐项比较两个128维岭预测器。输出当帧／后继各自的加权MAE、RMSE、正号准确率及预测为正组的实测均值。这些是拟合样本内检查，不是泛化成绩；不保存新模型或选择checkpoint。
2. **7个Stage-67共同起点：** 在相同protect历史、相同四候选上，用该视频held-out折的Stage-68模型重新评分。读取已有Q／protect各两条真实分支，核对当帧及32帧实测增益；future的预测误差只针对其训练定义的Q后继目标。六个二元规则各选q0或原Q提议，其窗口值来自对应的真实Q后继，不推算没有缓存的其他候选未来。Hann可能选第三个动作，不放入此二元未来值表。
3. **84条自身历史缓存：** 在每个提供状态的政策上，重新运行Hann／q0／Q／protect／rate_margin／immediate／future七条影子规则，比较同一候选集中的当帧IoU、漏选、索引及实际框变化。七条影子动作都不写回历史，不生成影子轨迹，不报告影子AUC；future分数不能用当前IoU当作“后继预测误差”。

不新增图像读取、GPU推理、分支、完整轨迹、SGLA训练或预测器拟合。CPU解线性方程仅为复核旧系数。COMPLETE代表诊断完成，无采用门槛或独立验证主张。

## 预声明窗口

每个窗口固定为锚点前后各30帧，边界截断，七种状态提供者都检查。事件依旧使用连续3帧IoU<0.2确认失效、连续3帧IoU≥0.5确认恢复，报告事件起始帧。编号沿用跟踪缓存相对帧，首个跟踪帧为1。

| ID | 视频简称 | 来源政策与规则 | Stage-68预期帧 | 要区分的问题 |
|---|---|---|---:|---|
| C01 | 0304 | immediate首次失效 | 554 | 退化当帧控制早期错误 |
| C02 | 0304 | future首次失效 | 1092 | 新增可恢复事件 |
| C03 | 0307 | future首次失效 | 394 | 恢复能力丢失前的选择 |
| C04 | 0307 | q0首次恢复 | 474 | 原生恢复窗口内候选可用性 |
| C05 | 0176 | future首次恢复 | 320 | 已有恢复进展附近的门控 |
| C06 | 0147 | future首次恢复 | 1116 | 暂时恢复与终末失效区分 |
| C07 | 0237 | future首次失效 | 240 | 相对Q损失最大视频 |
| C08 | 0204 | q0首次失效 | 664 | 已有大幅救援是否保留 |
| C09 | 0226 | q0首次恢复 | 338 | 已有恢复是否保留 |

帧由已封存事件解析；表中数字是已收到结果的预期，不作为新阈值。若指定来源无事件，记录`no_declared_source_event`，不换成效果更好的点。Stage-67已有7起点全部保留，包括0176:310与0307:370等，不能只展示正例。

## 指标与解释边界

- 同状态良好候选为IoU≥0.5，低候选为IoU<0.2；`missed_available`表示四候选存在良好输出而影子规则未选到良好输出。`all_low`与“无良好候选”不同；中间IoU候选也可能有价值。
- `q0_good_to_low`／`q0_low_to_good`描述当帧极端替换，不能据此判定整个后继安全。`changed_box`按实际浮点框比较，`nonq0`按索引比较；公共IoU仍使用原SGLA整数截断协议。
- `current_mean_iou_delta`是同一提供者状态上的future减控制当帧均值，先在视频内平均再宏平均。不是完整政策AUC，也不是共同后继收益。
- Stage-67二元值为“本次强制allow／veto后继续Q”的实测32帧均值；`oracle_binary_regret`为这两个已有分支中较好值减所选值。它不是门控后继的部署回报，也不是因果最优策略。
- 预测为正使用原1e-6判据；“非正”包含零，不把近零强行分成负例。fit精度乐观；7个转移点来自held-out视频组，但都是已消费、用于讨论设计的开发案例，且不是随机独立抽样。
- 每折128维特征的原始1%／99%边界使用该折真实fit对及等视频权重的逆经验分布；标准化中心／尺度沿用旧预测器。尺度>1e-6记为活跃维度，同时报告全部维度和活跃维度越界比例、最大绝对z及活跃维度RMS。常量维度的z可能很大，不截断、不变成新OOD门槛。
- 只在Q非q0且实际框改变的查询上描述特征支持。fit Hann到held-out Hann也改变了视频组、折模型与提议，不能把差异全称为“自身历史漂移”；同视频不同提供者使用相同held-out模型，但仍是描述性比较。
- 整条future与Q／immediate／rate的首次索引、框、所选裁剪、候选及特征分岔分别记录。晚窗口的差异不能直接归因于当帧模型；索引不同且框相同也不能算有效干预。

## 产物与完整性

`manifest.json`封存本轮代码、窗口、设计、Stage-68来源文件完整SHA集合和各级指纹。读取时保留原阶段全部收据、运行身份、原控制及折隔离校验；不修改旧校验函数、manifest或NPZ。相同输入可重复汇总，已有产物不同则拒绝覆盖。

脚本以`cat`合并以下13份材料为一个TXT：diagnostic；fit_prediction／fit_sign_groups；paired_case／paired_policy；same_state_summary／matched_summary／candidate_summary／feature_shift；anchors／focus／divergences／scope。默认分别为4／8／7／42／49／21／7／14／9／63／36／2行CSV。原始fit_points、逐视频同状态／匹配／特征表、scope_video、逐帧focus_frames及fit_reference JSON留服务器；必要时再定位，不默认全部回传。

报告只展示来源绑定集合指纹，完整SHA集合保留manifest，避免再次把数千缓存路径塞入回传文件。新调用预算全部为0；scope中的`cached_source_*`是旧来源的逻辑调用核对，不是本轮新增调用。没有置信区间、逐帧显著性或新阈值筛选。

## 运行

服务器同步新文件后，在原`sglatrack-test`环境和仓库根执行：

```bash
cd ~/workspace/paper
bash tools/run_stage69_gate.sh
```

脚本不需要checkpoint或GPU参数。上游数据检查和逐帧特征复核仍可能花时间；断线运行可用：

```bash
cd ~/workspace/paper
mkdir -p output/stage69
nohup bash tools/run_stage69_gate.sh > output/stage69/run_stage69.log 2>&1 < /dev/null &
```

日志放在实验根父目录，勿预先往新的实验根写文件。若已完成，相同命令仅复核并重新打包，不覆盖旧结果，不提交或上传Git。返回脚本提示的`docs/experiments/06_sgla_uav_tracking/results/stage69_results_<UTC>_<PID>.txt`即可。

## 验证

新增`tools/test_stage69_gate.py`覆盖真实成对值与未观察动作拒绝、GT-free门控、特征支持、好候选可用但漏选、无事件不替换、混合CSV列与拒绝覆盖，以及真实CPU CENTER的完整上游来源链。完整链在旧阶段使用真实Torch／OpenCV生成微型源缓存，然后阻止新图像读取／推理并令两库不可用，复核Stage-69输出与续跑，核对所有旧字节和网络状态未变，检验CSV／模型／缓存篡改被拒绝。该微型来源为2视频×4帧，2 fit对、1个Stage-67成对点；继承Stage-68仅测试的折内Q曝光系数，正式模型与原控制不替换。CPU测试不代表服务器GPU效果。

验证记录及服务器待运行状态以[交接第0.30节](../../../WORK_HANDOFF.md#030-2026-11-04stage-69已准备缓存同状态与真实成对目标诊断)为准。后续应先读结果，再决定需要哪种新干预；本轮不预判唯一病因。

本机4项测试全部通过，120.691秒、无跳过；新进程禁止导入Torch／OpenCV仍可加载入口。Python AST、Bash语法、187处所涉本地链接、维护完整性和diff检查均通过；旧Stage-68原件SHA不变。
