# Stage-62：已有候选输出的可用性与选择错误诊断

日期：2026-11-03，Asia/Hong_Kong。先阅读 WORK_HANDOFF 第0.14节及[Stage-61 L40结果复核](STAGE61_L40_RESULTS_REVIEW.md)。本轮保持原始SGLA V1／固定E头、VisDrone-SOT任务和已消耗开发组，仅分析服务器已有的Stage-61 L40缓存。此协议在取得Stage-62真实诊断结果前固定。

## 问题与新增价值

Stage-61原生历史搜索在0176／0204出现大收益，却使0226从可恢复失效变为末尾未恢复；E历史在0307的两个种子也丢失了恢复能力。完整视频结果不能回答：这些状态下，历史四候选本身是否仍有良好输出，还是评分选择已把良好输出排除？

Stage-62检验这两个环节，并对比同预算recrop及各自current。它是后续候选可靠性、保守选择或恢复机会研究的机制依据，没有新方法采用门槛，不修改Stage-61判定，不训练评分器或搜索新阈值。

## 数据边界与来源验证

输入为完整`output/stage61/v1_history_matched_recrop_l40_v1/`。严格读取原封存计划、运行记录、smoke、9表及decision、2份启动诊断、108条NPZ／receipt和完整上游来源链；核对实际运行身份、模型摘要、调用预算、因果父框、四候选构造、实际分数及选中状态。不会调用旧summary重写旧报告。

分析只使用`candidate_boxes`、`candidate_parents`、`features`、`selected`、`crop`、`image_hw`及已封存的开发集GT。GT只在输出冻结之后标注当前帧IoU、裁剪保留和几何误差。首帧初始化为逻辑帧0，缓存`frame_id=1`开始，关注帧均沿用原报告逻辑编号，不当作图像文件名中的绝对编号。

**候选指每次搜索通过原生Hann响应解码得到的一个最终框。** 现有缓存没有保留完整响应网格的size／offset／score map，因此“四个已解码输出均不良好”不等于网格内没有目标信息，也不等于目标不在裁剪中。GT覆盖率和中心是否落在可用裁剪内另外报告，包含原crop_geometry的右／下`+1`padding规则。

不读图像、不加载CUDA运行器、不重新执行网络、不训练或生成新轨迹。checkpoint及来源模型文件可以作为来源链做哈希／缓存读取，但不创建或运行新模型。当前GPU是否仍为L40不影响本轮离线分析；结果始终绑定已完成的L40来源。

## 固定统计定义

沿用官方`sgla_int_prediction_inclusive_v1` IoU：预测框先截断成整数、GT保持浮点、初始化不计入本轮跟踪帧。良好输出为IoU≥0.5，低IoU为IoU<0.2，均沿用此前门槛，不能根据本轮结果调节。

对每个政策**自身实际进入该帧的历史**，计算：

| 名称 | 定义 |
|---|---|
| 良好候选可用 | 当前缓存候选中至少一个IoU≥0.5 |
| 全部解码候选不足 | 所有已解码候选IoU<0.5 |
| 可用却漏选 | 至少一个良好候选，实际所选IoU<0.5 |
| q0良好但丢失 | q0 IoU≥0.5，实际所选IoU<0.5 |
| q0良好却选成低IoU | q0 IoU≥0.5，实际所选IoU<0.2 |
| 可救援机会／实际救援 | q0 IoU<0.5且有良好候选／实际所选IoU≥0.5 |
| 即时选择遗憾 | 当前缓存最好IoU−实际所选IoU，单位0～1 |
| 当前帧增量 | 实际所选IoU−同一状态下q0 IoU，单位0～1 |

“漏选率”以良好候选可用的状态为分母；“q0损失率”以q0良好的状态为分母；“救援率”以可救援机会为分母。分母为0时保持未定义，不当作0错误或100%保护。另给无条件状态比例、分子与分母计数，以及支持的种子／视频数量。

每个视频、种子、政策分别汇总，E先在视频内对有定义的种子指标等权平均，再对有定义的视频等权平均；原生每视频只计一个运行。条件分母为0的种子不进入该条件指标的均值，显式报告支持数量。状态总数只说明曝光，不当作独立样本数，不给帧级显著性或新方法PASS。

固定六类队列：全部状态、原生current可靠帧（≥0.5）、原生current低IoU帧（<0.2）、自身首次失败前30帧、自身首个失败事件、自身首次恢复起30帧。失败／恢复沿用三连帧规则；无事件或无恢复时对应队列为空。这些队列可重叠、自身事件队列按结果定义，只作离线描述，不能作为在线触发器或独立测试集。

所有“最好候选”都是已发生历史前缀下的当帧描述；**不拼接成oracle跟踪轨迹、不把均值写成新政策AUC、不假造任何候选的未来回报**。不同政策在同一逻辑帧的输入历史可能不同，横向比较也不是共同起点的因果重放。

## 全量审计与预声明关注点

全量覆盖全部12视频、108条实际轨迹；共121,725个已接受前缀下的状态观察和365,175个已付费搜索的解码输出。重复前缀或相同候选不另作独立样本。current只有q0，history／recrop各四候选。全量统计不受以下病例挑选或窗口重叠影响。

| 病例 | 来源 | 锚点政策／帧 | 已观察到的关注理由 |
|---|---|---|---|
| C01 | 原生0226 | history失败115 | 原生恢复丢失 |
| C02 | E0307／2132 | history失败392 | 恢复丢失 |
| C03 | E0307／2133 | history失败395 | 恢复丢失 |
| C04 | 原生0176 | current失败162 | 历史组避免长期失跟 |
| C05 | 原生0204 | current失败664 | 历史组避免失效 |
| C06 | E0304／2133 | current失败554 | 历史组大收益种子 |
| C07 | E0304／2132 | current失败1078 | 原控制晚期损伤 |
| C08 | E0304／2132 | history失败1093 | 历史组仍损伤 |
| C09 | E0307／2132 | current恢复471 | 恢复机会附近 |
| C10 | E0307／2133 | current恢复511 | 恢复机会附近 |
| C11 | 原生0226 | current恢复338 | 恢复机会附近 |

每个病例展示current／history／recrop三政策各自历史，窗口为锚点前后各30帧，含锚点共61帧，边界裁切。生产输入下为33条关注摘要、2,013条关注时序行、6,039条逐候选行；重叠窗口中的行不是额外推理或独立样本。病例锚点必须从对应缓存的第一个失败／恢复事件复算吻合，否则拒绝继续。

逐候选表保留父框／输出框、GT横纵中心误差、宽高比例、可用裁剪覆盖与中心保留、选层、原24项GT-free特征。Hann峰值来自原缓存，不重新校准。可据此检验“跨裁剪峰值是否可靠”这一假设，但本轮不预先认定它是唯一原因。

## 输出与服务器运行

新目录`output/stage62/cached_candidate_selection_v1/`，新schema为`stage62_cached_decoded_candidate_selection_v1`。旧文件及Stage-1～61／L40工具保持封存。新manifest绑定源文件及新工具哈希；已有输出只接受相同内容，源／报告篡改或来源混合均拒绝。

同步本次新工具后，在原环境运行；**不需要CHECKPOINT、GPU参数或CUDA设备分配**：

```bash
cd ~/workspace/paper
export STAGE61_L40_ROOT="$PWD/output/stage61/v1_history_matched_recrop_l40_v1"
export STAGE62_ROOT="$PWD/output/stage62/cached_candidate_selection_v1"
bash -n tools/run_stage62_candidate_audit.sh
bash tools/run_stage62_candidate_audit.sh
```

单独Python命令为：

```bash
python -B tools/stage62_candidate_audit.py --input "$STAGE61_L40_ROOT" --output_dir "$STAGE62_ROOT"
```

完成后状态为`COMPLETE_CACHED_CANDIDATE_SELECTION_DIAGNOSTIC`，含义是已完成缓存诊断，不是新方法通过。8张CSV包括policy_summary、video_summary、seed_summary、scope、transitions、focus、timeline、candidates。

一键脚本使用`cat`合并diagnostic JSON、policy_summary、scope、transitions、focus五份材料到仓库`results/stage62_results_<UTC>_<pid>.txt`，不自动commit／push。逐种子、逐视频、时序和逐候选长表保留在服务器，可按需进一步读取：

```bash
cat "$STAGE62_ROOT/analysis/stage62_diagnostic.json"
cat "$STAGE62_ROOT/analysis/stage62_policy_summary.csv"
cat "$STAGE62_ROOT/analysis/stage62_focus.csv"
```

当前仅准备本轮代码与本机验证；服务器真实缓存诊断尚未执行，不能预报0226或0307属于候选不足还是选择错误。若可用候选在分岔前存在，再设计候选可靠性／保守选择；若候选已经不足，需检验历史构造、尺寸条件或共同起点后继。下一阶段须依据实际诊断另行预声明。

## 本机验证记录

4项测试全部通过，16.587秒、无跳过，覆盖官方整数IoU、可用性／漏选／q0损伤／救援、GT标签不修改冻结动作、padding几何、空分母与视频等权聚合、真实小通道CENTER／公共decode来源链、重复审计旧文件逐字节不变及篡改拒绝。审计阶段禁用Torch／OpenCV导入、图像读取和网络前向仍通过；上游Stage-58边界、骨干、图像和设备为fixture，不代表服务器真实诊断结果。CLI help、Python AST及LF换行检查已通过。

本机Bash语法检查未完成：沙箱内MSYS权限错误，沙箱外只读`bash -n`又因自动审批服务用量限制而未执行，非脚本被判不安全。没有绕过审批或将其计作通过，请在服务器执行上述语法检查。
