# Stage-209：固定权重的适配器交互输入消融

固定协议，2026-11-26。未取得真实GPU结果。

208正确模板对搜索／错配两个主要完整AUC比较区间跨零，虽然自身状态训练有受控改善。分别训练的六条件同时改变了学习结果与交互输入，不能区分新增残差是否依赖正确模板。209不训练，以同权重干预检验推理时的输入依赖。干预发生在新增适配器，不能解释为移除了SGLA原有目标信息。

## 来源及冻结边界

读取208完整目录、原包SHA256 `1b5e2589d0e554792a07ab29d5f160087aa95d3c1c0c7aaa9a8c1430196c10bd`，decision `0058f9409c0437c4c8401e15f204fcdb41c2d1aa694dde56962486647bb8eadc`；核验16段、14表及完整物理文件／旧代码／协议。只取own_pair/matched的四模型（两fold×两seed），按原condition子plan、PT/receipt/scope/state绑定。错配参考复用208冻结native_pair-fit均值和映射，held不参与均值计算。无需额外194/207物理输入。

所有分支的Q/K/V/O数值、原生模板、原生L6输入、MLP、候选执行规则、CENTER、Hann、裁剪与写回公式不变。正确／搜索／错配均保持原残差注意力公式，仅选择键值来源：当前模板token、当前搜索token池化、冻结fit其他源模板均值。零残差直接返回输入，不把已训练O权重清零；其权重仍与其他三组完全一致，只旁路新增残差。没有新参数训练、优化器、调系数、选seed或选GT动作。

此处搜索／错配切换是针对匹配训练模型的输入干预，可能产生分布外输入；性能下降只支持该模型的输入依赖，不等于正确交互是所有可训练架构中的唯一有效机制。208的分别训练控制仍须一起解释。

## 预检与完整评价

按固定名单在每个held fold取首个视频，各前最多8帧，全部9策略（原生＋四模式×双seed）串行、双进程各一次，共18对。每帧框／父框／层／所有调用计数严格一致，计时排除。再在相同原生父框上运行全部9策略，逐帧核验适配前F6和原MLP分数严格相同、原生选择层相同，零残差输出框与原生完全一样。相同首帧模板初始化后执行固定父框，未来GT只在运行后计算IoU。这里的有限预检不代表所有未来输入上的普遍证明。

所有检查通过后，在38历史开发源组完整OOF评价342条：38原生＋4模式×2seed×38，每视频内部逐帧闭环。两seed零残差都必须与新原生完整轨迹逐字段相同，只有计时不比较。同完整runtime时，原生和正确输入114条还必须逐字段精确复现208旧轨迹。换L20/L40或UUID需新输出根、所有分支和原生同卡新跑，记录旧卡复现误差而不要求跨卡逐位一致；软件、权重、配置保持，续跑禁止换卡／workers／源码。

默认两个独立spawn进程跨视频，独立runtime和hook、不共享CUDA对象、不重复同视频任务；训练次数0。单总体进度条，worker日志单独保留，父进程持writer lock并汇总。沿用208调度器，不修改旧文件。墙钟可描述，但没有同任务完整串行对照，不声称GPU加速或隔离FPS。

## 分析与后续

两个主要比较：固定权重matched−search、matched−wrong完整AUC，seed先源内平均，38源bootstrap20000次seed24194，报告名义95%与两主要比较Bonferroni家族区间。matched−zero是适配总贡献的支持比较，zero−native是身份控制。继续报告低IoU、最长失败、匹配恢复丢失、好帧损失、新增稳定失败与逐源／seed；确认集封存。最终状态REVIEW，不自动采用。

如果正确输入仍没有可靠增量，降低新增模板交互为核心创新的优先级，保留训练状态和反馈目标的研究线索；如果有输入依赖但原生保护仍未过，下一步应检验恢复保护而不是宣布整体成功。两种情形都不能从此直接声称动态选层有效。

## 运行与回传

```bash
cd ~/workspace/paper
mkdir -p output/stage209
nohup bash tools/run_stage209_reference_intervention.sh > output/stage209/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage209/run.log
```

默认GPU0、STAGE209_WORKERS=2、根output/stage209/frozen_reference_intervention_v1，可覆盖STAGE209_ROOT／GPU／WORKERS、STAGE208_ROOT／BUNDLE、CHECKPOINT和七数据入口根。需208完整缓存与PT，只有结果TXT不够。完成后自动打包10张表＋decision＋manifest至stage209_results_<UTC>_<PID>.txt。--check-results复核／重新打包。若失败回传报错及worker_logs；不放宽身份或精确一致性条件。
