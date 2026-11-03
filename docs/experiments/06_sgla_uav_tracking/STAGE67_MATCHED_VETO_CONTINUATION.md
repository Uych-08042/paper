# Stage-67：同一起点的一次放行／否决及后继诊断

日期：2026-11-03，Asia/Hong_Kong。Stage-67服务器已完成，CSV修复成功，diagnostic／repair及7表共9份材料齐全。实际7支持起点／28短分支／3500搜索；0176:310一次放行有明显局部收益，0307:370却更差，详见[结果复核](STAGE67_RESULTS_REVIEW.md)和交接0.27。下文固定设计、原采集入口及恢复操作均为历史，不重复运行。

## 已完成采集后的汇总恢复

该恢复已成功完成并收到结果。以下保留问题原因和操作记录，不要求再次执行。

原`stage67_anchors.csv`直接导出`plan["requests"]`，首行F01没有`frame`字段，固定帧请求P01则有`frame=664`。原CSV发布器仅用首行生成表头，最后一行触发`dict contains fields not in fieldnames: 'frame'`；这是导出问题。原测试只有一种请求形状，漏掉了混合请求回归。

Stage-67的5个原脚本已经由manifest封存，直接修改summary或test会造成代码哈希不符。新增`tools/stage67_summary_repair.py`只在本次汇总调用内，为锚点表补齐统一的6列：case_id、sequence、selector、frame、status、resolved_frame。未指定的请求frame为空，已解析的实际锚点仍留在resolved_frame，未知字段继续拒绝，其他表沿用原发布器，已有不同字节结果不覆盖。

该入口完整调用原汇总及原来源／代码／receipt／对角线／预算校验，不修改manifest或缓存，不启动prepare、smoke、collect、训练、图像读取或GPU推理。另写`analysis/stage67_summary_repair.json`记录修复代码SHA、原代码指纹、诊断及9表哈希。

同步三个新增修复文件后，在原服务器仓库、`sglatrack-test`环境执行：

```bash
bash tools/run_stage67_summary_repair.sh
```

如原实验用了自定义根目录，先`export STAGE67_ROOT="原Stage-67根目录"`。新脚本只重建汇总并通过`cat`生成结果TXT，包含原8份材料和修复receipt。不要重跑原`run_stage67_veto.sh`，也不要编辑manifest／删除collection。新增回归覆盖8种真实请求形状、未支持的固定帧、原失败后恢复和原CENTER来源链；准确验证记录见交接0.26。

## 研究问题

Stage-66原生protect对连续质量选择Q下降2.8546 AUC，对fit接管率匹配的margin下降2.4791；0176的首次失效提前至315且未恢复。q0预测为良好，并不直接说明否决另一候选有利。总体接管率、当前候选IoU和两个不同历史上的完整成绩，无法单独确定某次否决的后继代价。

本轮固定SGLA V1、四候选生成、Q评分器和绝对q0预测器，检验：**同一进入历史上，否决与放行一次Q提议会怎样改变未来32帧；这种影响是否依赖后续继续采用protect或Q？** 结果用于定义后续接管目标，当前不新增评分器、不调0.5／0.02、不做新一轮训练。

## 固定锚点与缺失处理

所有起点均来自Stage-66原生protect自身完整轨迹。`changed_veto`要求来源确实否决了非q0的Q提议，且提议框与q0框至少一个浮点坐标不同；仅候选索引不同、输出框完全相同不算干预。

| 请求 | 视频 | 固定选择规则 |
|---|---|---|
| F01 | 0176 | 第一处changed_veto |
| F02 | 0307 | 第一处changed_veto |
| F03 | 0226 | 第一处changed_veto |
| F04 | 0204 | 第一处changed_veto |
| E01 | 0176 | protect首次失效起始帧及之前31帧中，最后一处changed_veto |
| E02 | 0307 | 同上 |
| R01 | 0226 | protect首次恢复起始帧及之前31帧中，最后一处changed_veto |
| P01 | 0204 | 664及之前31帧中，最后一处changed_veto；664是早期已消费的原生跳点对照 |

完整序列名由`stage67_veto_common.CONFIG`固定。事件采用原3连续低IoU帧确认、3连续良好帧恢复定义；仅离线确定事件锚点。F规则本身不按GT挑选，但视频名单来自已知开发结果。`prepare`把具体帧写入封存manifest，推理前即冻结。

缺事件、窗口没有有效否决、选中帧不足完整32帧，均在`stage67_anchors.csv`明确记为不支持，不找替代帧、不缩短、不换视频。全部不支持则退出。相同帧可能由不同规则选中，保留请求身份并分别计费；重叠窗口不能充作独立样本。

## 共同输入与2×2分支

在每个锚点恢复来源实际历史及原始首帧模板，重新执行一次共享四搜索，得到q0和Q提议等四个公共解码框。共享观测须精确复现来源：框、候选、裁剪、图像裁剪哈希、层选择、特征、评分及否决记录。dense解码误差另外按原固定容差审计。

只强制锚点的一次输出，然后各自在自身预测历史上推进31个后继帧：

| 锚点选择 | 后继控制器 | 作用 |
|---|---|---|
| veto：输出q0 | protect | 原来源对角线，全部后继须精确复现Stage-66 |
| allow：输出冻结Q提议 | protect | 同一未来控制器下，一次放行的局部效应 |
| veto：输出q0 | Q | 同一未来Q控制器下的否决分支 |
| allow：输出冻结Q提议 | Q | 同一未来Q控制器下的放行分支 |

同一控制器内`allow−veto`是匹配起点的一次决策效应。两种控制器的该差值之差另报交互，不能把控制器切换误写成一次否决的效果。若进入历史与来源Q也完全相同，额外要求Q／allow后继复现来源Q；否则只报告“不具备Q来源对角线”，不把两个自身历史混在一起比较。

原生V1仍为6个前缀块＋1个独立候选块，网络／首帧模板／原选择器不变，无E头。后继只接收当前图像与自身预测历史；GT仅用于离线选事件和计算指标，没有GT父框重置或在线oracle选择。

## 缓存审计、指标和预算

服务器先前60条来源轨迹全部校验并读取，汇总每条的否决次数、实际改变框次数、否决中“q0好而提议差／q0差而提议好”等当前机会。另列12视频protect与Q的第一次框、裁剪、索引分岔，并核对第一处框分岔是否对应相同四候选上的有效否决。这部分新增图像／网络为0，影子候选IoU不是后继效果。

分支报告当帧、32帧平均、仅31个后继平均、末帧IoU及良好／低IoU帧数。IoU单位0～1，不写作完整视频AUC百分点。`first_three_good_frame`仅指窗口内首次连续三帧良好的起始位置，窗口开始前事件状态未延续，不能直接当成完整视频恢复。

每个支持锚点4次共享搜索＋4分支×31帧×4搜索＝500搜索；最多8锚点、32短分支、4000搜索、28000块调用。首锚点veto／protect完整32帧用于smoke，128搜索直接复用，正式预算包含smoke。每条后继都做四搜索，按实际缓存调用数校验。没有新完整轨迹、模型拟合、阈值校准或采用门槛。

## 运行与回传

入口位于`tools/`，依赖Stage-66及其全部封存来源。工作目录必须为服务器仓库`/home/u25600003160604/workspace/paper`，原始V1 checkpoint和L40环境沿用；只允许显式绑定新的L40物理UUID，原来源对角线仍必须精确通过。

```bash
conda activate sglatrack-test
cd /home/u25600003160604/workspace/paper
bash tools/run_stage67_veto.sh
```

脚本依次prepare→smoke→collect→summary，失败立即停止；只复用已校验的完整缓存。可用`STAGE66_ROOT`、`STAGE67_ROOT`、`CHECKPOINT`、`STAGE67_GPU`覆盖默认路径／卡号。prepare要求独立空根目录，勿在该根目录内提前创建日志。

需要断开SSH后继续运行时：

```bash
mkdir -p output/stage67
nohup bash tools/run_stage67_veto.sh > output/stage67/run_stage67.log 2>&1 < /dev/null &
echo $!
tail -f output/stage67/run_stage67.log
```

以上后台方式适用于当前节点允许持续运行的环境；有调度器分配时仍受作业时限约束。脚本最后通过`cat`将diagnostic、anchors、case_summary、branch_summary、contrasts、scope、cached_video、divergences合并为`docs/experiments/06_sgla_uav_tracking/results/stage67_results_<UTC>_<PID>.txt`，提供此TXT即可复核。逐帧`veto_frames`与`timelines`保留服务器，先不合并长表。

本轮不自动commit／push。请先将本轮新文件及文档同步到服务器，再运行。旧Stage-1～66源码、模型及结果字节保持不变。

## 验证与结论边界

本机4项测试全部通过，130.555秒、无跳过。覆盖锚点缺失／不截短、实际框干预、后继历史独立拷贝、receipt篡改，以及真实CENTER的Stage-66→67来源链、共享起点、来源对角线、GT动作隔离、smoke计费与缓存续跑。四帧合成来源仅在测试中人为设置充分否决暴露；正式0.5／0.02等参数仍从真实封存来源原样读取。Bash语法、仓库保留校验及文档链接检查通过，具体记录在交接0.25。CPU小样本通过不代表服务器L40实测。

若一次放行在固定未来控制器下改善后继，可支持相应状态下“当前q0资格不足以判断否决价值”；若效应仅在切换控制器后出现，需研究未来控制器与状态的交互；若无效，则保留该负证据，不能认定本次锚点解释了315等更晚全视频失效。局部干预、重复开发视频和事件锚点不建立外部泛化、统一新方法成功或总体因果占比。
