# Stage-72：PPO同状态决策与训练动态缓存诊断

状态：服务器已完成，`COMPLETE_CACHED_PPO_DECISION_TRAINING_DIAGNOSTIC`，diagnostic＋9表齐全，见[结果复核](STAGE72_RESULTS_REVIEW.md)。本页保留结果前固定协议及实现验证历史，接续[Stage-71结果](STAGE71_RESULTS_REVIEW.md)与[固定PPO协议](STAGE71_BOUNDED_SEARCH_PPO.md)。新增训练／图像／SGLA调用为0；COMPLETE不表示方法采用，Stage-71仍REVIEW。旧代码、模型、轨迹和结果保持原样。

## 1. 本轮要回答的问题

Stage-71的PPO在两个种子上都低于相同曝光的监督控制。0304／0307是主要损伤，0204保留；当前汇总无法区分选择器与历史形成的问题。本轮检查：

- 在**同一个存储状态、同样四个输出框**上，PPO是否错过监督控制能够选择的好框，或接受有害替换？
- 失败和恢复窗中，是q0／Q二选一没有好框，还是四个公开候选都没有好框？
- 固定训练期间，actor概率与参数是否变化？critic与同段冻结GAE目标、实际行为轨迹折扣收益的误差如何？日志中的policy／value／梯度尺度是否提示值得验证的优化问题？

没有新训练、图像读取、SGLA搜索、完整轨迹或反事实分支。小控制器用NumPy在已有特征上前向重评分，因此“0 SGLA推理”不等于完全没有CPU计算。GT仅用于离线标签、事件定位与诊断。

## 2. 同状态比较及种子处理

读取Stage-71全部132条评价轨迹：5旧政策q0／Hann／Q／future／mixed_future各12条，PPO／myopic／ppo_rate各24条。每个状态比较q0、Q、PPO、myopic、ppo_rate五种规则，均选择同一组存储公开框，不改变历史。

旧政策没有学习seed，对每条旧历史分别使用两种子PPO／myopic配对模型评分；新政策历史使用其自己的同seed模型。得到192个“提供者轨迹×评分seed”上下文、216400次状态出现、1082000个规则决定。原评价缓存本身是148775帧出现；不要把两seed重评分、多个政策或相邻帧当成独立视频。

先在同视频、同提供者内平均提供者／评分种子，再对12视频作宏平均。五规则×八提供者形成40行same_state_summary，两个比较PPO−myopic／PPO−Q形成16行matched_summary。所有逐视频／seed记录保留在长表。仅使用相应模型的held-out组，不把另折fit表现混入。

指标包括当前IoU、good／low比例、公开候选最好框与二元最好框的当前regret、q0 good→low替换、q0 low→good、错过好框、接受当前更差Q、否决当前更好Q、非q0率，以及真实改框分歧。good为IoU≥0.5，low为IoU<0.2，当前严格收益epsilon1e-6。

**二元支持与公开支持分开。** q0／Q中没有好框，而其他公开候选有好框，说明提议限制可能重要；四个公开框都低质量，说明当前这些候选对任何选择器都不足。这里没有raw dense maps，不能把“四公开框无好框”扩大为整个网络或全图无可定位目标。

重评分API不接收GT、帧编号、视频名或未来信息。使用冻结归一化、±10特征裁剪、原actor零logit阈值及原rate门槛，不重拟合／校准／选阈值。批量NumPy前向对接近零logit（1e-12内）用原单样本前向确认，原记录规则必须复现其选择。

## 3. 固定关注窗与历史分岔

| ID | 视频 | 锚点来源 | 定位规则 | Stage-71已知帧，仅核对用 |
|---|---|---|---|---:|
| C01 | 0304 | PPO2172 | 首次失败 | 554 |
| C02 | 0304 | PPO2171 | 首次失败 | 1094 |
| C03 | 0307 | PPO2171 | 首次失败 | 395 |
| C04 | 0307 | myopic2171 | 首次恢复 | 470 |
| C05 | 0307 | PPO2172 | 首次恢复 | 475 |
| C06 | 0176 | PPO2172 | 首次恢复 | 386 |
| C07 | 0204 | q0 | 首次失败 | 664 |
| C08 | 0226 | ppo_rate2172 | 首次失败 | 119 |

每窗前后30帧，含锚点、视频边缘截断。锚点由已封存来源按规则解析；缺失事件明确标无支持，不改成邻近点。每窗同一帧检查全部11个提供者轨迹及上述评分seed，默认8×16＝128行关注摘要；长帧表留服务器。关注事件是已经消费的开发诊断，不能称独立验证。

另按全12视频×2种子输出24条PPO自身与监督自身的框、候选父框、候选输出、特征和所选裁剪首次分岔。索引相同可能框不同，索引不同也可能框相同；只将完全相同候选上的决定差标为同候选分歧。分岔先后不能独自证明失效因果。

## 4. 训练动态：读取已提交更新，不再更新

读取96段训练NPZ及每段更新前／后的JSON参数、Adam链和日志，核对原来源哈希与前链。每段对PPO／myopic各一行，共192行training_episode长表。按fold／seed／arm／pass聚合为32行training_summary；每格6视频的段均值，不称帧数池化或独立测试精度。另有8行optimization_summary。

每段在同一PPO行为轨迹的x上重算：

- 更新前后actor概率、绝对logit、eligible内确定性放行率、概率在[0.45,0.55]的比例、决定翻转、对当前myopic标签的准确率，以及actor／critic参数步长L2。
- 固定GAE target：沿Stage-71定义，从原behavior value计算一次；比较更新前／后critic在同目标上的MSE。
- 实际行为轨迹折扣MC：`G_t=r_t+gamma*G_(t+1)`，视频末尾为0；报告value误差和`1−Var(G−V)/Var(G)`，目标方差≤1e-12时置空。
- PPO更新后的实际已采动作相对原behavior log probability之比，计算clip0.2外比例和近似KL；不把myopic的监督分类器当作PPO行为分布来计算这两项。
- 原训练日志中的policy／value loss、entropy、approximate KL、clip比例、actor／critic／joint gradient范数和最大梯度。将“日志值”与本轮重评分值明确命名区分。

critic没有与actor共享参数，但原训练存在共同梯度范数裁剪。`mean(critic norm)/mean(actor norm)`是尺度描述，不能还原每个mini-batch的裁剪系数、Adam有效步长或证明critic压制actor。PPO当前标签准确率只是当帧描述，不代表其累计收益目标正确率。接近0.5的概率本身既不能证明未学习，也不能成为追加训练的依据。

更新后的actor／critic在**更新前PPO轨迹**上的误差仍是同曝光描述；myopic critic也使用这条共享return。不能将其称为更新后政策的真实value、独立泛化或新的闭环成绩。固定GAE target与全折扣MC定义不同，不能混写。

## 5. 证据解释与停止边界

如果同一组框上PPO漏选而监督选中，只能确定当帧决策差；若不同自身历史的候选支持分化，需要将输入状态形成与当前选择区分。即使actor步长小、概率近半且梯度尺度悬殊，也只是下一轮优化处理的依据，不能据此断定加步数会成功。

本轮无方法采用门槛、无模型选择或新成绩；COMPLETE只代表诊断执行完成。固定查看全12视频及正负控制，不按本轮结果改旧奖励、种子、配比或阈值。若以后需要动作的长期因果效果，再另行预声明有限共同起点分支，不能将旧缓存作为可任意推进的环境。

## 6. 服务器入口与结果文件

需要完整`output/stage71/bounded_on_policy_search_ppo_v1`及其Stage-70～61源链。合并结果TXT无法替代原模型／优化器／NPZ／receipt。此轮不需要GPU、checkpoint参数或Torch／OpenCV导入。

在原仓库根、已有NumPy的`sglatrack-test`环境同步新增代码后执行：

```bash
bash tools/run_stage72_ppo_audit.sh
```

关闭远程连接仍运行：

```bash
mkdir -p output/stage72
nohup bash tools/run_stage72_ppo_audit.sh > output/stage72/run_stage72.log 2>&1 < /dev/null &
echo $!
```

默认新空根`output/stage72/cached_ppo_diagnostic_v1`，日志位于其父目录；可指定STAGE71_ROOT／STAGE72_ROOT。来源及四个新代码文件写入新manifest，前后核对所有冻结字节；已存在同内容产物可重跑，内容不同拒绝覆盖，不删除或修补旧记录，不自动commit／push。

脚本用`cat`合并diagnostic＋9表为`docs/experiments/06_sgla_uav_tracking/results/stage72_results_<UTC>_<PID>.txt`。默认数据行数40／16／8／32／8／128／24／8／4，对应same_state_summary／matched_summary／candidate_summary／training_summary／optimization_summary／focus／divergences／anchors／scope。

15张CSV全部保存，逐状态关注窗、逐seed视频、192行训练动态及逐源scope等长表留服务器。回传合并TXT即可。当前Stage-71仍为REVIEW；本机检查记录及后续接续入口见[WORK_HANDOFF第0节](../../../WORK_HANDOFF.md#0-下一次对话的直接接续入口)。

## 7. 本机验证记录

三项语义测试已通过（0.483秒）：离线GT不进入决策、二元与公开候选支持分离、相同框强制保留、零logit平局、终止折扣MC／GAE与零方差空值、视频内先平均seed、缺事件不补点、可选CSV字段及不可覆盖。首次测试有两处新fixture错误（缺origin；相同框却预期可放行），已修正测试样例，没有放宽生产逻辑。

真实CPU工作流子测试也通过（该次含来源链构造的测试运行总378.526秒）：真实CENTER生成完整Stage-71微型来源后，在禁用Torch／OpenCV、图像读取、SGLA capture和optimizer update的条件下，Stage-72完整运行、重复产物一致、旧文件及网络状态不变，训练NPZ／检查点／上游CSV／结果篡改均拒绝。微型为22评价缓存／88源帧、128评分状态出现／640规则决定、8训练段／32帧；只是实现验证，不是服务器诊断成绩。修正后的语义测试与该工作流子测试分别通过，无跳过；未重复不受fixture影响的旧来源构造。

三份新Python AST、全新进程禁止Torch／OpenCV导入、CLI帮助及Bash `-n`通过。维护检查505保留／200移动／106活动模块、0缓存／0换行差异；七份相关文档216处本地链接、协议表格列数、`git diff --check`和Stage-71原件字节保护通过。实验及修复旧源码未改，未代commit／push。
