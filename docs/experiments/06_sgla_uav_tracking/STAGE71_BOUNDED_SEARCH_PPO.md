# Stage-71：冻结SGLA的有限预算搜索决策PPO对照

状态：服务器已完成，decision＋9表齐全，见[Stage-71结果复核](STAGE71_RESULTS_REVIEW.md)。PPO未超过匹配监督、能力保护未过，保留本轮结果，不重跑或追加预算。下文保留运行前固定协议及本机验证历史，接续[Stage-70结果](STAGE70_RESULTS_REVIEW.md)、[Stage-69诊断](STAGE69_RESULTS_REVIEW.md)及[Stage-68完整对照](STAGE68_RESULTS_REVIEW.md)。

## 1. 问题与本轮新增价值

现有单步或冻结Q后继预测能修复部分决定，却没有稳定保住完整自身轨迹：Stage-70消除0304新增事件，同时0204持续失跟、0307仍未恢复。本轮检验：**在自己实际造成的状态上，用累计跟踪收益训练接受／否决控制器，是否优于同状态曝光的当帧监督及既有Q／future？** 这次直接实施RL小试验，不再增加混合比例验证。

保持原SGLA V1、VisDrone-SOT、原checkpoint、CENTER头、选择器、独立候选、原首帧模板和四个既有搜索父框。V1每次搜索执行6前缀块＋1独立候选，不是串行12层。GT仅首帧初始化进入跟踪器，后续fit GT只供奖励／标签，held-out GT只供离线评价。原Stage-1～70代码、模型与结果不修改。

PPO和GAE采用既有算法：[PPO原论文](https://arxiv.org/abs/1707.06347)、[GAE原论文](https://arxiv.org/abs/1506.02438)。新增价值的候选是SGLA搜索决策的长期效果与能力保持，不把“采用PPO”单独称为创新成立。

## 2. 状态、动作和训练对照

每帧仍执行同样四次SGLA搜索，冻结的本折Q提出一个候选。二元动作0＝保留q0，1＝接受Q。若Q就是q0或框完全相同，只允许0；不引入新的重检测、模板更新、相机运动或动态层动作。

状态沿用Stage-68的128维：`[f(Q)−f(q0), f(q0)]`，其中f是已有64维在线候选特征。Q为0时前64维置零，后64维仍描述当前q0，供critic估值。标准化中心／尺度复用该折Hann-only fit统计，逐维裁到±10；无GT、帧编号、源视频名称或未来观测。没有新增RNN／记忆，既有预测历史仅用于四候选和因果特征。

| 项目 | 固定值 |
|---|---|
| actor／critic | 各128→32 Tanh→1，独立参数，无共享主干；actor4161、critic4161，每臂8322参数 |
| 学习臂 | PPO、myopic，两者同初始化、架构、归一化、状态、mini-batch排列和更新次数 |
| PPO actor目标 | clipped policy loss，clip0.2，eligible状态上归一化GAE advantage |
| myopic actor目标 | 当前同状态Q框IoU严格高于q0＋1e-6的二元标签，BCE；平局选q0 |
| critic目标 | 两臂均拟合由该次冻结PPO rollout产生的相同GAE return；critic不与actor共享参数 |
| 奖励 | 实际接受框的IoU，0～1；无额外失效／动作／时延惩罚，无reward搜索 |
| gamma／lambda | 0.99／0.95 |
| Adam | LR3e-4、epsilon1e-5、无weight decay；梯度范数clip1 |
| loss系数 | value MSE0.5、entropy0.01；相同entropy项用于两actor |
| 每视频收集后更新 | 4 epochs、mini-batch256；最后不足256也更新 |
| 收集行为 | PPO随机动作；整段视频收集期间该actor／critic固定；视频收集完才更新 |
| 评价行为 | 最终PPO／myopic logit>0放行，零或负保留q0；不采样、不挑checkpoint |

每条fit视频始终从首帧初始化，之后跟随自己的接受框推进到视频结束；失败后不使用GT重置，不跳转到已知事件窗。完整视频是自然terminal，最后next value为0，不把下一视频value接到上一视频；GAE在整段内计算，不把mini-batch边界当终止。actor梯度仅来自eligible状态，critic使用全部状态；没有提议的帧仍产生真实奖励和后继影响。

**监督控制只使用PPO所收集的同一状态矩阵，不另外收集自己的fit历史。** 它在评价时独立滚动。这样严格匹配新增数据曝光和优化预算，但比较同时包含PPO累计目标、随机行为收集和监督分类损失的区别，不能声称单独隔离“长程目标”或证明所有监督方案逊于RL。当前候选IoU作为fit标签不代表额外执行反事实未来分支。

## 3. 划分、曝光和预算

沿用12个已经反复消费的源组、2折每折6 fit／6 held-out；种子固定2171／2172，两种子同折使用相同预定视频顺序。每模型4遍完整fit视频，共24段；PPO从零输出logit的0.5随机策略开始，无旧PPO checkpoint或先验行为克隆预热。每次收集完同时更新PPO及匹配myopic。每模型固定最后一次更新作为评价模型。

| 产物／计算 | 正式预算 |
|---|---:|
| PPO训练模型 | 2折×2种子＝4 |
| 匹配监督模型 | 4；共8个新actor＋critic模型包，0 SGLA参数训练 |
| 实时fit完整轨迹 | 96＝2折×2种子×6视频×4遍 |
| 实时环境步 | 108200 |
| 实时训练搜索 | 432800，四搜索／帧；myopic复用相同曝光，无额外搜索 |
| 完整评价 | 132条，595100搜索 |
| 同输入来源复现 | 每折4次，共8搜索，额外smoke且已计入总量 |
| 合计 | 1027908搜索／7195356块 |

两臂各1888次mini-batch更新（按当前12视频长度计算，代码按manifest精确计算）。每视频曝光4遍，不声称每视频梯度权重相同，长视频有更多mini-batch。时间／峰值显存不是这里的搜索预算，也不据此声称部署加速。

训练smoke复用第一折第一种子的第一条完整视频及其更新；其长度按固定顺序在manifest列明，正式训练复用。评价smoke为首视频前32帧×11个政策／种子实例＝1408搜索，正式评价复用。中断时已提交轨迹可重做尚未提交的CPU更新；已提交更新直接跳过，未提交的视频重新从首帧和固定随机种子执行，不从GT失败帧续接。上述为正式提交证据的调用数，异常中断的未提交尝试耗时不冒充零成本。

所有学习和轨迹推进仅使用本折fit源组。二折fit集互为对方held-out，检查的含义是每个被评价模型不使用其held-out组，不能把全项目未接触这些视频当事实。

## 4. 必要完整评价与采用条件

五个旧控制q0、Hann、Q、future、mixed_future各12条，共60，必须同卡逐帧复现Stage-70；PPO、myopic、ppo_rate各2种子×12条，共72。全部政策都四搜索／帧，均沿自己的历史推进，初始之后无GT控制。

ppo_rate在**同折全fit Hann状态**上，用旧Q gain阈值匹配最终PPO确定性放行率；无新增推理／GT。每种子独立校准，报告同分误差；不称评价实际动作／干预完全匹配。

先在视频内平均两种子，再对12视频作固定seed2171／20000次bootstrap。主要PPO−myopic；四项关键AUC家族为对myopic、Q、future、ppo_rate，Bonferroni分位数0.05/8与1−0.05/8；另报告对mixed_future、q0、Hann的名义区间。均报告逐视频差、胜平负、中位、最差与留一，两个种子各自宏差保留。

只有同时满足以下才标记PROMISING，否则REVIEW，结果均保留：

- 四项关键AUC家族下界均>0，且对myopic和Q均值均≥+0.5 AUC。
- 每个种子对myopic和Q宏AUC差均>0。
- 对原生q0稳定视频任何种子0新增失效；低IoU宏差≤0；原生可靠帧AUC差名义下界≥−1.0。
- 至少4视频实际放行、至少4视频实际否决，避免退化成常量动作。

重点描述0304稳定保护、0307恢复、0204救援是否丢失、0176／0147获益是否仍在；这些案例不驱动样本／奖励／阈值／checkpoint选择。PROMISING仍是消费开发证据，不是独立泛化或创新已成立。失败不追加本轮预算、不换seed挑好结果、不继续调PPO奖励或训练步数。

## 5. 启动、断线运行和结果回传

需要完整`output/stage70/fit_history_source_v1`及Stage-69～61源链、原checkpoint、原数据路径。合并TXT不足以运行。不登录或声称本机已执行服务器GPU。

同步代码后，在服务器仓库根和`sglatrack-test`环境执行：

```bash
bash tools/run_stage71_ppo.sh
```

关闭连接仍运行：

```bash
mkdir -p output/stage71
nohup bash tools/run_stage71_ppo.sh > output/stage71/run_stage71.log 2>&1 < /dev/null &
echo $!
```

日志位于实验根父目录，不污染prepare的空根。脚本prepare→source_bridge→train_smoke→train→smoke→evaluate→summary，不自动commit／push。原checkpoint、STAGE70_ROOT、STAGE71_ROOT及GPU可用相应环境变量指定；设计参数不提供结果后CLI调节。

默认新根`output/stage71/bounded_on_policy_search_ppo_v1`。manifest封存来源、源码、硬件、种子和预算；CPU float64学习actor／critic，冻结SGLA在L40执行。每段轨迹NPZ＋receipt，每次更新保存不可覆盖的JSON模型、Adam状态及前一步指纹；每段随机行为及排列由固定种子重建，不依赖未保存的进程RNG。状态与权重不同或缓存篡改直接拒绝。

脚本用`cat`合并decision＋9表为`docs/experiments/06_sgla_uav_tracking/results/stage71_results_<UTC>_<PID>.txt`。默认行数：policy8、paired28、seed_effects2、calibration4、training_scope4、fit_support24、focus132、scope132、video_influence12。训练逐段96行、seed_video132、video_summary96及events长表留服务器，与模型、优化器检查点、原始轨迹一并保留。

本机测试记录及最新接续状态见[WORK_HANDOFF第0节](../../../WORK_HANDOFF.md#0-下一次对话的直接接续入口)。Stage-70固定结果保持REVIEW；服务器未跑前不填写任何Stage-71精度。

本机`python -B -m unittest discover -s tools -p test_stage71_ppo.py -v`四项全部通过，179.968秒，无跳过：GAE延迟信用／自然终止、PPO clipped loss、在线特征／动作合法性、固定生产预算／fold隔离；真实CPU Torch actor／critic与NumPy推理一致，Adam序列化后连续更新完全一致；完整真实CENTER来源链延伸到Stage-70后，Stage-71实时收集／更新／完整评价／汇总与零推理续跑。断点测试先提交轨迹、模拟更新前中断，再复用轨迹补更新；改变后续GT不改变动作与历史，改变当前模型held-out GT不改变其fit更新，模型／缓存／来源与结果篡改拒绝。微型fixture为2视频×4帧、2遍、hidden4、epochs2，32训练步、22完整评价、488搜索；非服务器精度，完整生产配方另有预算语义测试。五份Python AST、纯NumPy帮助入口、Bash语法和维护检查通过。

最终种子边界检查仅使用CPU default_generator初始化控制器，避免重设原SGLA CUDA随机流；真实Torch定向测试禁止全局／CUDA reseed入口并核对外部CPU RNG保持，2.544秒通过。
