# Stage-64：GT-free候选可靠性选择原型

2026-11-03结果更新：服务器已完成96条完整轨迹，状态`REVIEW_CANDIDATE_RELIABILITY_PILOT`，主要对比和保护未过。七份回传材料已齐，详见[Stage-64结果复核](STAGE64_RESULTS_REVIEW.md)。下文保留结果前协议及准备验证记录，不按结果修改设计或重复运行。

日期：2026-11-03，Asia/Hong_Kong。已先读WORK_HANDOFF第0.18节及[Stage-63结果复核](STAGE63_RESULTS_REVIEW.md)。本协议在得到Stage-64服务器结果前固定；本机只准备代码与验证，未运行服务器推理。

## 问题与本轮新增差异

Stage-63的共同历史干预说明，保留有效q0可以避免局部持续丢失，选择非q0候选也可能恢复；当帧更好并不总有更好后继。本轮准备一个实际在线选择器：用可观测历史、候选几何一致性和响应形态决定接受哪个公共框，目标是兼顾有效状态保护和替代恢复机会。

本轮不训练SGLA网络。原始SGLA V1仍执行6个前缀块＋1个独立候选块，原首帧模板、结构、选择器及预测头不更新；E只使用既有OOF几何头。候选生成仍是Stage-61的history四候选，不新加搜索、输出平滑或GT重置。R01的recrop干预只提供机制动机，本轮不把“history选择器”写成已复现R01的在线修复。

与Stage-60通用未来均值评分不同，本轮使用**当前良好／低IoU标签的可靠性目标**，并给出完全匹配的当前IoU评分控制。未来回报没有被观测到的旧候选不被伪造未来标签；本轮不是未来价值学习。是否改善后继由完整自身预测轨迹评价。

## 训练状态与防止折间信息混入

输入完整Stage-63目录，验证封存计划、完成记录、diagnostic、6张CSV及60个NPZ／receipt和上游Stage-62／61来源。旧代码和原件均不改写，不重跑旧summary。

沿用原12源视频、2折来源组划分，每折6 fit／6 held-out。这些组已反复用于设计，仍是开发OOF，不是新外部验证。

每折**仅使用其fit组的原生SGLA history／recrop两类既有完整缓存**：所有跟踪帧及每帧四个公共输出，初始化帧不计。每视频等权，视频内两种政策等权，政策内各帧等权，每帧三个相对q0的候选对等权。不按GT事件挑帧，不围绕R01／R05额外采样。

不把E轨迹放进选择器拟合。既有E轨迹是按各视频OOF头生成的；将互补折E轨迹直接当当前折训练状态，可能带入评价组训练过的头。这里使用不适配视频标签的原生SGLA缓存避免该路径。E只在完整轨迹评价中作为迁移检验，两种子使用同一相应fold选择器，种子先在视频内平均。

拟合不需要新图像前向。两折共27050个原生四候选状态，108200个已解码框标签；两个目标重复使用相同状态，总计4个确定性小型评分器，没有新增SGLA权重训练。

## GT-free特征与匹配评分目标

每个候选使用旧24项GT-free特征，加8项：与其他候选输出的连续IoU均值／最大值、与q0框IoU、相对自身历史速度预测的中心残差、相对历史中位尺寸的对数误差、候选间中心／尺寸分歧、Hann峰值的相对排名。长度5的历史仅含自身已接受预测；距离以最近预测框面积平方根归一化，尺寸比也只用预测值。

32项先截断到[−10,10]，再附加各自平方，共64项。特征不含GT、视频名、帧号、源组ID、头身份或未来图像。折号只用于离线OOF模型分配，不是评分特征。

两个学习目标如下，IoU只在fit侧给冻结输出作标签：

| 评分器 | 当前候选标签 | 用途 |
|---|---|---|
| quality | IoU | 完全匹配的当前质量控制 |
| reliability | `1[IoU≥0.5] − 1[IoU<0.2]` | 奖励良好、惩罚低IoU，中间区间为0 |

两者都拟合候选相对q0的标签差，使用同一64项特征差、加权RMS归一化和固定岭系数0.01；不加截距、不搜索系数、不选择checkpoint。可靠性输出是相对效用评分，不解释为已校准成功概率。所有替代评分未超过q0＋1e-6时接受q0；其他并列沿原候选顺序。比较quality与reliability时只有标签目标不同。

## 完整自身历史评价与预算

所有评价策略都真正执行四次原生搜索，然后只把自己接受的框推进到下一帧，包括固定q0控制。原生四组、E两组固定为：

| 来源 | 政策 | 完整轨迹数 |
|---|---|---:|
| 原生V1 | Hann原选择、固定q0、quality、reliability | 12×4＝48 |
| 固定E | Hann、reliability；种子2132／2133 | 12×2×2＝48 |

E迁移不做quality／q0四调用新组，不能据此给E内标签目标消融结论。旧E current仅用于离线稳定／失效队列及来源描述。

本轮全96条轨迹均新执行在同一个已绑定L40上，不把旧卡结果混作主比较；允许prepare观察到的物理L40 UUID与来源不同，型号、软件、原权重／封存配置／模型状态仍需符合来源。Hann轨迹逐项精确复现旧history来源，原生q0公共框精确复现旧current；不通过则停止，不放宽容差。

正式共432800次搜索、3029600个块调用。首视频8个政策／来源单元各32帧的1024次smoke搜索直接复用到正式轨迹，额外smoke为0。有receipt的完整轨迹不重复执行；中断时未提交的视频尾段可能重跑，失败尝试开销不计入完成缓存预算。多候选和特征计算成本不等于部署加速，本轮不报告实时性或能耗收益。

## 预定判断与报告

主对比为原生reliability−quality；必要比较为reliability−Hann和reliability−q0。三项AUC家族区间用同一视频bootstrap、种子2164、20000次，Bonferroni端点为0.05/6和1−0.05/6。原生每视频只计一个确定性运行，E先平均种子再按视频等权；不按帧计算置信区间。

开发pilot的积极条件固定为：上述三个AUC家族下界均>0，且对quality／Hann的均值提升均≥0.5；相对q0无新增原稳定视频失效、平均低IoU不增加、可靠帧AUC名义CI下界≥−1；至少4视频实际改变Hann选择。E对Hann作为次要迁移比较，保留任何负例。所有条件只决定开发pilot为PROMISING或REVIEW，不建立新方法采用或外部泛化。

报告AUC、低IoU、可靠帧、最长失效、恢复／末尾未恢复、原稳定视频损伤、改变选择比例、逐视频影响及完整事件。所有视频都保留，不因N类没有良好网格而移除。可靠性标签只是代理，不能提前宣称已学会后继风险，或已经修复无良好网格的问题。

## 服务器执行与结果回传

新增tools下5个文件：`stage64_reliability_common.py`、`stage64_reliability_experiment.py`、`stage64_reliability_summary.py`、`test_stage64_reliability.py`、`run_stage64_reliability.sh`。

同步代码后，在`sglatrack-test`环境、同一个L40任务内执行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE63_ROOT="$PWD/output/stage63/matched_one_choice_continuation_v1"
export STAGE64_ROOT="$PWD/output/stage64/causal_candidate_reliability_v1"
export STAGE64_GPU=0
bash -n tools/run_stage64_reliability.sh
bash tools/run_stage64_reliability.sh
```

脚本顺序prepare→fit→smoke→evaluate→summary。prepare绑定设备且不做图像前向；fit为NumPy小评分器拟合，SGLA始终冻结。也可手工顺序执行：

```bash
python -B tools/stage64_reliability_experiment.py --mode prepare --input "$STAGE63_ROOT" --output_dir "$STAGE64_ROOT" --gpu 0
python -B tools/stage64_reliability_experiment.py --mode fit --input "$STAGE64_ROOT"
python -B tools/stage64_reliability_experiment.py --mode smoke --input "$STAGE64_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage64_reliability_experiment.py --mode evaluate --input "$STAGE64_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage64_reliability_summary.py --input "$STAGE64_ROOT"
```

脚本用cat合并decision及policy_summary／paired／scope／fit_support／video_influence／focus六表，生成`docs/experiments/06_sgla_uav_tracking/results/stage64_results_<UTC>_<pid>.txt`。回传一个TXT即可，其他逐视频／种子／事件表、模型及NPZ留服务器。不自动commit／push。

## 本机验证

3项测试通过，34.707秒、无跳过。覆盖64项因果特征及两种标签、q0并列规则、模型／缓存绑定，真实小通道CENTER及Stage-59→60→61→L40→62→63→64完整小流程。核对fit只读本折原生fit组、改变held-out GT不改拟合输入、改变未来GT不改演员输出；原Hann与q0控制复现、E头冻结、smoke复用、续跑零新推理、旧来源字节不变与模型／缓存／报告篡改拒绝。

小流程为2视频、16条完整轨迹、256次搜索；smoke128复用，余下128新调用。Stage-58上游边界、骨干、图像与设备为fixture；不是服务器96条真实轨迹验证。目前没有Stage-64服务器收益结论。

CLI help、4文件AST、Bash LF、维护检查通过。只读`bash -n`经自动审批在沙箱外执行通过，没有运行服务器或本机GPU实验。
