# 工作交接：SGLATrack 无人机跟踪与模型压缩探索

更新日期：2026-11-03（Asia/Hong_Kong）

当前结果阶段：Stage-67条件诊断已完成，状态`COMPLETE_MATCHED_VETO_CONTINUATION_DIAGNOSTIC`。汇总修复成功，9份回传材料齐全、原件逐字节归档；7支持起点／28短分支／3500搜索。0176:310一次放行在同一后继protect下32帧均值+0.4961、低IoU少21帧，0307:370却更差，见0.27。最新完整政策仍为Stage-66保护未过的固定负结果（0.24）。Stage-1～67实验及修复源码冻结，Stage-45检测支线暂不执行。

Stage-68已实现并通过本机4项测试，服务器尚未运行。先读0.28的[执行协议](docs/experiments/06_sgla_uav_tracking/STAGE68_FIT_PAIRED_TAKEOVER_ADVANTAGE.md)，再读0.27和[Stage-67复核](docs/experiments/06_sgla_uav_tracking/STAGE67_RESULTS_REVIEW.md)。在fit Hann历史上用本折Q采成对真实32帧后继，对比immediate／future／全fit率匹配门槛，再作7政策完整OOF。0.25／0.26已执行完毕，不重跑；当前没有Stage-68结果。

2026-11-01新增交接：**SGLATrackV2源码发现、训练日志判读与后续启发见第0.11节**。当前实验仍使用V1／E，不因V2发布默认切换基线；优先补原始V1权重上的候选选择与同预算控制，再考虑V2迁移及搜索状态—特征记忆一致性。此次只整理知识与建议，未启动新实验。
整理前 Git 基点：`91052cd71d49d1397197ea9d5f532f12d37d2c91`

## 0. 下一次对话的直接接续入口

本节更新至2026-11-03。**最新结果为[Stage-67条件后继复核](docs/experiments/06_sgla_uav_tracking/STAGE67_RESULTS_REVIEW.md)，Stage-68已准备、4项CPU测试通过、待服务器运行，先读0.28，再读0.27和0.24。Stage-63／67共同起点后继、Stage-65影子选择、Stage-66自身完整政策是不同证据，不混用。Stage-61～67材料齐全，不重跑。**

**先读本节0.28、0.27和0.24，再读0.22、0.20、0.18的共同状态／完整政策／单次选择后继；需要执行修复历史时才读0.25／0.26**；[Stage-60](docs/experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md)、[Stage-54](docs/experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md)和[Stage-55](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)保留较早证据。下文早期阶段的“下一步”和“尚无结果”均为当时历史，不覆盖当前状态；研究范围文档末尾的Stage-46待运行说明也是历史。

### 0.1 固定目标与当前进度

- 用户目标是**以SGLA为基础开展无人机单目标跟踪研究**，数据目前只有VisDrone-SOT；不默认改为检测、多目标跟踪或要求补齐XS-VID／VisDrone-VID。
- 评价采用首帧GT初始化后的完整自身预测轨迹。当前12个源组、2折每折6 fit／6 held-out、种子2132／2133已反复用于设计，是开发证据；不要称为新的独立外部验证。
- 当前使用的V1原生SGLA执行6个前缀块＋L7～L12中1个独立候选块，保留原始首帧模板，原生实现没有在线模板更新。它不是串行12层ViT。V2另有4＋1路径与历史特征，详见第0.11节，勿混用版本描述。
- Stage-54训练和完整OOF已完成，状态为`REVIEW_SHORT_ROLLOUT_ENDPOINT_PILOT`。Stage-55诊断已完成，状态为`COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`；该COMPLETE不是新方法PASS。
- **Stage-56服务器已报告完成并为REVIEW。** 12头／72条新完整OOF、4个完整末帧控制精确复现检查通过；主要AUC差−1.3179、名义CI[−5.3575,+1.5984]、6胜6负，种子差−2.0925／−0.5432。五份结果已齐，不重复索取，不重跑Stage-54～56。
- **Stage-57已完成并为REVIEW。** 主要质量选择对完整保持−3.6057、两个种子均负，对随机+1.0364但区间跨零；0204双种子无失效，整体恢复与原稳定保护未过。五份材料已齐，见第0.7节；Stage-58已报告完成，结果与接续边界见第0.8节。
- **Stage-58已完成并为REVIEW。** 12头／72条OOF，四个完整控制精确复现检查通过，投影曝光充分；主比较未过，能力保护未过。五份结果已齐，详见第0.8节。
- **Stage-59已完成条件诊断。** 7锚点×2种子、56个共享输入／112条分支，五份材料已齐，见第0.9节。中心×尺寸可强烈交互，特定混合父框能恢复同一冻结头的后继表现；没有新方法采用门槛。
- **Stage-60已完成并为REVIEW。** 96条OOF、8个scorer，SGLA权重固定。response方向积极但总体区间跨零；future未超过response，且新增current稳定视频失效。五份材料已齐，见第0.10节；Stage-1～60实验Python冻结。
- **原Stage-61的L20复用方案在启动前因设备不符退出。** 代码和冻结计划保留为来源；准备／诊断历史见第0.12节。
- **Stage-61 L40六组完整重测已完成并为REVIEW。** 完整12份材料已齐，原生history／current／recrop AUC65.2483／63.4512／58.7334；E对应67.7506／64.0079／61.0880。原生主比较及可靠帧保护未过，原件与已重算复核见第0.14节。
- **Stage-62已完成条件诊断，五份材料已齐。** 全量108条来源、121725状态／365175输出，新增0训练／图像／网络调用；4表SHA、指纹、来源和关注记录相符。失败附近候选不足与部分事件内漏选并存，恢复损伤有出裁剪和可见却输出差两类。详见第0.16节，不重复请求现有五份材料，不把COMPLETE当采用门槛。
- **Stage-63已完成条件后继诊断，六份材料已齐。** 12例／48短分支／6000搜索／0训练；原生0304一次q0保护和E0307一次q2恢复各有32帧效果，统一q0回退不足；4个N负例锚点dense也无良好框，后继均未恢复。5表哈希／48分支算术／36对比相符，详见第0.18节；不重复索取或重跑。
- **Stage-64已完成并为REVIEW，七份回传材料已齐。** 4个评分器、96条新完整轨迹／432800搜索，0 SGLA训练。原生reliability对匹配quality −4.3668、2胜10负；对Hann／q0也负，保护未过。E0304／2132局部修复未抵消0226双种子损伤，E总体两种子均负。见0.20，不重跑。
- **Stage-65已完成缓存诊断，六份回传材料已齐。** 96条自身轨迹＋24个fit参考、135250状态出现次数；新增网络／图像／训练0。0003有害替换成立，0226／0307不同历史的候选可用性分化；两评分器同状态IoU差小。见0.22，不重跑。
- **Stage-66已完成并为REVIEW，七份回传材料齐全。** 原生五组60轨迹／270500搜索，0 SGLA训练、2个q0预测器；protect对Q −2.8546、对margin −2.4791，0176恢复受损、0307未修复，fit接管率匹配接近。稳定集0新增失败但保护未过，见0.24。
- **Stage-67已完成条件诊断，汇总修复成功，9份回传材料齐全。** 8请求／7支持起点／28短分支／3500搜索、0训练。0176:310一次放行在protect后继下均值+0.4961、低IoU少21帧；0307:370放行反而更差，0204:664当帧收益大而后继差小。见0.27，不重跑；局部32帧不是全视频成绩。
- **Stage-68已准备，尚无服务器结果。** fit组成对Q后继相对增益，对比匹配immediate与全fit接管率控制；最多384对、84新完整轨迹、475468搜索，SGLA不训练，4个新CPU预测器。本机4项测试167.173秒通过，无跳过，见0.28。
- 当前以“保住多数适配收益，减少持续失跟及原稳定轨迹损伤”为主问题。新增价值可以建立在已有方法上；用户不要求绝对原创，也不希望因有相似工作就结束探索。

### 0.2 Stage-54：主要对照、结果及不能丢失的取舍

原生骨干、候选、选择器、响应参数和BN缓冲冻结，只训练原头size／offset。每步从相同的原生／fit-only donor预测父框分别推进两帧，第二帧使用当前训练头第一帧的实际输出。两条新训练策略预算匹配：普通监督`rollout_supervised`（S）与增加原生短分支末帧框保持的`rollout_preserve`（P）。资格仅使用fit GT：原生两帧IoU均≥0.5，且原生末帧中心落在学生末帧可用裁剪内。父框、取整、argmax均detach；保持梯度直接进入末帧几何，**不是完整BPTT**。

| 策略 | 全视频宏AUC | 低IoU比例 | 原生可靠帧AUC | 原稳定视频新增失效数 |
|---|---:|---:|---:|---:|
| native | 63.4512 | 14.4573% | 75.5602 | 0 |
| geometry_control（旧固定状态普通几何适配） | 63.6819 | 17.7348% | 72.6131 | 1 |
| static_anchor（Stage-53 anchor_reliable） | 61.9882 | 17.7123% | 71.3898 | 1 |
| rollout_supervised | 63.2814 | 17.6339% | 70.3889 | 2 |
| rollout_preserve | 64.0079 | 14.7953% | 72.0353 | 1 |

预定主要对比P−S：AUC+0.7265，名义95% CI[−2.3364,+3.9123]，4胜8负，中位数−0.4797，最小留一−0.2589；其获益视频为0176、0304、0307、0147。对旧静态保持+2.0197、CI[+0.2479,+5.0494]、12胜0负、最小留一+0.5753；对旧几何控制仅+0.3260、3胜9负。旧静态对照的训练状态和计算安排不同，不能把12/12全部归功于末帧损失。普通短分支自身对静态也有+1.2932、10胜2负。

保持组对原生AUC+0.5567，CI[−9.2140,+9.7376]，但可靠帧−3.5250、低IoU+0.3381个百分点，0304仍新增长期失效，保护要求未过。原10条获益视频全部仍优于原生，平均收益+6.0741，却低于本轮普通监督的+7.4575。主比较中0304／0307贡献+1.8794，被其余10条−1.1529部分抵消。宏正均值不能代替能力保持；可靠帧对原生9胜3负，损伤集中0304／0307，另有0176小幅下降。

逐种子必须保留：0304的S两种子均554失效；P2132推迟到1078，P2133仍554，两者最终均未恢复。0204的S2132为76.7902且无失效，P2132降至53.7408并在664失效；P2133虽也664失效，但893恢复。0307的P相对S两个种子AUC分别+14.8817／+8.2502；P2132在471／732恢复，P2133在511恢复但后来仍有末尾未恢复。0176的两条新策略均出现双种子收益，保持进一步改善，但没有避免最初失效。0170的普通监督新增短事件被保持消除；静态保持在0147丢失的正收益也被恢复。

### 0.3 Stage-55：5个锚点及直接机制证据

S／P分别指Stage-54普通短分支头／末帧保持头；hS／hP指来源完整轨迹在该帧之前的预测父框。5种历史和5个评价头均为native、S2132、P2132、S2133、P2133。

| 案例 | 源视频与锚点 | 32帧窗口 | 观察目的 |
|---|---|---|---|
| C01 | uav0000204_00000_s:664 | 664～695 | 无失效高分适配分支丢失 |
| C02 | uav0000304_00253_s:554 | 554～585 | 早期种子分岔 |
| C03 | uav0000304_00253_s:1078 | 1078～1109 | 推迟之后的剩余损伤 |
| C04 | uav0000307_04531_s:389 | 389～420 | 失效附近的后继发展 |
| C05 | uav0000307_04531_s:471 | 471～502 | 恢复附近的后继表现 |

每个锚点另重放−1／0／+1三个当帧输入；先共用同一个实际GPU特征比较各头，再从同一个预测父框分别用各头推进32帧。分支后续只用自身输出，没有GT重置或教师接管。协议总计75个共同输入、125条分支，4,075次完整前向＋300次仅头前向，新增训练为0；smoke缓存已含在其中。自身头×自身历史须复现来源32帧，首帧须与共同特征探针一致。

记账定义：`D = value(P,hP)−value(S,hS) = M+H`，其中`M = value(P,hS)−value(S,hS)`、`H = value(P,hP)−value(P,hS)`。当帧M是同状态换头差，32帧M还包含换头之后的自身历史演化；两条分解路径均精确但不是唯一因果百分比分摊。IoU是0～1，不能直接写成完整视频AUC百分点。

- **C01／2132**：S2132父框上五头当帧IoU均≥0.6471，其他四种父框五头均为0，coverage却均为1。当帧自身差D=−0.6768、同S起点M=−0.0297；32帧D=−0.7977、M=−0.0116、H=−0.7861。保持头在较好输入上的基本定位能力仍在，主要差与进入664前形成的状态相关；尚未定位更早哪一帧或哪种几何变化产生了它。
- **C02／2132**：P2132父框五头IoU均为0.7457～0.7994；S2132／S2133／P2133父框五头均为0，coverage均为1。当帧D=+0.7753、M=0；32帧D=+0.8010、M=0。P头在已经较好的P2132父框上相对S头的32帧均值反而略低0.0063。2133的P头放在该好父框当帧可达0.7994，其自身历史却失败，不能只按当前头能力解释种子差异。
- **C03**：P2132父框coverage=1，但五头最终输出均约0.187～0.193；其他三条适配历史coverage=0。在共同native起点，两种子P相对S的32帧均值为−0.0566／−0.0546，分别多4个低IoU帧；当帧却为0／+0.0107。这是匹配起点上的滚动损伤。P2132自身32帧差只剩+0.0430，末帧差0。不能仅凭这些表认定中心、尺寸或dense候选是哪一种问题。
- **C04**：10组共同起点P−S的32帧均值均为小正，约+0.0031～+0.0214；没有显示稳定避免首次失效或大幅恢复的证据。
- **C05**：恢复附近出现当帧小差、后继大差，数值见下表。2132五种起点的32帧均值均改善，2133四正一负；2133在S2132起点为−0.1884、低IoU多13帧。收益有状态和种子依赖。

| C05共同起点、同种子P−S | 当帧IoU差 | 32帧均值差 | 末帧差 | 低IoU帧数变化 |
|---|---:|---:|---:|---:|
| S2132起点，2132头 | +0.0038 | +0.1545 | −0.0329 | −17 |
| P2132起点，2132头 | +0.0236 | +0.3569 | +0.7716 | −13 |
| S2133起点，2133头 | +0.0163 | +0.3620 | +0.6601 | −10 |
| P2133起点，2133头 | +0.0043 | +0.1039 | +0.1130 | −5 |

C05／2133自己的32帧均值差仍为−0.1732：M=+0.3620被H=−0.5351抵消。其实际511恢复在窗口截止502之外；该窗口负差不否定完整视频+8.2502 AUC，共同起点大增益也不是自身实际轨迹已实现的增益。50条比较均为当帧框分岔、下一帧取整裁剪／响应分岔；这也出现在零IoU差及受损分支中，不能只按分岔先后判效果方向。零IoU差不等于框相同、没有干预或未学习。

### 0.4 接下来可以做什么，哪些还没有确定

当前检验两个相互关联的问题：**分岔前正确状态的保护，以及恢复附近的稳健推进**，兼顾0307条件收益与0204／0304损伤。Stage-56已将其具体化为三臂：Lsup+Pend、Lsup+0.5Pend、Lsup+0.5Pend+0.5Pfirst。保持两帧长度、原头size／offset更新范围和其他训练配置；三臂执行相同的首末梯度审计。主要对比双时刻保持与完整末帧控制，同时对比半强度控制，区分第一步监督与减弱旧约束的效果。完整末帧控制须复现Stage-54对应权重及轨迹。

Stage-56上述固定配方已经完成，未达预定保护与改进目标。后续优先检验原生参考与当前有效修正是否冲突：原生IoU≥0.5不表示比当前预测更优；这仍是假设，现有汇总不能证明梯度冲突。Stage-57已将其落实为完整末帧控制／条件等数量随机选择／相对IoU质量选择三个训练臂，主要对比质量选择与完整末帧，必要控制为质量选择与随机选择。只在fit内用参考IoU严格高于当前预测+1e-6来保留约束；不使用在线GT，也不搜索首末系数。该轮已完成：质量选择未超过完整保持，只在0204等出现局部进展。下一轮应区分当前IoU代理、持续保持强度与优化／后继风险之间的差别，不能把梯度冲突当成已证实原因。若设计确实依赖精确中心／尺寸／选点信息，先读服务器现有`stage55_point_details.csv`、`stage55_rollout_summary.csv`、`stage55_timelines.csv`，不重跑GPU采集。长表可让用户用文件附件提供；当前三张关注表已足够支持本次机制判断。

设计时核对真实梯度：响应参数冻结时，固定特征上的响应排序损失无法更新该分支；父框detach时，未来损失不能直接反传进前帧父框。普通回归用raw峰值，原生推理用Hann峰值，实际监督格点要写清并测试。不要未经证据把这处差异认定为当前全部病因。前轮的冻结范围是实验处理，不是用户要求未来永久冻结响应；若下一轮改变范围，单独预声明并匹配对照。

不能据现有结果写成：全部损伤都是当前头退化、全部收益都是既有历史、coverage=1就没有输入问题、两帧末帧保持已经实现长期稳定、32帧窗口等于整视频、COMPLETE诊断等于方法采用、开发OOF等于独立验证。也不要仅因存在相似工作否定方向；应明确本轮新增价值和仍需补足的机制。

### 0.5 文件、环境、产物与验证入口

本机仓库`E:\Program Files\paper\paper`，PowerShell；服务器仓库`/home/u25600003160604/workspace/paper`，环境`sglatrack-test`。原权重是`output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar`，不是本轮学习头。VisDrone-SOT训练根目录为`/home/u25600003160604/workspace/datasets/VisDrone-SOT/VisDrone2019-SOT-train`。

| 相对仓库路径 | 下一次接续用途 |
|---|---|
| `tracking/stage54_rollout_common.py`、`stage54_rollout_experiment.py` | 完整Stage-53来源与两帧训练／OOF入口，冻结不改 |
| `tracking/stage54_rollout_runtime.py` | 实时自身短分支、实际梯度与解码边界 |
| `tracking/stage55_matched_common.py` | 完整Stage-54来源校验、5个固定锚点、缓存与对角线审计 |
| `tracking/stage55_matched_collect.py` | 同GPU特征探针及原生32帧分支，无训练 |
| `tracking/stage55_matched_summary.py` | 分解定义、共同起点对比、局部窗口指标 |
| `tracking/test_stage55_matched.py` | 3项真实CPU测试，前轮90.476秒全部通过，无跳过 |
| `tracking/stage56_transition_common.py`、`stage56_transition_runtime.py` | 完整Stage-54来源、三臂系数、首末直接梯度及原控制复现 |
| `tracking/stage56_transition_experiment.py`、`stage56_transition_summary.py` | prepare／smoke／train／evaluate与结果汇总 |
| `tracking/test_stage56_transition.py` | 4项真实CPU测试＋增强目标函数检查已通过 |
| `tracking/stage57_quality_common.py`、`stage57_quality_runtime.py` | 完整Stage-56来源、相对质量／条件等数量选择、直接末帧梯度 |
| `tracking/stage57_quality_experiment.py`、`stage57_quality_summary.py` | 已完成阶段的prepare／smoke／train／evaluate及统计入口 |
| `tracking/test_stage57_quality.py` | 5项真实CPU测试通过，170.601秒、无跳过 |
| `tracking/stage58_gradient_common.py`、`stage58_gradient_runtime.py` | Stage-57完整来源及gS／gP、单侧投影／同长度控制 |
| `tracking/stage58_gradient_experiment.py`、`stage58_gradient_summary.py`、`test_stage58_gradient.py` | 已完成阶段与CPU验证入口，结果见第0.8节 |
| `tracking/stage59_parent_common.py`、`stage59_parent_collect.py`、`stage59_parent_summary.py`、`test_stage59_parent.py` | 已完成的父框分量交叉与32帧后继诊断；4项真实CPU测试此前已通过，结果见第0.9节 |
| `tracking/stage60_state_common.py`、`stage60_state_runtime.py` | 自身历史四候选、GT-free白名单特征、岭回归、原生探测与后继标签 |
| `tracking/stage60_state_experiment.py`、`stage60_state_summary.py`、`test_stage60_state.py` | 已完成prepare／smoke／collect／fit／evaluate及汇总；4项真实CPU测试此前通过，结果见第0.10节 |
| `tracking/stage61_search_common.py`、`stage61_search_runtime.py`、`stage61_search_experiment.py`、`stage61_search_summary.py`、`test_stage61_search.py` | 原始V1／E匹配重裁剪流程与3项已通过CPU测试；L20方案未运行，L40完整重测已完成，见第0.12～0.14节 |
| `tools/run_stage61_search.sh`、`tools/stage61_runtime_replay.py` | 读取封存配置严格复现启动身份，顺序运行prepare／smoke／evaluate／summary；cat合并9份结果及2份启动记录，不执行Git提交或推送 |
| `tools/run_stage61_l40.sh`、`tools/stage61_l40_common.py`、`stage61_l40_experiment.py`、`stage61_l40_summary.py`、`test_stage61_l40.py` | 当前L40独立入口；全部六组新完整轨迹，来源L20数据只进入描述性桥接表，见第0.13节 |
| `docs/experiments/06_sgla_uav_tracking/STAGE61_L40_RESULTS_REVIEW.md`及`results/stage61_l40_results_20261102T143543Z_40176.txt` | 完整闭环结果复核与逐字节归档原件；完整12份材料已齐，见第0.14节 |
| `tools/stage62_candidate_common.py`、`stage62_candidate_audit.py`、`test_stage62_candidate_audit.py`、`run_stage62_candidate_audit.sh` | 已完成NumPy候选缓存诊断，代码冻结；全量108条来源＋11关注点，4项准备阶段本机测试通过，见第0.15～0.16节 |
| `docs/experiments/06_sgla_uav_tracking/STAGE62_RESULTS_REVIEW.md`及`results/stage62_results_20261103T004524Z_148362.txt` | 全量候选诊断复核与逐字节原件，五份材料已齐；未上传长表不等于未执行，见第0.16节 |
| `tools/stage63_choice_common.py`、`stage63_choice_experiment.py`、`stage63_choice_summary.py`、`test_stage63_choice.py`、`run_stage63_choice.sh` | 已完成共同历史一次选择后继，源码冻结；4项准备阶段真实CPU测试通过，见第0.17～0.18节 |
| `tools/stage64_reliability_common.py`、`stage64_reliability_experiment.py`、`stage64_reliability_summary.py`、`test_stage64_reliability.py`、`run_stage64_reliability.sh` | 下一轮GT-free可靠性选择，3项真实CPU测试通过；服务器96条新完整轨迹待运行，见第0.19节 |
| `docs/experiments/06_sgla_uav_tracking/STAGE63_RESULTS_REVIEW.md`及`results/stage63_results_20261103T015043Z_159855.txt` | 当前最新局部后继复核及原样原件；diagnostic＋5表已齐，timelines和响应数组仍在服务器 |
| `output/stage54/short_rollout_endpoint_v1/` | 服务器训练、完整OOF及6张已收到表所在目录 |
| `output/stage55/matched_parent_continuation_v1/` | 服务器共同输入及分支缓存、3张已收到关注表和详细几何 |
| `output/stage56/first_transition_preservation_v1/` | 服务器已完成12头／72轨迹；五份结果已收到，完整GPU产物仍在服务器 |
| `output/stage57/quality_selective_endpoint_v1/` | 服务器已完成12头／72条新OOF，五份结果已齐，完整GPU产物仍在服务器 |
| `output/stage58/projected_endpoint_gradient_v1/` | 服务器已完成12头／72条新OOF，五份结果已齐；内部校验通过，完整GPU产物仍在服务器 |
| `output/stage59/parent_component_continuation_v1/` | 服务器已完成56共享输入／112条32帧分支，五份材料已齐；完整8表／缓存仍在服务器 |
| `output/stage60/causal_search_state_value_v1/` | 服务器已完成96条OOF；五份优先材料已齐，完整9表／fit缓存／模型／轨迹仍在服务器 |
| `output/stage61/v1_history_matched_recrop_l40_v1/` | 服务器108条L40完整轨迹与9表；12份优先材料已齐，完整来源链必须保留 |
| `output/stage62/cached_candidate_selection_v1/` | 服务器只读诊断已完成；diagnostic及4张优先CSV已齐，其余4张长表仍在服务器 |
| `output/stage63/matched_one_choice_continuation_v1/` | 服务器48条分支／6000搜索已完成；diagnostic及5CSV已齐，逐帧表、60个NPZ及receipt留服务器 |
| `output/stage64/causal_candidate_reliability_v1/` | 预定新输出目录，尚无服务器产物；prepare／fit／smoke／evaluate／summary入口 |
| `docs/experiments/06_sgla_uav_tracking/results/stage60_results_20261031T042013Z.txt` | 用户已上传并同步的五份合并原件，保持字节不变 |

Stage-54已收到scope／policy／retention／focus／paired／video_summary六表；Stage-55已收到focus／crossed_matrix／branch_contrasts三表，分别10／25／50行。Stage-56 decision及policy／scope／paired／focus也已收齐，四表6／12／40／30行。Stage-57及Stage-58同样已收到decision及四表，四表行数各为6／12／40／30。Stage-59已收到diagnostic及focus／crossed_matrix／factor_summary／exposure，四表14／56／42／14行。Stage-60已收到decision及policy／paired／scope／fit_support，四表5／30／8／24行，原件已存本机仓库results目录。**不要在新对话中再次请求这些材料或重跑已完成阶段。** 本机未持有服务器完整缓存／模型及未提供的详细表；Stage-56～60提供的decision或diagnostic完整文本均已重算自身指纹及四表SHA，但未独立重验服务器全部模型／缓存SHA或真实调用。仍未收到的Stage-54总体种子差不影响判断已有保护条件未过，无需为此默认追加索取。

用户查看文件偏好`cat`。定位服务器现有产物时先进入仓库再设置变量，防止此前出现的`~/workspace/output`误指；以下只定位，不启动实验：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE54_ROOT="$PWD/output/stage54/short_rollout_endpoint_v1"
export STAGE55_ROOT="$PWD/output/stage55/matched_parent_continuation_v1"
```

仓库忽略CSV／NPZ等实验产物，Git同步不是结果备份。服务器要保留Stage-54～63及上游来源链的manifest、run、收据、模型、特征缓存和轨迹；不能只留下analysis表或删除被新阶段校验引用的来源文件。

本机前轮CPU测试使用下面的已有运行时；仅在新改动需要验证时按需使用，CPU合成测试不是GPU精度复现：

```powershell
$env:PYTHONPATH='D:\Users\23920\Documents\Codex\stage46_cpu_deps;D:\Users\23920\Documents\Codex\stage34_cpu_torch'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Users\23920\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B tracking/test_stage60_state.py -v
```

维护检查为`python -B tools/check_workspace.py`，最近通过：505个保留文件、200个归档移动、106个原活动依赖、0个再生缓存。Stage-1～63及L40实验源码保持冻结；Stage-64为新工具／目录／协议，本轮保留先前未提交的Stage-63复核与原件。本助手未替用户commit或push。实验自身指纹校验严格，后续改变设计应新建阶段／目录，不直接改旧代码或收据。

可在下一次对话给出以下接续提示：

> 先读取WORK_HANDOFF.md第0节，特别是0.14～0.15。核心仍为SGLA V1／E无人机单目标跟踪。Stage-61 L40全套结果已齐且为REVIEW，原生历史的0226恢复损伤与E0307双种子损伤未解决。下一轮Stage-62已实现并通过4项本机测试，服务器尚未执行：运行tools/run_stage62_candidate_audit.sh，读取既有108条L40轨迹、121725个状态观察、365175个已解码候选。无图像读取／GPU推理／新训练。全量统计及11个固定锚点区分解码候选不足、漏选、q0即时损伤与救援机会；不推成dense网格能力、oracle政策AUC或未来收益。不要重跑旧GPU或再次索取Stage-61材料。Bash本机语法检查因MSYS与自动审批服务限制未完成，需服务器bash -n；其余验证见0.15。若已收到Stage-62结果，先读结果，不重复让用户运行。

### 0.6 Stage-56最新结果与接续边界

完整末帧E、半强度末帧H、首末各半T的AUC为64.0079／62.2953／62.6900，低IoU14.7953%／18.6970%／17.1608%。T−E为−1.3179（CI[−5.3575,+1.5984]，6胜6负），T−H为+0.3947（CI[−0.5834,+1.7709]，3胜9负），两个家族区间也均跨零。主要总体种子差−2.0925／−0.5432。T对原生−0.7612、可靠帧−3.9896、低IoU+2.7035个百分点，0304仍新增失效，未达积极开发条件。

执行审计：scope12行均冻结响应／缓冲，响应误差0，解码最大0.00024414；四个E均报告精确复现Stage-54。正式24,576次训练完整前向＋12,288次仅头、72条OOF共81,150次head／568,050个block，吻合原预算。三臂都计算首末影子梯度，非零first_gradient不表示控制也优化了第一步项。新方案首步梯度确有实际优化系数，但不能由此推断稳定性。

0176：E的双种子61.0138／59.1152，在T变成16.4485／64.6329，2132丢失44.5653，平均T−E−19.5238。0304：T仍1079／555首次失效至末尾，对E两种子都微降，没有原稳定能力恢复。0307：T对H的+7.2213主要来自2132（+15.5907），而对E两种子均下降，2133丢失原511恢复（T无恢复且末尾未恢复）。0204：T2132无失效75.1978，但2133在664失效且末尾未恢复；H的高分无失效分支是2133，T为2132。T对E+9.4230，对H却−0.9152，不是跨种子稳定保护。

五条关注视频贡献主要宏差−1.3148，其余7条均值约−0.0052。T−H的+0.3947也集中0176／0307，其余7条平均−0.4168。原固定10条获益队列仍10/10对原生为正，但E／H／T平均幅度约+6.0741／+5.2380／+5.0287。少数大损伤、种子分支互换与多数小收益必须同时记录。

五份材料在本机完成内部自洽核验：decision指纹、四张CSV按原CRLF还原的SHA、40条宏指标差、report／paired对应及scope计数。没有从全部12条video_summary重算bootstrap或独立验证服务器完整GPU产物。Stage-56全部Python保持冻结，复核详见链接。当前不需索取更多表作主要判断；如新设计依赖逐样本梯度或参考与GT误差，可读服务器现有history及详细轨迹，不能从宏结果补造几何病因。Stage-57也已由用户报告完成并提供五份材料，本助手没有登录服务器或执行GPU采集；不代用户提交或推送。


### 0.7 Stage-57最新结果与接续边界

完整保持E／条件等数量随机R／质量选择Q的AUC64.0079／59.3657／60.4022，低IoU14.7953%／23.9019%／21.8560%。Q−E为−3.6057，名义CI[−12.7607,+3.6011]，8胜4负，两个总体种子差−4.8582／−2.3532；Q−R为+1.0364，CI[−4.7107,+6.9128]，6胜6负。家族区间均跨零，主要与能力保护未过。

四个E精确复现检查通过，scope均冻结响应／缓冲、响应误差0、解码最大0.00024414。正式预算24,576次完整训练＋12,288次仅头，72条OOF共81,150次head／568,050个block。5项真实CPU测试170.601秒通过是前轮实现验证，本次只核对结果，不重新训练或推理。

每臂4模型eligible合计1550个。E全保留，R保留315（20.3226%），Q保留318（20.5161%）。Q无误选／漏选，R保留71个非较优、漏掉71个较优，二者实际总量接近但不是相同状态轨迹。Q当前状态上与影子随机有154个位置分歧＝77步／1024步（7.52%），混合批次161。Q所选参考当前平均IoU优势+0.02541，R为+0.01607；规则成功选到了当前更优参考，仍未转化为整体闭环收益。

0176：E两个种子约61／59，Q约16／16，两种子都无记录恢复、末尾未恢复；0307的Q两个种子也均丢失E的恢复。0304的Q均555失效至末尾，R反而两种子约1077／1082才失效。0226的Q两个种子均338恢复；R2133掉到7.1941且不恢复，Q−R+25.2906均值主要来自这一个种子。0204的Q首次双种子无失效（76.4593／76.7644），比E+21.5346、比R+9.6117，是应保留的局部积极证据，不能抵消其他大损伤。

Q对原生−3.0490、可靠帧−5.4126、低IoU+7.3987个百分点，有两个原稳定视频新增失效；已知0304，另一条不在focus中，名称和严重度未知，不猜0170。原固定10条获益队列正收益数量降为Q的8/10、R的7/10。Q−E多数小胜与少数大损伤并存；五条关注视频宏贡献−3.8143，其余7条平均+0.3576。Q−R五条宏贡献+1.0732，其余7条平均−0.0631。

五份材料在本机核对了decision指纹、按原CRLF还原的四张CSV SHA、40条宏指标差、report／paired对应及选择计数恒等式。未独立重验完整模型／缓存，也未从全12视频明细重算bootstrap。Stage-54无保持普通短分支63.2814仍高于Q，不能把丢掉约80%约束直接认定为唯一损伤原因。当前IoU代理、持续正则、梯度方向及后继搜索风险仍需区分。

随后Stage-58已检验全体合格项的梯度方向，服务器结果已收到，见第0.8节。不搜索本轮阈值／权重／种子，也不默认重复扩充锚点。完整history／video／seed／events仍在服务器，具体设计确需时再读，不重复请求已有五份材料。Stage-1～57实验Python冻结；未代用户提交／推送或登录服务器。

### 0.8 Stage-58最新结果与接续边界

[Stage-58完整复核](docs/experiments/06_sgla_uav_tracking/STAGE58_RESULTS_REVIEW.md)，[协议与历史命令](docs/experiments/06_sgla_uav_tracking/STAGE58_PROJECTED_ENDPOINT_GRADIENT.md)。输入完整Stage-57，产物`output/stage58/projected_endpoint_gradient_v1/`；Stage-1～58实验Python冻结。

E完整保持／N同长度保持／P单侧梯度投影AUC64.0079／59.6043／61.2503，低IoU14.7953%／22.0800%／19.8301%。P−E−2.7575，95%CI[−7.7311,+1.7310]，7胜5负，中位数+0.1413，两个总体种子−1.1681／−4.3469；P−N+1.6460，CI[−5.0702,+9.0367]，6胜6负，中位数−0.0019。两条家族区间跨零。P对原生−2.2008、可靠帧−6.7321、低IoU+5.3728个百分点，0304仍新增持续失效。主比较与能力保护未过，不写成已确认普遍劣效。

四个E精确复现检查通过；所有scope响应／缓冲冻结，响应梯度和重算误差0，最大decode0.0002441406252273737。正式12头／72OOF，训练24,576次完整前向／172,032个block＋12,288次仅头，OOF81,150次head／568,050个block，符合预算。前轮5项真实CPU测试219.308秒通过，本次只复核，没有重跑训练或GPU。

每臂均1550／1550个eligible保留，840／1024步保持梯度非零。E／N／P原始冲突647／650／647步；两个处理臂实际修改650／647步，P每模型139～182步，曝光充分。N／P实际保持梯度范数累计859.8988／859.4551，为自身原始累计的70.1179%／70.2513%；全部参考仍入损失，但梯度作用变弱。影子长度匹配与投影残差最大约4.62e-8／6.35e-8。实际保持与gS内积累计E−2547.5689、N−835.1935、P+548.2583，方向处理确实不同。范数比例不是保留参考数量，也不是两个真实状态全程完全匹配。

P的647个冲突批次含33批全参考当帧IoU都较优、614批含平局／较差参考；它是批次聚合，不能解释逐项原因或估计“全参考较优”条件冲突率。实际AdamW位移相对gS反向次数却仅E3／N4／P4（各1024步）。原始保持梯度冲突常见，不等于完整实际更新常反向；一阶点积正常也不保证真实损失和后继轨迹改善。

0176：P两种子27.4163／66.5622，N15.2627／14.3902，P−N两种子皆正（+12.1536／+52.1720）；但对E分别−33.5975／+7.4470，P2132虽首次失效推迟至315仍无记录恢复，未稳保E收益。0226：P2133只有8.3973、129失效后未恢复，E和N两种子均恢复；平均P−E−24.4624，损伤主要来自2133−48.9964。0304：P1086／555失效后都末尾未恢复，对E仅+0.1643，原稳定能力未恢复。0307：P474恢复仅2132有，两个种子均末尾未恢复，对E分别−3.7644／−10.7221，对N一正一负。0204：P2132无失效75.7200，2133仍664失效、892恢复；对E+11.0361、对N+12.1762均值是局部积极进展，未保住Stage-57在此双种子无失效。

五条关注视频对P−E宏贡献−2.7984，其余7条平均+0.0700；对P−N贡献+1.6523，其余7条平均−0.0108。P对原生9胜3负，负例均为0226／0304／0307；其余7条正。原固定10条几何获益队列保留9/10（N也9/10但丢0176），由胜负与focus推导而非已读retention。P事件数比E少，却恢复更少、末尾未恢复及最长失效更高，不能以事件少认定更稳定。

原几何控制固定失效帧区间（9视频）P比旧几何控制+6.5981、名义CI[+1.3344,+13.4681]、7胜2负，有局部价值；但对E仅+0.5821且区间跨零，对N+8.0239也跨零、4胜5负。它是离线GT定义区间、非共同起点因果恢复收益，不能代替全视频与可靠帧保护。

五份材料已核对decision指纹、四CSV按CRLF还原的SHA、40条宏指标差、报告字段与scope计数恒等式；未独立加载服务器完整模型／缓存，也未重算12视频bootstrap。此前中断轮次附件decision与本次完全相同，不算重复实验。材料足以作当前判读，不追加请求已有五份表。

Stage-58当时尚不足以指定中心／尺寸／选点唯一病因；随后Stage-59已完成预测父框中心×尺寸×冻结头交叉后继，具体条件效应见第0.9节。若新设计确需更细响应信息，可读服务器既有详细表；不据旧结果搜索投影阈值／权重／种子／checkpoint。不重跑旧训练，不代用户commit／push或登录服务器。


### 0.9 最新结果：Stage-59预测父框中心／尺寸后继诊断

[完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE59_RESULTS_REVIEW.md)，[协议与历史命令](docs/experiments/06_sgla_uav_tracking/STAGE59_PARENT_COMPONENT_CONTINUATION.md)。用户已报告`COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC`；五份材料已齐。输入完整Stage-58，输出独立`output/stage59/parent_component_continuation_v1/`。4个Stage-59 Python文件及Stage-1～58来源均保持冻结。

E＝完整末帧保持，P＝梯度投影。每种子分别取来源预测父框，交叉EE／PE／EP／PP（中心／尺寸来源），再以同种子E和P冻结头推进32帧。只交换起点，随后各分支用自己输出；没有后续来源框注入、GT父框或在线GT选择。原组合直接复制原父框避免重建误差；共享GPU特征探针另记录原生头。

7锚点固定：0226:338（E2132恢复）、0176:238（P2133恢复）、0176:315（P2132失效）、0304:555（P2133失效）、0304:1086（P2132失效）、0307:511（E2133恢复）、0204:664（E2132失效）。prepare逐项校验来源事件，每个锚点仍同时评价两个种子，不挑高分种子。它们是消耗过结果指导的条件案例，不是独立验证。

共56个共享输入＋112条32帧分支＝3640次完整搜索前向／25480个block＋112次仅头，总head3752。新训练及新模型为0。smoke先完成C01（520次完整＋16次仅头），缓存被collect复用而非额外预算。所有同来源父框／同头32帧对角线须复现来源，所有分支首帧须复现共享特征探针。

报告当帧／窗口均值／末帧的两条精确分解路径：先换头，再依次换中心／尺寸，或先尺寸／中心。另报三个顺序平均效应及两两交互；存在高阶交互时平均主效应不直接加成自身差，不声称唯一因果百分比分摊。尺寸干预改变搜索范围／分辨率／padding／响应竞争，不能直接等同尺寸回归问题；相同取整ROI也可能有浮点映射差。窗口IoU不是全视频AUC，局部事件有窗口截断。

4项真实CPU测试此前通过，301.661秒、无跳过；覆盖真实CENTER／原生decode、后继GT隔离、完整合成来源、缓存复用与篡改拒绝等。本次只复核结果、修改文档，不重跑CPU训练或GPU实验。服务器预算符合协议；来源32帧对角线、当帧／首图／响应误差均0，最大分支解码误差0.0002441406252273737像素。本机重算diagnostic指纹、四表CRLF SHA、42组两条分解及14组当帧完整析因均相符，最大记账残差5.55e-17；没有独立重验服务器全模型和缓存。13／14组各4个实际不同ROI及裁剪像素，C07／2133为4父框但2ROI／2像素，不能称为完全无干预。

主要结果如下，单位均为32帧**平均IoU差**，不是全视频AUC。前两列按标注比较头／起点，最后一列固定同种子P头只换起点；EE／EP／PE／PP的第一个字母为中心来源：

| 案例／种子 | 自身差P(PP)−E(EE) | 共同EE换头P−E | 关键父框分量对比 |
|---|---:|---:|---|
| C01 0226:338／2133 | −0.8930 | +0.0026 | PE−PP +0.8917；EP−PP +0.8838，联合PP坏而两种单分量修复都较好 |
| C02 0176:238／2133 | +0.0653 | +0.0162 | EP−EE +0.0431，缩小搜索有益；2132同一对比却−0.0844 |
| C03 0176:315／2132 | −0.7350 | +0.0115 | PE−PP +0.7514；EP−PP +0.7280，PP覆盖0.8479、其他1 |
| C04 0304:555／2133 | 0 | 0 | PE−PP +0.8177，末帧+0.8498；当帧只有PE约0.8078，其他三组0 |
| C05 0304:1086／2132 | +0.0106 | −0.0001 | 四组当帧均≤0.1902，当前混合无大幅后继修复；2133四组coverage=0 |
| C06 0307:511／2133 | −0.5868 | +0.0058 | EE−PP +0.5926；只恢复E中心+0.0058，只恢复E尺寸0，需两E分量一起恢复 |
| C07 0204:664／2132 | +0.7885 | 0 | EP−EE +0.7870、PP−PE +0.7885；搜索边长196→205，两中心coverage都1 |

C01／C03的中心×尺寸均值交互约−0.8837／−0.7378；C04约−0.8161。C06反为+0.5835：任一P分量已足以破坏较好EE，不是同一种交互模式。C07的交互仅+0.0010，尺寸效应在两种中心下均明显。两条分解路径都是条件差，不按路径给尺寸或中心分摊唯一原因百分比。

其余种子也必须保留：C01／2132当帧自身−0.5480，但32帧仅−0.0294；C03／2133均值−0.0115；C04／2132均值+0.0079；C06／2132当帧−0.2010但均值+0.0026；C05／2133与C07／2133当帧所有头IoU=0，窗口各比较差=0，不能据此补造窗口绝对均值为0或推断完全没干预。C07／2133的来源892恢复在本窗口之外。

这轮把问题具体化为**中心与搜索范围的条件兼容性**：混合两个失败路径的分量仍可形成较好起点；较小中心误差、coverage=1、更大目标像素都不是充分条件。共同EE换头的窗口均值范围−0.0083～+0.0162，部分自身差却很大，支持重视输入条件，但不说明所有头作用均小。尺寸交换改变搜索范围／分辨率等，不能等同输出尺寸回归。0176较小搜索有时好、有时坏，0204较大搜索好，不默认统一放缩或平滑。

Stage-59没有建立新方法采用、独立外部泛化或全视频提升。单个在线系统的因果候选及选择能力随后已由Stage-60检验，见第0.10节。不要按GT在线挑框，不重跑Stage-54～60，也不重复索取已齐材料。Stage-59详细point_details／rollout_summary／branch_contrasts／timelines仍留服务器，仅在设计确需时利用。

Stage-59 plan为`f572d132a9d39091d3dc251855f54a6b112eb8f3c65e633a61a9665ba701c647`，run为`771caa5ef9fffadbd8d0ecf6e75d89f906be1630a32fccba01cd2731a43224fc`，diagnostic为`94d7c73c7030bfe73d7fb8b21a5d2e116b3620484d8be5002a3908673405c6a4`；附件SHA及完整细节见结果复核。助手没有commit／push或登录服务器。

### 0.10 最新结果：Stage-60自身历史候选与后继收益

[完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md)，[预定协议与历史命令](docs/experiments/06_sgla_uav_tracking/STAGE60_CAUSAL_SEARCH_STATE_VALUE.md)。用户已报告`REVIEW_CAUSAL_SEARCH_STATE_PILOT`，五份材料已齐。输入完整Stage-59，输出`output/stage60/causal_search_state_value_v1/`。沿用来源E完整末帧保持头，SGLA全部参数／BN和首帧模板固定，仅拟合8个小岭回归scorer。

最近至多5个已接受框，组成当前中心／t−2滞后中心×当前尺寸／log中位历史尺寸四候选。current整个框逐位复制。response按Hann峰值选择；immediate_value学当帧相对收益；future_value学8帧均值相对收益。两种学习器共享24维GT-free响应／几何／历史交互特征、候选、拟合样本、RMS缩放、λ=0.01及无截距模型。current收益恒0，需高1e-6才改选；不搜索窗口、系数或threshold。四候选都走原生完整网络，只把选中原始输出接入自身历史，不平滑输出，不从其他OOF轨迹取在线父框。

每折／种子的E头在本折fit六视频新生完整轨迹，每视频24个均匀时刻、四个初始候选各推进8帧，之后普通自身推进；全部预测完成才附加fit GT。每个current短分支核对来源。两目标仅改离线标签；held-out数据不进入scorer拟合或RMS，未来图像／GT不进入在线选择。单次干预后普通推进的8帧标签不是反复选择的最优价值，需完整OOF检验。

固定正式预算按原13,525帧为：24条fit轨迹27,050次完整搜索，576个fit时刻共18,432次分支搜索，四策略×12视频×2种子96条OOF共351,650次搜索，合计397,132次；block 2,779,924次，无额外仅头。current每帧1次，其余三策略每帧4次，不声称加速。smoke的首个fit视频／前2窗口缓存复用，另4策略×32帧有416次额外搜索，总计397,548；模板初始化不计入搜索口径。

主要对比future−immediate，必要控制future−response／current；三个AUC对比用家族区间。积极开发要求主均值及对current均值均≥+0.5、三个家族下界>0、两个种子主要收益均正；不新损伤current稳定视频，低IoU不升，可靠帧差下界≥−1.0，current原获益视频仍全获益，至少4视频有实际预测变化。只标记PROMISING／REVIEW开发pilot，不等于独立方法通过。

本机原先4项真实CPU测试通过（10.333秒、无跳过），包含真实CENTER／原生解码、float32 Hann近并列峰、GT隔离、Stage-59→60合成流程和篡改拒绝。服务器现已完成，本机重算JSON指纹、四表CRLF SHA、30条配对均值及主要对照均相符；current来源框误差0，最大解码0.00048828125。调用397,132＋额外smoke416符合固定预算；support通过。未独立重验服务器全模型／原始缓存，本次不重跑CPU或GPU。

| 策略 | 宏AUC | 低IoU | 可靠帧AUC | 相对current稳定视频新增失效 |
|---|---:|---:|---:|---:|
| native | 63.4512 | 14.4573% | 75.5602 | — |
| current | 64.0079 | 14.7953% | 72.0353 | 0 |
| response | 67.7506 | 10.7647% | 73.3873 | 0 |
| immediate | 60.5784 | 21.0583% | 67.5821 | 1 |
| future | 64.1921 | 15.0800% | 69.7859 | 1 |

主要future−immediate +3.6137，CI[−2.9970,+10.6309]、9胜3负、最小留一+1.4009；种子+0.0536／+7.1738，大部分均值来自2133。对current仅+0.1842，CI[−8.1315,+7.9157]、6胜6负，可靠帧−2.2494、低IoU+0.2847个百分点，保护未过。对response −3.5585，名义CI[−9.5467,−0.2118]、家族CI[−12.0978,−0.1294]、1胜11负；去掉最差负视频后仍约−0.6866。不能把优于一个退化immediate控制写成优于更强简单控制。

response对current +3.7427，CI[−1.0196,+9.5075]、9胜3负、最小留一+1.7356；低IoU−4.0306个百分点，未引入额外current稳定失效。对固定current失败帧队列（8视频）+18.2745，名义CI[+0.6839,+38.5658]、6胜2负。积极信号值得保留，但全视频区间仍跨零、中位差仅+0.1265、最大视频差约+25.8206；不能追认PASS。对native描述性宏差+4.2994，直接配对区间未提供；可靠帧仍−2.1730，仍有1条原生稳定视频失效。

24条自身轨迹的事件由均值还原：current 23失效／14恢复／9末尾未恢复；response 13／6／7；immediate 16／3／13；future 13／2／11。future事件减少却恢复更少、未恢复更多，不是恢复机制已增强。原生可靠帧与current失败帧队列不是互斥分区，不能加和解释全视频贡献。

fit为576个唯一状态／1,728个非current动作样本，两个标签共享；平均3.2587份不同裁剪，314／576有正8帧选项，事后最佳均值+0.007263（0～1 IoU，非AUC上限）。future非零目标257～353／432，fit MSE甚至可下降约85.1%，仍不代表OOF选择可靠。在线future宏预测改变68.95%、取整裁剪改变63.76%，所有12视频有处理；不是未学习或没有干预。fold0拟合0003／0204／0226／0237／0307／0342，部署另6；fold1相反，不能把同名fit行当其自身OOF训练覆盖。

仪表化中位pipeline current6.39ms、response22.86ms、immediate23.02ms、future23.15ms；response约3.58倍，实际搜索4倍。不含读图／哈希，不能当部署FPS。immediate P95=353.65ms显著异常，其他选择组约26.6ms；仅凭汇总不能指定是算法固有开销或运行负载。

response−current还同时增加了搜索预算和多裁剪选择，未隔离历史构造优于同预算普通多裁剪。后续若验证该分支方法价值，需对应计算控制或验证减少探测次数仍保留收益，不能把全部正差归功于历史先验。

已有五份材料足够主判读，不重复索取。逐视频／种子／retention／focus／events及四候选缓存仍留服务器；后续需要定位response负收益或future损伤时优先离线利用。可以同状态对比缓存即时排序，不能把反复选择的实际后继冒充全部候选的普通8帧回报。Stage-60复核时下一轮尚未设计，现已由第0.12节落实Stage-61；Stage-1～60代码与结果冻结，不搜索λ／阈值／窗口／种子或重跑本轮。

解释字段时注意：Stage-60的`worst_video_delta`统一存原始delta最小值，只有AUC等正向指标可称最差；低IoU／最长失效的该字段不是最差损伤。均值／CI／REVIEW无此问题。原封存Python和CSV保留，后续新阶段按有利方向重算最差，勿沿用错误命名。

Stage-60 plan为`5dbcda645746c948a3bcb30b5e0879e67eba8f399fbbbcb14e924f76dd28c873`，run为`f7d13fbe13ddec2b3739eb07e12f4c48d871d34ded2f93d20842688618104cc4`，decision为`f127ade8e2b00691c815de4dc06b7654a7dd12b94354849667119850f754ed15`；合并TXT SHA及细节见结果复核。无独立外部验证或部署收益结论，助手无commit／push或服务器登录。


### 0.11 2026-11-01：SGLATrackV2源码、日志与下一轮启发

**本节是已核对的相关方法事实、研究假设和建议顺序，不是已冻结的Stage-61协议，也不表示用户已要求切换V2或启动训练。** 用户要求将这些启发写入交接，便于下次实验接续。SGLA无人机单目标跟踪目标不变；已有相似方法不构成否定方向的理由，需检验新增价值。

**版本与来源。** 官方[V2目录](https://github.com/GXNU-ZhongLab/SGLATrack/tree/main/SGLATrackV2)的README称其为SGLATrack扩展版，阅读时仍写TPAMI在审，没有可直接核对的V1／V2成绩对照表。本轮比较公开源码，未加载V2权重或运行V2推理。2026-11-01查询GitHub API时，上游main提交为`6eb0f977d3a5f6c080b8b2ec89c3e97705384afe`，提交时间2026-08-14T07:44:29Z；下列链接固定该提交方便追溯。逐文件SHA下载核验遇超时，未完成；不要描述为已完整归档／逐字节验证V2来源，下次实际接入前应自行固定代码与权重SHA。

- [V2 README](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/README.md)。
- [活动CAE骨干与历史token生成](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/lib/models/sglatrackv2/vit_cae_async.py)。
- [模型训练／推理入口](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/lib/models/sglatrackv2/sglatrackv2.py)。
- [在线跟踪器及记忆更新](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/lib/test/tracker/sglatrackv2.py)。
- [多搜索帧训练Actor](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/lib/train/actors/sglatrackv2.py)。
- [公开配置baseline_WOCE_cae.yaml](https://github.com/GXNU-ZhongLab/SGLATrack/blob/6eb0f977d3a5f6c080b8b2ec89c3e97705384afe/SGLATrackV2/experiments/sglatrackv2/baseline_WOCE_cae.yaml)。

**已经确认的代码变化（针对当前V1 DeiT与上述V2 CAE配置，不能推广为所有变体）。**

| 模块 | 当前V1 | V2公开实现 | 接续解释 |
|---|---|---|---|
| 模板与交互 | 每帧联合编码固定首帧模板与搜索图 | 初始化`mode='z'`编码并缓存`self.z`；搜索queries读取模板／历史／搜索信息 | 缓存配套改变交互结构，不能在V1直接缓存后宣称输出不变 |
| 历史状态 | 上一预测框决定裁剪，无原生特征记忆 | 将`hlist`作为前缀加入搜索；选中候选块按搜索对模板／历史前缀注意力排序，提取6个搜索特征token | 当前帧信息影响后续特征；高注意力不等于GT确认可靠 |
| 记忆更新 | 原模板固定 | 活动代码为`self.hlist = aux_dict['hlist']` | 每帧替换紧凑特征记忆，并非注释中的每5帧累积多帧库 |
| 选层 | 6公共前缀＋L7～L12中1个独立块 | `start_layer=3`，4公共前缀＋L5～L12中1个独立块 | 自适应仍主要选择候选身份，不能说V2串行执行12层或由难度任意改变深度 |
| 骨干与输入 | DeiT-Tiny、192维，搜索256、模板128，搜索倍率4 | CAEv2-Small、384维，输入尺寸不变，搜索倍率5，预训练`caev2_small_300ep.pth` | 不只改变记忆；块数减少不等于已测出速度提升，V1／V2总差有多项混杂 |
| 训练 | 一模板、一搜索帧 | 一模板、四搜索帧，顺序处理并传递历史；逐帧GIoU／L1／focal／相似度选层损失相加 | 四搜索帧是训练输入数量，不是同一当前图像四次候选探测，也不证明自身预测裁剪的闭环稳定性已解决 |

推理中新增模板图像采集／模板库更新代码被注释，配置中`TEMPLATE_NUMBER:3`不能据此认定动态模板更新已启用；`CE_LOC:[]`／`CE_KEEP_RATIO:[]`不应被说成当前配置已使用token剔除。选层仍用候选块输入／输出余弦相似度生成标签，并非未来IoU／强化学习选层。V2在线仍从上一预测框生成一个搜索裁剪，没有本项目的中心×尺寸四候选选择。

**与当前实验、方法框架的关系。** 我们目前用V1 `sglatrack_ep0297.pth.tar`作原始网络基础；native为未适配原头，Stage-60 current／response／两种value策略在同折同种子共享E几何适配头。E对应Stage-54 rollout_preserve／Stage-58 endpoint_control，只曾训练size／offset，Stage-60网络完全固定，两种value仅额外拟合8个评分器。尚未用到V2的CAE或hlist。原生63.4512是VisDrone2019-SOT train来源的12个开发视频宏AUC，不是官方全测试集成绩，也不是V2成绩。裁剪本身属于原生机制；当前新增价值假设是因果历史中心—尺寸组合的搜索输入选择，而非“首次引入裁剪”。[初始中英文方法图及图注](docs/figures/sgla_search_state_v1/README.md)明确画的是V1／E，尚未接入V2记忆。

V2提供三条可发展的启发：

1. **迁移对照：** 在已有时序特征记忆的模型上，搜索父框是否仍造成持续失跟？若历史几何选择仍有效，支持其价值超出单个E头；若收益变小，可能表明特征记忆缓解部分问题，但需匹配证据。须先做同一模型内部比较，不能将V1／V2总差唯一归因于记忆。
2. **搜索状态与记忆的一致性：** 可检验“不合适裁剪→关注干扰物→生成错误记忆→后续定位与裁剪继续受影响”的双反馈假设。未来选择准则可同时关注即时定位与后续记忆质量，而不只取当前最大响应。该错误链及相应收益目前均未验证，不写成既有发现或已实现新模块。
3. **成本设计：** V2的模板缓存有减少重复模板处理的设计意义，但不能直接移植到V1联合自注意力而假定等价。不同裁剪的搜索特征仍需分别算；每帧候选次数、模板初始化、额外memory读写和实际时延分别记账。

**若接入V2，候选探测的状态隔离是必要工程要求。** 每个候选都从同一份`t−1`已接受状态开始，包含预测框、框历史、hlist与其他可变跟踪状态；首帧缓存模板保持一致。为各分支克隆会被原地修改的张量，不仅复制Python引用。四次探测不得串行覆盖共享hlist；仅在确定选择后一次性提交“选中原始框＋该分支生成的记忆”，其他候选均丢弃。逻辑frame_id每个真实图像推进一次，不能把四次probe计成四帧。先核对单路径V2原生控制重放，再检查候选调用顺序不改变各自结果、未选分支不污染状态、未来GT不改变动作，保留实际调用／状态审计。

```text
同一已接受状态：b(t−1)、H(t−1)、h(t−1)、固定模板缓存
  ├─ q0 → 原始输出b0、分支记忆h0
  ├─ q1 → 原始输出b1、分支记忆h1
  ├─ q2 → 原始输出b2、分支记忆h2
  └─ q3 → 原始输出b3、分支记忆h3
选择k后仅提交bk、hk；未选输出和特征不进入已接受历史。
```

未来若做匹配起点后继诊断，必须连同特征记忆一起绑定起点；只交换父框而混用另一条轨迹的hlist不再是同状态比较。应明确处理对象是“几何”“记忆”还是二者联合，并匹配相应对照；不默认联合改动就能唯一归因。GT仍仅限初始化及fit离线监督／评价。

**建议的下一轮顺序（尚未立项／实现）。** 先利用Stage-60现有详细表／四候选缓存定位响应负例；优先补“原始V1权重＋历史候选响应选择”，明确当前收益是否依赖E适配头，再增加同预算普通多裁剪控制，分离历史构造与4次搜索的作用。固定配方后在未参与设计的源组评价；既有12组继续标记开发证据。随后才将V2作为相关方法／迁移对照，比较V2自身基线、V2历史候选和同预算普通多裁剪；必要时再针对记忆一致性提出新处理。当前不因追求创新立即全网重训、叠加记忆模块或搜索窗口／阈值／种子。推理策略可以固定网络形成方法；若学习评分，先训练和验证评分器本身的独立价值。若普通同预算多裁剪已取得相同效果，应如实调整历史构造的贡献定位，而非只改命名。

**用户提供的V2训练日志已读取，无须重复索取。** 原件在本机仓库外`E:\Program Files\LETrack\SGLATrackV2\logs\odtrack-baseline_WOCE_cae.log`，4,466,676字节，SHA256=`b5da971fcc4f36525ba53e358d8ce493433538697ed5deedd8376a0ef98e6652`，未复制进Git／未修改。6000条均为train，epoch1～300完整，每轮234 batch，50／100／150／200／234打印，同坐标4条不同记录，兼容4个分布式进程但不能单凭日志确认GPU数或认作4次独立实验。已记录指标全部有限，无报错文本；没有验证／测试AUC、LR、配置展开、预训练加载或checkpoint保存记录，不能证明泛化、最佳轮次、最终权重已保存或模型具体身份。FPS是训练吞吐，不是部署推理速度。

轮末四条等权后再对四个搜索位置平均：epoch1 loss6.85437／IoU0.26871，10为1.16119／0.72551，50为0.86831／0.79077，100为0.79195／0.80802，240为0.71022／0.82751，300为0.66797／0.83762。epoch297 IoU0.83737，与300接近，但不能据训练IoU选择测试最优checkpoint。epoch300四位置IoU分别0.850958／0.838595／0.833533／0.827403，最后比第一低0.023555；这是待解释的位置差，不能单独证明记忆污染或真实闭环误差累积。全日志每帧loss符合`2*giou+5*l1+location+0.2*cos`，舍入最大残差0.000042。cos是选层辅助损失，不是余弦值或选层准确率；日志也不能隔离历史记忆相对无记忆的贡献。若下次环境没有此外部文件，先利用上述已记录分析，不默认日志丢失意味着必须重跑训练。

本次只修改交接文档；Stage-1～60实验代码／封存结果不改，V2没有安装、训练、推理或与现有source链混用；未commit／push，也未登录服务器。下一次明确要求“进行下一轮实验”时，以本节及0.10设计具体有限对照，不把研究建议误写为已经完成的新阶段。


### 0.12 2026-11-02：Stage-61已准备，等待服务器推理

用户本次明确要求先读交接再进行下一轮。已按0.11的建议实现[Stage-61固定协议](docs/experiments/06_sgla_uav_tracking/STAGE61_NATIVE_HISTORY_MATCHED_RECROP.md)，没有切换V2或新增训练。**当前只有本机CPU验证，没有Stage-61真实AUC／GPU结果。**

本轮明确两个问题：历史候选是否在**原始未适配V1**上也有效；同样四次调用下，历史是否超过只用**当帧初次解码框**构造的重裁剪。网络来源为原始V1／已有E，政策为current／history／recrop，六个策略分别报告。原始V1每视频只计一个确定性运行，不伪造两个种子；E沿用原折／2132和2133，先在视频内平均。

历史组原样使用Stage-60的当前／滞后中心×当前／近5框log中位尺寸。重裁剪令B为自身上一接受框，先用B预测当帧D；四候选是B中心×B尺寸、B中心×D尺寸、D中心×B尺寸、D本身。**q0直接复用已执行的这次调用，仅另做3次，所以每帧4次而非5次。** 不增加扰动幅度、尺度超参或触发规则；统一Hann峰值及原1e−6并列规则，只提交选中原始输出。重裁剪候选／评分不依赖更早历史；旧24维日志保留但历史特征不参与控制。

新增native history 12条、native recrop 12条、E recrop 24条，合计**48条完整轨迹、216,400次搜索／1,514,800次block**。复用封存native current 12条、Stage-60 E current／response各24条，合计60条旧轨迹，旧推理不重跑。第一视频前32帧的native及两E种子×三政策smoke有864次额外搜索，校验native/E current及E history来源前缀精确复现；正式＋smoke共217,264次。模板初始化不计入搜索次数。匹配的是每帧网络调用，不把历史时延和本轮时延冒充同时重测。

主要比较native history−native recrop，必要比较native history−native current；两项AUC给Bonferroni家族区间。均值均≥+0.5、家族下界均>0，并通过原稳定失效、低IoU与可靠帧容差保护及有效曝光，才标PROMISING开发pilot；否则REVIEW。E内对比和基线×候选来源交互为次要描述。这个当帧重裁剪控制不穷尽所有普通多裁剪，不以一个控制替代外部确认。

新增5个Python文件：`stage61_search_common.py`、`stage61_search_runtime.py`、`stage61_search_experiment.py`、`stage61_search_summary.py`及`test_stage61_search.py`。prepare只读取验证完整Stage-60及上游源，不写旧报告。新输出`output/stage61/v1_history_matched_recrop_v1/`，入口`prepare → smoke → evaluate → summary`，无collect／fit／train阶段。分项NPZ／receipt支持只读恢复及篡改拒绝。

本机最终**3项CPU测试通过，16.958秒，无跳过**，5文件AST及两个NumPy-only CLI help通过。测试覆盖真实小通道CENTER／原公共解码、真实调用hook、修改全部未来GT不改变预测／候选／特征／动作、候选调用顺序、只接受选中状态；真实运行合成Stage-59→60→61协议流程（上游Stage-58边界模拟，骨干／采样为fixture），验证旧Stage-60所有文件字节不变、smoke来源前缀误差0、缓存恢复不重复前向和源／缓存／报告篡改拒绝。来源评分器只验封存包，不重新拟合。维护检查PASS（505保留／200归档／106模块／0缓存）、本地链接与diff检查通过；Bash只读`-n`语法检查通过，Windows沙箱阻止MSYS初始化后只在沙箱外做了该语法检查，没有执行实验。这不代表真实服务器效果或重新训练全部来源链。

服务器同步代码并进入`sglatrack-test`后：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
bash tools/run_stage61_search.sh
```

脚本默认CHECKPOINT为原V1权重、STAGE60_ROOT为`$PWD/output/stage60/causal_search_state_value_v1`、STAGE61_ROOT为`$PWD/output/stage61/v1_history_matched_recrop_v1`，GPU0；可用`STAGE61_GPU`指定设备，但原runtime/GPU身份必须仍符合封存源。单独命令详见协议。`.gitattributes`只对该新脚本固定LF，避免Windows换行造成Bash错误。

结束后使用cat合并JSON及8张CSV、另附2份启动身份记录到`docs/experiments/06_sgla_uav_tracking/results/stage61_results_<UTC>_<pid>.txt`；9份分析材料含逐视频、种子和事件，便于直接读完整新结果。不自动commit／push。来源模型、原始NPZ和所有上游产物继续保留在服务器。

新增汇总按指标有利方向修正最差视频值，**不修改旧Stage-60字段**。只为新轨迹报告新pipeline计时；原生确定性基线不复制成两种子。当前尚无Stage-61真实决策或表现，下一次若用户上传结果先读它们，不重复跑本轮或替换基线。

**同日启动故障与修复：** 用户首次服务器启动在`stage46_motion_experiment.make_runtime`比较`rt.identity != source_run["runtime"]`时退出，未进入图像推理；堆栈没有差异字段，实际根因尚待服务器启动诊断，不能认定一定是错误权重或YAML漂移。为了保留已prepare的manifest及封存代码，未改Stage-1～61实验Python或任何旧JSON；新增`tools/stage61_runtime_replay.py`和4项启动回归。新入口从Stage-61 manifest的`runtime.config`复制EasyDict／TrackerParams，临时在内存替换原参数加载，仍调用原工厂；权重SHA、完整配置、Torch／CUDA、设备及UUID、网格、OpenCV／NumPy、原模型摘要必须精确匹配。若当前YAML／全局配置漂移，会记录差异并按封存快照构造；真正的权重／环境差异仍拒绝。4项回归通过，使用真实旧工厂，但CUDA／Runtime构造由fixture替代，不能作为GPU验证。

一键脚本的smoke／evaluate现改用上述独立入口，prepare／summary不变。原Stage-61目录可继续使用，不删manifest或旧来源，也不修改校验条件。启动记录写入`diagnostics/stage61_runtime_smoke.json`及`stage61_runtime_evaluate.json`，带launcher SHA与计划指纹，成功为`EXACT_SOURCE_RUNTIME_VERIFIED`，不一致为`RUNTIME_MISMATCH`且逐字段列expected／actual；模型构造失败为`RUNTIME_CONSTRUCTION_FAILED`并保留原异常。如仍失败，优先`cat "$STAGE61_ROOT/diagnostics/stage61_runtime_smoke.json"`读取新证据，再按实际字段定位，不盲目换权重、改GPU或重跑旧实验。修复后本机4项启动测试及3项原Stage-61测试全部通过，后者20.689秒、无跳过；新入口help、Bash只读语法检查、维护检查通过。尚无服务器smoke／AUC结果。

**后续已定位设备差异：** 用户新堆栈明确`tracker.device expected='NVIDIA L20' actual='NVIDIA L40'`；当前退出原因已知是所选GPU型号不符，不再仅写“根因未知”，也不能认定已证实YAML漂移或错误权重。补充的`--list-gpus`只读列出CUDA逻辑编号／型号／UUID，`--gpu auto`仅自动选中源型号＋UUID唯一匹配的可见原卡；7项启动回归通过，包括实际CLI参数传递及重映射编号。**用户随后明确回答目前只能使用L40，因此当前应使用第0.13节的新入口**，而非继续尝试原卡自动选择。服务器仍未进入本轮图像推理。

### 0.13 2026-11-02：仅有L40，新增全部六组同卡重测

新固定协议为[STAGE61_L40_DEVICE_MATCHED_RUN.md](docs/experiments/06_sgla_uav_tracking/STAGE61_L40_DEVICE_MATCHED_RUN.md)。用户确认仅有L40后，已新增4个Python文件与1个Bash脚本到`tools/`，不改Stage-1～61封存实验Python，不编辑原L20 manifest或旧报告。原Stage-61目录继续作为冻结计划及完整Stage-60来源链入口；其smoke没有通过也可以读取已封存计划。

新目录`output/stage61/v1_history_matched_recrop_l40_v1/`，schema为`stage61_l40_device_matched_recrop_v1`。新manifest同时记录原L20来源与实际L40运行身份，只声明GPU型号／UUID变化，原checkpoint／配置／模型状态／软件仍严格校验。新运行的smoke、evaluate和resume绑定新设备身份；当前只支持此次声明的L40，同型号旧卡自动选择不是迁移校验的替代。

在原始V1和固定E上，current／history／recrop全部重新推理。原生每视频只一个确定性运行，E保留两个原种子和原折；不重训、不选checkpoint、不改候选或评分规则。所有主要指标、可靠帧、失败帧和原稳定视频都从**本轮L40原生current**定义；L20旧轨迹只进入`stage61_l40_device_bridge.csv`作描述，不参与主比较或方法采用门槛。

按既有13525跟踪帧：原生36条121725次搜索，E72条243450次搜索，合计**108条新完整轨迹／365175次正式搜索／2556225次block**，复用完整评价轨迹数0。首视频前32帧的三网络×三政策864次smoke，另三条独立公共current路径共96次，smoke总960；正式＋smoke366135次搜索。模板初始化单列、不算搜索，无新模型或仅头额外前向。

smoke要求原生及两个E种子的current actor与独立原生公共`NativeObserver.step`路径在本次L40上框及层精确一致；跨L20／L40的前缀差只记录为描述性诊断。原四次调用／因果历史／裁剪与解码审计沿用原actor。新NPZ／receipt绑定计划与实际runtime指纹，旧L20 receipt不能填补新缓存。已完成项只读恢复，篡改拒绝。主要及必要比较、家族区间、保护条件沿用原Stage-61，不因迁移追改阈值。

服务器同步本次新增工具后，在`sglatrack-test`中执行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE61_ROOT="$PWD/output/stage61/v1_history_matched_recrop_v1"
export STAGE61_L40_ROOT="$PWD/output/stage61/v1_history_matched_recrop_l40_v1"
bash tools/run_stage61_l40.sh
```

默认CUDA逻辑GPU0，可设置`STAGE61_L40_GPU`。顺序prepare／smoke／evaluate／summary；prepare只读设备与源，summary无需GPU。一键脚本用cat合并decision＋9张CSV（含device_bridge）及2份启动记录到仓库results的`stage61_l40_results_<UTC>_<pid>.txt`，不自动commit／push。仍需保留全部原来源链与新完整产物，Git报告不是缓存备份。

本机L40新增**3项测试通过，23.837秒，无跳过**，另有7项启动回归通过。覆盖仅GPU身份变化及软件保持、旧L20 receipt拒绝、真实小通道CENTER与原公共解码的完整Stage-59→60→61来源及新六组流程、独立public smoke误差0、预算、原文件逐字节不变、恢复不重复前向、源／缓存／报告篡改拒绝。主动污染旧L20数组后，device_bridge改变，L40主比较／必要比较／交互／保护不变。上游Stage-58边界、骨干、图像和设备为fixture，不等于真实GPU验证。新CLI help／AST、Bash只读语法、LF、维护检查通过。

上述是准备时的本机验证记录；服务器L40现已完成，真实结果见第0.14节。CPU合成测试不代替服务器结果，方法REVIEW也不等于新外部验证或部署加速。

### 0.14 2026-11-02：Stage-61 L40结果已齐，原生恢复能力保护未过

用户上传`D:\Users\23920\Desktop\stage61_l40_results_20261102T143543Z_40176.txt`，已用cat读取并原样归档到仓库`docs/experiments/06_sgla_uav_tracking/results/`。完整复核见[STAGE61_L40_RESULTS_REVIEW.md](docs/experiments/06_sgla_uav_tracking/STAGE61_L40_RESULTS_REVIEW.md)。完整12份材料为decision、9张CSV及smoke／evaluate两份启动记录，**无需再次索取或重跑本轮**。原件169724字节／1278行，SHA`5a46ec176287486124e0ec45f2a8791900c69d8bdc44e76d5db1de98be8854e7`；新增原件专用`-text`属性防Git换行转换。

状态`REVIEW_L40_NATIVE_HISTORY_RECROP_PILOT`，pilot_pass及retention_guard为false，support_pass为true。宏AUC／低IoU：native current63.4512／14.4573%、history65.2483／13.0572%、recrop58.7334／20.7305%；E current64.0079／14.7953%、history67.7506／10.7647%、recrop61.0880／20.2354%。原生current有5条稳定视频，history无新增稳定视频失效，recrop在0304新增1条；E各组仍有1条原生稳定视频在任一种子上失效。

主要native history−recrop为+6.5150，名义CI[−7.6692,+20.9566]、家族CI[−9.7870,+23.0008]、9胜3负、中位+0.0970、最小留一+2.3973；必要history−current为+1.7972，名义CI[−10.8179,+14.4368]、家族CI[−12.3696,+15.9896]、6胜6负、中位−0.0046、最小留一−2.7343。均值目标虽达，家族证据未过。

保护失败具体来自可靠帧：native history−current为−5.5054，名义CI[−16.2543,−0.0763]，低于预定下界−1.0。0226单视频可靠帧差−64.4288，其余11条平均约−0.1488。原生current与history事件总数同为7，恢复同为4，末尾未恢复同为3，但事件身份不同，**不能只按总数或零新增稳定失效判断保护成功**。

E内次要history−current为+3.7427，CI[−1.0318,+9.6159]、9胜3负、最小留一+1.7356；history−recrop为+6.6626，名义CI[+0.0039,+17.4080]、8胜4负、最小留一+2.4589。后者下界仅略高于0，未对该次要比较做家族校正，且recrop自身比E current低2.9198；不能替代原生主比较或推广到所有普通多裁剪。E历史对current两个种子均值差为+1.2887／+6.1967；对recrop为+4.4883／+8.8368，收益量不均。来源×候选交互+0.1476，CI[−8.0736,+10.4374]，未建立依赖或独立于E的总体结论。

必须接续的病例：

- **原生0226负例：** current AUC55.0614，110失败后338恢复；history仅5.0819，115起直到1860末尾未恢复，1744低IoU帧。recrop53.5740且337恢复。E历史两种子约56～57均恢复。该视频原先已有暂时事件，所以不计新增稳定失效，却是主要保护损伤。
- **原生0176／0204正例：** history相对current分别+51.6436／+18.5358；0176避免162起的长期丢失，history到407才首个短事件、412恢复，另458／467短事件；0204完全没有664失效。原生history在0304保持约72.39，recrop却555起未恢复，AUC22.1587。
- **E0307负例：** history相对current视频均值−9.9661，两个种子约41.48／41.24，392／395起均末尾未恢复；丢失current2132的471／732和2133的511恢复。不能因首次失效略延后便称恢复能力改善。
- **E0304种子分岔：** history2133从current22.4624升至73.6315且无失效，2132只从44.3506到44.8226且1093起未恢复。视频均值+25.8206不代表双种子均解决。E0176两个历史种子均约69.09；0204历史和recrop两种子都约75.3～75.6且无失效，后者不是历史独占优势的正例。

本机重算decision指纹、9表字节SHA、全部31条配对及bootstrap／家族区间、种子→视频→policy聚合、scope预算均相符，最大配对数值误差0。表行数policy6、paired31、influence12、scope108、proposal108、video72、seed_video108、events72、bridge60。108条新轨迹／365175head／2556225block，另960smoke；原公共current前缀框差0、最大decode约0.00048828125。两份启动均为EXACT_SOURCE_RUNTIME_VERIFIED，当前YAML无差异且未恢复配置；此前GPU型号不符是已证实问题，不能写成YAML漂移已证实。60条完整桥接的最大框差、改变帧数、AUC差均0；只证明本次已比较输出复现，不推广为跨硬件普适保证。服务器原始NPZ／模型未在本机独立读取验证。

native历史／recrop实际改变比例约57.79%／76.87%，E约57.00%／76.59%，有真实非平凡使用；更高干预比例不是因果病因。仪表化pipeline中位原生current6.11ms、history23.09ms、E current6.07ms／history23.14ms，多候选约3.78～3.83倍耗时及4倍搜索。native_history P95异常为362.89ms，其余多候选约26～28ms，未有系统负载证据，不能指定算法固有原因或宣称部署实时加速。

**当时的下一步建议现已由第0.15节落实为Stage-62：** 保留因果历史中心—尺寸候选分支，以既有evaluation NPZ为主，诊断原生0226、E0307，配原生0176／0204与E0304两种子。先在各自已冻结的同一状态下核对q0与四候选质量、排序、选中动作及恢复分岔，区分候选不足与选择错误；跨裁剪峰值可靠性和恢复机会是待检验假设，不能从当前表认定唯一病因。若需要共同起点的未来因果证据，再单独设计后继重放，不从自身持续选择轨迹假造全部候选未来回报。不调已看过结果的窗口／阈值／种子，不编辑Stage-61或L40工具／原件／既有报告，不默认训练新头或切换V2。

### 0.15 2026-11-03：Stage-62结果前固定方案与准备记录

**以下保留准备时记录；服务器现已完成，五份结果及复核见第0.16节，无需重跑。**

用户要求先读交接再继续实验，已按第0.14节落实[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE62_CACHED_CANDIDATE_SELECTION_AUDIT.md)。新增`tools/stage62_candidate_common.py`、`stage62_candidate_audit.py`、`test_stage62_candidate_audit.py`和`run_stage62_candidate_audit.sh`，不改Stage-1～61／L40封存代码。

输入完整`output/stage61/v1_history_matched_recrop_l40_v1/`，输出新目录`output/stage62/cached_candidate_selection_v1/`。来源验证读取完整上游计划／模型哈希／回执、Stage-61 L40的decision／9表／2启动记录及108条NPZ，重验父框、候选、实际选择及调用预算；不调用旧summary写回旧产物。**0新增网络调用、0图像读取、0训练／新模型，无需CHECKPOINT或GPU参数**。GPU设备变更不影响只读诊断，来源身份仍绑定原L40记录。

全量108条来源包含121725个状态观察及365175个已付费搜索的已解码输出，重复前缀／相同候选不另作独立样本。计算当前帧所选／q0／GT最好已解码IoU、候选可用性（≥0.5）、漏选、q0良好却损失／变为低IoU（<0.2）、救援机会、即时选择遗憾、GT可用裁剪覆盖及几何。**缓存无完整响应网格**，候选不足只表示已解码框不足；不能推成dense候选缺失。GT仅用于冻结输出的离线标签，不产生新控制器，不拼oracle轨迹或假造未来收益。

六类队列固定为all、native_reliable、native_low、pre_first_failure、first_failure_episode、first_recovery_window；前失败30帧、恢复起30帧，空队列和条件分母0保持未定义。E先在视频内对支持种子等权平均，再对支持视频等权平均，逐项保留支持数；队列可重叠，事件队列依赖自身结果，不是在线触发器或独立验证。

11个固定锚点：原生0226历史失败115／原生current恢复338；E0307历史两种子失败392／395及current恢复471／511；原生0176 current失败162、0204 current失败664；E0304种子2133 current失败554、2132 current失败1078及history失败1093。每个锚点展示current／history／recrop各自历史前后30帧：生产输入下33条focus、2013条timeline、6039条逐候选行，重叠不当作额外样本。锚点从源首次事件复算，不吻合即拒绝；跨政策同帧不是共同起点因果对比。

运行（同步新增代码后，在原环境即可）：

```bash
cd ~/workspace/paper
export STAGE61_L40_ROOT="$PWD/output/stage61/v1_history_matched_recrop_l40_v1"
export STAGE62_ROOT="$PWD/output/stage62/cached_candidate_selection_v1"
bash -n tools/run_stage62_candidate_audit.sh
bash tools/run_stage62_candidate_audit.sh
```

也可单独执行`python -B tools/stage62_candidate_audit.py --input "$STAGE61_L40_ROOT" --output_dir "$STAGE62_ROOT"`。一键脚本自动cat合并diagnostic JSON／policy_summary／scope／transitions／focus五份材料到仓库results的`stage62_results_<UTC>_<pid>.txt`；8张完整CSV中其余seed／video／timeline／candidates长表留服务器。不自动commit／push。

本机**4项测试通过，16.587秒、无跳过**。覆盖候选不足／漏选／q0损伤／救援分类、官方整数IoU、GT标签不改动作／候选、历史padding几何、空分母及视频等权聚合；真实小通道CENTER及原公共decode的Stage-59→60→61→L40来源链、读取／重复审计不改旧字节、源表／缓存／输出篡改拒绝。审计段禁用Torch／OpenCV导入并拦截图像及NN调用仍通过。上游Stage-58边界、骨干、图像及设备为fixture，不能替代真实服务器数据。新CLI help、Python AST和LF检查通过。

**Bash验证限制：** 沙箱内MSYS因`NtCreateDirectoryObject`权限错误不能初始化；请求沙箱外只读`bash -n`时，自动审批服务因用量限制未能完成审查，命令未执行，不是判定脚本不安全。未绕过审批或标记语法检查通过，需在服务器运行上述`bash -n`。其余任务继续完成，无需再次修改源实验或请求GPU权限。

准备时尚无Stage-62真实候选分类结论；现已按上述固定方案收到结果，见第0.16节。不得把COMPLETE诊断当作新方法PASS或默认启动训练／V2。上述本机Bash限制是历史验证记录，本次收到完成结果不等于本机语法检查曾执行通过。

### 0.16 2026-11-03：Stage-62完成，候选退化与部分漏选并存

用户上传`D:\Users\23920\Desktop\stage62_results_20261103T004524Z_148362.txt`，已用cat读取，逐字节归档到仓库results；完整[Stage-62复核](docs/experiments/06_sgla_uav_tracking/STAGE62_RESULTS_REVIEW.md)是下次接续入口。五份材料为diagnostic及policy_summary／scope／transitions／focus，4表36／108／108／33行。原件100888字节、552行，SHA`d93a89da4fe63d2a0cfb200326fffb7ecf95b46d1db843688af94bae58e9cd50`；诊断指纹`93885c0f2cd262ee91fa4b0c4f7c29c8b9a52ad322b7768f5406ebf011d5b0f3`，专用`-text`属性保护原件。

状态`COMPLETE_CACHED_CANDIDATE_SELECTION_DIAGNOSTIC`。全量108条来源、121725状态观察、365175已有解码输出；新增训练／网络调用／图像读取／完整轨迹均为0。GT只作离线标签，候选是每次裁剪的一个Hann解码框；**无完整响应网格，无反事实未来分支，无新方法采用门槛**。

全部状态宏统计：native history良好候选可用83.8993%、无条件漏选0.6293%、q0良好却选低0.0557%、平均即时遗憾0.01618；E history对应84.2875%、0.6974%、0.0146%、0.01654。native／E current可用80.6095%／78.2981%，recrop74.7630%／75.8767%，均为各自不同历史下的描述。条件漏选native history9.0672%、E history0.9460%为先条件比例后视频等权平均，不能用无条件宏比例相除重算，也不是帧池化错误率。空分母未当0；current零遗憾仅因一个候选。

**候选不足负例：** C01原生0226 history:115、C02／C03 E0307 history:392／395，±30帧三个61帧窗口分别0个良好候选；所选裁剪逐帧完整覆盖GT、GT中心保留。锚点所选／最好IoU为0.1907／0.2270、0.1730／0.1791、0.1762／0.2393；三个窗口current／recrop各自历史也无良好候选。只重排现成框或保留同状态q0无法在这些锚点选出≥0.5输出，不能推成dense网格没有目标、全部责任在几何或更早选择无关。原生0226首次漏选15，E0307首次漏选及q0丢失42／41，早于关注窗口，尚无共同起点未来因果证明。

**恢复丢失分两类：** C11原生0226 current:338 IoU0.7446恢复，recrop同帧0.8744；history四候选IoU与GT覆盖均0，197首次所有候选GT中心出裁剪，308～368窗口仅1帧任一裁剪保留中心。E0307历史则整条轨迹未出现所有候选GT中心出裁剪：C09 current／2132:471为0.6070，history最好0.2603；441～501窗口完整覆盖但仅1帧良好候选且漏选，具体帧／候选需已有长表，不能猜。C10 current／2133:511为0.5357，history最好0.1216；481～541窗口完整覆盖却0个良好候选。

**正例已有状态较好：** C04原生0176:162、C05原生0204:664、C06 E0304／2133:554，history自身q0已为0.6846／0.7254／0.7705，对应current为0.1722／0／0；不能全解释为锚点当帧选到替代框。C04窗口58帧良好、6次漏选、2次救援；C05／C06各61帧良好且无漏选。跨政策历史不同，不能计算纯当前候选贡献或宣称双种子均解决0304。

**不能放弃选择环节：** 首次失败事件队列native history条件漏选95%（支持4视频，事件队列共6视频）、E history100%（支持5，事件队列共7），表示部分事件仍有即时机会，未证实选它就会持续恢复。native_recrop／0304首次q0良好却选低恰在555失败起点；556首次全候选低，561首次所有GT中心出裁剪，是有良好q0却即时损伤的具体对照。其`before_failure`字段不含555本帧，为空不矛盾。E0304／2132 history:1093所选=q0=最好0.1876，说明事后回退q0也不能解决该状态；1063～1123窗口19帧有良好候选、2次漏选，之后覆盖逐渐丢失。

本机已重算JSON指纹及**4张已提供CSV**的字节SHA，核对L40来源身份、108条scope帧数／调用／头哈希／decode误差，transitions首次失败／事件／末尾未恢复与Stage-61逐种子表一致；11锚点、33行分类／窗口计数／差分自洽。2013时序／6039候选展开量与报告一致。没有服务器NPZ／模型／receipt及未上传的4张长表，**未独立重算种子→视频→policy聚合或全部服务器来源哈希**；不能把附件自洽核对写成完整原始缓存复验。

接下来具体方向是**在SGLA因果历史搜索中保护有效候选及恢复机会**，同时处理可见却输出差和更晚覆盖丢失。先按设计需要读取服务器`stage62_seed_summary.csv`、`stage62_video_summary.csv`、`stage62_timeline.csv`、`stage62_candidates.csv`，定位C09唯一机会及候选失效前的过渡；再预声明小规模共同父框后继／保留q0／中心尺寸构造对照，在线规则只用因果GT-free信息。GT最好可用于离线机制分支，不当在线方法。可以边诊断边试清楚的小原型，不要求先证明所有机制，也不因相似工作否定方向。

Stage-62复核时Stage-63尚未实现；现已按第0.17节方案完成，结果见0.18。上述Stage-62五份材料足够完成其复核，不重复索取；Stage-1～63代码及旧结果冻结，不代commit／push。

### 0.17 2026-11-03：Stage-63结果前固定方案与准备记录

**本节保留结果前安排；服务器现已完成，收到材料与结果复核见第0.18节，不重跑。**

用户要求继续实验，已落实[Stage-63固定协议](docs/experiments/06_sgla_uav_tracking/STAGE63_MATCHED_ONE_CHOICE_CONTINUATION.md)及5个新tools文件：`stage63_choice_common.py`、`stage63_choice_experiment.py`、`stage63_choice_summary.py`、`test_stage63_choice.py`、`run_stage63_choice.sh`。读取完整Stage-62封存计划／diagnostic／8张CSV哈希及上游Stage-61 L40来源，代码／模型／receipt／缓存继续绑定，不重跑旧summary或训练。准备时尚无服务器后继结果。

本轮问题：同一个已发生预测历史下，**只改变一次接受哪个解码框，能否改善随后31帧；当帧最好是否也有更好后继**。原始V1及固定E头均保持不训练，原首帧模板固定，不换V2。演员不接收GT；GT仅作事后标签和已消费开发病例选择。q0保留是可实现的一次干预，GT当帧最好与看完分支后的最好都只是离线条件对照，不是在线评分器。

固定12例、5源视频：R01原生0304 recrop首次q0良好却选低（现记录555），R02原生0226 history首次漏选（15），R03／R04 E0307 history两种子首次良好q0丢失（42／41），R05 E0307／2132 history在441～501唯一漏选；N01原生0226、N02／N03 E0307两种子、N04 E0304／2132 history首次失败且无良好候选（115／392／395／1093）；P01原生0176 history:162、P02原生0204 history:664、P03 E0304／2133 history:554要求q0≥0.5。首次条件从缓存重新解析，R05具体帧自动定位且必须唯一；缺支持或不足完整32帧即拒绝，不换点。

每例读取源实际5框历史，在同一进入历史执行共享的4次原生搜索；四条分支分别接受q0～q3的公共框。随后每分支31帧按来源history／recrop原规则做4次搜索并只接受自己所选输出。仅首次选择不同，不持续强制q0、不改候选构造、不GT重置或借用其他分支未来。共享锚点四候选及原选择的整条32帧分支必须逐项精确复现源框、候选父框／输出、特征、分数、裁剪哈希、选层与调用数组，否则停止。重复输出照样保留并报告独特数量，不作独立样本。

预算：每例`4+4×31×4=500`次搜索，共6000次／42000块、48条短分支、1536分支帧输出，0新完整轨迹、0训练。第一例共享锚点＋原选择分支作smoke的128次全部复用，非额外128次；缓存有receipt则续跑不重复网络。完整32帧单条候选政策逻辑预算128次，与全实验共享后的实际6000次区别报告。未提交的中断分支可能从头重跑，失败尝试额外开销不属于已完成缓存的预算。

允许本次获得的物理L40 UUID与Stage-61 L40来源不同，prepare显式绑定并记录唯一差异；型号仍必须L40、原软件／权重／配置／模型状态不变，所有新分支在同一已绑定设备。新GPU上的源对角线仍要求精确0差。prepare之后换卡不能覆盖旧manifest。启动复用封存配置构建器，保持原CHECKPOINT哈希验证，避免之前的L20或可变YAML问题。

本轮会新增保存所重放帧完整score／size／offset图，事后分析锚点四裁剪的dense最好IoU、格点排名、Hann前5格点、GT输入尺寸、中心误差、宽高比例和覆盖。前5为网格格点，不做旧局部峰值NMS；dense与公共浮点框误差可能导致整数IoU边界不同，分别报告。Stage-62无完整图，不能把新图说成旧缓存已有。

产物diagnostic＋6表：case_summary、branch_summary、contrasts、scope、dense_focus、timelines。比较q0−原记录、当帧GT最好−原记录、事后32帧最好−原记录，均报告即时／随后31帧／32帧均值／末帧IoU及低IoU帧变化。不拼oracle完整轨迹、不给新政策AUC或病例／帧级显著性；多个锚点和两种子依赖同一开发视频。本轮COMPLETE只表示诊断完成。

同步新代码后在原环境、同一个L40任务内运行：

```bash
cd ~/workspace/paper
export STAGE62_ROOT="$PWD/output/stage62/cached_candidate_selection_v1"
export STAGE63_ROOT="$PWD/output/stage63/matched_one_choice_continuation_v1"
export STAGE63_GPU=0
bash -n tools/run_stage63_choice.sh
bash tools/run_stage63_choice.sh
```

默认原CHECKPOINT，脚本顺序prepare→smoke→collect→summary，用cat合并diagnostic及case_summary／contrasts／scope／branch_summary／dense_focus到`docs/experiments/06_sgla_uav_tracking/results/stage63_results_<UTC>_<pid>.txt`。回传一个TXT即可；逐帧表和响应图缓存留服务器，不自动commit／push。

本机**4项测试通过，27.903秒、无跳过**，含真实小通道CENTER／公共解码、共同预测历史与一次干预、GT改变不改演员输出、E载入、精确来源对角线、Stage-59→60→61→L40→62→63小流程、仅L40 UUID差异、预算、smoke复用、重复采集零新调用、旧文件逐字节不变与来源／缓存／结果篡改拒绝。小流程2例／8分支／3帧72次，smoke12次计入；上游Stage-58边界、骨干、图像、设备为fixture，不是服务器6000次验证。CLI／4文件AST／脚本LF／维护检查通过。只读`bash -n`在沙箱内MSYS权限失败，随后经自动审批在沙箱外成功，Stage-63 Bash语法检查已通过；不改变第0.15节的历史未完成记录。

### 0.18 2026-11-03：Stage-63完成，单次选择的持续后效与能力限制

用户上传`D:\Users\23920\Desktop\stage63_results_20261103T015043Z_159855.txt`，已用cat读取并逐字节归档。完整[Stage-63复核](docs/experiments/06_sgla_uav_tracking/STAGE63_RESULTS_REVIEW.md)为下次接续入口。六份材料是diagnostic及case_summary／contrasts／scope／branch_summary／dense_focus，5表12／36／12／48／48行。原件70209字节、609行，SHA`a149c54fed46fec612f77ad75cc9c0d99dfcdb1d5eb2a0079bf1af653d90d214`；诊断指纹`cad93f1d6049d061046fedd1b12afa4ab05b7409c00018171563c306558a2953`。新增原件专用`-text`属性。不要重复请求这六份材料或重跑本轮。

状态`COMPLETE_MATCHED_ONE_CHOICE_CONTINUATION_DIAGNOSTIC`。12例／5源视频、48条32帧分支完成；共享48＋尾段5952=6000搜索、42000块，smoke128包含其中、额外0，0训练／模型／完整轨迹。runtime与Stage-61 L40来源相同，包括同一物理GPU UUID，声明差异为空。报告来源分支框差0、最大decode误差0.00006103515625。这里只改变首步接受哪个公共框，此后同规则自行滚动；下列IoU是0～1、非全视频AUC。

**R01原生0304 recrop:555：** 原选择q2当帧／32帧均值／末帧IoU全0，32帧全低；一次接受q0后窗口555～586连续32帧良好，均值0.7913、末帧0.8982，低帧32→0。q1也32帧全良好、均值0.7872；q3仍全低。四裁剪锚点覆盖均1，所选q2网格有12个良好格点、最好0.7825且Hann排第2，公共输出却0。q0／q2 Hann峰约0.28691／0.29840，较高峰未代表安全后继。这里验证的是接受q0／q1，未验证q2网格改选第2格点的未来，也未证明超过原生history／current；R01来源是recrop对照。

**R05 E0307／2132 history唯一恢复机会自动定位为475帧q2：** 原q1当帧0.1846、32均值0.2035、末帧0.1351、0良好／16低；接受q2后窗口475～506连续32良好、均值0.7606、后31均值0.7687、末帧0.7571，均值差+0.5571、低帧16→0。q2是滞后中心＋最新尺寸，q1是最新中心＋历史中位尺寸。q2 Hann峰0.40808略低于q1的0.41096，且仅q2网格有1个良好格点，其余q0／q1／q3 dense最好0.4721／0.2230／0.2715。不能只在所选q1图内重排获得良好框。

R05单纯接受q0当帧虽+0.1158，32均值仅+0.0026、后31反而−0.0010，低帧16→18、良好帧仍0。**有效q0保护与非q0恢复机会选择不能由“总回退q0”统一解决。** R05的q2效果是有共同进入历史的局部干预证据，选择该帧／该候选仍使用离线GT描述，尚无GT-free触发或完整视频506之后结果。

**N01／N02／N03／N04四个负例：** 对应原生0226:115、E0307两种子392／395及E0304／2132:1093。16个锚点裁剪均完整覆盖GT，四裁剪中最大dense IoU分别0.3956／0.2548／0.3008／0.2254，全部dense良好格点数0。每例四条32帧分支也均0个良好帧；这是锚点网格不足加后继未恢复，不能说所有后继图均已分析或整段无目标信息。N01中心偏移明显；N02／N03所选高度约2.726／2.701倍GT并纵偏；N04宽约3.811倍GT，属于几何／输入线索而非唯一原因。N04 q2当帧较差0.0224却均值+0.1115、少11低帧，但良好帧仍0且末帧0，不算恢复成功。

**其他病例：** R02原生0226:15的当帧最好替代均值仅+0.0048，三替代分支各仅1个良好帧、无连续3帧良好；窗口止46，未到115失效。R03／R04早期41／42良好q0损失，保留q0均值差−0.0020／+0.0151，原选择分支已从43有连续良好段；后来392／395失效不在这两个早期32帧窗口内，不直接认定根因。P01当帧最好q2反而均值−0.0068，P02／P03四分支均32帧良好、差异小。当帧最好、后继均值最好、末帧与良好帧目标不等价，不把事后最好候选当作学得策略。

本机已重算JSON自身指纹和5张已提供CSV的字节SHA，来源Stage-62指纹／完整runtime／12例head SHA与原件一致；由48条分支重建12个case摘要及36条contrast，32帧／后31加权关系、计数、索引、范围和预算均相符。既有首次条件／事件及7个关注锚点IoU与Stage-62表相符；120个evidence条目正好60个NPZ及receipt。**没有服务器实际响应数组／NPZ／receipt或timelines，未独立复算dense格点、每帧调用或来源GPU对角线**；表间自洽核对与服务器固定代码的原始检查区分。逐帧表1536行及完整新响应图保留服务器，已有摘要足够本次判断。

上述接续方向现已由0.19落实为GT-free候选可靠性原型；Stage-64已准备，尚无服务器结果。Stage-60通用未来均值评分器负结果保留，新目标与匹配控制的差异必须写清。Stage-1～63代码和旧结果冻结，不重跑旧GPU／训练或默认切换V2，不代commit／push。

### 0.19 2026-11-03：Stage-64 GT-free候选可靠性原型已准备

本节为结果前准备记录；现已完成，实际结果与下一次接续以0.20为准，以下“尚无结果”是历史叙述。

用户要求下一步实验，已实现[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE64_CAUSAL_CANDIDATE_RELIABILITY.md)和tools下5个新文件：`stage64_reliability_common.py`、`stage64_reliability_experiment.py`、`stage64_reliability_summary.py`、`test_stage64_reliability.py`、`run_stage64_reliability.sh`。输入完整Stage-63及全部上游来源，读取6表／分支receipt／代码绑定，不重跑旧summary。新root为`output/stage64/causal_candidate_reliability_v1/`，服务器尚未运行。

仍是原始V1无人机SOT，SGLA骨干／候选块／选择器／头及首帧模板均冻结。只拟合2折×2标签目标＝4个确定性小评分器。**训练仅使用该折fit组原生SGLA的history／recrop完整缓存**，所有帧均保留；每视频、视频内两政策、政策内帧、帧内3个相对q0候选对依次等权。不使用E轨迹拟合，以免互补折E头把评价组训练过的模型带入当前折。E仅迁移评价，两个种子用相同fold评分器。

GT-free输入为旧24项＋8项几何／候选一致性：候选连续IoU均值与最大、q0框一致性、自身历史速度预测残差、历史中位尺寸误差、候选中心／尺寸分歧、相对Hann排名。32项截断[−10,10]再加平方，共64项；无GT／帧号／视频ID／头ID／未来图像。历史只含自身接受的5个预测。候选生成保持原history四候选，原始公共框不平滑／重解码。

匹配控制quality标签＝当前IoU；reliability标签＝`1[IoU≥0.5]−1[IoU<0.2]`。两者都回归相对q0标签差，同状态池／特征／权重／RMS归一化／岭系数0.01，只有标签不同；不是已校准概率，也不是未观测的未来价值标签。候选没有超过q0＋1e-6则接受q0，其余按原顺序处理并列，不搜索阈值或checkpoint。两折原生fit状态合计27050个、候选框标签108200个，拟合不新增网络调用；不声称已学习真实32帧后继风险。

原生四组Hann／固定q0／quality／reliability各12轨迹；E两组Hann／reliability各12×2种子，共96条**全部新执行**的完整自身轨迹。每帧每政策真正4次搜索，包括q0控制；总432800搜索／3029600块，首视频1024次smoke直接复用，额外0。E没有quality／q0新消融，不作E内目标效应结论。允许新L40 UUID显式绑定，其他软件／权重／配置不变；Hann整轨迹按来源history逐项精确复现，原生q0框按来源current精确复现，不通过则停止。中断未提交视频可重跑，失败尝试额外成本不计入完成缓存预算。

主对比原生reliability−quality，必要对比−Hann／−q0，三项AUC用视频bootstrap种子2164、20000次、家族端点0.05/6与1−0.05/6。积极pilot要求三个家族下界>0、对quality／Hann均值≥+0.5；相对q0无新增稳定失效、低IoU均值不增、可靠帧名义CI下界≥−1，且至少4视频改变Hann选择。E为次要迁移，种子先在视频内平均。12源组均已消耗，PROMISING也仅是开发pilot，不是新方法或独立外部验证。N类不移除，本轮不声称修复其无良好网格问题。

本机**3项测试通过，34.707秒、无跳过**。真实小型CENTER及Stage-59→60→61→L40→62→63→64小流程，覆盖fit只读本折原生组、held-out GT改变不改拟合输入、未来GT改变不改演员输出、两目标标签与q0并列、Hann／q0控制、E权重冻结、smoke复用、重复运行零新前向、来源字节不变及模型／缓存／结果篡改拒绝。小流程2视频／16轨迹／256搜索，smoke128计入；上游Stage-58边界、骨干、图像、设备为fixture，不代表服务器真实96轨迹。CLI／4文件AST／LF／维护检查通过；沙箱外经自动审批的只读`bash -n`已通过。

同步新文件后在原环境同一L40任务运行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE63_ROOT="$PWD/output/stage63/matched_one_choice_continuation_v1"
export STAGE64_ROOT="$PWD/output/stage64/causal_candidate_reliability_v1"
export STAGE64_GPU=0
bash -n tools/run_stage64_reliability.sh
bash tools/run_stage64_reliability.sh
```

脚本顺序prepare→fit→smoke→evaluate→summary，cat合并decision及policy_summary／paired／scope／fit_support／video_influence／focus六表到`docs/experiments/06_sgla_uav_tracking/results/stage64_results_<UTC>_<pid>.txt`。回传一个TXT即可；其他逐视频／种子／事件表及评分器和NPZ留服务器。未代commit／push，Stage-63原件及复核修改保持。

### 0.20 2026-11-03：Stage-64已完成，可靠性标签原型未超过连续质量控制

用户上传`D:\Users\23920\Desktop\stage64_results_20261103T071456Z_203326.txt`，已用cat读取并逐字节归档。完整[Stage-64复核](docs/experiments/06_sgla_uav_tracking/STAGE64_RESULTS_REVIEW.md)为当前入口。七份材料为decision及policy_summary／paired／scope／fit_support／video_influence／focus，6表6／16／96／24／12／96行；其他video_summary／seed_video_summary／events和模型／NPZ仍在服务器。回传集合已齐，不重复索取或重跑。

状态`REVIEW_CANDIDATE_RELIABILITY_PILOT`，pilot／retention false，support true。原生q0／Hann／quality／reliability AUC分别63.4512／65.2483／66.2438／61.8770，低IoU14.46%／13.06%／11.01%／17.98%；E Hann／reliability为67.7506／63.6500、10.76%／17.73%。原生仍冻结V1、E为旧冻结头，只用本折原生fit侧history／recrop拟合4个小评分器；64特征、其他拟合配置匹配，quality当前IoU、reliability当前三级效用，不是长期风险标签。

主要reliability−quality −4.3668，名义CI[−12.5716,+0.1290]、家族[−16.1392,+0.2732]，2胜10负、中位−0.1956、最小留一−4.9389；剔除最差0226后仍−0.4020、2胜9负，所有留一最大也是负。对Hann −3.3714，名义[−8.2604,−0.2369]、家族[−9.8992,+0.0530]；不能把名义全负写成家族全负。对q0 −1.5742，家族跨零、2胜10负；低IoU+3.5255个百分点、可靠帧−8.5727（名义CI[−20.9026,−0.5937]）及0003新增失效，保护未过。非q0宏比例69.72%、自身状态相对Hann改选73.32%，使用充分。

**0226：** native quality55.1997、112失效339恢复，q0也338恢复；reliability7.2197、124起未恢复，Hann5.0819、115起未恢复。reliability对quality损失47.9800，尚无共同起点证据定位唯一干预病因。**0307：** native Hann／q0分别471／474恢复，AUC67.7834／66.5098；quality／reliability分别386／396起未恢复，40.6989／40.1468，两目标均破坏恢复。不能把新自身历史等同Stage-63 E0307:475的共同起点。

**0003：** 原生reliability从138起未恢复、AUC75.9921，另外三政策均无失效；总144跟踪帧，为短末尾事件，不叫数百帧长失跟，但确实新增稳定视频失败。E两种子也都138起未恢复，E Hann都无事件。**0176／0204：** 两学习政策保留相对q0的收益，但0176低于Hann，0204与Hann相近。原生0304四政策均稳定，本轮不是修复Stage-63 recrop:555，不能混算。

**E0304：** 2132 Hann44.8226、1093起未恢复；reliability73.0987且无失效，局部+28.2761；2133都无失效但−0.8502，视频均值+13.7129。值得保留的GT-free完整政策局部改进，但未做共同起点归因；E无quality组，不能归功于三级标签胜过连续IoU。E0226两Hann种子恢复、两reliability未恢复，视频−48.6367。E总体对Hann −4.1005、3胜9负，种子宏差−3.0637／−5.1374；任意单视频留一仍负，最大−0.0518。E低IoU+6.9615个百分点、可靠帧−4.7923。

本机额外探索性重算quality对Hann +0.9954、名义CI[−7.1894,+11.6862]、7胜5负、最小留一−3.4702；对q0 +2.7926、区间跨零、7胜5负。quality对Hann在0226 +50.1178、0307 −27.0845；去掉这两条后其余10条−1.1088。quality相对q0可靠帧仍−2.9749，不因本轮最高AUC就列为新方法PASS，也不在本结果上选配方。

全部96新轨迹／432800搜索／3029600块，smoke1024计入、额外0；fit27050状态／108200候选框、0 SGLA训练。runtime与Stage-61 L40完全相同、同物理UUID，差异为空；报告来源框差0、scope最大decode核对误差0.00048828124954525265。原生q0与旧current框严格复现、Hann与旧history逐项复现属于服务器固定代码检查，本机没有原NPZ。

本机验证JSON指纹、6表字节SHA、96 scope／focus组合及六政策宏AUC、四项AUC完整视频bootstrap／三家族区间／胜负／留一／12条influence；16比较宏均值关系相符。fit仅native、每折6 fit／6评价不交叉、权重和1、状态数与来源长度相符；scope调用预算、96 head SHA、48条Hann／q0来源AUC及完整runtime与旧L40原件相符。192 evidence路径正好96 NPZ及receipt。**未读取服务器实际模型系数／候选逐帧数组／receipt，未独立重建拟合、逐帧事件、来源框差或重跑GPU**；区分表间复核和服务器原始审计。

原件85785字节／704行，SHA`eea618f75667c0b7658cce453958aa177f4b7f77d7814c7999e9f2947b24858e`；decision指纹`44b6a945b0ef75eef1c16e6f9b227ae566c7f06283d41c82c1285a2401dc8b8e`。plan `3633adaa9b4ee1daf8a4d23e985fb83aab7d62ef51bcbfe0d633b3a1232780b5`，run `9186508a3f31f8c13c3e29578f30b5cd7b74a235c98d71e2dca2f56d557a27b9`，training `f44ba087e41cba70b3f4004fe6459b695468780a4254a5ac293479e0dffb3e27`。原件新增`-text`属性，源码与旧结果冻结，不代commit／push。

**下一步尚未实现Stage-65。** 优先缓存内共同状态重评分，统计有效q0被错误替换、良好候选漏选、全候选不足，以及fit／自身历史分布；保留0226／0307／0003与全12视频。然后用连续质量依据配合有效状态保护／低置信拒绝干预，保留非q0恢复路径，按新阶段固定fit规则和匹配控制。未选择候选的未来在旧轨迹中不可见，不能把当帧重评分当成已验证后继收益；现有表也不够指定标签损失或分布偏移为唯一病因。用户不追求绝对原创，继续以明确增量价值推进SGLA无人机SOT，不默认换V2／检测。

### 0.21 2026-11-03：Stage-65共同缓存状态诊断已准备

本节为准备历史；服务器已完成，实际结果与下一次接续以0.22为准，以下“尚未运行”为当时叙述。

用户要求下一步实验，已实现[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE65_CACHED_SAME_STATE_SELECTION.md)及4个tools文件：`stage65_selection_common.py`、`stage65_selection_audit.py`、`test_stage65_selection.py`、`run_stage65_selection.sh`。输入完整Stage-64，新root为`output/stage65/cached_same_state_selection_v1/`。**服务器尚未运行，不能预报错误率或分布差异。** Stage-1～64源码及旧权重／模型／缓存／报告冻结；旧Stage-64结果复核的“尚未实现Stage-65”为准备前历史。

这是一轮快速CPU缓存实验，不是另训评分器或方法采用门槛。对同一保存预测历史和四个公共框分别应用q0／Hann／quality／reliability，影子选择不推进状态；GT只作离线标签。分析有效q0被丢弃、良好替代漏选与四候选不足。E的quality／q0只是影子选择，不能写成新完整E政策。

状态池为96条Stage-64自身轨迹、108200状态；另24个本折原生fit history／recrop参考、27050状态，共135250状态出现次数、541000个缓存框出现次数和被报告的规则应用。它们有重复内容与不同历史，不能称作独立样本或新增搜索。Stage-64搜索432800已付出，fit另复用108200个旧L40公共框；本轮0新网络、图像、完整轨迹、训练或模型。原4评分器通过fit输入数值重建核对，不保存新模型、不改系数、不搜索阈值。

全量报告当帧已选／最好／q0 IoU、regret、良好／低帧比例；条件率包括有好候选却漏选、q0良好却失去良好／降为低IoU、q0不足且有良好替代时救回。空分母保持None并列支持数；种子先在视频内有支持平均，再视频等权，fit_reference与held_out分开。另列quality−reliability／quality−Hann／reliability−Hann的共同状态当帧差，绝不写成全视频AUC或未观测未来收益。

特征支持采用该fold原生fit输入、原视频／政策／帧／三个候选对权重和冻结模型scale；64项相对q0标准化特征的1%／99%逆加权经验CDF区间。输出坐标超界／任一维超界比例、绝对标准化均值与最大坐标99%分位数；是描述，不是在线拒绝阈值、已校准OOD或分布偏移病因证明。另统计三级标签相同但IoU不同的候选对及对应目标的平均绝对预测误差；两目标单位不同，不直接比绝对误差大小断定校准优劣。

10个病例按原封存事件解析：F01原生0003 rel首次失效；F02原生0226 rel首次失效、F03 quality首次恢复；F04原生0307 quality首次失效、F05 q0首次恢复；F06／07 E0226两种子rel首次失效；F08 E0304／2132 Hann首次失效；F09／10 E0307两种子旧自身E current首次恢复。预期帧138／124／339／386／474／127／127／1093／471／511仅为来源提示，不作在线规则。各自±30帧、边界截断，同视频origin种子全部状态提供政策×四影子规则，预期30个历史窗口／120行摘要；跨提供政策仍不是共同进入历史。缺失事件停止，不能另挑病例。

完整Stage-64来源plan／models／run／smoke／decision、9表、代码和cache receipt均核对；继承全上游绑定，检查96头身份、原生q0／Hann对角线、原fit支持和系数。新manifest／reference封存本轮来源、代码、锚点及fit区间，发布前再核对旧文件；同结果可重入，不同旧结果拒绝覆盖。没有调用旧summary补报告，不加载checkpoint或创建网络。脚本只运行新audit，用cat合并diagnostic及policy_summary／matched_summary／feature_shift／scope／focus五表，回传一个TXT；逐视频、逐种子、focus_frames等六表留服务器。

3项真实CPU测试通过，50.983秒、无跳过；包含同状态有害替换／救回、空分母、GT隔离、加权分位数、种子→视频平均、缺锚点拒绝，以及真实小型CENTER Stage-59→64缓存后接65。审计期间阻断Torch／OpenCV导入、图像原字节读取／哈希和新演员推理；重复结果相同、来源字节不变，模型／源表／新结果篡改及不匹配头身份拒绝。小流程2视频／16条64轨迹、64 held-out＋16 fit状态、320缓存框／规则应用、16行窗口摘要；上游骨干／图像／设备为fixture，不代表服务器实际120数据集已审计。CLI、AST、LF及Bash语法通过；沙箱阻止Git Bash初始化后，只读`bash -n`经自动审批在沙箱外通过。

同步新增文件后，在原`sglatrack-test`环境运行，无需GPU分配：

```bash
cd ~/workspace/paper
export STAGE64_ROOT="$PWD/output/stage64/causal_candidate_reliability_v1"
export STAGE65_ROOT="$PWD/output/stage65/cached_same_state_selection_v1"
bash -n tools/run_stage65_selection.sh
bash tools/run_stage65_selection.sh
```

结果为`docs/experiments/06_sgla_uav_tracking/results/stage65_results_<UTC>_<pid>.txt`。未代commit／push，上一轮Stage-64归档与交接修改保留；现阶段不在这些结果上直接挑在线阈值。用户目标仍是SGLA无人机SOT，有相似工作不是否定增量价值的理由。

### 0.22 2026-11-03：Stage-65完成，有害接管与退化历史的候选不足并存

用户上传`D:\Users\23920\Desktop\stage65_results_20261103T081800Z_253219.txt`，已用cat读取、逐字节归档；完整[结果复核](docs/experiments/06_sgla_uav_tracking/STAGE65_RESULTS_REVIEW.md)为当前入口。diagnostic＋policy_summary／matched_summary／feature_shift／scope／focus五表32／24／8／120／120行齐全；其余六张逐视频／种子／focus_frames表与reference、原NPZ留服务器，不重复索取同一回传集合。

状态`COMPLETE_CACHED_SAME_STATE_SELECTOR_DIAGNOSTIC`，不是采用PASS。96条64轨迹108200状态＋24原生fit参考27050，共135250状态出现次数、541000缓存框／规则应用；0新网络／图像／训练／模型／完整轨迹。旧评分器只重建核对，Stage-64原成绩保留。四规则在同一保存状态内输入相同，影子选择不推进；跨提供政策的同帧历史不同。E的quality／q0仍只是影子选择，不能写成已执行新E政策。

**同状态差与自身历史差：** quality−reliability的当前IoU宏差，在native H／0／Q／R历史分别+0.001304／+0.001352／+0.000903／+0.001055；E H／R为+0.001857／+0.000607。选择索引分歧约44%～49%，不等于有意义改框。native Q自身平均IoU0.671970、R自身0.625095，差0.046875；在R历史换Q只+0.001055，Q跨两种历史差0.045821，两者相加为自身差。是描述性代数分解，不是97%以上因果归因。R历史逐帧四框最好均值0.641975仍低于Q自身；只限制固定输入重选的当帧空间，不是未来或全视频oracle上界。

**F01原生0003，108～144：** H／0／Q历史37状态q0全良好；R影子分别把1／2／1个良好q0换为低框，Q没有同类损伤。实际R历史已有5状态四框全低，q0选31良好／6低、R选30良好／7低，含1次良好q0降为低；H／Q影子也只能31良好／6低。本表不能定位那次替换是否恰为138或唯一根因，逐帧表仍在服务器。支持接管保护，同时不能只靠末尾改评分挽回全部状态。

**F02／F03与F06／F07：** 0226原生94～154四历史均0/61有好候选；E两种子97～157 H／R也全部0/61。到原生恢复309～369，0／Q历史各33/61有好候选，H／R历史则61状态四框全低。R历史换Q仍0良好；Q历史q0选32良好，Q／R均31。R在早期F02自身历史当前均IoU0.1859还高于同状态Q0.1813，却没实现完整恢复，不能把即时均值视为长期效用。

**F04／F05原生0307：** 356～416，H历史有31个好候选状态，0／Q各9，R为0。恢复444～504，H有39、0有33，Q／R历史都0。H历史q0好35，H选35（丢2、救2）；Q选30、丢5个好q0且救0；R选31、丢4且救0。Q／R各自历史却分别53／45低帧，换当帧评分不能创造良好框。需要保护恢复依赖的状态，未定位更早唯一几何干预。

**F08 E0304／2132，1063～1123：** R历史61/61有好候选，R选60好、均值0.7283；同状态H影子61全好、0.7706反而更高。原H历史只有19状态有好候选，H／R各选17、均值0.2709／0.2605。Stage-64完整+28.2761 AUC伴随更好进入历史，不能说R每帧都更优，也不能声称H移到该好历史后已得到全视频收益。

**F09／F10 E0307：** 2132 H历史441～501恰1个好候选状态，四影子规则全0良好，与Stage-63 475帧q2唯一机会吻合；仍未学会该恢复选择，q0保护不能替代非q0救回。R自身该窗口61状态四框全低。2133在481～541两历史都0好候选，R也全低。Stage-63一次接受q2的32帧后效与本轮无推进的影子比较保持区分。

**特征／标签线索：** native fit history／recrop任一维超界候选对20.22%／16.44%，held H／0／Q／R为25.54%／24.84%／22.17%／26.69%，E H／R为22.94%／25.09%。轨迹内P99 max绝对标准化坐标的宏均值对应fit12.61／11.34、native53.56／37.00／17.59／47.51、E14.50／49.01；不是全帧合池P99或在线OOD阈值。fit history与held native H按来源控制是同一批原生历史，不同折fit／held视角使用不同scale和参考区间，标准化差异包含参考变化，不能全归于原始状态迁移。H尾部更大但完整成绩好于R，不能用尾部单判病因。三级效用同值但连续IoU不同的候选对约53.17%～64.69%，说明信息压平，不证明唯一根因。两目标误差单位不同，只作同目标跨来源比较。条件率是支持视频宏均值，不是全帧池比；重点病例用整数计数解释，避免8.49%等稀疏分母比例误读。

本机验证JSON自身指纹、5表字节SHA、source64报告／training指纹及runtime；120组合／两折fit与held不交叉、预算；六实际政策低IoU／非q0／Hann改选率与64相符。24同状态差值由政策表重建，120关注窗口整数计数／条件率／守恒／regret相符，30×4规则齐全；10锚点与旧64／L40事件相符。**未读服务器逐帧候选／模型系数／reference数组／receipt或留存六表，未独立重算4模型、每帧分数／IoU、具体有害接管帧、完整32／8行宏聚合。** 本机表间核对与服务器固定代码原始审计分别陈述。

原件103080字节、509行，SHA`e2cdaffe622614515a73471e7c6925b9c3f59d3e58fd9cb73bf20d9e52816b04`；diagnostic指纹`0d840346bce337d1df0327798e3ce0497599f573b3b84f707ea16c9d7944c451`。plan `6924322621361b6d5fae1d148eb01f1ae313cc2be7fbd6f246d21c4e136056b3`，reference `9f012d8f0450febab94b76ecb48245f0fbb21007d7136212cae4fb3b734a659f`。新增原件`-text`属性，Stage-1～65源码／旧结果冻结；不代commit／push，不重复测试未变代码或GPU。

**Stage-66尚未实现。** 优先连续质量＋绝对q0可靠性判断的接管保护，匹配单纯提高门槛的控制、保留非q0恢复；弱候选推进可另列处理。旧差分评分器把q0固定0，没有直接预测q0是否好，不能用该0作安全判断。新判别与阈值限fit侧确定，再检验完整自身轨迹；不用GT帧号或全开发结果挑规则。无需先证明所有病因，也不因相似工作否定SGLA无人机SOT的增量空间。

### 0.23 2026-11-03：Stage-66原生SGLA接管保护原型已准备

本节为准备历史；服务器已完成，实际结果与当前接续以0.24为准，以下“尚未运行”保留为当时叙述。

用户要求下一步实验，已实现[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE66_NATIVE_Q0_TAKEOVER.md)和5个tools新文件：`stage66_takeover_common.py`、`stage66_takeover_experiment.py`、`stage66_takeover_summary.py`、`test_stage66_takeover.py`、`run_stage66_takeover.sh`。输入完整Stage-65及全部上游，新root为`output/stage66/native_q0_takeover_v1/`；源码／旧模型／结果Stage-1～65冻结，先前归档和交接未提交修改保留，不代commit／push。**服务器尚未运行，不能预报原型收益。**

研究仍为原始SGLATrack V1无人机SOT，6个前缀＋1个独立候选块、首帧模板和原头全部冻结。先只评原生五组Hann／q0／quality／margin／protect，不加E，降低本轮成本并先检验基础对照；E迁移后续单独设计。原quality已经在0003稳定，保护保留不应宣传为新增修复R失败；主要关注Q尚存0307等损伤及全12视频取舍。

每帧四个原history候选都执行原网络。原Stage-64 quality系数和scale原值复用；q0为本帧从最新父框得到的公共解码输出。新增2折×1绝对q0 IoU岭回归：输入同一64项特征的q0行，不再做相对q0相减；fit加权中心／标准差（下限1e-6）、截距、岭系数0.01，输出截断[0,1]。仅用本折6组原生history／recrop完整缓存，27050状态，GT只为fit标签，E／held-out标签不进入拟合。它预测IoU，不是校准概率或安全保证。

protect规则固定：原quality提议增益g；预测当前q0 IoU≥0.5时，只有g>0.02才允许替代，否则q0；预测<0.5时沿用quality，包括非q0恢复。0.5／0.02不按本轮结果搜索。返回q0仍更新当前解码框和历史，不是冻结上一帧；本轮没有所有候选弱时的独立几何修复。

margin控制在相同fit状态上按quality增益统一截断，以最接近protect加权接管率为目标；整组同分不拆，完全同距时保留较少接管的前缀。门槛为最高未接收增益、严格大于，避免相邻浮点数中点取整问题。记录Q／P／margin fit率、匹配差和最大同分组权重。它是fit侧率匹配，实际评价历史不同可产生不同接管率，不能声称评价精确等干预；五组网络预算相同。

12视频×5＝60条**全部新完整自身轨迹**，270500搜索／1893500块；首视频五组32帧smoke共640搜索直接复用，额外0。新权重只有2个CPU小预测器，0 SGLA训练。允许新L40 UUID显式声明，其余运行身份保持；同新卡新执行Hann／q0／quality控制，分别对原history完整字段／current框／Stage-64 quality完整字段精确复现，失败即停止，不放松阈值。原始V1 checkpoint同0.5节，不能换学习头。

主比较P−Q，必要P−margin／Hann／q0；原12组全保留，视频bootstrap种子2166、20000次，4项AUC家族端点0.05/8与1−0.05/8。积极pilot要求4家族下界均>0，对Q／Hann均值≥+0.5，且相对q0无新增原稳定视频失败、低IoU均值不增、可靠帧名义CI下界≥−1，至少4视频有保护否决、至少4视频仍接收非q0；否则REVIEW。即便PROMISING也只是已消耗开发pilot，不是外部确认。

每帧保存绝对q0预测／资格、原Q提议与增益、实际否决和既有候选数据，供summary重建全部选择、原拟合与率匹配。报告实际接管率、否决率、绝对q0离线误差／条件精度，以及AUC／失效／可靠帧／尺寸／恢复取舍；不声称速度或能耗收益。输入Stage-65 plan／reference／diagnostic、11表及其64模型／receipt／来源代码全核对，不运行旧summary补来源。

最终4项真实CPU测试通过，102.906秒、无跳过。覆盖绝对状态门控、弱q0保留非q0、并列、加权率匹配含相邻浮点、schema／模型绑定，以及真实小型CENTER Stage-59→64缓存／65诊断后接66。验证fit只读本折原生组、held-out GT改变不改拟合输入、未来GT改变不改演员输出，3条控制复现、smoke复用、续跑零新推理、旧来源字节与原权重不变，源表／模型／缓存／新报告篡改拒绝。小流程2视频、10轨迹、160正式搜索（80 smoke已含）；另有GT隔离重放，未算正式预算。上游骨干／图像／设备为fixture，不代表服务器60轨迹已完成。CLI／4 Python AST／Bash LF／只读bash -n通过，Bash检查经自动审批在沙箱外执行。

同步新文件后，在原`sglatrack-test`环境并取得L40 GPU运行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE65_ROOT="$PWD/output/stage65/cached_same_state_selection_v1"
export STAGE66_ROOT="$PWD/output/stage66/native_q0_takeover_v1"
export STAGE66_GPU=0
bash -n tools/run_stage66_takeover.sh
mkdir -p output/stage66
nohup bash tools/run_stage66_takeover.sh > output/stage66/stage66_run.log 2>&1 < /dev/null &
```

脚本prepare→fit→smoke→evaluate→summary，cat合并decision及policy_summary／paired／calibration／scope／focus／video_influence六表，生成`docs/experiments/06_sgla_uav_tracking/results/stage66_results_<UTC>_<pid>.txt`，回传一个TXT即可。日志在root父目录，不阻碍prepare空目录检查；`tail -f output/stage66/stage66_run.log`查看，Ctrl+C仅退出查看。nohup不延长GPU任务分配，仍需保持计算资源有效。

### 0.24 2026-11-03：Stage-66完成，绝对q0条件保护未建立收益

用户上传`D:\Users\23920\Desktop\stage66_results_20261103T115116Z_276612.txt`，已用cat读取并逐字节归档。完整[Stage-66复核](docs/experiments/06_sgla_uav_tracking/STAGE66_RESULTS_REVIEW.md)是当前入口。decision＋policy_summary／paired／calibration／scope／focus／video_influence六表5／16／2／60／60／12行齐全；video_summary／events／fit_support、模型和NPZ／receipt留服务器，不重复请求相同回传集合。

状态`REVIEW_NATIVE_Q0_TAKEOVER_PILOT`，pilot和retention false、support true。五政策Hann／q0／quality／margin／protect宏AUC65.2483／63.4512／66.2438／65.8682／63.3892；低IoU13.06%／14.46%／11.01%／11.01%／14.89%；原生可靠帧AUC70.0548／75.5602／72.5853／72.3764／72.4639。全部原稳定视频新增失败0，但不能据此忽略已有获益视频损伤。

主要P−Q −2.8546，名义CI[−8.2785,+0.0818]、家族[−10.8249,+0.2160]，3胜9负，中位−0.2226、最小留一−3.2461；去掉0176仍−0.2106，所有留一最大也是负。P−margin −2.4791，名义[−7.8147,+0.4243]、家族[−10.3757,+0.5826]，3胜1平8负，最小留一−2.8795；去掉0176后+0.1231、3胜1平7负，不能删除该例宣称条件信息有效。P−Hann −1.8592、5胜7负；P−q0 −0.0620、4胜2平6负，家族均跨零。相对q0低IoU+0.4353个百分点、可靠帧−3.0963（名义CI[−8.5077,−0.0933]），保护仍未过；相对Q低IoU+3.8867个百分点。

**fit率匹配：** fold0／1原生fit状态12336／14714，统一阈值0.0079342085／0.0053832194；P率14.35493%／19.21275%，margin14.35823%／19.21026%，差0.003299／0.002483个百分点，小于各最大同分组权重。两折等权Q率71.6070%，P／margin16.7838%／16.7842%。评价Q实际非q0率76.75%，margin19.72%、P15.09%；对应自身状态原Q提议率76.58%／76.56%，否决率56.86%／61.47%。这是fit侧率匹配，评价自身历史不同、实际率不等，不能称精确等干预。五组网络都4次搜索／帧。额外探索性margin−Q −0.3756、名义CI[−1.0126,+0.1379]、3胜9负，未建立单纯少干预的增益，也不能将P额外损伤全归于一般保守化。

**0176主要损伤：** P28.8479，315首次失效、无首次恢复、末尾未恢复1；Q60.7865，380失效387恢复；margin59.9508，432失效439恢复，但后来仍有末尾未恢复事件。Hann67.7419，407→412；q016.0983，162起未恢复。P比Q −31.9386、margin −31.1029、Hann −38.8940，虽仍较q0 +12.7496，但损伤此前恢复收益。原q0本已失败，故新增稳定视频失败不计此例；不能把新增失败0当完整保护。未有逐帧资格／否决和共同起点后继，不认定315帧某次高估q0为唯一根因。

**0307：** Q386、margin393、P394首次失效，均未恢复；AUC40.6989／37.5118／39.4379。P虽对margin+1.9261，对Q仍−1.2611，对q0−27.0720；Hann／q0在471／474恢复。较晚首次失效没有转为成功恢复。**0226：** P与margin AUC同为55.1485，首次失效126／114，均339恢复，不能用同AUC推出同轨迹；Q339、q0338恢复，Hann未恢复。**0204：** Q／margin／P均无失效，P72.5557，比q0+18.6340、Q−0.3051。**0147：** P比Q+1.4522，1036首次失效推至1073但仍未恢复。0003／0170／0126／0304／0329五个原稳定视频都保留无失败，Q本来就在0003稳定，不作新增修复R失败主张。

P平均失败事件0.5低于Q0.8333，同时平均末尾未恢复事件0.3333高于0.25、平均恢复事件0.1667低于0.5833；不同轨迹事件不是匹配因果比较，更少事件不代表更可靠。P自身q0预测良好占80.36%，资格精度88.22%（12视频支持）、q0 IoU MAE0.1616；Q历史精度91.14%、MAE0.1489，但模型相同、历史不同。资格精度不是否决正确率，当前q0好也不等于拒绝替代的后继更好。

60条新完整轨迹／270500搜索／1893500块，640 smoke包含且复用，额外0；0 SGLA训练，2个新绝对q0小预测器；原quality系数复用、E迁移／弱候选几何修复未执行。runtime与Stage-64完全相同、同物理L40，差异为空；报告来源框差0、scope最大decode核对误差0.00048828124954525265，二者含义分开。

本机验证JSON自身指纹、6表字节SHA；由60 focus重建5宏AUC及4项AUC配对、名义／家族bootstrap、胜平负、最差视频与留一；12影响值、16比较宏均值关系相符。2fold状态合计27050、率差及同分权重关系相符；60唯一scope每帧4搜索／每搜索7块、总预算相符；head SHA和runtime对齐64，3控制36条AUC／首事件／末尾未恢复字段相同；120 evidence路径正好60 NPZ及receipt。**未读原模型系数、逐帧预测／否决／框／receipt或留服务器三表，未独立重建critic／率匹配最近前缀、来源逐帧框差或非AUC配对区间。** 原始审计由固定服务器代码报告，本机表间核对与其区分。

原件58558字节、557行，SHA`3cadeb50ebefc8ea19b30313602e239924687010ed13d60c12c1be147cbf681a`；decision指纹`85e139b06020c9b918cbb76b642702c27eb2cf8c80cba3072673b5d3d30187a9`。plan `8351c4a03155cbe15b2b82a25c979f99d348f2ae0c7d8d03ec5063d3e3631941`，run `abcdecfb20ca8dc9336dec90cd6f6bdfa75bd65ac7e26d2339f5b8ab5c69031a`，training `32366a30715670fa3610d6b00dadc5784494a8dac1a56078a7105b1d024c9cea`。原件加`-text`属性，Stage-1～66代码和旧产物冻结，不代commit／push、不重跑未改代码的测试。

**当时的下一步建议（现已落实为0.25的Stage-67）：** 优先从0176缓存定位错误否决／历史分岔，保留0307受损及0226／0204保留收益；不能根据聚合精度确定唯一病因。若要学习放行／否决的条件收益，fit侧新采共同起点短分支并匹配控制，区分q0当前质量与接管后继价值；未观测未来不能由影子评分虚构。Stage-60未来均值评分器负结果保留，不能只更名为风险模型。不在本轮0.5／0.02等参数上补搜，也不因该规则失败终止有明确新增价值的SGLA无人机SOT研究。

### 0.25 2026-11-03：Stage-67已准备，共同起点放行／否决后继诊断

本节保留准备历史。服务器已运行到汇总并暴露锚点CSV字段不一致，当前恢复入口以0.26为准，不按下面原采集命令重新启动GPU。

用户授权继续下一轮；已先读0.24。新增[Stage-67协议](docs/experiments/06_sgla_uav_tracking/STAGE67_MATCHED_VETO_CONTINUATION.md)、`tools/stage67_veto_common.py`／`stage67_veto_experiment.py`／`stage67_veto_summary.py`／`test_stage67_veto.py`／`run_stage67_veto.sh`；只添加新命名空间，Stage-1～66源码和旧产物冻结。服务器尚未执行，无Stage-67成绩，未代commit／push。

**核心问题：** 当前q0质量预测／fit接管率不能判断某次否决的后继价值。固定全部SGLA V1权重、原首帧模板、四候选生成和Stage-66两类评分参数，直接做一次选择的局部干预，不更名重试Stage-60未来均值评分器，不新增训练或阈值搜索。

固定8个请求：0176／0307／0226／0204各自第一处实际改变输出框的protect否决；另0176／0307在其首次失效起始及之前31帧取最后有效否决、0226在首次恢复起始及之前31帧取最后有效否决、0204在已消费固定664及之前31帧取最后有效否决。`prepare`从封存来源解析确切帧并写入manifest；若无事件／无否决／不足完整32帧，明确记录unsupported，不回退、不缩短、不换视频。重复或重叠锚点仍单列计费，不能当独立样本。

每个起点只重放一次共享四搜索；强制输出q0(veto)或冻结Q提议(allow)，然后分别用Q／protect在自身预测上续跑31帧，形成2×2。原始protect／veto必须精确复现来源；若同一历史也匹配来源Q，Q／allow后继额外精确复现Q来源。实际裁剪／图像哈希、候选、特征、评分、层／块／head预算逐帧校验，raw／size／offset保留新缓存并做dense解码审计。GT只在离线选事件／评价，后继动作无GT重置。

最多8起点／32短分支／4000搜索／28000块，无新完整视频；首起点protect／veto的128次smoke包含且复用。对同一未来控制器报告`allow−veto`，另报其Q与protect之差的交互；IoU为0～1，分开当帧／32均值／31后继均值／末帧与好坏帧数。全60条来源另做缓存否决暴露、全12视频Q／protect首次框／裁剪分岔。`first_three_good_frame`为窗口局部三帧好，不能直接称完整恢复；不输出采用PASS、不为事件选择样本计算总体CI。

**本机验证：** `python -B tools/test_stage67_veto.py -v`四项全部通过，130.555秒、无跳过；真实CPU Torch／OpenCV和仓库CENTER，完整Stage-66→67来源链，smoke 8次＋正式新增12次＝20次搜索，缓存续跑0推理，q0／Q一次选择、原protect来源对角线、未来GT隔离、模型／来源字节不变、NPZ与来源／结果表篡改拒绝均通过。四帧合成来源仅在测试里设置明确否决暴露（q0资格阈值0、保护gain为1e6）；正式Stage-67仍原样拷贝真实Stage-66的0.5／0.02，没有调整。测试过程中修复逐帧汇总q0列名重复，最终全流程及幂等汇总已过。Bash `-n`通过；`check_workspace.py` PASS（505保留、200移动、106模块、0缓存、0换行差异）；171处本轮所涉本地文档链接有效，`git diff --check`通过。CPU小样本不代表L40服务器结果。

服务器执行：激活`sglatrack-test`，进入`/home/u25600003160604/workspace/paper`，同步本轮文件后`bash tools/run_stage67_veto.sh`。默认Stage-66输入`output/stage66/native_q0_takeover_v1`；新根`output/stage67/matched_veto_continuation_v1`，GPU0原始V1 checkpoint，观察并绑定L40 UUID。需要后台时先`mkdir -p output/stage67`，然后`nohup bash tools/run_stage67_veto.sh > output/stage67/run_stage67.log 2>&1 < /dev/null &`，日志在新根的父目录，勿污染prepare空目录。

脚本依次prepare／smoke／collect／summary，以`cat`汇总8份文件为`docs/experiments/06_sgla_uav_tracking/results/stage67_results_<UTC>_<PID>.txt`：diagnostic、anchors、case_summary、branch_summary、contrasts、scope、cached_video、divergences。逐帧veto_frames／timelines留服务器，先回传TXT即可。不得用本轮局部32帧干预推断更晚315等整视频失效已被避免；等待结果再确定可学习的新目标。

### 0.26 2026-11-03：Stage-67锚点CSV导出修复，保留已完成分支

本节为修复历史。修复已经成功、正式结果9份材料齐全，当前接续以0.27为准，不再重复执行下述恢复命令。

用户回传`stage67_veto_summary.py:115 → stage46_shift_audit.publish_table → csv.DictWriter`报错`dict contains fields not in fieldnames: 'frame'`。汇总已通过来源／完成标记／分支审计，失败在第一张anchors表导出。首F01请求没有frame、P01固定请求有frame=664，原发布器按首行取表头；原小样本测试仅一种请求，未覆盖此混合形状。尚未收到任何Stage-67效果数值，不推断是否达成局部收益。

**修复方式：** 原Stage-67 common／experiment／summary／test／run五文件均已纳入manifest代码SHA，因此保持字节不变。新增`tools/stage67_summary_repair.py`、`tools/test_stage67_summary_repair.py`、`tools/run_stage67_summary_repair.sh`；仅在原汇总调用内替换CSV发布适配器，将锚点请求规范为固定6列（case_id、sequence、selector、frame、status、resolved_frame），仅缺失的可选frame补None。保留frame=664与resolved_frame的区别，不丢弃未知列；其他表不变，已存在不同字节的结果仍拒绝覆盖。完整执行原load_plan／代码哈希／来源／receipt／对角线／预算校验，未放宽任何校验或重签旧manifest。新增`analysis/stage67_summary_repair.json`绑定修复代码SHA、原封存代码SHA、原计划／run指纹及汇总表哈希。

**执行：** 将三个新增修复文件同步到服务器，进入原仓库及`sglatrack-test`环境，运行`bash tools/run_stage67_summary_repair.sh`。自定义路径沿用原`STAGE67_ROOT`；默认`output/stage67/matched_veto_continuation_v1`。无需checkpoint或GPU参数，不再运行原全流程shell，不删collection、不改manifest、不重跑prepare／smoke／collect。新shell汇总原8份材料＋修复receipt为`docs/experiments/06_sgla_uav_tracking/results/stage67_results_<UTC>_<PID>.txt`，回传此文件即可。

**验证完成：** 三项导出回归单独运行3/3通过（0.040秒），覆盖真实8请求形状、原报错复现后恢复、空可选frame、unsupported P01、已解析帧不混淆、其他表原始字节、幂等／不同结果拒绝和来源失败不绕过。真实CPU Torch／OpenCV／CENTER完整来源链恢复用例也通过（含上游生成，所在批次131.039秒）；在同一原封存计划内加入一种固定帧unsupported请求后恢复全部汇总，核对manifest／run／smoke／所有collection NPZ和receipt字节不变、0新增推理、原对角线／篡改拒绝。首次批次另3项受默认TEMP沙箱权限影响，已把新测试临时目录移入workspace后单独全部通过，无跳过；未修改原封存测试。Bash语法检查通过，check_workspace为PASS（505／200／106、0缓存、0换行差异），171处本地文档链接及git diff --check通过；原Stage-67五封存文件git diff为空。未代commit／push。

### 0.27 2026-11-03：Stage-67完成，0176一次否决具有明显局部后继代价

用户回传[原件](docs/experiments/06_sgla_uav_tracking/results/stage67_results_20261103T142847Z_336860.txt)，38546字节／505行，SHA`9887bc003cdd614fce72da9250c343fa5a40de55bf9736183a6c1456173ea9a9`。diagnostic＋repair receipt＋anchors／case_summary／branch_summary／contrasts／scope／cached_video／divergences共9份齐全；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE67_RESULTS_REVIEW.md)。状态`COMPLETE_MATCHED_VETO_CONTINUATION_DIAGNOSTIC`，无采用门槛。8请求中7支持，0226恢复附近R01无有效否决而明确不支持，不补挑；共28短分支、3500搜索／24500块，128 smoke包含复用、额外0，无新训练／模型／完整轨迹。

**关键E01：0176:310，窗口310～341。** 同一个protect进入历史、四个共享实际候选，veto接受q0当帧0.488186，allow放行q2为0.507042，当帧差仅+0.018856。未来仍用protect时，32均值0.060053→0.556111（+0.496058），31后继均值0.046242→0.557693（+0.511451），末帧0→0.782559；低IoU27→6、良好0→23，窗口首个三帧良好段从320开始。未来用Q时，veto汇总指标相同，allow均值0.562656、末帧0.804236，均值差+0.502603；两未来控制器的放行效应交互仅+0.006546。这是该共同状态下一次否决的局部代价，不能直接当作310以后的全视频修复或Stage-66 AUC差的因果占比。

E01真实q0仅0.4882，未到0.5，不能把它写成“真实良好q0仍不值得保护”的纯例。按固定规则可推知当时critic资格已过且Q预测增益未越0.02，但精确预测值／逐帧图留服务器，不能杜撰误差幅度。allow仍有6低IoU帧，不是完全无短暂失跟；窗口止341，之后是否持续、是否再失效未知。该事件离线按已知首次失效挑选，不是在线无GT控制器已发现此时刻。

**阴性与保留控制：** 0307:370（E02），q0／提议0.452345／0.420561；protect后继下allow−veto均值−0.036971、低帧8→15，Q后继−0.009306、低帧8→9，四分支良好帧均0、末帧均<0.2。不能统一取消否决，也不能据此说其余两候选或dense网格都无好框。0204:664（P01）当帧+0.091161，protect后继均值仅+0.006103、末帧−0.060378；Q后继均值−0.000926；四分支均32好、0低，未显示恢复增益。该起点q0已0.7057，不是早期原生664失效的同一输入。

**首个实际分岔：** F01／F02／F03／F04为0176:3、0307:17、0226:4、0204:20，protect后继allow−veto均值+0.004566／+0.008187／−0.006869／−0.002691，Q后继+0.004054／+0.002591／−0.006487／+0.001111。当前IoU差与后继方向可不同，早窗均值差小也不能否定更长累积作用。四个F起点与来源Q历史相同，Q／allow来源对角线通过；E01／E02／P01不具备该Q来源对角线，但同控制器内allow／veto匹配仍成立。全部7个protect／veto对角线通过。

**全缓存描述：** 60条来源67625状态出现，每政策13525；protect8379否决，其中7895改变实际框、484仅索引差而框相同。12视频首次框分岔均为相同候选上的有效否决，裁剪下一帧分岔；0176 3→4、0307 17→18、0226 4→5、0204 20→21。0329索引差2／框差4／裁剪差5，不混同索引与实际干预。两种极端当帧好坏翻转计数全0（q0≥0.5而提议<0.2，或q0<0.2而提议≥0.5），不能解释为所有否决正确；E01的中间IoU值不满足这些极端定义。

**本机验证：** 两JSON自身指纹、repair所绑diagnostic字节SHA、7CSV字节SHA均过；5原脚本＋修复Python共6源码SHA与服务器完全一致，原Stage-66 report／training及完整runtime相同、同物理L40 UUID、差异为空。8／7／28身份、21对比及交互、全部32均值与31后继关系、35 scope的4搜索／帧和7块／搜索、18条对角线标记关系、70证据路径集合、60缓存计数与12分岔关系核对通过。未读服务器NPZ／receipt正文／GT／网格，没有独立逐帧重算来源框误差或干预轨迹；最大来源框差0是服务器原审计报告，不当成本机逐帧重验。veto_frames／timelines长表仅有哈希留服务器，已收9份不重复索取，不重跑未改源码的CPU测试或GPU。

指纹：diagnostic `7ffd6bbfa0ed237779cf76dc01f4a69a2a70ff1a18280808b1fe2623d832af9c`；repair `d3e53d8ebca7eb171498086dfcc6a9a437faa7691dcb04181f2617b4f37e4ba9`；plan `e2abf26a28870109844eee69a461453dc1178dc85f07304e4e4e55d6ba7eff09`；run `c2c5061ec2111d23b6a7084a20dcf2671bf955b868f56cb5e158c03455a1cb73`。scratch解析在`D:/Users/23920/Documents/Codex/stage67_bundle_parsed.json`，不作为新实验产物。原件加-text保留字节。本次check_workspace为PASS（505保留／200移动／106模块、0缓存、0换行差异），177处所涉本地文档链接及git diff --check通过，tracking／lib／tools未改。

**当时的下一步建议（已落实为0.28）：** 将新增价值落到“一次放行相对q0否决的条件后继增益／持续失跟风险”而非当前q0资格、接管更少或绝对未来均值；优先在各折fit组统一无GT触发条件下采成对分支，固定同一后继控制器／预算，保留冻结Q、率匹配margin和稳定轨迹保护对照，再用完整自身轨迹验证。不要只挑310等已知成功点或据7诊断窗口补搜阈值。若需量化310对整视频贡献，另行预定从共同起点续跑至末尾，本轮并未完成该检验。Stage-1～67实验和修复源码／旧模型结果冻结，Stage-67结果复核未代commit／push。

### 0.28 2026-11-03：Stage-68已准备，fit成对后继增益与匹配控制

用户要求下一步实验，已先读0.27并完成[Stage-68固定协议](docs/experiments/06_sgla_uav_tracking/STAGE68_FIT_PAIRED_TAKEOVER_ADVANTAGE.md)，新增`tools/stage68_advantage_common.py`／`stage68_advantage_experiment.py`／`stage68_advantage_summary.py`／`test_stage68_advantage.py`／`run_stage68_advantage.sh`。服务器尚未运行，当前无成绩；Stage-1～67实验／修复Python和旧产物不变，不代commit／push。

**具体目标：** 预测同一起点一次allow(Q提议)相对veto(本帧q0)，之后两条均执行冻结Q时的真实32帧均值IoU差。它是固定Q后继下一步偏离收益，既非绝对未来均值，也非新门控的自洽价值或概率风险。Stage-60负结果仍保留。

**fit隔离是核心：** 只以冻结原生Hann历史作为状态来源；对本折fit组重算本折fit-local Q提议并用它执行后继。不能用这些fit视频的原OOF Q轨迹，其控制器来自相反折，可能含held-out监督。Hann缓存中的OOF Q／q0critic辅助决定不进入输入或标签；只使用其不依赖Q的物理候选／特征／历史。每fit视频完整32帧可用、Q非q0且实际框不同时，在可用时刻列表均匀取最多32点，不读GT／已知事件、不补挑310等点。整折无支持退出，单视频无支持明确记录。

每点共享4实际搜索，强制接受q0／Q提议后，各自在自身历史上执行31帧本折Q；初始候选观测须复现原Hann，后继逐帧审计raw／size／offset、裁剪、解码和原生块。完成采集后才读fit GT产生两种标签：immediate为当帧IoU差、future为32帧IoU均值差。两个岭模型共享128因果维（64候选差＋64 q0状态）、同一对样本／视频等权／ridge0.01／数值零门槛1e-6，仅目标不同。最多4个新CPU预测器，SGLA权重与原首帧模板不动。

七完整政策为Hann／q0／冻结Q／冻结protect／rate_margin／immediate／future。rate_margin在**全量fit原生Hann帧**上按视频等权匹配future预测的非q0接管率，同分组整体接纳，报告目标／实现率、误差、最大同分权重及calibration_states；不以held-out率调门槛。评价历史自演化，实际率不保证相同。未来与当帧两模型输入／目标外配置匹配；在线只选本帧Q提议或q0，不读未来GT。记录索引接管率与实际改变框比例。4个冻结控制在新L40上须精确复现Stage-66。

最多384个fit点／768短分支／96768搜索，7×12＝84条新完整轨迹／378700搜索，合计最大475468搜索／3328276块；pair smoke252、eval smoke896均包含并复用，额外0。主要future−immediate，关键future−rate_margin，另比Q／protect／Hann／q0；12视频bootstrap种子2168、20000次，6 AUC家族端点0.05/12。pilot需六项家族下界>0，对immediate／Q均值至少+0.5 AUC，原生q0稳定集0新增失败、低IoU不恶化、可靠帧CI下界≥−1.0，以及足够支持；不满足即REVIEW。重复开发组、重叠窗口、Hann训练状态与部署自身状态差异均保留为限制。

**验证完成：** `python -B tools/test_stage68_advantage.py -v` 4项全部通过，167.173秒、无跳过；真实CPU Torch／OpenCV／CENTER、完整封存来源链到本轮，2折fit对采集／模型拟合／14条完整微型轨迹／汇总／续跑通过。微型正式预算pair24搜索（12 smoke包含）＋eval224（112 smoke包含）；验证fit标签只来自本折fit、held-out GT改变不影响本折模型、在线和后继动作不读GT、4原控制精确复现、网络／所有旧来源字节不变、NPZ及结果表篡改拒绝。微型4帧来源本折Q可能无提议，**仅测试的fit路径使用确定性因果Q系数曝光**保证双折分支；原评价Q控制和正式模型均未替换，纯语义测试另核对真实折内Q算式及忽略缓存OOF辅助决定。已测试混合可选CSV字段，避免Stage-67导出问题重现。Bash语法、4份新Python解析、check_workspace（505保留／200移动／106模块、0缓存、0换行差异）、177处所涉本地链接和git diff --check均通过；Stage-67原件SHA未变，tracking／lib／已有tools源码未改。

服务器同步本轮新文件后，在`sglatrack-test`环境、`/home/u25600003160604/workspace/paper`执行`bash tools/run_stage68_advantage.sh`。默认输入`output/stage67/matched_veto_continuation_v1`，新空根`output/stage68/fit_paired_takeover_advantage_v1`，GPU0／原V1 checkpoint。流程prepare→pair_smoke→collect→fit(CPU)→smoke→evaluate→summary。可复用已提交缓存续跑；源码一旦正式prepare即冻结。

后台：`mkdir -p output/stage68`后，`nohup bash tools/run_stage68_advantage.sh > output/stage68/run_stage68.log 2>&1 < /dev/null &`；日志在实验根父目录，勿污染空根。脚本通过cat汇总decision／policy_summary／paired／calibration／fit_summary／pair_support／scope／focus／video_influence为`docs/experiments/06_sgla_uav_tracking/results/stage68_results_<UTC>_<PID>.txt`，回传此TXT即可；fit_pairs、pair_scope和事件／逐视频长表留服务器。

## 1. 接手后首先要知道的结论

用户于 2026-10-29 明确重申：**核心目标是以 SGLA 为基础开展无人机跟踪研究。**主任务保持给定首帧目标框的单目标跟踪，主要基线保持原生 SGLATrack；后续方法应在其上形成可测的改进。既有模型压缩探索继续保留，跟踪性能与资源收益分别验证。文献中的检测方法可提供机制参考，不能据此默认更换任务和主基线。

用户重视明确且有意义的新增价值，不要求“绝对原创”或完全没有先例。可以在已有方法上解决不足、适应新需求、改善成本或重新组织问题，并用对照验证贡献；应如实说明借鉴与改进，避免仅通过更名宣称创新。

截至 Stage-44，**尚未建立经独立外部验证的新压缩方法，也没有成立的部署加速或能耗收益主张**。多个阶段通过了执行/梯度 smoke，但主要方法或机制门槛未通过。不要把 smoke、GT hindsight 空间、同状态候选收益或少数视频改善当作方法成功。

最新 Stage-44 没有建立“删除原生候选结构后，在相同适配预算下仍有稳定性能缺口”这一先决前提；保留结构的适配对照也没有通过保护门槛。后续不在这组结果上搜索阈值、层、步数、学习率或 checkpoint。

此前助手将下一轮默认转成 [Stage-45 视频检测原型](docs/experiments/05_uav_detection/STAGE45_TEMPORAL_SIGNAL_PILOT.md)，这与用户重申的核心任务不一致，**现标记为暂不执行的检测支线**。其代码和 CPU 检查记录保留供参考，不作为当前服务器操作入口；未由助手运行服务器 GPU 实验。现阶段无需为这条支线新增 XS-VID 数据、YOLOFT 权重或检测环境。

后续立项边界见 [SGLA 无人机跟踪研究范围](docs/research/SGLA_UAV_TRACKING_RESEARCH_SCOPE.md)。[Stage-46 原生 SGLA 相机补偿搜索原型](docs/experiments/06_sgla_uav_tracking/STAGE46_CAMERA_SEARCH_PILOT.md)从已消耗的 Stage-27 源组固定选12段完整视频，四组48条轨迹已由用户在服务器完成。原生、普通速度、仅相机补偿、相机加速度AUC分别为63.4512、63.6449、64.8664、60.8608，状态为 `REVIEW_CAMERA_SEARCH_PILOT`；原生框误差0，预算与权重检查通过。

仅相机补偿在`0176`上+48.8111、`0204`上+18.5254，在原稳定`0304`上−50.1173；其余9个视频描述均值约−0.0262。全部背景拟合通过，仍不能区分运动是否适合搜索中心修正。[Stage-46B 搜索偏移缓存时序审计](docs/experiments/06_sgla_uav_tracking/STAGE46B_SEARCH_SHIFT_AUDIT.md)已由用户完成，原代码与产物保留。`0204:664`原生父框中心误差约0.071但输出约1.362；`0304:555`仅相机父框约0.086但输出约0.992，均coverage=1。`0176:158`组合策略相对速度项远大于相机项。`0304:1087`组合输出中心误差约0.065、IoU约0.172，需要宽高与横纵误差而非直接认定中心跳变。三个案例各组均无相对原生的选层序列差异。以上是时序/几何线索，不是唯一因果解释。

[Stage-47 固定历史搜索与响应重放](docs/experiments/06_sgla_uav_tracking/STAGE47_MATCHED_SEARCH_RESPONSE.md)已由用户完成400次独立原生前向，状态 `COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC`。其最直接的新证据是：`0304:555`固定camera_only历史上，不偏移IoU0.7703；约0.22像素速度提议改变实际裁剪后IoU0，GT仍保留但从第1降至第2，浮点映射单改没有造成该损伤。`0304:1087`组合策略中心误差仅0.065，宽度却是GT的4.22倍，离线GT尺寸置于预测中心IoU约0.8524，表明尺寸是另一条问题。`0176:158`各干预均无IoU≥0.5的dense候选；`0204:664`仅相对速度的当帧IoU0.7189，但单帧相机干预仍IoU0，不能用完整相机轨迹收益替代当帧证据。

[Stage-48 原生预测头排序与尺寸稳定性](docs/experiments/06_sgla_uav_tracking/STAGE48_NATIVE_HEAD_STABILITY.md)已由用户报告完成。固定原生骨干／选择器／独立候选块，仅训练原CENTER头，BN缓冲冻结；普通两视图监督、加排序、加尺寸、两者组合四臂。复用Stage-46的12个已消费源组，2折每折6 fit／6 held-out、2种子2132/2133、每视频48个均匀时刻、原生父框及预定±1像素视图、256步，16个正式头及96条新完整轨迹。各臂共享缓存特征与训练预算。此为开发OOF，非新外部验证，也不证明压缩收益。

最新[Stage-48结果复核](docs/experiments/06_sgla_uav_tracking/STAGE48_RESULTS_REVIEW.md)：native／supervised／rank／shape／rank_shape宏AUC为63.4512／60.9638／58.5856／60.8346／61.5648。组合较普通监督+0.6010（CI[−5.5756,+6.4664]），较原生−1.8864，低IoU比例+6.7857个百分点并在2个原稳定视频新增失效。组合对原生9胜3负、中位数+1.6085；3条负收益为`0226`−22.2286、`0304`−27.4272、`0307`−27.5770。后两条四臂两种子均退化；`0226`则rank种子2133、rank_shape种子2132分别掉到约7.17／7.19 AUC，而其普通监督和单尺寸对照约58～59。

组合在`0176`的均值收益来自2133单种子（59.0599 vs 2132的15.9508）。组合较普通监督的总体种子差为−2.3949／+3.5970；正交互主要集中于`0176`与`0304`，不能单独证明稳定协同机制。新增失效的两视频均为`0170`与`0304`，前者组合仅4／6帧低IoU且AUC提高，后者为648／645帧低IoU；原保护条件保持未过，严重度需分开描述。冻结父框探针改善仍未转为闭环收益；需以缓存时序区分中心、尺寸与历史传播，不能仅凭汇总认定因果。明细已收到，不再重复请求；不重跑训练、不改本轮协议。本机仍不持有完整Stage-16～48产物。

[Stage-49冻结轨迹时序诊断](docs/experiments/06_sgla_uav_tracking/STAGE49_HEAD_TRAJECTORY_AUDIT.md)已由用户提供focus与onsets，详细解释见[Stage-49结果复核](docs/experiments/06_sgla_uav_tracking/STAGE49_RESULTS_REVIEW.md)。`0304`有554附近的当帧中心错误及1080附近的尺寸失真两类路径：组合2132在1078输出中心误差仅0.0498，但宽比4.0717、IoU0.1786；2133在1081也类似。`0307`原生398失效、474恢复；适配模型大多385～392首次失效后持续到末尾，组合2132虽471恢复仍在487复发。`0226`各臂首次失效相近，但正常模型329～339恢复，rank2133与组合2132一直未恢复至1860。`0176`获益组合2133仍在160严重失效，240才恢复，收益不是避免最初失跟。`0170`4／6帧失效位于视频末尾并被截断，不能称为已经观察到恢复的短暂波动。

新增实现核对：Stage-48普通回归框来自raw最高响应位置，原生实际输出来自Hann加权最高位置，额外尺寸约束在GT格点；三者可能不同。[Stage-50共同父框×冻结头重放](docs/experiments/06_sgla_uav_tracking/STAGE50_SHARED_PARENT_HEAD_REPLAY.md)固定95个输入，每输入一次原生完整前向，再由原生头及所在held-out折8个训练头共享GPU特征；协议预算95次完整前向＋760次仅头前向，smoke的15个输入复用。4项CPU测试通过；用户已提供crossed_matrix与focus两张表。本机未核验服务器95个完整输入的缓存、diagnostic及全部调用计数。

最新[Stage-50结果复核](docs/experiments/06_sgla_uav_tracking/STAGE50_RESULTS_REVIEW.md)：大多数关注输入跨头差异小，换历史后共同成功或失败。0204:664在原生／组合2133历史上9头全为0，在组合2132历史上9头均≥0.7343；0226:338失败历史上各头均≤0.0103，正常历史上均≥0.5063；0176:240成功历史上各头均≥0.5623，原生历史coverage=0、所有头IoU0。0304:554的rank2132历史是明确例外：原生头0.6096、8个适配头全0，不能全归因于历史。0304两起晚期尺寸失真样本的Hann与GT格点相同；0204成功历史上Hann反而将raw IoU0纠正为0.7479，故单纯监督格点不一致不宜列为当前首要解释。Stage-1～50实验Python源码及既有成绩保持冻结。

[Stage-51拟合组预测状态匹配训练](docs/experiments/06_sgla_uav_tracking/STAGE51_MATCHED_STATE_SOURCE.md)已由用户报告完成。保留原生SGLA结构，只用普通监督训练原头；`native_replay`采用两份相同原生视图，`state_mix`采用同一原生视图＋固定fit内监督模型的预测父框。±1像素旧视图仅用于来源审计与分布参考，不进入优化。每折固定Stage-48 supervised_seed2132，仅在该折fit组生成12条新增轨迹，禁止复用互补折的held-out轨迹作训练状态。每视频48个均匀时刻，额外576次原生特征前向，保留全部出裁剪／失效样本；原始头初始化、两视图、抽样与256步预算匹配。新训练8个头，评价48条完整OOF轨迹；固定离线donor混合不是新模型迭代on-policy训练。4项真实CPU测试通过；本机未持有完整服务器产物，不假称已重新核验全部预算与哈希。

最新[Stage-51结果复核](docs/experiments/06_sgla_uav_tracking/STAGE51_RESULTS_REVIEW.md)：native／source_supervised／native_replay／state_mix AUC为63.4512／60.9638／60.5504／61.9121。混合相对匹配控制+1.3618，CI[−4.2677,+7.3063]，7胜5负、中位数仅+0.0642；0226贡献+25.7565、0204贡献+11.3670，0176丢失控制臂的恢复收益−21.2104。其余9视频均值仅+0.0475。混合对原生仍−1.5391，低IoU+5.8167个百分点、可靠帧AUC−2.8875，原稳定视频新增失效仅0304。共同适配损伤0304／0307对原生分别−27.3347／−28.3895，均有末尾未恢复。对旧监督失败队列的+8.0669约74%净均值来自0170末尾少量帧的能力保持，不能当作长时恢复增强。两折各282/288裁剪有实际新暴露；fold0无GT中心出裁剪，fold1有68/288、中心误差P95为42.43。按冻结代码重建，fold1 held-out反而含0226／0204主要收益，不能直接认定其极端训练状态有害。五张表已收到，不重复请求。下一轮可考虑固定状态池的响应／几何分支更新对照，重点保持0304／0307能力，并保留0176／0226／0204恢复取舍；具体新种子／事件仍未提供，不凭AUC猜时刻。Stage-1～51代码与结果保持冻结。

Stage-46用户报告的标识：plan `445dc1c25a5b48662f8d796c52cf4ff4a828ee40b48ddfb485aa692fbbbd0015`，run `4c58e05cf3b15cc304922ef16336015d6ca42cd0737def4f57028f1aced71fa6`，decision `8ba76d6de665755786f6fd70a347f7a64f91d7d96686c67132a52c239e61cba0`。

本文为当前状态入口。`docs/archive/manuscript/` 的旧组会、Introduction 和 novelty review 只反映历史设想，不应直接用于当前论文结论。

[Stage-52固定状态池分支适配](docs/experiments/06_sgla_uav_tracking/STAGE52_FIXED_STATE_BRANCH_ADAPTATION.md)已由用户报告完成8个正式模型、48条OOF轨迹；3项CPU测试已通过。[结果复核](docs/experiments/06_sgla_uav_tracking/STAGE52_RESULTS_REVIEW.md)：8条scope检查全部通过；native／response_only／geometry_only／joint AUC63.4512／59.3268／63.6819／61.9121。几何组对原生仅+0.2307，低IoU+3.2776个百分点、可靠帧−2.9471且有新增失效，不能直接采用。0304响应组554／555失效至末尾，几何组均1083失效至末尾；0307仅响应组保留451／442恢复，几何及joint持续失效。0176几何组高均值来自2133（61.2903），该种子160首次失效、240恢复后仍有末尾未恢复。0204几何与joint同为2132失效、2133不失效；0226两组均正常恢复。两个全局固定更新范围不能同时保护全部关注视频；下一步可检验有实际梯度路径的原生行为保持或受约束适配。五张表已收齐；几何对原生10胜2负，中位数+2.6419；对joint仅3胜9负、均值收益集中0176；几何因子最小留一+0.3498，但区间仍跨零。本机未独立重验完整服务器产物。Stage-1～52代码及结果保持冻结。

[Stage-53原生几何保持](docs/experiments/06_sgla_uav_tracking/STAGE53_NATIVE_GEOMETRY_PRESERVATION.md)已由用户报告REVIEW，详见[结果复核](docs/experiments/06_sgla_uav_tracking/STAGE53_RESULTS_REVIEW.md)。scope冻结／梯度／抽样检查通过，资格暴露约69%～84%，有条件平均权重0.971～1.001。native／geometry_control／anchor_all／anchor_reliable AUC为63.4512／63.6819／61.8208／61.9882；新两组对控制−1.8612／−1.6937，可靠帧继续下降，均仍有0304新增失效。原10条获益队列仍正收益10／9条，但收益幅度对控制−1.8601／−1.9386；这部分算术贡献解释整体下降的约83%／95%。0307全状态保持两个种子均恢复，0176高分种子首次失效推迟且无末尾未恢复；与此同时0304全状态保持两种子均555失效至末尾，有条件组一早一晚，0204原无失效的2133高分分支丢失。响应参数冻结仍可早期失效，不能仅凭时刻归因某个塔。四表已收齐，不重复请求；paired区间及完整服务器产物未在本机核验。单状态单格点保持未达目标，Stage-1～53代码／结果冻结。

[Stage-54实时两帧自身状态与末帧保持](docs/experiments/06_sgla_uav_tracking/STAGE54_SHORT_ROLLOUT_ENDPOINT.md)已由用户报告完成。沿用原生SGLA，冻结响应与BN，只训练size／offset；每步从一个fit时刻的原生及fit-only donor预测父框各自推进两帧，当前训练头的实际第一帧输出作为第二帧父框。`rollout_supervised`用普通监督，`rollout_preserve`再保持另一条原生短分支的末帧图像框，资格要求原生两帧IoU均≥0.5且教师末帧中心可见；GT只用于推进后fit损失。父框、取整和argmax均截断，保持梯度只进入末帧几何；不是完整BPTT。每步4个自身监督视图，256步共1,024视图；正式8模型／48条新OOF，复用60条参考轨迹。scope报告两组调用相同，共16,384次完整前向、114,688个block及8,192次仅头前向，smoke另计。旧静态控制的训练状态安排不同，主要匹配对比固定为两个新组。3项真实CPU测试通过，包括当前后继父框、future GT动作隔离、原生解码／梯度、fit隔离、来源篡改、AdamW续训和完整48条OOF合成流程。

最新[Stage-54结果复核](docs/experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md)：六表已齐，scope冻结、解码与预算读数符合协议，90.82%～96.09%后继取整裁剪实际变化。native／geometry_control／static_anchor／rollout_supervised／rollout_preserve AUC为63.4512／63.6819／61.9882／63.2814／64.0079。保持对匹配普通监督+0.7265、低IoU−2.8385个百分点、可靠帧+1.6464，但对原生仍低IoU+0.3381个百分点、可靠帧−3.5250且0304新增长期失效。原10条获益视频均保留正收益，平均+6.0741，略高于旧几何控制+5.8861，但比匹配普通监督的+7.4575下降1.3835。主比较的0304／0307贡献+1.8794，被其余10条−1.1529部分抵消；分别移除0304／0307后主均值约−0.2062／−0.2589；相对原生移除0176后约−3.3896。0307两种子主比较+14.8817／+8.2502，0304主要只改善2132（554失效后移至1078，仍末尾未恢复），0204丢失普通监督2132的无失效高分分支−23.0495。0176新两组均有双种子收益，保持进一步改善但没有避免最初失效。新增paired／video表确认主比较4胜8负、中位数−0.4797、CI[−2.3364,+3.9123]，获益为0176／0304／0307／0147；对旧几何控制3胜9负，对静态保持12胜0负、CI[+0.2479,+5.0494]、最小留一+0.5753。普通短分支自身对静态也有+1.2932、10胜2负，不能把整体12/12归因末帧损失。0170是普通监督额外新增的短事件视频，保持消除该事件；静态丢失正收益的视频为0147，本轮恢复。可靠帧对原生9胜3负，严重损伤集中0304／0307，另有0176小幅下降。四项AUC区间从用户12视频数据按固定种子bootstrap重算一致，完整原始产物仍未在本机重验，总体种子差未提供。后续优先共同父框与后继诊断，不重跑旧训练。Stage-1～54代码与结果冻结。

[Stage-55共同父框与32帧后继诊断](docs/experiments/06_sgla_uav_tracking/STAGE55_MATCHED_PARENT_CONTINUATION.md)已由用户报告完整诊断完成。固定0204:664、0304:554／1078、0307:389／471共5个已消费锚点，逐项核对来源事件。每个锚点−1／0／+1帧交叉5历史×5头，共75个共享原生GPU特征输入及300次仅头前向；另在锚点从5个预测父框各由5个冻结头推进32帧，共125条分支、4,000次完整前向。完整预算4,075次网络、300次仅头、28,525个block、4,375次head；smoke的175次网络／60次仅头缓存直接复用。每个自身头×自身历史32帧必须复现完整来源轨迹；首帧还核对共享特征探针，后续仅使用自身实际输出父框。固定响应、几何、覆盖和实际裁剪分岔分别记录；当帧与窗口末帧／平均IoU分解仅为条件记账，窗口IoU不是全视频AUC。新增训练为0，没有在线策略采用门槛。完整Stage-54来源108条种子指标、60条视频及5条宏指标重算后绑定。3项真实CPU测试通过（90.476秒，无跳过），覆盖真实CENTER、来源到本轮全流程、缓存复用、130／120次小样本预算、后续GT隔离和篡改拒绝。

最新[Stage-55结果复核](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)：三表已齐，10条focus两条当帧及后继记账路径相符。0204:664 S2132父框五头IoU均≥0.6471，其他父框均0且coverage=1；2132自身32帧差−0.7977，在共同S起点换头仅−0.0116，主要差随进入窗口前状态变化。0304:554 P2132父框五头均≥0.7457、S两种子／P2133父框全0；2132当帧+0.7753、32帧+0.8010均落在指定分解起点项，当前好起点上换头作用很小。1078的P2132父框仍coverage=1，但五头输出均约0.187～0.193，32帧自身差仅+0.0430、末帧差0；共同native起点两种子保持头均多4个低IoU帧。0307:389共同起点均值差皆小正；471恢复附近，同S起点换保持头的32帧均值差2132为+0.1545、2133为+0.3620，2133末帧+0.6601，当帧仅+0.0163。2132五起点均获益，2133四正一负（S2132起点−0.1884），收益状态依赖。2133自身均值差仍−0.1732，起点项−0.5351抵消共同S起点模型收益；窗口止502，Stage-54记录的511恢复在窗口外。50条比较均当帧框分岔、下一帧裁剪／响应分岔，IoU差0不等于没有干预。不能将全部效果归因当帧头能力或全部归因既有历史；下一轮优先研究分岔前状态保护和恢复阶段的稳健推进，不预设中心／尺寸唯一机制。Stage-1～55代码及结果冻结，Stage-56也已报告REVIEW，见第0.6节，不重复请求已有表。

## 2. 目录与文件定位

| 路径 | 用途与保留理由 |
|---|---|
| `README.md` | 原 SGLATrack 项目说明及本项目入口；原论文的成绩不是后续实验成绩 |
| `WORK_HANDOFF.md` | 当前研究、结论、环境与交接说明 |
| `docs/README.md` | 全部阶段说明的分类索引 |
| `tracking/` | 当前保留的运行、评估、几何、损失和测试依赖；至Stage-61，L40重测和Stage-62缓存诊断入口在tools |
| `lib/`、`experiments/` | 原模型、数据接口、训练/推理实现与 YAML 配置 |
| `environment_sgla1.yml` | 原环境说明；实际服务器环境另见下节 |
| `output/checkpoints/` | 原始 SGLA 权重，是后续初始化、来源校验与复现的基础 |
| `output/test/tracking_results/` | 原始 UAV123 预测与时间文件，保留用于复核 |
| `output/stage4/`、`output/stage5/` | 本机已有早期 manifest/元数据证据，保持路径与字节不变 |
| `archive/legacy_tracking/` | 当前依赖链外的 136 个历史 Python 脚本/测试；需恢复后运行 |
| `docs/archive/manuscript/` | 3 份旧汇报/论文构想，不是当前已成立主张 |
| `tools/` | 整理后校验与历史代码恢复工具 |
| `docs/maintenance/` | 整理清单、逐文件 SHA256、移动与删除原因 |
| `STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json` | 历史预声明协议，保留原位置与内容 |

实验说明已从仓库根目录、`tracking/` 集中到 `docs/experiments/`。代码块仍应在仓库根目录执行。

没有改写冻结的实验 Python 源码。早期阶段的公共文件即使不再是新方法入口，也可能参与导入或代码哈希校验，不能按阶段编号直接删除。

## 3. 本机和服务器的边界

### 本机

- 工作区：`E:\Program Files\paper\paper`，PowerShell。
- 本机 `output/` 约 99 MiB，主要是原始权重、UAV123 原始预测及 Stage-4/5 元数据。
- **本机没有完整 Stage-16～59实验产物。**后文近期数值来自用户在会话中提供的服务器终端输出/附件，不能假称已在本机对全部原始结果文件重新核验。
- 原始权重：`output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar`。
- 本次整理核对的本机权重 SHA256：`88447c7724704edaf03488f519d3e300385fef05c2f1fac02fe22f7941cf334e`。服务器运行仍以其原始 manifest 中的权重哈希为准。

### 用户服务器

- 项目：`/home/u25600003160604/workspace/paper`。
- 环境：`sglatrack-test`，Python 3.10；用户报告 Torch `2.7.1+cu118`。
- GPU：NVIDIA L20，驱动 `550.54.14`，约 46 GiB。
- 报告过的物理 UUID：`GPU-32d96966-117c-354f-01f2-b2363a738210`，换设备/容器后不能沿用其身份假设。
- VisDrone-SOT 根目录：`/home/u25600003160604/workspace/datasets/VisDrone-SOT`，训练集子目录为 `VisDrone2019-SOT-train`。
- `lib/test/evaluation/local.py` 当前包含该用户服务器的绝对路径；`lib/train/admin/local.py` 还含原项目旧路径。不要在冻结来源上随意重写配置；新环境先核对 manifest、配置及 checkpoint 解析。

本次目录整理只作用于本地工作树，没有登录服务器、删除服务器数据或重新运行 GPU 实验。

## 4. 模型、指标和评估口径

- **Fast / 六层模型**：L1–L6 后接预测头。后期实验物理删除候选块和选择器。
- **原生 SGLA**：L1–L6 后，由原选择器挑选 L7–L12 中一个独立候选块执行，再预测。
- 原生 SGLA 保存候选结构，但通常每次执行 **6+1 个块**，不是串行十二层 ViT。把 L7–L12 连续执行称为原生 SGLA 是错误的。
- Stage-34～44后期适配主要只训练L5–L6，其他前缀、候选、选择器、归一化与头冻结；这不是完整模型充分训练后的能力比较。Stage-48改为仅训练原CENTER头，骨干与候选选择结构保持冻结。
- `tracking/validation_metrics.py` 与阶段公共模块定义官方框/IoU口径；当前 AUC 使用离散成功阈值，不能把单状态平均 IoU 差值直接当成 AUC 百分点。
- **同状态诊断**：同一父框/裁剪上比较输出；不能代替各自闭环轨迹。
- **完整闭环**：首帧初始化后，每个策略用自身预测生成下一帧搜索裁剪。
- **GT 干预/上界**：只用于离线机制诊断。GT 选候选、GT 失败时刻、另一策略的父框不是可部署控制器输入。
- GT 在搜索裁剪内，不等于目标可见；低 IoU 框不自动代表语义干扰物。
- 视频/源组是统计单位。先在视频内平均种子/锚点，再 bootstrap；种子、帧、分支不是额外独立视频。重叠 OOF 拟合的区间仍是条件开发证据。

## 5. 阶段性证据总览

数字来自会话中已提供材料；详细设计在文档索引中。不同阶段的数据、窗口和模型不同，AUC 不能跨表直接排序。

| 阶段/路线 | 已知结果 | 后续意义 |
|---|---|---|
| Stage-1～15：层路由、时序预算、反事实诊断 | 实现了 Fast/独立候选/闭环与预算工具；部分早期研究设想只见于历史稿 | 用作工具和历史证据，不当作当前成功方法 |
| Stage-16B/C：离线 Q 到闭环 | 16B 离线筛查通过；16C 最终为 `FAIL_CLOSED_LOOP_GAIN`。TAVRTrack-P30 AUC 84.0203，Uniform-L9-P30 84.4904，SGLA 85.5591 | 离线价值相关性没有转化为闭环优势；早期 integrity 状态曾被后续审计更正，不能混用 |
| Stage-17：安全残差优势 | `FAIL_TRAIN_OOF_RESIDUAL_SCREEN`。30%预算对 Uniform-L9 lift 为 −0.001834，候选层活跃数仅 2 | 停止该状态表示，不继续搜阈值 |
| Stage-18：记忆表示＋PPO | 表示/优化 smoke 通过；PPO seed2054/2055/2056 最佳 tune AUC 83.0071/83.3869/82.9256，SGLA 对照 84.8323 | 未超过 SGLA；2056 是中断后的部分训练结果，不能写成完整训练 |
| Stage-19：持续压缩损害反馈 | 24 clips 完成；`MECHANISM_NOT_ESTABLISHED` | 闭环损害选择准则未建立 |
| Stage-20：固定压缩学生＋纠错 KD | `PASS_MATCHED_TRAINING_SMOKE`；小 smoke 中纠错交互为正 | 只有 smoke，不是充分验证的 KD 方法 |
| Stage-22：单 GPU 设备状态 | L20 在 33.33ms 截止下无有意义 deadline 压力 | 按规则停止资源感知 PPO；不人为收紧时限制造压力 |
| Stage-23/24：长时信用 PPO 与部署动作审计 | `NO_CREDIT_EFFECT_ESTABLISHED`；H1 三种子退化为完全相同的 Static-L9-P30 行为 | 不把同分数解释为独立鲁棒成功；动作审计很重要 |
| Stage-25：关系状态 probe | `FAIL_RELATIONAL_STATE_SCREEN`，GT hindsight 有空间，学习状态未优于 raw/全部静态对照 | 上界不等于可学习价值 |
| Stage-26：干预价值对齐表示 | smoke 通过；32 train/16 checking/3 seeds pilot 为 `FAIL_VALUE_REPRESENTATION_PILOT` | 对齐表示与闭环保护机制均未成立 |
| Stage-27：VisDrone 失败定位 | 数据审计 86 视频、69,941 图像、几何异常 0；选24视频/27,887跟踪帧，宏 AUC 63.3250，11失败事件，10次起点 coverage>=0.9 | 原始小尺度、搜索缩放与错误排序需要区分，不能把出裁剪当作初因 |
| Stage-28：响应竞争 | 11个起点中，仅3个正确目标留在 weighted Top-5 且不是Top-1；另有5个不存在匹配GT的 dense框 | 不能只做候选重排来处理所有失败 |
| Stage-29/30：一次纠正与复发 | GT选择起点候选的持续改善门槛仅1/3通过；另两条纠正后失败点目标仍在Top-5 | 一次输出正确不保证持续跟踪，属于离线机制线索 |
| Stage-31/32：GT-free身份关联与归因 | identity 6 checking视频 AUC65.9168，SGLA66.0447；低 IoU与非平凡使用门槛失败。训练大多数pair是容易样本，难状态信号较弱 | 简单外观/历史/线性排序未建立闭环价值 |
| Stage-33：pooled descriptor probe | 设计与代码仍有保留依赖；本交接不补造附件中未完整复录的数值 | 需要精确结果时读取服务器原decision |
| Stage-34：空间身份表示 | `NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED`；contrast表示较standard +1.1163 AUC但CI跨零，较原SGLA低1.1722 | 无稳定空间表示收益 |
| Stage-35/35B：保留＋纠错 OOF | preserve_correct64.0169，普通supervised68.6324；`NO_PRESERVE_CORRECT_GAIN_ESTABLISHED`。普通监督均值收益也被少数视频主导，存在正常片段损失 | 不把普通适配结果当作可靠普遍改进 |
| Stage-36/B/C：同裁剪及坐标跳变审计 | `0089` 两种子失效前后 peak136→69，grid项解释大部分中心误差跳变，GT仍Top-5 | 是两个事后案例，不能推成所有失败都由响应头导致 |
| Stage-37/38：响应/几何交叉闭环 | teacher_response 相对学生 −2.1361；teacher_geometry +1.3741但CI跨零；单独size −2.5727、offset +0.2228均无稳定收益 | 不能直接用教师响应、size或offset替换作为成立方法 |
| Stage-39：匹配父框扰动 | `NO_EXCESS_SENSITIVITY_ESTABLISHED`；Fast相对SGLA敏感性excess −0.006447，CI跨零 | 六层压缩更敏感这一前提未建立 |
| Stage-40/41：匹配裁剪增强及转移审计 | 增强AUC68.0144 vs clean68.6324，差−0.6179；`NO_MATCHED_AUGMENTATION_GAIN_ESTABLISHED`。扰动损失减小主要对应未扰动项下降，绝对扰动表现无平均提升 | 普通增强没有建立压缩稳定机制 |
| Stage-42：冻结模型×父框交换 | 92起点/368分支，87完整后续窗口；增强失败后+10帧换一次父框，4支持视频平均后40帧IoU +0.1441；当帧及固定时刻无稳定收益 | 条件性状态效应，GT时刻/另一模型父框不可部署，不能变成在线收益主张 |
| Stage-43：GT-free历史恢复 | `NO_GT_FREE_RECOVERY_GAIN_ESTABLISHED`。augmented history相对native −1.2926；clean −6.1104；局部搜索对augmented近乎持平 | 当前运动/响应门控恢复停止，不继续调阈值 |
| Stage-44：匹配结构适配 | `MATCHED_RETAINED_CONTROL_DEGRADED`，压缩缺口未建立；详见下一节 | 研究前提未通过，不自动展开新损失/恢复模块 |
| Stage-46：原生SGLA搜索运动 | `REVIEW_CAMERA_SEARCH_PILOT`；仅相机补偿均值+1.4153点但受单视频收益/损伤主导，相机加速度较原生−2.5904点 | 以Stage-46B定位偏移、裁剪与失败的时序，暂未采用新策略 |
| Stage-47：固定历史搜索与响应重放 | 用户报告400次完成；E03微小实际裁剪变化令保留GT从第1降至第2，E04宽度达GT的4.22倍 | 支持分别检验选择间隔与尺寸校准，不是全视频训练收益 |
| Stage-48：原生预测头稳定性 | `REVIEW_NATIVE_HEAD_STABILITY_PILOT`；组合61.5648 vs 原生63.4512，低IoU与稳定视频保护未过；固定父框探针改善却未转成闭环收益 | 先定位3条受损视频及普通头适配已有损伤；不以正交互认定方法成立 |
| Stage-49：冻结轨迹时序诊断 | 用户已提供focus／onsets；0304起点有中心／尺寸两类，0226与0176主要在恢复分岔，0170末尾截断 | 区分当帧输出与后续恢复；下一步共同父框重放，不新增方法成绩 |
| Stage-50：共同父框×冻结头重放 | 用户提供25行交叉矩阵及focus；多数差异随历史变化，0304:554仍有原生／适配头当前能力差异 | 优先检验自身状态训练覆盖；晚期Hann=GT仍尺寸失真，格点不一致不足以解释 |
| Stage-51：拟合组预测状态匹配训练 | `REVIEW_FIT_STATE_SOURCE_PILOT`；0226／0204获益、0176丢失恢复收益，0304／0307共同重损伤仍在；状态暴露充分但两折难度不对称 | 0170末尾保护有改进；下一轮优先研究共同适配损伤与能力保持，保留恢复取舍对照 |
| Stage-52：固定状态池分支适配 | `COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC`；五表齐；几何对原生10胜2负但宏均值+0.2307、CI跨零，对joint仅3胜9负 | 保留多数视频适配收益、控制0304／0307长期退化；几何因子最小留一为正，仍需新保护机制对照 |
| Stage-53：原生几何保持 | `REVIEW_NATIVE_GEOMETRY_PRESERVATION_PILOT`；约束有梯度，但两组均低于控制；0307／0176改善伴随0304／0204损伤 | 多数收益幅度下降；后续检验跨状态保持，保留当前失效路径与恢复取舍，非简单重复调系数 |
| Stage-54：两帧自身状态末帧保持 | `REVIEW_SHORT_ROLLOUT_ENDPOINT_PILOT`；对静态12胜0负，对匹配普通监督4胜8负、CI跨零 | 整体相对静态进步，末帧独立增益仍有取舍；0307／0304改善伴随0204损伤，可靠帧／新增失效保护未过 |
| Stage-55：共同父框×冻结头×32帧后继 | `COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`；三张关注表已齐，无新增训练 | 0204／0304当帧差随起点变化；0307恢复附近有共同起点模型收益及状态依赖损伤，无方法采用门槛 |

## 6. Stage-44 压缩前提的精确记录

### 四个角点

| 方法 | AUC | 宏平均低 IoU 比例 | 平均失败事件数 |
|---|---:|---:|---:|
| six_initial | 65.274498 | 8.478679% | 1.2500 |
| retained_initial | 67.392394 | 7.893546% | 1.1250 |
| six_adapted | 68.632370 | 4.952532% | 1.1250 |
| retained_adapted | 66.702954 | 9.740018% | 1.0000 |

- 数据：8个已反复使用的开发源组，4折、每折6 fit/2 held-out，种子2110/2111。
- 六层适配模型复用 Stage-40 clean；本轮仅新训练8个保留结构模型，16条新完整轨迹。
- 两结构仅L5–L6可训练；相同256更新、batch2、两张相同clean视图、AdamW LR1e-4/WD1e-4/clip1；训练父框来自冻结SGLA。
- 主差值 retained_adapted−six_adapted：**−1.92941561**，95% CI **[−6.41407660,+1.18354074]**，5赢/0平/3负。
- 两种子主差值：**−3.35760126 / −0.50122996**。
- retained_adapted−retained_initial：**−0.68944013**，CI **[−11.67736886,+9.35055744]**；4赢4负。保护门槛未过，不等于显著普遍下降。
- `0089` 主差值 **−16.51311249**，去掉该视频其余平均 **+0.15396966**；不能说六层普遍更强，也不能说保留结构有稳定优势。
- 保留适配模型全部13,292次输出选择 **L9**，其他候选均0次。这是选择器输出，不是脚本手动固定L9；本轮没有观察到动态切层。
- 实际头调用13,292，块调用93,044＝7×13,292。最大dense解码差3.05176e-5。调用审计不能替代真实性能/能耗测试。

用户提供的原结果标识（本机未持有原完整文件用于重新核验）：

```text
plan_fingerprint: 51f142d6a73027b991b8eea144fde863eba54b348fe2ccc77e01dcbc560f8cef
run_fingerprint: 0f9846c567645671a8594bc5b7ca2404521399f681801790d96a81e6b3a5d602
decision_fingerprint: b393b7f9a835b36ac264fbc772659596369da6086d4432dbd6fa0c689149baea
```

### 应保留的解释边界

1. 实验按固定规则完成；“保留结构对照退化”不是运行报错，不应丢弃此结果。
2. 该状态是预定保护条件的判断，不是模型结构能力的普遍统计结论。
3. 只适配L5–L6不能代表充分训练的架构最优能力；候选和选择器冻结的兼容性仍未分离。
4. 原始结构差值、适配后结构差值与交互区间均跨零，尚不能确立待修复的稳定压缩缺口。
5. 不应基于这些8个视频继续调层、训练步数或挑选checkpoint后称作新独立验证。

## 7. 服务器结果位置与冻结依赖

以下均为服务器项目根目录下相对路径；旧协议中还可能嵌入绝对路径。后续阶段读取前序manifest、decision、checkpoint、receipt和代码SHA，因此不能只保留最后一份CSV。

| 阶段 | 目录 |
|---|---|
| Stage-18 | `output/stage18/temporal_layer_policy_v1/` |
| Stage-23B | `output/stage23/`，精确子目录以原环境变量/manifest为准 |
| Stage-27 | `output/stage27/uav_failure_diagnostic_v1/` |
| Stage-28 | `output/stage28/response_competition_v1/` |
| Stage-29 | `output/stage29/`，读取原manifest确认子目录 |
| Stage-30 | `output/stage30/`，读取原manifest确认子目录 |
| Stage-31/34/35 | `output/stage31/`、`output/stage34/`、`output/stage35/`，保留完整源链 |
| Stage-36 | `output/stage36/same_crop_mechanism_v1/` |
| Stage-37 | `output/stage37/frozen_crossed_closed_loop_v1/` |
| Stage-38 | `output/stage38/fixed_response_geometry_v1/` |
| Stage-39 | `output/stage39/matched_parent_perturbation_v1/` |
| Stage-40 | `output/stage40/matched_crop_augmentation_v1/` |
| Stage-41 | `output/stage41/cached_crop_transfer_audit_v1/` |
| Stage-42 | `output/stage42/frozen_model_parent_swap_v1/` |
| Stage-43 | `output/stage43/online_history_recovery_v1/` |
| Stage-44 | `output/stage44/matched_structure_adaptation_v1/` |
| Stage-46 | `output/stage46/sgla_camera_search_v1/` |
| Stage-46B | Stage-46下`analysis/search_shift_audit_v1/`，用户已报告完成 |
| Stage-47 | `output/stage47/matched_search_response_v1/`，用户已报告完成 |
| Stage-48 | `output/stage48/native_head_stability_v1/`，用户已提供完整汇总及逐视频／种子明细；下一步可作缓存时序诊断 |
| Stage-49 | Stage-48下`analysis/trajectory_audit_v1/`，用户已提供focus／onsets；其余产物仍在服务器 |
| Stage-50 | `output/stage50/shared_parent_frozen_heads_v1/`，用户已提供crossed_matrix／focus完整文本；其余缓存与诊断报告在服务器，本机未核验 |
| Stage-51 | `output/stage51/fit_only_state_source_v1/`，用户已提供五张汇总／支持／逐视频表；完整缓存、逐样本exposure、seed／events与decision未在本机核验 |
| Stage-52 | `output/stage52/fixed_state_branch_adaptation_v1/`，用户已提供scope／policy／focus／paired／factor；完整轨迹／模型／diagnostic未在本机核验 |
| Stage-53 | `output/stage53/native_geometry_preservation_v1/`，用户已提供scope／policy／retention／focus；paired、完整轨迹／模型／decision未在本机核验 |
| Stage-54 | `output/stage54/short_rollout_endpoint_v1/`，六张结果表已收到；正式8模型／48条新OOF，训练16,384次完整前向＋8,192次仅头前向；完整服务器模型／轨迹／decision未在本机核验 |
| Stage-55 | `output/stage55/matched_parent_continuation_v1/`，已提供focus／crossed_matrix／branch_contrasts三表；协议75输入／125分支／4,075次完整＋300次仅头，完整缓存与diagnostic未在本机重验 |
| Stage-56 | `output/stage56/first_transition_preservation_v1/`，REVIEW，五份材料已齐；12头／72条新OOF，完整模型及轨迹未在本机重验 |
| Stage-57 | `output/stage57/quality_selective_endpoint_v1/`，REVIEW，五份材料已齐；12头／72条新OOF；材料内部校验通过，完整GPU产物未在本机重验 |
| Stage-58 | `output/stage58/projected_endpoint_gradient_v1/`，REVIEW，五份材料已齐；12头／72条新OOF，投影曝光通过但主要／保护未过，完整产物未在本机重验 |

Stage-44沿Stage-43→42→41→40继续校验更早来源。单独复制一个末端结果目录到新路径，不能保证可复核；不要改原指纹来绕过路径/来源错误。

Stage-18 的 seed2056 曾在146,144步中断，导出 `partial_best_at_146144.pt`，最佳tune点140,000；2054/2055已到250,000。用户说明重新登录后服务器数据仍保留。它是历史部分结果，当前没有补训授权。

Git主要管理代码与协议。当前`.gitignore`延续忽略`*.csv`和`*.npz`，因此**Git仓库不等于服务器实验产物备份**。需要交接原数据时，完整保留manifest、decision、evidence、checkpoint与其收据、轨迹/缓存及数据文件和路径。

## 8. 接手操作

### 整理后的本机校验

在仓库根目录执行：

```bash
python -B tools/check_workspace.py
python -B tools/test_workspace_maintenance.py
python -B tracking/stage44_matched_experiment.py --help
```

`check_workspace.py`核对本次整理快照、移动归档内容和活动依赖，不代表服务器GPU结果校验。清单保留原始字节SHA，并为UTF-8文本另存仅将CRLF转LF的SHA；Windows/Linux检出的纯换行差异单列为`line_ending_only_differences`，不会被误报为内容改动。权重/二进制仍严格逐字节核对，实验自身的`file_hash`和指纹校验未修改。将来主动修改README或其他被快照文件时，应明确维护该快照，不把正常变更误认为科研结果被篡改。

### 需要恢复历史PPO/路由实验时

```bash
python -B tools/restore_legacy_tracking.py
python -B tools/restore_legacy_tracking.py --apply
```

第一条只预览；第二条把全部136个归档脚本复制回原`tracking/`路径，保留归档，不覆盖内容不同的现有文件。恢复后再使用相应历史说明。不要直接在`archive/legacy_tracking/`中运行旧命令。

### 当前服务器入口与历史结果

Stage-52五表、Stage-53四表、Stage-54六表已收齐，最新结论见[Stage-54复核](docs/experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md)。不重跑旧训练或重复请求现有表。[Stage-54协议](docs/experiments/06_sgla_uav_tracking/STAGE54_SHORT_ROLLOUT_ENDPOINT.md)保留历史复现入口。paired／video_summary也已收到，并用固定种子从12视频AUC重算主要区间、胜负及留一，全部相符；不再重复索取。当前已有结果足够判断主要统计条件及能力保护未通过，同时保留对旧静态方案12/12改善的开发价值。

Stage-55 focus／crossed_matrix／branch_contrasts也已收到，最新结论见[Stage-55复核](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)。已有三表足够作当前机制判断，不重复索取或重跑采集。[Stage-55协议](docs/experiments/06_sgla_uav_tracking/STAGE55_MATCHED_PARENT_CONTINUATION.md)保留历史复现入口。精确中心／宽高／选点归因可使用既有point_details及timelines，当前没有追加采集或新训练。Stage-56也已报告REVIEW并提供五份材料，详见最新复核；Stage-57也已报告REVIEW并提供五份材料，Stage-58也已报告REVIEW并提供五份材料，结果见第0.8节；Stage-59已完成父框分量交叉诊断，当前结果与边界见第0.9节。

以下仅用于复核Stage-44历史压缩对照（无新训练）：

```bash
cd ~/workspace/paper
export STAGE44_ROOT="$PWD/output/stage44/matched_structure_adaptation_v1"

python tracking/stage44_matched_experiment.py --input "$STAGE44_ROOT" --mode status

cat "$STAGE44_ROOT/analysis/stage44_decision.json"
cat "$STAGE44_ROOT/analysis/stage44_policy_summary.csv"
cat "$STAGE44_ROOT/analysis/stage44_paired.csv"
cat "$STAGE44_ROOT/analysis/stage44_video_influence.csv"
```

不要重复启动已经完成的训练。确需恢复既有中断运行时，重跑原stage、原root、原seed指令；训练从已提交参数/Adam状态继续，视频级推理的未提交视频重跑。

### 依赖与路径常见问题

- 所有默认`$PWD/output/...`都应先`cd ~/workspace/paper`再设置；过期变量曾误指`~/workspace/output`。
- `CHECKPOINT`应为原SGLA权重，不是目录或学生模型。解析顺序通常为显式参数、环境变量、原tracker配置，并校验SHA。
- `STAGE18_REP_BUNDLE`的有效冻结划分为95 fit/24 tune，不能强行修成96/24。
- 旧Stage-17曾有NumPy高级索引转置错误，已修复；Stage-31的float32响应审计已做精度兼容；不应再次放宽全部特征容差。
- 当前轻量命令帮助/几何工具采用延迟Torch导入；真实SGLA运行仍需服务器CUDA环境。

## 9. 验证记录与未完成事项

前轮本机CPU测试曾通过 Stage-43 的4项测试和 Stage-44 的3项测试，使用真实小型Torch模型及合成数据验证执行/续训/来源隔离，不是服务器精度复现。**本次整理后，Stage-44真实CPU测试3项再次全部通过（28.100秒），维护工具2项测试通过，文件字节与活动依赖校验通过。**详见整理报告。

本机可用CPU Torch运行时位于用户Codex缓存，并通过`D:\Users\23920\Documents\Codex\stage34_cpu_torch`提供包。普通系统Python可能没有Torch；不要把跳过Torch测试当作全部通过。

待后续明确的研究工作：

1. 重新定义压缩的实际目标（参数、存储、时延或能耗）及允许的精度损失；这些目标不能互相代替。
2. 基于明确的失败机制或资源瓶颈立项，不默认压缩敏感性、动态选层价值或历史恢复已经成立。
3. 在新的设计中预先固定对照、数据使用、验证门槛与停止规则；若需要独立结论，先重新审计哪些源组/基准未用于设计。
4. 若讨论创新性，重新检索并与已有压缩、跟踪训练及恢复方法区分。旧novelty review仅是2026-10-11的历史判断。

Stage-46、46B、47、48已由用户报告完成，Stage-49已提供focus／onsets结果，不重复启动旧训练。Stage-48原实现的6项CPU测试与Stage-49的5项纯NumPy测试通过；维护清单检查通过。最新服务器数值、逐视频／种子明细和诊断表均来自用户终端，本机未重算完整模型和轨迹产物。Stage-50原实现4项CPU测试通过（6.245秒），包含共同特征真实小型Torch头重放、完整prepare／smoke／collect／summary、缓存复用与篡改拒绝；用户现已提供两张关注表，解释已记录，服务器完整缓存与诊断哈希未在本机核验。Stage-51的4项CPU测试覆盖源结果绑定、fit／held-out方向、普通两视图及出裁剪负例、真实Torch完整流程、带AdamW状态的中断恢复、冻结网络、缓存／结果／checkpoint篡改拒绝；用户现已提供五张支持与结果表，主要对象已定位，完整缓存及新逐种子／事件尚未在本机核验。Stage-52已完成3项真实CPU测试（35.652秒）：仓库实际CENTER小通道分支、梯度与BN冻结、完整Stage-51来源校验、fit隔离、续训一致性及48条OOF合成流程；用户已报告服务器完成并提供scope／policy／focus／paired／factor五表；实现审计通过，但未建立同时保护0304／0307的固定更新范围。完整服务器产物未在本机核验。Stage-53的3项真实CPU测试通过（53.814秒）：连续Hann格点资格、平均权重匹配、size／offset保持梯度、冻结响应和BN、完整Stage-52来源、fit隔离、断点AdamW恢复与48条OOF合成流程。用户已报告服务器REVIEW并提供四张完整表，本机未独立重验全部产物；未达到本轮整体改进与保持目标。Stage-54的3项真实CPU测试通过（92.571秒，无跳过），覆盖当前两帧actor、后续GT不改变输出、末帧size／offset梯度、冻结响应、原生解码、完整Stage-53来源、fit隔离、断点恢复及48条新OOF合成流程；纯NumPy CLI和5文件AST通过。用户已报告Stage-54服务器REVIEW并提供六表；主比较区间跨零、原生能力保护未过，对旧静态方案12/12改善；四项AUC bootstrap区间及逐视频算术核对相符，本机未独立重验完整产物。Stage-55的3项真实CPU测试通过（90.476秒，无跳过），覆盖完整Stage-54来源、实际CENTER与原生decode／hook、共同特征、自己预测作为后继父框、GT隔离、缓存恢复、精确预算及来源／checkpoint／缓存／结果表篡改拒绝；4文件AST和纯NumPy CLI通过。用户现已报告Stage-55完成并提供三张关注表；分解算术相符，最大残差约5.55e-17，完整服务器调用与缓存哈希未在本机重验。Stage-56也已报告REVIEW并提供五份材料，详见最新复核；Stage-57也已报告REVIEW并提供五份材料，Stage-58也已报告REVIEW并提供五份材料，结果见第0.8节；Stage-59已完成父框分量交叉诊断，当前结果与边界见第0.9节。提交与推送遵循下列协作约定。

## 10. 用户协作约定

- 核心任务保持以原生 SGLA 为基础的无人机单目标跟踪。未经用户另行要求，不把检测文献调研解释为切换到检测任务，也不以 YOLOFT 等检测器替换主要基线。
- 选题以“新增价值、可验证差异与后续发展空间”为重点。存在相似工作是建立基线和寻找改进点的依据，不是直接否定方向的理由。
- 想法不成熟时，具体说明可以改善哪个环节、形成什么差异、怎样用最小实验验证；诊断和小原型可并行推进，不要求先完成所有机制证明才允许尝试。已有冻结实验结论和证据标准保持不变。
- 用户通过Git同步仓库，不需要额外脚本zip包；未要求时不代为commit/push。
- 需要行动时直接完成实现、合理测试和服务器指令，不反复确认已经授权的步骤。
- 用户在服务器跑GPU实验，本地助手不能假称已登录服务器。
- 查看结果优先给`cat`命令；长输出可通过文件附件交付。
- 实验代码、原模型、已冻结来源路径和哈希不随意改写；需要改变设计时新建阶段和输出目录。
- 报告结论要区分执行通过、机制证据、闭环收益、速度结果和独立验证，保留失败与保护门槛未过的结果。

相关入口：[实验说明索引](docs/README.md) · [历史恢复说明](archive/README.md) · [整理报告](docs/maintenance/CLEANUP_REPORT.md)
