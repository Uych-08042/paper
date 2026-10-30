# 工作交接：SGLATrack 无人机跟踪与模型压缩探索

更新日期：2026-10-30（Asia/Hong_Kong）

当前结果阶段：Stage-55，用户报告`COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`，三张关注表已齐。0204／0304当帧大差主要随起点变化；0307恢复附近存在共同起点上放大的模型收益，也有种子与起点依赖。Stage-56第一步状态更新保护现已实现并完成CPU验证，服务器GPU结果尚未产生；Stage-45检测支线暂不执行。
整理前 Git 基点：`91052cd71d49d1397197ea9d5f532f12d37d2c91`

## 0. 下一次对话的直接接续入口

本节记录截至2026-10-30本次对话结束时的具体状态。**最新工作为[Stage-56第一步状态更新保护](docs/experiments/06_sgla_uav_tracking/STAGE56_FIRST_TRANSITION_PRESERVATION.md)，实现及CPU测试完成，等待用户在服务器执行。**

**先读本节，再读[Stage-54结果复核](docs/experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md)和[Stage-55结果复核](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)**；下文早期阶段的“下一步”均为当时历史，不应覆盖此处当前状态。研究范围文档末尾的Stage-46待运行说明也是历史文字。

### 0.1 固定目标与当前进度

- 用户目标是**以SGLA为基础开展无人机单目标跟踪研究**，数据目前只有VisDrone-SOT；不默认改为检测、多目标跟踪或要求补齐XS-VID／VisDrone-VID。
- 评价采用首帧GT初始化后的完整自身预测轨迹。当前12个源组、2折每折6 fit／6 held-out、种子2132／2133已反复用于设计，是开发证据；不要称为新的独立外部验证。
- 原生SGLA执行6个前缀块＋L7～L12中1个独立候选块，保留原始首帧模板，原生实现没有在线模板更新。它不是串行12层ViT。
- Stage-54训练和完整OOF已完成，状态为`REVIEW_SHORT_ROLLOUT_ENDPOINT_PILOT`。Stage-55诊断已完成，状态为`COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`；该COMPLETE不是新方法PASS。
- **Stage-56已经实现，服务器尚未运行。** 读取已完成的Stage-54作为来源，三臂为完整末帧控制、半强度末帧控制、半末帧＋半第一步保持。4项真实CPU测试通过（74.919秒，无跳过），另有2项增强目标函数检查通过（2.267秒）。不重跑Stage-54／55，也不在收到Stage-56结果之前默认跳到Stage-57。
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

下一次先查看Stage-56是否已在服务器执行及是否有新结果，运行指令见协议。该轮是12个新头、72条完整OOF轨迹，不必重新从头总结55个阶段，也不必默认继续扩充诊断锚点。若设计确实依赖精确中心／尺寸／选点信息，先读服务器现有`stage55_point_details.csv`、`stage55_rollout_summary.csv`、`stage55_timelines.csv`，不重跑GPU采集。长表可让用户用文件附件提供；当前三张关注表已足够支持本次机制判断。

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
| `output/stage54/short_rollout_endpoint_v1/` | 服务器训练、完整OOF及6张已收到表所在目录 |
| `output/stage55/matched_parent_continuation_v1/` | 服务器共同输入及分支缓存、3张已收到关注表和详细几何 |
| `output/stage56/first_transition_preservation_v1/` | 新实验预定目录，服务器尚未生成结果；prepare输入是Stage-54 |

Stage-54已收到scope／policy／retention／focus／paired／video_summary六表；Stage-55已收到focus／crossed_matrix／branch_contrasts三表，分别10／25／50行。**不要在新对话中再次请求这些表或重跑已完成阶段。** 本机保存了复核文档和数值记录，没有服务器完整缓存、模型和diagnostic／decision原件；未独立重验服务器全部SHA或调用预算。仍未收到的Stage-54总体种子差在服务器decision中，但它不妨碍判断已有保护条件未过，无需为此默认追加索取。

用户查看文件偏好`cat`。定位服务器现有产物时先进入仓库再设置变量，防止此前出现的`~/workspace/output`误指；以下只定位，不启动实验：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE54_ROOT="$PWD/output/stage54/short_rollout_endpoint_v1"
export STAGE55_ROOT="$PWD/output/stage55/matched_parent_continuation_v1"
```

仓库忽略CSV／NPZ等实验产物，Git同步不是结果备份。服务器要保留Stage-54／55及上游来源链的manifest、run、收据、模型、特征缓存和轨迹；不能只留下analysis表或删除被新阶段校验引用的来源文件。

本机前轮CPU测试使用下面的已有运行时；仅在新改动需要验证时按需使用，CPU合成测试不是GPU精度复现：

```powershell
$env:PYTHONPATH='D:\Users\23920\Documents\Codex\stage46_cpu_deps;D:\Users\23920\Documents\Codex\stage34_cpu_torch'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Users\23920\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B tracking/test_stage55_matched.py -v
```

维护检查为`python -B tools/check_workspace.py`，最近通过：505个保留文件、200个归档移动、106个原活动依赖、0个再生缓存；`git diff --check`和文档链接检查通过。Stage-1～55 Python源码保持冻结。Git工作树有本次结果／交接文档修改，以及新增`docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md`；这些需要随仓库一起同步，但本助手未替用户commit或push，不覆盖现有未提交修改。实验自身指纹校验严格，后续改变设计应新建阶段／目录，不直接改旧代码或收据。

可在下一次对话给出以下接续提示：

> 先读取WORK_HANDOFF.md第0节及Stage-54、Stage-55结果复核。核心任务仍是基于SGLA的无人机单目标跟踪。Stage-55已完成、三表已齐，Stage-56已实现并完成CPU测试，服务器GPU结果尚未产生。请先查看Stage-56协议及本轮执行状态，接续运行或分析新结果；不要重复请求Stage-54／55已有表或重跑旧阶段。

### 0.6 Stage-56当前执行状态

本轮为用户要求“继续下一轮实验”后完成的新实现，尚未登录服务器。沿用两个种子、2折、256步，每模型1,024个普通监督视图。正式总预算为24,576次完整训练前向、12,288次仅头重算，另执行72条完整OOF；smoke单独计费。原生6+1结构、响应和BN冻结，第一帧已有普通监督，本次新增的是该帧原生保持项；跨帧父框仍detach，不能称为BPTT。

输入`STAGE54_ROOT`，新输出`STAGE56_ROOT=$PWD/output/stage56/first_transition_preservation_v1`。按协议先prepare，再smoke，依次train seed2132／2133，evaluate，summary。第一步和末帧的资格都来自fit内原生两帧质量及各自可用裁剪，不进入在线actor。完整末帧控制若不能精确复现旧权重和boxes／previous／peak／layer，则先排查实现／环境，不放宽门槛后解释收益。

预定主要为transition_preserve−endpoint_control，另必须对比transition_preserve−endpoint_half；两条AUC对比用0.0125／0.9875家族区间检查。仍检查原生可靠帧至少−0.5、低IoU不恶化、无新稳定视频失效及双种子方向，不只看宏AUC。结果尚未产生，不能宣称保护或恢复已改进。

回传优先为Stage-56 decision、policy_summary、scope_summary、paired、focus五个文件，使用cat查看，大表可附件。新的原生保持系数是预先固定的三臂控制，不在同一结果上追加搜索。Stage-1～55源码未改；5个Stage-56 Python文件和新协议尚未代用户提交或推送。

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

最新[Stage-55结果复核](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)：三表已齐，10条focus两条当帧及后继记账路径相符。0204:664 S2132父框五头IoU均≥0.6471，其他父框均0且coverage=1；2132自身32帧差−0.7977，在共同S起点换头仅−0.0116，主要差随进入窗口前状态变化。0304:554 P2132父框五头均≥0.7457、S两种子／P2133父框全0；2132当帧+0.7753、32帧+0.8010均落在指定分解起点项，当前好起点上换头作用很小。1078的P2132父框仍coverage=1，但五头输出均约0.187～0.193，32帧自身差仅+0.0430、末帧差0；共同native起点两种子保持头均多4个低IoU帧。0307:389共同起点均值差皆小正；471恢复附近，同S起点换保持头的32帧均值差2132为+0.1545、2133为+0.3620，2133末帧+0.6601，当帧仅+0.0163。2132五起点均获益，2133四正一负（S2132起点−0.1884），收益状态依赖。2133自身均值差仍−0.1732，起点项−0.5351抵消共同S起点模型收益；窗口止502，Stage-54记录的511恢复在窗口外。50条比较均当帧框分岔、下一帧裁剪／响应分岔，IoU差0不等于没有干预。不能将全部效果归因当帧头能力或全部归因既有历史；下一轮优先研究分岔前状态保护和恢复阶段的稳健推进，不预设中心／尺寸唯一机制。Stage-1～55代码及结果冻结，Stage-56已实现并完成CPU验证，不重复请求三表。

## 2. 目录与文件定位

| 路径 | 用途与保留理由 |
|---|---|
| `README.md` | 原 SGLATrack 项目说明及本项目入口；原论文的成绩不是后续实验成绩 |
| `WORK_HANDOFF.md` | 当前研究、结论、环境与交接说明 |
| `docs/README.md` | 全部阶段说明的分类索引 |
| `tracking/` | 当前保留的运行、评估、几何、损失和测试依赖，最新入口为 Stage-55 |
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
- **本机没有完整 Stage-16～55实验产物。**后文近期数值来自用户在会话中提供的服务器终端输出/附件，不能假称已在本机对全部原始结果文件重新核验。
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

Stage-55 focus／crossed_matrix／branch_contrasts也已收到，最新结论见[Stage-55复核](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)。已有三表足够作当前机制判断，不重复索取或重跑采集。[Stage-55协议](docs/experiments/06_sgla_uav_tracking/STAGE55_MATCHED_PARENT_CONTINUATION.md)保留历史复现入口。精确中心／宽高／选点归因可使用既有point_details及timelines，当前没有追加采集或新训练。Stage-56已实现并完成CPU验证，服务器结果尚未产生。

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

Stage-46、46B、47、48已由用户报告完成，Stage-49已提供focus／onsets结果，不重复启动旧训练。Stage-48原实现的6项CPU测试与Stage-49的5项纯NumPy测试通过；维护清单检查通过。最新服务器数值、逐视频／种子明细和诊断表均来自用户终端，本机未重算完整模型和轨迹产物。Stage-50原实现4项CPU测试通过（6.245秒），包含共同特征真实小型Torch头重放、完整prepare／smoke／collect／summary、缓存复用与篡改拒绝；用户现已提供两张关注表，解释已记录，服务器完整缓存与诊断哈希未在本机核验。Stage-51的4项CPU测试覆盖源结果绑定、fit／held-out方向、普通两视图及出裁剪负例、真实Torch完整流程、带AdamW状态的中断恢复、冻结网络、缓存／结果／checkpoint篡改拒绝；用户现已提供五张支持与结果表，主要对象已定位，完整缓存及新逐种子／事件尚未在本机核验。Stage-52已完成3项真实CPU测试（35.652秒）：仓库实际CENTER小通道分支、梯度与BN冻结、完整Stage-51来源校验、fit隔离、续训一致性及48条OOF合成流程；用户已报告服务器完成并提供scope／policy／focus／paired／factor五表；实现审计通过，但未建立同时保护0304／0307的固定更新范围。完整服务器产物未在本机核验。Stage-53的3项真实CPU测试通过（53.814秒）：连续Hann格点资格、平均权重匹配、size／offset保持梯度、冻结响应和BN、完整Stage-52来源、fit隔离、断点AdamW恢复与48条OOF合成流程。用户已报告服务器REVIEW并提供四张完整表，本机未独立重验全部产物；未达到本轮整体改进与保持目标。Stage-54的3项真实CPU测试通过（92.571秒，无跳过），覆盖当前两帧actor、后续GT不改变输出、末帧size／offset梯度、冻结响应、原生解码、完整Stage-53来源、fit隔离、断点恢复及48条新OOF合成流程；纯NumPy CLI和5文件AST通过。用户已报告Stage-54服务器REVIEW并提供六表；主比较区间跨零、原生能力保护未过，对旧静态方案12/12改善；四项AUC bootstrap区间及逐视频算术核对相符，本机未独立重验完整产物。Stage-55的3项真实CPU测试通过（90.476秒，无跳过），覆盖完整Stage-54来源、实际CENTER与原生decode／hook、共同特征、自己预测作为后继父框、GT隔离、缓存恢复、精确预算及来源／checkpoint／缓存／结果表篡改拒绝；4文件AST和纯NumPy CLI通过。用户现已报告Stage-55完成并提供三张关注表；分解算术相符，最大残差约5.55e-17，完整服务器调用与缓存哈希未在本机重验。Stage-56已实现并完成CPU验证，服务器结果尚未产生。提交与推送遵循下列协作约定。

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
