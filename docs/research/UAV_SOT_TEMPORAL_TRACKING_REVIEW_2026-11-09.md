# 无人机视角单目标跟踪的时序建模与搜索更新

## 摘要

无人机视角的单目标跟踪需要从首帧指定目标出发，在相机运动、尺度变化、遮挡和相似物干扰下持续输出目标框。历史信息可以帮助辨认目标，也可能把一次定位错误传给后续搜索。本文综述截至 2026 年 11 月 9 日可核验的代表性研究，重点覆盖 2025–2026 年的顶会主会和相关期刊论文，并以较早工作追溯技术来源。综述按历史信息在当前定位、可靠性判断、运动预测与计算分配、搜索更新与恢复中的作用组织。现有工作在这些环节分别给出了有效设计，但相同的当帧定位增益可以对应不同的失跟与恢复过程。面向轻量无人机跟踪，一个可检验的研究问题是，在同等搜索预算下，怎样让下一帧搜索利用可靠的定位信息，同时限制错误更新的传播。文末据此给出 SGLATrack 的基线选择与实验判读框架。

**关键词**　无人机视角，单目标跟踪，时序建模，搜索区域，失跟恢复，SGLATrack

## 1. 任务与文献范围

本文以在线 RGB 单目标跟踪（single-object tracking，SOT）为主。跟踪器接收首帧目标框，以后仅使用已到达的图像和已有状态，逐帧输出同一目标的边界框。无人机既可能主动转向、平移和变焦，目标也可能突然移动或被遮挡。同样的图像坐标跳变因此可能对应真实位移、相机运动或错误匹配。局部搜索使这种区分尤为重要，因为当前输出常被用来决定下一帧的裁剪区域。较早的 TCTrack 将时序上下文用于特征提取与相似图细化，期刊扩展还把运行延迟纳入现实场景评估。 [6]

本综述为机制导向的叙述性综述。检索使用 `visual single object tracking`、`UAV tracking`、`temporal modeling`、`search region`、`drift`、`recovery` 和 `VisDrone-SOT` 等关键词组合，以会议论文集、期刊原始页面和作者项目为主要来源。入选工作至少涉及在线定位、历史更新、搜索控制或相应评测之一。CVPR、ICCV、ECCV、NeurIPS、ICML 和 AAAI 主会及 IEEE TPAMI、TIP、TCSVT 等期刊构成主体。另纳入与搜索反馈直接相关的期刊论文，并标明工作坊、相邻任务及在线发表状态。结果比较以同篇论文或同一实验协议为单位。 [25] [28] [29]

为区分时序信息的作用，本文将当前发布框记作 $b_t$，将决定后续裁剪的搜索状态记作 $s_t$。搜索状态至少包含中心与尺度，也可以包含可靠性或遮挡信息。发布框用于回答目标此刻在哪里，搜索状态用于决定下一帧在哪里观察。即使两者都由同一个框初始化，这两个功能也可分别评估。

## 2. 高效无人机跟踪与时序特征

### 2.1 轻量网络如何保留定位能力

SGLATrack 和 AVTrack 从不同角度压缩每次搜索的计算。SGLATrack 利用轻量 ViT 内层表示的相似性，跳过冗余层并从后续候选层中保留一层。AVTrack 通过输入相关的激活模块调节 Transformer 块的执行，并学习对视角变化更稳健的表示。两者都是直接的效率参照。ORTrack 则把重点放在遮挡下的特征可辨认性，通过模拟遮挡的掩蔽训练约束表征，并蒸馏得到更紧凑的 ORTrack-D。改善这类表征，有时可以减少相同搜索区域里的定位错误。 [1] [2] [3]

EATrack 将轻量非对称网络、特征和预测两级蒸馏以及推理时的时序适配组合起来。其原文表 5 在 EATrack-DeiT 内部比较启用和关闭时序线索，VisDrone2018 成功率从 59.8% 升至 65.6%，DTB70 从 63.3% 升至 66.5%，UAV123 均为 68.1%。这一组结果说明，时序线索的收益会随数据分布变化。它也给出了比只比较新旧整网更有解释力的同模型消融。 [4]

### 2.2 怎样以更少计算传递历史

ODTrack 将目标信息压缩成可逐帧传播的 token，以既往状态为当前定位提供提示。SPMTrack 用跟踪专用的混合专家扩展模板、搜索帧和时序上下文之间的关系建模。LoRATv2 使用逐帧因果注意力、键值缓存与不同输入流的低秩适配器，降低重复编码历史的开销。SMTrack 则把历史压缩进状态空间模型的隐状态，逐帧传递和更新。几条路线共同处理的是如何让定位模块以较低代价访问时间信息，其差异在于历史以何种形式保存、怎样与当前帧交互。 [7] [8] [9] [10]

**LETrack 是与当前 SGLATrack 项目最接近的近期外部基线。** 它与 SGLATrack 共享部分作者，直接面向无人机视角跟踪。LETrack 的 LETL 模块选择、合并并传播有代表性的外观 token，论文引入每 FLOP 精度增益（precision per FLOP，PPF）讨论时序模块的计算收益。官方实现提供 `VisDrone2018-SOT-test-dev`、UAV123、UAVDT、DTB70 等评估入口。对搜索状态更新研究，PPF 的启发是将新增计算与增益一起报告。搜索调用数、完整管线延迟和持续失跟仍各有独立的解释价值。 [5] [LETrack 官方实现](https://github.com/GXNU-ZhongLab/LETrack)

## 3. 历史可靠性、运动预测与定位决策

### 3.1 有历史之后，哪些信息值得保留

历史可能因遮挡、错选或外观变化而失真。MUTrack 分别维护长、短时记忆，再融合两者以平衡身份稳定性和对即时变化的适应。DTPTrack 对历史状态分配可靠性权重，并以首帧模板为可信锚点，把历史信息合成可输入跟踪器的先验 token。DASTrack 发现，即使网络包含时序 token，强外观线索仍可能让模型绕过时序分支。作者通过训练期的辅助监督使定位依赖时序 token，正式推理时去除该辅助过程。这三类工作分别处理记忆分工、历史质量和训练时的时序依赖。应用到搜索更新时，还需要检验历史信息是否改变了下一帧的观测位置，以及这一改变能否减少失跟。 [11] [12] [13]

### 3.2 从预测下一状态到分配观测资源

TGTrack 通过预测后续模板的外观学习时序变化，并提供 RGB 跟踪结果。ARTrack-AC 则预测短期轨迹的不确定性，用它在较高和较低推理容量之间切换。前者把未来预测作为学习时序表示的手段，后者把未来不确定性转化为计算分配决策。PVT++ 将运动和视觉信息用于补偿在线处理延迟。更直接针对无人机运动的 SiamMDTP 将相机运动与目标运动分离，用预测结果更新搜索区域并处理遮挡，其原文报告包括 VisDrone2019-SOT 的实验。这些工作说明运动预测、预算分配和搜索区域更新已有各自的技术参照。 [14] [15] [16] [17]

对使用历史框做速度外推的局部跟踪器，预测中心更接近标注中心是第一层证据。改变裁剪还会影响目标在输入中的大小、位置以及与背景的相对关系，最终应检查 SGLA 是否真的输出了更好的框。若四个搜索区域都裁到了目标，但四个输出都偏向干扰物，进一步改善几何覆盖也未必能解决该帧定位错误。相反，若同一帧已经有准确候选，下一步应检验候选比较和历史更新，避免把解码、选择和搜索范围的损失混在一起。 [3] [18] [项目同历史搜索与响应诊断](../experiments/06_sgla_uav_tracking/STAGE47_MATCHED_SEARCH_RESPONSE.md)

### 3.3 候选关联与强化学习控制的作用位置

KeepTrack 通过跨帧关联同时维护目标和相似干扰物的身份，为「当前分数接近、位置却相互冲突」提供了比单帧置信度更丰富的证据。RELO 把强化学习策略用于空间定位，奖励结合当帧交并比（intersection over union，IoU）和序列成功率曲线下面积（area under the curve，AUC）。AdaSAM2 在分割式跟踪中用 Double DQN 选择裁剪尺度，并通过事件驱动逻辑控制搜索扩大和重新决策。对应到 SGLATrack，四选一 PPO 属于搜索之后的候选选择，搜索状态更新则决定下一帧如何构造观测。两者可以用相同候选或相同搜索预算分别比较，以确定增益来自哪一步。 [18] [19] [20]

## 4. 搜索中断、恢复与评测协议

CPDTrack 用精细局部观察和低分辨率的广域观察处理时空不连续，并提出 STDChallenge 评价这类行为。RTracker 用正负样本树判断目标状态，在局部跟踪和检测器之间切换。2026 年在线可查的 OASAMT 把遮挡识别、不可见时的位置预测和重现后的再检测分开建模，并用遮挡与非遮挡双记忆减少污染。其 OccTrack 来源包含 VisDrone-SOT、UAV123 等已有数据，复用时需按原始视频检查重叠。SAM2Long 则用受限的记忆树保留多条分割路径，原方法在视频结束后选择最终路径，因此主要适合作为多假设传播的机制参照。 [21] [22] [23] [24]

同一组逐帧 IoU 即使按不同顺序出现，也可以得到相同的成功率 AUC。持续失跟和间歇抖动由此可能拥有相同的平均分。评估搜索更新应同时记录首次失跟、最长失跟、恢复时间和视频结束时未恢复的事件。涉及遮挡时应说明哪些帧有目标可见性标注，并区分「目标暂时不可见」与「仍可见但跟到别处」。不同数据集的初始化、重置、缺失帧处理和聚合方式都会影响解释，比较时需要保留协议。 [21] [23] [29]

| 数据或协议 | 适合检验的问题 | 比较时需要对齐 |
|---|---|---|
| VisDrone-SOT [25] | 无人机视角下的完整框轨迹，适合与本项目直接接轨 | 年份、split、视频清单、初始化与标注处理。LETrack 的官方示例为 2018 test-dev |
| UAV123、UAV123@10fps、UAV20L [26] | 航拍泛化、增大帧间位移、较长跟踪 | UAV20L 是 UAV123 子集，低帧率版本源于原视频，分析时处理源重叠 |
| WebUAV-3M [27] | 大规模跨场景评估与难例分层 | 模型输入是否只用 RGB，训练视频与测试视频是否重叠 |
| XS-VID v2 [28] | 极小目标的补充压力测试 | 选择 SOT 轨道。YOLOFT 是该论文的检测框架，场景也包括非无人机视频 |
| STDChallenge 与 OccTrack [21] [23] | 时空不连续与遮挡恢复 | 报告难例来源和标注定义，避免将衍生子集当成全新独立样本 |

相邻任务也能帮助理解边界。CVPR 2026 的 DynUAV 对剧烈相机运动下的多目标关联进行压力测试，能启发运动难例的划分。MUST 研究多光谱无人机 SOT，UAST 将 RGB-D 观测与无人机飞行控制结合。三者分别涉及不同的任务输出、传感输入或决策对象，其论文分数与首帧框初始化的 RGB SOT 应分开呈现。 [30] [31] [32]

## 5. 面向 SGLATrack 的研究定位

### 5.1 把同历史收益与闭环收益分开

项目的 Stage-84 在 12 个开发视频、27,050 个冻结历史状态上比较固定四候选和运动生成四候选。运动生成使最佳候选 IoU 增加 0.00179，同时新增 107 帧、丢失 110 帧 IoU 至少 0.5 的候选机会。Stage-85 使用同一冻结 PPO、每帧各四次搜索，允许两臂各自用选中的框推进。运动版 AUC 为 62.69，固定版为 67.24，配对差 −4.54 点，95% 区间为 [−11.92, +0.94]。这组开发结果反映了同历史候选表现与完整轨迹的差别。由于运动外推和裁剪布局一起变化，且闭环历史随后分叉，已有结果无法单独归因给运动、PPO 或后续状态传播。 [Stage-84 复核](../experiments/06_sgla_uav_tracking/STAGE84_RESULTS_REVIEW.md) · [Stage-85 复核](../experiments/06_sgla_uav_tracking/STAGE85_RESULTS_REVIEW.md)

更贴近论文目标的问题是，**在相同搜索预算下，能否用可在线取得的证据选择下一帧搜索状态，使 SGLATrack 在候选冲突时少发生持续失跟，同时保持快速跟随与恢复？** 这个问题与可靠记忆和自适应搜索相邻，区别需要通过干预对象和可测结果来建立。一个可能的做法是固定 SGLATrack 的视觉网络与当帧定位规则，只改变搜索状态怎样接纳新位置或新尺度。Stage-79 从同一历史出发的个别续跑显示，初次选择的差异可以对应完全不同的短窗后续轨迹。限幅和旧中心保留在部分窗口改善，也在完整开发轨迹上出现代价。局部机制与整段收益都需独立验证。 [12] [17] [20] [Stage-78 至 Stage-83 交接记录](../../WORK_HANDOFF.md#047-2026-11-08-stage-78-匹配尺度反馈原型)

### 5.2 基线和证据应各自回答一个问题

| 对照 | 回答的问题 |
|---|---|
| 原生 SGLATrack 与改进后的 SGLATrack | 改动对原跟踪系统的完整轨迹和成本有何影响？ |
| 固定四候选 PPO、零速度覆盖布局、运动覆盖布局 | 同一历史、同一选择器下，收益来自布局去重与覆盖，还是来自速度外推？ |
| 同一当前输出、不同搜索状态更新 | 后续变化有多少来自搜索中心或尺度的写入？关键帧从同一历史起步，只替换一次更新，后续使用共同策略 |
| LETrack、EATrack | 与近期专用无人机时序跟踪器在相同视频和协议下相比，精度、稳定性与速度如何？ |
| ORTrack-D、AVTrack、LoRATv2 或 DTPTrack | 根据最终贡献选择补充对照，分别覆盖遮挡表征、轻量计算或通用时序建模 |

完整轨迹报告 AUC、低 IoU 帧比例、最长失跟及恢复。机制实验报告同历史的裁剪覆盖、最佳候选质量、好框被漏选的比例，以及共同起点短续跑的结果。若要将论文结论推广到真实运动或遮挡，应依据预先定义的属性、可见性标注或可核验的视频观察进行分层。标注只用于离线打分和诊断。候选数、搜索次数、预处理、模型及策略延迟应分别记录。若主张机载效率，还需在目标设备上测量完整管线。开发视频用于分析与选择，独立源视频用于定型后的确认。 [5] [16] [29] [项目确认集清单](../experiments/06_sgla_uav_tracking/results/stage79_confirmation_v1.json)

## 6. 结论

近期研究分别改善了历史表征、历史可靠性、运动预测、观测预算和失跟恢复。对资源受限的无人机 SOT，值得把当前定位和下一帧观测放在同一个闭环里考察。LETrack 和 EATrack 提供了直接的外部比较，DTPTrack、RELO 和 AdaSAM2 为可靠历史、定位决策和动态搜索提供了机制参照。基于 SGLATrack 的搜索更新只有在同预算完整轨迹中减少持续失跟、保留原有恢复能力，并经独立视频确认后，才能形成相应的论文性能主张。 [4] [5] [12] [19] [20]

## 参考文献

1. Xue C. 等. [Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking][1]. CVPR, 2025.
2. Wu Y. 等. [Learning an Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking][2]. IEEE TCSVT, 2026 卷期，2025 在线发表.
3. Wu Y. 等. [Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking][3]. CVPR, 2025.
4. Yang H. 等. [Dual-branch Distilled Transformer for Efficient Asymmetric UAV Tracking][4]. CVPR, 2026.
5. Xue C. 等. [Toward Low-Cost yet Effective Temporal Learning for UAV Tracking][5]. CVPR, 2026.
6. Cao Z. 等. [Towards Real-World Visual Tracking With Temporal Contexts][6]. IEEE TPAMI, 2023. 会议先行版为 [TCTrack，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_TCTrack_Temporal_Contexts_for_Aerial_Tracking_CVPR_2022_paper.html).
7. Zheng Y. 等. [ODTrack: Online Dense Temporal Token Learning for Visual Tracking][7]. AAAI, 2024.
8. Cai W. 等. [SPMTrack: Spatio-Temporal Parameter-Efficient Fine-Tuning with Mixture of Experts for Scalable Visual Tracking][8]. CVPR, 2025.
9. Lin L. 等. [LoRATv2: Enabling Low-Cost Temporal Modeling in One-Stream Trackers][9]. NeurIPS, 2025.
10. Ma Y. 等. [SMTrack: State-Aware Mamba for Efficient Temporal Modeling in Visual Tracking][10]. IEEE TIP, 2026.
11. Wu W. 等. [MUTrack: A Memory-Aware Unified Representation Framework for Visual Tracking][11]. AAAI, 2026.
12. Huang Y. 等. [Drift-Resilient Temporal Priors for Visual Tracking][12]. CVPR, 2026.
13. Zhang D. 等. [Rethinking Temporal Modeling in Visual Object Tracking via Decoupled Auxiliary Supervision][13]. ECCV, 2026.
14. Geng W. 等. [TGTrack: Temporal Generative Learning for Unified Single Object Tracking][14]. CVPR, 2026. [作者项目](https://github.com/wtg1/TGTrack)
15. Lin T. 等. [Adaptive Capacity Autoregressive Visual Tracking][15]. CVPR, 2026.
16. Li B. 等. [PVT++: A Simple End-to-End Latency-Aware Visual Tracking Framework][16]. ICCV, 2023.
17. Xiao T. 等. [A UAV tracking framework via motion-decoupled trajectory prediction and occlusion handling][17]. Information Sciences, 2026.
18. Mayer C. 等. [Learning Target Candidate Association to Keep Track of What Not to Track][18]. ICCV, 2021.
19. Chen X. 等. [RELO: Reinforcement Learning to Localize for Visual Object Tracking][19]. ICML, 2026.
20. Liu B. 等. [Learning Adaptive Search with Reinforcement Learning for Small and Fast Object Tracking][20]. Sensors, 2026.
21. Zhang D. 等. [Beyond Accuracy: Tracking more like Human via Visual Search][21]. NeurIPS, 2024.
22. Huang Y. 等. [RTracker: Recoverable Tracking via PN Tree Structured Memory][22]. CVPR, 2024.
23. Chen Z. 等. [Occlusion-aware visual object tracking with explicit temporal state modeling and dual-memory mechanism][23]. Pattern Recognition, 2026.
24. Ding S. 等. [SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree][24]. ICCV, 2025.
25. Du D. 等. [VisDrone-SOT2019: The Vision Meets Drone Single Object Tracking Challenge Results][25]. ICCV Workshops, 2019. 数据协议来源.
26. Mueller M., Smith N. 与 Ghanem B. [A Benchmark and Simulator for UAV Tracking][26]. ECCV, 2016.
27. Zhang C. 等. [WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking][27]. IEEE TPAMI, 2023 卷期.
28. Guo J. 等. [XS-VID: A Large-Scale Benchmark for Small Object Detection and Tracking in Videos][28]. IEEE TPAMI, 2026 年 10 月 Early Access.
29. Kristan M. 等. [A Novel Performance Evaluation Methodology for Single-Target Trackers][29]. IEEE TPAMI, 2016.
30. Ye J. 等. [Breaking Smooth-Motion Assumptions: A UAV Benchmark for Multi-Object Tracking in Complex and Adverse Conditions][30]. CVPR, 2026.
31. Qin H. 等. [MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking][31]. CVPR, 2025.
32. Qin L. 等. [UAST: Unified Active Search and Tracking for Arbitrary Targets with UAVs][32]. CVPR, 2026.

[1]: https://openaccess.thecvf.com/content/CVPR2025/html/Xue_Similarity-Guided_Layer-Adaptive_Vision_Transformer_for_UAV_Tracking_CVPR_2025_paper.html
[2]: https://ieeexplore.ieee.org/document/11129088/
[3]: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Learning_Occlusion-Robust_Vision_Transformers_for_Real-Time_UAV_Tracking_CVPR_2025_paper.html
[4]: https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Dual-branch_Distilled_Transformer_for_Efficient_Asymmetric_UAV_Tracking_CVPR_2026_paper.pdf
[5]: https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Toward_Low-Cost_yet_Effective_Temporal_Learning_for_UAV_Tracking_CVPR_2026_paper.html
[6]: https://doi.org/10.1109/TPAMI.2023.3307174
[7]: https://ojs.aaai.org/index.php/AAAI/article/view/28591
[8]: https://openaccess.thecvf.com/content/CVPR2025/html/Cai_SPMTrack_Spatio-Temporal_Parameter-Efficient_Fine-Tuning_with_Mixture_of_Experts_for_Scalable_CVPR_2025_paper.html
[9]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/ad7e42e7b1f638e991d822724969be45-Abstract-Conference.html
[10]: https://doi.org/10.1109/TIP.2026.3661393
[11]: https://ojs.aaai.org/index.php/AAAI/article/view/38052
[12]: https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.pdf
[13]: https://link.springer.com/chapter/10.1007/978-3-032-37432-5_12
[14]: https://openaccess.thecvf.com/content/CVPR2026/html/Geng_TGTrack_Temporal_Generative_Learning_for_Unified_Single_Object_Tracking_CVPR_2026_paper.html
[15]: https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_Adaptive_Capacity_Autoregressive_Visual_Tracking_CVPR_2026_paper.pdf
[16]: https://openaccess.thecvf.com/content/ICCV2023/html/Li_PVT_A_Simple_End-to-End_Latency-Aware_Visual_Tracking_Framework_ICCV_2023_paper.html
[17]: https://doi.org/10.1016/j.ins.2025.122784
[18]: https://openaccess.thecvf.com/content/ICCV2021/html/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.html
[19]: https://proceedings.mlr.press/v306/chen26ea.html
[20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13611299/
[21]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/050f8591be3874b52fdac4e1060eeb29-Abstract-Conference.html
[22]: https://openaccess.thecvf.com/content/CVPR2024/html/Huang_RTracker_Recoverable_Tracking_via_PN_Tree_Structured_Memory_CVPR_2024_paper.html
[23]: https://www.sciencedirect.com/science/article/pii/S0031320326011209
[24]: https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf
[25]: https://openaccess.thecvf.com/content_ICCVW_2019/html/VISDrone/Du_VisDrone-SOT2019_The_Vision_Meets_Drone_Single_Object_Tracking_Challenge_Results_ICCVW_2019_paper.html
[26]: https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset
[27]: https://ieeexplore.ieee.org/abstract/document/10004511/
[28]: https://doi.org/10.1109/TPAMI.2026.3741044
[29]: https://ieeexplore.ieee.org/abstract/document/7379002
[30]: https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Breaking_Smooth-Motion_Assumptions_A_UAV_Benchmark_for_Multi-Object_Tracking_in_CVPR_2026_paper.pdf
[31]: https://openaccess.thecvf.com/content/CVPR2025/html/Qin_MUST_The_First_Dataset_and_Unified_Framework_for_Multispectral_UAV_CVPR_2025_paper.html
[32]: https://openaccess.thecvf.com/content/CVPR2026/html/Qin_UAST_Unified_Active_Search_and_Tracking_for_Arbitrary_Targets_with_CVPR_2026_paper.html
