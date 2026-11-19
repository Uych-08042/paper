# 实验说明索引

当前研究结论与接手说明见[工作交接文档](../WORK_HANDOFF.md)。以下保留原阶段协议，文档内容未改写；目录分组只是导航，不代表方法已通过。

最新完整开发评价 Stage-131 已完成。单槽放大对缩边 AUC +0.1497 点，区间跨零，丢失两次匹配恢复。Stage-121 备用搜索与原生 SGLA 是同卡对照。 [Stage-131 结果](experiments/06_sgla_uav_tracking/STAGE131_RESULTS_REVIEW.md) · [Stage-121 结果](experiments/06_sgla_uav_tracking/STAGE121_RESULTS_REVIEW.md)

Stage-132 首次分叉审计已完成。112 对缓存轨迹中，两组在状态匹配时均未让已有好候选全部消失。放大组丢失的两次恢复都发生在 q2 替换改变此前输出之后。 [结果复核](experiments/06_sgla_uav_tracking/STAGE132_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE132_FIRST_DIVERGENCE.md)

Stage-133 原生单搜索失跟诊断已完成。19 个起点都保有目标中心，11 个仍有好格点框，8 个充分覆盖却无好框。直接改用原始最高峰在起点救回 0 次，固定状态 AUC −0.4218 点。 [结果复核](experiments/06_sgla_uav_tracking/STAGE133_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE133_NATIVE_FAILURE_AUDIT.md)

Stage-134 同起点短程续跑已完成。Hann 第二峰在 19 个失跟起点新增 3 次并丢失 1 次恢复，在 28 个稳定窗口中引入 4 个低 IoU 窗口。 [结果复核](experiments/06_sgla_uav_tracking/STAGE134_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE134_NATIVE_PEAK_CONTINUATION.md)

Stage-135 缓存改选诊断已完成。三条条件均新增 3 次恢复且保住原生 6 次恢复，28 个稳定窗口均未触发。历史连续性条件减少改选，整段闭环收益待验证。 [结果复核](experiments/06_sgla_uav_tracking/STAGE135_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE135_CACHED_PEAK_GATE.md)

Stage-136 整段闭环已完成。仅分差／完整规则相对原生 AUC −2.8782／−0.2968 点，分别在 2／1 个原生稳定视频引入新失跟。 [结果复核](experiments/06_sgla_uav_tracking/STAGE136_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE136_PEAK_GATE_CLOSED_LOOP.md)

Stage-137 已完成。原选块无好框的 8 个起点中，其他块标准输出补出 0 个。原选块网格已有好框的 11 个起点中，其他块标准输出找回 8 个。 [结果复核](experiments/06_sgla_uav_tracking/STAGE137_RESULTS_REVIEW.md) · [实验协议](experiments/06_sgla_uav_tracking/STAGE137_SHARED_CROP_LAYER_SUPPLY.md)

Stage-138 固定跨块新增好框和稳定受损起点，验证单次写回后的 32 帧轨迹。服务器结果待跑。 [实验协议](experiments/06_sgla_uav_tracking/STAGE138_CROSS_LAYER_CONTINUATION.md)

Stage-122 缓存诊断已完成。358 次替换中 80 次前史和前三框观测匹配，补出 9 个好框却仅选中 1 个；两次前三框重排直接改变输出，分别是 0174 和 0089 的首次分叉。 [结果复核](experiments/06_sgla_uav_tracking/STAGE122_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE122_ALTERNATE_RERANKING_DIAGNOSTIC.md)

Stage-123 固定分数诊断已完成。80 个匹配点中改选 5 个，新增好框采用由 1/9 升至 3/9，收益集中在 0176 和 0237。视频配对差 +0.0419，区间下界 0；没有新增闭环评价。 [结果复核](experiments/06_sgla_uav_tracking/STAGE123_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE123_FROZEN_RETAINED_SCORES.md)

Stage-124 已完成同框评分对照。独立评分普通帧对关系重训 +0.001098、区间跨零，对原记录 −0.002650；两种重训头各只保住原记录的 2/6 次普通帧救援。备用触发有局部信号，暂不支持全局替换。 [结果复核](experiments/06_sgla_uav_tracking/STAGE124_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE124_CANDIDATE_INDEPENDENCE.md)

Stage-125 已完成 346 个备用搜索触发点的短程闭环。独立对原评分后续 31 帧平均 IoU +0.005629，区间跨零，新增持续好窗都来自同一种子。 [结果复核](experiments/06_sgla_uav_tracking/STAGE125_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE125_TRIGGER_CONTINUATION.md)

Stage-126 已完成当帧提交与下一帧备用搜索交叉对照。独立评分换框在两路均禁止备用时后续 IoU +0.030208，自然请求下为 +0.005629。备用搜索同时出现补框恢复和好框漏选。 [结果复核](experiments/06_sgla_uav_tracking/STAGE126_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE126_FACTORIAL_CONTINUATION.md)

Stage-127 已完成搜索裁剪诊断。333 个无好输出起点中，54 个四裁剪均与目标无交集，276 个覆盖充分，其中 226 个在 256×256 输入中的目标短边仍不到 4 像素。 [结果复核](experiments/06_sgla_uav_tracking/STAGE127_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE127_TRIGGER_CROP_COVERAGE.md)

Stage-128 已完成 346 个同帧对照。输入过小的 226 点短边中位数从 1.49 升到 4.43 像素，仍无新增好候选或好选框。 [结果复核](experiments/06_sgla_uav_tracking/STAGE128_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE128_PADDING_TRIM.md)

Stage-129 四槽粗搜与局部放大已完成。346 点好候选新增 8、丢失 6，丢失框全来自被替换的 q3。真值居中 54 点成功，其中 38 点实际局部搜索覆盖不足。 [结果复核](experiments/06_sgla_uav_tracking/STAGE129_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE129_COARSE_TO_FINE.md)

Stage-130 缓存重评分已完成。三个保留 q3 的单槽方案均保住原 13 个好候选起点并新增 5 个，第一位置保住原 3 个好选框并新增 2 个，位置投票仍漏选一次好 q3。 [结果复核](experiments/06_sgla_uav_tracking/STAGE130_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE130_PRESERVE_Q3.md)

Stage-131 在 28 段开发视频评估了 196 条轨迹。原生／备用搜索／缩边／单槽放大的整段 AUC 为 63.9605／54.0502／51.4511／51.6007。 [结果复核](experiments/06_sgla_uav_tracking/STAGE131_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE131_ZOOM_CLOSED_LOOP.md)

历史入口脚本如已归档，先按[代码恢复说明](../archive/README.md)恢复；代码块在仓库根目录执行。

最新论文式图示见 [真实图像简洁框架v5](figures/sgla_search_state_v5/README.md)，仅保留(a)历史条件搜索与(b)SGLA共享质量选择，提供中英文3840×1300 PNG及内嵌真实图片的可编辑SVG；图片来源及示意框说明写入图注。[完整说明版v4](figures/sgla_search_state_v4/README.md)另外展示Stage-120尚未持续部署验证的提交检验。历史 [PPO图v3](figures/sgla_search_state_v3/README.md) 对应Stage-74／76，不能代表当前主干；[v2](figures/sgla_search_state_v2/README.md) 和 [v1](figures/sgla_search_state_v1/README.md) 描述Stage-54～60的方案。

## Stage-1～15：基线、候选层与动态计算

- [STAGE10_CLOSED_LOOP_TRANSFER.md](experiments/01_baseline_and_routing/STAGE10_CLOSED_LOOP_TRANSFER.md) — Stage-10: Frozen Periodic-L9 Closed-Loop Transfer
- [STAGE11_EXTERNAL_VALIDATION.md](experiments/01_baseline_and_routing/STAGE11_EXTERNAL_VALIDATION.md) — Stage-11: Frozen External Full-Sequence Validation
- [STAGE12_EXTERNAL_FAILURE_AUDIT.md](experiments/01_baseline_and_routing/STAGE12_EXTERNAL_FAILURE_AUDIT.md) — Stage-12 External Failure Audit
- [STAGE13_LONG_HORIZON_VALUE.md](experiments/01_baseline_and_routing/STAGE13_LONG_HORIZON_VALUE.md) — Stage 13: Long-Horizon Value-Driven Computation
- [STAGE14_MULTIACTION_AUDIT.md](experiments/01_baseline_and_routing/STAGE14_MULTIACTION_AUDIT.md) — Stage-14 Multi-Action Long-Horizon Counterfactual Audit
- [STAGE15_ACTION_STABILITY.md](experiments/01_baseline_and_routing/STAGE15_ACTION_STABILITY.md) — Stage-15 disjoint-window action-value stability audit
- [STAGE1_VALIDATION.md](experiments/01_baseline_and_routing/STAGE1_VALIDATION.md) — SGLATrack 第一阶段三路径验证
- [STAGE2_CANDIDATE_VALIDATION.md](experiments/01_baseline_and_routing/STAGE2_CANDIDATE_VALIDATION.md) — SGLATrack 第二阶段候选层验证
- [STAGE3_ROUTER_VALIDATION.md](experiments/01_baseline_and_routing/STAGE3_ROUTER_VALIDATION.md) — SGLATrack 第三阶段离线路由验证
- [STAGE4_AB_VALIDATION.md](experiments/01_baseline_and_routing/STAGE4_AB_VALIDATION.md) — Stage-4A/B: 评估口径、计算开销与独立训练数据
- [STAGE4C_ROUTER_VALIDATION.md](experiments/01_baseline_and_routing/STAGE4C_ROUTER_VALIDATION.md) — Stage-4C：候选收益路由器离线验证
- [STAGE4D_L9_GATE_VALIDATION.md](experiments/01_baseline_and_routing/STAGE4D_L9_GATE_VALIDATION.md) — Stage-4D：Fast/L9 单动作收益门控
- [STAGE4E_L9_GATE_STABILITY.md](experiments/01_baseline_and_routing/STAGE4E_L9_GATE_STABILITY.md) — Stage-4E：L9 门控的在线阈值与稳定性验证
- [STAGE4F_L9_ONLINE_CONTROLLER.md](experiments/01_baseline_and_routing/STAGE4F_L9_ONLINE_CONTROLLER.md) — Stage-4F：因果在线预算控制器
- [STAGE5_PERIODIC_L9_CLOSED_LOOP.md](experiments/01_baseline_and_routing/STAGE5_PERIODIC_L9_CLOSED_LOOP.md) — Stage-5：周期 L9 闭环基线
- [STAGE6_BUDGET_ORACLE.md](experiments/01_baseline_and_routing/STAGE6_BUDGET_ORACLE.md) — Stage-6：序列级预算 Oracle
- [STAGE7_TEMPORAL_HOLDOUT.md](experiments/01_baseline_and_routing/STAGE7_TEMPORAL_HOLDOUT.md) — Stage 7: Frozen Temporal-Layout Holdout
- [STAGE8_CAUSAL_INTERVENTION.md](experiments/01_baseline_and_routing/STAGE8_CAUSAL_INTERVENTION.md) — Stage-8B: Single-L9 Causal Intervention
- [STAGE8_TAIL_DIAGNOSTIC.md](experiments/01_baseline_and_routing/STAGE8_TAIL_DIAGNOSTIC.md) — Stage 8A: Closed-loop Tail and Temporal Diagnostic
- [STAGE9_EQUAL_COST_SCHEDULE.md](experiments/01_baseline_and_routing/STAGE9_EQUAL_COST_SCHEDULE.md) — Stage-9: Equal-Cost Temporal Schedule Intervention

## Stage-16～26：离线价值、PPO与早期压缩

- [SINGLE_GPU_PPO_EXPERIMENT.md](experiments/02_policy_and_compression/SINGLE_GPU_PPO_EXPERIMENT.md) — Single-GPU task/resource-aware PPO: proposed experiment
- [STAGE16_CONTINUOUS_BEHAVIOR.md](experiments/02_policy_and_compression/STAGE16_CONTINUOUS_BEHAVIOR.md) — Stage-16A Continuous Multi-Action Behavior Data
- [STAGE16B_RECURRENT_CQL.md](experiments/02_policy_and_compression/STAGE16B_RECURRENT_CQL.md) — Stage-16B Recurrent Conservative Offline Q Screen
- [STAGE16C_TRUE_CLOSED_LOOP.md](experiments/02_policy_and_compression/STAGE16C_TRUE_CLOSED_LOOP.md) — Stage-16C: Frozen TAVRTrack True Closed Loop
- [STAGE17_SAFE_RESIDUAL_ADVANTAGE.md](experiments/02_policy_and_compression/STAGE17_SAFE_RESIDUAL_ADVANTAGE.md) — Stage-17A: Safe Residual Advantage Screen
- [STAGE18_FULL_PPO_DEVELOPMENT.md](experiments/02_policy_and_compression/STAGE18_FULL_PPO_DEVELOPMENT.md) — Stage-18 Full PPO Development Run
- [STAGE18_LONG_TERM_LAYER_POLICY.md](experiments/02_policy_and_compression/STAGE18_LONG_TERM_LAYER_POLICY.md) — Stage-18: Long-Term Layer Policy With Persistent Target Memory
- [STAGE18A_EXECUTION_PILOT.md](experiments/02_policy_and_compression/STAGE18A_EXECUTION_PILOT.md) — Stage-18A Execution Pilot
- [STAGE18B_REPRESENTATION_WARMUP.md](experiments/02_policy_and_compression/STAGE18B_REPRESENTATION_WARMUP.md) — Stage-18B: Chronological Supervised Representation Warm-up
- [STAGE18C_ONPOLICY_PPO.md](experiments/02_policy_and_compression/STAGE18C_ONPOLICY_PPO.md) — Stage-18C: Frozen Representation and Live On-Policy PPO
- [STAGE19_COMPRESSION.md](experiments/02_policy_and_compression/STAGE19_COMPRESSION.md) — Stage-19: persistent structural-compression mechanism pilot
- [STAGE20_COMPRESSION.md](experiments/02_policy_and_compression/STAGE20_COMPRESSION.md) — Stage-20: fixed dense student and rollout-state correction
- [STAGE22_SINGLE_GPU_CALIBRATION.md](experiments/02_policy_and_compression/STAGE22_SINGLE_GPU_CALIBRATION.md) — Stage-22: single-GPU calibration for the new resource-aware PPO design
- [STAGE23_ONPOLICY_BRANCH_CREDIT.md](experiments/02_policy_and_compression/STAGE23_ONPOLICY_BRANCH_CREDIT.md) — Stage-23: Same-State Branch Credit for Accuracy-Oriented PPO
- [STAGE23B_FIXED_BUDGET_DEVELOPMENT.md](experiments/02_policy_and_compression/STAGE23B_FIXED_BUDGET_DEVELOPMENT.md) — Stage-23B: Fixed-Budget Three-Seed Accuracy Experiment
- [STAGE24_DEPLOYMENT_ACTION_AUDIT.md](experiments/02_policy_and_compression/STAGE24_DEPLOYMENT_ACTION_AUDIT.md) — Stage-24: Read-Only PPO Deployment Action Audit
- [STAGE25_RELATIONAL_STATE_PROBE.md](experiments/02_policy_and_compression/STAGE25_RELATIONAL_STATE_PROBE.md) — Stage-25: Equal-Cost Relational State Feasibility
- [STAGE26_VALUE_REPRESENTATION_PILOT.md](experiments/02_policy_and_compression/STAGE26_VALUE_REPRESENTATION_PILOT.md) — Stage-26: Intervention-Value Representation A/B Pilot

## Stage-27～34：无人机失败定位与身份表示

- [STAGE27_EVENT_REVIEW.md](experiments/03_uav_failure_and_identity/STAGE27_EVENT_REVIEW.md) — Stage-27 follow-up: read-only onset review
- [STAGE27_UAV_FAILURE_DIAGNOSTIC.md](experiments/03_uav_failure_and_identity/STAGE27_UAV_FAILURE_DIAGNOSTIC.md) — Stage-27: UAV failure localization before method development
- [STAGE28_RESPONSE_COMPETITION.md](experiments/03_uav_failure_and_identity/STAGE28_RESPONSE_COMPETITION.md) — Stage-28：冻结 SGLA 的空间响应竞争审计
- [STAGE29_CANDIDATE_PERSISTENCE.md](experiments/03_uav_failure_and_identity/STAGE29_CANDIDATE_PERSISTENCE.md) — Stage-29：一次空间候选纠正能否产生持续改善
- [STAGE30_CORRECTED_BRANCH_RESPONSE.md](experiments/03_uav_failure_and_identity/STAGE30_CORRECTED_BRANCH_RESPONSE.md) — Stage-30：纠正分支首次失效帧的响应审计
- [STAGE31_CONTINUOUS_IDENTITY.md](experiments/03_uav_failure_and_identity/STAGE31_CONTINUOUS_IDENTITY.md) — Stage-31：无 GT 的持续候选身份关联小规模验证
- [STAGE31_PRECISION_COMPAT.md](experiments/03_uav_failure_and_identity/STAGE31_PRECISION_COMPAT.md) — Stage-31：原始 float32 响应分数的回放审计修复
- [STAGE32_IDENTITY_ATTRIBUTION.md](experiments/03_uav_failure_and_identity/STAGE32_IDENTITY_ATTRIBUTION.md) — Stage-32：冻结候选、特征与选框的只读归因
- [STAGE33_POOLED_DESCRIPTOR_PROBE.md](experiments/03_uav_failure_and_identity/STAGE33_POOLED_DESCRIPTOR_PROBE.md) — Stage-33：困难状态均衡的完整候选描述子留一验证
- [STAGE34_SPATIAL_IDENTITY_STUDENT.md](experiments/03_uav_failure_and_identity/STAGE34_SPATIAL_IDENTITY_STUDENT.md) — Stage-34：单路径学生的空间身份表征与困难负样本机制验证

## Stage-35～44：闭环诊断、压缩与匹配对照

- [STAGE35_PRESERVE_CORRECT_OOF.md](experiments/04_closed_loop_compression/STAGE35_PRESERVE_CORRECT_OOF.md) — Stage-35：保留可靠定位能力的条件纠错蒸馏
- [STAGE35B_SUPERVISED_BASELINE_AUDIT.md](experiments/04_closed_loop_compression/STAGE35B_SUPERVISED_BASELINE_AUDIT.md) — Stage-35B：普通监督学生的事后只读审计
- [STAGE36_SAME_CROP_MECHANISM.md](experiments/04_closed_loop_compression/STAGE36_SAME_CROP_MECHANISM.md) — Stage-36：获益与受损案例的同裁剪机制定位
- [STAGE36B_GEOMETRY_AUDIT.md](experiments/04_closed_loop_compression/STAGE36B_GEOMETRY_AUDIT.md) — Stage-36B：缓存中的中心偏移与宽高分解
- [STAGE36C_ONSET_JUMP_AUDIT.md](experiments/04_closed_loop_compression/STAGE36C_ONSET_JUMP_AUDIT.md) — Stage-36C：失跟起点的响应峰与中心跳变审计
- [STAGE37_CROSSED_CLOSED_LOOP.md](experiments/04_closed_loop_compression/STAGE37_CROSSED_CLOSED_LOOP.md) — Stage-37：冻结响应／几何四臂闭环实验
- [STAGE38_SIZE_OFFSET_CLOSED_LOOP.md](experiments/04_closed_loop_compression/STAGE38_SIZE_OFFSET_CLOSED_LOOP.md) — Stage-38：固定学生响应的 size／offset 闭环消融
- [STAGE39_MATCHED_PERTURBATION.md](experiments/04_closed_loop_compression/STAGE39_MATCHED_PERTURBATION.md) — Stage-39：固定预测父框扰动下的压缩敏感性验证
- [STAGE40_MATCHED_CROP_AUGMENTATION.md](experiments/04_closed_loop_compression/STAGE40_MATCHED_CROP_AUGMENTATION.md) — Stage-40：匹配父框扰动增强训练对照
- [STAGE41_CACHED_TRANSFER_AUDIT.md](experiments/04_closed_loop_compression/STAGE41_CACHED_TRANSFER_AUDIT.md) — Stage-41：增强收益与轨迹损伤的缓存审计
- [STAGE42_FROZEN_STATE_SWAP.md](experiments/04_closed_loop_compression/STAGE42_FROZEN_STATE_SWAP.md) — Stage-42：冻结模型与初始预测父框的交叉干预
- [STAGE43_ONLINE_HISTORY_RECOVERY.md](experiments/04_closed_loop_compression/STAGE43_ONLINE_HISTORY_RECOVERY.md) — Stage-43：单模型历史状态恢复的小规模验证
- [STAGE44_MATCHED_STRUCTURE_ADAPTATION.md](experiments/04_closed_loop_compression/STAGE44_MATCHED_STRUCTURE_ADAPTATION.md) — Stage-44：压缩结构 × 相同适配预算的基础对照

## 当前研究范围与暂存支线

- [SGLA 无人机跟踪研究范围](research/SGLA_UAV_TRACKING_RESEARCH_SCOPE.md) — 用户已明确原生 SGLA 基线与单目标跟踪任务，后续立项遵循此范围
- [STAGE46_CAMERA_SEARCH_PILOT.md](experiments/06_sgla_uav_tracking/STAGE46_CAMERA_SEARCH_PILOT.md) — 原生 SGLA 四组48轨迹已由用户报告完成；收益与损伤集中，尚无稳定改进
- [STAGE46B_SEARCH_SHIFT_AUDIT.md](experiments/06_sgla_uav_tracking/STAGE46B_SEARCH_SHIFT_AUDIT.md) — 用户已报告完成；关键起点目标仍在裁剪内，需继续区分输出响应与形状问题
- [STAGE47_MATCHED_SEARCH_RESPONSE.md](experiments/06_sgla_uav_tracking/STAGE47_MATCHED_SEARCH_RESPONSE.md) — 用户已报告400次重放完成；固定历史下观察到裁剪引起的选框翻转与独立的宽度膨胀案例
- [STAGE48_NATIVE_HEAD_STABILITY.md](experiments/06_sgla_uav_tracking/STAGE48_NATIVE_HEAD_STABILITY.md) — 原生SGLA预测头的排序×尺寸约束；用户已报告四臂OOF完成，状态REVIEW
- [STAGE48_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE48_RESULTS_REVIEW.md) — **当前结果**：已定位0304／0307共有损伤、0226种子分岔及0176单种子收益；需继续冻结轨迹时序诊断
- [STAGE49_HEAD_TRAJECTORY_AUDIT.md](experiments/06_sgla_uav_tracking/STAGE49_HEAD_TRAJECTORY_AUDIT.md) — 六视频冻结轨迹时序诊断；用户已提供focus与onsets
- [STAGE49_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE49_RESULTS_REVIEW.md) — **当前结果**：0304起点分中心／尺寸两类；0226与0176的差别集中于恢复
- [STAGE50_SHARED_PARENT_HEAD_REPLAY.md](experiments/06_sgla_uav_tracking/STAGE50_SHARED_PARENT_HEAD_REPLAY.md) — 95个共同预测父框输入×9个冻结头；用户已提供关注交叉矩阵与focus
- [STAGE50_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE50_RESULTS_REVIEW.md) — **当前结果**：多数关注输入的差异随历史改变，保留0304的原生／适配头差异；优先检验自身状态训练覆盖
- [STAGE51_MATCHED_STATE_SOURCE.md](experiments/06_sgla_uav_tracking/STAGE51_MATCHED_STATE_SOURCE.md) — 原生状态与固定fit内预测状态的匹配普通监督；用户已报告REVIEW并提供policy／paired
- [STAGE51_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE51_RESULTS_REVIEW.md) — **当前结果**：0226／0204获益、0176恢复取舍、0304／0307共同损伤；两折状态暴露不对称，下一步优先研究原生能力保持
- [STAGE52_FIXED_STATE_BRANCH_ADAPTATION.md](experiments/06_sgla_uav_tracking/STAGE52_FIXED_STATE_BRANCH_ADAPTATION.md) — 用户已报告完成；scope／policy／focus／paired／factor五表已收到，协议与复现命令保留
- [STAGE52_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE52_RESULTS_REVIEW.md) — **当前结果**：几何对原生10胜2负、对joint仅3胜9负；0304／0307长期损伤抵消多数收益，区间跨零，下一步检验适配能力保持
- [STAGE53_NATIVE_GEOMETRY_PRESERVATION.md](experiments/06_sgla_uav_tracking/STAGE53_NATIVE_GEOMETRY_PRESERVATION.md) — 用户已报告REVIEW，scope／policy／retention／focus四表已收到，协议与复现命令保留
- [STAGE53_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE53_RESULTS_REVIEW.md) — **当前结果**：约束确有梯度，但多数收益幅度下降；0307／0176局部改善伴随0304早期失效与0204高分分支丢失
- [STAGE54_SHORT_ROLLOUT_ENDPOINT.md](experiments/06_sgla_uav_tracking/STAGE54_SHORT_ROLLOUT_ENDPOINT.md) — 用户已报告REVIEW，scope／policy／retention／focus／paired／video_summary六表已收到；协议及历史复现命令保留
- [STAGE54_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md) — **结果复核**：对旧静态保持12胜0负、名义CI为正；对匹配普通监督4胜8负、CI跨零，0307／0304改善伴随0204损伤，可靠帧及新增失效保护未过
- [STAGE55_MATCHED_PARENT_CONTINUATION.md](experiments/06_sgla_uav_tracking/STAGE55_MATCHED_PARENT_CONTINUATION.md) — 用户已报告完整诊断完成，focus／crossed_matrix／branch_contrasts三表已齐；协议及历史复现入口保留
- [STAGE55_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md) — **结果复核**：0204／0304当帧大差随起点变化；0307恢复附近同起点换头可放大收益，也存在种子与起点依赖的损伤
- [STAGE56_FIRST_TRANSITION_PRESERVATION.md](experiments/06_sgla_uav_tracking/STAGE56_FIRST_TRANSITION_PRESERVATION.md) — 用户已报告REVIEW，decision／policy／scope／paired／focus五份材料已齐；协议保留历史复现入口
- [STAGE56_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE56_RESULTS_REVIEW.md) — **结果复核**：对完整末帧−1.3179、双种子均负；对half +0.3947但3胜9负；控制复现通过，第一步保护仍有收益取舍
- [STAGE57_QUALITY_SELECTIVE_PRESERVATION.md](experiments/06_sgla_uav_tracking/STAGE57_QUALITY_SELECTIVE_PRESERVATION.md) — 用户已报告REVIEW，decision／policy／scope／paired／focus五份材料已齐；协议保留历史入口
- [STAGE57_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE57_RESULTS_REVIEW.md) — **结果复核**：质量选择保留20.52%参考，对完整保持−3.6057、双种子均负；0204双种子无失效，0176／0307恢复及0304保护未保住
- [STAGE58_PROJECTED_ENDPOINT_GRADIENT.md](experiments/06_sgla_uav_tracking/STAGE58_PROJECTED_ENDPOINT_GRADIENT.md) — 用户已报告REVIEW，decision／policy／scope／paired／focus五份材料已齐；协议保留历史复现入口
- [STAGE58_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE58_RESULTS_REVIEW.md) — **结果复核**：全部保持与647／1024步投影生效，对完整保持−2.7575、双种子均负；对同长度+1.6460，主要来自0176，0226单种子恢复崩溃
- [STAGE59_PARENT_COMPONENT_CONTINUATION.md](experiments/06_sgla_uav_tracking/STAGE59_PARENT_COMPONENT_CONTINUATION.md) — 用户已报告条件诊断完成，diagnostic／focus／crossed_matrix／factor_summary／exposure五份材料已齐；保留预定协议及历史入口
- [STAGE59_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE59_RESULTS_REVIEW.md) — **结果复核**：0304特定混合父框有32帧条件修复，0204主要受尺寸来源影响；0226／0176与0307呈不同中心×尺寸交互，没有统一放缩或全视频新方法结论
- [STAGE60_CAUSAL_SEARCH_STATE_VALUE.md](experiments/06_sgla_uav_tracking/STAGE60_CAUSAL_SEARCH_STATE_VALUE.md) — 用户已报告REVIEW、96条完整OOF完成，decision／policy／paired／scope／fit_support五份材料已齐；保留预定设计与历史入口
- [STAGE60_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md) — **当前结果**：response对current +3.7427但CI跨零；future对immediate +3.6137，对response −3.5585且家族区间全负，能力保护未过；四候选与响应分支值得继续检验
- [STAGE61_NATIVE_HISTORY_MATCHED_RECROP.md](experiments/06_sgla_uav_tracking/STAGE61_NATIVE_HISTORY_MATCHED_RECROP.md) — 原始L20复用方案因GPU身份不符未运行；保留固定协议，后续改用L40全部六组重测
- [STAGE61_L40_DEVICE_MATCHED_RUN.md](experiments/06_sgla_uav_tracking/STAGE61_L40_DEVICE_MATCHED_RUN.md) — L40全部108条完整轨迹已完成，保留设备匹配方案与历史执行入口
- [STAGE61_L40_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE61_L40_RESULTS_REVIEW.md) — **完整视频复核**：原生history对current +1.7972、对recrop +6.5150但家族区间跨零，0226恢复保护未过；E收益保留0307损伤与种子依赖
- [STAGE62_CACHED_CANDIDATE_SELECTION_AUDIT.md](experiments/06_sgla_uav_tracking/STAGE62_CACHED_CANDIDATE_SELECTION_AUDIT.md) — 已完成缓存诊断；保留结果前固定方案与4项本机测试记录，0新增训练／图像／网络调用
- [STAGE62_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE62_RESULTS_REVIEW.md) — 全量候选诊断：0226与E0307失败附近候选不足，恢复附近有覆盖丢失和可见却输出差；全量事件仍有漏选机会
- [STAGE63_MATCHED_ONE_CHOICE_CONTINUATION.md](experiments/06_sgla_uav_tracking/STAGE63_MATCHED_ONE_CHOICE_CONTINUATION.md) — 已完成12例／48短分支／6000搜索／0训练，保留结果前固定方案与4项CPU测试记录
- [STAGE63_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE63_RESULTS_REVIEW.md) — 共同起点后继证据：0304一次q0保护、0307／2132在475一次q2选择均实现32帧良好；统一q0回退不足，四负例锚点dense也无良好框
- [STAGE64_CAUSAL_CANDIDATE_RELIABILITY.md](experiments/06_sgla_uav_tracking/STAGE64_CAUSAL_CANDIDATE_RELIABILITY.md) — 已完成96完整轨迹／432800搜索，保留原生fit侧连续质量／离散可靠性匹配协议、3项CPU测试与历史执行入口
- [STAGE64_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE64_RESULTS_REVIEW.md) — 最新完整政策结果：reliability对quality −4.3668、2胜10负，对Hann／q0也负且保护未过；E0304单种子修复，E两种子总体均负
- [STAGE65_CACHED_SAME_STATE_SELECTION.md](experiments/06_sgla_uav_tracking/STAGE65_CACHED_SAME_STATE_SELECTION.md) — 已完成120缓存数据集、135250状态出现次数；保留共同状态四规则／fit支持／10事件的固定协议和3项CPU测试历史
- [STAGE65_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE65_RESULTS_REVIEW.md) — **最新诊断**：0003良好q0被低框替换；0226／0307退化历史常无好候选，E0307唯一机会四规则都漏选；接管保护与弱候选推进需要区分，标准化差异包含跨折参考变化
- [STAGE66_NATIVE_Q0_TAKEOVER.md](experiments/06_sgla_uav_tracking/STAGE66_NATIVE_Q0_TAKEOVER.md) — 已完成原生五组60完整轨迹／270500搜索；保留绝对q0条件保护、fit率匹配门槛协议与4项CPU测试历史
- [STAGE66_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE66_RESULTS_REVIEW.md) — **最新完整结果**：protect对quality −2.8546、对margin −2.4791，0176恢复受损、0307未修复；稳定集0新增失败但保护未过，fit率匹配接近、评价实际率不同
- [STAGE67_MATCHED_VETO_CONTINUATION.md](experiments/06_sgla_uav_tracking/STAGE67_MATCHED_VETO_CONTINUATION.md) — 已完成7支持起点／28短分支／3500搜索，CSV修复成功；保留共同起点2×2协议及恢复历史，不重跑
- [STAGE67_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE67_RESULTS_REVIEW.md) — **最新条件诊断**：0176:310一次放行在protect后继下均值+0.4961、低IoU少21帧；0307对应点反而受损，不能统一放行或当作整视频改善
- [STAGE68_FIT_PAIRED_TAKEOVER_ADVANTAGE.md](experiments/06_sgla_uav_tracking/STAGE68_FIT_PAIRED_TAKEOVER_ADVANTAGE.md) — 已完成384对／84完整轨迹、475468搜索；保留匹配当帧／后继增益与全fit接管率控制的固定协议和测试历史
- [STAGE68_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE68_RESULTS_REVIEW.md) — **最新完整结果**：future AUC66.9254，修复0304退化当帧控制、0176／0147恢复有进展；对Q仅2胜10负，0307仍损伤，保护未过
- [STAGE69_CACHED_PAIRED_GATE_AUDIT.md](experiments/06_sgla_uav_tracking/STAGE69_CACHED_PAIRED_GATE_AUDIT.md) — 已完成384对／7真实成对起点／84缓存诊断；保留固定协议与4项CPU测试历史，零新推理／训练
- [STAGE69_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE69_RESULTS_REVIEW.md) — **最新诊断**：0176:310大恢复收益被future错判且边际特征未越界；0307恢复窗自身历史无好候选，预测与候选支持需分别改进
- [STAGE70_FIT_HISTORY_SOURCE.md](experiments/06_sgla_uav_tracking/STAGE70_FIT_HISTORY_SOURCE.md) — 已完成固定32对／16-8-8混合历史对照、24 fit历史／120完整评价／697592搜索；保留结果前协议与测试历史
- [STAGE70_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE70_RESULTS_REVIEW.md) — **最新完整结果**：mixed_future65.2286，对旧future−1.6968；0304稳定及旧E01／E02方向修复，0204新持续失跟、0307未恢复，保护未过；结束本轮覆盖验证，可准备有限RL对照
- [STAGE71_BOUNDED_SEARCH_PPO.md](experiments/06_sgla_uav_tracking/STAGE71_BOUNDED_SEARCH_PPO.md) — 已完成96实时fit视频／132完整评价、1027908搜索；保留冻结SGLA、PPO与同曝光监督／率控制的运行前协议和验证历史
- [STAGE71_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE71_RESULTS_REVIEW.md) — **最新完整结果**：PPO61.8565对监督−2.3570、两种子均负；0204双种子无失效，0304新增持续失跟、0307弱于监督，保护未过；不直接扩训当前配方
- [STAGE72_CACHED_PPO_DIAGNOSTIC.md](experiments/06_sgla_uav_tracking/STAGE72_CACHED_PPO_DIAGNOSTIC.md) — 已完成132评价缓存／96训练段诊断、1082000次规则决定；保留结果前协议与实现验证，零新训练／图像／SGLA调用
- [STAGE72_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE72_RESULTS_REVIEW.md) — **最新诊断**：同状态PPO／监督当前净差很小，自身滚动候选质量分化；概率多数近半、critic／actor梯度尺度差180～231倍，建议固定预算优化尺度对照，未确立唯一病因
- [STAGE73_SEPARATE_GRADIENT_CLIP_PPO.md](experiments/06_sgla_uav_tracking/STAGE73_SEPARATE_GRADIENT_CLIP_PPO.md) — 已完成96实时fit段／204完整评价、1352508搜索；保留分别裁剪与旧共同裁剪固定对照及验证历史
- [STAGE73_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE73_RESULTS_REVIEW.md) — **最新完整结果**：PPO62.3800，对原PPO+0.5235、两seed反向，对新监督−2.3894；actor步长增大但0304未修复、保护未过，不能只靠裁剪解决
- [STAGE74_DIRECT_FOUR_CANDIDATE_PPO.md](experiments/06_sgla_uav_tracking/STAGE74_DIRECT_FOUR_CANDIDATE_PPO.md) — 已完成96 fit段／144完整评价、1082008搜索；保留PPO直接选q0～q3、匹配监督／旧二元／Q及零恢复支线的固定协议
- [STAGE74_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE74_RESULTS_REVIEW.md) — **最新完整结果**：四选一PPO67.2356，对匹配监督+4.0722且双seed正，对Q+0.9918尚不稳；0176／0307恢复、0304单seed保稳，0003及0304剩余损伤使保护未过
- Stage-75 L20 已完成 13 个起点和 52 条短分支。0003 首败有好框却未选中，0304／2172 在首败前已失去好候选。统一退回 q0 会破坏部分恢复。 [Stage-75 L20 结果复核](experiments/06_sgla_uav_tracking/STAGE75_L20_RESULTS_REVIEW.md) [固定方案](experiments/06_sgla_uav_tracking/STAGE75_FOUR_CHOICE_FAILURE_REPLAY.md)
- Stage-76 L20 已完成 108 条完整轨迹。8 帧收益监督比匹配当帧监督低 6.2616 AUC 点，PPO 稳定保护仍未通过。 [Stage-76 结果](experiments/06_sgla_uav_tracking/STAGE76_RESULTS_REVIEW.md) [固定方案](experiments/06_sgla_uav_tracking/STAGE76_MATCHED_HORIZON_SUPERVISION.md)
- Stage-77 已完成 105 个固定开发状态和 420 次搜索。0304 早期首败仍有公开好框，晚期 15 帧覆盖完整但网格无好框，伴随预测框持续偏宽。 [Stage-77 结果复核](experiments/06_sgla_uav_tracking/STAGE77_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE77_SGLA_MODULE_DIAGNOSTIC.md)
- Stage-78 已完成 9 个短窗。只改搜索尺度使 0304 主窗口均值 IoU 提高 0.1627，但使 0204 已有能力对照下降 0.6535，恢复保护未通过。 [Stage-78 结果复核](experiments/06_sgla_uav_tracking/STAGE78_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE78_MATCHED_SCALE_FEEDBACK.md)
- Stage-79 已完成首帧交叉。0204 限幅后强制一次 q2，整窗 28 帧均达到 IoU ≥ 0.5；0304 的同动作尺度收益保留。 [Stage-79 结果复核](experiments/06_sgla_uav_tracking/STAGE79_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE79_CONFIRMATION_AND_SCALE_CROSS.md)
- Stage-80 完成 84 条完整轨迹。限幅加保护比原尺度 PPO 低 0.9463 AUC，修复 0204 但 0304 首败提前，开发门槛未通过。 [Stage-80 结果复核](experiments/06_sgla_uav_tracking/STAGE80_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE80_FULL_SCALE_CONSENSUS.md)
- Stage-81 完成 84 条完整轨迹。混合尺度比原尺度 PPO 低 3.7707 AUC，两套混合方案均未通过开发门槛。 [Stage-81 结果复核](experiments/06_sgla_uav_tracking/STAGE81_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE81_MIXED_SCALE_FULL.md)
- Stage-82 完成 1,203 个同历史活跃状态。两种来源上混合布局的候选上界均下降，选择器重训门槛未通过。 [Stage-82 结果复核](experiments/06_sgla_uav_tracking/STAGE82_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE82_SAME_HISTORY_SCALE_SUPPLY.md)
- Stage-83 完成 60 条完整轨迹。短时保留旧中心比原尺度 PPO 低 1.6294 AUC，低 IoU 与最长失跟比例均升高，五项开发门槛均未通过。 [Stage-83 结果复核](experiments/06_sgla_uav_tracking/STAGE83_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE83_CONFLICT_CENTER_MEMORY.md)
- Stage-84 完成 27,050 个同历史配对状态。运动生成四候选的最佳 IoU 提高 0.00179，候选供给门槛通过。好框比例按视频等权略升，按帧累计新增 107、丢失 110。 [Stage-84 结果复核](experiments/06_sgla_uav_tracking/STAGE84_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE84_MATCHED_MOTION_CANDIDATES.md)
- Stage-85 完成 12 个开发视频的 60 条闭环轨迹。运动生成四候选比固定布局低 4.5442 AUC 点，五项开发门槛均未通过。 [Stage-85 结果复核](experiments/06_sgla_uav_tracking/STAGE85_RESULTS_REVIEW.md)
- Stage-86 完成 144 对同输出短续跑。单次中点干预在高预测位移组的 16 帧平均 IoU 低 1.38 点。 [Stage-86 结果复核](experiments/06_sgla_uav_tracking/STAGE86_RESULTS_REVIEW.md)
- Stage-87 完成 576 个布局与动作交叉条件。高预测位移组的平均 IoU 损失主要出现在 PPO 改选对照。 [Stage-87 结果复核](experiments/06_sgla_uav_tracking/STAGE87_RESULTS_REVIEW.md)
- Stage-88 完成 144 个固定布局窗口的备选写入诊断。三项信号均未显示稳定收益，实质差异集中于一个源视频。 [Stage-88 结果复核](experiments/06_sgla_uav_tracking/STAGE88_RESULTS_REVIEW.md)
- Stage-89 扩至 28 个开发源组。冲突窗口实质 IoU 差异为 11/38，普通窗口为 3/120。 [Stage-89 结果复核](experiments/06_sgla_uav_tracking/STAGE89_RESULTS_REVIEW.md)
- Stage-90 在冻结续跑中检验三项写入信号。排序 AUC 为 0.467 至 0.550，三项规则均丢掉冲突组唯一的 PPO 独有恢复。 [Stage-90 结果](experiments/06_sgla_uav_tracking/STAGE90_FROZEN_SIGNAL_AUDIT.md)
- Stage-91 两项固定模板外观信号保住一个原 PPO 恢复，也漏掉一个备选恢复，未建立稳定的切换依据。 [Stage-91 结果](experiments/06_sgla_uav_tracking/STAGE91_RESULTS_REVIEW.md)
- Stage-92 筛选后的短时历史在冲突窗口比固定首帧高 4.49 个 IoU 点，完整闭环仍待验证。 [Stage-92 结果](experiments/06_sgla_uav_tracking/STAGE92_RESULTS_REVIEW.md)
- Stage-93 完成 148 条完整轨迹。筛选历史比原 PPO 高 0.74 AUC 点，区间跨零，4 个 PPO 恢复未在规定窗口内保留。 [Stage-93 结果](experiments/06_sgla_uav_tracking/STAGE93_RESULTS_REVIEW.md)
- Stage-94 完成 25 次 PPO 恢复的复查，4 次未匹配事件在原恢复首帧均缺少好框。 [Stage-94 结果](experiments/06_sgla_uav_tracking/STAGE94_RESULTS_REVIEW.md)
- Stage-95 完成 274 个状态回放，四次未保留恢复分为全格点有好框但峰未保留、覆盖充分但定位失败、裁剪外目标。 [Stage-95 结果](experiments/06_sgla_uav_tracking/STAGE95_RESULTS_REVIEW.md)
- Stage-96 完成搜索中心与尺寸交叉。部分恢复关键帧只换尺寸即可补好框，部分需中心与尺寸共同调整。 [Stage-96 结果](experiments/06_sgla_uav_tracking/STAGE96_RESULTS_REVIEW.md)
- Stage-97 完成 50 条后继分支，新增 1 次恢复、丢失 2 次已有恢复，尺寸写入资格仍会接纳错误状态。 [Stage-97 结果](experiments/06_sgla_uav_tracking/STAGE97_RESULTS_REVIEW.md)
- Stage-98 完成尺寸记忆审计。两次恢复损失的首次分叉早于窗口内低 IoU 写入，后期错误引用也确实影响裁剪。 [Stage-98 结果](experiments/06_sgla_uav_tracking/STAGE98_RESULTS_REVIEW.md)
- Stage-99 完成25窗口／75分支。原规则／一次调整／持续更新恢复21／23／20；一次调整保留原21恢复并新增2次，仍有0085中途损伤。 [Stage-99 结果](experiments/06_sgla_uav_tracking/STAGE99_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE99_MATCHED_SINGLE_SCALE_PULSE.md)
- Stage-100 兼容运行完成28组／188轨迹。pulse AUC65.2127，比gated低0.6221；PPO恢复保留22/25，但新增丢失2次并损伤原稳定轨迹，开发保护未过。 [Stage-100 结果](experiments/06_sgla_uav_tracking/STAGE100_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE100_ONLINE_SCALE_EPISODE.md)
- Stage-101 完成1,834次同状态触发回放及61起点／122短分支。当帧跳过为339胜1,157平338负；0307局部改善，0304、0169和0076关键损伤未修复，没有新增完整AUC。 [Stage-101 结果](experiments/06_sgla_uav_tracking/STAGE101_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE101_MATCHED_TRIGGER_REPLAY.md)
- Stage-102 完成85,144个缓存状态审计。0304／0169关键帧有好框却错选，拒绝记忆写入也未阻止错框进入普通搜索历史；0307局部恢复补回，0076仍缺好框。 [Stage-102 结果](experiments/06_sgla_uav_tracking/STAGE102_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE102_CACHED_STATE_CHAIN.md)
- Stage-103 完成28组／188轨迹。两项四候选核验AUC60.9853／62.7318，均低于pair65.8348；局部0304可修复，但排序、稳定性和恢复损失未解决。 [Stage-103 结果](experiments/06_sgla_uav_tracking/STAGE103_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE103_FOUR_CANDIDATE_VERIFICATION.md)
- Stage-104 完成127,716个缓存状态审计。两项新法同状态丢好85／84帧、补好58／70帧；拒绝外观写入仍会提交几何，部分首败已缺好候选，需要检验提交机制的后继效果。无新完整AUC或GPU推理。 [Stage-104 结果](experiments/06_sgla_uav_tracking/STAGE104_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE104_CACHED_REPLACEMENT_STATE_AUDIT.md)
- Stage-105 完成259起点／1,036短分支。共同提交后继平均IoU提高0.01875／0.01219；部分窗口明确受益，也有严重损伤，17个参考恢复均未补回。已有条件机制证据，仍缺可靠线上提交条件和完整验证。 [Stage-105 结果](experiments/06_sgla_uav_tracking/STAGE105_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE105_MATCHED_STATE_COMMIT.md)
- Stage-106 完成56个源组留出的CPU门控和259起点评价。future后继效用+0.012838，但低于always+0.015469、myopic+0.014028及同率+0.013800；0304严重错判保留。另检查85,144个旧状态曝光，无新GPU或完整AUC，门控新增价值尚未建立。 [Stage-106结果](experiments/06_sgla_uav_tracking/STAGE106_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE106_CROSS_FITTED_COMMIT_GATE.md)
- Stage-107 完成56模型审计及85,144旧状态检查。计算重放通过；future训练内自身R²约0.142，留组MSE比固定截距高8.58%；0304损伤与0176救援互为最近训练近邻，0348有模式与状态交互线索，尚无可靠新策略。 [Stage-107结果](experiments/06_sgla_uav_tracking/STAGE107_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE107_FIXED_GATE_SUPPORT_AUDIT.md)
- Stage-108 完成四表示／224门控评价。组合future效用+0.013909，较原future+0.001071但6胜15平7负、最小留一负，仍低于always+0.015469；0304／0176局部决定改善，0348／0349等漏正收益，无新增完整AUC。 [Stage-108结果](experiments/06_sgla_uav_tracking/STAGE108_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE108_RELATIONAL_MODE_COMMIT_GATE.md)
- Stage-109 已完成960个均匀抽样开发起点／1,920短分支。组合future效用+0.003121，低于原future+0.003240、always+0.003937和同率+0.003495；主差−0.000120、6胜11平11负。提交干预有空间，冻结门控筛选价值未建立；无新增完整AUC。抽样表seed为随机种子，PPO种子须从context恢复。 [结果复核](experiments/06_sgla_uav_tracking/STAGE109_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE109_GT_FREE_SAMPLED_COMMIT_BRANCHES.md)
- Stage-110 已完成1,920共同状态／28预测包。校准生成集合最佳IoU0.670099，对固定+0.019737、22胜6负，超过四项生成控制；好框可用率+0.781个百分点但集中0349，新增35／丢21。点预测更差，旧PPO影子选框0.632668→0.479921，尚无完整AUC。 [结果复核](experiments/06_sgla_uav_tracking/STAGE110_RESULTS_REVIEW.md) · [固定设计](experiments/06_sgla_uav_tracking/STAGE110_LEARNED_SEARCH_PROPOSALS.md)
- Stage-111 已完成1,920状态／六集合／69,120次CPU actor。原PPO精确复现；校准臂几何重排和概率汇总IoU降至0.444759／0.436110，均低于原0.479921；GT挑最好排列也仅0.604447，低于q0的0.631912。排列依赖成立，简单换序未修复，需验证匹配新候选的评分训练。无新GPU/训练/完整AUC。 [结果复核](experiments/06_sgla_uav_tracking/STAGE111_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE111_PPO_SLOT_PERMUTATION_AUDIT.md)
- Stage-112 已完成4生成包/32监督评分器、84,468搜索及1,920状态评价。generated4共享评分IoU0.631974，对Hann+0.005600、对旧PPO+0.085889，但对q0仅+0.000062（11胜17负、两训练种子一正一负），低于fixed4/shared。2/4/6/8候选最佳持续增加，实际选择为0.629812/0.631974/0.630023/0.627226；未证明四最优。适配进展成立，稳定净收益和完整AUC未建立。 [结果复核](experiments/06_sgla_uav_tracking/STAGE112_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE112_NESTED_SHARED_SCORING_BUDGET.md)
- Stage-113 已完成五臂196条完整自身轨迹、725,283搜索。native/fixed_hann/generated_hann/fixed_shared/generated_shared AUC63.9605/64.5947/46.2415/64.1309/53.5418；生成共享对固定−10.5890、7胜21负、两训练种子均负，新增5原稳定失败并丢14/26原生参考恢复。固定历史供给进展未转成闭环收益，下一步先统一审计自身历史首个有害替换/供给消失，不扩K或挑seed。 [结果复核](experiments/06_sgla_uav_tracking/STAGE113_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE113_FROZEN_SHARED_CLOSED_LOOP.md)
- Stage-114 已完成196轨迹/203,329帧次缓存诊断。生成共享45失败中20起点仍有好框、17好q0被换低；0089/305、0304/114等随后持续供给不足。也有0080中等q0被换低和原已无好框事件；几何影子改变后继裁剪，但替代未来IoU未观测。零新GPU/训练，接续优先一次q0提交的匹配后继与救援对照。 [结果复核](experiments/06_sgla_uav_tracking/STAGE114_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE114_CACHED_FAILURE_CHAIN.md)
- Stage-115 已完成34起点/68条匹配短分支、8568新搜索。好q0被换低组后31帧IoU0.097310→0.315195、中等组+0.161010；原救援组0.634385→0.304051，6支持视频全负。L20同身份policy控制零漂移，无训练/新完整AUC；局部后继作用成立，统一保q0不足，需可靠公开的提交判别条件。 [结果复核](experiments/06_sgla_uav_tracking/STAGE115_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE115_MATCHED_Q0_CONTINUATION.md)
- Stage-116 已完成56条/58,094状态缓存审计及34匹配起点评价。连续性后继IoU0.299838对原0.180024有局部收益，但仍误伤救援；Hann近统一q0。同率可辨支持前两规则均0，联合仅3上下文/6点且对同率−0.018791；预测兼容AUROC0.830290是待验证线索，尚无可靠提交规则。零新GPU/训练/完整AUC。 [结果复核](experiments/06_sgla_uav_tracking/STAGE116_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE116_CACHED_ONLINE_EVIDENCE.md)
- Stage-117 已完成744去重起点/1488匹配分支、187488搜索。uniform448点主比较预测−连续性−0.001930，区间跨零、13胜1平14负；分层309点也未建立优势。控制同卡零漂移，无训练/新AUC。严格救援支持0：排除旧8救援及末尾约束后母体最多1个，不能据此验证救援保护；需区分普通状态误干预与救援压力目标。 [结果复核](experiments/06_sgla_uav_tracking/STAGE117_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE117_GT_FREE_SAMPLED_Q0.md)
- Stage-118 已完成112个CPU小模型/778点预测。均匀future−myopic−0.003097，CI[−0.006399,−0.000331]，8胜1平19负；旧救援0.634385→0.508706，虽减轻部分损伤仍未保护。future留组MSE比自身常数高14.03%，无新GPU/完整AUC，不采用当前头；后续先查公开证据与训练支持。 [结果复核](experiments/06_sgla_uav_tracking/STAGE118_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE118_PROVIDER_ISOLATED_COMMIT_LEARNING.md)
- Stage-119 已完成778点固定头诊断。均匀5严重错放/3漏益全部未超训练距离参照，旧8救援全部超参照；每头96例只有0～3大正/0～4大负、救援全0。697点近邻无材料性大效应，微小正负冲突不证明关键效果不可分。零新训练/GPU/AUC，优先有对照的关键支持扩展或表示增量。 [结果复核](experiments/06_sgla_uav_tracking/STAGE119_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE119_FIXED_COMMIT_SUPPORT_DIAGNOSTIC.md)
- Stage-120 已完成168个新CPU模型／778点评价、1484次RGB解码。均匀主比较扩展外观对扩展13维+0.001960、CI跨零；原支持外观对旧13维／打乱有次要正向信号，但分层不支持。扩展组合旧8救援全部接管q0、后继0.634385→0.304051，保护失败；无新GPU／完整AUC。 [结果复核](experiments/06_sgla_uav_tracking/STAGE120_RESULTS_REVIEW.md) · [固定协议](experiments/06_sgla_uav_tracking/STAGE120_SUPPORT_APPEARANCE_COMMIT_FACTORIAL.md)
- [STAGE45_TEMPORAL_SIGNAL_PILOT.md](experiments/05_uav_detection/STAGE45_TEMPORAL_SIGNAL_PILOT.md) — **暂不执行**的 YOLOFT 视频检测支线；保留实现与 CPU 检查记录
- [无人机检测研究方向调研](research/UAV_DETECTION_RESEARCH_DIRECTIONS_2026-10-29.md) — 检测文献与机制参考，不代表切换当前主任务

## 交接与维护

- 接续时先读 0.129 的 Stage-138 跨块写回、0.128 的 Stage-137 候选块供给和 0.127 的 Stage-136 整段闭环。确认集与论文目标见 0.48／0.44。 [工作交接与下次会话入口](../WORK_HANDOFF.md#0-下一次对话的直接接续入口)
- [历史写作材料说明](archive/README.md)
- [整理报告](maintenance/CLEANUP_REPORT.md)
- [逐文件整理清单](maintenance/CLEANUP_MANIFEST.json)
- [Stage-18历史预声明修订](../STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json)
