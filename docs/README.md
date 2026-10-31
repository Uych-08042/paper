# 实验说明索引

当前研究结论与接手说明见[工作交接文档](../WORK_HANDOFF.md)。以下保留原阶段协议，文档内容未改写；目录分组只是导航，不代表方法已通过。

历史入口脚本如已归档，先按[代码恢复说明](../archive/README.md)恢复；代码块在仓库根目录执行。

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
- [STAGE59_RESULTS_REVIEW.md](experiments/06_sgla_uav_tracking/STAGE59_RESULTS_REVIEW.md) — **当前结果**：0304特定混合父框有32帧条件修复，0204主要受尺寸来源影响；0226／0176与0307呈不同中心×尺寸交互，没有统一放缩或全视频新方法结论
- [STAGE60_CAUSAL_SEARCH_STATE_VALUE.md](experiments/06_sgla_uav_tracking/STAGE60_CAUSAL_SEARCH_STATE_VALUE.md) — **当前新实验入口**：SGLA自身历史四候选，匹配当帧／8帧后继收益与响应选择；4项真实CPU测试通过，服务器未运行，四候选成本单列
- [STAGE45_TEMPORAL_SIGNAL_PILOT.md](experiments/05_uav_detection/STAGE45_TEMPORAL_SIGNAL_PILOT.md) — **暂不执行**的 YOLOFT 视频检测支线；保留实现与 CPU 检查记录
- [无人机检测研究方向调研](research/UAV_DETECTION_RESEARCH_DIRECTIONS_2026-10-29.md) — 检测文献与机制参考，不代表切换当前主任务

## 交接与维护

- [工作交接与下次会话入口](../WORK_HANDOFF.md#0-下一次对话的直接接续入口) — 先读第0节：Stage-54～59已完成结果与材料、父框分量效应边界、Stage-60设计／CPU验证及当前服务器入口
- [历史写作材料说明](archive/README.md)
- [整理报告](maintenance/CLEANUP_REPORT.md)
- [逐文件整理清单](maintenance/CLEANUP_MANIFEST.json)
- [Stage-18历史预声明修订](../STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json)
