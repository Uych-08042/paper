# 实验说明索引

当前研究结论与接手说明见[工作交接文档](../WORK_HANDOFF.md)。以下保留原阶段协议，文档内容未改写；目录分组只是导航，不代表方法已通过。

历史入口脚本如已归档，先按[代码恢复说明](../archive/README.md)恢复；代码块在仓库根目录执行。

论文图示见 [SGLA 四候选 PPO 框架图 v3](figures/sgla_search_state_v3/README.md)。新版参考 IEEE TPAMI 的 TCTrack++，按 Stage-74／76 展示四候选搜索、在线决策和训练，图注记录截至 Stage-83 的证据。早期的 [v2](figures/sgla_search_state_v2/README.md) 和 [v1](figures/sgla_search_state_v1/README.md) 描述 Stage-54～60 的方案。

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
- Stage-97 将用自身可靠写入历史更新搜索尺寸，开展 50 条匹配后继分支，脚本已就绪。 [Stage-97 协议](experiments/06_sgla_uav_tracking/STAGE97_OWN_TRUSTED_SCALE_CONTINUATION.md)
- [STAGE45_TEMPORAL_SIGNAL_PILOT.md](experiments/05_uav_detection/STAGE45_TEMPORAL_SIGNAL_PILOT.md) — **暂不执行**的 YOLOFT 视频检测支线；保留实现与 CPU 检查记录
- [无人机检测研究方向调研](research/UAV_DETECTION_RESEARCH_DIRECTIONS_2026-10-29.md) — 检测文献与机制参考，不代表切换当前主任务

## 交接与维护

- 接续时先读 0.66 的 Stage-97、0.65 的 Stage-96 结果、0.48 的确认集和 0.44 的论文目标。 [工作交接与下次会话入口](../WORK_HANDOFF.md#0-下一次对话的直接接续入口)
- [历史写作材料说明](archive/README.md)
- [整理报告](maintenance/CLEANUP_REPORT.md)
- [逐文件整理清单](maintenance/CLEANUP_MANIFEST.json)
- [Stage-18历史预声明修订](../STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json)
