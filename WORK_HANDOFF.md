# 工作交接：SGLATrack 无人机跟踪与模型压缩探索

更新日期：2026-11-26（Asia/Hong_Kong）

**Stage-206已实现并通过3项CPU测试，待服务器GPU运行，见0.204。** 在原生L6选层之前保存原输入，MLP只运行一次；选定块使用零初始化的空间残差。三组为搜索自交互／正确模板／fit错配参考，147456参数匹配、原定位监督、12模型3072更新；新跑38原生＋228适配完整轨迹，总266条，另做fit／held读出。总体进度条，9表；确认集封存。本机无真实194特征或GPU，不预报收益。服务器`bash tools/run_stage206_spatial_adaptation.sh`，首次支持L20／L40、同root续跑身份必须一致。 [固定协议与后台命令](docs/experiments/06_sgla_uav_tracking/STAGE206_NATIVE_ROUTED_SPATIAL_ADAPTATION.md)

**Stage-205真实结果已复核，见0.203，接近先验保护，未建立动态选择收益。** 正确关系shrunk IoU .66662117，prior .66660835；3648动作仅33个非L7，一个seed完全固定。对prior／raw／concat／自身无对应期望的主要区间均跨零，仅对错配为正。12系数、16表及全状态／日程复算通过，0新完整轨迹／确认组。停止单独扫系数，该206建议现已实现，见0.204；待服务器GPU运行。 [真实结果与下一步](docs/experiments/06_sgla_uav_tracking/STAGE205_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE205_CROSSFIT_RESIDUAL_SHRINKAGE.md)

**Stage-204真实结果已复核，见0.201。有小幅状态对应和排序线索，整体收益尚未建立。** 正确关系held IoU .66127519，对原输入+.00090616、对错配+.00052654，两区间均跨零，仍低于prior .66660835。自身状态对应+.00159258、名义区间为正，但原输入自身也为正；不能直接归因正确关系。14表全状态／日程／参考映射复算通过，0新完整轨迹／确认集。该建议现已由205实现，见0.202；真实运行待服务器完成。 [真实结果与下一步依据](docs/experiments/06_sgla_uav_tracking/STAGE204_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE204_TARGET_SEARCH_RELATION.md)

**Stage-203真实结果已复核，见0.199，取消裁剪没有建立收益。** point clip／off held IoU .66061429／.66064190；regret clip／off .66104727／.66096011，均低于prior .66660835，五项主要区间均跨零。regret实际梯度范数改变，但平均参数更新量仅+0.10%；8个clip控制权重哈希／分数／动作与202精确一致，14表全状态复算通过。降低裁剪瓶颈假设优先级，不继续扫阈值，不采用。该关系输入对照已由Stage-204完成，真实结果见0.201；0新完整轨迹／确认组。 [真实结果与下一步依据](docs/experiments/06_sgla_uav_tracking/STAGE203_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE203_LOSS_CLIPPING_FACTORIAL.md)

**Stage-202真实结果已复核，见0.197，三个主要比较均未建立收益。** regret held IoU .66104727，对point +.00043298、CI跨零，仍低于prior .66660835；状态对应+.00057849、CI跨零且双seed异号。训练有效，但优化交互明显：regret有1022／1024次裁剪前范数>1，point只有4／1024次；不能直接认定裁剪造成失败。point模型哈希／逐状态分数与200精确一致；13表／代码／原协议及全状态复算通过。0新完整轨迹／确认组，层号不是实际执行块数。该四组匹配训练已由Stage-203实现，见0.198，待服务器真实训练。 [结果与下一步依据](docs/experiments/06_sgla_uav_tracking/STAGE202_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE202_REGRET_MARGIN_ROUTING.md)

**Stage-201已在本机用真实缓存完成，无需服务器补跑，见0.195。** 差分raw／离线视频均值／严格过去均值held IoU .66165173／.66079708／.65945720，固定先验.66660835。因果规则三个主要比较均未建立收益，状态对应增量仅+.00005178、区间跨零；两个种子校正后IoU均下降。降低“单独移除视频均值就能解决”的优先级，不进入闭环采用。6项测试及全状态／9表复算通过，0新训练／模型／图像／完整轨迹／确认组。该同预算监督对照已由Stage-202实现，见0.196，待服务器真实训练。 [真实结果、推导与建议](docs/experiments/06_sgla_uav_tracking/STAGE201_RESULTS_REVIEW.md) · [固定协议与复现](docs/experiments/06_sgla_uav_tracking/STAGE201_SCORE_BASELINE_IDENTIFICATION.md)

**Stage-200真实训练结果已复核，见0.194，有小幅状态对应信号但整体未通过。** 差分held IoU .66165173，低于固定先验.66660835（−.00495661、名义区间为负），对匹配point +.00103745但区间跨零。对自身无状态对应期望+.00086284、区间[+.00015161,+.00161365]，双seed均正；不能称全部无效或动态选层已成功。12表／代码／原协议／指纹核验和逐状态复算通过，0新完整轨迹／确认组。该分数基准诊断已由Stage-201完成，见0.195；两种校正未建立收益。 [结果与下一步依据](docs/experiments/06_sgla_uav_tracking/STAGE200_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE200_WITHIN_VIDEO_DIFFERENCE.md)

**Stage-199真实训练已完成，见0.192，仍未建立held逐帧增量。** 三组held IoU Sigmoid .666462／additive .661409／residual .661389，固定先验.666608。residual三个主要比较均未通过；fit排序77.75%→held54.91%，固定先验59.30%。梯度和训练有效，held约49.67%状态换层；探索性分解显示74.8%训练MSE下降来自视频均值分量，两分量held均变差。决策／manifest／10表／代码／原协议哈希及全部统计重算通过。0新完整轨迹／确认集。该建议已由Stage-200完成，真实结果见0.194。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE199_RESULTS_REVIEW.md) · [原协议](docs/experiments/06_sgla_uav_tracking/STAGE199_FIT_CENTERED_INTERACTION.md)

**Stage-198已在本机用真实回传缓存完成，见0.190，无需服务器补跑。** 固定fit层均值校准后，三种路由held IoU同为.6666083461，但10944条held模型状态全部选L7，与fit标签均值常数完全相同；两主比较无动态增量。原ordered状态对应增量+.00007556、CI跨零。全部21888分数重算差0，无对应期望经全循环移位独立核验差≤5.55e-16，0新训练／网络／图像／完整轨迹／确认组。校准提升是固定偏好修正，不能称新AUC，更不能采用固定L7（194全视频反而受损）。Stage-199现已完成，结果见0.192。 [真实结果与下一步依据](docs/experiments/06_sgla_uav_tracking/STAGE198_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE198_FIT_MEAN_STATE_CORRESPONDENCE.md)

**Stage-197空间顺序对照已完成，见0.189，未建立可靠逐帧增量。** 12模型／21888预测／3072更新，决策与10表哈希相符，汇总复算差≤2.22e-16。held ordered／unordered／global IoU .659283／.659153／.655827，主差+.0001301区间跨零，两个seed为0／+.0002603。排序有小幅正信号，但ordered对自身固定预测均值仅+.0000349，低于fit标签固定均值.666608。主要提升来自固定层偏好变化。0新完整轨迹／确认组，暂不采用。该建议现已由Stage-198完成，见0.190。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE197_RESULTS_REVIEW.md) · [原协议](docs/experiments/06_sgla_uav_tracking/STAGE197_SPATIAL_ROUTING_INTERFACE.md)

**Stage-196同预算损失对照已完成，见0.187，未建立relative收益。** 决策与9表哈希相符，8模型／2048步／14592状态，复算差≤2.22e-16。held mean IoU absolute .655827、relative .654338，差−.00148860，名义CI[−.00311982,−.00020158]。relative实际训练相对误差四模型均改善，但held相对误差变差31/38、差值加权排序下降5.7059个百分点。对自身固定预测均值−.00117592，未过增量要求，0新完整视频／图像／骨干／确认组。下一步建议Stage-197同参数检验保留空间顺序与逐样本无序读取，需区别Stage-164旧局部匹配路线；不能预设空间缺失是主因。Stage-197现已完成，结果见0.189。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE196_RESULTS_REVIEW.md) · [原协议](docs/experiments/06_sgla_uav_tracking/STAGE196_MATCHED_RELATIVE_LOSS.md)

**Stage-195冻结路由审计已完成，见0.186。** 4个R-only模型、7296状态、76条既有路径，决策和9表哈希相符，复算误差≤2.22e-16，CPU／GPU动作差异0。四模型实际抽样MSE均优于初始化和抽样均值控制，但下降的99.15%～99.97%来自共同质量项。held逐帧R对预测均值常数仅+0.0001376 IoU，区间跨零；训练抽样均值与预测均值均偏好fold0 L7／fold1 L10。69/76路径固定，38条L7路径（19源组×2种子）与已有固定L7控制逐帧相同。下一步建议Stage-196同网络同预算比较绝对MSE与仅中心化层间MSE，先检验差异学习，不能宣称共同项已经被证明造成退化。该同预算对照现已完成，结果见0.187；确认集继续封存。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE195_RESULTS_REVIEW.md) · [原协议](docs/experiments/06_sgla_uav_tracking/STAGE195_ROUTER_OBJECTIVE_AUDIT.md)

**Stage-194 同预算模块消融已完成，仍为REVIEW，见0.185。** 38源组、36次训练、798条完整OOF、40517跟踪帧／策略种子，决策和11表哈希均相符，汇总复算最大误差0。原生／仅Q／仅A／仅R／三模块AUC为64.4185／64.7794／63.1884／58.2466／56.1599。R在四个因素背景中均下降6.17～7.86点；20个含R模型15个完全固定选层，全部未选L9。仅Q剔除0176后−0.9668且新增0080、0304失败。目标条件对同参数current主比较+0.4397，区间跨零。原生状态L7即时IoU高于L9，完整固定L7反而−5.9821。下一步优先审计现有R的全池拟合、共同难度分量与相对排序，区分未学会和即时标签／闭环错配；该审计已由Stage-195完成，见0.186。确认集封存。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE194_RESULTS_REVIEW.md) · [原协议](docs/experiments/06_sgla_uav_tracking/STAGE194_MATCHED_MODULE_ABLATION.md)

**Stage-1～193 的综合证据与修改依据已整理。** 用户要求所有改进具有明确理论依据，不能先固定模块或数字再寻找解释。见[全阶段综述与方法依据](docs/research/SGLA_EXPERIMENT_SYNTHESIS_AND_METHOD_BASIS_2026-11-25.md)。当前直接证据最强的是有框漏选及错误写回的后效；相似性路由与任务质量的对齐具有结构依据；时序骨干适配仍是待证假设，尚未证明缺少历史 token 是失跟主因。下文的三模块方案是候选研究设计，L6 接入、记忆长度、投影维数及风险权重均未确定为最优。下一阶段须用参数／训练预算匹配和完整轨迹消融区分表征、路由、预测头的贡献，确认集保持封存。本次整理未运行新 GPU 实验。

**研究主线再次锁定为修改 SGLA 的 Transformer 表征、选层和预测机制，形成新方法。** 用户明确要求改进 Transformer 骨干与选层，同时改进预测头。Stage-194 已实现共享边界的目标条件残差、实际定位质量路由和格点质量评分及其消融。时序历史是后续候选研究方向，是否加入取决于目标条件与同参数当前帧对照的结果，不预先假定可靠历史可用。原生组、仅骨干适配、仅路由、仅预测头和组合分别评价，以同训练机会和正常跟踪入口检验模块贡献。Stage-48／52 的头适配退化、Stage-167／170 的冻结骨干换层退化须纳入解释。仅扩大峰池或调第二峰门槛的优先级降低。

**Stage-193 同裁剪分离第二峰对照已完成，见 0.184。** 第二峰复现 4 个真值挽救中的 2 个，36 个稳定窗有 7 个新增失跟，后效 −0.1229。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE193_SECOND_PEAK_REPLAY.md)

**Stage-192 跨开发源组的候选与写回对照已完成，见 0.183。** 16 个失跟窗中 10 个漏选好框，真值一次写回安全挽救 4 个源组，排除两个先导个案后仍有 3 个。稳定窗存在明显精度损失个案。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE192_DEVELOPMENT_GRID_RESCUE.md)

**Stage-191 网格好框一次写回对照已完成，见 0.182。** `0173` 四种输入中有三条于第 547 帧恢复至窗末，`0107` 两条只维持 5 帧好框。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE191_GRID_WRITEBACK.md)

**Stage-190 裁剪与 CENTER 读出对照已完成，见 0.181。** `0173:503` 和 `0107:161` 的六种输入都有网格好框，原始响应与 Hann 均选中同一坏框。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE190_SUCCESSOR_READOUT.md)

**Stage-189 轨迹持续性审计已完成，见 0.180。** `0173:502` 的三个高 IoU 分支都只维持一帧，503 帧 IoU 降至 0.091～0.094，主要表现为中心跳偏。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE189_CONTINUATION_AUDIT.md)

**Stage-188 补充开发源组与历史位置对照已完成，见 0.179。** 10 个源组提供 2 个完整失跟窗、9 个稳定窗。两个失跟窗均无安全挽救候选，历史位置相对最高 Hann 的后 128 帧平均 IoU 为 −0.00317。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE188_TEMPORAL_SUPPORT.md)

**Stage-187 相邻裁剪预测框一致性已完成，见 0.178。** 一致性和最高 Hann 均选中 2 个长窗挽救；一致性后效较最高 Hann 低 0.0326，并多 1 次事件伤害。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE187_CROP_CONSENSUS.md)

**Stage-186 搜索裁剪取整与移动方向对照已完成，见 0.177。** 同一取整窗口的预测图完全一致，浮点回映仍可改变后续轨迹。0091 的等幅反向像素同样挽救，未建立稳定的移动方向规律。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE186_CROP_QUANTIZATION.md)

**Stage-185 写回框中心与尺寸对照已完成，见 0.176。** 只换中心复现三个长窗挽救源组。0091 中相同尺寸的裁剪平移 1 像素后，Hann 改选并恢复跟踪。0348 仍需要尺寸保护。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE185_WRITEBACK_COMPONENTS.md)

**Stage-184 固定尺寸和层的搜索中心分解已完成，见 0.175。** L9 只改映射的当帧收益最大，完整中心修正在 128 帧后效和挽救上表现更好。下一步检查起点框中心和宽高怎样改变后续搜索。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE184_CENTER_MAPPING.md)

**Stage-183 搜索反馈与六层换层机制对照已完成，见 0.174。** 19 个关键点的全网格均有好框。恢复较早父框使 12 个点恢复常规好框，受保护挽救源组从 2 个增至 4 个。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE183_SEARCH_FEEDBACK.md)

**Stage-182 疑难与稳定窗口六层续跑已完成，见 0.173。** 9 个疑难窗口均无安全挽救备选，唯一挽救发生在失跟前 31 帧的历史低波动窗口。下一步按裁剪和响应网格的变化定位诊断状态，检验搜索反馈是否会限制换层的挽救空间。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE182_LAYER_OPPORTUNITY.md)

**Stage-181 同裁剪双层 CENTER 对照已完成，见 0.172。** 35 个留出窗口缺少达标挽救和伤害，响应联合分的连续收益排序一致率为 44.4%。下一步补充独立源组的换层后效，优先区分备选供给和选层能力。 [实验和结果](docs/experiments/06_sgla_uav_tracking/STAGE181_PAIRED_CENTER_AUDIT.md)

**Stage-180 当前帧 CENTER 匹配证据审计已完成，见 0.171。** 主次峰竞争可提示部分状态存在有益备选。当前原生响应信号尚无法可靠决定是否接受排序层。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE180_CENTER_MATCH_AUDIT.md)

**Stage-179 选择器残差离线审计已完成，见 0.170。** 残差分在 Stage-177 留出续跑的连续收益排序一致率为 48.1%，尚未支持原生重开或非原生层排序。 [实验和结果](docs/experiments/06_sgla_uav_tracking/STAGE179_SELECTOR_RESIDUAL_AUDIT.md)

**Stage-178 选择器 Sigmoid 前分数审计已完成，见 0.169。** 原始分差有变化，但 265 帧最高分均在 L9，六层原始分数的第一主成分解释约 99.95% 的方差。已有后效尚不支持用该分数决定换层。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE178_SELECTOR_LOGITS_AUDIT.md)

**Stage-177 原生提议状态下的换层审计已完成，见 0.168。** 243 个扫描点的选择器输出分差全部为 1.0，35 个实际续跑点均无达标有益换层。未达在线重开条件。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE177_NATIVE_PROPOSAL_AUDIT.md)

**Stage-176 同起点六层续跑已完成，见 0.167。** 七个已知有害排序起点中五个有安全的其他层。16 个抽样普通起点中没有达到有益标准的受保护替代。三个挽救起点原门控均提议留在原生层。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE176_LAYER_EVENT_SWEEP.md)

**Stage-175 失跟与恢复附近的同状态续跑已完成，见 0.166。** 主分析 41 点仅 1 个源组出现严重事件风险，匹配普通窗口也出现 1 例。未达到事件监督试验条件。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE175_EVENT_RISK.md)

**Stage-174 同放行预算的长窗退化监督已完成，见 0.165。** 32 帧与 128 帧重拟合放行完全相同的 40 点。主样本仅一个源组出现严重退化，0304 定点仍被放行，未达到完整视频试验条件。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE174_LONG_HORIZON_RISK.md)

**Stage-173 门控评估层与实际执行层的对齐诊断已完成，见 0.164。** 取样组放行从 33 次降到 13 次，挡住两次短窗有害换层，未恢复失跟仍是 5 次，0304 的严重失跟仍被放行。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE173_GATE_LAYER_ALIGNMENT.md)

**Stage-172 的跨视频和续跑时长诊断已完成，见 0.163。** 同卡复现 33 个起点的自然分支，23 个起点的门控提议与排序推荐不同。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE172_COOLDOWN_HORIZONS.md)

**Stage-171 的同起点交叉续跑已完成，见 0.162。** 0304 第 11 帧沿用原生后较早失跟，0176 第 140 帧需要换层与冷却配合。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE171_MATCHED_COOLDOWN.md)

**Stage-170 的固定冷却完整视频对照已完成，见 0.161。** 成对排序加冷却 AUC 63.779，低于原生 63.961。冷却保护未通过，0176 恢复，0304 失跟提前。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE170_FULL_VIDEO_COOLDOWN.md)

**Stage-169 的定点阻断已完成，见 0.160。** 0304 跳过第 530 帧后避免第 554 帧失跟。0089 暂停 16 帧后第 1194 帧恢复。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE169_TARGETED_LAYER_VETO.md)

**Stage-168 的同状态续跑已完成，见 0.159。** 17 个起点、85 条分支原样复现缓存对照。0304 的单次换层可引入失跟，0089 的持续换层可阻断恢复。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE168_MATCHED_LAYER_ROLLOUTS.md)

**Stage-167 的在线完整视频对照已完成，见 0.158。** 成对排序完整轨迹 AUC 为 62.598，原生为 63.961。成对排序较原提议高 0.215 点，剔除 0304 后低 0.569 点，开发验证未通过。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE167_ONLINE_LAYER_RANKING.md)

**Stage-166 的同帧层间排序已完成，见 0.157。** 固定起点后 32 帧平均 IoU 比四帧对照高 0.014 个点，剔除 0176 后高 0.022 个点，恢复保护通过。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE166_PAIRWISE_LAYER_RANKING.md)

**Stage-165 的候选层重提议已完成，见 0.156。** 保住 0176 的恢复。重提议并重排起点比四帧对照低 0.011 个 IoU 点，剔除 0176 后也低 0.011 个点。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE165_LAYER_REPROPOSAL.md)

**Stage-164 的 L6 局部匹配对照已完成，见 0.155。** 两组投影对照一致。局部匹配比投影对照低 0.186 个 IoU 点，丢掉 0176 的恢复，在线验证门槛未通过。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE164_L6_LOCAL_MATCH_GAIN.md)

**Stage-163 收益预测历史消融已完成，见 0.154。** 长历史收益组比四帧收益组低 0.027 个 IoU 点，剔除 0176 后低 0.028 个点。0176 的恢复保留，在线验证门槛未通过。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE163_HISTORY_GAIN.md)

**Stage-162 历史特征消融已完成，见 0.153。** 16 帧趋势组后 32 帧平均 IoU 比四帧基础组高 0.168 个点，保住 0176 的一次恢复。剔除 0176 后差值为 −0.011 个点。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE162_HISTORY_ABLATION.md)

**Stage-161 同预算的退化预测已完成，见 0.152。** 相对 Stage-160 +0.051 个 IoU 点，去掉贡献最大的视频后略负，新增连续恢复和连续低 IoU 均为零。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE161_LAYER_DOWNSIDE.md)

**Stage-160 保留原生层的六选一已完成，见 0.151。** 校准后换层 68 次、后 32 帧相对原生 −0.067 点，两个新增低 IoU 和两次有效恢复均被挡住。下一轮建议分开估计收益与下行风险。 [实验和运行说明](docs/experiments/06_sgla_uav_tracking/STAGE160_LAYER_VETO.md)

**Stage-159 六选一训练对照已完成，见 0.150。** 后续监督相对原生 +0.192 点，相对当帧监督 +0.267 点，新增连续恢复和连续低 IoU 各两次。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE159_LAYER_SELECTOR.md)

**Stage-158 等预算六选一已完成，见 0.149。** 保护性事后选层后 32 帧相对原生 +3.14 个 IoU 点、相对留出固定 +3.59 点，新增／丢失连续恢复 3／0。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE158_LAYER_ROUTING.md)

**Stage-157 固定加算预算下的写回对照已完成，见 0.148。** 峰值增益／再加重叠门槛的后 32 帧分别比原生低 0.491／0.579 个 IoU 点，均丢掉 0348 的原生恢复。 [实验与结果](docs/experiments/06_sgla_uav_tracking/STAGE157_DEPTH_WRITEBACK.md)

**Stage-156 动态加算触发离线验证已完成，见 0.147。** 低峰值和峰偏离中心能触发加算，但后 32 帧低于原生及同次数随机。预测未来收益仅触发 5 个稳定点。 [结果](docs/experiments/06_sgla_uav_tracking/STAGE156_DEPTH_ROUTING_AUDIT.md)

**Stage-155 只训练串行路径的 L10 已完成，见 0.146。** 适配相对冻结加算的后 8／32 帧平均 IoU 差为 +0.014948／+0.000080，未新增连续好框恢复。 [原始结果](docs/experiments/06_sgla_uav_tracking/results/stage155_results_20261120T134722Z_730065.txt)

**Stage-154 强制加算一块的同裁剪对照已完成，见 0.145。** 47 个起点均执行 L9 → L10。19 个失跟起点新增 1 个当帧好框，后 32 帧新增和丢失的连续好框各 1 次，平均 IoU 差 −0.0098。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE154_RESULTS_REVIEW.md)

**Stage-153 难帧串行加算的运行已完成，见 0.144。** 28 个视频的 29,047 帧均选 L9，0 帧触发加算，两组 AUC 均为 63.9605。串行加算的效果仍待测量。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE153_RESULTS_REVIEW.md)

**Stage-152 只换搜索中心的整段闭环已完成，见 0.143。** 只换中心 AUC 55.4456，相对完整灰度／回归／原生低 1.2245／5.1994／8.5149 点。0089 局部收益在整段保留，0085 的短程关键改选未触发。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE152_RESULTS_REVIEW.md)

**Stage-151 写回框几何交叉对照已完成，见 0.142。** 只换中心的后继 8／32 帧平均 IoU 差为 +0.0687／+0.0451，只换宽高为 −0.0664／−0.0885。0089 的宽高交换使次帧全网格失去好框，0085／690 只换中心即可恢复。32 帧结果对个别源组敏感。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE151_RESULTS_REVIEW.md)

**Stage-150 校准灰度改选的同起点续跑已完成，见 0.141。** 灰度写回相对回归写回的后续 8／32 帧平均 IoU 差为 −0.0130／−0.0457。0089／1159 当帧改善后失跟，0085／690 当帧 IoU 仅 0.2896，续跑后恢复。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE150_RESULTS_REVIEW.md)

**Stage-149 准入后的接受分数消融已完成，见 0.140。** 仅用首帧灰度并留出校准，0089／1159 新选中好框，110 个采样稳定窗口无改选。零门槛选中两个目标起点的好框，同时让 36 个稳定帧跌破好框阈值。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE149_RESULTS_REVIEW.md)

**Stage-148 放宽框间重叠后的外观辅助对照已完成，见 0.139。** 两个新增失跟起点好框在三种评分中都排在备选首位，但相对默认框的分差均为负，两套接受门槛都未采纳。重新校准后，两种外观没有增选纯响应与几何漏掉的好框。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE148_RESULTS_REVIEW.md)

**Stage-147 触发条件覆盖审计已完成，见 0.138。** 在 172 个原生严重低 IoU 的补救机会中，162 个状态至少有一个好框仅被回归分差挡住。失跟起点的 6 个漏选好候选均受框间重叠限制。分差放宽至 0.25 会放行 41 个补救状态，同时把含严重错框的稳定帧从 6 帧增至 137 帧。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE147_RESULTS_REVIEW.md)

**Stage-146 候选冲突时的外观辅助选择已完成，见 0.137。** 4,754 个状态中三组均未改选。281 个原生严重低 IoU 且响应网格有好框的状态里，全网格候选池有 172 个原评分漏选的好框机会，触发条件在外观可用性过滤之前已排除全部。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE146_RESULTS_REVIEW.md)

**Stage-145 同池外观证据诊断已完成，见 0.136。** 全网格池用近期颜色选中 6／19 个失跟起点，保留逐框回归选中的 3 个。首帧灰度相似度在 18 个全网格独有好框机会中选中 3 个，但直接使用在 31／110 个原稳定窗口引入连续低 IoU。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE145_RESULTS_REVIEW.md)

**Stage-144 同帧成对排序已完成，见 0.135。** 普通排序在 18 个全网格独有好框机会中选中 1 个，19 个失跟起点的选择与逐框 IoU 回归完全相同。难例加权选中 0 个全网格独有好框，并在局部峰／全网格的 68／4 个原稳定窗口引入连续低 IoU。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE144_RESULTS_REVIEW.md)

**Stage-143 固定候选池的质量评分已完成，见 0.134。** 645 个原生稳定窗口中，两池质量评分均无连续低 IoU。两池都有好框的 161 个严重低 IoU 帧中，质量评分只选中 8／7 个，原始响应选中 47 个。全网格独有的 18 个好框机会均未被质量评分选中。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE143_RESULTS_REVIEW.md)

**Stage-142 等候选预算的全网格提取已完成，见 0.133。** 281 个原生低 IoU 且网格有好框的帧中，含好框的候选池从 162 帧增至 179 帧，新增 18 帧、丢失 1 帧。0089 第 1159 帧补出好框，第 1165 帧的跨块分支复发仍未补出。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE142_RESULTS_REVIEW.md)

**Stage-141 外圈背景干预已完成，见 0.132。** 0237／243 大裁剪两种填充均小幅提高全网格最佳 IoU，均未补出好框。后七帧改善不一致，0237 全部 64 个标准输出均为 0。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE141_RESULTS_REVIEW.md)

**Stage-140 搜索中心与尺寸交叉已完成，见 0.131。** 0237／243 只恢复原生尺寸比只恢复中心改善更多稠密供给，两个分量都恢复才有好格点框，四组实际输出均 0。后续七帧四组均无好框。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE140_RESULTS_REVIEW.md)

**Stage-139 写回后的裁剪和响应诊断已完成，见 0.130。** 960 个分支帧均完整覆盖目标。0237 写回后的跨块分支全网格无好框，0089 复发时全网格有好框却漏选。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE139_RESULTS_REVIEW.md)

**Stage-138 跨块候选写回与续跑已完成，见 0.129。** 跨块真值最优在 3 个失跟起点均给出好框，后续新增恢复 0，32 帧平均 IoU 比原生低 1.82 个百分点。稳定对照中 3 个跨块最高峰起点误选均未新增后续低 IoU。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE138_RESULTS_REVIEW.md)

**Stage-137 同裁剪补算候选块诊断已完成，见 0.128。** 原选块无好框的 8 个起点中，其他块标准输出补出 0 个。原选块网格已有好框的 11 个失跟起点中，其他块标准输出找回 8 个，包含原块有限峰池漏掉的 3 个。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE137_RESULTS_REVIEW.md)

**Stage-136 同裁剪第二峰整段闭环已完成，见 0.127。** 原生／仅分差／完整规则 AUC 63.9605／61.0823／63.6637。两规则分别引入 2／1 个原生稳定视频的新失跟。Stage-137 以原生为起点单独验证补算候选块。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE136_RESULTS_REVIEW.md)

**Stage-135 同裁剪第二峰改选条件已完成，见 0.126。** 三条规则均新增 3 次恢复且保住原生 6 次，28 个稳定窗口均未触发。仅限分差已获得同样恢复数，完整规则把失跟窗口改选从 10 次降到 6 次。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE135_RESULTS_REVIEW.md)

**Stage-134 同起点改选与短程续跑已完成，见 0.125。** 19 个失跟起点上，Hann 第二峰新增 3 次并丢失 1 次恢复。28 个稳定窗口中强制换第二峰引入 4 个低 IoU 窗口，其中 2 个后续全程低 IoU。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE134_RESULTS_REVIEW.md)

**Stage-133 原生 SGLA 单搜索失跟诊断已完成，见 0.124。** 19 个失跟起点的目标中心全在裁剪内，11 个仍有好格点框，8 个充分覆盖却无好框。原始最高峰在起点救回 0 次，全帧由低转好 47、由好转低 271，固定状态 AUC −0.4218 点。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE133_RESULTS_REVIEW.md)

**Stage-132 首次分叉审计已完成，见 0.123。** 112 对缓存轨迹。匹配状态下缩边与单槽放大均未让已有好候选全部消失。缩边有 1 次仍有好框却漏选、3 次同框但备用请求改变。放大丢失的 0076／1485、0085／1478 两次恢复，都在替换 q2 导致先前选框分叉之后。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE132_RESULTS_REVIEW.md)

**最新完整评价 Stage-131 已完成，见 0.122。** 28 视频／196 轨迹，同 L40 新跑 112 条、复用 84 条。缩边／单槽放大 AUC 51.4511／51.6007，+0.1497、区间跨零、3 胜21 平4 负，双种子反向，丢失缩边的 2 次匹配恢复。仅缩边相对原 Stage-121 −2.5991，区间低于零；放大仍低于原生 12.3598 点。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE131_RESULTS_REVIEW.md)

**Stage-130 保留 q3 的单槽放大已完成，见 0.121。** 346 同帧点，三个单槽方案均保住原 13 个好候选点并新增 5 个；第一位置好选框 3→5 且无丢失，第二／投票 3→4 且各漏掉同一个保存下来的好 q3。位置投票 31 次转第二位置，没增加好候选起点。第一位置可作保留对照，暂不据此部署。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE130_RESULTS_REVIEW.md)

**Stage-129 四槽粗搜与局部放大已完成，见 0.120。** 346 点／265 点启用／2,179 次搜索。好候选新增 8、丢失 6，6 个原好框全来自被替换的 q3；好选框新增 4、丢失 1。226 个原小目标点新增好候选 6，第一和第二局部位置各独占 3 个。265 个真值居中点有 54 个好框，其中 38 点实际局部覆盖不足。固定双槽替换尚不能采用。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE129_RESULTS_REVIEW.md)

**Stage-128 裁剪缩放对照已完成，见 0.119。** 346 个同帧起点，L40 同身份，288 次新搜索，覆盖率逐槽一致。原 226 个目标输入过小起点的短边中位从 1.49 到 4.43 像素，199 个仍低于 8 像素，全部最高 IoU <0.2。全部 346 点的新好候选和好选框均为 0，原 13 个好候选点和 3 个好选框点均保留。单用画面长边限幅不足以恢复。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE128_RESULTS_REVIEW.md)

**Stage-127 覆盖与尺度诊断已完成，见 0.118。** 333 个无好输出起点中，54 个四裁剪均与目标无交集，3 个仅部分覆盖，276 个至少 90% 覆盖。后者有 226 个即使选最小的充分覆盖裁剪，缩放后目标短边也不足 4 像素。该组原帧短边中位 41 像素、裁剪边长中位 5,712 像素、补边 93.8%、输入短边 1.97 像素。改小裁剪能否恢复输出仍待验证。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE127_RESULTS_REVIEW.md)

**Stage-126 交叉对照已完成，见 0.117。** L40 同身份，346 个起点／17 视频，新增 295 条续程和 36,580 次搜索。两路均禁止备用时，独立评分换框后续 IoU +0.030208，主要由 0089、0237、0348 驱动；自然请求下为 +0.005629、区间跨零。0237 第 414 帧留有好候选却漏选，第 525 帧备用裁剪补出好框并跟稳。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE126_RESULTS_REVIEW.md)

**Stage-125 短程闭环已完成，见 0.116。** L40 同身份复用原评分分支，346 个完整窗口／17 个开发视频，579 条新分支，71,796 次新搜索。独立对原评分后续 31 帧 IoU +0.005629、区间跨零；对关系重训 +0.012156，正向均值集中。新增 3 个持续好窗均来自种子 2313，严格救援仅 2 个起点且有一次当帧好框丢失。原结果已归档。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE125_RESULTS_REVIEW.md)

**Stage-124 候选独立评分已完成，见 0.115。** 112 个 CPU 模型、7,561 个状态。普通帧对同批重训的关系评分 +0.001098，区间 [−0.001091,+0.003306]；对原记录 −0.002650。备用触发对原记录 +0.046293，但对同批关系重训区间跨零。普通帧原记录救援的 6 次缓存观察，两种重训头各只保住 2 次（每个模型种子）。暂不支持全局替换。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE124_RESULTS_REVIEW.md)

**Stage-123 固定候选评分对照已完成，见 0.114。** 80 个匹配点共享同一候选集合与备用框分数；固定前三框后改选 5 点、2 点改善当帧 IoU，新增好框采用 1/9→3/9。视频配对差 +0.041936，95% CI [0,+0.106004]。原对照实际只选中 1 个好框，不能据零丢失得出保护结论。0174 与 0089 的重排均撤销，仍未选中备用好框。零新 GPU、图像、训练、完整轨迹。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE123_RESULTS_REVIEW.md)

**Stage-122 缓存诊断已完成，见 0.113。** 56 对轨迹、358 次替换中 80 次前史和前三框观测匹配。补出 9 个好框却只选中 1 个。0174／2313 第 746 帧仅改第 4 个候选，前三框实际改选 q0→q2，单独更换平均尺度差即可复现前三框翻转。0089／2312 第 802 帧反向 q2→q0。新增 GPU 搜索、读图、训练均为零。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE122_RESULTS_REVIEW.md)

**Stage-121 备用搜索完整对照已完成，见 0.112。** L40 四臂同卡重测 28 个开发视频、196 条轨迹。单帧备用搜索 AUC 54.0502，同卡生成共享 53.4103，配对 +0.6399 点、区间 [−0.3953, +2.3317]、5 胜 18 平 5 负，两种子 −0.4079／+1.6876。相对原生仍低 9.9103 点，5 个原稳定视频出现失败，丢 13/26 个匹配恢复。直接对照丢 1/20 个恢复。0174／2313 大幅改善，两次启用均未选中备用槽位，接续优先检查候选重排与首次输出分叉。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE121_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE121_ALTERNATE_SEARCH_STATE.md)

**最终交付是提出并验证一个改进后的 SGLA 无人机单目标跟踪方法，再根据完整实验证据写出论文。** 最终方法须包含可独立验证的 SGLA 跟踪机制改动，例如搜索状态更新、搜索区域构造或输出解码。原生 SGLA 是主基线。先用同预算消融量化模块和选择策略各自的贡献，再做独立源视频确认、结果图表与失败分析。最新Stage-120支持×外观提交消融结果见0.111，固定设计见0.110；Stage-119结果见0.109；Stage-118结果见0.107；Stage-117结果见0.105；Stage-116结果见0.103；Stage-115结果见0.101；Stage-114结果见0.99；Stage-113完整结果见0.97；Stage-112固定状态结果见0.95。

Stage-113 是此前生成布局的完整 L20 对照。28 源组、196 条自身轨迹，native／fixed_hann／generated_hann／fixed_shared／generated_shared AUC 63.9605／64.5947／46.2415／64.1309／53.5418。生成共享对固定 −10.5890、7 胜 21 负，两种子均负，新增 5 个原稳定视频失败，丢 14/26 个原生参考恢复。Stage-121 的新处理采用同轮 L40 生成共享作直接对照。 [Stage-113 结果](docs/experiments/06_sgla_uav_tracking/STAGE113_RESULTS_REVIEW.md)

前次完整Stage-103原生／PPO／pair／four_matched／four_pairwise AUC63.9605／65.0943／65.8348／60.9853／62.7318仍保留作历史背景，保护同样未过。Stage-113没有新PPO训练，不能当作本轮PPO消融。Stage-100的pulse结果见0.71，Stage-99的23/25是窗口成绩，不能当整视频结果。 [Stage-103结果](docs/experiments/06_sgla_uav_tracking/STAGE103_RESULTS_REVIEW.md)

Stage-112已完成4生成包／32监督评分器、84,468搜索及1,920状态评价。generated4共享评分IoU0.631974，对Hann+0.005600、旧PPO+0.085889，但对q0仅+0.000062，11胜17负、两个训练种子一正一负；低于fixed4/shared。2/4/6/8的候选最佳持续提高，实际选择0.629812／0.631974／0.630023／0.627226。适配进展成立，稳定新增价值及完整AUC尚未建立；未证明四最优。 [Stage-112结果](docs/experiments/06_sgla_uav_tracking/STAGE112_RESULTS_REVIEW.md)

Stage-113服务器已完成，decision及八表齐全，原件已归档并核对内部统计。冻结Stage-112四生成包与两训练种子共享评分器，无新训练/确认推理；L20同身份native框误差0，725,283实际搜索。随后Stage-114已完成全量缓存诊断，见0.99；暂不扩K、挑seed或直接加训PPO。 [Stage-113协议](docs/experiments/06_sgla_uav_tracking/STAGE113_FROZEN_SHARED_CLOSED_LOOP.md)

Stage-114已完成196轨迹/203,329帧次诊断。生成共享45次失败中20次起点仍有好框、17次好q0被换低，25次起点已无好框；0089/305、0304/114等出现错选后持续供给不足，0080也有中等q0被换低。零新GPU/训练，已定位可检验触发点；替代未来IoU未观测，接续优先匹配一次q0提交的实际后继分支及救援对照。 [Stage-114结果](docs/experiments/06_sgla_uav_tracking/STAGE114_RESULTS_REVIEW.md)

Stage-115已完成34起点/68条匹配短分支。好q0被换低组后31帧IoU0.097310→0.315195、差+0.217885；中等组+0.161010；但原救援组0.634385→0.304051、差−0.330334，6个支持视频全负。L20同身份policy控制零漂移，8,568新搜索，无训练/新完整AUC；已建立局部动作/反馈后继作用，未建立可靠线上判别条件，不能统一保q0。 [Stage-115结果](docs/experiments/06_sgla_uav_tracking/STAGE115_RESULTS_REVIEW.md)

Stage-116已完成56条/58,094状态审计及34起点固定规则评价。连续性后继IoU0.299838对原0.180024有局部收益，但仍损伤原救援；Hann近似统一q0。同率可辨支持Hann/连续性均0，联合仅3上下文/6点且对同率−0.018791。预测兼容性描述性AUROC0.830290提供待检验线索，尚无可靠提交规则。零新GPU/训练/完整AUC。 [Stage-116结果](docs/experiments/06_sgla_uav_tracking/STAGE116_RESULTS_REVIEW.md)

Stage-117已完成744起点/1488实际短分支，均匀448点、分层309点、交集13。主比较预测兼容性对连续性−0.001930、CI[−0.006461,+0.001589]，13胜1平14负，未建立收益；均匀原policy/连续性/预测后继IoU0.570277/0.570739/0.568809。L20同身份控制零漂移，187488新搜索，无训练/新AUC。严格救援支持0，旧救援8个被排除、另1个末尾不合格，剩余母体最多1个，不能评价救援保护。 [Stage-117结果](docs/experiments/06_sgla_uav_tracking/STAGE117_RESULTS_REVIEW.md)

Stage-118已完成112个CPU小模型／778点预测。均匀主比较future−myopic为−0.003097、CI[−0.006399,−0.000331]、8胜1平19负，未建立后继监督优势；对原policy−0.003530。旧救援后继0.634385→0.508706，损伤虽减轻仍未保护。每头同提供者6组／96例拟合，压力只评；留组future MSE比常数高14.03%。无新GPU／图像／完整AUC，不采用当前门控。 [Stage-118结果](docs/experiments/06_sgla_uav_tracking/STAGE118_RESULTS_REVIEW.md)

Stage-119已完成778点固定模型诊断。均匀5个严重错放／3个严重漏益均未超训练距离参照，旧8救援全部超参照；每头96例只有0～3大正／0～4大负，严格救援全0。697点近邻无材料性大效应，虽常见微小正负混合，不能证明大效果不可分。无新训练/GPU/AUC；接续优先有对照的关键支持扩展或表示增量，单纯分布外拒绝不足。 [Stage-119结果](docs/experiments/06_sgla_uav_tracking/STAGE119_RESULTS_REVIEW.md)

Stage-120已完成168个新CPU模型／778点预测、1484次RGB解码。均匀主比较扩展外观对扩展13维+0.001960、CI[−0.001724,+0.006073]，尚未建立收益；原支持外观对原13维+0.004521、对打乱+0.004214，名义次要区间均正，但分层不支持。扩展组合旧8救援全部接管q0，后继0.634385→0.304051、6视频全负；未形成可靠规则。零新SGLA/GPU／完整AUC，见0.111。 [Stage-120结果](docs/experiments/06_sgla_uav_tracking/STAGE120_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE120_SUPPORT_APPEARANCE_COMMIT_FACTORIAL.md)

2026-11-16新增[当前中英文框架图v4及图注](docs/figures/sgla_search_state_v4/README.md)，参考用户所附两图的分区／配色和时间轴重新绘制，未复用原照片。实线为Stage-113历史预测→32场景／34备选→四搜索区域→冻结SGLA→64→32→1共享质量评分→自身历史；独立虚线区为Stage-120的13＋24维后继提交检验，尚未持续门控完整验证。场景／响应／层突出均示意，不展示PPO、扩散或在线模板更新。新增4320×2400 PNG、可编辑SVG及标准库／Windows原生导出器；不改实验、旧图或冻结模型，未启动Stage-121。

同日按用户进一步要求新增[真实图像简洁框架v5](docs/figures/sgla_search_state_v5/README.md)：去除v4的c／d分区，仅保留a／b与必要历史反馈，删减图内说明。采用公开VisDrone示例实景照片，原图、14裁剪及坐标／SHA／来源记录随图归档；同一图片复用及手工框均为架构示意，不是连续帧或实验预测，不接入SOT实验／确认集。输出中英文3840×1300 PNG与内嵌JPEG的可编辑SVG，图注保留图片来源与示意说明；v4继续保留作完整研究说明。

Stage-111已完成1,920状态／六集合／69,120次CPU actor，原PPO精确复现。校准候选的几何重排／概率汇总IoU0.444759／0.436110，均低于原0.479921；固定q0的六排列离线最好也仅0.604447，低于q0的0.631912。简单换序不能完成适配，接续优先匹配新候选的评分训练；无新完整AUC。 [Stage-111结果](docs/experiments/06_sgla_uav_tracking/STAGE111_RESULTS_REVIEW.md)

Stage-110已完成1,920共同状态／28预测包／35,768搜索。校准生成集合最佳IoU0.670099，相对固定+0.019737，22胜6负、最小留一+0.012099，且超过其他生成控制；好框可用率+0.781个百分点但集中0349，新增35／丢21。点预测未改善，旧PPO影子选框0.632668→0.479921；已有候选供给进展，尚未转化为可用选择或完整AUC。 [Stage-110结果](docs/experiments/06_sgla_uav_tracking/STAGE110_RESULTS_REVIEW.md)

Stage-109已完成960个均匀抽样开发起点／1,920短分支。组合future效用+0.003121，低于原future+0.003240、always+0.003937、同率+0.003495；主要均差−0.000120，6胜11平11负。提交干预有平均空间，旧门控选择价值未建立，无新增完整AUC。抽样表seed字段为随机种子，连接样本需从context恢复PPO种子，详见0.89。 [Stage-109结果](docs/experiments/06_sgla_uav_tracking/STAGE109_RESULTS_REVIEW.md)

最新Stage-108已完成四表示／224门控评价，新增168个CPU模型、0新GPU或完整轨迹。组合future效用+0.013909，较原future+0.001071，但收益集中0304／0176，仍低于always+0.015469；0348／0349等新漏益，留组MSE也恶化。局部提交选择改善成立，整体稳定新增选择价值尚未建立；接续优先考虑代表性开发状态与成对后继支持，避免在259旧起点反复加特征。 [Stage-108结果](docs/experiments/06_sgla_uav_tracking/STAGE108_RESULTS_REVIEW.md)

最新Stage-107已完成固定56模型审计，正规方程与重放检查通过；future训练内R²约0.142、正／负放行84.93%／71.05%，留组后MSE比固定截距高8.58%、正／负放行82.82%／78.39%。存在训练内区分有限和留组损失；0304损伤与0176救援互为最近训练近邻，0348显示模式与状态交互线索。无新训练／GPU／完整AUC，当前策略仍未建立可靠提交条件。 [Stage-107结果](docs/experiments/06_sgla_uav_tracking/STAGE107_RESULTS_REVIEW.md)

最新Stage-106已完成56个CPU提交门控、259起点评价及85,144旧状态曝光。future相对keep后继效用+0.012838，但低于always+0.015469、myopic+0.014028及同率+0.013800；0304有害提交放行、有益提交拒绝，线上判别条件尚未建立。无新GPU或完整AUC，接续优先检查训练支持和大错判，不直接采用当前门控。 [Stage-106结果](docs/experiments/06_sgla_uav_tracking/STAGE106_RESULTS_REVIEW.md)

最新GPU短分支诊断Stage-105已完成259起点／1,036分支。共同改变几何与外观提交，后继平均IoU较keep+0.01875／+0.01219；0176与部分0304窗口有明确匹配收益，也有0304另一窗口从0.7984降至0的严重反例。17个参考恢复均未补回。已有条件机制证据，仍缺可靠线上提交条件及完整方法验证，没有新增完整AUC。 [Stage-105 结果](docs/experiments/06_sgla_uav_tracking/STAGE105_RESULTS_REVIEW.md)

最新缓存诊断Stage-104已完成120条／127,716状态。两项四候选同状态丢好85／84帧、补好58／70帧；错误框拒绝外观写入仍提交普通几何，部分严重首败已经没有公开好候选。没有新完整AUC；后续优先用匹配后继干预检验评分、外观写入与搜索状态提交，不能只扩大候选或清理坏记忆。 [Stage-104 结果](docs/experiments/06_sgla_uav_tracking/STAGE104_RESULTS_REVIEW.md)

Stage-101已完成全1,834触发的同状态回放及61起点／122短分支。跳过一次调整的当帧IoU为339胜1,157平338负，视频等权均差−0.000755；0307／2172恢复附近窗口改善，但0304、0169和0076关键损伤仍未消除。它是诊断，没有新增整视频AUC。 [Stage-101 结果](docs/experiments/06_sgla_uav_tracking/STAGE101_RESULTS_REVIEW.md)

Stage-102已完成全缓存状态链审计，0307／2172跳过466调整后在471匹配恢复，0076仍未匹配。0304／555及0169／653都有好候选却错选，且错误框虽然被拒绝入记忆，仍进入普通搜索历史。下一步优先检验候选核验范围与几何状态提交保护。 [Stage-102 结果](docs/experiments/06_sgla_uav_tracking/STAGE102_RESULTS_REVIEW.md)

Stage-75 L20 已完成 13 个固定起点和 52 条短分支。0003 两个首败起点有好框却未选中，0304／2172 在首败前就已缺少好框。接续先读 0.43 的结果，再看 0.42 的协议与 0.41 的 Stage-74 完整评价。 [Stage-75 L20 结果复核](docs/experiments/06_sgla_uav_tracking/STAGE75_L20_RESULTS_REVIEW.md)

2026-11-01新增交接：**SGLATrackV2源码发现、训练日志判读与后续启发见第0.11节**。当前实验仍使用V1／E，不因V2发布默认切换基线；优先补原始V1权重上的候选选择与同预算控制，再考虑V2迁移及搜索状态—特征记忆一致性。此次只整理知识与建议，未启动新实验。
整理前 Git 基点：`91052cd71d49d1397197ea9d5f532f12d37d2c91`

## 0. 下一次对话的直接接续入口

最新入口为0.204的Stage-206实现与服务器命令，再回看0.203的205真实结果。198～205均已完成，勿重复索取或重跑；Stage-1～205实验Python、原协议与产物保持冻结。206本机合成CPU测试通过，待服务器真实GPU训练与完整视频评价。

本节更新至 2026-11-25。接续先读0.187的Stage-196损失对照结果、0.186的Stage-195路由目标审计、0.185的 Stage-194 完整模块消融结果及全阶段方法依据。每次分析实验结果后，附上下一步实验建议及其验证标准。运行使用总体进度条，复用 `tools/experiment_progress.py` 中的 `OverallProgress`，在后台日志定时刷新。以往全视频结果见 0.143、0.122 和 0.112，候选生成背景见 0.90～0.95 和 0.53～0.54，确认边界和论文目标见 0.48／0.44。 [Stage-196结果](docs/experiments/06_sgla_uav_tracking/STAGE196_RESULTS_REVIEW.md)

需要回溯历史与配对干预时读 0.33、0.31、0.29、0.27、0.24、0.22、0.20 和 0.18。较早阶段的运行状态按各节日期理解。 [Stage-60](docs/experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md) · [Stage-54](docs/experiments/06_sgla_uav_tracking/STAGE54_RESULTS_REVIEW.md) · [Stage-55](docs/experiments/06_sgla_uav_tracking/STAGE55_RESULTS_REVIEW.md)

### 0.1 固定目标与当前进度

- 用户目标是**修改 SGLA 的跟踪机制，形成新方法并完成无人机单目标跟踪论文**。数据目前只有 VisDrone-SOT，原生 SGLA 保留作对照。
- 评价采用首帧GT初始化后的完整自身预测轨迹。当前12个源组、2折每折6 fit／6 held-out已反复用于设计，种子按各阶段固定（Stage-71～74为2171／2172），是开发证据；不要称为新的独立外部验证。
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
- **Stage-68已完成并为REVIEW，decision＋8表齐全。** 实际384对／84完整轨迹／475468搜索、4个新CPU预测器、0 SGLA训练。future对immediate +4.3644（4胜8负），对Q +0.6816（2胜10负），对率控制+0.9084，区间均跨零；0304大幅修复却新增可恢复事件，0307仍损伤，保护未过，见0.29。
- **Stage-69已完成缓存诊断，diagnostic＋12表齐全。** 384 fit对／7真实成对起点／84提供者、94675状态出现，零新训练／图像／推理。future在E01漏掉大救援、E02放行不利动作；0307恢复窗future无好候选而Hann／q0有。见0.31，不重跑，不修改已封存模型或阈值。
- **Stage-70已完成，decision＋11表齐全。** 同视频32对，Hann-only对16／8／8混合历史，24 fit历史／192新起点／120完整评价／697592搜索。mixed_future对原future −1.6968、6胜6负；旧E01／E02方向改对、0304稳定事件消除，却丢失0204救援，0307未修复。见0.33，不执行后验配比搜索；下一步可准备预算受限RL对照。
- **Stage-71已完成并为REVIEW，decision＋9表齐全。** 固定两seed／四遍fit、96训练视频／132完整评价，总1027908搜索；PPO对myopic −2.3570、3胜9负、两seed均负，名义CI负但家族上界微正；0304新增持续失跟、0307弱于监督，0204两seed无事件。保护未过，见0.35，不追加本轮训练。
- **Stage-72缓存诊断已完成，diagnostic＋9表齐全。** 132评价缓存／96训练段，192评分上下文／216400状态出现／1082000决定，零新SGLA／图像／训练；同状态PPO−监督当前IoU仅−0.000078～+0.000480，自身PPO好候选少3.4679个百分点，critic／actor尺度差180～231倍且概率多数近半。优化因果未确立，见0.37。
- **Stage-73已完成并为REVIEW，decision＋12表齐全。** 新8包／96实时fit段／204完整评价、1352508搜索，旧132政策同卡复现。PPO对原PPO+0.5235、9胜3负但两seed+3.6321／−2.5851，对新监督−2.3894；0307贡献主要均值收益，0304未修复、保护未过。见0.39，不重复扩训／调clip。
- **Stage-74已完成并为REVIEW，decision＋11表齐全。** 96实时fit段／144完整评价、1082008搜索，无恢复支线。四选一PPO67.2356对匹配监督+4.0722、双seed正；对Q+0.9918、5胜7负，保护未过。0176／0307恢复及0304单seed保稳有进展，0003末尾短失败和0304另一个seed持续失跟保留，见0.41。扩展输入与动作是整体处理，不是纯动作数量消融。
- **Stage-75 L20 已完成条件诊断。** 5 视频、13 起点、52 短分支、5092 搜索，回传 diagnostic 与 5 表齐全。13 个基准窗口的动作和输出框与 L40 缓存相同。见 0.43。
- **Stage-76 L20 已完成选择策略对照。** 1152 个 fit 起点、108 条完整轨迹。8 帧监督较当帧监督 −6.2616 AUC，低 IoU +10.43 个百分点。PPO 同卡复测最高，保护仍未通过。见 0.45。
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

本节保留结果前方案和测试历史；服务器已完成，实际结果与当前接续以0.29为准，不重复运行。

用户要求下一步实验，已先读0.27并完成[Stage-68固定协议](docs/experiments/06_sgla_uav_tracking/STAGE68_FIT_PAIRED_TAKEOVER_ADVANTAGE.md)，新增`tools/stage68_advantage_common.py`／`stage68_advantage_experiment.py`／`stage68_advantage_summary.py`／`test_stage68_advantage.py`／`run_stage68_advantage.sh`。服务器尚未运行，当前无成绩；Stage-1～67实验／修复Python和旧产物不变，不代commit／push。

**具体目标：** 预测同一起点一次allow(Q提议)相对veto(本帧q0)，之后两条均执行冻结Q时的真实32帧均值IoU差。它是固定Q后继下一步偏离收益，既非绝对未来均值，也非新门控的自洽价值或概率风险。Stage-60负结果仍保留。

**fit隔离是核心：** 只以冻结原生Hann历史作为状态来源；对本折fit组重算本折fit-local Q提议并用它执行后继。不能用这些fit视频的原OOF Q轨迹，其控制器来自相反折，可能含held-out监督。Hann缓存中的OOF Q／q0critic辅助决定不进入输入或标签；只使用其不依赖Q的物理候选／特征／历史。每fit视频完整32帧可用、Q非q0且实际框不同时，在可用时刻列表均匀取最多32点，不读GT／已知事件、不补挑310等点。整折无支持退出，单视频无支持明确记录。

每点共享4实际搜索，强制接受q0／Q提议后，各自在自身历史上执行31帧本折Q；初始候选观测须复现原Hann，后继逐帧审计raw／size／offset、裁剪、解码和原生块。完成采集后才读fit GT产生两种标签：immediate为当帧IoU差、future为32帧IoU均值差。两个岭模型共享128因果维（64候选差＋64 q0状态）、同一对样本／视频等权／ridge0.01／数值零门槛1e-6，仅目标不同。最多4个新CPU预测器，SGLA权重与原首帧模板不动。

七完整政策为Hann／q0／冻结Q／冻结protect／rate_margin／immediate／future。rate_margin在**全量fit原生Hann帧**上按视频等权匹配future预测的非q0接管率，同分组整体接纳，报告目标／实现率、误差、最大同分权重及calibration_states；不以held-out率调门槛。评价历史自演化，实际率不保证相同。未来与当帧两模型输入／目标外配置匹配；在线只选本帧Q提议或q0，不读未来GT。记录索引接管率与实际改变框比例。4个冻结控制在新L40上须精确复现Stage-66。

最多384个fit点／768短分支／96768搜索，7×12＝84条新完整轨迹／378700搜索，合计最大475468搜索／3328276块；pair smoke252、eval smoke896均包含并复用，额外0。主要future−immediate，关键future−rate_margin，另比Q／protect／Hann／q0；12视频bootstrap种子2168、20000次，6 AUC家族端点0.05/12。pilot需六项家族下界>0，对immediate／Q均值至少+0.5 AUC，原生q0稳定集0新增失败、低IoU不恶化、可靠帧CI下界≥−1.0，以及足够支持；不满足即REVIEW。重复开发组、重叠窗口、Hann训练状态与部署自身状态差异均保留为限制。

**验证完成：** `python -B tools/test_stage68_advantage.py -v` 4项全部通过，167.173秒、无跳过；真实CPU Torch／OpenCV／CENTER、完整封存来源链到本轮，2折fit对采集／模型拟合／14条完整微型轨迹／汇总／续跑通过。微型正式预算pair24搜索（12 smoke包含）＋eval224（112 smoke包含）；验证fit标签只来自本折fit、held-out GT改变不影响本折模型、在线和后继动作不读GT、4原控制精确复现、网络／所有旧来源字节不变、NPZ及结果表篡改拒绝。微型4帧来源本折Q可能无提议，**仅测试的fit路径使用确定性因果Q系数曝光**保证双折分支；原评价Q控制和正式模型均未替换，纯语义测试另核对真实折内Q算式及忽略缓存OOF辅助决定。已测试混合可选CSV字段，避免Stage-67导出问题重现。Bash语法、4份新Python解析、check_workspace（505保留／200移动／106模块、0缓存、0换行差异）、177处所涉本地链接和git diff --check均通过；Stage-67原件SHA未变，tracking／lib／已有tools源码未改。

服务器同步本轮新文件后，在`sglatrack-test`环境、`/home/u25600003160604/workspace/paper`执行`bash tools/run_stage68_advantage.sh`。默认输入`output/stage67/matched_veto_continuation_v1`，新空根`output/stage68/fit_paired_takeover_advantage_v1`，GPU0／原V1 checkpoint。流程prepare→pair_smoke→collect→fit(CPU)→smoke→evaluate→summary。可复用已提交缓存续跑；源码一旦正式prepare即冻结。

后台：`mkdir -p output/stage68`后，`nohup bash tools/run_stage68_advantage.sh > output/stage68/run_stage68.log 2>&1 < /dev/null &`；日志在实验根父目录，勿污染空根。脚本通过cat汇总decision／policy_summary／paired／calibration／fit_summary／pair_support／scope／focus／video_influence为`docs/experiments/06_sgla_uav_tracking/results/stage68_results_<UTC>_<PID>.txt`，回传此TXT即可；fit_pairs、pair_scope和事件／逐视频长表留服务器。

### 0.29 2026-11-04：Stage-68完成，后继学习有局部进展，普遍收益及保护未建立

用户回传[原件](docs/experiments/06_sgla_uav_tracking/results/stage68_results_20261103T175411Z_351135.txt)，603991字节／2999行，SHA`e34a961599b348a1ed163cc7e913d42cf89244d6a6815dfc215587c3464a7f32`；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE68_RESULTS_REVIEW.md)。decision＋policy／paired／calibration／fit_summary／pair_support／scope／focus／video_influence共9份齐全。状态REVIEW，pilot=false、retention_guard=false、support=true，不重复索取已有材料。

实际12开发源组、两折6 fit／6 held-out、每fit视频32点，共384对／768短分支，fit96768搜索／677376块；7政策84新完整自身轨迹，eval378700搜索／2650900块；合计475468搜索／3328276块，pair smoke252及eval smoke896包含复用、额外0。新增CPU岭预测器4个、SGLA不训练，Hann状态／本折Q提议及Q后继／128因果维／ridge0.01与目标外匹配配置保持；未来标签是同一起点实际allow−veto的32均值差，不是新门控后继的自洽价值。

**宏成绩：** Hann／q0／Q／protect／rate_margin／immediate／future AUC65.2483／63.4512／66.2438／63.3892／66.0170／62.5610／66.9254；低IoU13.06%／14.46%／11.01%／14.89%／10.88%／15.71%／9.66%；可靠帧70.0548／75.5602／72.5853／72.4639／72.4983／68.2632／72.9460。未来均值最高，但未来及当帧组均在0304有原稳定新增失败1，其他组0。

主要future−immediate +4.3644，名义CI[−0.1986,+12.7879]、6项家族[−0.3052,+16.8506]，4胜8负、中位−0.1340、最小留一+0.2803。0304的+49.2892按12视频记账贡献+4.1074，其他11条合计贡献+0.2570；移除0304仍平均+0.2803，不能说全部收益仅来自它，但不能把修复退化当帧控制写成普遍超越强控制。对rate_margin +0.9084，CI[−0.1170,+2.1662]、家族[−0.2881,+2.6735]、6胜6负。对Q +0.6816，CI[−0.4089,+2.1759]、家族[−0.5211,+2.7602]、2胜10负、中位−0.1467；仅0176+6.4025及0147+5.1201为正，其余10条平均−0.3343。对protect +3.5362、7胜5负；对Hann +1.6771、4胜8负；对q0 +3.4742、3胜9负，名义与家族区间均跨零。

**明确的进展与仍需保护：** 0304 immediate在554失效到末尾，AUC22.2195；future在1092失效、1100开始恢复、末尾未恢复0，AUC71.5086，对Q仍−0.2981。不能把可恢复新事件与终末崩溃等同，也不能忽略严格稳定集保护失败。0176 future 314→320恢复、末尾未恢复0，AUC67.1889；对Q+6.4025、对protect+38.3410、对率控制+5.7327，却对immediate−0.5960、对Hann−0.5530。没有避免首次失效；320与Stage-67相符也不能证明本轮在310执行了相同干预。0147 future1073失效、1116恢复，但末尾仍未恢复1，AUC37.4710，对Q+5.1201；不能写成最终恢复成功。0307 future394失效未恢复、AUC40.3978，对Hann／q0仍−27.3857／−26.1121。0204未来无事件、72.8401（对Q−0.0207）；0226为112→340恢复、54.9821（对Q−0.2176），主要已有收益保留。0237对Q−1.1387为该比较最差视频。

**保护未过原因：** 相对q0，future低IoU平均−4.7948个百分点、最长失败比例−5.2681个百分点，但可靠帧AUC−2.6143，名义CI[−7.6223,−0.0147]，下界未到−1.0；再加0304稳定集新增失败。对Q可靠帧均值+0.3607却1胜11负且CI跨零。事件宏数future1.0833高于Q0.8333，恢复数0.8333也高于Q0.5833，末尾未恢复均0.25；事件次数、低IoU和持续失跟严重度分开解释。

**fit与率匹配：** 可用状态9306，12视频各32点；两折各192。future标签正173／负184／绝对值≤1e-6共27，是否支持放行分歧142对（36.98%，含零附近）；两折mean_future −0.002121／−0.001885，等权−0.002003，mean_immediate等权+0.001249。标签不同不是已证明预测准确。全fit Hann calibration状态6168／7357，门槛0.0035143239／0.0021361568；future率38.37246%／41.33680%，margin38.38088%／41.34394%，误差0.008415／0.007143个百分点，小于最大同分权重。两折等权future39.8546%、rate39.8624%、Q73.1559%。评价宏非q0：future39.64%、rate41.84%、immediate44.34%、Q76.75%、P15.09%；实际框改变37.16%、41.82%、41.65%、72.49%、14.78%。fit率匹配有效，不能称评价实现率或逐状态干预完全相等。对率控制低IoU−1.2157个百分点，名义CI[−2.8538,−0.0178]有改善支持，但非跨指标校正结论，AUC仍跨零。

**本机核对：** JSON指纹、8CSV字节SHA；从84 focus及12 influence重算7宏AUC、6项AUC名义／家族bootstrap和胜平负／中位／最差／留一；24项宏均值关系；48原控制focus及控制宏共同字段同Stage-66；runtime相同、同物理L40、84 head SHA相符，最大decode0.00048828124954525265；84 scope均4搜索／帧、7块／搜索，总预算相符。fit evidence384案例／12×32及全32帧范围、fold与评价相反、2304路径集合；eval168路径集合；calibration率差、同分上界、13525状态合计核对相符。**未读服务器manifest、模型系数、NPZ／receipt正文、fit_pairs、pair_scope、video_summary和events；未独立重拟合、核对均匀选点／逐标签fit隔离、逐帧控制复现或非AUC bootstrap。** 这些由固定服务器代码报告，本机表间验证不替代原始审计；不重跑未改代码的CPU测试或GPU。

指纹：decision `8f1dd49f4bdabf321893c554aac4d51bb25a37d3c5a9e0f7a736483d0a1c063d`；plan `d8ea1c0d4abc4349fd926763ddf708c14b95763b951558eb0459373362823a64`；training `12691a0a9b6bbfaba6cef6511598a20d5c611d99da4a7384c535f059af61a49e`；pair run `1bfd3d2eff3e4ab0a5d48bce713a582fdb4543dcb129fba7706a6026a294add3`；eval run `99a113e3ad7aacb04e627156c16a1c493ee7a80d750fb69288c89cfc8d240d57`。scratch为`D:/Users/23920/Documents/Codex/stage68_bundle_parsed.json`，非新实验产物；原件加-text保留字节。维护检查PASS（505保留／200移动／106模块、0缓存、0换行差异），182处所涉本地文档链接及git diff --check通过；tracking／lib／tools源码未改。

**结果复核当时的下一步（现已完成，见0.31；0.30保留实现历史）：** 保留成对后继学习在恢复及长失跟修复上的进展，目标收敛到保住Q多数轨迹并修复0307。优先用存量缓存诊断同一候选上的Q／q0／immediate／future／rate分歧、fit标签与预测／特征支持、0304新事件及0307恢复窗口。当前影子IoU不是未执行动作的后继收益，必要时另行预定共同起点续跑。小标签均值、Hann到自身历史及Q后继到门控后继失配不直接当唯一病因，也不据本轮再搜阈值／窗口。Stage-1～69实验与修复源码、模型和旧结果冻结，未代commit／push。

### 0.30 2026-11-04：Stage-69已准备，缓存同状态与真实成对目标诊断

本节保留结果前方案及测试历史。服务器现已完成，先读0.31；不重复运行本节流程。

用户要求继续下一步实验。已先读0.29、Stage-68复核及缓存接口，新增[Stage-69固定协议](docs/experiments/06_sgla_uav_tracking/STAGE69_CACHED_PAIRED_GATE_AUDIT.md)、`tools/stage69_gate_common.py`／`stage69_gate_audit.py`／`test_stage69_gate.py`／`run_stage69_gate.sh`。本机没有服务器完整NPZ／模型，**准备时服务器尚未运行、没有新成绩**；Stage-68依然为REVIEW，不因实现完成转为采用。

**问题与固定设计：** 本轮不继续搜阈值。先在384 fit对上核对标签、权重、旧预测系数与样本内误差，再把冻结Stage-68模型应用到Stage-67的7个共同起点，和已有真实allow／veto后继比较符号。7点均用该视频held-out折模型，继续Q的实测32帧目标与训练定义一致；protect后继只作控制敏感性描述。这些仍是消费过的开发案例，不是独立验证，二元影子值也不是新门控继续运行的回报。Hann可能选其他动作，无其真实分支时不虚构未来。

**全缓存与关注窗口：** 84个已存自身历史提供者逐帧复核，七条影子规则都作用于同一当前四候选、不滚动历史。报告当前均值IoU／好候选漏选／q0好→低与低→好／索引及实际框变化、future相对Q／immediate／rate的同状态当帧差；不算影子AUC或用当帧差评估future目标误差。逐视频内平均再宏平均，无逐帧显著性或采用门槛。关注锚点固定C01～09：0304 immediate首次失效554、future首次失效1092；0307 future首次失效394、q0首次恢复474；0176 future首次恢复320；0147 future首次恢复1116；0237 future首次失效240；0204 q0首次失效664；0226 q0首次恢复338。每点前后各30帧、七种提供者，缺事件记录不支持、不替换。

**特征支持和历史分岔：** 原128维pair特征按各折真实fit对、等视频权重求1%／99%逆分布边界，沿用预测器中心与尺度；报告全部／尺度>1e-6活跃维的越界比例及z，不裁剪、不开新OOD门槛。只计Q非q0且实际框改变的查询。fit Hann与held-out Hann同时换视频组、折和提议模型，不把差异当纯历史漂移。另记录future与三控制首次索引／框／所选裁剪／候选／特征分岔，区分无效索引变动和随后输入变化。

**预算与完整性：** 新模型、训练、图像读取、GPU搜索、分支和完整轨迹全部0；重解旧回归仅为系数复核。保留原模型／run／receipt／原控制／折隔离校验，封存新manifest绑定完整旧SHA及新代码；相同输入可重复汇总，旧产物不同则拒绝覆盖。报告仅含来源集合指纹，避免数千路径占满TXT。代码不修改Stage-1～68、模型／阈值或原件，未代commit／push。

**本机验证已通过：** `python -B tools/test_stage69_gate.py -v` 4项、120.691秒、无跳过。三项语义测试检查真实二元分支值、零附近符号、GT-free打分、特征支持、可用好候选漏选、无事件不替换及混合CSV列。完整真实CPU CENTER来源链走到Stage-68后，阻止图像读取／capture并让Torch／OpenCV不可用，验证Stage-69两次汇总一致、49同状态／21匹配／14特征组与7个微型关注行、1个真实成对转移点、旧字节／网络未变及原CSV／模型／缓存篡改拒绝。微型来源2视频×4帧、2 fit对，继承Stage-68仅测试的折内Q曝光以保证分支支持，正式Q与评价控制未替换。CPU测试不是服务器成绩。

补充检查全部通过：3份新Python的AST、全新进程禁止导入Torch／OpenCV仍能加载诊断入口、Bash `-n`语法、187处所涉本地文档链接、`git diff --check`、维护清单505保留／200移动／106活动模块且0缓存／0换行差异；Stage-68回传原件SHA保持不变，tracking／lib及既有tools源码未改。

**服务器运行：** 同步新文件后，在原`sglatrack-test`环境和仓库根执行`bash tools/run_stage69_gate.sh`。默认输入`output/stage68/fit_paired_takeover_advantage_v1`，输出独立新根`output/stage69/cached_paired_gate_audit_v1`，不需要checkpoint／GPU参数。需要完整缓存链，不能仅用已归档TXT。后台执行如下，日志位于实验根父目录：

```bash
cd ~/workspace/paper
mkdir -p output/stage69
nohup bash tools/run_stage69_gate.sh > output/stage69/run_stage69.log 2>&1 < /dev/null &
```

脚本以cat合并diagnostic＋12张紧凑CSV为`docs/experiments/06_sgla_uav_tracking/results/stage69_results_<UTC>_<PID>.txt`。回传该TXT即可；fit_points、scope_video、same_state_video／matched_video／feature_video、focus_frames及fit_reference留服务器，不默认全部回传。下一次先检查这些结果，若需要新因果干预，另行预声明共同起点续跑，不将本轮影子分数当成已实测后继。

### 0.31 2026-11-04：Stage-69完成，恢复预测错误与历史候选不足已区分

用户回传[原件](docs/experiments/06_sgla_uav_tracking/results/stage69_results_20261103T230930Z_471659.txt)，68607字节／504行，SHA`3c2353633f1075382ab5379e99957de4fc1a4cde4b619e38bbb1592a1246a561`；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE69_RESULTS_REVIEW.md)。diagnostic＋12CSV已齐，行数4／8／7／42／49／21／7／14／9／63／36／2；COMPLETE仅为诊断完成，无采用门槛，原Stage-68 REVIEW与成绩不变。

**范围：** 384旧fit对、7个Stage-67真实共同起点、84条自身历史，94675状态出现＝13525×7。两折51499／43176状态；核对旧搜索205996／172704、旧块1441972／1208928，合计378700／2650900，全部是Stage-68旧评价预算。本轮新模型／训练／图像／搜索／分支／完整轨迹均0。来源五指纹与runtime同Stage-68；服务器报告原控制和逐帧决定精确复现、旧系数重构通过。仍为原SGLA V1、固定首帧模板和6＋1结构。

**fit不等于学会恢复：** future两折样本内MAE0.008595／0.003947，正号准确率54.69%／61.46%，预测正组实际正比例44.21%／62.11%；TP／FP／FN／TN为42／53／34／63和59／36／38／59，合计223/384＝58.07%。fold0多数非正猜测60.42%反高于模型，fold1模型则高于多数类50.52%；不能概括为全局随机。两折正组实测均值仅+0.000650／+0.001227，而预测+0.006789／+0.003046，负组实测约−0.0048／−0.0049，存在均值信号但单例区分有限。小MAE及回归整体均值吻合不证明稀有大收益识别。

**真实共同起点的关键新增结论：** future在7个已消费开发起点正号仅2/7一致，immediate 4/7，非总体泛化准确率。E01 protect历史0176:310，future预测−0.032445、实际Q后继allow−veto +0.502603（0.562656对0.060053），protect后继也+0.496058；future仍veto，Q／rate／immediate放行。该点全部128维边际fit分位内、outside=0、max|z|=1.760，不能只归因越界或靠简单OOD报警识别；边际内也不证明联合或语义支持。Stage-68 future自身历史恢复不等于在这个protect起点学会同一救援。E02 0307:370则预测+0.020676而Q／protect真实−0.009306／−0.036971，被future错误allow。统一取消或增强否决均不能同时解决。F01／F02／F03也错符号，F04／P01正确，许多差很小；七点全部保留，不按个例再调阈值。二元窗口值均是强制一次动作后继续Q，不是影子门控自身未来回报。

**同状态与完整轨迹不同：** future对immediate在7种提供者上的当前均值IoU差全部负，−0.000169～−0.000552，而索引有31.82%～34.64%分歧；对Q约−0.000460～+0.000062、索引36.71%～40.60%不同。不能把Stage-68 AUC增益解释成当前平均定位普遍更准，也不能因均值小认定所有动作无用。future自身有好候选85.17%、全四候选低8.90%、所选漏好候选0.81%；Q84.04%／10.12%，immediate79.36%／14.90%，跨历史分布是结果，非独立候选生成器因果增益。

**0307需要分岔前和候选支持：** C03 364～424 future有好候选0/61、四候选全低28；C04 444～504仍0/61、全低35，七影子当前均值0.1653～0.1774。Q／protect／rate／immediate同窗也全无好候选；Hann／q0有39／33好候选帧、全低均0。在这段固定退化历史仅重排四当前输出不能直接给出良好跟踪；未执行连续重排后继，不能说永久无效。需要检验较早状态保护／历史父框或候选机制，尚未定位出裁剪、尺寸或网格哪个是唯一病因。

**0304与收益窗：** C01 524～584 immediate好候选32、全低29，future为61／0。C02 1062～1122 future好候选39、全低0，自身漏4、Hann影子漏2，当前均值0.558788对Hann0.581478；存在少量当帧改选空间，但没有锚点1092替代框IoU和新分支，不能宣称切Hann能消掉完整新事件。0176 290～350 future好候选58／全低0、protect26／32、q00／61；0147 1086～1146 future32／0、Q0／59、immediate8／8；进入状态和候选质量有差异，不能据此认定当前评分更准或0147最终恢复。0204 future等六种历史同窗61帧好候选、q0仅32／全低29，0226 Hann全低61而其他约31～32好帧，已有对比分化保留。

**特征和分岔边界：** fold1标准化外推明显大于fold0，Hann平均max|z|2.43／79.19、immediate2.69／78.65、future2.45／24.11；活跃维越界约1.4%～4.5%，不是越界帧比例。每视频P95的平均不等于合并P95，逐视频长表未回传，不能指认全部极值视频；折间也换视频和模型。E01范围内大错提示外推并非全部解释。36组首框／所选裁剪同帧分岔，候选及特征晚1帧；0304对Q／I／rate框2／7／4、0307三组均框2。0170和0329有索引先变而实际框后变；最早分岔也不自动是致害转移。

**本机已核对：** diagnostic指纹、12CSV字节SHA、五来源指纹及runtime；fit加权组均值／符号／混淆计数；7成对标签与Stage-67的14条Q分支均值、42二元值／遗憾；49当前分解与21同状态差；7自身政策的low／nonq0／changed_box共21项与Stage-68一致；9锚点、63窗口、36分岔次序、2折预算相符。未读取服务器manifest、系数、NPZ／receipt、fit_points、scope_video、逐视频同状态／特征／配对、focus_frames和fit_reference正文，不能独立复算所有逐预测／边界／候选IoU或定位确切致害转移。材料已齐，不默认索取长表或重跑未改代码测试。

指纹：diagnostic `a18a806f1e28d8f04d80cab4e827d2ec005936878b748a72d246529a3e3f90ff`；plan `2caa36bfbc8cc92453cc19558addc687ff474e569d5ed0093f2d50b4bcab8235`；source bindings `b6c91dc51dbefc6528335d68f53f0fb09c41730aaea6b4a1105194e9ff6aeb87`。scratch解析与核对脚本位于`D:/Users/23920/Documents/Codex/stage69_bundle_parsed.json`和`review_stage69.py`，非新实验产物。原件加-text保持字节；未代commit／push，Stage-1～69源码、模型和旧产物冻结。

维护检查PASS：505保留／200归档移动／106活动模块，0再生缓存、0换行差异；196处所涉本地文档链接、复核文档表格列数和`git diff --check`通过。tracking／lib／tools既有源码未改，不因纯文档与结果归档重跑旧CPU或GPU实验。

**结果复核当时的下一步（现已完成，见0.33；0.32保留实现历史）：** 优先验证fit状态来源与稀有恢复覆盖，而不是对这七点调门槛。在固定总配对数、相同128维／ridge／32帧Q标签／零门槛下，比较Hann-only与本折冻结Q／protect生成的fit历史混合，保持immediate／future匹配、Q与率控制，并检查0307分岔前原生恢复能力。新fit历史要本折生成，不将held-out已知正负点直接当训练样本；新阶段单独封存。Stage-51的几何头state-mix是不同处理，不能把它当此处门控目标已验证；完整SGLA研究方向继续保留。

### 0.32 2026-11-04：Stage-70训练历史来源对照已准备，服务器待运行

本节保留结果前固定方案与测试历史。服务器现已完成，先读0.33，不重跑本节流程。

用户讨论后要求执行下一步，已按“先完成一次有边界的状态覆盖验证，再决定小规模RL”实现[Stage-70协议](docs/experiments/06_sgla_uav_tracking/STAGE70_FIT_HISTORY_SOURCE.md)。新增`tools/stage70_history_common.py`／`stage70_history_experiment.py`／`stage70_history_summary.py`／`test_stage70_history.py`／`run_stage70_history.sh`。**准备时服务器尚未运行、无新成绩**；Stage-68／69结论不变，未代commit／push。

**处理：** 每fit视频32对，控制复用Stage-68 Hann-only 32；mixed固定16 Hann＋8 Q自身历史＋8 protect自身历史。Hann16从原32案例按索引均匀取，Q／protect8从各自有完整32帧、Q提议非q0且实际框改变的时刻均匀选，GT／已知事件／七个关注点不参与采样。任一来源配额不足报错，不补其他来源、不拿held-out补齐。新增Q／protect各12条完整fit历史必须用该折fit-local控制器生成，不能使用旧的相反折OOF Q／protect轨迹当训练历史。

**其余固定：** 原始SGLA V1及首帧模板／网络／CENTER头／候选选择器／4搜索／128维／ridge0.01／32帧冻结本折Q后继标签／1e-6门槛不变。mixed当帧和后继目标用同一384对、等视频权重拟合，新增CPU预测器4个；原Hann-only 4模型及率控制完整复用并重构核对，不按结果重选模型。两个来源的rate_margin都在同一全fit Hann总体上匹配相应future预测率，评价自身历史的真实率仍可能不同。新增fit轨迹／分支生成成本单列，不称训练算力完全相等。

**运行身份与预算：** 只用L40、原V1 checkpoint及来源软件／精度；允许物理L40 UUID变化，先逐折重现一个旧fit Hann共享输入，检查raw／size／offset和所有旧观测／决定，共8新搜索。再生成24 fit完整历史108200搜索，192个新配对点／384短分支48384搜索，192个Hann配对直接复用；十政策统一新评价120完整轨迹541000搜索。合计144条新完整历史／评价轨迹＋384短分支，**697592搜索／4883144块**。history256／pair252／eval1280 smoke包含复用，另8次来源复现已计入合计，不重复计费。

**评价与判读：** 原Hann／q0／Q／protect／rate／immediate／future七控制逐帧复现，另mixed_rate_margin／mixed_immediate／mixed_future三组；所有新完整评价都本轮同卡，不混合新旧设备成绩。主比较mixed_future−future，另mixed_future对mixed_immediate／mixed_rate／Q组成4项AUC家族（seed2170、20000次、视频为单位、端点0.05/8）。报告来源×目标交互、当帧来源对照及逐视频影响。保护沿用0新增原稳定失败、对q0低IoU均值不差及可靠帧下界≥−1.0。pilot_pass还要求4比较家族下界>0及对原future／Q≥+0.5；这是强证据标记，不作为尝试RL的硬性前提。

**本轮边界：** 七个Stage-67真实起点仅对baseline／mixed重新打分，14行诊断，不作为训练点或独立准确率。fit历史支持表离线用GT描述覆盖，不反馈采样。12组仍是反复消费开发数据，32帧Q后继不等于新门控自洽价值；无独立泛化、压缩、速度或能耗主张。本轮不运行PPO，结果无论正负先封存，不继续搜配比／时间点／窗口／门槛；有动作后继信号时可进入预算受限RL对照，不要求现监督方法先完美。

**本机验证全部通过：** `python -B tools/test_stage70_history.py -v` 4项、149.798秒、无跳过。语义测试验证16／8／8固定配额、均匀抽样忽略GT与旧OOF决定、本折Q确实重算、配额不足／跨held-out拒绝、生产预算与receipt不可覆盖。完整真实CPU Torch／OpenCV／CENTER来源链到Stage-69后，新流程完成来源复现、fit Q／protect历史、真实allow／veto分支、模型拟合、20条微型完整评价、汇总两次和全流程零推理续跑；原七控制逐帧一致。验证改变held-out GT不影响相应fold的mixed模型、改GT不影响历史／分支／在线动作；旧网络及全部旧字节不变，原件／模型／缓存和报告篡改拒绝。CPU微型仅2视频×4帧，H0／Q1／P1、每视频2对；另外真实读取／审计了复用Hann配对，生产16／8／8由独立语义测试覆盖。微型新增预算8 bridge＋64历史＋48配对＋320评价＝440搜索，复用smoke16／12／80已包含，另8 bridge单列；不是服务器成绩。

测试继承Stage-68仅测试的因果fold-local Q曝光系数，保证4帧微型源有足够可用点，不用GT；正式模型和held-out旧控制均未替换。开发中真实链发现旧历史接口需要controller_design适配，已在新Stage-70包装层修正，旧模块未改。4份新Python AST、全新进程禁止导入Torch／OpenCV仍可加载CPU入口、Bash `-n`、202处所涉本地文档链接、协议表格列数及`git diff --check`通过；维护505保留／200移动／106活动模块、0缓存／0换行差异。Stage-68／69回传原件SHA保持不变。准备与测试期间未修改Stage-1～69实验／修复代码和旧模型／原件。

**服务器命令：** 同步本轮新增文件后，在原`sglatrack-test`环境／仓库根执行`bash tools/run_stage70_history.sh`。默认输入`output/stage69/cached_paired_gate_audit_v1`，新空根`output/stage70/fit_history_source_v1`，GPU0／原V1 checkpoint；需要完整旧缓存链，合并TXT不足以运行。

```bash
cd ~/workspace/paper
mkdir -p output/stage70
nohup bash tools/run_stage70_history.sh > output/stage70/run_stage70.log 2>&1 < /dev/null &
```

流程prepare→source_bridge→history_smoke→histories→resolve(CPU)→pair_smoke→collect→fit(CPU)→smoke→evaluate→summary。日志在根父目录，勿污染prepare空根；已提交缓存可续跑，receipt／源码／模型变化拒绝覆盖。脚本以cat合并decision＋11 CSV为`docs/experiments/06_sgla_uav_tracking/results/stage70_results_<UTC>_<PID>.txt`；回传此TXT即可。fit_pairs／pair_scope／video_summary／events及模型／NPZ／receipt留服务器，报告只放来源绑定集合指纹，避免海量路径回传。Stage-71尚未准备。

### 0.33 2026-11-04：Stage-70完成，固定混合历史没有整体收益，可结束覆盖验证并准备有限RL对照

用户回传[原件](docs/experiments/06_sgla_uav_tracking/results/stage70_results_20261104T044504Z_498206.txt)，65472字节／648行，SHA`97ce85ad448dd98f032ad358d33ee1affb7335e3f2353d8410f074438fcc496c`；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE70_RESULTS_REVIEW.md)。decision＋11CSV齐全，行数10／28／3／4／8／36／24／14／120／120／12；状态REVIEW，来源增益／保护／pilot均false。不重跑、不默认索取长表。

**执行：** 12已消费开发源组，两折各6 fit／6 held-out，每视频32对，mixed16 Hann＋8本折Q＋8本折protect，总384，与Hann-only等量；新CPU预测器4个、0 SGLA训练、0 RL。24 fit历史108200搜索，192新共同起点／384短分支48384搜索、192 Hann起点复用，120完整评价541000搜索，来源复现8，共697592搜索／4883144块。smoke256／252／1280包含复用，额外8 bridge单列且已入合计。七旧完整政策84 focus及宏共同字段同Stage-68，baseline fit／calibration／probe同Stage-69，runtime同一L40且差异空；来源控制逐帧0误差为服务器报告。

**宏结果：** mixed_immediate／mixed_future／mixed_rate AUC63.8024／65.2286／63.3604，低IoU13.73%／11.93%／15.00%，可靠帧69.9575／72.8193／72.4885；新增原稳定失败1／0／0。旧future66.9254、Q66.2438、Hann65.2483、q063.4512不变。主mixed_future−future −1.6968，名义CI[−5.2680,+0.2262]、4项关键AUC家族[−6.9591,+0.3122]，6胜6负、中位−0.00718、最差−20.8107、最小留一−1.9343；区间跨零，不称总体显著下降，也无收益成立。0204贡献−1.7342 AUC，其余11合计+0.03745，去掉0204均值仅+0.04086，不能后验排除最差视频来采用。

**关键比较：** mixed_future对mixed_immediate +1.4262（7胜5负）、mixed_rate +1.8682（6胜6负）、Q −1.0152（4胜8负），关键家族区间均跨零；对q0 +1.7774、Hann −0.0197也跨零。mixed_immediate对旧immediate +1.2414（5胜1平6负），主要来自0304+21.6259，去掉后平均−0.6117。来源平均效应−0.2277、future目标平均+2.8953、交互−2.9382，区间均跨零。

**局部修复与新损伤：** 0304 mixed_future无失效，消除旧future1092→1100可恢复新增事件，AUC71.4507对旧−0.0580；mixed_immediate推迟到1084仍终末失跟，43.8454，不能当完全修复。0204 mixed_future664失效不恢复，AUC52.0294对旧future72.8401−20.8107；Q／旧future／mixed_immediate／mixed_rate均无事件，连q0都在894恢复、53.9217。q0该视频原本有事件，所以新损伤不计入“原稳定新增失败”。0307 mixed_future390失效不恢复，39.7766，对旧−0.6211，对Hann／q0仍约−28.0068／−26.7332；未修复。0176 mixed_future仍314→320、末尾未恢复0，66.7465；mixed_rate却314后不恢复、28.3871。0147 mixed_future首次失效1202、无恢复、末尾未恢复1，较旧37.4710低0.4118；旧1116中途恢复消失，不能因起点推迟称成功。0226 mixed_future126→338、55.0384，恢复保留。

**保护与率：** mixed_future虽然原稳定0新增、低IoU对q0−2.5294个百分点，但可靠帧AUC−2.7409，CI[−7.9004,+0.0687]，下界未达−1.0，guard失败。对旧future低IoU+2.2655个百分点、最长失败+2.4593个百分点，终末未恢复宏数0.25→0.3333、恢复数0.8333→0.5。mixed率控制仍在同一fit Hann6168／7357帧校准，阈值0.0033596622／0.0028033046，future拟合率39.35671%／35.30101%、rate39.35277%／35.30709%，误差0.003933／0.006088个百分点在同分上界内；评价mixed F／R／I非q0率41.59%／40.66%／49.68%，实际框40.30%／40.63%／46.70%，不称真实干预完全匹配。

**覆盖验证发生，但不是全面解释：** Hann可用栏384指原32集合，不是原始全帧候选数；新Q／protect可用9282／9342、各选96，固定每视频配额16／8／8全满足。fit新历史与Hann的实际裁剪显著不同，包含坏轨迹：fold0 0226 Q／P低IoU约93.98%／94.03%，fold1 0304约67.83%／37.33%。全轨迹描述不证明所选8点覆盖关键恢复；本机未读新fit_pairs，不推断具体采样缺陷。future样本内fold0准确率54.69→64.58%、MAE0.008595→0.006945；fold1为61.46→55.21%、MAE0.003947→0.014831。合计58.07→59.90%但训练数据和标签已变，不能当同集泛化提升。

**真实旧起点probe：** mixed future对7点从2/7变4/7，immediate4/7→3/7，都是定向开发案例。E01 0176:310预测−0.032445→+0.008300，真实Q差+0.502603，放行方向修复但幅度仍严重低估；E02 0307:370 +0.020676→−0.012063，真实−0.009306，否决方向修复。F01／F02也修复，F03仍错，F04／P01原正确变错。P01 0204:664新预测+0.217238、真实−0.000926，明显过高。该probe是旧protect历史，不能因同帧664就认定它是新完整0204失跟的直接原因；没有新轨迹共同起点续跑。

**本机核对完成：** decision指纹、11CSV字节SHA／行数／唯一键；源指纹与runtime；84旧focus、7旧宏共同字段、baseline fit／calibration／probe；120 focus重算10宏AUC、28均值、7项AUC bootstrap和4家族区间／胜平负／中位／最差／留一、3因子区间、12 influence；36配额／24fit历史／120held-out身份隔离；scope head SHA和4×7调用、总预算、fit混淆整数与14probe算术及校准。未读取服务器manifest／pair_plan／系数、fit_pairs／pair_scope／video_summary／events、NPZ／receipt／来源bridge正文，未独立重构每个标签／模型／共享maps或核实具体失效因果；这些是固定服务器代码报告。无需重跑旧CPU／GPU。

文档与维护检查通过：交接／索引／Stage-68～70相关六份文档的203处本地链接，Stage-70协议／复核54行Markdown表格列数，回传与归档逐字节一致及-text规则；`check_workspace.py`为505保留／200移动／106活动模块、0缓存／0换行差异，`git diff --check`通过。`git ls-files --modified -- tracking lib tools`为空；本次仅结果归档与文档更新，不重跑未改的实验测试。

指纹：decision `7a618d0d2bd31e7dc53a634b439e894da3376887f9f978338021f4af5f79e227`；plan `5c8e81a8d28e57fff5b567782c3924f57fbfb7769686018def211ec82a75a257`；training `25892a2b9e6692391bf5e2ba97fef7aeccefdeb1d40491594e534dc9767f2654`；history run `c8c46476b32f870f269a3cfdc0c4cc12d5a9ce86cf9903eea6e855bbf5a2f23d`；pair plan `885db635c27abe47ec8e6d492c932d6bda22c55ca308100244595ae5ea4d0cca`；pair run `43f1222b7483883615b3aa71d36d2c69a26c75f50c4ddb30ef1511567290d601`；eval run `0d6fbfa5cf7256d587f1cfa3b2c6918275a0ebe8b00db8ac4076c6de6f65b236`。scratch为`D:/Users/23920/Documents/Codex/stage70_bundle_parsed.json`及`review_stage70.py`；原件-text保护字节。未改Stage-1～70实验／修复代码、模型或旧结果，未代commit／push。

**结果复核时提出的下一步（现已实现，见0.34）：** 这次约定的一轮覆盖验证已结束，不继续扫混合比例或门槛。准备有预算上限的搜索决策PPO对照：冻结原SGLA，先保留q0／接受Q提议二元动作，在自己实际滚动轨迹上优化累计收益，GT只用于fit奖励／离线评价；旧缓存不充当任意动作环境。保留原Q／future及同容量／表示、匹配数据暴露与预算的监督控制，预定种子、训练量和评价，不在12组反复挑checkpoint。现监督未过不阻止RL小试验，PPO也不保证修复候选不足／特征信息不足；改记忆或动作另列处理，独立确认需另行预定。

### 0.34 2026-11-04：Stage-71运行前固定方案与验证历史

本节保留结果前固定方案与测试历史。Stage-71服务器现已完成，先读0.35，不重跑本节流程。

用户要求进行下一步，已实现[Stage-71运行前固定协议](docs/experiments/06_sgla_uav_tracking/STAGE71_BOUNDED_SEARCH_PPO.md)，新增`tools/stage71_ppo_common.py`／`stage71_ppo_runtime.py`／`stage71_ppo_experiment.py`／`stage71_ppo_summary.py`／`test_stage71_ppo.py`／`run_stage71_ppo.sh`。准备时Stage-1～70实验及修复源码、旧模型与结果保持冻结；Stage-70仍REVIEW、尚无Stage-71服务器精度，未代commit／push。

**研究处理：** 继续原生SGLA V1／VisDrone-SOT、6＋1独立候选结构、原首帧模板。每帧同样四次搜索，冻结本折Q提出候选，二元动作保留q0／接受Q；无有效异框提议时强制0。沿用128维`[fQ−f0,f0]`，原fit Hann中心／尺度、clip±10，无GT／帧号／视频名／未来观测输入；q0无提议时前64为0，后64仍描述q0。actor和critic各128→32 Tanh→1、4161参数，各臂合8322，独立无共享主干；SGLA不进优化器。

**实际交互与匹配监督：** PPO随机动作形成自己的预测历史，完整fit视频从首帧到自然终止，失败后不GT重置，不用旧缓存模拟任意动作；每段期间policy固定，结束才更新。奖励实际框IoU，gamma0.99／lambda0.95，整段GAE末尾bootstrap0。PPO actor clipped0.2、eligible advantage标准化；myopic actor对同状态Q IoU>q0＋1e-6作BCE，平局q0。两臂同初始化、归一化、PPO状态矩阵、mini-batch排列及更新次数、相同return critic监督和entropy0.01；value MSE0.5，Adam LR3e-4／eps1e-5、clip1、无WD。myopic不额外采自己的fit历史，评价各自滚动；这是累计PPO对当帧监督的共享曝光比较，不声称只隔离长程目标。

**预定有限预算：** 旧12消费源组、两折各6 fit／6 held-out，2171／2172，固定4遍fit，4 epochs／mini256、固定末checkpoint。4 PPO＋4 myopic模型包、96完整fit视频／108200步／432800搜索，每臂1888优化更新；没有按GT失效选窗口或预热挑权重。五旧控制q0／Hann／Q／future／mixed_future各12条，3新政策PPO／myopic／ppo_rate各2×12，合132完整评价／595100搜索；额外每折4次来源复现共8，正式总1027908搜索／7195356块。train smoke复用首job首完整视频及更新；eval smoke首视频前32帧×11＝1408搜索复用。更长视频有更多mini-batch，不写每视频梯度等权。

**控制与保护：** 旧五政策同卡逐帧精确复现，解码差沿原固定容差。ppo_rate在同折全fit Hann状态上按Q gain匹配最终PPO确定性放行率，不使用GT、不新增推理；实际评价率可以不同。主要PPO−myopic；四项家族AUC为对myopic／Q／future／ppo_rate，视频内先平均种子、bootstrap2171／20000，关键区间分位0.05/8和1−0.05/8。PROMISING需四下界>0、对myopic及Q均≥+0.5、每种子对两者均正、原q0稳定任何种子0新增、低IoU差≤0、可靠帧差下界≥−1，并至少4视频放行且4视频否决；否则REVIEW保留。仍属消费开发证据，不据PPO本身宣称创新或外部泛化。

**续跑与源码：** manifest封存完整Stage-70及上游链、源码、设计和L40身份，仅允许物理UUID变化且重测全部对照。每段NPZ／receipt绑定更新前模型，更新保存JSON参数、Adam状态及前链指纹；行为种子和排列由job／段编号固定重建。已提交段在中断后可不推理而补CPU更新，已提交更新不重复；未提交视频按同一policy与seed整段重做，异常未提交尝试耗时不冒充零成本。优化器JSON元组／列表序列化已在新模块规范化，不修改旧模块。

**服务器入口：** 在仓库根／`sglatrack-test`同步新增文件后运行`bash tools/run_stage71_ppo.sh`。默认完整输入`output/stage70/fit_history_source_v1`，新空根`output/stage71/bounded_on_policy_search_ppo_v1`，原V1 checkpoint／GPU0；只有合并TXT不能运行。断线后台：

```bash
mkdir -p output/stage71
nohup bash tools/run_stage71_ppo.sh > output/stage71/run_stage71.log 2>&1 < /dev/null &
echo $!
```

流程prepare→source_bridge→train_smoke→train→smoke→evaluate→summary，日志在新根父目录。脚本用cat合并decision＋9表为`docs/experiments/06_sgla_uav_tracking/results/stage71_results_<UTC>_<PID>.txt`；回传这一份即可。默认policy8／paired28／seed_effects2／calibration4／training_scope4／fit_support24／focus132／scope132／video_influence12；训练96行及seed_video／video／events长表与模型／优化器／轨迹留服务器。

**本机验证通过：** `python -B -m unittest discover -s tools -p test_stage71_ppo.py -v`四项、179.968秒、无跳过。覆盖GAE自然终止／延迟信用、无提议动作约束、128维因果状态、生产预算／fit-held-out隔离；真实CPU float64 actor／critic、NumPy推理一致、PPO clipping绑定及双臂实际梯度、Adam JSON续跑逐值相同。真实CENTER完整来源链到Stage-70后，新阶段来源复现、实时自身PPO收集、同曝光双臂更新、22条微型完整评价与两次汇总通过，五旧控制精确复现；模拟提交NPZ后更新前中断，续跑复用该轨迹，不重做已提交优化。变后续GT只改变奖励／标签、不变动作／历史；本折held-out GT不可读时，该折fit更新仍相同。旧网络／全部旧文件字节不变，checkpoint／轨迹／上游CSV／结果篡改拒绝；全已提交流程可零推理续跑，汇总无需Torch／OpenCV。微型为2视频×4帧、2遍、hidden4／epochs2，32环境步／128训练搜索、22评价／352搜索、8 bridge，合488；生产4遍／hidden32／epochs4由独立预算语义测试覆盖，不把微型称服务器精度。

最终审查将控制器初始化种子限定为CPU default_generator，避免`torch.manual_seed`同时重设原SGLA的CUDA随机流；补充禁止这两个CUDA／全局种子入口且CPU RNG前后相同的真实Torch测试，定向测试2.544秒通过，CPU初始化分布不变。五新Python AST、纯NumPy CLI、Bash `-n`、所涉四份文档199处本地链接／协议24行表格、`git diff --check`通过；维护505保留／200移动／106活动模块、0缓存／0换行差异。Stage-70原件SHA保持`97ce85ad448dd98f032ad358d33ee1affb7335e3f2353d8410f074438fcc496c`，旧tracking／lib／tools已跟踪源文件无修改。`.gitattributes`新增Stage-71 shell LF规则，未代commit／push。

### 0.35 2026-11-04：Stage-71完整PPO比较完成，未超过匹配监督，保护未过

用户回传[原件](docs/experiments/06_sgla_uav_tracking/results/stage71_results_20261104T114757Z_599124.txt)，60239字节／661行，SHA`19a144d62588c310566fb01e901beca121122f078f79581eb8de4279306ab519`；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE71_RESULTS_REVIEW.md)。decision＋9表齐全，行数8／28／2／4／4／24／132／132／12；REVIEW，support=true、seed／guard／pilot=false。不默认索取长表、不重跑旧实验。

**实际执行：** 原SGLA V1／首帧模板／四搜索冻结，128维因果状态，二元q0／Q，4 PPO＋4匹配myopic模型包。12消费源组、两折各6 fit／6 held-out、2171／2172、四遍fit，96完整训练段／108200步／432800搜索，每臂1888更新；132完整评价／595100搜索，额外bridge8，合1027908搜索／7195356块。train smoke2392（首训练视频598帧）／eval1408包含复用，8额外已入合计。五旧控制60 focus、宏共同字段及scope与Stage-70一致；同一物理L40 UUID、checkpoint／模型状态，runtime差异空，服务器报告逐帧误差0；最大解码0.00048828125沿原容差。

**宏结果：** PPO／myopic／ppo_rate AUC61.8565／64.2136／61.1751，低IoU16.22%／12.81%／18.54%，可靠帧70.1909／72.6735／68.6359，原稳定新增失败均1（0304，任一seed计视频）。旧Q66.2438、future66.9254、mixed65.2286、Hann65.2483、q063.4512不变。主PPO−myopic −2.3570，名义CI[−5.5489,−0.1038]、四关键家族[−6.5281,+0.0020]，3胜9负、中位−0.05088、最差−15.2801、最小留一−2.5920；名义提示下降，不能写家族校正后显著下降，更不能据跨零称两者等效。

**其他比较：** 对Q −4.3872、2胜10负，家族[−14.8803,+1.9869]；对future −5.0689、5胜7负，家族[−16.0528,+1.7657]；对rate +0.6814、6胜6负，家族[−7.0232,+8.0276]；对mixed／q0／Hann为−3.3721／−1.5946／−3.3918，名义区间均跨零。PPO对监督低IoU多3.4141个百分点、名义CI[+0.2516,+7.9371]、0改善5平7恶化；最长失败多3.5067个百分点，可靠帧−2.4826。对q0保护三项都不满足：0304新增长期失败、低IoU+1.7661个百分点、可靠帧−5.3693，CI[−13.6227,−0.0455]，下界未达−1.0。

**逐种子与集中损伤：** 2171 PPO60.2778／监督62.5533／rate62.8340，主−2.2755、对Q−5.9659／future−6.6476／rate−2.5561；2172 PPO63.4352／监督65.8738／rate59.5163，主−2.4386、对Q−2.8085／future−3.4902／rate+3.9190。两seed均不支持主比较，2172对q0仍约−0.01593，不能挑它采用。0304与0307主差−10.7260／−15.2801，按12视频贡献−2.1672，其余十条合计−0.1899；去掉两条后均值仍−0.22784。主比较正视频仅0003／0226／0204，均不足+0.23。

**关键轨迹：** 0304原q0／Hann／Q／mixed均无事件，旧future1092→1100可恢复；PPO2171／2172在1094／554后均终末失跟，AUC44.1297／22.0649；监督1083／1082后也终末失跟、43.8923／43.7543。PPO2171相对监督+0.2374仍是受损，2172−21.6894。0307 PPO2171在395失效无恢复、41.2573；2172在398→475首次恢复但末尾仍未恢复1、53.5479；监督398→470／400→472、66.1020／59.2634，末尾均0，主seed差−24.8447／−5.7155。PPO虽然比旧Q／future平均约+6.70／+7.00，仍低于q0／Hann约19.11／20.38。

0204两臂双seed均无事件，PPO72.6229／72.8246、监督72.4626／72.5299，保留救援但非PPO独有。0176 PPO15.9263／60.6452，对应158后不恢复／381→386首次恢复，末尾均1；监督16.0860／62.8449也末尾均1；旧Hann／future末尾0。0147四条均无恢复，PPO首次1073／1263、监督均1266；PPO35.0794／37.5204低于监督37.6019／37.6190。0226 PPO两seed338首次恢复、监督340，接近原q0；rate2172在119后不恢复、AUC7.2683，构成PPO对rate宏正差主要来源：该视频贡献+1.9904，去掉后均值−1.4280。PPO每seed末尾未恢复5视频（0176／0304／0307／0147／0342），监督4视频，宏0.4167对0.3333；首次恢复不代表末尾成功。

**训练与率支持：** 每fold／seed均24段，fold0各24672步／432更新，fold1各29428步／512更新；每臂8322、actor／critic各4161，SGLA trainable0，同初始化／曝光／mini-batch均true。PPO最大actor梯度0.4399／0.5822／1.3495／1.7974，监督1.0751／0.8764／1.4742／1.4641，不能用非零梯度证明收敛。fit eligible内随机接受约50.14%／50.04%／50.48%／50.46%，不证明没有学习或必须扩训；逐段entropy／KL／value／clip仍在服务器。四rate门槛0.00422113／0.00401534／0.00222027／0.00120544，PPO拟合率33.43367%／34.64453%／40.56524%／51.65520%，误差0.003496／0.019827／0.008419／0.001501个百分点在同分上界内；同fit Hann6168／7357状态。评价实际改框PPO40.37%、监督22.51%、rate41.59%；监督较少干预并存较好表现，但没有匹配监督率来证明数量是唯一解释。

**本机复核：** decision指纹、9CSV字节SHA／行数／唯一键、Stage-70来源与runtime；五旧政策60 focus和宏共同字段／scope；132 focus重算8宏AUC、视频内种子平均／任一seed新增／末尾数，28配对均值、7项AUC bootstrap／胜平负／中位／最差／留一及4家族，2 seed宏差／12 influence；24 fit支持、4训练scope、132评价fold身份／4×7预算／1888更新／总数及4校准差额。未读取服务器manifest、96行训练动态、系数／Adam、NPZ／receipt、seed_video／video／events及bridge正文，未在本机逐更新重构或证明新失效因果；服务器审计与表间验证分开陈述。材料已齐，不重跑未改的CPU／GPU。

维护及文档验证通过：八份相关文档216处本地链接、Stage-71协议／复核64行表格列数、上传与归档逐字节一致及-text规则；`check_workspace.py`为505保留／200移动／106活动模块、0缓存／0换行差异；`git diff --check`通过，`git ls-files --modified -- tracking lib tools`为空。本次仅结果归档和文档更新。

原件-text；decision `abe82fceed4098b1b1b48f8bd9e1df13038239689dcbcf4d132d61faddefb931`；plan `06a61806fe66adaaf453d12a43f657c8e5e6ddb81090076adae290b62be2f162`；training `8be277dfaf308405fd6c6ee102b5652095147aed99dc93088bba8735ead0347b`；run `e2ebcf9184d13805260d48c06d417c0d715c59edd3e375ae511a686e75c14e97`；bridge `a92cccbd6010497a454492e7859f720b3a5c3ac19e3b3a144ff61170e9600d33`。scratch为`D:/Users/23920/Documents/Codex/review_stage71.py`及`stage71_bundle_parsed.json`。Stage-1～71实验／修复源码、模型和旧结果冻结，未代commit／push。

**结果复核时提出的下一步（现已完成，见0.37；方案见0.36）：** 不把当前PPO替换Q／future，不直接扩训或调奖励；用现有缓存将PPO／myopic放在同状态上核查0304／0307选择与候选可用性，并读取既存训练动态检查policy／value／梯度尺度。保留全12视频和正负控制；缓存重评分不能声称任意动作的真实未来，若需因果续跑再单独预定有限共同起点分支。此结果不否定所有RL／SGLA搜索控制；改变状态、动作或训练机制应明确新处理。

### 0.36 2026-11-04：Stage-72固定诊断方案及实现验证历史（结果见0.37）

用户要求进行下一步，按0.35判断实现[Stage-72协议](docs/experiments/06_sgla_uav_tracking/STAGE72_CACHED_PPO_DIAGNOSTIC.md)，新增`tools/stage72_ppo_common.py`／`stage72_ppo_audit.py`／`test_stage72_ppo.py`／`run_stage72_ppo_audit.sh`。以下保留结果前方案和实现验证，当时服务器待执行；现已完成，结果见0.37。不更改旧实验／修复代码、模型和结果，不执行新的RL或SGLA前向，Stage-71仍REVIEW。

**同状态检查：** 读取全部132自身评价轨迹、148775源状态出现，原五控制各12、新三政策各两seed×12。每个旧提供者对两评分seed各评分一次，新提供者只用自身seed；共192上下文／216400评分状态出现，q0／Q／PPO／myopic／ppo_rate五规则1082000决定。评分不收GT／帧号／视频名／未来，使用冻结128维、归一化、clip±10、原零门槛及rate值；临近零logit以单样本原前向确认。既存规则选择必须复现；只是NumPy小控制器重评分，不称完全没有CPU推理。

当前公开最好IoU与q0／Q二元最好IoU分开，报告错过好框、q0 good→low／low→good、当前regret、有害接受／漏当前收益及真实改框分歧；先平均同视频seed，再宏平均视频。40行同状态、16行比较、8行候选摘要；全12视频保留，不从相邻帧／多seed当独立证据。四公开框无好框不等于dense maps或全图无目标，当前标签不等于动作后继效果。

**关注与分岔：** 固定8个±30帧窗：0304 PPO2172首失败554、PPO2171首失败1094；0307 PPO2171首失败395、监督2171首恢复470、PPO2172首恢复475；0176 PPO2172首恢复386；0204 q0首失败664；0226 rate2172首失败119。服务器按封存事件解析，缺事件不补点，默认128关注摘要，长帧表留服务器。全12×2seed另24条自身框／父框／候选／特征／所选裁剪分岔，顺序本身不当因果分摊。

**训练动态检查：** 读取96段NPZ和更新前后JSON／Adam链及原日志，不调用optimizer。每段双臂192行训练长表；按fold／seed／arm／pass为32行摘要，每格6视频等段均值；另8行优化摘要。重算前后概率／近半[.45,.55]／logit／放行／当前标签准确率／参数步长；固定原behavior GAE target MSE、完整原行为折扣MC误差及explained variance（target variance≤1e-12为空），PPO更新后对原采样动作的likelihood clip／近似KL。日志loss／entropy／clip／KL及三种梯度范数与本轮重算值分开；ratio of mean critic/actor norms不能还原每mini-batch裁剪或证明critic压制actor。更新后critic在旧PPO行为上的MC不叫新政策value验证，监督critic也是共享行为return；概率近半不证明未学习或应扩训。

**来源与产物：** 严格读取完整Stage-71及上游模型、状态、receipt、训练链和13旧表SHA，新manifest封存来源／四新源码；输入输出不得互为父子，旧源码／结果不修补。同内容可重复、不同内容拒绝覆盖。新根`output/stage72/cached_ppo_diagnostic_v1`；诊断COMPLETE不是方法采用、`method_adoption_gate=false`，新模型／图像／SGLA搜索／完整轨迹／反事实分支均0。

**服务器入口：** 同步新增代码后，在原仓库根／NumPy环境执行`bash tools/run_stage72_ppo_audit.sh`。默认完整输入`output/stage71/bounded_on_policy_search_ppo_v1`，不需要GPU／checkpoint参数或Torch／OpenCV。需断线后台：

```bash
mkdir -p output/stage72
nohup bash tools/run_stage72_ppo_audit.sh > output/stage72/run_stage72.log 2>&1 < /dev/null &
echo $!
```

脚本用cat合并diagnostic＋9表为`docs/experiments/06_sgla_uav_tracking/results/stage72_results_<UTC>_<PID>.txt`；默认数据行40／16／8／32／8／128／24／8／4，回传一份即可。15张CSV均留服务器，长表含逐seed提供者重评分／关注帧／192训练段臂记录／scope，保留完整旧缓存。

**本机验证分项通过：** 三项语义测试0.483秒通过，覆盖GT不进入评分、同框强制0／零logit平局、四公开支持与q0/Q二元支持分离、当前good→low／漏选与regret、终止折扣MC／GAE／零方差空值、先视频内均seed后宏平均、缺事件不补点、CSV可选列和不可覆盖。首次全套有两处新fixture错误（缺origin、同框却预期放行），仅修正测试样例后重跑三语义项通过，未放宽生产代码。

真实CPU工作流子测试通过（该次包含完整来源链构造的运行总378.526秒，无跳过）：真实CENTER来源链到Stage-71后，禁用Torch／OpenCV、图像读取、SGLA capture及optimizer update仍可完整诊断、重复结果一致；源文件逐字节与旧网络状态不变。训练轨迹／更新检查点／上游结果CSV／新报告篡改拒绝，输入输出嵌套拒绝。微型22评价缓存／88帧，128评分出现／640规则决定，8训练段／32帧，40同状态摘要／16比较／16训练pass摘要／8优化摘要／4分岔；不是服务器结果。与修正后三语义项分别完成验证，不重复不受fixture影响的来源构造。

三新Python AST、fresh进程禁止Torch／OpenCV导入、CLI与Bash `-n`通过；七份相关文档216处本地链接／协议表格列数、`git diff --check`通过；维护505保留／200移动／106活动模块、0缓存／0换行差异。旧tracking／lib／tools已跟踪源文件无修改，Stage-71原件SHA保持`19a144d62588c310566fb01e901beca121122f078f79581eb8de4279306ab519`；`.gitattributes`仅新增Stage-72 shell LF规则，未代commit／push。

### 0.37 2026-11-04：Stage-72缓存诊断完成，同状态净差小，优化尺度对照可继续

用户回传[Stage-72原件](docs/experiments/06_sgla_uav_tracking/results/stage72_results_20261104T124206Z_685785.txt)，已逐字节归档并加`-text`；110674字节／498行，SHA`958edfd8b4b66dfa8521731845906915dd02f178a061460a2f72d5f2acda56c4`。[完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE72_RESULTS_REVIEW.md)。状态`COMPLETE_CACHED_PPO_DECISION_TRAINING_DIAGNOSTIC`，method_adoption_gate=false；没有新AUC，Stage-71仍REVIEW，PPO61.8565／myopic64.2136／Q66.2438及两seed负差保持。**本次只归档与复核，Stage-73尚未固定设计／实现／运行，未代commit／push。**

**材料及预算：** diagnostic＋9CSV齐全，行数same_state40／matched16／candidate8／training32／optimization8／focus128／divergences24／anchors8／scope4；报告绑定15CSV，6长表same_state_video／matched_video／candidate_video／focus_frames／training_episode／scope_video留服务器，不默认索取。132评价来源、148775源帧出现；旧5提供者各用两评分seed，新3用自身seed，共192上下文／216400评分状态出现／5规则1082000决定；96段训练／108200帧、双臂192段记录。8锚点全支持。new_models／image_reads／search_calls／full_trajectories／counterfactual_branches全0，training／RL=false；NumPy controller rescoring=true，原选择精确复现／旧控制不变true。

scope评价fold0／1各66条，80927／67848帧、117712／98688评分出现，cached搜索323708／271392；fit各48段、49344／58856帧，cached197376／235424搜索。595100评价＋432800训练都是旧调用，Stage-71额外8 bridge没有重做。不把216400重复评分当独立帧或新轨迹，先视频内均seed，再宏12视频；来源同L40 UUID `ca7219a3-ba4a-8d0e-3299-2f85a384374c`。

**同状态结果：** 按q0／Hann／Q／future／mixed／PPO／myopic／rate提供者，PPO−myopic当前平均IoU为+0.000480／+0.000372／+0.000333／−0.000078／+0.000073／+0.000068／+0.000226／+0.000335。0～1 IoU不是AUC点；实际框与选择分歧约34.83%～37.16%，净差小不等于规则相同、局部无害或两者等效。PPO自身history有公开good78.4199%，myopic自身81.8878%，少3.4679个百分点；全公开low15.6386%对12.2049%，多3.4336个百分点。公开与二元good支持全局只差约0.28～0.47个百分点，局部仍有例外；四公开框不代表dense maps或全图无好输出。

PPO自身状态上，PPO有害接受当前较差Q10.9571%，监督6.9193%；错过当前更好Q11.6466%对16.6148%，监督更保守但也多漏收益。PPO实际当前IoU0.6276649813，myopic自身0.6519802563，PPO在myopic历史0.6522060003；自身差−0.024315275=指定同myopic输入规则差+0.000225744＋PPO跨输入差−0.024541019。是精确描述性代数记账，不是唯一因果分摊／百分比，不能据此说历史独立导致全部损伤或当前门控没有价值。缓存规则不滚动，未采取动作的未来不可见。

**优化动态：** 按fold0/2171、0/2172、1/2171、1/2172，PPO critic／actor平均梯度范数比230.53／215.09／180.31／189.32；actor均0.1530／0.1502／0.1695／0.1653，critic35.2747／32.3156／30.5569／31.2906，joint约30～35。原共同梯度范数clip=1，actor／critic无共享参数。PPO每段平均actor step L2为0.03884／0.03327／0.03889／0.04655，监督0.10182／0.07862／0.15610／0.15576，比0.3815／0.4231／0.2491／0.2989，均非零。平均entropy0.691920／0.692988／0.693021／0.692904近ln2；日志KL2.3481e−5／3.6972e−6／3.6612e−6／5.7847e−6，PPO全部pass组日志ratio clip及更新后behavior ratio clip均0。**PPO比率clip不等于梯度norm clip；平均范数比不能还原每mini裁剪、Adam有效步长或证明critic压制actor。** 监督也有约75～108倍梯度尺度差。

**最后一遍同曝光重评分：** PPO更新后eligible内均概率0.492170／0.498944／0.501422／0.507273，eligible概率[.45,.55]比例85.17%／99.93%／100%／100%；零阈值确定性接受42.72%／48.64%／57.13%／78.81%。概率近半仍能产生不同确定性率，不是随机50%、没学习或校准证明；这是各段更新后旧行为数据均值，不是最终checkpoint独立评价。监督eligible内均概率0.469409／0.481852／0.445145／0.459317，接受3.18%／15.12%／45.25%／42.06%。没有据结果调阈值、选种子或断言需要更多步数。

**value诊断：** 32个双臂pass组固定behavior GAE target MSE更新后均下降；PPO末遍124.091→120.263／110.828→107.951／112.542→107.662／113.359→108.387，critic并非没学。完整旧行为折扣MC EV末遍更新后0.03831／0.04243／0.05183／0.04759，前0.03625／0.03937／0.04813／0.04495，MC MSE约2673～3178。全MC均值45.14～53.32与固定GAE均值10.33～12.21定义不同，不据差距认定bug；监督critic同PPOreturn，也有弱解释。更新后在旧行为上的value不是更新政策真实return／held-out确认；PPO当前标签准确率低不等于其累计奖励目标“错误率”。

**关注窗与正负例（全61帧）：**

- C01 0304:554 PPO2172：PPO p=.500549接q1、监督p=.462457保q0，两者当前IoU均0；公开／二元good31、全low30，同窗PPO−监督−.010270。同期q0与监督2172历史均good61／全low0，后者当帧PPO .747038、监督 .758786。说明好坏随输入变化，不能拿另一历史好输出声称一次门控就修复坏历史。
- C02 0304:1094 PPO2171：两规则均q0、IoU.193514；Q .312715可改善但仍非good≥.5；good20、全low28，同窗差+.009456。
- C03 0307:395 PPO2171：两规则与Q均q0、IoU.155696；公开good1／二元0／全low26。原q0同窗也仅1公开good，但不能据此解释各自后来恢复差异。
- C04 0307:470 PPO2171历史公开good0，监督2171历史31；当帧两规则在前者均q0 .209126、后者均q0 .717258；当前换规则小，不能称匹配起点因果证明。
- C05 0307:475 PPO2172：两规则q0 .617438；公开good37／二元32／全low6，同窗PPO−监督+.005390。恢复正例保留，末尾再失跟在窗外。
- C06 0176:386 PPO2172：两规则q0 .646182；公开45／二元43／全low0，同窗差+.016875。不能从首次恢复推断后续不复发。
- C07 0204:664原q0历史、评分2171：PPO p=.504185接q3 .753846、监督p=.450007保q0 0；公开32／二元31／全low29，同窗差+.016197，说明有需要接管的局部正例。同q0输入评分2172时两规则都保q0 0，局部正例并非双seed一致。Stage-71自身PPO／监督0204双seed均无事件，不能把shadow点当成监督实际失效。
- C08 0226:119 rate2172历史：公开／二元good0、全low26；PPO／监督均q0 .163921，rate原框.169110；换当前门控不创造good，不能唯一解释率控制长期失跟。

关注特征裁剪坐标比例0～2.31%；不能称严重整体饱和已确立，也不能据坐标均值排除关键维度／分布问题。24条全视频分岔均第k帧框／所选crop不同、第k+1帧候选／父框／特征不同，共同候选阶段只有1次选框分歧，k为2～62；0304两seed k2、0307 k14／3、0204 k19／2。顺序符合闭环，不证明首次分歧单独导致后来损伤。

**本机核验：** diagnostic自身fp、9CSV原字节SHA／计数／唯一键、15表清单；完整来源report／plan／training／runtime；scope对旧帧数／fold／调用；同状态支持与regret恒等式／配对差、own-rule low及非q0率；32组×6重构原帧数／eligible／接受／更新、双臂共享数据目标、8优化摘要和最大actor梯度；8锚点时间与原focus、128窗身份／计数／差值、24分岔顺序均通过。服务器manifest／逐帧NPZ／参数Adam／6长表未在本机独立重放，不将服务器审计写成本机已逐更新证明，也不唯一归因。Scratch `D:/Users/23920/Documents/Codex/review_stage72.py`及`stage72_bundle_parsed.json`；原件及Stage-71来源字节保护核验。

diagnostic fp `e221aa1f4f456b4d7d11760f34c7ba57ae768242ebad236c6ca154f2a469d121`；plan `2188da23a3fc31167a7d3082d6a746d63fc1c8625d2381192500f6efbdbfef30`；source bindings `3c7e8908f3e9fe13e46f5b277bf41735b8f944852b6a27813909c91b344736f3`；来源Stage-71 report／plan／training见0.35。Stage-1～72实验／修复源码与模型／旧结果冻结，没有重跑旧CPU／GPU实验。

本轮维护及文档验证PASS：505保留／200移动／106活动模块、0缓存／0换行差异；8文档221处本地链接、Stage-72协议／复核49行表格列数、原件逐字节保护及`git diff --check`通过；已跟踪tracking／lib／tools源码修改为空。没有重跑未改的实验测试。

**本轮复核时的下一步建议（现已落实，见0.38）：** 缓存核查完成，准备固定预算的actor／critic优化尺度对照，单独检验分开梯度norm处理；保持状态、二元动作、reward／GAE、seed、曝光／更新次数、最终checkpoint规则、全12视频评价和匹配监督。不能把critic因果当已证实，也不同时加warm-start／记忆／新奖励／扩训。训练KL／value好转仍需完整跟踪和0304稳定／0307恢复保护同时检验。不要无限追加前置诊断，也不直接采用／扩训旧PPO。

### 0.38 2026-11-04：Stage-73固定裁剪对照及实现验证历史（结果见0.39）

用户要求下一步，实现[Stage-73固定协议](docs/experiments/06_sgla_uav_tracking/STAGE73_SEPARATE_GRADIENT_CLIP_PPO.md)及六新文件：`tools/stage73_ppo_common.py`／`stage73_ppo_runtime.py`／`stage73_ppo_experiment.py`／`stage73_ppo_summary.py`／`test_stage73_ppo.py`／`run_stage73_ppo.sh`。输入完整Stage-72，新根`output/stage73/separate_gradient_clip_ppo_v1`。以下保留结果前方案和实现验证，当时服务器待执行；现已完成，结果见0.39。

**固定处理：** 原actor＋critic共同norm clip1改为各分支分别norm clip1，同一次loss反传、同一个原Adam和LR，不拆分优化器。分别处理后合范数最多约√2，而非仍≤1，这是处理的一部分。直接复用Stage-71网络／CPU局部初始化／Adam／PPO或BCE loss代码；每个新job的初始参数与norm逐值比对原step0，视频／pass顺序直接复用原jobs。2171／2172、四遍fit、四epochs／mini256／最终step、gamma.99／lambda.95、reward IoU、value.5／entropy.01、128维clip±10和零logit门槛全部保持；不加warm-start／记忆／新reward／扩训。

**对照与曝光：** `joint_ppo`／`joint_myopic`／`joint_rate`按字节复用Stage-71三政策最终包和率阈值（率无新网络）；新`ppo`／`myopic`／`ppo_rate`用分别裁剪训练／校准。每个条件内部myopic只看其PPO实际fit状态、标签／return和mini-batch；跨裁剪条件PPO各自实时收集，更新后状态可以不同，不能宣称逐帧曝光相同。主比较是整个训练处理的闭环效果，交互也含相应曝光改变，不能从大梯度直接认定根因。冻结SGLA V1／Q／首帧模板／四候选，不变检测或V2。

**同卡来源与正式预算：** 五旧q0／Hann／Q／future／mixed各12条，三旧joint与三新政策各2seed×12条，共204；旧132轨迹在本轮L40逐字段精确复现Stage-71，decode沿旧容差。只允许物理L40 UUID变动。4 PPO＋4匹配监督新模型包，原8包复用；96新完整fit段／108200环境步／432800训练搜索，每臂1888更新；204评价／919700搜索，bridge另8，合1352508搜索／9467556块。训练smoke第一完整段2392、评价首视频32帧×17实例2176均正式复用，额外8另计；异常未提交尝试耗时不当0。自然终止、无GT失效重置。

**比较与保护：** 主PPO−joint_ppo；四关键家族对joint_ppo／新myopic／Q／future，视频内均seed后12视频bootstrap2171／20000、分位.05/8。`optimization_contrast_supported`需主家族下界>0、均值≥+.5、两seed均正且guard；不是采用等价。PROMISING另需四关键家族全正、对myopic／Q均≥+.5、每seed对joint／myopic／Q均正、至少4视频接受且4否决、guard通过；否则REVIEW。guard为q0稳定任一seed0新增失败、low均值不增、reliable差CI下界≥−1。额外10对照差及myopic−joint_myopic、2行factor、seed与全视频影响均保留；primary_leave_one_out按joint差计算。消费开发证据，不称独立外部确认。

**训练动态与可续跑：** 新日志保存裁剪前／后分支范数、系数、loss／entropy／KL／ratio clip；同时对原joint及new separate各自更新前后行为状态重评分，384训练段臂记录、64 pass摘要、16优化摘要。原joint没有post-clip分支日志就留空，不从均值伪造。value是各自旧behavior的固定GAE／MC描述，不当更新政策真实return。JSON参数／Adam前链＋NPZ receipt支持已提交轨迹后中断补CPU更新，已提交更新不重做；旧来源／模型／缓存／新表篡改拒绝。初始与加载模型都重新核对原joint初始包。

**服务器入口：** 仓库同步新增文件后，在原repo根／`sglatrack-test`运行`bash tools/run_stage73_ppo.sh`。默认完整输入`output/stage72/cached_ppo_diagnostic_v1`、原V1 checkpoint、GPU0；只有上传TXT不够。断线后台：

```bash
mkdir -p output/stage73
nohup bash tools/run_stage73_ppo.sh > output/stage73/run_stage73.log 2>&1 < /dev/null &
echo $!
```

流程prepare→source_bridge→train_smoke→train→smoke→evaluate→summary；用`tail -f output/stage73/run_stage73.log`查看。脚本用cat输出decision＋12摘要表为`docs/experiments/06_sgla_uav_tracking/results/stage73_results_<UTC>_<PID>.txt`；预期policy11／paired44／seed2／factor2／calibration4／training_scope4／fit_support24／optimization16／training_summary64／focus204／scope204／influence12。17CSV全保存，5长表training_episode384／training_log96／seed_video／video／events及模型／Adam／NPZ留服务器，回传合并TXT一份即可。

**本机验证最终通过：** `python -B -m unittest discover -s tools -p test_stage73_ppo.py -v`三项、432.698秒、无跳过。生产规模预算／单处理约束／fit隔离；真实CPU float64分支clip机制、同原loss／Adam、CPU局部RNG不重设CUDA、NumPy前向／Adam JSON续跑；真实CENTER完整来源链到Stage-72后，新阶段prepare／bridge／实时PPO共享监督更新／34评价／重复summary／22旧控制精确复现通过。提交轨迹后中断可不重复该搜索而补CPU更新，全提交后零新增推理／更新续跑；未来GT只变奖励／标签，不变动作／历史，对应held-out GT不可读仍同fit更新。全部旧文件与SGLA权重不变，源Stage-72表／Stage-71 step、新step／NPZ／新结果表篡改及嵌套输出拒绝；summary禁用Torch／OpenCV仍通过。joint未记录post-clip为空，主留一从视频表独立重算。微型为2视频×4帧、2遍、hidden4／epochs2／mini2，32环境步／128训练搜索、34评价／544搜索、bridge8，总680，不是正式服务器成绩。

首次流程在新Stage-73入口继承上游汇总fixture的`sys.modules cv2=None`，仅修正新测试恢复真实Torch／OpenCV后完整通过；生产条件未放宽。新汇总代码审查修正primary_leave_one_out为joint_ppo并补算术验证。独立额外数值核对在仅测试的超大clip上限使裁剪不触发时，新旧双臂参数及序列化Adam逐值相同，支持唯一更新差异为裁剪；正式上限仍1。Bash `-n`通过（沙箱MSYS对象受限后，只读语法检查以沙箱外执行）；5新Python AST、全新进程禁止Torch／OpenCV的CLI帮助通过。9文档225本地链接／协议20表格行、shell LF、Stage-72原件字节保护通过；维护505保留／200移动／106活动模块、0缓存／0换行差异及`git diff --check`通过，旧tracking／lib／tools已跟踪源码未修改。没有Stage-73服务器成绩，未代commit／push。

### 0.39 2026-11-05：Stage-73分别裁剪结果完成，更新变大但PPO未稳定改善

用户回传[Stage-73原件](docs/experiments/06_sgla_uav_tracking/results/stage73_results_20261104T201738Z_703850.txt)，148598字节／955行，SHA`f4b076427b7af853923db08a280689da6631431f8e3bb1e66dff67db306f277b`，逐字节归档、`-text`保护；[完整复核](docs/experiments/06_sgla_uav_tracking/STAGE73_RESULTS_REVIEW.md)。UTC时间戳对应香港2026-11-05 04:17:38。REVIEW，support=true，optimization_contrast_supported／guard／seed／pilot=false。decision＋12表齐全，不默认索取长表、不重跑旧实验。本次只复核及交接，**Stage-74尚未固定设计／实现／运行**，未代commit／push。

**来源与执行：** 只改共同norm clip1→actor／critic各norm clip1，合范数上界约√2是处理一部分，不称总上限同为1。原SGLA V1／Q／首帧模板／四搜索／128维状态／reward／GAE／Adam／2171-2172／四遍fit／最后step均保持，新初始包与norm比原step0。每条件内myopic同PPO曝光，跨条件on-policy状态可不同；完整训练处理效果不是唯一梯度因果分解。新8包、原8包复用；96新fit段／108200步／432800搜索，每臂1888更新；204评价／919700搜索，原132重测＋新72，另bridge8，总1352508搜索／9467556块。train smoke2392／eval2176均复用，额外8已入总数。

runtime与Stage-71／72完全相同，差异列表空，物理L40 UUID `ca7219a3-ba4a-8d0e-3299-2f85a384374c`。原132 focus、宏共同字段和scope共同字段全部对应；服务器逐字段复现／旧模型与Q不变true、error0，最大decode0.00048828124954525265沿原容差。没有L20混用或新SGLA训练。

**宏结果：** 新PPO／myopic／rate AUC62.3800／64.7694／62.5611，low15.82%／12.37%／16.77%，reliable70.6662／73.7202／69.8210，原稳定新增视频1／1／0。旧joint PPO61.8565、joint监督64.2136、joint rate61.1751；q063.4512／Hann65.2483／Q66.2438／future66.9254／mixed65.2286均原样。主要PPO−joint_PPO +0.5235、名义CI[−0.1869,+1.6960]、四项家族[−0.2699,+2.1949]，9胜3负、中位+.09404、最小留一−.01756；不能仅因均值过+.5及多数胜就判通过。

对新监督−2.3894，名义[−7.2974,+1.1853]、家族[−9.1218,+1.6594]，9胜3负；对Q−3.8638、家族[−14.8405,+3.4675]、4胜1平7负；对future−4.5454、家族[−15.6395,+3.3894]、8胜4负。对新rate−.1811、8胜4负、名义[−8.9136,+6.8406]；对旧joint监督−1.8336、8胜4负、名义[−4.1924,−.0375]，不能与四关键家族结论混写。主比较low少.4034个百分点、reliable+.4753均区间跨零；longest少.5483个百分点、名义CI[−1.1758,−.0483]个百分点，是积极描述但不代替主比较和保护。

**种子与集中性：** 2171旧joint PPO60.2778→新63.9099，+3.6321，新监督66.2830，PPO−监督−2.3731；2172旧63.4352→新60.8501，−2.5851，新监督63.2558，PPO−监督−2.4057。主比较方向相反，两seed对匹配监督仍均负，不挑2171。0307主差+6.4747、按12视频贡献+.5396，其余11条合计贡献约−.0161，移除0307后−.01756。PPO对新监督虽9胜3负，0304／0307为−25.1953／−11.4813，另0126−.02650，其余小胜不足抵消。

新myopic对旧joint_myopic+.5558、5胜7负、CI[−1.7189,+3.5239]；0304+14.2636、0176−6.6175、0147−3.4588，移除0304后−.69035。clip平均+.5396、CI[−.7845,+2.1483]；目标×clip交互−.03235、CI[−3.1509,+2.2783]，未建立PPO特有收益，且交互包含曝光改变。

**关键轨迹：**

- 0304：旧PPO2171／2172 AUC44.1297／22.0649，新43.7543／22.0290，分别1090／554后均无恢复；宏对q0−39.3954，仍是原稳定新增失败视频。新监督2171为72.2208无事件，2172为43.9531、1093后未恢复；新rate两seed71.8233／71.5059均无事件。相同冻结SGLA下存在保稳轨迹，但不同自身历史不是在PPO坏输入上换当前门控即可修复的因果证明。
- 0307：新PPO2171为41.5898、397后无恢复；2172为66.1648、394→472首次恢复、末尾未恢复0。对旧PPOseed差+.3325／+12.6169，收益集中2172。新监督64.9413／65.7758，389→474／392→472，末尾均0；新rate39.9649／39.7578，均无恢复。匹配fit率不自动保住恢复能力。
- 0176：新PPO61.1920／16.2089，对旧+45.2657／−44.4363；2171在381→388恢复、末尾0，2172在167后无恢复、末尾1，成功seed互换，宏仅+.4147。新监督37.6406／28.0553，首恢复747／空；不能当成稳定恢复增强。
- 0204：新PPO72.7367／72.5402，新监督72.6281／72.5402，新rate72.8504／72.7935，两seed各组均无事件，保留既有救援，非PPO独有。
- 0226：新PPO54.3216／55.7527，首次恢复343／176、末尾均0；focus不含全部事件，不把176当整段持续恢复提前。新rate2171 AUC7.2683，119后无恢复；灾难seed由旧rate2172换到2171。
- 0147：新PPO33.0909／37.3252，新监督34.2557／34.0476，两臂两seed均终末未恢复。

旧PPO每seed终末未恢复5视频，新PPO各4（2171去掉0176，2172去掉0307）；新监督2／4。PPO对q0仍AUC−1.0712、low+1.3627个百分点、reliable−4.8941（名义CI[−12.6695,−.000216]），0304新稳定失败仍在，保护三项均未满足。

**优化动态：** 按fold0/2171、0/2172、1/2171、1/2172，PPO平均actor参数step原.03884／.03327／.03889／.04655→新.10202／.11234／.10663／.11061，倍率2.627／3.377／2.742／2.376；新myopic倍率约2.193／2.454／1.520／1.595。新PPO actor裁剪系数均1／1／.99886／.99511，critic .07621／.07050／.06181／.05303；post actor约.150～.166、critic .993～1.000。joint旧post日志空，不从范数均值推造。处理确实增大更新，不能再说改动未到actor；参数步长不代表收敛或策略质量。新两臂critic日志／step／value指标逐项一致，符合独立clip下共享return／初始条件的代码预期，不是强value学习证明。

新PPO平均日志KL1.525e−5／1.546e−5／7.875e−6／1.643e−5，对旧三job增加一job减少；所有pass日志及重算behavior ratio clip均0。最后一遍eligible均概率.50247／.52768／.50556／.50826，近半[.45,.55]比例98.16%／55.08%／99.96%／94.26%，确定性接受48.55%／73.13%／70.40%／62.56%；多数仍近半但不等于随机50%或没学。64新旧双臂pass的固定GAE MSE均下降；新末遍119.671→116.789／132.424→128.093／121.885→116.488／112.683→107.820。新behavior MC EV .04261／.03075／.04585／.05686，旧.03831／.04243／.05183／.04759，未一致改善；跨条件target不同，旧behavior上误差不是新政策真实value或泛化。

实际评价改框PPO40.37%→46.48%，监督22.51%→12.46%，新rate45.15%。新fit率37.55%／44.74%／51.79%／35.93%，率误差.000973／.002467／.004061／.002926个百分点，在同分界内；评价率／候选支持不一定相同。更多干预未稳定改善跟踪。

**本机复核与边界：** decision自身fp、12CSV原字节SHA／行数／唯一键、17表清单、Stage-72／71来源及runtime；132旧focus／宏／scope共同字段；204 focus重算11宏AUC／seed内均值／introduced-any／终末计数；44配对宏均值、11 AUC名义区间／胜平负／中位／最差／留一、4家族、2因子、两seed及12影响；24 fit／4scope／204评价折隔离、4×7调用／1888更新／总预算、4校准误差；64pass共享目标、joint与Stage-72一致、16优化聚合及新clip范围均相符。服务器manifest／系数Adam／逐帧NPZ／receipt、5长表和bridge正文未独立重放；初始化／逐字段／GT隔离是服务器固定代码审计，本机不据摘要重新证明所有更新或失败唯一因果。材料齐，不索取本轮未要求的长表，不重跑未改的测试。

原件`-text`；decision `ea0fb015a2386818eaca22fe4f1e4a0efce5709c6818b5ae157281d9d476dcde`；plan `f8f412b4a93fb2d2c7a609c0fe0a9c89e092ab189d51542654412fe3ad893918`；training `0b7c081dfc12f0f7609accec614b0b2d98f4b0c139f658d09c291b530eaefbd0`；run `9379a9f7862cc832ddf3b2759d44c3d87f1b107c0a1e5e0c6f2f6d54310b11ff`；source bindings `4e48c26e9e368cbeef45ff45fdc86bff2791224d4981e2014dab2be03ee5ab26`。Scratch `D:/Users/23920/Documents/Codex/review_stage73.py`／`stage73_bundle_parsed.json`／`project_stage73.py`。

本轮维护及文档验证PASS：505保留／200移动／106活动模块、0缓存／0换行差异；10文档233处本地链接、Stage-73协议／复核63表格行、原件逐字节／旧Stage-71-72 SHA保护和`git diff --check`通过；已跟踪tracking／lib／tools源码修改为空。未重跑未改的实验测试。

**本次结果复核时的下一步边界（用户后续已指定0.40四选一实验）：** 不采用或直接扩训当前PPO，不继续只调clip／步数／阈值，也不将本轮扩大为所有RL无效。分别裁剪已生效但未稳定缩小PPO对匹配监督的差距；下一轮若继续，应固定一个新的观测或信用分配假设，保留同曝光监督及q0／Q和完整滚动控制。不能从现表断言记忆或reward为唯一病因，也不无限追加旧缓存诊断。Stage-1～73源码、模型和旧结果冻结。

### 0.40 2026-11-05：Stage-74结果前固定方案与实现验证历史（结果见0.41）

用户明确要求“先不做恢复支线，验证四个候选→PPO直接选择q0、q1、q2或q3”。已实现[Stage-74固定协议](docs/experiments/06_sgla_uav_tracking/STAGE74_DIRECT_FOUR_CANDIDATE_PPO.md)及六新文件：`tools/stage74_ppo_common.py`／`stage74_ppo_runtime.py`／`stage74_ppo_experiment.py`／`stage74_ppo_summary.py`／`test_stage74_ppo.py`／`run_stage74_ppo.sh`。输入完整Stage-73根和其上游NPZ／模型，新根`output/stage74/direct_four_candidate_ppo_v1`。以下保留结果前设计及验证；当时无服务器新精度，**现已完成，结果见0.41**。Stage-73仍REVIEW，未代commit／push。

**具体假设：** 旧二元PPO只可保留q0或接受Q单个推荐，可能无法直接选另两个候选。新PPO输入四候选现有各64维因果描述、展平256维，直接输出q0～q3 categorical概率；Q仅用于记录／对照，不限制新actor。保持原SGLA V1／CENTER、权重／模板／四搜索生成、历史父框定义、奖励和训练次数；没有恢复支线、额外搜索、未来图像、GT动作触发或失败重置。四个公开候选均差的问题本轮不由新搜索解决。

**表示／动作的准确边界：** actor256→32 Tanh→4，critic256→32 Tanh→1独立，actor8356＋critic8257＝每臂16613参数；控制器CPU float64。每折从fit组已有Hann缓存按视频等权拟合256维标准化，clip±10，不读held-out GT。只有完全相同的公开输出框合并到最小索引，q0始终合法；不同的低质量框不能因GT或评分被屏蔽。全相同强制q0且该状态不算actor损失。初始对不同框均匀采样，四框不同则q0为1/4；确定性评价argmax／平局最小索引。对旧二元的比较同时改变输入信息、维数、容量和探索分布，**不是纯动作数量消融**。

**匹配训练控制：** 新`ppo`和`myopic`同架构／初始化／标准化／数据／minibatch／更新预算；由新PPO实际滚动采样两臂共同fit曝光，监督不另采自身轨迹。myopic用当前四个合法框最高IoU的交叉熵，1e−6并列最小索引；两臂共用旧行为GAE return作为value辅助目标，value0.5／entropy0.01相同。奖励为实际所选框IoU，gamma.99／lambda.95／eligible标准化adv／clip.2／Adam3e−4 eps1e−5，actor／critic分别norm clip1。沿用两折6fit／6held-out、2171／2172、四遍／四epoch／minibatch256，固定最后一步，无warm-start、奖励／阈值／checkpoint搜索。训练标签只在真实动作／完整轨迹生成后附加；监督对照是同PPO曝光下的目标对照，不代表所有监督方案最优。

**评价八政策：** q0／hann／quality／future各12条；旧Stage-73分别裁剪`binary_ppo`／`binary_myopic`各24条原包复用、新四选一`ppo`／`myopic`各24条。共144完整轨迹＝96条选定旧控制重测＋48条新政策，不是重跑Stage-73全部204条。要求选定旧控制共同物理搜索／响应／输出字段精确复现；新categorical占位格式不与旧二元logit混比。新阶段先source_bridge，仍绑定L40，只允许物理UUID变化，其他身份及来源哈希严格不变。

**预算：** 8新控制器包＋8复用旧二元包；96实时fit完整视频／108200环境步／432800 fit搜索，每新增臂1888更新；144评价／649200搜索，额外bridge8。总1082008搜索／7574056块。train_smoke第一完整段2392搜索复用；eval_smoke首视频32帧×12实例×4＝1536搜索复用。故障／未提交重试另记，不能称零成本；原SGLA可训练参数0，新增恢复搜索0。

**判读：** 主比较新四选一PPO−同容量myopic，同时对Q／旧二元PPO／future四个AUC关键比较做家族区间，另保留旧二元监督／q0／Hann。视频内先平均seed、再12视频bootstrap20000次／seed2171；开发组反复消耗，不能称独立确认。PROMISING需四关键家族下界均>0、对myopic和Q均值≥+.5；两seed各自对myopic／Q／binary_ppo正；q0稳定任一种子0新增失效、低IoU不恶化、reliable CI下界≥−1；至少四视频实际改q0输出且至少四视频使用q0。否则REVIEW，不只凭超过旧二元PPO就宣称RL收益。全12视频及0304长失跟／0307恢复／0176和0204收益都要看。

**诊断及回传：** decision＋11CSV合为一个`stage74_results_*.txt`：policy_summary、paired、seed_effects、training_scope、fit_support、training_summary、optimization_summary、candidate_summary、focus、scope、video_influence。服务器另保留candidate_video、seed_video_summary、video_summary、events、training_episode、training_log，合计17CSV。候选使用／超出{q0,Q}的实际框比例／当前支持与漏选都是离线描述，不是未执行动作的长期收益；训练前后概率和value指标是旧行为数据读数，不是新政策回报。默认回传一个TXT即可，不重复索取已有长表。

服务器同步新增文件后，在`sglatrack-test`环境及仓库根目录执行；脚本依次prepare→source_bridge→train_smoke→train→smoke→evaluate→summary，最后用`cat`合并，不自动上传Git：

```bash
mkdir -p output/stage74
nohup bash tools/run_stage74_ppo.sh > output/stage74/run_stage74.log 2>&1 < /dev/null &
echo $!
```

用`tail -f output/stage74/run_stage74.log`查看；默认`STAGE73_ROOT=output/stage73/separate_gradient_clip_ppo_v1`，新根如上，可指定CHECKPOINT／STAGE74_GPU（默认0）及真实根目录。相同源码／身份下重启会复用已提交步骤和smoke；不可删cache、改mask／代码后继续封存根，也不要同根并发启动两份。

**本机验证：** 4项CPU测试全部通过，407.744秒、真实Torch／OpenCV、无跳过：四动作mask／平局／Q以外直接动作；生产预算及折隔离；真实PPO clipped／四分类CE／强制q0数值与Adam连续性；真实CENTER微型Stage-63→73→74源链全流程。新阶段实际32fit步／128训练搜索、24完整评价／384搜索、bridge8，共520搜索。覆盖训练中断续跑、完整评价及旧共同字段精确复现、fit／评价GT改变不影响动作、held-out GT隔离、来源／step／缓存／结果篡改拒绝、候选／主比较留一算术；完全提交重启零推理／更新、重复汇总一致且不依赖Torch／OpenCV。5新Python AST、全新禁止Torch／OpenCV进程CLI帮助、Git Bash只读`-n`和维护505／200／106、0缓存／0换行差异通过。11文档237处本地链接、本轮协议29表格行、shell LF、Stage-73原件逐字节／Stage-71～72旧SHA保护及`git diff --check`通过。Scratch验证`D:/Users/23920/Documents/Codex/check_stage74_docs.py`。旧已跟踪tracking／lib／tools源码未改。本机没有服务器完整VisDrone／L40，CPU测试不代表服务器精度。后续接续先检查是否已有Stage-74回传，尚无则运行本轮，不默认另建Stage-75／恢复支线，不重跑Stage-1～73。

### 0.41 2026-11-05：Stage-74完成，四选一PPO有双seed监督对照收益，稳定保护仍未过

用户回传[Stage-74原件](docs/experiments/06_sgla_uav_tracking/results/stage74_results_20261105T080357Z_823287.txt)，96126字节／752行，SHA`99ce26e52c6289c09f81ddbe98fa8eab546b3ffc26afa5a93dda941c8209f398`，逐字节归档、`-text`保护；[完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE74_RESULTS_REVIEW.md)。UTC文件名对应香港2026-11-05 16:03:57。状态`REVIEW_DIRECT_FOUR_CANDIDATE_PPO_PILOT`，support=true，guard／seed／pilot=false。**有值得继续推进的进展，不等于毫无收益；尚不能稳定替换Q。** 本次仅归档／复核／交接，未代commit／push，Stage-75未固定设计／实现／运行。

**材料与正式执行：** decision＋11CSV齐全，policy8／paired28／seed2／training_scope4／fit_support24／training_summary32／optimization8／candidate8／focus144／scope144／influence12。17表清单中的candidate_video／seed_video／video／events／training_episode／training_log六长表留服务器，不默认索取。8新模型包＋8旧二元包、96实时fit段／108200环境步／432800搜索，每新增臂1888更新；144评价＝96旧政策重测＋48新政策、649200搜索，bridge另8，合1082008搜索／7574056块。smoke2392／1536均复用。SGLA权重／模板／四搜索冻结，256维四候选actor直接选q0～q3，无恢复支线／新增恢复调用。

**成绩：** q0／Hann／Q／future／旧二元PPO／旧二元监督／新四选一PPO／新四选一监督AUC63.4512／65.2483／66.2438／66.9254／62.3800／64.7694／67.2356／63.1634；低IoU14.46／13.06／11.01／9.66／15.82／12.37／8.04／14.48%。新PPO原生可靠帧73.1461，新监督68.6464，原q0 75.5602。

主比较PPO−同容量四选一监督+4.0722，名义CI[+0.3955,+8.2025]、四关键家族[−0.01084,+9.4016]、8胜4负、中位+0.2340、最小留一+2.7304；不能将微负家族下界四舍五入成通过。low差−6.4370个百分点、名义CI[−13.6031,−0.3072]，reliable+4.4998、CI[+.2987,+9.2400]。对Q+0.9918、CI[−2.9810,+5.4411]、家族[−3.9808,+6.8324]、5胜7负；对旧二元PPO+4.8556、CI[−.0390,+11.2451]、家族[−.4812,+13.2719]、7胜5负；对future+.3102、CI[−3.4324,+4.8304]、家族[−4.4056,+6.3603]、6胜6负。对旧二元监督+2.4662名义区间跨零、6胜6负，新监督自身比旧监督低1.6060，不能把主比较写成RL普遍优于所有监督。

**两seed与集中性：** 新PPO2171／2172 AUC68.2439／66.2273，新监督61.2565／65.0704，P−S+6.9875／+1.1569；对Q+2.0001／−.01646，对旧二元PPO+4.3340／+5.3772，对future+1.3185／−.6981。两seed主比较均正，是相对前两轮二元对照的积极变化；2172微低Q使seed条件未过。主比较0226／0304／0307贡献+18.8313／+13.8068／+14.4143，按12视频记账共+3.9210、占净均值96.29%，其余9视频平均+.2015；没有单视频支撑全部收益，却也不是普遍大幅改善。去掉0307，对Q−.7135、对future−1.4844。

**关键完整轨迹：**

| 视频 | 四选一PPO2171／2172 AUC及事件 | 关键对照与解释 |
|---|---|---|
| 0176 | 66.7158／66.6359，381→388／314→321恢复，末尾均无未恢复 | 旧二元PPO61.1920／16.2089；新监督67.8525／67.9570且均无事件，四选一收益并非PPO独有 |
| 0226 | 54.3779／54.8413，343／337首次恢复，末尾均无未恢复 | 新监督2171仅17.3016、125后不恢复，是主比较集中来源之一 |
| 0304 | 72.3948／43.8068，2171无事件，2172在1090后未恢复至1725 | 由旧二元两seed持续失败变为一个保稳，宏58.1008仍低q0 14.1863；监督2172无事件但44.3533也明显受损，不能按事件0称保稳 |
| 0307 | 57.4001／63.4983，403→474／397→471首次恢复，末尾均无未恢复 | 对Q／future宏约+20，但对q0−6.0606、旧二元监督−4.9093；旧PPO2172本已66.1648，新2172反而−2.6664 |
| 0204 | 72.4833／72.6333，两seed无事件 | 保留已有救援，新监督亦无事件 |
| 0003 | 75.7606／76.6534，138／140首次失败，末尾未确认恢复 | 共144帧，仅剩7／5帧事件窗口，属于新增末尾短失败；旧控制均无事件，新监督也双seed138。不能据末尾截断说已观察到长期失跟 |

0147／0342新PPO两seed仍末尾未恢复。PPO末尾未恢复视频2171／2172为3／4、旧二元4／4、新监督6／3。新增稳定失败0003／0304共2视频；对q0 reliable−2.4141、名义CI[−5.1879,−.4474]、2胜10负；整体low虽少6.4169个百分点，保护仍因新增失败及可靠帧容差未满足。focus没有全部事件／逐帧IoU，不从首次恢复反推全部复发，也不把事件窗口每帧都当low。

**直接动作实际使用：** PPO q0／q1／q2／q3宏21.90／36.56／32.29／9.26%，改q0输出78.10%，实际输出在{q0,Q推荐框}之外55.42%；监督71.85／.41／4.94／22.80%、超范围19.75%。PPO自身公开良好支持85.93%、二元支持85.37%，当帧新增支持仅.5561个百分点；55.42%动作超范围不等于55.42%独有救援机会。PPO／监督选良好84.79／79.35%、有好漏选1.14／2.89%、current regret .01796／.03225、好q0替成low .0832／2.1427%。两者历史不同，当前描述不等于同状态因果；PPO对旧二元current regret反而更大（旧.01420）。

**训练动态：** 四job fit随机四动作均覆盖、eligible99.90%～99.92%；q0标签50.11／50.74／56.48／58.77%，包含并列最小索引约定。PPO actor段步长L2 .16117／.15657／.16295／.15923，actor clip系数1／1／.98898／.99287、critic .02565／.04266／.03605／.03938；日志KL8.014e−5／5.989e−5／1.086e−4／1.033e−4、更新后behavior KL .000293／.000219／.000391／.000387，ratio clip仍0。末遍PPO最大概率均.3446／.3249／.3325／.3275、当帧标签准确29.52／26.10／31.60／35.94%，监督48.23／47.68／55.94／50.69%。监督标签精度更高并不替代完整跟踪，也未唯一证明累计奖励机制。32 pass两臂固定GAE MSE均下降、critic共享目标／初始化读数相同；末遍旧behavior MC EV .02568／.03325／.04205／.02962，仍不能当新政策value验证。

**核验与边界：** decision fp、11CSV原字节SHA／行数／唯一键、17清单、Stage-73来源及完整runtime、96旧focus／宏共同字段／scope、144 focus→8宏／两seed／introduced-any／终末数、28配对宏差及7 AUC bootstrap／4家族／12影响、fit／评价折隔离、四动作和标签计数、1888更新／4×7／总预算、32pass共享critic／clip、8优化聚合与候选恒等式相符。最大decode .00048828124954525265。服务器manifest／参数Adam／NPZ／receipt／bridge及六长表未本机独立重放；可靠／low逐视频区间及逐动作也未从摘要重算，GT隔离和逐字段精确复现按固定代码服务器报告。材料齐，不重复请求或重跑旧训练。Scratch `D:/Users/23920/Documents/Codex/review_stage74.py`及`stage74_bundle_parsed.json`。

decision `c87fc620a45b063b9a07b6658ef15b9490d31211ca75e37637e2106d394cf9ac`；plan `0a914c0c7c41f80c1d5964f2420119fad325b5ed6aa97cbb3d2adf66ae97c8fc`；training `3cabcdf33e0a3cd382cf35ba59be7825a02909ea10a3d52603b5e6921554398b`；run `6167c17530c2ade5eedc1064365d5eb4ac617fbd9fcff4eeb6f6718e3a65cc09`。维护505／200／106、0缓存／0换行差异通过；12文档246处本地链接、协议／复核64表格行、原件逐字节／Stage-71～73 SHA保护和`git diff --check`通过；已跟踪tracking／lib／tools源码修改为空，未重跑未改实验测试。文档验证`D:/Users/23920/Documents/Codex/check_stage74_results_docs.py`。

**下一步边界：** 保留四选一PPO这条正向路线，优先保住0176／0307／0204收益、查清0003末尾与0304／2172长期失跟的具体分岔；区分候选不足、好q0被替换、漏选和先前历史变化。可先用新阶段已有缓存，若需因果后继再预定有限共同起点续跑。不仅加训练／改阈值／挑种子，不把当前监督控制视为所有SL最优；保持Q与同容量四选一监督。继续用户“不做恢复支线”范围，不自动增加搜索、换V2／检测，不把12消费开发组称外部确认。Stage-75尚未设计／实现／运行，Stage-1～74源码、模型和结果冻结。

### 0.42 2026-11-07 Stage-75 固定失跟诊断与短续跑

Stage-75 用 Stage-74 的完整封存运行和上游缓存定位失败机制。本轮固定 0003 两种子首败 138／140，0304／2172 首败 1090，并分别取提前 8 帧的起点。稳定对照为 0304／2171 的 1090。收益对照为 0176 的 388／321、0307 的 474／471、0204 两种子的 664。13 个参考帧及窗口长度已与回传 focus、scope 表核对。 [固定方案与起点表](docs/experiments/06_sgla_uav_tracking/STAGE75_FOUR_CHOICE_FAILURE_REPLAY.md)

每例缓存同帧取 q0、Q、future、双种子 PPO、双种子监督七个历史，使用对应折和目标种子的 PPO、监督重新打分。统计四框支持、良好 q0 被替为低框、漏选、候选缺失，以及参考历史仍有良好候选的时刻。续跑使用实际 PPO 历史和原始模板，起点只改选一次，随后执行相同 PPO。主要读数为强制 q0 相对基准动作的后续均值 IoU 差和低 IoU 帧数差。

窗口最多 32 帧，0003 首败只剩 7／5 帧。上限 52 分支、5092 次搜索、35644 次块调用。训练更新、完整新轨迹、在线恢复支线均为 0。结果适用于这些开发起点。输入仍为 `output/stage74/direct_four_candidate_ppo_v1` 的完整目录。

2026-11-08，原 L40 入口完成缓存诊断后停在设备检查，当次作业分到 L20。独立入口用 L40 缓存 PPO 历史固定起点，在 L20 重算共享观测和 PPO 基准，同卡运行所有不同候选的短分支。L40 缓存的动作和输出用于差异表。继续使用原权重、配置与来源软件版本。 [L20 服务器命令](docs/experiments/06_sgla_uav_tracking/STAGE75_FOUR_CHOICE_FAILURE_REPLAY.md#l20-服务器命令)

```bash
mkdir -p output/stage75
nohup bash tools/run_stage75_l20_replay.sh > output/stage75/run_stage75_l20.log 2>&1 < /dev/null &
echo $!
```

同步后在 `sglatrack-test` 环境执行，默认 CUDA 逻辑编号 `0`。用 `tail -f output/stage75/run_stage75_l20.log` 查看。新根 `output/stage75/four_choice_failure_replay_l20_v1` 独立于原 L40 根。回传 `docs/experiments/06_sgla_uav_tracking/results/stage75_l20_results_*.txt`，包含 diagnostic、device_bridge、branches 及 L40 缓存 anchors、same_state、divergence。运行中断后使用已绑定 L20 UUID 继续，换卡时用 `STAGE75_L20_ROOT` 指定新根。

运行前，本机 5 项 Stage-75 回归测试通过，包含原 L40 入口的 3 项及 L20 入口的 2 项。L20 测试使用小型真实 CENTER 与模拟输出漂移，核验共享裁剪、同卡比较、重复框掩码及中断续跑。L20 正式结果已回传，见 0.43。

### 0.43 2026-11-08 Stage-75 L20 回传结果

状态 `COMPLETE_L20_MATCHED_FOUR_CHOICE_FAILURE_REPLAY`。5 个开发视频、13 个预定起点、91 个缓存上下文、52 条 L20 分支、5092 次搜索和 35644 次块调用。13 个起点及基准窗口的 PPO 动作和输出框与 L40 缓存一致。`exact_source_windows=0` 由内部特征和打分的逐字段数值差异产生，起点最大 logit 差约 `2.04e-8`。 [完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE75_L20_RESULTS_REVIEW.md)

- **0003**。f138／140 的 q0 当帧 IoU 为 0.7208／0.6900，PPO 都选了 IoU 为 0 的 q3。D01 改 q0 只修复首帧，D02 后续均值 IoU +0.2975、低 IoU 少 3 帧且末帧仍为 0。提前 8 帧改选在这两个起点未减少低 IoU 帧数。
- **0304／2172**。f1078–1090 自身历史连续 13 帧无 IoU ≥ 0.5 的候选，四条参考历史同帧都有。f1082 自身最佳仅 0.2597，四个分支 32 帧内均未出现好框。f1090 最佳降至 0.1926，所有续跑的 32 帧均低于 0.2。稳定种子同帧各分支均保持好框。较早的状态分岔仍待定位。
- **保留已有收益**。0176／2172 f321 的 q2 是唯一好框，改任何其他选项都导致 32 帧零 IoU。0204／2172 f664 改 q0 也导致 32 帧零 IoU。0307 的两个恢复起点改 q3，后续均值 IoU 分别 +0.3599／+0.3586。固定改 q0 的后续均值为 1 改善、6 持平、6 降低。

接续时分别处理好框被错过与候选池退化，保留 R02、R06 为恢复保护对照。当前证据限于固定开发窗口，完整视频表现以 Stage-74 为准。运行目录 `output/stage75/four_choice_failure_replay_l20_v1`。回传原件、诊断指纹及 5 表 SHA 已核对，逐帧证据留服务器。 [服务器回传原件](docs/experiments/06_sgla_uav_tracking/results/stage75_l20_results_20261107T170631Z_35417.txt)

### 0.44 2026-11-08 论文目标与 Stage-76

最终方法需要改动 SGLA 的一项实际跟踪机制，并用完整轨迹检验贡献。Stage-76 是选择策略的组件对照，用来判断后续收益监督和 PPO 哪种更适合保留。Stage-75 提供失跟诊断及后续模块定位的依据。

Stage-76 用同一批 fit 历史、同一四选一模型比较当帧 IoU 标签与 8 帧同起点续跑标签，并在同一张卡上重测原 PPO、Q、future 和 q0。固定开发预算为 1152 个 fit 起点与 108 条评价轨迹。PPO 与两套 CE 的历史训练预算分别披露，标签影响由两套匹配 CE 的对照识别。L20 正式运行已回传，见 0.45。 [协议和命令](docs/experiments/06_sgla_uav_tracking/STAGE76_MATCHED_HORIZON_SUPERVISION.md)

模块方向由证据确定。先检查 0304／2172 失跟前后的四个搜索裁剪对目标的覆盖、响应网格可解出的最佳框与实际输出。覆盖不足时优先改原生跟踪流程中的搜索状态与区域构造。覆盖充分且网格中仍有好框时优先改 CENTER 的峰值选择与解码。覆盖充分而整个网格都缺好框时，先定位中心或宽高回归的误差。

确认单一模块改动后，用原／新模块 × 基准／选定策略的同预算消融分别估计两项贡献。另报告原生单搜索 SGLA 与最终方法的完整视频 AUC、长失跟、稳定轨迹损伤和实际开销。模块和策略定型后在未参与设计的源视频上验证，论文按所得证据确定收益和适用边界。

### 0.45 2026-11-08 Stage-76 回传结果

状态 `COMPLETE_MATCHED_HORIZON_SUPERVISION_DEVELOPMENT`。同一张 L20 上完成 1152 个 fit 起点、4062 条短分支、108 条完整轨迹、605244 次搜索。匹配 8 帧监督 AUC 57.1320，当帧监督 63.3936，差 −6.2616、2 胜 10 负，家族区间 [−14.4867, +1.5992]。两种子差 −1.1109／−11.4123，低 IoU 增加 10.43 个百分点。fit 标签有 547 处变化，事后 8 帧均值仅增加 0.00455。 [Stage-76 结果复核](docs/experiments/06_sgla_uav_tracking/STAGE76_RESULTS_REVIEW.md)

冻结 PPO 为 67.2356，比 8 帧监督 +10.1036、家族区间 [+2.8728, +18.4926]，两套训练曝光与预算不同。PPO 比 Q 和既有 future 的差仍为 +0.9918／+0.3102，稳定保护未过。8 帧监督在 0176、0226 的种子 2172 丢失恢复，在 0304 两个种子分别从第 556／554 帧持续失跟。0304 的这个较早失效段与 Stage-75 的 PPO 第 1078–1090 帧候选缺口可分开做模块诊断。Stage-63 在另一条历史的第 555 帧已有裁剪覆盖充分、网格可解好框却未输出的对照，新的两条历史需各自复核。 [Stage-63 结果](docs/experiments/06_sgla_uav_tracking/STAGE63_RESULTS_REVIEW.md) [Stage-75 结果](docs/experiments/06_sgla_uav_tracking/STAGE75_L20_RESULTS_REVIEW.md)

接续优先定位 SGLA 搜索／解码模块的限制。Stage-76 保留为标签消融，回传原件及随包表 SHA 已核对，另由逐种子摘要重建了服务器 `video_summary.csv` 的哈希。服务器根 `output/stage76/matched_four_candidate_horizon_v1`。 [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage76_results_20261108T021842Z_144458.txt)

### 0.46 2026-11-08 Stage-77 搜索与解码定位

服务器 L20 已完成并导出 105 个状态、420 次搜索。报告指纹与三张表 SHA-256 已核对，重放框最大误差 0 像素、公开好候选变化 0。初次运行的退出清理异常发生在分析写出之后，结果通过 `--pack-only` 从原目录导出。 [原始结果](docs/experiments/06_sgla_uav_tracking/results/stage77_results_20261108T043231Z_242387.txt) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE77_SGLA_MODULE_DIAGNOSTIC.md)

0304 早期首败仍有好候选。晚期 PPO 2172 在 1078–1092 连续 15 帧中四个裁剪全覆盖，但完整网格无好框，60 个输出宽度为真值的 2.27–5.73 倍。只替换中心时 60 个均未达 0.5，只替换尺寸时 13 帧至少一个候选达标。下一轮优先做因果尺度更新，把发布框尺寸和后继搜索尺度拆开做匹配对照，再用同预算完整轨迹检验稳定保护与恢复保留。当前没有新模块闭环结果。 [详细判读与下一轮设计](docs/experiments/06_sgla_uav_tracking/STAGE77_RESULTS_REVIEW.md)

### 0.47 2026-11-08 Stage-78 匹配尺度反馈原型

服务器 L20 已完成 9 个窗口、每臂 244 帧、1,952 次实际搜索。原生组与 Stage-76 框、裁剪及动作精确复现，回传三表 SHA-256 及报告指纹已核对。比较原生更新、只限幅后继搜索、只限幅发布框，历史五帧逐轴对数中位数和 1.25 增长上限固定。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE78_MATCHED_SCALE_FEEDBACK.md) · [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage78_results_20261108T051056Z_248317.txt)

0304／2172 主窗口均值 IoU 0.2110→0.3737，好候选帧 6→13，首败推迟 3 帧，窗口内仍未恢复。只修发布框为 0.2339。0204 则 0.7472→0.0936，664 帧好候选仍有 0.7656，PPO 从 q2 改选 q0，666 帧开始四候选均为 0。恢复保护未过。下一轮在 0204:664 和 0304:1075 交叉比较同历史下的裁剪与动作，拆开首帧作用与后续漂移，再决定尺度保护怎么做。 [详细结果与下一轮](docs/experiments/06_sgla_uav_tracking/STAGE78_RESULTS_REVIEW.md)

### 0.48 2026-11-08 确认集与 Stage-79 首帧交叉

服务器 L20 已完成 6 条短分支、672 次搜索，四条自然分支精确复现 Stage-78。确认集清单已回传并核对指纹，共 28 个源组、28 条序列、21,103 帧（去掉初始化后 21,075 帧），与去重后的 40 个已知排除组无重叠。Stage-79 确认集推理为零，清单留到方法定型后使用。 [原始结果](docs/experiments/06_sgla_uav_tracking/results/stage79_results_20261108T055823Z_265937.txt) · [确认集清单](docs/experiments/06_sgla_uav_tracking/results/stage79_confirmation_v1.json)

0204:664 限幅后 q2 仍为 0.7057，PPO 因约 0.00080 的 logit 差改选 q0。固定一次 q2 后 28 帧均达 0.5；两尺度固定 q0 均整窗为 0。0304:1075 同 q2 使起点 IoU 0.3898→0.6111，窗口均值 0.1735→0.3531，仍有未恢复事件。下一轮冻结可部署的尺度保护，转向 12 个开发视频完整轨迹消融；确认集留到方法定型。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE79_RESULTS_REVIEW.md)

### 0.49 2026-11-08 Stage-80 完整轨迹尺度保护消融

服务器 L20 已完成 12 个开发视频、84 条完整轨迹、338,125 次搜索。原生单搜索／原尺度 PPO／限幅 PPO／加保护 AUC 为 63.4512／67.2356／65.4064／66.2893。加保护相对原尺度 −0.9463 AUC 点，95% 区间 [−2.9652,+0.4302]，原生稳定视频仍有 0003 和 0304 新增失跟。0204／2172 完整轨迹恢复至 72.6436 AUC 且无失跟，0304／2172 首次失跟从原尺度的第 1090 帧提前到第 556 帧。保护换框 15 次。开发门槛未通过，确认集推理为零，保留冻结清单。后续如继续改尺度机制，转向固定预算内保留原尺度候选的完整开发轨迹消融。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE80_RESULTS_REVIEW.md) · [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage80_results_20261108T104119Z_342748.txt)

### 0.50 2026-11-08 Stage-81 混合尺度完整轨迹

服务器 L20 完成 12 个开发视频、84 条完整轨迹、338,125 次搜索。原生／原尺度 PPO／混合 PPO／混合加保护的 AUC 为 63.4512／67.2356／63.4649／63.7399。主比较 −3.7707 AUC，95% 区间 [−8.8137, −0.1862]，两种子均负。0304 两种子分别在 555／554 帧失跟，0204／2172 两混合臂均在 664 帧失跟。两套方案均未通过开发门槛，确认集零推理。停止微调当前混合布局；后续若拆解机制，优先做完整开发样本的共同历史候选上界与漏选对照。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE81_RESULTS_REVIEW.md) · [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage81_results_20261108T124540Z_389385.txt)

### 0.51 2026-11-08 Stage-82 同历史候选供给与选择诊断

服务器 L20 已完成，扫描 48 条预测历史的 54,100 个状态，回放 1,203 个活跃状态、7,218 次搜索。两种来源对应的 Stage-81 框与动作精确复现。原尺度／混合历史的活跃帧候选上界差为 −0.01668／−0.00613，好候选支持差为 −2.74／−1.15 个百分点，选择损失点估计未增加。重训门槛未通过，结束当前混合尺度配方；本轮是同历史诊断，不能从局部 IoU 解释全部整段 AUC 差。确认集零推理，保持封存。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE82_RESULTS_REVIEW.md) · [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage82_results_20261108T141916Z_445506.txt)

### 0.52 2026-11-09 Stage-83 冲突前中心短时保留结果

同一 L20 跑完 12 视频、60 条完整轨迹和 229,925 次搜索。原生、原尺度 PPO、中心保留 PPO 的 AUC 分别为 63.4512、67.2356、65.6062。中心保留相对原尺度 PPO 下降 1.6294 点，95% 配对区间 [−4.8929, +0.0927]，低 IoU 与最长失跟分别增加 2.44 和 2.72 个百分点。五项开发门槛均未通过，原尺度 PPO 已有的两个原生稳定视频失跟仍在。

46 次触发实际改变 45 帧裁剪。只有一帧在该方案的四候选中由保留对独家提供 IoU ≥ 0.5 的框，PPO 仍选到 IoU 为零的框。最大退化来自 `uav0000176_00000_s` 的种子 2172，两臂首败均在 314 帧，原尺度 PPO 后续恢复，中心保留版这次失跟持续至结束。候选日志来自中心保留版自己的历史，无法分离同一历史下候选替换的效果与后续历史分叉的效果。当前配方退出候选，28 组确认集推理为零。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE83_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE83_CONFLICT_CENTER_MEMORY.md)

### 0.53 2026-11-09 Stage-84 固定四候选与运动生成四候选

服务器 L20 完成 12 个开发视频、24 条冻结 PPO 历史、27,050 个配对状态。固定四候选精确复现 Stage-76。两臂各 108,200 个逻辑槽位，合计实际搜索 187,760 次。新规则保留 q0 与各槽位宽高，只根据运动和互补覆盖改其余中心。26,978 帧改变了取整裁剪。视频等权平均的不同裁剪数为 3.322 → 3.994。

最佳候选 IoU 由 0.70046 升至 0.70225，差 +0.00179，95% 视频配对区间 [+0.00020, +0.00353]，9 个视频改善。好框比例按视频等权 +0.124 个百分点，区间跨零。逐状态共新增 107 帧、丢失 110 帧好框，两种口径的符号差来自视频长度权重。0226 的好框净减 21 帧，0176 净减 5 帧。三项预设候选供给门槛通过，幅度较小。下一阶段只在同预算闭环中检验选框与稳定性。候选回放无 PPO 训练、无新闭环轨迹，28 组确认集仍封存。本机已核对回传的三张表并重算配对统计，逐帧表及缓存留在服务器。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE84_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE84_MATCHED_MOTION_CANDIDATES.md)

### 0.54 2026-11-09 Stage-85 运动生成四候选完整闭环

服务器 L20 完成 12 个开发视频、60 条完整轨迹、229,925 次搜索。两 PPO 臂共用冻结的 Stage-74 策略，每个跟踪帧各搜索四次。原生 SGLA、固定四候选 PPO、运动生成四候选 PPO 的 AUC 分别为 63.4512、67.2356、62.6914。运动对固定的 AUC 差为 −4.5442 点，95% 配对区间 [−11.9225, +0.9403]，低 IoU 和最长失跟分别增加 7.03 和 7.45 个百分点。五项开发门槛均未通过。`0304` 和 `0176` 的提前且持续失跟是主要退化来源。

各自历史上，运动生成的最佳候选 IoU 从 0.70046 降至 0.65308，实际选框 IoU 从 0.68250 降至 0.63664，选框与最佳候选的平均差距略缩小。Stage-84 的 +0.00179 来自固定历史上的候选回放。Stage-85 的闭环统计涵盖已经分叉的轨迹，首次偏离的原因仍需同历史诊断。固定四候选 PPO 继续作为开发对照，当前运动生成配方退出待确认候选。28 组确认集推理为零。本机从回传表重算并核对结论，逐帧 GPU 缓存留在服务器。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE85_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE85_MOTION_CANDIDATE_CLOSED_LOOP.md)

若接续候选生成方向，先做同历史、等搜索预算的三臂诊断，比较固定布局、零速度覆盖、运动覆盖，再用同一冻结 PPO 比较实际选框。把首次输出分叉和失跟前窗口分开分析，必要时只替换一帧布局、用共同后续策略重放，以区分候选损失、选择错误与误差传播。 [待补的机制对照](docs/experiments/06_sgla_uav_tracking/STAGE85_RESULTS_REVIEW.md#待补的机制对照)

### 0.55 2026-11-10 Stage-86 同输出搜索中心干预

服务器 L20 已完成 Stage-86，固定臂精确复现 Stage-76。12 个开发视频、两个种子、144 对 16 帧窗口、18,432 次搜索，83 个窗口首帧裁剪像素发生变化。两臂共用起点发布框，只在下一帧移动干预臂 q0、q1 的搜索中心。高预测位移组平均 IoU 差 −1.38 点，95% 视频配对区间 [−3.50, +0.04]，低 IoU 帧比例 +1.74 点。低位移组平均 IoU 差 +0.04 点。无训练，确认集推理为零。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE86_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE86_SEARCH_CENTER_WRITE.md)

高位移组下降主要来自 `0147` 与 `0307`。`0147` 种子 2172 在起点第 1164 帧已低 IoU，后继帧两臂仍有相同的好候选 q2。中点臂改选 q1，16 帧均为零，固定臂恢复。统一中点写入没有改善依据，继续保留固定 PPO 开发对照。接续诊断可做同起点、同动作续跑，区分搜索几何本身与 PPO 改选对恢复的影响。回传原件和四张表已核对，分支 NPZ 留服务器。 [回传原件](docs/experiments/06_sgla_uav_tracking/results/stage86_results_20261110T014013Z_1027331.txt)

### 0.56 2026-11-10 Stage-87 搜索几何与 PPO 首次动作交叉

服务器 L20 已完成 Stage-87。144 个 Stage-86 起点、576 个条件、55 条新增续跑、47 个窗口首帧精确复现、3,676 次新增搜索，无训练与确认集推理。高预测位移组 16 帧平均 IoU 总差 −1.38 点。固定原槽位仅改布局 −0.06 点 [−0.51, +0.27]，中点布局内改选 −1.32 点 [−3.42, −0.08]。逆向分解也以改选为主要平均损失。低 IoU 帧比例两项各增加约 0.87 点。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE87_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE87_SEARCH_GEOMETRY_ACTION_CROSS.md)

`0147` 种子 2172 第 1164 帧后，两布局选 q2 均恢复，选 q1 均为零。`0307` 种子 2171 第 697 帧后，同选 q1 时中点布局的首帧 IoU 略升、后续明显下降。接续可先在冻结缓存上验证单套布局的在线信号能否识别有害写入，同时保留恢复动作，再决定是否做条件化状态更新。回传表已复算，逐帧 NPZ 留服务器。

### 0.57 2026-11-10 Stage-88 单布局写入信号诊断

服务器 L20 已完成 Stage-88。144 个固定布局窗口、143 个有不同备选、120 条新增续跑、7,680 次新增搜索。Hann、速度和共识规则的高位移后续平均 IoU 差为 −0.13、+0.17、+0.09 点，区间均跨零。低位移三规则均增加 1.13 点低 IoU 帧。实质差异只有 1 个收益、2 个损失，均在 `0307`。未达到排序 AUC 支持门槛，起点已低 IoU 的组也没有仅一臂恢复的可比较窗口。无训练、无确认集推理。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE88_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE88_SINGLE_LAYOUT_WRITE_SUPPORT.md)

`0226` 的两个低位移窗口中，三项信号都支持备选，但低 IoU 帧增加，后续平均 IoU 损失低于 0.05。若继续，应预先扩大源组与分歧状态的覆盖，保留普通窗口作为对照，单列恢复和低 IoU 尾部。回传表已核对，逐帧缓存留服务器。

### 0.58 2026-11-10 Stage-89 候选冲突抽样

服务器 L20 已完成 Stage-89。28 个开发源组、40 条基准历史、158 对窗口、72,200 次新增搜索，无训练与确认集推理。实质 IoU 差异在冲突组为 11/38、普通组为 3/120。同有两类窗口的 15 个源组上，冲突抽样使差异发生率高 25.33 点 [9.78, 43.78]，低 IoU 帧数实质差异发生率高 20.89 点 [5.33, 39.56]。追加组 5/19 冲突窗口有实质 IoU 差异，48 个普通窗口无一达到。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE89_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE89_CONFLICT_SAMPLED_WRITE_SUPPORT.md)

冲突组备选有 6 次实质收益、5 次实质损失，来自 6 个和 4 个源组，平均 IoU 差 +2.34 点，区间跨零。冲突组起点已低 IoU 比例更高，结论是抽样效率提高。独有恢复两种方向各一次。接续可直接用冻结续跑检查三项预设信号的切换方向和低 IoU 代价。回传表已重算，逐帧缓存留服务器。

### 0.59 2026-11-10 Stage-90 固定写入信号方向诊断

本机完成 158 个 Stage-89 窗口复算，无新训练、搜索或确认推理。冲突组 Hann、速度、共识的排序 AUC 为 0.467、0.533、0.550，后续平均 IoU 差 +2.13、+2.74、+1.04 点，三个区间都跨零。速度和共识各避开 5 次实质损失中的 3 次，但三项规则均切换 `0091` 的唯一原 PPO 恢复窗口，改选多 10 个低 IoU 帧。普通组 `0147` 的备选独有恢复由 Hann 和共识选中。保持原 PPO 作开发对照，冲突可作诊断取样，方向判据仍待验证。 [结果与复算](docs/experiments/06_sgla_uav_tracking/STAGE90_FROZEN_SIGNAL_AUDIT.md)

### 0.60 2026-11-10 Stage-91 首帧模板外观一致性

服务器 L20 已完成。158 个动作对，632 次诊断搜索与 4,424 次骨干块调用，取样帧逐项复现，外观分数均有效。冲突组 patch 后续 IoU 差 +1.36 点 [−1.35, +5.40]，RGB −0.38 点 [−0.91, +0.01]，排序 AUC 0.333 与 0.400。两项外观均保住 `0091` 的原 PPO 独有恢复，也都漏掉 `0147` 的备选独有恢复。复用后续续跑，无训练和确认推理。回传表已复算，特征缓存留在服务器。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE91_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE91_TEMPLATE_APPEARANCE.md)

### 0.61 2026-11-10 Stage-92 短时历史外观

服务器 L20 已完成。158 个原动作对、926 次诊断搜索与 6,482 次骨干块调用，固定模板复现 Stage-91，无训练、新续跑或确认集推理。38 个冲突窗口内，筛选历史相对固定首帧后续平均 IoU +4.49 点 [+0.41, +11.75]，低 IoU 比例 −8.22 点 [−22.44, −0.03]，排序 AUC 0.700。相对原 PPO 后续平均 IoU +5.85 点 [+0.35, +14.31]。`0016` 贡献约 78% 的固定首帧相对增益，去掉该组仍为 +1.08 点。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE92_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE92_RECENT_APPEARANCE.md)

筛选历史保住 `0091` 的原 PPO 独有恢复，也选中 `0147` 的备选独有恢复，各只有一例。仍有 `0226` 的实质有害切换。接续可冻结筛选历史规则，进入完整开发轨迹的同预算闭环比较与稳定性保护，确认集继续封存。回传表已重算，逐窗口特征留服务器。

### 0.62 2026-11-10 Stage-93 筛选历史外观闭环

服务器 L20 已完成。28 个开发源组、148 条完整轨迹、356,098 次新增搜索。原生、PPO、首帧、筛选历史 AUC 为 63.96／65.09／61.92／65.83。筛选历史相对 PPO +0.74 点 [−2.10, +3.69]，相对固定首帧 +3.92 点 [+0.23, +8.43]。原队列相对 PPO −1.28 点，追加队列 +2.26 点。PPO 稳定视频无新增失跟，但 25 个原 PPO 恢复有 4 个未在对齐窗口内保留，开发门槛未过。无训练与确认集推理。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE93_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE93_APPEARANCE_CLOSED_LOOP.md)

筛选历史换框 21,920 次，占 51.5%。恢复损失在 `0147`、`0226`、`0307`，其中 `0226` 平均 AUC −23.68 点。接续可从四个恢复事件做首次分叉与历史写入的匹配回放，区分单次误写与重复反馈。回传表已核对，逐帧轨迹及记忆缓存留服务器。

### 0.63 2026-11-10 Stage-94 筛选历史外观的恢复诊断

服务器已完成。25 次原 PPO 恢复中，4 次未匹配。原 PPO 恢复首帧，4 次在筛选历史状态上均无 IoU ≥ 0.5 的候选，最佳 IoU 分别为 0.343、0.019、0.195、0.000。其窗口共 79 帧筛选历史无好框而 PPO 有好框，未见严重同状态换框或屏蔽单条引用即可从低 IoU 选回好框。从各自 PPO 失跟起点至视频结束，筛选历史均无连续三帧 IoU ≥ 0.5。

21 个保留恢复中，5 个在原恢复首帧也缺好框但后来恢复，4 个窗口共出现 6 帧严重同状态换框。下一步优先检查恢复前的搜索状态和候选供给，可交叉几何历史检验目标覆盖与响应图，再做同起点短分支。首次裁剪分叉在第 3 或 10 帧，尚不能定位哪次写入引起后续失跟。本机核对决策、事件表、四个源码哈希与 25 次恢复对齐，逐帧和引用屏蔽明细留在服务器。没有新增搜索、训练或确认集推理。 [Stage-94 结果](docs/experiments/06_sgla_uav_tracking/STAGE94_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE94_APPEARANCE_RECOVERY_AUDIT.md)

### 0.64 2026-11-13 Stage-95 恢复失败的裁剪与完整响应

服务器 L20 已完成 25 个恢复事件的 137 个时刻、两历史 274 个状态、1,096 次搜索及 7,672 个块调用。源框最大误差 0、可用性变化 0、硬件／软件身份差异为空。4 次未保留恢复的首恢复帧分别为 0147:1165 全格点有好框但保留峰未覆盖（最佳 0.667），0226:343 与 0307:474 覆盖 100% 但全格点无好框（最佳 0.055／0.263），0307:712 覆盖 0%。当前证据没有定位到环境成因或某次早期写入。

未保留事件的 23 个关键状态分为全格点有好框但未被保留峰覆盖 6、覆盖充分但无好框 12、裁剪支持有限 5，既定多峰规则补出好框为 0。已保留事件 114 状态中，已有单峰好框 70，保留峰另补好框 23。下一步优先共同图像下交叉中心和尺寸检查输入几何，多峰保留为针对另一类失败的候选分支。训练、确认推理与完整新轨迹均为零。本机核对决策、两表哈希与事件对齐，逐状态／裁剪及 NPZ 留服务器。 [Stage-95 结果](docs/experiments/06_sgla_uav_tracking/STAGE95_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE95_RECOVERY_DENSE_RESPONSE.md)

### 0.65 2026-11-13 Stage-96 搜索中心与尺寸交叉

服务器同一 L20 已完成 137 个时刻、548 个组合状态，复用 PP／GG，新增 1,096 次搜索、7,672 个块调用。原组合误差 0，供给可用性变化 0。P 为原 PPO 来源，G 为筛选历史来源，首字母取中心、次字母取尺寸。0147:1165 的 PG／GP 最佳单峰 IoU 为 0.672／0.632，均较 GG0.343 变为好框。0307:474 只换尺寸 GP 得 0.664；0226:343 的 GP 全格点达 0.580，但单峰仍 0，PP 单峰 0.859。0307:712 的 PG 覆盖恢复 100% 却最佳单峰仅 0.022，PP 才有 0.554 好框，固定 G 槽位仍仅 0.491。

未保留事件的 23 个关键状态中，GG／PG／GP／PP 有好单峰为 0／4／6／14。最佳单峰均值差 PG−GG +0.0567、GP−GG +0.0705、PP−GG +0.3609，交互 +0.2337。21 个已保留事件的恢复帧，GP 保留 GG 的全部 16 个有好单峰时刻并新增 1 个，PG 丢 5 增 3。优先研究自身可观测历史下的尺寸更新及中心尺寸配合，P 来源是另一条已运行轨迹，不能直接当免费在线参考。固定 G 槽位分类与原 P 策略类别须区分。训练、确认推理及完整新轨迹为零。本机核对三表哈希、事件、缓存名、预算和因子等式，逐状态／裁剪缓存留服务器。 [Stage-96 结果](docs/experiments/06_sgla_uav_tracking/STAGE96_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE96_SEARCH_CENTER_SIZE_CROSS.md)

### 0.66 2026-11-13 Stage-97 自身可信历史尺寸后继试验

服务器同一 L20 已完成 25 窗口／50 分支，新增 12,648 次搜索、88,536 个块调用，control 原框误差 0。trusted_scale 对原 4 个未保留事件新增 0226／2171 的 343 帧恢复，但丢失原已保留的 0307／2172（control 475 恢复）和 0076／2172（control 1477 恢复），两条新分支在窗口末均未恢复。总对齐恢复 21→20。此为固定起点窗口结果，完整轨迹 AUC 尚未检验。

新规则改 q1／q3 的自身历史尺寸，可靠记录近十二帧最多五条、至少两条，资格沿用外观写入与有限描述符。3,162 个分支帧中，引用 3,156、父框改变 1,378、取整裁剪改变 1,250（39.53%）。0307／2171 的 652～729 窗口平均 IoU0，仍有 77 次尺寸写入，可信资格可接受错误状态。优先核对损失前的首次分叉、引用和错误写入，再决定更新机制。训练、完整新轨迹和确认推理为零。本机核对三表哈希、事件、恢复时刻、预算，并从窗口表重算视频／队列汇总，分支缓存留服务器。 [Stage-97 结果](docs/experiments/06_sgla_uav_tracking/STAGE97_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE97_OWN_TRUSTED_SCALE_CONTINUATION.md)

### 0.67 2026-11-13 Stage-98 尺寸记忆与首次分叉缓存审计

服务器已完成25事件／50分支／6,324分支帧审计，重算15,614条尺寸引用。0307／2172与0076／2172的损失分支分别在379／611帧首次裁剪和输出分叉，窗口内首次低 IoU 写入却在403／639帧，首次分叉引用均无 IoU<0.2 记录。当帧新组 IoU 略高，之后470／1475帧才首次丢失好候选。因此不能直接把最初分叉归给后期低 IoU 写入，尚未定位唯一因果链。

新组2,581次写入中1,165次低 IoU，7,241次低 IoU 引用中5,203次屏蔽可改变裁剪。0226获益分支122帧首次分叉时五条引用全低 IoU，仍在343恢复，不能仅用低 IoU 引用存在与否判断有害。下一步可共同起点比较一次尺寸改动与持续更新，并独立检验后期反馈。替代裁剪未推理。新增图像、网络、训练与确认推理均为0。本机核对两表哈希、事件转变、分支写入和范围，逐帧及引用明细留服务器。 [Stage-98 结果](docs/experiments/06_sgla_uav_tracking/STAGE98_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE98_SCALE_MEMORY_AUDIT.md)

### 0.68 2026-11-14 Stage-99 一次尺寸调整与持续更新结果

服务器同一L20已完成25窗口／75分支，control与continuous复用50来源尾段，once新增11,516搜索／80,612块。三组逻辑34,548搜索，2,879尾段帧／8,637分支帧，来源最大框误差0，共同前缀与首步一致性报告通过。SGLA、PPO和外观写入资格冻结；训练、完整新轨迹和确认推理均0。本轮COMPLETE仅为诊断完成。

对齐恢复control／once／continuous为21／23／20。once在原4个未保留事件补回0226／2171／r343（起点122、恢复343）和0307／2171／r474（起点371、恢复474），全部保留原21次恢复。持续组丢失的0307／2172／r471与0076／2172／r1477在once均保留，恢复471／1477，末帧IoU0.7363／0.9180；continuous末帧0.1684／0且未匹配。两组同首步后结局不同，支持继续研究持续规则及状态反馈，尚不能唯一归给某条错误尺寸写入。另两未保留事件0147／2172／1165和0307／2171／712仍未修复。

once仅25次启用、25次改父框、22次改取整裁剪；continuous激活2,873帧、改父框1,378、改裁剪1,250。once对control在原未保留队列按视频等权平均IoU+0.05143、低IoU−4.5956个百分点；原已保留队列平均IoU−0.00572、低IoU+1.1190个百分点。0085／2171／r317虽仍在317恢复，但286～334尾段平均IoU0.6545→0.4743、低IoU0→16/49；continuous该窗口反而0.6837且无低IoU。23/25不能当作能力保护或完整轨迹成功，新增两次还均为2171种子。

锚点是Stage-98缓存中首次搜索父框分叉，采用事件窗口重置，尚未形成完整视频在线启停规则。下一步应固定只依赖自身观测的有限启用与退出机制，并检验完整开发轨迹、恢复与中途质量；不要直接采用once，不开放确认集。三组后续各走自身预测与外观记忆，continuous−once包含这些状态分化，不能称为单条写入的纯效应。

本机核对decision指纹、三表字节哈希、25锚点、75缓存名称、50来源末帧及恢复记录，并从窗口表重算39条视频对比与6条队列汇总，误差0；逐帧NPZ与manifest留服务器，未本地重演。原件已归档，SHA256`cb35fc217885c3d8e4f00a7e61347841268f80844de67bd44e4d4d181b6bf3a3`，decision指纹`ccc86856f8b6fdd7a3c195a373cbb10b32c6890ed54ea184db4d1383e7161310`。无需重复索取四份已回传材料，也不重跑／改写Stage-99代码。 [Stage-99 结果](docs/experiments/06_sgla_uav_tracking/STAGE99_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE99_MATCHED_SINGLE_SCALE_PULSE.md)

### 0.69 2026-11-14 Stage-100 完整视频在线尺寸启停设计（结果见0.71）

已新增`tools/stage100_online_scale.py`、`tools/run_stage100_online_scale.sh`和五项CPU测试。接续目标是把Stage-99一次调整的条件收益转成可在线执行的SGLA搜索尺寸改动。每帧搜索前用自身过去资格尺寸和普通历史尺寸计算两套父框；可信尺寸若改变实际取整裁剪则记分歧。pulse只在分歧从无到有时启用一帧，随后回原规则，分歧消失后再次允许触发；无GT／旧事件锚点／恢复时刻输入，也不按事件重置。资格沿用十二帧最多五条、至少两条及原外观写入检查，SGLA／PPO冻结。

全28个已消费开发源组、40个来源单位，原12组双种子、追加16组各沿用来源一个种子。五组native／ppo／gated／continuous／pulse共188条逻辑完整轨迹；同Stage-93身份复用108参照，新80条／340,576搜索／2,384,032块；另一L20／L40或UUID则五组重跑188条／710,199搜索／4,971,393块。四候选组均每帧四搜索。新几何组相对gated首次父框变化前的全部预测、特征和外观前缀必须一致，换卡记录来源误差。训练和确认推理为0，完整轨迹从首帧到结尾独立滚动，不能将23/25局部结果直接带入。

主要比较pulse−gated和pulse−continuous，两项AUC家族区间；另报对PPO／原生、continuous−gated。种子先视频内平均，按视频等权，并保留historical／expanded。固定门槛含对gated至少+0.5 AUC、两主区间下界正、低IoU／最长失跟不增、可靠帧不降、PPO／gated稳定视频无新增失跟、gated恢复全保留，且按具体事件集合不新增丢失PPO恢复。恢复总数相同也不能交换一个损失抵消另一个。continuous沿用旧规则，有引用即启用；pulse额外要求裁剪分歧的开始，差异是整个在线启停方案，非纯持续时长因果消融。全部通过也仅为开发PROMISING。

五项合成网络CPU测试、Bash语法与命令行检查通过，实际几何、解码、特征、记忆及混合队列／多种子采集发布均运行。SSH再次被服务器拒绝认证，GPU试验未启动。同步新文件并激活环境后`bash tools/run_stage100_online_scale.sh`；默认输出`output/stage100/online_scale_episode_pulse_v1`。需后台运行用协议中的nohup命令，中断后换卡须新目录。回传`stage100_results_*.txt`（decision＋policy_summary／paired／video_summary／seed_video／scope／recovery六表）。不要重复运行Stage-99，不开放28组确认集。 [Stage-100 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE100_ONLINE_SCALE_EPISODE.md)

### 0.70 2026-11-14 Stage-100 历史原生q0缓存兼容修复

服务器报`Call accounting shape changed: block_calls`，发生在原生参照审计。Stage-76的q0轨迹始终选择q0但存储／执行四候选，调用形状为`[N,4,12]`；Stage-100把复用native当作`[N,1,12]`，本地旧测试只覆盖单槽参照，这是接口兼容错误，不是方法成绩。

新增`tools/stage100_native_compat.py`及`tools/run_stage100_native_compat.sh`。先核对全部源候选调用、恒选q0及发布框等于q0，再建立q0单槽副本，并按单搜索自己的历史、裁剪和调用逐帧审计。历史四槽物理调用数保留在`native_source_slots`／`native_source_head_calls`及最终报告逐视频`native_source_views`；native好候选比例只评q0。无新增适配推理，不改变在线尺寸规则、PPO、预算或28组范围。

原`stage100_online_scale.py`、原runner、全部上游代码／来源和旧manifest均保持字节不变，避免代码哈希导致第二次启动拒绝。适配器另存`stage100_native_compat.json`，绑定旧计划、旧代码与新适配代码哈希；decision加入兼容回执绑定。原有缓存先核对原receipt再读，既有字节不改写。七项CPU测试通过，含真实四槽格式复现、未用候选坏调用拒绝、混合一／四槽、两种硬件模式完整发布、manifest不变及零推理缓存续跑；Bash语法和CLI通过。

当前续跑命令改为`bash tools/run_stage100_native_compat.sh`，默认原`STAGE100_ROOT`无需删manifest；若GPU型号或UUID改变仍须指定新目录。后台命令见修复文档，旧脚本不会安装兼容视图。回传`stage100_results_*.txt`现增加兼容回执，保留原decision＋六表。本次仅完成修复，服务器正式评价尚未完成，不新建Stage-101、不改实验设计，不开放确认集。 [Stage-100 修复与续跑](docs/experiments/06_sgla_uav_tracking/STAGE100_NATIVE_CACHE_COMPAT.md)

### 0.71 2026-11-14 Stage-100 完整视频在线尺寸启停结果

兼容入口已在同一L20完成28源组、188条逻辑完整轨迹；108参照复用，新增80轨迹／340,576搜索／2,384,032块，逻辑710,199搜索槽位，来源框误差0，runtime差异为空。回传兼容回执、decision和六表齐全，无需重复索取或重跑。训练与确认推理0，状态`REVIEW_ONLINE_SCALE_PULSE_DEVELOPMENT`。

native／PPO／gated／continuous／pulse宏AUC63.9605／65.0943／65.8348／64.0206／65.2127。pulse对gated−0.6221，95%区间[−3.9434,+2.4006]，家族区间[−4.5725,+2.8106]，16胜1平11负；低IoU+0.4768个百分点、最长失跟+0.8048个百分点、可靠帧AUC−0.6650。对continuous+1.1922，区间跨零，12胜1平15负，去掉0091均值−0.2774；对PPO仅+0.1185、14胜14负、区间跨零。历史12组对gated+0.8389、追加16组−1.7178，均非独立确认。

25个PPO恢复中gated保留21、continuous20、pulse22。pulse补回0226／2171／343及0307／2171／474、712，但新增丢失0307／2172／471和0076／2172／1477；0147／2172／1165仍丢失。相对gated自己的17个恢复，pulse保留15、continuous13。21−2+3=22，具体事件保护未过，不能只看净增一次。相对gated原稳定种子轨迹新增失跟按视频去重4个：0304／2172、0003／2172、0169／2171、0080／2171；PPO稳定种子轨迹无新增不抵消这项损伤。

主损伤为0304视频宏AUC72.4707→47.2781（2172种子22.1366、低IoU67.88%）及0169单种子62.4229→35.0163、低IoU0.36%→41.37%。0226宏30.9268→53.8978，0307宏54.8058→65.6534，有局部机制价值但未建立整体稳定收益。pulse启用并改实际裁剪1,834/42,572帧（4.31%）；continuous引用42,448帧，改裁剪6,248帧。减少改动数量仍不足以保护能力。在线从首帧启动且每次分歧重新允许，与Stage-99已知起点一次干预的23/25处理不同。

十项开发检查仅PPO稳定保护和非平凡使用两项通过。暂不采用当前pulse，也不直接搜索冷却或阈值；优先在已生成轨迹中对齐0304／0169有害分叉与0226／0307获益，做匹配状态与后继检查，连接触发、候选选择和下一次搜索反馈。不能由本轮否定全部尺寸方向，28组确认集继续封存。

本机核对两JSON指纹、兼容回执绑定、六表原始字节哈希、31原绑定文件＋2适配文件字节哈希、188记录及预算；从种子表重算15宏汇总／140视频汇总／120配对，误差0。108来源共同指标与Stage-93精确一致，恢复集合核对相符。原生来源12四槽＋16单槽，历史head69,622，q0视图29,047，未抹掉旧额外调用。NPZ和manifest留服务器，未本地重演网络／逐帧审计。paired中failure_events／unrecovered_events方向字段仍被旧helper标positive，解释时按越少越好，主要AUC与门槛不受影响。原件SHA256`10d112e78f7889b476d6d269f63999e90e3bddf2f6213a4b43007d0bdc49f1c7`，decision指纹`9b237b9c93ad05f7e9fbcf38d203c38ad33cfad432ff184c6a3069f353054915`。本次仅归档复核，冻结代码不变。 [Stage-100 结果](docs/experiments/06_sgla_uav_tracking/STAGE100_RESULTS_REVIEW.md)

### 0.72 2026-11-14 Stage-101 全部触发匹配回放与后继诊断（准备记录，结果见0.73）

已新增`tools/stage101_trigger_replay.py`及`tools/run_stage101_trigger_replay.sh`，绑定Stage-100完整决策`9b237b9c93ad05f7e9fbcf38d203c38ad33cfad432ff184c6a3069f353054915`和兼容回执。首先逐帧重建40单位gated／pulse决策与记忆，再对pulse全部1,834个实际触发做同一过去状态的keep／skip比较：保留尺寸调整或只撤销本次。q0／q2父框、候选、特征、裁剪哈希及调用须相同，原PPO和外观选择在各自候选上重新计算。当前GT仅作输出评价，无新训练／完整轨迹／确认推理。

后继锚点在新推理前根据既有Stage-100确定：每单位首次触发、首次gated好而pulse差以前最近触发、首次pulse好而gated差以前最近触发；另覆盖0307／2172／471、0076／2172／1477两次新丢失恢复前最近触发。同帧合并，上限122个唯一锚点，实际数在manifest冻结。默认16帧，失去恢复角色延长到原恢复+17并按视频截断。GT参与离线角色选择，不能将角色当在线信号。

keep与skip共用各自当帧匹配缓存，后续两支均恢复同一pulse规则，各自更新预测、外观和尺寸记忆及上帧分歧，不复制来源未来，也不永久关闭更新。同卡keep当帧／后继均复用来源切片并审计，只有skip新推理；换L20／L40或UUID则两支都重放，共同过去状态仍来自冻结轨迹。设后继首帧以外总帧数S，新搜索同卡4×(1834+S)、换卡8×(1834+S)，每搜索7块；同卡当帧部分7,336搜索。实际预算先打印，首帧不重复计费。

返回decision＋trigger_summary／trigger_video／anchors／branches／branch_summary五张汇总；逐触发和候选两表及NPZ留服务器。无方法采用门槛，COMPLETE仅表示诊断完成。以全触发当帧比较和固定分支检查正常候选、PPO／外观错选及后继反馈，不能拿短窗口均值替代整视频AUC，不能把已分叉来源的观察获益／损伤提前当因果结论。

四项CPU测试通过，含共用过去状态、q0／q2、首帧复用、未来GT隔离、实际调用预算、两硬件模式及缓存续跑；Bash语法和CLI通过。SSH仍拒绝认证，GPU正式运行未启动。本机没有源NPZ，锚点实际数量需服务器先行重建，禁止假称已有新结果。同步后`bash tools/run_stage101_trigger_replay.sh`，默认输出`output/stage101/matched_scale_trigger_v1`；后台命令见协议。回传`stage101_results_*.txt`，不重跑Stage-100、不改其冻结代码、不开放确认集。 [Stage-101 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE101_MATCHED_TRIGGER_REPLAY.md)

### 0.73 2026-11-14 Stage-101 全触发匹配结果（已完成）

回传decision及五张汇总表齐全；状态`COMPLETE_MATCHED_SCALE_TRIGGER_DIAGNOSTIC`，没有方法采用门槛。同一L20及UUID、runtime差异为空，keep复用Stage-100来源且报告框误差0，q0／q2和共享首帧检查通过。全40单位中38单位／27视频发生1,834次触发，61个唯一固定起点产生122短分支，3,790缓存标识；新搜索11,000、块77,000，训练／新完整轨迹／确认推理0。每处理后继尾帧916，总帧977，预算`4×(1834+916)`。57窗口16帧，0003末尾9、0016末尾7、0307恢复23、0076恢复26。

skip仅在同一过去状态取消当次尺寸替换，下一帧仍按同一pulse规则自身推进。全触发当帧IoU339胜1,157平338负，平局占63.09%只表示IoU相等，不表示框／记忆完全不变。最终IoU视频等权差−0.000754816、最佳候选−0.002364843；历史1,136状态／11视频−0.002037976，追加698／16视频+0.000127356，27视频17正10负。没有稳定的统一跳过依据，也没有新的完整AUC。

四角色短分支skip−keep平均IoU／末帧IoU／低IoU比例差：首次触发38起点／27视频−0.000678／−0.003871／0；观察损伤11／11为+0.019837／+0.105380／−0.025613；观察获益13／11为+0.008331／+0.065616／+0.014205；新增恢复损失2／2为+0.141397／+0.117695／0。角色已用来源GT离线固定，可重叠，不是在线信号或独立确认。窗口平均IoU不能当AUC或恢复保留判定。

**0307／2172：** 466～488取消466调整，平均IoU0.411629→0.694422、末帧0.451977→0.687367，好候选11/23→19/23；两支低IoU均0。支持此状态下干预影响后继，但需逐帧NPZ核定原PPO471起十五帧内是否连续三帧最终IoU≥0.5，五表不能直接宣布补回恢复。**0304／2172：** 548～563两支平均0.349261、末帧0、低IoU9/16和好候选9/16均相同，减少一次启用仍未修复555附近损伤。**0169／2171：** 646～661均值0.290188→0.320255、低IoU少1帧、好候选多1帧，末帧均0，仍失跟。**0076／2172：** 1469～1494两支最终IoU全0、无IoU≥0.5公开候选；取消该次未恢复供给，需检查更早状态，不能未经覆盖／图像证据称裁剪外或环境所致。

**0003／2172：** 136～144跳过反而均值0.304717→0.222021，不能统一禁止尺寸调整。0016／2171在632～638则0.406391→0.625013、末帧0.327188→0.728047；0080／2171在199～214为0.466348→0.520459、低IoU少2/16帧。来源获益也非最近动作的唯一贡献：0226／2171在321～336两支均0.150306，0307／2171在473～488跳过反而0.749226→0.782959；过去已含更早干预。

本机独立复算decision指纹、五表原始字节SHA256、来源runtime／fingerprint、3,790缓存标识配对、逐视频触发数、38个首次锚点与points最小帧、61个anchors／branches对应及预算；从视频表重算3汇总，从branches重算4角色汇总，误差≤1e-14。未回传manifest、逐帧NPZ和triggers／candidates，本机未重放网络、全部逐触发IoU或服务器q0／q2逐帧审计，不重复索取已有五表、不重跑该轮。原件已归档SHA256`90730ab74aa3dfd09b2be416cc3f4cfbb1c4b9d4616b2265c6114bbfcd8a9da4`，decision指纹`979f901ca4e7884821a5f32f06bdbdd35c579f47a1fa79b83b4bf0a9d18135ce`，plan指纹`1f2dcb1f87d10d3fe11b8f77dd8a70f40f37baa36eab898fef658b0a03728a2f`。

建议下一步利用现有100／101缓存向前定位0304、0169、0076的进入坏状态链条，以0307／2172作对照，连接触发→候选供给→PPO及外观最终选择→实际写入→下一次搜索。最近一次触发的单点撤销不足以解释全部严重失败；先补状态链证据，再决定候选保留、干预资格或写入机制的修改，不直接搜频率／冷却。本次只归档复核，未新建Stage-102或修改冻结代码，28组确认集继续封存。 [Stage-101 结果](docs/experiments/06_sgla_uav_tracking/STAGE101_RESULTS_REVIEW.md)

### 0.74 2026-11-14 Stage-102 缓存搜索状态链审计（准备记录，结果见0.75）

新增`tools/stage102_state_chain_audit.py`和`tools/run_stage102_state_chain.sh`。绑定Stage-101决策`979f901ca4e7884821a5f32f06bdbdd35c579f47a1fa79b83b4bf0a9d18135ce`及其Stage-100来源，CPU复算全部40开发单位／28组的gated和pulse共80条缓存、85,144个状态，重验1,834对触发、61对分支及64角色；不只追踪三条失败视频。SGLA、PPO和已有规则均冻结，图像解码／新搜索／新骨干调用／训练／确认推理0，不需要GPU或checkpoint参数。

主体区分缺少IoU≥0.5公开候选、PPO漏选且外观未补回、外观丢掉PPO好框、外观补回与选中好框；统计实际接纳的低IoU输出、后续尺寸及有效外观读取。每条实际尺寸引用删除只重算父框／裁剪敏感性，不宣称输出IoU；外观引用删除可在已有四框／查询上重算当帧选择与IoU，但不推进新轨迹。低IoU引用由离线GT标记，不能作在线信号。完整来源分叉是观察描述；与同状态触发干预和两种局部删除的结论范围分开解释。

沿用Stage-101全部锚点／角色，报告目标帧之前最近好输出、之后首次候选缺失／漏选／低IoU写入与读取，以及两完整来源最早父框、裁剪、输出、PPO、外观和写入差异。另附关键帧、最近触发、前两帧预测及实际读取引用的一层依赖摘录，不跨来源目标帧。已有短分支中两次新丢失恢复要逐帧核定，不能凭Stage-101均值直接认定修复。

恢复边界已核对：Stage-93实际代码允许连续三帧好输出的起点在`[参考恢复帧, 参考帧+15]`闭区间，需到+17，先前“十五帧窗口”文字不够精确；本轮沿用原实现，未改旧实验或其结果。无效GT打断连续判定，末尾不足报告截断。

输出十二表与decision。结果包含summary、unit_summary、cases、trigger_summary、recovery、branches、chain_excerpt、reference_summary八表；完整触发、引用边、分支逐帧、来源逐帧四表留服务器。先重算Stage-101触发汇总／视频与分支表，必须相符，再发布；主要比例按种子再视频等权，引用边计数只作相关描述。来源／代码绑定、缓存回执、q0／q2和共享首帧完整性均校验，无方法采用门槛。

五项CPU测试通过，含错误写入及实际读取、引用删除、非有限描述子不计有效外观读取、GT隔离、当前／未来引用拒绝、分类与权重、原恢复闭区间及无效GT、缓存流水线／原表重现、依赖摘录、重复发布及篡改拒绝；Bash语法及CLI通过。综合测试为合成网络缓存与替代选择函数，服务器将复算真实封存PPO。本机没有服务器NPZ，正式审计未运行。同步后`bash tools/run_stage102_state_chain.sh`，默认输出`output/stage102/cached_state_chain_v1`；后台命令见协议。回传`stage102_results_*.txt`，不重跑GPU旧实验、不改冻结代码、不开放确认集。 [Stage-102 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE102_CACHED_STATE_CHAIN.md)

### 0.75 2026-11-14 Stage-102 状态链结果（已完成）

完成40开发单位／28组gated和pulse共80完整缓存、85,144状态、1,834对触发、61对短分支／64角色；262,146条引用边、1,954分支逐帧、2,806摘录。来源runtime仍为Stage-101的L20，CPU审计没有新GPU性能比较；新图像解码／搜索／块／训练／确认推理均0。状态`COMPLETE_CACHED_SEARCH_STATE_CHAIN_AUDIT`，没有方法采用门槛。

视频等权gated／pulse：无好候选13.7773%／14.6280%，有好框最终漏选1.1538%／1.0850%，低IoU10.6570%／11.1339%，低IoU尺寸写入8.5367%／9.0955%，有效外观读取低IoU历史10.6789%／11.1855%。长期状态多数是供给不足，但不能反推首发错帧也是缺框。11个来源首次观察损伤目标分为5个PPO及最终漏选、5个无好候选、1个外观丢掉PPO好框。历史组无好候选改善15.76%→14.54%，追加组恶化12.29%→14.70%，均为开发证据。

**0304／2172／555关键新证据：** 前帧最终IoU0.8008，本帧PPO q2=0、外观最终q0=0，但四候选最佳0.813839，因此好框在q1／q3。现外观只比较PPO与一个challenger，本帧比较对{q2,q0}未覆盖好框。引用553／550／547的自身IoU0.8206／0.7825／0.8078，不能归因历史全是坏框。555未启用尺寸替换、记忆和尺寸写入均拒绝。取消最近548触发的匹配分支中，最终输出、PPO／外观选择及写入／外观引用全程无分叉，不能把548动作当直接修复点。完整pulse无好框1,227帧、低IoU1,171帧、其中尺寸接纳1,033帧；gated仅无好框58帧且低IoU0。

**0169／2171／653：** 前帧最终0.5970，本帧PPO及最终q3=0，另有好候选0.6511，覆盖1，used_history=false、记忆写入拒绝。gated同帧PPO q3也为0，外观换q1得到0.5964；两条过去已不同，不能当作同状态移植成功。取消646动作后短分支仍末帧0。**重要共同路径：** 当前`stage100_online_scale.py:91`无条件把最终box追加普通history，外观／可信尺寸的写入资格不控制这条路径。故0304／555和0169／653的错误框虽拒绝入库，仍成为后续普通搜索中心／尺寸输入；代码和状态重建确认依赖，尚未验证阻断能保住完整能力。

**0307／2172恢复已核定：** 从466共同过去状态只取消这次调整，skip在471开始连续三帧IoU≥0.5，keep未匹配；窗口起点至486、完成至488，完整。469／471帧pulse的PPO已有0.6157／0.6222好框，外观都换至约0.4109；471引用470／467／464 IoU0.4089／0.3921／0.2926，不含<0.2引用，单按低IoU过滤命不中这些读取。局部恢复补回不能将Stage-100完整策略22/25改写为23/25。

**0076／2172：** 1469～1494两支均未匹配1477参考恢复，允许起点至1492且窗口完整，最终输出无分叉。目标前pulse最后好输出646，647无好框、648选框0、649在这个局部阶段接纳低IoU写入、652读取低IoU尺寸；更早640附近两组都曾低IoU，647不是全视频首次失跟。1476／1477 gated输出0.7916／0.7517，pulse最佳框0、最大覆盖约0.5929／0.6323；1469覆盖1仍没有好框，不能仅解释为完全裁剪外。

1,834触发中两组均选好1,110、均无好框656、有好框均漏36；跳过丢／补好候选10／3，两组有好框时跳过救回／破坏选择9／10，无效GT0。低IoU尺寸引用2,219次，单条删除改裁剪1,390次（62.64%），只证输入几何敏感性。低IoU外观引用34,742次，删除后601胜33,530平611负，未支持简单逐条删除全面修复；同帧多引用不是独立试验。原始帧合并低IoU数gated6,053／pulse5,869虽下降，不同于宏比例上升；低IoU输出的尺寸接纳4,950／4,936，帧合并接纳率81.78%／84.10%，勿混用分母。

本机核对decision指纹、八表字节哈希／行数、80单位唯一标识／有效帧／比例／漏选恒等式，重算六宏汇总误差≤1e-14；80单位及六宏的低IoU／供给比例与Stage-100一致，61分支共同字段与Stage-101一致；触发分割、引用总量／暴露和删除计数相符。2,806摘录为2,188唯一来源状态，重复相符、引用严格过去、有效引用为原引用子集，64角色目标相符。manifest、NPZ、完整触发／引用边／两逐帧表留服务器，本机未重放原缓存/PPO或独立重算skip连续恢复，服务器标志与本机核对范围分开。已归档原件SHA256`4faebb8394a980257dedd38a8f60f548644fb647503ce6d7fd31aa01c99a91ef`；decision`2385b6d968cb5aa69ae89ad055a87b02a0591472eedcdb8a9e5fada4454fcf4b`；plan`e0c69502d7b7a408e9cfb4380ea49fd8ad138d560e55676647d8d61285f8ed18`。

下一步优先比较现有两框核验与覆盖四候选的核验，再独立量化“发布输出”和“下一次搜索几何状态”的提交条件，保留相同搜索预算与完整稳定性／恢复对照。已有旧中心保持与尺度干预失败仍须纳入，不能直接宣布新机制有效。0304／0169检验错选传播，0307检验外观恢复保护，0076作为缺少公开好框的边界；单纯多训PPO、调pulse频率或按GT删坏引用不足以回应全部问题。本次仅归档复核，未新建Stage-103或改冻结代码；确认集继续封存，最新完整AUC仍为Stage-100。 [Stage-102 结果](docs/experiments/06_sgla_uav_tracking/STAGE102_RESULTS_REVIEW.md)

### 0.76 2026-11-14 Stage-103 四候选核验完整轨迹实验（准备记录，结果见0.77）

新增`tools/stage103_candidate_verification.py`及`tools/run_stage103_verification.sh`，绑定Stage-102决策`2385b6d968cb5aa69ae89ad055a87b02a0591472eedcdb8a9e5fada4454fcf4b`。本轮独立改候选核验范围，不同时改几何状态提交，不把已失败pulse接入新完整方法。三种核验均按自身输出更新原普通几何历史及原资格外观记忆，SGLA／四候选生成／PPO／全部旧阈值冻结。

完整开发组28视频／40单位：native每视频一次28，PPO／pair／four_matched／four_pairwise各40，共188逻辑完整轨迹。pair为原PPO＋challenger；主要对照four_matched保留原pair使用域、原两查询有限性、原历史／首帧参照，仅扩大到合法有限其他候选。four_pairwise把原中心距离或Hann差条件逐候选应用，允许不同备选具备资格；用于分清候选范围与启用条件。分数为原描述子差点乘原参照，沿用switch_margin，平局优先原challenger，精确重复框去重。实际最终描述子写入原资格记忆；拒绝记忆仍接纳几何历史的旧路径本轮不变。

补充Stage-102全部64角色目标，现从归档重算恰64唯一单位／帧。以冻结pulse父框／过去框／外观记忆共同重放，每点只做四次搜索，三种核验共用四个描述子，GT仅评价；用于0304／555等好框能否进入并通过核验的机制说明，不把离线角色作在线触发。同卡必须复现来源候选／特征／裁剪／旧查询和pair选择，换卡记录漂移且仅条件于旧来源过去状态。

同型号＋UUID＋软件完全相同复用native／PPO／pair共108完整来源，只新增80条完整轨迹340,576搜索＋64点256搜索＝340,832；换L20／L40或UUID时在当前卡重测全部188完整对照，总710,199＋256＝710,455搜索；每次7块，无新训练或确认推理。新增ROI描述子来自原四次前向patch grids，无第五次网络搜索，但不声称CPU／内存、延迟或能耗零增加。旧原生四槽来源保留物理调用元数据，只取q0评价视图，未修改旧文件。

主要完整AUC比较two four modes分别对pair，使用两比较家族区间；沿用+0.5实用幅度和低IoU／最长失跟／可靠帧门槛，保护pair稳定单位及恢复，按具体单位／事件集合拒绝新的native／PPO稳定损伤或原已保留PPO恢复丢失。计数型失跟／未恢复按越少越好，本轮未继承旧helper方向错误。通过仅为PROMISING开发结果，确认集仍封存。

八表与decision全部回传：policy_summary、paired、video_summary、seed_video、scope、recovery、matched、matched_candidates；完整NPZ和64点四描述子另存。五项CPU测试通过，含隐藏候选、域控制、tie／NaN、过去引用、pair复现、自身反馈／未来GT隔离、混合队列两硬件模式实际调用预算、点查询、续跑／篡改和指标方向；网络与PPO选择在测试中合成／替代，正式成绩未知。Bash语法、CLI、LF及AST通过。旧代码未改，本机缺少服务器NPZ和GPU访问，正式实验未启动。

同步后`bash tools/run_stage103_verification.sh`，默认GPU0、原V1 checkpoint、`output/stage103/four_candidate_verification_v1`；L20／L40首次运行自动识别，续跑换型号或UUID须新`STAGE103_ROOT`，不能覆盖旧manifest。后台命令见协议，回传`stage103_results_*.txt`。 [Stage-103 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE103_FOUR_CANDIDATE_VERIFICATION.md)

### 0.77 2026-11-14 Stage-103 四候选核验结果（已完成）

同一L20／UUID，runtime差异为空，188逻辑完整轨迹、108参照复用、80新增，64固定点；新搜索340,832、块2,385,824，来源误差0，训练／确认推理0。状态`REVIEW_FOUR_CANDIDATE_VERIFICATION_DEVELOPMENT`。原生／PPO／pair／four_matched／four_pairwise宏AUC63.9605／65.0943／65.8348／60.9853／62.7318，低IoU13.20%／11.30%／10.66%／17.53%／14.65%。两新组11检查均只非平凡使用通过，不采用。

four_matched对pair−4.8495，名义95%[−10.1901,−0.3485]，两比较家族[−11.1046,+0.1811]，12胜16负；four_pairwise−3.1030，名义[−8.0678,+1.2988]、家族[−8.7959,+1.9643]，12胜16负。不能把前者名义负区间写成家族也全负。后者对前者+1.7465、区间[−0.1480,+4.1801]，较少退化不等于有效。

主要损伤视频宏pair／matched／pairwise：0304为72.4707／22.4707／33.1498，0169为62.4229／35.3160／35.3845，0091为66.7495／29.8965／29.8965，0307为54.8058／41.6808／41.5992，0176为67.2811／43.8648／62.2058。0349有+16.2785／+16.5804、0226 pairwise+23.5817，局部收益不足抵消损伤。matched新增pair稳定单位0304两seed、0169／2171、0091／2171（4单位3视频）；pairwise再添0003／2172（5单位4视频）。新增PPO稳定损伤两组皆有0091／2171、0304／2171，按具体单位判断，不能总数相抵。

PPO参考25恢复，pair保21、matched15、pairwise19；pair自身17恢复分别保17／12／15。两项都补回0226／2171／343，但matched新丢7：0085／2171／208，0176／2172的321／388／415／468，0226／2172／337，0307／2172／471；pairwise新丢3：0076／2172／1477、0085／2171／208、0307／2172／471。故21+1−7=15、21+1−3=19，恢复保护未过。

64个冻结pulse状态中56个有公开好框；允许好框参与比较的状态pair52→matched54／pairwise55，但最终好输出49→47／47，可访问好框却漏选3→7／8。matched当帧13胜45平6负、视频等权条件IoU差+0.000771，pairwise14胜43平7负、+0.008721；前者补好2丢好4，后者补3丢5。11损伤目标2胜9平／3胜8平，不抵消其他目标损伤，离线角色非随机在线队列。

**0304／2172／555：** 两新法选q3得到0.813839，pair q0为0。好框q1／q3历史相似度0.98983／0.99016，高于坏q0／q2的0.96893／0.96859，覆盖假设局部成立。**0169／2171／653：** pair和matched仅支持q3、用anchor，仍IoU0；pairwise用history并开放q0得到0.65109，涉及资格／参照变化，不能只归因四个数目。**0307／2172／471：** 好q2 IoU0.62223分数0.90486，坏q0 IoU0.44404分数0.91558；新两法选q0，仍没保好框，说明当前分数排序不稳定。**0076／1477：** 四框IoU都0，三法都无法补出好框。

完整三法从首帧普通历史自身推进，无pulse；固定点却来自旧pulse过去，两种历史不同。完整新法与pair在0304两seed第10／4帧、0169第5、0091第4帧已首次输出分叉，不能把局部555／653修复当完整修复，也不能把首次差异直接叫首次错误。自身状态下选原两框之外候选15,528／15,277帧（36.47%／35.89%），是频繁处理。宏有好框但漏选略降1.1538%→1.1154%／1.0952%，但好框供给86.22%→79.40%／81.63%，不能据漏选比例下降称选择更好。

源码既有switch_margin实际为1e-12，属数值级条件，未在本轮标定为身份可信度门槛；描述子为原patch embedding ROI与历史均值。后续可检验表示／参照／评分与状态提交，不能只继续扩范围或事后搜阈值。新NPZ已保存pair_selected、四描述子、分数，可在新增完整轨迹每个同状态下核对新选择相对原pair的直接效果、实际写入和几何提交，不需重复GPU。

本机核对decision及八CSV字节哈希，从188seed记录重算15policy／140video／210paired及区间一致，188scope预算与新增80一致；108参照共同指标与Stage-100一致，64点候选／原pair／资格／IoU／matched相同参照分数一致。具体稳定与恢复集合已核对；原生12四槽＋16单槽、旧head69,622及q0视图29,047区分保留。manifest／NPZ留服务器，未本地重放GPU、全描述子或逐帧后继；服务器标志不冒充本机再验。归档SHA256`239669bd95cf1eac19311541a1c881a5597b7efeb3468aa37176dfab27b37d79`；decision`81222ffbb0e81639ab6db572089694f2382e5a0a77d16c9d45423cf808bd6c97`；plan`1a28ae6b64fb22aae0ca3ee8593326f6abed6665e2be3f87cadeebdfa089e4f7`。

本次只归档复核，未新建Stage-104或改冻结代码。保留pair作当前对照，优先审计新缓存的有害替换，再独立设计候选可靠性及普通搜索状态保护；线上信号不读GT，兼顾真实运动与恢复。没有新增方法通过完整能力保护，确认集继续封存。 [Stage-103 结果](docs/experiments/06_sgla_uav_tracking/STAGE103_RESULTS_REVIEW.md)

### 0.78 2026-11-14 Stage-104 错误替换与普通搜索状态提交审计（准备记录，结果见0.79）

新增`tools/stage104_replacement_audit.py`、`tools/run_stage104_replacement_audit.sh`与五项CPU测试。绑定Stage-103决策`81222ffbb0e81639ab6db572089694f2382e5a0a77d16c9d45423cf808bd6c97`；覆盖全部28组／40单位，两项新法80条85,144状态，加pair40条42,572状态，共120条127,716状态。原生／PPO和全部方法188条逻辑轨迹的policy、paired、video、seed、recovery重算，scope的单位／帧数／首次输出分叉及额外选择次数亦核对。

同状态pair使用每条新法的实际过去、四查询及记忆，不能与独立完整pair同帧混为一谈。沿原0.5／0.2阈值描述好框丢失／补回、低IoU引入／消除；槽位替换和真实框变化分开。所有合法帧统计按种子再视频等权；条件均值报告支持数，不把平均当前IoU差当新政策AUC。

逐帧重演实际外观队列，排除非有限查询、拒绝当前／未来引用；对替换帧的每条实际有效参照，删除该条后按原规则重选参照和候选，允许补入较早引用或退回anchor，只报告固定查询的当帧结果，不延展未来。GT只在决定之后标记引用质量，不能用来在线删坏引用。分数差、首帧分数差、逐历史一致性只做描述，不搜阈值。

普通几何历史仍接纳每帧最终输出。固定t−1前历史，分别提交实际框与同状态pair框，计算t+1父框／取整裁剪差；实际下一父框必须与缓存一致。替代路径没有新网络输出或未来IoU，不能据此唯一确定长期失跟原因。记录拒绝外观写入后仍低IoU几何提交的路径。

统一纳入各新轨迹首次替换、首次好框丢失、首次低IoU引入、首败、与完整pair首次分叉，及全部新增pair稳定失败／新增丢失PPO和pair恢复。目标窗口前32后17帧；摘录目标邻帧、最近好输出／有害替换及实际过去引用，避免只看0304／0169。恢复仍用原+15闭区间和三帧好输出，按事件集合保护。

新增训练／图像读取／GPU搜索／块／确认推理均0，不需GPU或checkpoint，当前L20／L40均不触发跨卡重测，因为只有CPU审计。所有源文件、代码和九表SHA绑定；完成状态`COMPLETE_CACHED_REPLACEMENT_STATE_AUDIT`，无方法采用门槛。结果包decision＋summary、unit_summary、cases、recovery、signal_summary、reference_summary、chain_excerpt七表；全量state_frames及reference_drops留服务器。

本机五项CPU测试通过：有害替换／补回、有效低质量参照、拒绝写入后的几何反馈、GT隔离、非有限与未来引用、两级权重、恢复逐事件、失跟start索引、混合队列流水线、六张来源表核对、幂等和篡改拒绝。合成缓存／替代网络和PPO不代替正式来源结论。本机没有正式NPZ，尚未运行服务器审计，确认集继续封存，最近完整AUC仍是Stage-103。

同步后执行`bash tools/run_stage104_replacement_audit.sh`，默认`output/stage104/cached_replacement_state_audit_v1`；来源沿用Stage-76／89／93／103，可用对应`STAGE*_ROOT`覆盖。后台nohup命令见协议；完成回传`stage104_results_*.txt`，`--pack-only`只校验和打包。 [Stage-104 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE104_CACHED_REPLACEMENT_STATE_AUDIT.md)

### 0.79 2026-11-14 Stage-104 错误替换与普通搜索状态提交结果（已完成）

状态`COMPLETE_CACHED_REPLACEMENT_STATE_AUDIT`，28开发组／40单位，pair与两新法120条127,716状态；新增图像／搜索／块／训练／确认推理0，无新完整AUC。来源L20 runtime与Stage-103完全一致。回传decision＋七表齐：summary9、unit_summary120、cases289、recovery168、signal_summary24、reference_summary18、chain_excerpt1,537；全量state_frames127,716和reference_drops90,581留服务器。

同状态pair用新法自己的过去框／候选／查询／记忆，不能与独立完整pair同帧混淆。matched／pairwise替换15,528／15,277帧，同状态丢好85／84、补好58／70，低IoU引入55／48、消除37／35；视频等权平均当前IoU差−0.00044145／−0.00043530。替换视频宏率36.56%／36.12%，原始帧池率36.47%／35.89%，不可混写；全部28视频有替换。丢好13视频／15或17单位，不能当独立事件或据小的平均当帧差唯一解释完整AUC下降。

pair／matched／pairwise低IoU帧6,053／9,650／7,150，其中写入外观4,950／6,837／5,063，拒绝外观却提交几何1,103／2,813／2,087；有效读取低IoU历史宏率10.68%／16.96%／14.54%，无公开好框13.78%／20.60%／18.37%。几何仍无条件接纳最终输出，拒绝外观写入没有保护搜索状态；这些共变不证明全部失败唯一由某条路径造成。

**0176／2172／four_matched第251帧：** 四框IoU0.913871／0.790112／0／0，PPO q1、pair q0、新法q2。q2对pair分数差+0.08233，但对最高其他候选差仅0.00206。历史246／242／239一致支持q2，最小差+0.05758；首帧也支持，差+0.04988。三个引用自身IoU0.26491／0.33640／0.22175，虽不是<0.2仍不良好，reads_low=false不能当可信。q2拒绝外观写入却提交几何；替代提交pair会改变下一帧全部四裁剪、最大父框中心差402.35像素，尚无替代未来网络输出。

**0304／2172／four_pairwise第1083帧：** 四框0.628492／0.358974／0.808458／0.687525，PPO好q2、pair好q3、新法坏q1。参照1081／1075／1072均好（0.64229／0.64870／0.62573），仍排错；平均分数差仅+0.000939、历史不一致、首帧却也支持错框。外观写入拒绝，替代几何会改变全部四裁剪、中心最大2.61像素、log尺寸差0.58814；1084最佳0.47222，1085又有好框，1095形成首败。不能说坏历史是唯一原因或1083一次改动必然修复。

**缺框与资格仍并存：** 0304／2171两新法555、0304／2172 matched555、0169两新法653四框全0；0091两新法94最佳仅0.01661；0304 pairwise1095最佳0.20708。同状态pair也无法给好输出。0304／2171第556帧却又有q2／q3好框0.66927／0.80782而漏选，不能把缺框扩大到永远。0003／2172 pairwise138有q0／q1好框0.69398／0.77644且分数最高，但掩码0|0|1|1排除，实际q2和pair q3均0，资格限制也是问题。

首个分叉常是两框都好：0304／2171第10、2172第4，0169第5两者IoU相同但框不同，0091第4新增选择当帧更好。次帧父框／裁剪已分叉；不能将首次差异直接当首次错误或后期失跟唯一原因。严重失败需分岔前共同状态的后继干预证据。

参照删除90,581条只在替换帧对实际有效引用进行，固定已有查询。删除低IoU参照的选框改变宏比例24.92%／22.63%，平均当前IoU差+0.001823／+0.001186，条件于16个视频；有补引用／退anchor且边重复相关，没有新未来轨迹，不建立GT清洗在线方法。丢好状态margin平均高于补好，仅为不同支持集描述；0176已反证简单margin高／历史一致／首帧赞成就可靠。

稳定单位损伤matched4单位3视频、pairwise5单位4视频，与Stage-103集合相同。PPO恢复pair21/25、matched15/25、pairwise19/25；pair自身17次恢复保17／12／15，新增丢失PPO7／3、pair5／2，逐事件复现，回传recovery与Stage-103原CSV字节完全相同。0307／2172第471帧两新法最佳0.20536、实际和同状态pair约0.15759，不能引用旧pulse固定点好q2=0.62223作为本轨迹可用候选。

本机核对decision、来源及runtime、七CSV字节哈希；从120单位表重算九宏汇总、每单位比例和分解；逐单位与Stage-103低IoU／供给／漏选一致，恢复字节一致；替换分类、稳定及恢复集合一致；1,537唯一摘录的候选／选择／分类／过去索引／裁剪掩码一致，289case目标与摘录一致。完整NPZ、两全表及manifest未回传，未独立重算全部查询／删除／父框或计划指纹，服务器标志与本机复核范围分开。

原件SHA256`34e18fc05478817eb45ba21c5104977faefa81f43661cd62334b7bb2c9ab0e55`；decision`f923f8d0138e1e77458ce8f54f1f1058518f321d1980fb0f66cfa72f0ae6e2da`；plan`98fa60250947a41d3ea9925fc92dcc11262b5b99da75466b44e2571c9f69a7fb`。本次仅归档复核，未新建Stage-105或改冻结代码；下一步优先匹配后继干预，分别检验候选依据／外观写入与普通几何提交，不再只调阈值、增加候选或一律保持旧中心。需要保住真实移动和恢复，确认集继续封存。 [Stage-104 结果](docs/experiments/06_sgla_uav_tracking/STAGE104_RESULTS_REVIEW.md)

### 0.80 2026-11-14 Stage-105 共同起点的几何与外观提交交叉干预（准备记录，结果见0.81）

新增`tools/stage105_commit_branches.py`、`tools/run_stage105_commit_branches.sh`与本机测试。绑定Stage-104决策`f923f8d0138e1e77458ce8f54f1f1058518f321d1980fb0f66cfa72f0ae6e2da`及其Stage-103来源。覆盖28开发组／40既定单位，four_matched与four_pairwise各40条来源，共80上下文；SGLA与PPO、候选生成、评分及阈值冻结。

四组keep／geometry_pair／memory_pair／both_pair，在共同t帧观察上都发布原四候选框b4；仅第一步分别把同状态pair框b2用于普通几何提交、外观写入候选及其资格，或同时用于两者。t+1开始恢复各自原四候选规则，最长32帧。拒绝写入仍可有几何干预；不强迫写入pair、不按GT删过去记忆、不一律保持旧中心。普通历史也派生PPO时序特征、外观使用条件及资格，因此几何因素是整个普通历史状态处理，不是纯裁剪中心归因。

统一起点包括：首次替换／首次完整pair分叉／首次丢好／首次低IoU引入；首败和新增pair稳定失败之前最近替换及前一帧；新增丢失PPO／pair恢复前一帧及较早最后丢好（没有则最近替换）；完整来源首次补好作为收益控制。同上下文同帧去重保留全部角色，角色由已消耗GT和既有结果定义，只作诊断，不是线上触发器。服务器先审核80来源并冻结起点、角色、末帧及精确预算，再开始新GPU推理。

一个起点重算一次四搜索共同观察，四组共享第一帧，后续每组真实四搜索；长度L新增4+16(L−1)，完整32帧500搜索／3,500块，逻辑缓存16L调用与实际新增分开记。每起点一个point＋四branch五个回执缓存；中断复用已提交缓存。来源过去框／描述子／anchor固定；同来源型号、UUID与软件时keep整个窗口必须复现，换卡时四组全部同卡重测并记录源误差，不能把新卡分支直接与旧卡分支作同卡比较。

第一帧发布输出一致，主要看后31帧均值IoU、低IoU、好候选供给、漏选比例与末帧IoU。比较几何−keep、记忆−keep、共同−keep、交互及两个顺序平均；唯一去重起点总体和角色分层分别按起点→单位→视频汇总，窗口重叠不作独立样本。恢复仍按三帧好输出与参考起点+15闭区间；未覆盖到参考+17且未找到恢复时右截断，不当作恢复失败。较早丢好到恢复很远时不额外延长窗口。

状态`COMPLETE_MATCHED_GEOMETRY_MEMORY_COMMIT_DIAGNOSTIC`，无新方法采用门槛，无训练／完整视频重测／确认推理。回传decision＋summary、paired、scope、branches、recovery、focus六表；全量branch_frames留服务器，七表与全部NPZ均SHA绑定。七项CPU测试通过，包括非平凡真实搜索后继分岔、两硬件模式实测调用预算、续跑及篡改拒绝。实测尚待服务器运行，本机替代网络／PPO测试不代替正式结果；最近完整AUC仍是Stage-103，确认集继续封存。

同步后`bash tools/run_stage105_commit_branches.sh`，默认GPU0、原V1 checkpoint、`output/stage105/matched_geometry_memory_commit_v1`，读取Stage-76／89／93／103／104，可用对应`STAGE*_ROOT`覆盖。首次自动识别L20／L40；续跑换型号或UUID须新`STAGE105_ROOT`，软件／权重变化仍拒绝。后台nohup命令见协议，完成回传`stage105_results_*.txt`；`--pack-only`只校验与打包。 [Stage-105 固定协议](docs/experiments/06_sgla_uav_tracking/STAGE105_MATCHED_STATE_COMMIT.md)

### 0.81 2026-11-14 Stage-105 几何与外观提交交叉干预结果（已完成）

同一L20／UUID、runtime差异为空，keep整段复现最大框误差0。28开发视频／40既定单位、两个四候选来源80上下文；matched131起点、pairwise128起点，共259／1,036短分支、32,648逐帧记录、1,295缓存。新搜索127,484、块892,388，逻辑缓存130,592调用因首帧共享与新推理数不同。训练／完整新轨迹／确认推理0。状态`COMPLETE_MATCHED_GEOMETRY_MEMORY_COMMIT_DIAGNOSTIC`，无采用门槛。

四组同过去、同当帧发布输出，只在起点改普通历史／外观写入，之后恢复原规则。总体keep／geometry／memory／both的后继平均IoU：matched0.562567／0.572108／0.563375／0.581319；pairwise0.561443／0.566321／0.562231／0.573629。both对keep+0.018753／+0.012186（约1.88／1.22个IoU百分点，非AUC），19胜9负／17胜11负，中位+0.002080／+0.001501。几何差+0.009541／+0.004878，记忆差仅+0.000808／+0.000788，交互+0.008404／+0.006520。

both低IoU宏比例26.47%→24.36%、26.09%→24.87%；好候选供给67.92%→70.38%、67.67%→69.96%。单改记忆漏选比例反增约0.452／0.398个百分点。普通历史还派生PPO时序特征和资格，且之后两条路径继续耦合，识别的是首步处理的总后继作用，不是纯裁剪中心归因。

队列边界：matched历史／追加的both均差+0.013796／+0.022470；pairwise为+0.001307／+0.020345。pairwise历史组几何−0.003119（5胜7负），低IoU+0.952个百分点；both虽小正但低IoU+0.307个百分点、末帧−0.009441。首次丢好角色几何+0.021964／+0.044796；首次低IoU引入却−0.024909／−0.020242；首次补好pairwise几何−0.036318、both−0.032588。首次分叉两来源相同，几何−0.002211、both−0.000490。不能一律回退pair或遇分叉就阻断。

**0176／2172／matched，251～282：** keep后31帧均值0、好候选1/31、低IoU31/31；几何与both均值0.893676、好候选31/31、低IoU0、末帧0.956522。起点四组发布IoU0相同；当帧两种外观写入都拒绝，记忆因素没有实际改变，memory组保持0。这为拒绝外观写入后仍需处理普通历史提交提供直接后继证据。窗口止282，321等参考恢复在外，不能声称补回原参考事件或完整视频。

**0304／2172／matched，552～583：** keep均值0.054210，几何／both0.787438，低IoU29/31→0，好候选3/31→31/31，末帧0.742433；memory保持原结果。当帧b4与pair IoU同约0.80645，仍有巨大后继分化。**0304／2172／pairwise，1083～1114：** keep0.124987、几何／both0.725163、memory0.107588；几何低IoU20/31→0，好候选1/31→31/31，末帧0.851970。四组当帧仍发布0.358974，未混入当帧换框直接收益。

**0091／2171，87～118：** 两来源keep0.114349、memory0.114573；geometry在matched0.147711／pairwise0.379144；both均0.676507、供给31/31、低IoU0、末帧0.691748，显示两条提交路径有条件互作。**未解：** 0304／2171两来源553起点虽554已有输出／裁剪分叉，四组均值仍0.026253、末帧0；554起点是无效处理。0169两来源651起点改变普通历史却无记录的后继输出／裁剪分叉，均值0.018350；652无效处理、均值0。

**严重反例：0304／2172／pairwise，556～587首次补好控制：** 起点b4 IoU0.669269，pair0；keep后继均值0.798406、供给31/31、低IoU0，几何／both均值0、31帧无好候选且低IoU，memory未实际改变保持原结果。**0176／2172第165帧：** b4 IoU0.162910、pair0.348106（当帧更好），keep后继0.441850，geometry／both在matched0.072675、pairwise0.055610，低IoU27/31、29/31。当前pair更好也不能保证后继更好。both的pairwise视频宏损伤0176−0.052607、0348−0.026772、0304−0.022480。

恢复136行=34起点／来源／参考上下文×4处理，17个事件身份（matched12、pairwise5，PPO与pair参照可对应相邻的同一物理恢复）。每处理23个完整观察、11个右截断上下文，四组均0匹配；17个身份均有至少一个完整窗口也均未补回。right_censored不当失败；本轮只追踪已丢失事件，未完整检验保住已有恢复能力。

scope几何有效111/131、113/128；记忆有效90/131、85/128，均属于几何也可改变的起点。总体与角色按去重起点→单位→视频汇总，两个来源仍是同28视频，窗口重叠不当独立样本。事后描述性both最小留一均差+0.012417／+0.005607，但不构成新置信或采用门槛。

本机核对decision／来源／runtime及六CSV原始字节SHA；259点／80上下文／28视频、预算、同当帧输出、无效因素指标恒等与首次分叉时序一致；由1,036 branches与scope重算288summary（重建CSV哈希完全一致）和432paired（最大差5.55e-17、均在1e-12，未声称字节一致）；2,072首末记录的候选／选择／过去引用／恢复原提交与branch指标一致，240个keep首末记录与Stage-104摘录一致；恢复边界／截断一致。manifest、完整NPZ和全branch_frames留服务器，未本地重放GPU或全帧actor／查询，也未独立重算未回传计划指纹。

原件SHA256`e331209d6811f22b96908d60619a4d9455a379b217f5be700dddfa9d16198dbc`；decision`a01a169a4cb8d0e9211f2252cddfe7ff739e9a1758ed8f53789ecdba91c57363`；plan`ef8de45bd9b328e62d60448c5b944cacbda91c1ea569c705b2d81a0f771d64ef`。已有部分状态下真实后继改善的机制证据；下一步可设计可部署的提交条件，独立于最终框发布并维护几何／PPO时序输入／外观写入一致性，用fit确定后做完整开发同预算对照。不能使用GT角色作线上触发。本次仅归档，未新建Stage-106或改冻结代码；最近完整AUC仍是Stage-103，确认集继续封存，没有新增方案通过完整能力保护。 [Stage-105 结果](docs/experiments/06_sgla_uav_tracking/STAGE105_RESULTS_REVIEW.md)

### 0.82 2026-11-14 Stage-106 源组留出的提交门控可学习性（准备记录，结果见0.83）

用户要求继续下一轮。已新增`tools/stage106_commit_gate.py`、`tools/run_stage106_commit_gate.sh`与本机测试，协议`STAGE106_CROSS_FITTED_COMMIT_GATE.md`。绑定Stage-105决策`a01a169a4cb8d0e9211f2252cddfe7ff739e9a1758ed8f53789ecdba91c57363`及原Stage-103来源。目标是用线上可见证据辨别一次both_pair提交何时有益，而不是一律回退pair。SGLA与PPO、生成器、核验规则及其资格保持冻结；本轮新增CPU门控拟合，不可写成零训练。

259唯一锚点从既有keep／both_pair分支重算标签。future为后31帧均值IoU差；myopic为当帧pair提交框与原四候选框的IoU差，是提交框质量代理，四组当帧发布框其实相同。两头使用完全相同的有限且有效样本、49维特征、训练权重与正则。特征只取当前两项选择的中心／尺寸／响应／外观、过去自身几何及已接纳描述子、冻结PPO概率与value、全候选离散度和来源模式；GT／角色／种子／序列身份／未来不进入输入。

28源组逐一留出，两个来源模式合并fit，每头最多28模型，56个模型文件。按组→种子→模式→锚点等权，训练组加权均值／标准差归一化，ridge=1.0、权重和1、截距不正则、预测收益>0才提交，无系数或阈值搜索。无实际提交变化、必要证据无效或缺拟合支持时keep。被评估组全部片段／种子／模式的标签与归一化数据排除；旧SGLA／PPO可能与其重叠，整个项目仍是已消耗开发证据，不是外部确认。

对照keep／always_pair／myopic／future／相同提交率均匀随机的解析期望／离线未来oracle；后两项只是诊断，不能当部署策略。报告相对keep的后继效用、提交率、有害提交、遗漏正效用和遗憾。按上下文锚点→模式→种子→源组汇总，再分历史／追加与两来源；重叠窗口不作独立置信样本。它评价首步干预后恢复原规则的收益，不是重复新门控的整视频结果。

另重放80条原来源、85,144个缓存状态，只记录门控合法率、提交率、训练min/max外比例与最大标准化值分布，门控不回写普通历史或外观。曝光NPZ中当前GT仅在预测后附描述性标签；范围外比例不作额外门槛。无新图像／搜索／GPU／完整轨迹／确认推理，最近完整AUC仍是Stage-103。状态固定`COMPLETE_CROSS_FITTED_COMMIT_GATE_DIAGNOSTIC`，没有采用门槛；若有积极信号再预声明完整闭环，不能直接称精度或保护已通过。

模型／样本／曝光缓存有指纹、哈希及回执，续跑复用相同fit输入的封存模型，不要求跨CPU重新求解逐位一致；来源或代码变化拒绝混用。回传decision＋七表（samples／fit_support／matched_summary／paired／sample_decisions／exposure_units／exposure_summary），56模型与样本特征、80曝光NPZ留服务器。本机4项CPU测试已通过，含非平凡特征／正负目标／留组隔离／同率对照、完整缓存流程、GT隔离、续跑与篡改拒绝；正式来源NPZ仍只在服务器，未正式拟合或推理。

同步后`bash tools/run_stage106_commit_gate.sh`，默认`output/stage106/cross_fitted_commit_gate_v1`，不需GPU或checkpoint参数，可用`STAGE76_ROOT`／89／93／103／105／106覆盖来源路径。后台：`mkdir -p output/stage106`后`nohup bash tools/run_stage106_commit_gate.sh > output/stage106/run.log 2>&1 < /dev/null &`；`--pack-only`只验证与打包，不自动提交或推送。接续先看结果中的future对myopic／always／同率控制，以及旧轨迹曝光外推程度，再决定是否值得完整自身闭环。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE106_CROSS_FITTED_COMMIT_GATE.md)

### 0.83 2026-11-15 Stage-106 提交门控结果（已完成，新增选择价值未建立）

已收到并逐字节归档`stage106_results_20261114T161430Z_716160.txt`，590,164字节。decision＋七CSV齐全；56个有支持CPU模型、259唯一起点、80原来源上下文／85,144缓存状态。224起点有实际提交变化、35无变化；259样本两目标均有效。27组fit／整源组留一，fit201～222条／76～78上下文；SGLA与PPO训练0、新图像／搜索／完整轨迹／确认推理0。状态`COMPLETE_CROSS_FITTED_COMMIT_GATE_DIAGNOSTIC`，不是方法通过。

相对keep的后继均值IoU效用：always+0.015469、myopic+0.014028、future+0.012838、同率期望+0.013800、离线oracle+0.021464。不是AUC。future对keep+0.012838，20胜2平6负，最小留一+0.006283；但对always−0.002631（4／12／12）、对myopic−0.001190（9／9／10）、对同率−0.000962（7／15／6），尚无超越必要控制的新增选择价值，无重叠窗口独立显著性结论。

宏提交率always90.71%、myopic54.96%、future与同率72.48%；按全部锚点的有害提交宏比例31.88%／18.49%／23.65%／24.13%，不是已提交动作中的错误率。实际224合法起点中后继正117／负84／零23；future提交182次，接受96/117正例（82.05%）及69/84负例（82.14%），符号区分很弱；myopic提交146次，正85、负50、零11。

历史12组future效用+0.001494，低于myopic+0.009600和同率+0.005250；追加16组future+0.021346，对myopic+0.003997、对同率+0.001133，但对always仍−0.000062。matched总体future+0.017625，对myopic+0.003374，但低于always且对同率仅+0.000352、最小留一负；pairwise总体future+0.008050，对myopic−0.005754，对always−0.004135，对同率−0.002276。历史pairwise future更为−0.008098。不能事后挑来源或队列采用，追加组仍为已消耗开发证据。

**0304／2172／pairwise556：** 实测both−keep后继−0.798406，future预测+0.042914而放行，myopic预测−0.076667而拒绝。**同来源1083：** 实测+0.600176，future预测−0.018235而拒绝，myopic+0.149325而接纳。**0176／2172／matched251：** 实测最大救援+0.893676，future−0.202572、myopic−0.433130均拒绝；165的有害提交−0.369175／−0.386240两头却都接受。**正例仍有：** 0304／matched552的+0.733228、0091两来源87的+0.562158，两头都接纳。0348／2171／364，matched真实+0.234587被future接纳、myopic拒绝；pairwise真实−0.126191，future仍接纳、myopic拒绝。

0304是future对myopic最差组，组内差−0.070737；后验移除它总体差转+0.001386，但不允许删除后采用。0091贡献future对keep总体均值约+0.006779，四个非oracle处理在该组相同，不能归因future独有选择；移除后future对keep仍+0.006283。

原轨迹曝光仅观察旧自身历史：30,805合法状态，future／myopic记录27,352／20,175次提交；宏合法率36.34%、提交32.30%／24.48%。7,694合法状态至少一维超fit min/max，宏比例22.78%（历史26.91%、追加19.68%）；最大绝对标准化值宏均1.8891、单位p95宏均3.1352。两头fit支持一致，范围统计一致。原始计数与宏比例权重不同；范围外不是必然错误或新增触发阈值。门控没有回写，不能报告新完整AUC／恢复／稳定性通过。

本机核对decision自身指纹、全部七CSV原始字节SHA与行数259／56／54／36／1554／160／18；逐条用Stage-105 branches重算两个标签、focus核对b4／pair索引、scope核对变化。对1,554条决定复算触发／同率期望／效用／遗憾，从其重建54summary及36paired，从160单位重建18exposure_summary并还原整数计数；所有差≤1e-12、最大1.14e-13。56条fit名单排除、支持数、加权目标均值相符。正式manifest、模型与样本／曝光NPZ仍仅在服务器，本机未重拟实际系数／归一化、输入指纹或重放全曝光，勿扩写为已独立验证所有缓存／模型。

原件SHA`89576939edd30387ad9695b9ad6ae82318b63a3c4de012dc30ac8b9e37ab99cb`；decision`644e44f73492caf63b5a8dcecb475b3728f3415b13d01eb9651a482e7d8b06d6`；plan`0265964f07c3e8f088a3aa978d675210f8ecf2f9965d16b55fa026d9c95f8e91`，未本地独立重算plan。

下一步优先审计现有fit支持与大错判，区分训练内也难分和留组后失效，再确定补充按线上条件抽样的状态或改状态表示／目标。尚不能唯一归因线性容量、GT质量或PPO，不直接扩训PPO或搜索旧阈值。本次仅归档复核，未新建Stage-107或改冻结代码，确认集封存；最近完整AUC仍为Stage-103，没有新增方法采用。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE106_RESULTS_REVIEW.md)

### 0.84 2026-11-15 Stage-107 固定门控训练支持与大错判审计（准备记录，结果见0.85）

用户要求继续下一步，已新增`tools/stage107_gate_support_audit.py`、`tools/run_stage107_gate_support.sh`、`tools/test_stage107_gate_support.py`及`STAGE107_FIXED_GATE_SUPPORT_AUDIT.md`。绑定Stage-106决策`644e44f73492caf63b5a8dcecb475b3728f3415b13d01eb9651a482e7d8b06d6`。只读取既有56模型、259样本及80份／85,144状态曝光，区分训练内不足与留组后失效，不重训门控／SGLA／PPO，不新增图像／GPU／完整轨迹／确认推理。

按原共同合法支持核对56模型的fit输入指纹、源组排除、加权归一化／范围／截距，验证固定beta正规方程（相对残差≤1e-9、浮点核对atol1e-10／rtol1e-9）。不优化或写新系数。分别评价fit_resubstitution与held_out，共112条，每头每阶段28模型宏平均4条；自身目标MSE／MAE／R²与固定训练截距基线，自身目标及真实future正负放行率，真实future效用／always／同率期望／oracle／遗憾。指标以共同有限且有实际提交变化样本为条件，不能直接与Stage-106全锚点均值比较。fit回代及28次重叠训练不是独立泛化。

全259样本双头518条状态支持；逐条复现Stage-106分数与动作。固定13重点例：0304／2172／pairwise556、1083；0176／2172／matched251、165及pairwise165；0304／2172／matched552；0091／2171两来源87；0348／2171两来源364；0003／2171／pairwise138；0176／2171两来源157。无需结果出来后再选择重点例。

逐状态分解49维z*beta与截距，检查和等于未加资格掩码的分数；无变化时实际score为0，不误作贡献全0。训练近邻只按fit尺度下49维RMS距离选，每训练源组至多一个锚点、再取5个不同源组，排名后才附标签；不引入查询GT／角色或留出组作近邻。它是支持描述，不是新策略。另计算固定矩阵A下`h_i=w_i*(1+z_i' A^-1 z)`，按训练源组分解`sum(h_i*y_i)`，允许负影响权重；固定尺度／权重／A，不是删除组重训或因果占比。两类分解都只解释冻结分数。

80曝光只读x／eligible，重算两头分数、动作、范围外与最大标准化值，核对旧NPZ／单位表。按49特征列示合法状态范围外计数与宏比例，分队列／来源441条；不使用离线当前IoU决定检查、不新增范围回退，逐特征计数可重叠。没有新完整AUC或保护结论，最近完整评价仍为Stage-103，确认集封存。

13CSV与decision全部SHA绑定。回传10表：model_scope56、prediction_summary112、prediction_macro4、state_support518、exposure_units160、exposure_features441、focus26、focus_features1274、focus_neighbors130、focus_sources702；完整neighbors2240／feature_attribution25382／source_contribution13986留服务器。不写新模型；完成后重跑只检查已有结果，中断重跑重建固定审计并要求已发表表一致。旧模型／来源／代码变化时拒绝混用。

本机4项CPU测试通过，含正规方程与留组隔离、精确分解和标签无关的不同源近邻、合成训练内不足与留出失效区分、完整缓存流程／曝光GT列隔离／续跑／篡改拒绝。正式模型／NPZ仍仅在服务器，本机没有正式审计结果；测试中的合成来源可拟合，实际Stage-107在阻止新拟合及网络调用的补丁下完成。状态固定`COMPLETE_FIXED_COMMIT_GATE_SUPPORT_AUDIT`，无新方法采用门槛。

同步后`bash tools/run_stage107_gate_support.sh`，默认读`output/stage106/cross_fitted_commit_gate_v1`，写`output/stage107/fixed_gate_support_audit_v1`，可用`STAGE106_ROOT`／`STAGE107_ROOT`覆盖；无需GPU、checkpoint或L20／L40参数。后台：`mkdir -p output/stage107`后`nohup bash tools/run_stage107_gate_support.sh > output/stage107/run.log 2>&1 < /dev/null &`。`--pack-only`只验证与打包，不自动Git提交或推送。结果回传后先比较future训练内／留组后区分，再看13例贡献与近邻及全曝光支持，据此决定补数据或改表示／目标，不能预设原因或搜索旧阈值。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE107_FIXED_GATE_SUPPORT_AUDIT.md)

### 0.85 2026-11-15 Stage-107 固定门控训练支持与错判结果（已完成）

已收到并逐字节归档`stage107_results_20261114T234540Z_840517.txt`，697,870字节，decision＋10回传CSV齐全；13表全有SHA，另外3长表按协议留服务器。固定56模型、259样本／518双头状态，80来源／85,144旧状态；新模型／训练／图像／搜索／完整轨迹／确认推理均0。状态`COMPLETE_FIXED_COMMIT_GATE_SUPPORT_AUDIT`，无采用门槛。服务器正规方程、归一化、源组排除、原分数／动作、分解和曝光检查通过；最大正规残差5.55e-17，不是新学习效果证明。

**训练内与留组损失并存。** future自身目标MSE训练0.011554对截距0.013462（改善14.18%），留组0.014869对0.013694（恶化8.58%）；myopic训练0.011605对0.027858（改善58.34%），留组0.027340对0.028086（改善2.66%）。两头目标不同，不能直接比MSE绝对值。按真实future正／负标签的条件放行：future训练84.93%／71.05%，留组82.82%／78.39%；myopic训练71.24%／57.22%、留组66.08%／55.86%。当前future训练内也仍接纳多数损伤，但并非完全没学到信号。

条件后继效用future训练+0.022826、留组+0.016713；always均+0.020175、同率+0.018956／+0.018298、oracle均+0.027545。myopic训练+0.018266、留组+0.018579。上述以共同合法有变化样本为条件，排除了无变化锚点，不能直接与Stage-106全锚点均值比较。future在训练支持上对always+0.002651，留组后−0.003462；不能仅称训练内未拟合或只称泛化，当前两者皆有不足。

训练future R²28模型均正、宏均0.1420（范围0.1257～0.1951），myopic宏均0.5817；留组MSE优于固定训练截距的模型future9/28、myopic15/28。留组R²两头仅22组目标方差非零，均值−75.62／−808.25、中位−1.967／−0.313、正5/22／8/22；6空组不入R²。0126仅4有效锚点，myopic MSE0.002701却R²−17749，反映小方差放大，不是AUC下降几百倍或计算错误；28模型字段不代表每指标28项支持。

**互为最近邻的反例：** F01=0304／2172／pairwise556，真实−0.798406、future+0.042914；其最近训练锚点是F03=0176／2172／matched251，真实+0.893676，RMS1.64934。反向留出0176，F03最近训练锚点是0304／556，RMS1.63280、future预测−0.202572。不同fit尺度所以距离不对称。F01固定future源组贡献中0176为+0.088877；F03中0304为−0.168168。支持现有度量／线性表示与后继方向混淆的线索，不是删除源组重训效应或物理因果归因。

F01没有任何逐列越界，但仍有害；F03有5维越界、max|z|9.6067，主要为横向位移／移动／速度、匀速偏差及方向。F03 future中过去横速贡献−0.090575、pair−four横移−0.043380、four横移−0.041350，正截距未抵消。F04／F05的0176第165帧无越界且近邻future均值约−0.003648，仍接受真实−0.369175／−0.386240，不能把所有错判归因范围外。

F02=0304／pairwise1083，真实+0.600176，future−0.018235；pair anchor与PPO概率差越界，主要负项含pair anchor−0.015297、PPO熵−0.011257及four anchor−0.009785，虽近邻future均值+0.023825仍拒绝。myopic+0.149325接受，主要正项为log宽比等。F06=0304／matched552的+0.733228、0091两来源87的+0.562158仍被两头接纳。F11=0003／pairwise138真实+0.301069，future−0.014620拒绝，全部维度在范围内。

**模式×状态关系线索：** F09／F10=0348／2171／364两来源，49维只有mode_pairwise不同，其余48维完全相同；真实future+0.234587→−0.126191，差−0.360777。future只能给固定线性模式偏移−0.003512，两项都接受；myopic偏移+0.001494，两项都拒绝。0091两来源87也只有模式不同，真实future却相等。支持预声明检验模式／状态交互，但不是加入交互必然泛化的证明。

全曝光与106计数相同：30,805合法、两头27,352／20,175提交记录、7,694至少一维越界。宏逐特征越界率最高PPO value7.30%（2501原始状态）、pair参照4.25%（1057）、log尺寸离散2.45%（775）、pair anchor2.13%（1003）、four anchor1.86%（1233）。多列可同帧越界，宏率与原始数权重不同。严格min/max会标记极小越界，不能凭排名删value或加新阈值。

本机核对原件／decision指纹、10表原始字节SHA及行数56／112／4／518／160／441／26／1274／130／702；56模型scope对106输入及模型指纹相符，518状态对106目标／预测／动作相符，56held_out指标用106标签与固定分数重算，112→4宏表重算。26重点例49维分解／范围、130近邻排除／身份／标签／均值／两头排名、702源贡献和式均相符；从重点字段完整重建5源组双头10模型体指纹，与106完全一致，并重算有完整特征的部分近邻距离。160曝光单位及441分层加和／宏比例与106相符，最大算术差2.28e-13。原始全部X／80曝光NPZ及其余46模型体仍在服务器，本机未独立重放全部正规方程、训练内预测、全近邻排名或逐帧曝光，也未独立重算plan。

原件SHA`b6d5351d6fff581e70947bd2bcf1e0ae0351b948773fcf1b97d82dc43c991638`；decision`7782ea245f6f17fe2b0bd3575edcadcee3da12a23a9bafbb2987d24a465cd0ba`；plan`2922d514078c44271d0d6e476122ea651007efac0653174e315930bfa6e4a81c`。

接续可预声明检验一种状态关系／模式交互表示，对照原49维，在相同样本、源组留出、正则预算及myopic／always／同率控制下评估；再判断是否需要补充按线上条件抽样的代表性状态。不能只修13例、删value、选模式或后验调阈值，也没有理由直接扩大PPO训练。本次只归档复核，未新建Stage-108或改冻结代码；确认集封存，最近完整评价仍为Stage-103，没有新增完整AUC或方法采用。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE107_RESULTS_REVIEW.md)

### 0.86 2026-11-15 Stage-108 状态关系×模式交互门控（准备记录，结果见0.87）

用户要求继续下一轮，已新增`tools/stage108_relational_commit_gate.py`、`tools/run_stage108_relational_gate.sh`、`tools/test_stage108_relational_gate.py`及`STAGE108_RELATIONAL_MODE_COMMIT_GATE.md`。绑定Stage-107决策`7782ea245f6f17fe2b0bd3575edcadcee3da12a23a9bafbb2987d24a465cd0ba`与原Stage-106。检验当前证据的关系与模式交互是否新增选择价值，不改SGLA／PPO或原核验与资格。

四组2×2表示：baseline49维；relation65维；mode97维；relation_mode129维。原49维全部保留。增加16维关系：中心冲突C=log1p两框中心距离、尺寸冲突S=log1p log宽高差范数、相对过去匀速误差的tanh偏好M、相对历史中位log尺寸误差的tanh偏好G；C×anchor／参照／Hann／PPO差4项、S／M／G各×anchor／参照差6项、两写入资格差×anchor／参照差2项。全从已有当前49维导出，不读GT／角色／ID／未来。

模式交互对非模式stem列做`(2m−1)*fit-standardized(stem_j)`，只用fit组中心与尺度，常量列不重复成模式偏置；再对完整拼接向量作fit-only最终标准化。每留出组排除全部片段／种子／模式；同样259唯一样本、224共同合法有效支持、相同权重／目标／ridge=1和收益>0，不搜超参数。新维数改变容量和有效正则，不能宣称完全相同计算或纯相同容量。

原56个baseline门控原样引用与复现；三个新表示×28源组×future／myopic=168个新CPU门控。正规方程与映射核对，保存输入指纹、两层fit归一化及系数；同fit输入续跑复用已封存模型，不重复求解beta。future和myopic目标仍分别为真实一次both−keep后继均值差、当帧提交框IoU差。所有结果为已消耗开发证据，旧SGLA／PPO可能与留组重叠，确认集封存。

15政策：keep／always／离线oracle，加四表示双监督8个门控、每表示future的同上下文同率均匀选择解析期望4个。主比较固定`relation_mode__future - baseline__future`，必要比较combined对relation-only／mode-only／同表示myopic／always／自己的同率／keep；另给两因素交互与顺序平均。不能只看主均值或13例。全259锚点层次宏汇总，与合法支持条件下的训练指标分母区分；不对重叠窗口给独立置信。

预期224 model_scope、448 training_summary、16 training_macro（各指标另列有效模型数）、135 policy_summary、90 paired、2072 state_predictions、3885 sample_decisions、104 focus、640 exposure_units、72 exposure_summary。13重点状态沿Stage-107不变。80旧来源／85,144状态仅曝光8门控，不回写、不新增GPU或完整轨迹；baseline逐条复现。168模型JSON和80新预测缓存共248产物SHA绑定，10表全部cat回传。新增CPU拟合，不是零训练；SGLA／PPO更新及新图像／搜索／完整轨迹／确认推理为0。状态`COMPLETE_RELATIONAL_COMMIT_GATE_DEVELOPMENT`，没有采用门槛。

本机4项CPU测试通过：四表示维数／固定公式／新增关系镜像性质／常量中心化、fit-only映射与源组／标签隔离、合成模式依赖目标可学习及正规方程、掩码与同率控制／两因素算术、完整缓存流程／旧控制不重拟合与精确重放／GT列隔离／续跑不重复fit／篡改拒绝。正式NPZ只在服务器，本机没有本轮正式结果；合成测试不替代实际验证。最近完整成绩仍是Stage-103。

同步后`bash tools/run_stage108_relational_gate.sh`，默认读Stage-106／107，写`output/stage108/relational_mode_commit_gate_v1`，用`STAGE106_ROOT`／`STAGE107_ROOT`／`STAGE108_ROOT`覆盖路径。无需GPU、checkpoint或L20／L40参数。后台：`mkdir -p output/stage108`后`nohup bash tools/run_stage108_relational_gate.sh > output/stage108/run.log 2>&1 < /dev/null &`；`--pack-only`只验证打包，不自动提交推送。结果后先看留组combined对原表示及简单控制，再看两因素消融、正负例保留与全曝光；即使积极仍须完整自身轨迹和封存源确认，不能改成只挑过关来源／种子／阈值。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE108_RELATIONAL_MODE_COMMIT_GATE.md)

### 0.87 2026-11-15 Stage-108 关系×模式交互门控结果（已完成，局部改善未转成稳定优势）

已收到并逐字节归档`stage108_results_20261115T003253Z_857451.txt`，1,366,374字节；decision＋10表齐全。28开发源组、259起点／224合法有效支持；baseline56模型复用，65／97／129维三表示新168个CPU模型，总224。新SGLA／PPO更新、图像／搜索／完整轨迹／确认推理均0；旧80来源／85,144状态仅曝光。状态`COMPLETE_RELATIONAL_COMMIT_GATE_DEVELOPMENT`，无采用门槛。

全锚点后继效用future：baseline+0.012838、relation+0.012134、mode+0.012645、combined+0.013909；对应myopic+0.014028／+0.013703／+0.013856／+0.013100。combined自己的同率期望+0.013203、always+0.015469、oracle+0.021464。均为最多31后继帧IoU差，不是AUC。主combined−baseline +0.001071（约0.107个IoU百分点），中位0、6胜15平7负、最小留一−0.000368。对relation+0.001775、mode+0.001264、同表示myopic+0.000809、always−0.001560、自己同率+0.000706；对同率10／12／6、最小留一+0.000209，但其他必要控制仍不够，旧baseline myopic也略高于combined future。

两因素交互+0.001968（6／19／3，最小留一+0.000689）；关系／模式顺序平均+0.000280／+0.000791，最小留一均负。单因素未提高整体future，不能把交互写唯一因果分摊。baseline／combined实际提交182／168次，宏提交72.48%／65.81%，全锚点有害提交宏23.65%／21.12%，combined同率为21.27%。合法样本中提交正／负／零从96／69／17变90／65／13，避损与漏益并存。

主差历史12组+0.006169，追加16组−0.002752；matched−0.000399，pairwise+0.002542。历史pairwise从−0.008098→+0.002176，但追加pairwise+0.020161→+0.016904，低于自己的同率+0.017974。主比较最大正组0304+0.039920／0176+0.036498，最大负组0349−0.027040／0348−0.018167／0237−0.006200；去掉0304均差−0.000368、去掉0176−0.000241，两者都去掉−0.001786。不得后验挑来源或删除负例。0091旧大收益保留，但四future表示该组相同。

**局部正进展：** F01=0304／2172／pairwise556，实测both−keep−0.798406；mode-only／combined改拒绝，分数−0.033721／−0.042268。F03=0176／2172／matched251，实测+0.893676，只有combined接纳（+0.002947）。F05=0176／pairwise165，实测−0.386240，combined拒绝（−0.000738）。F06=0304／matched552 +0.733228和0091两来源87 +0.562158仍被各future接受。这是旧起点上选择已有真实短分支的改变，不是完整视频已修复。

**仍未解决：** F02=0304／pairwise1083 +0.600176，四future都拒绝，combined−0.036666；F04／F12／F13的0176损伤仍接受；F10=0348／pairwise364 −0.126191仍接受，combined错误正分升到+0.070105，对应matched F09 +0.012127也接受。模式交互有容量不保证学对，分数不是置信概率，不能后验调阈值救例子。

全量相比baseline改动作22处，8改善／10损伤／4零效用，窗口不独立。新漏益包括0348／2171／matched363（+0.218004，原+0.021775→新−0.000379）、0349／2172／pairwise408（+0.162240，+0.041208→−0.055690）、0237／2172／matched299（+0.099441，+0.149452→−0.000435）、0237／2171／pairwise242（+0.025560）。不能只看13重点表；F09保住364不等于保住0348其他起点。

训练内更好而留组更差：future自身训练R²从baseline0.1420→combined0.2891、训练MSE0.011554→0.009582；留组MSE0.014869→0.015916，比原future恶化约7.04%，比固定截距0.013694差16.23%。combined训练future正／负放行84.42%／64.16%，留组76.42%／73.86%，仍弱；myopic组合训练R²0.7537，留组MSE0.032614对截距0.028086差16.12%。这些条件指标只对合法有变化支持，不与全259均值直接比；留组R²仍22非零方差模型，勿误读极端负均值。不能据此唯一归因容量、正则、某特征或标签噪声。

旧曝光合法30,805；四future提交记录27352／27848／26810／27269，宏提交32.30%／32.81%／31.58%／32.10%；范围外宏比例22.78%／24.05%／24.27%／25.48%，组合myopic提交18829、宏23.00%。高维更易至少一列越界，不是同尺度风险概率；这些记录未回写，不是新政策真实干预或新完整成绩。

本机核对原件／decision指纹、10CSV原始字节SHA及行数224／448／16／135／90／2072／3885／104／640／72。224模型scope维数／支持相符，56旧模型指纹与107一致，scope正规残差均≤9.72e-17；248新产物为168模型＋80预测缓存。2072预测标签／资格与106一致，104focus相符；3885决定／率控制／效用／遗憾及135policy／90paired／主比较重算一致；224held_out评价由原标签与新分数重算，训练截距／R²关系、448→16宏与支持数相符；640曝光单位整数与72宏可重建。常规误差≤1.14e-13；极端R²下向量化小舍入放大至基线个别R²差3.64e-12，仍低于1e-9容差且动作不变，不能声称分数逐位相同。正式新模型JSON／完整X与80预测NPZ仍在服务器，本机未重算新系数／映射／fit输入FP、全训练打分或逐帧曝光，未独立重算plan。

原件SHA`fc804f7d120e8613c41d06a53b0ca645ba4a2efcc90afc598ffd74761fa4d1a2`；decision`e8cf2e2928d42715b69956355858e04d2952eedd0b0b4ac05c7c541a1d9397a9`；plan`25cceb40976b2b5b922e5c9b8ebb7f728810e3635fe9d138af922613fe57b42e`。

接续优先考虑冻结本轮配方，按预声明线上冲突／写入条件抽取更代表性的开发状态与成对后继标签，保留源组隔离和简单控制，再检验是否有稳定新增选择价值；不在259旧起点继续后验堆特征、选13例或调阈值。之后仍需完整自身轨迹与封存源确认。本次仅归档复核，未新建Stage-109或改冻结代码；确认集封存，最近完整评价仍为Stage-103，尚无新增完整AUC或方法采用。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE108_RESULTS_REVIEW.md)

### 0.88 2026-11-15 Stage-109 无GT结果筛选的开发起点抽样（准备记录，结果见0.89）

用户要求继续实验，新增`tools/stage109_sampled_commit.py`、`tools/run_stage109_sampled_commit.sh`、`tools/test_stage109_sampled_commit.py`和协议`STAGE109_GT_FREE_SAMPLED_COMMIT_BRANCHES.md`。目标是检验旧门控在未按失败／救援／未来收益挑选的新开发起点上的选择价值，同时补充成对后继支持；本轮不再加特征或重训门控。

绑定Stage-108决策`e8cf2e2928d42715b69956355858e04d2952eedd0b0b4ac05c7c541a1d9397a9`。来源Stage-103的40单位×2模式＝80上下文，28开发源组。用Stage-106曝光缓存的线上eligible资格，保留`1..n-31`完整32帧窗口，按身份排除已有259标签起点；每个剩余池PCG64均匀无放回最多12个，全局seed2309／上下文SHA派生种子。池不足全取、空池保留，不按结果补样。manifest先封存池、选择、概率和预算再推理。总体仅为剩余合格完整窗口状态，不是所有帧，也不是新独立确认；可与旧窗口重叠。

每个起点共享一次四搜索，成对实测keep与both_pair，起点公开输出相同，只改变起点几何和外观提交，后面31帧恢复原核验、自身推进。252次新搜索／256逻辑调用，最多960起点／1,920分支／241,920搜索／1,693,440块。复用Stage-106 baseline49与Stage-108 relation_mode129各future／myopic、共112个原留组模型；GT只在四模型分数／动作保存后评价标签。新硬件导致fresh资格变化不换样，门控不合格则keep，但成对分支照常采集。

future标签是t+1..t+31有效帧平均IoU差，myopic为当帧pair框−原四候选公开框IoU代理。9政策为keep、always、离线oracle、两表示future／myopic及各自同率期望。主要relation_mode future−baseline future，必要对旧／自身myopic、always、自身同率、keep；按源组等权并报告两模式／两队列、支持与缺失，无帧独立CI。没有新完整轨迹AUC或采用门槛，也不打开确认集。

首次支持L20／L40自动绑定型号、UUID和软件，续跑必须同运行身份，换卡用新`STAGE109_ROOT`。除型号／UUID外变化拒绝；同运行keep须精确复现来源，换卡来源误差单列桥接，keep／pair两者当前卡实测。每起点四NPZ及收据，已完成缓存不重推理，所有产物和来源哈希封存。

本地5项Stage-109合成CPU测试通过，覆盖抽样／标签隔离、空短池、源池准备和只读模型加载、硬件绑定、同率和缺失、完整2起点4分支504搜索／512逻辑调用、决策先于分支、禁止新训练、来源不变、缓存零推理续跑和收据篡改拒绝。另7项Stage-105回归通过，包含有效提交改变实际后继搜索、GT轨迹隔离、一次提交及两卡模式缓存流程；新CLI帮助与Bash只读语法检查通过。正式数据、服务器NPZ／GPU本地不可用，尚未执行正式Stage-109。

同步代码后执行`bash tools/run_stage109_sampled_commit.sh`，默认`output/stage109/gt_free_sampled_commit_branches_v1`，支持Stage76／89／93／103／106／108／109路径、CHECKPOINT及STAGE109_GPU覆盖。后台先`mkdir -p output/stage109`，再`nohup bash tools/run_stage109_sampled_commit.sh > output/stage109/run.log 2>&1 < /dev/null &`；完整后自动生成decision＋8表的TXT，`--pack-only`只验证打包。不自动commit／push。收到结果先比较主要及简单控制、分层与错判，再决定是否用新标签启动另一期训练；不要把本轮抽样称独立确认或完整策略效果。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE109_GT_FREE_SAMPLED_COMMIT_BRANCHES.md)

### 0.89 2026-11-15 Stage-109 均匀抽样后继结果（已完成，旧门控未超过简单控制）

已收到并逐字节归档`stage109_results_20261115T023058Z_871630.txt`，4,502,572字节、SHA256`da7610e6d67a045ad6d621e9193ce967175cee08bbd35821bfb72907637e78e3`。decision指纹`1094eac91bf60b49ab1073a2f59779b7956449d15332f260d5b6acbd58702d80`，plan指纹`dded2f9442648004c3a825ce0eac530dcd438c0dbc791ff499acf029c70e3b98`；来源Stage-108与预声明一致。status为`COMPLETE_GT_FREE_SAMPLED_COMMIT_BRANCHES`，不是方法PASS。

L20、runtime差异空。80上下文／28开发组、960起点／1,920条32帧分支、241,920新搜索／245,760逻辑搜索／1,693,440块。112模型复用，新训练／完整轨迹／确认推理0。服务器报告keep来源框最大误差0、首帧共享通过；资格变化0，全部31帧有效后继，几何变化960、外观变化897。不重复采集。85,144旧状态中合格30,805、完整窗口29,819，排除其中219个已有标签身份后池29,600；每池16～926，均抽12，概率1.296%～75%。新身份与全部旧259身份无重合，但窗口可重叠、来源已消耗，不是独立确认。

全组效用：keep0、always+0.003936755、原future+0.003240232、原myopic+0.003356407、组合future+0.003120613、组合myopic+0.003307900、组合同率+0.003494755、oracle+0.006573505。组合−原future主差−0.000119619，中位0，6胜11平11负，最小留一−0.000481213；对旧myopic−0.000235794、自身myopic−0.000187287、always−0.000816141、自身同率−0.000374141；对keep+0.003120613，17胜11负、最小留一+0.001742124。均为一次提交后的最多31帧平均IoU差，不是完整AUC或重复门控效果。

历史12组主差+0.001080811，扩展16组−0.001019941；matched+0.000111209、pairwise−0.000350447。扩展pairwise组合+0.000492183，低于原+0.002358994、always+0.002773920和同率+0.002479142。源组主要改善0304+0.009643422、0226+0.003661079，损伤0091−0.010433598、0016−0.003719355、0076−0.001131996；0176−0.000181801。去0304主差−0.000481213，去0091为+0.000262380，同时去0304／0226为−0.000640532。只能作敏感性描述，不据此选组或改模式。

原future提交854、组合867；57动作变化＝35新增／22取消，效用19改善／29损失／9零。0304／2172／pairwise1077实测+0.470371，原分数−0.000118→组合+0.005189，提交救援被选中；keep后继0.236894→both0.707265，末帧0→0.822185。0226／2171／matched209实测+0.178119被新增接纳，但窗口质量仍低，不称恢复。0091／2171／pairwise78实测+0.345871却从+0.003360→−0.007024被新拒绝，keep后继0.286142→both0.632013；该处myopic−0.055821。0016／pairwise605的+0.089265、0076／pairwise634的+0.027168也被新拒绝。共同漏挡0237／2172／pairwise276：myopic0，future−0.234572，keep末帧0.914286、both0，原／组合future都正分放行；0226／2172／pairwise204的−0.174033也均放行。0176／2172／pairwise243的+0.108408均被拒绝。

标签449正／361负／150零；仅69正与46负超过绝对0.01，其余845为小效应或零。组合对正例400/449＝89.09%接纳、负例328/361＝90.86%接纳；原为401/449＝89.31%、319/361＝88.37%，仍近于多数放行。宏提交率原89.81%／组合90.92%，不要混用原始计数。future宏预测均值原+0.027181／组合+0.029073，而实测宏+0.003937；新标签MSE0.002450／0.002781，组合约高13.5%，训练范围外宏比例22.99%／25.67%。支持标度和迁移存在问题的线索，不能唯一归因特征维数或训练分布。

本机核对decision自指纹、8表SHA／行数、采样身份排除／随机种子派生／范围概率、1,920分支到960标签、3,840预测阈值、8,640决定、81汇总／54配对全部重算一致，7,680个NPZ／收据名称账本完整。原manifest池、实际NPZ和模型本机缺失，未在本机重放完整采样、核验实际缓存或复现GPU；不要假称本机执行完整check-results。

**导出字段问题：sampling_scope.csv的seed被抽样随机种子覆盖，是128位sampler_seed，不是PPO种子。** context里的`__s2171__`／`__s2172__`及其他表的seed正确，主分析未受影响。读取该表连接样本时必须从context恢复actor_seed；下一阶段分开sampler_seed和actor_seed命名。保留Stage-109封存源码／结果／哈希，不为字段问题重跑GPU或改写旧表。

下一步优先利用这960个新标签，冻结表示与门槛，在源组留出和简单控制下验证训练支持改变能否改善选择，而不是继续后验加特征／调阈值。本轮未新建Stage-110或启动新训练；确认集封存，最新完整AUC仍Stage-103。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE109_RESULTS_REVIEW.md)

### 0.90 2026-11-15 Stage-110 学习预测与校准驱动的候选生成（准备记录，结果见0.91）

用户再次质疑固定四布局的证据，明确要求按“由算法生成并建立强证据链”推进。新建`tools/stage110_proposal_common.py`、`tools/stage110_search_proposals.py`、`tools/run_stage110_proposals.sh`、`tools/test_stage110_proposals.py`及协议`STAGE110_LEARNED_SEARCH_PROPOSALS.md`。接续主线改为预测依据→误差校准→同预算实际候选供给，不直接继续Stage-109提交门控重训；已有机制与负例保留，冻结旧代码。

绑定Stage-109决策`1094eac91bf60b49ab1073a2f59779b7956449d15332f260d5b6acbd58702d80`和原Stage-103完整历史。28开发组／40单位×两来源＝80上下文，t≥5每上下文最多96个等间隔拟合身份，从剩余帧按seed2310派生PCG64均匀抽最多24评价帧。采样不看GT、失败、分数或eligible；短池保留，不补样，评价缺GT不删。拟合与评价身份分开，最多7,680拟合记录／1,920评价状态；整条源历史只读，不用新结果替换后续状态，不打开确认集。

预测32维只含过去五框的中心／对数尺寸增量16、均值／标准差／转折12、历史比例和过去raw／Hann／memory_write四项。目标是GT相对最新预测框的中心位移／side和宽高log比。每留组包只用其余27组：按排序每三组取一校准，18组拟合／9组校准／1组评价；固定ridge=1均值，拟合组内部三折残差再回归log(error²+1e-4)预测条件尺度，另取固定尺度。两尺度各用校准组的源组平衡90%最大标准误差分位数校准；没有时间相关数据的分布无关覆盖保证。旧PPO仍是已消耗来源，留组隔离只针对新预测包。

六组fixed、旧Stage-84 motion、learned_constant、learned_uncalibrated、learned_calibrated、calibrated_random；四为相同搜索预算，不是四布局最优主张。均保留原q0；新生成组三个额外父框的中心／尺寸可共同改变，来自32个固定反向配对高斯场景及预测均值。按假设中心与尺度的软覆盖增益分配，random共用校准提议池和裁剪去重约束。q0保留、对角高斯、soft coverage仍是预声明假设；生成几何更广不能代替真实SGLA解码质量。

六组每状态24逻辑搜索，共同q0及完全相同浮点父框共享，最多19物理搜索／状态，总上限46,080逻辑／36,480物理／255,360块。只同状态供给，不完整推进，不新增PPO／SGLA更新；拟合28个CPU预测包。先保存在线输入与提议，再实测候选与附GT；同运行fixed框／特征／裁剪／层须精确复现来源，换卡误差单列，六组同卡共同实测。

主要learned_calibrated−fixed最佳解码IoU，必要对motion／constant／uncalibrated／random；报告好候选比例、得到与丢掉好框、输入目标像素、有效覆盖／裁剪数及预测中心尺度误差、90%经验区间覆盖与宽度。另有冻结旧PPO影子选框，不能单独当生成器有效性或新策略闭环证据。按来源→种子→源组宏平均、两队列／模式分层，最多135对比，无帧独立CI、无最终方法采用门槛。若预测和真实供给有积极证据，再做旧／新生成器×旧／匹配训练选择器整视频验证。

6项本地CPU测试通过：过去输入和GT隔离、采样身份分离、拟合／残差／校准源组隔离、校准标签不改变预测系数、条件误差尺度、校准经验覆盖、中心尺度共同生成、q0和随机同池、实际原生解码／去重调用、完整合成流程与来源不变、缓存零GPU续跑／篡改拒绝、L20／L40运行绑定。恢复数值审计rtol=1e-9／atol=1e-12，不改封存系数或指纹。Stage-84几何生成旧测试1项通过，真实Torch／OpenCV回放缺依赖跳过1项，不能算通过；新CLI、Python／Bash语法、LF通过。正式数据与GPU本机不可用，没有正式Stage-110成绩。

同步后`bash tools/run_stage110_proposals.sh`；默认`output/stage110/calibrated_search_proposals_v1`，支持Stage76／89／93／103／109／110路径、CHECKPOINT和STAGE110_GPU。后台先`mkdir -p output/stage110`，再`nohup bash tools/run_stage110_proposals.sh > output/stage110/run.log 2>&1 < /dev/null &`。首次L20或L40自动绑定，续跑换卡需新STAGE110_ROOT；完成打包decision＋8表TXT，`--pack-only`只校验打包，不自动commit／push。 [完整协议](docs/experiments/06_sgla_uav_tracking/STAGE110_LEARNED_SEARCH_PROPOSALS.md)

### 0.91 2026-11-15 Stage-110 学习生成候选结果（已完成，供给积极但预测和实际选择仍不足）

已收到并逐字节归档`stage110_results_20261115T044713Z_948813.txt`，4,226,624字节，SHA256`8367c4e1527ab0206431651ca9bb203aeb325299595ccc3cf0bec224c5c70806`。decision指纹`dd804631d678a6e112c45af5df33d3ebbbf88bd3113dfdd7af0be7f7fb31b254`，plan指纹`c539321d81f7df6558ae09373818b449150a2fcb41f09c83d545b9de921c0dab`，Stage-109来源一致。状态`COMPLETE_LEARNED_SEARCH_PROPOSAL_SUPPLY_PILOT`，不代表方法采用通过。

L20、runtime差异空。28开发组、80上下文，96 fit／24 eval各上下文，7,680有效fit身份、1,920共同评价状态且GT全有效；28个预测包的18 fit／9校准／1 held组及行数相符。35,768物理搜索、250,376块、46,080逻辑搜索，逐状态16～19次、均18.629；fixed来源框误差0，共同q0通过。SGLA／PPO更新、完整轨迹及确认推理均0，不重复运行。所有候选均在同一旧历史上，不回写未来。

全组最佳IoU／好框率／旧PPO影子选框：fixed0.650362／81.03%／0.632668，motion0.651938／80.77%／0.632911，constant0.645502／80.36%／0.346476，uncalibrated0.661859／81.21%／0.631635，calibrated0.670099／81.81%／0.479921，random0.657992／81.06%／0.444855；各臂q0同为0.631912。calibrated最佳IoU对fixed+0.019736813（22胜0平6负，中位+0.013265807，最小留一+0.012098769），对motion+0.018160926／constant+0.024596687／uncalibrated+0.008239652／random+0.012106524，四项最小留一均正，分别19／24／17／18胜。已有同预算候选质量增益，不是完整AUC或实际方法成功。

好框率对fixed仅+0.0078125，8胜13平7负、最小留一−0.004243827。0349从39.58%→72.92%，去该组其余好框率均差−0.4244个百分点。原始好框状态1,541→1,555，新增35／丢21；uncalibrated新增14／丢8，motion8／13，constant13／25，random23／21。最佳IoU去0349仍+0.012099，去0349／0348仍+0.009903，不能把“最佳质量广泛改善”与“缺框状态广泛补回”混为一谈。

分层最佳IoU主差：历史12组+0.013562630，扩展16组+0.024367450，matched+0.018236173，pairwise+0.021237454；旧PPO选择四层均明显负。源组最佳IoU主要正0349+0.225964、0348+0.069179、0342+0.055205，负0085−0.043261、0070−0.026303、0016−0.023280、0176−0.020182。不删负例、不后验改倍率。

预测均值未通过更准的解释：中心误差latest0.638011／velocity0.641143／learned0.700920，学习较latest增9.86%；log尺寸误差0.234355→0.244974，增4.53%。历史尺寸小改善、扩展恶化，中心两队列均差。四维覆盖constant93.86%／倍率1 uncal13.69%／cal93.01%；uncal并未声称90%覆盖。条件校准倍率9.155～25.850，中位13.313。条件半宽四维均值宏10.599，状态中位1.560／P95=5.137／最大21,950.57，0307源组宏232.831拉高；matched18.700、pairwise2.497。该量混合归一化中心与log尺寸，不能解释像素；93%覆盖不等于高效或正确概率模型。

旧PPO calibrated−fixed影子IoU−0.152747342，2胜26负；最佳与选择差距0.017694→0.190178，有好框却漏选宏1.34%→23.96%、原始23→438。331状态q0≥0.5却选IoU<0.2，fixed为0；constant673、random417。候选槽位语义／分布与原拼接256维PPO不匹配是应检验假设，不是已排除其他解释的唯一原因。不要将旧PPO直接退化当生成算法无价值，也不预设重训必能修复。

代表正例：0091／2171／pairwise143 fixed最佳0→cal0.858173但PPO仅0.028958；0349／2172／pairwise325的0→0.773595却选0；0349／matched364的0→0.768802且选中，但无后继；0304／2171／pairwise560的0→0.711161却选q0=0；0003／pairwise143的0→0.769231却选0。负例：0176／2172／matched746的0.815089→0.119715；0016／matched577的0.905172→0.179152；0237／2171／matched293的0.719368→0，最大GT裁剪覆盖仍1。0003／matched33 q0=0.946092却选q1=0，0076／pairwise375 q0=0.947948也选0。实际供给损伤与有框错选并存。

本机已核对decision自指纹、8表SHA／行数、80上下文采样身份和分列seed、fit／校准身份及行数、1,920状态／11,520供给、共同q0和好框变化算术、调用预算、54供给汇总／9预测汇总／135对比全部重算一致，另独立宏平均复核全组指标。7,709模型／输入／NPZ和收据名称账本完整。服务器manifest、拟合输入、实际28模型／3,840 NPZ及收据未在本机取得，未本机复核模型系数或GT预测标签、复演GPU，不声称执行完整check-results。

接续优先用已有proposals／observations作缓存审计：中心与尺度分布、极端预测范围和图像约束、好框所在槽位、原PPO对槽位排列的依赖；再设计匹配选择器训练或顺序无关评分对照。新选择器外层留组时，生成器fit／cal也必须排除同一评价组，不能复用“其他组预测包含该评价组标签”的缓存造成间接泄漏。保留uncalibrated与简单控制，不在1920结果上搜倍率／删异常点。之后仍需完整自身轨迹和封存源确认。本次只归档复核，未新建Stage-111或新训练；最新完整AUC仍Stage-103。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE110_RESULTS_REVIEW.md)

### 0.92 2026-11-15 Stage-111 同候选 PPO 槽位排列诊断（准备记录，结果见0.93）

用户要求继续下一轮。针对Stage-110“集合最佳改善、旧PPO实际选择明显退化”，本轮保持全部1,920状态、六个原候选集合、共同q0、原历史和冻结PPO，枚举q1/q2/q3的六种排列。基础特征与框同步重排，重新计算关系特征，保持原256维逐槽位归一化、重复掩码和argmax。共69,120次CPU actor前向；不读取图像、不训练、不作GPU推理或新完整轨迹，确认集封存。

新增`tools/stage111_slot_audit.py`、`tools/test_stage111_slot_audit.py`、`tools/run_stage111_slot_audit.sh`。默认根`output/stage111/cached_ppo_slot_audit_v1`；运行`bash tools/run_stage111_slot_audit.sh`，无需checkpoint/GPU。自动打包decision与六表；断开SSH继续时用协议中的nohup命令。来源需保留完整Stage-76/89/93/103/110目录，Stage-110完成指纹固定为`dd804631d678a6e112c45af5df33d3ebbbf88bd3113dfdd7af0be7f7fb31b254`。

九项控制：identity、按原fixed父框几何对应的canonical_geometry、六排列概率映射/重复输出合并后平均的probability_symmetrized、uniform_order_expected、q0、Hann、best_order_offline、worst_order_offline、candidate_oracle。前三种及简单控制不使用GT；最好/最差排列仅为离线上界。无GT预测及概率先保存为1,920个NPZ+收据，再附标签。identity动作和IoU、候选最佳及q0必须逐状态复现Stage-110；误差≤1e-12。

主要比较预先固定为learned_calibrated中的probability_symmetrized−identity当前IoU，另报几何对应/随机排列/最好排列与identity、概率汇总与q0/Hann；所有六集合、历史/扩展、matched/pairwise完整保留。报告有好框漏选、有害q0替换、物理框排列敏感性、六排列IoU范围、旧归一化截断前越界比例。先上下文平均再模式/种子/源组等权。表行数scope1,920、state_summary11,520、policy_summary486、paired972、permutation_summary324、sensitivity_summary54。无方法采用门槛；槽位变化同时涉及旧归一化与掩码，不能唯一归因神经网络结构。未通过六排列也不能单独否定新生成候选。

5项本机CPU测试通过、无跳过：真实旧gate动作/概率映射，槽位偏好反例，六重编号下概率汇总不变性（含重复/并列），几何对应与GT/未来历史隔离，归一化截断统计，完整合成缓存、预测先保存、来源不变、续跑及篡改拒绝。测试禁止拟合/GPU初始化/图像读取。CLI、Bash语法及LF通过。正式缓存本机没有，尚无Stage-111正式结果；最新完整AUC仍Stage-103。收到服务器bundle后先核对identity复现和物理不变性，再评估是否需要顺序无关且匹配训练的选择器。不要直接从最好排列上界宣称在线收益。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE111_PPO_SLOT_PERMUTATION_AUDIT.md)

### 0.93 2026-11-15 Stage-111 排列诊断结果（已完成，换序不足以修复旧PPO）

已收到并逐字节归档`stage111_results_20261115T055052Z_976171.txt`，6,125,566字节，SHA256`c149cd7ee5ea4721cc762c46504cbf7bf7c1c34201d0ccb5bce1bf46414ce0c0`。decision指纹`61d3d1e808b8df3f317489c2b9f0dc4d69fe7f65a5e7f8e04d2aa492c39b13f5`，plan指纹`4d398df46fb370a4ee0bc772a118e61c371e1c24197e39300275b97840713eaa`；Stage-110来源一致。1,920状态／六原集合／69,120 CPU actor，原动作、选择IoU、best和q0逐项与Stage-110相符，最大误差0；零新图像／SGLA／训练／完整轨迹／确认推理，不重跑已完成阶段。

校准集合identity0.479920739、canonical0.444758945、symmetrized0.436110145、uniform-order0.446482546、q0=0.631911800、Hann0.625630938；逐状态GT挑六排列最好0.604446705，四候选oracle0.670098611。主比较概率汇总−原PPO为−0.043810594，8胜4平16负、中位−0.016438265、最小留一−0.051441461。几何对应−0.035161793、8胜4平16负；随机排列期望−0.033438192。概率汇总对q0−0.195801655（0胜6平22负），对Hann−0.189520793（4胜2平22负）。六个全局排列均低于q0，最好0312也仅0.486742；不能后验挑排列当方法。

排列效应存在但不等于可用修复。宏物理选择变化fixed89.40%／校准74.03%，六排列IoU范围fixed0.023991／校准0.320680：原固定框也常变化但后果小，新校准框变化后果大。校准raw有害q0替换identity331／canonical440／sym451，漏好438／563／602；q0漏41，Hann漏49且严重替换10。原331例中222例某排列可选好，但另76状态q0≥0.5而六排列最好仍<0.2。0147／2172／matched1053 q0=0.805085，六排列均0；0126／2172／pairwise272 q0=0.896266，六排列最好0.184456。范围限于q0固定的六排列，不声称遍历移动q0的24排列。

同状态正例0237／2172／matched380从0→0.925822，0003／2172／matched36从0→0.893459；严重负例0174／2172／matched1319从0.981132→0、0329／2172／pairwise172从0.931818→0。概率汇总历史12组0.398714<原0.490887，扩展16组0.464157<原0.471696；matched0.432076<0.480842，pairwise0.440144<0.478999。不能删负例或只看局部正例。

原槽位归一化后截断前越界比例fixed0.3826%／uncal6.7610%／cal26.8061%／constant38.4171%／random24.8990%。校准六排列约26.55%～27.27%，换序未消除特征支持压力。这是新候选与旧接口不匹配的线索，不能唯一归因截断、MLP结构或PPO算法；本轮保持旧归一化及重复掩码。fixed概率汇总几乎不变0.632668→0.632665，constant有小改善0.346476→0.372754仍差，uncal0.631635→0.632756接近简单控制，生成范围与旧评分交互仍需研究。

本机已核对decision自指纹、六表SHA／行数、全部状态身份／GT有效、调用／零新增账本、Stage-110逐项连接、六排列IoU上下界与几何对应、486策略／972配对／324排列IoU及有害替换／54敏感性汇总重算，另独立宏平均核对54个全组策略；54分层截断原排列和六排列平均一致。3,840 NPZ/收据名称及SHA格式完整；实际服务器manifest、actor、观测/预测NPZ及收据未取得，未本机复演概率/特征或完整check-results，逐排列截断比例也无逐状态值可完全重算。

接续保留算法生成候选供给主线，但不采用当前换序/概率汇总。优先预先固定共享逐候选、匹配新集合的监督或成对评分控制，超过q0/Hann/旧PPO后再检验PPO后继增益；顺序无关不自动等于正确评分。外层评价源组须同时从生成器拟合／校准和选择器拟合中排除，不能直接用Stage-110其他组的OOF生成缓存导致评价标签间接进入训练。之后仍要完整自身轨迹与封存源确认。本次只归档和复核，未建立Stage-112或新训练；最新完整AUC仍Stage-103。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE111_RESULTS_REVIEW.md)

### 0.94 2026-11-15 Stage-112 匹配评分训练与2/4/6/8搜索数量对照（准备记录，结果见0.95）

用户讨论旧PPO流程、新生成器与选择器适配及“四个仅为预算控制”后要求继续实验。本轮保持生成算法和SGLA权重，新增共同外层源组留出、监督质量训练及数量迁移。新增`tools/stage112_scoring_common.py`、`tools/stage112_shared_scoring.py`、`tools/test_stage112_shared_scoring.py`、`tools/run_stage112_shared_scoring.sh`。来源完整Stage-76/89/93/103/110/111，111完成指纹固定`61d3d1e808b8df3f317489c2b9f0dc4d69fe7f65a5e7f8e04d2aa492c39b13f5`。

四外层fold各留出3历史+4扩展=7源组，其余21组中14拟合生成器、7校准；评分器与标准化使用21训练组。四个生成包重新拟合，不能复用Stage-110其他组包含外层评价标签的生成缓存。沿用原96拟合帧，从中等间隔取24帧作评分器训练；原24评价帧不变。每fold训练1,440/评价480，合计7,680个新采集fold状态，外层评价1,920覆盖全部28源组。所有上游历史已消耗，仍是开发证据，确认集封存。

两个生成集合fixed4/generated最多8；新generated2/4/6/8严格取同一贪心序列前缀。共享质量MLP每候选64→32→1（2,113参数），拼接控制256→32→4（8,356参数），相同IoU加权MSE、训练状态/标签/标准化/批次/更新预算，各两种子2312/2313。共享64维标准化跨训练槽位拟合，K=4精确复用原描述，其他K按实际候选数计算关系。20epochs、minibatch128、Adam3e−4、梯度clip1、固定末步，无评价挑选；共32评分器、通常7,680次优化更新。原PPO冻结作K=4控制，q0/Hann及GT oracle完整保留。K=2/6/8复用K=4训练的共享模型，不分别重训或后验宣布最优K。

主比较generated4/shared−Hann当前IoU；必看q0、旧PPO、重训concat和fixed4/shared，另报2/6/8−4。新搜索父框先存，评价选择/分数先存再附标签。每状态fixed4+generated8共12逻辑槽位，最多11唯一搜索；上限92,160逻辑/84,480唯一/591,360搜索块，另7,680模板初始化。计时同步GPU并按集合内唯一观测求和，排除图像/模板/提议/评分，不能称端到端时延；q0按已采整集合记账仅供同预算诊断，单独q0可只搜索一次。

默认根`output/stage112/nested_shared_scoring_budget_v1`。入口`bash tools/run_stage112_shared_scoring.sh`，需要GPU及原checkpoint；首次L20/L40自动绑定，续跑同根运行身份必须一致，换卡新设`STAGE112_ROOT`。后台命令见协议；完成自动生成stage112_results TXT。预计fold_support4、sampling_scope320、training_scope32、collection_scope7,680、state_decisions59,520、policy_summary216、paired486七表。完整缓存续跑零重复搜索/评分器训练，生成器会数值重建核验；未提交CPU模型从固定种子重算。没有SGLA/PPO更新或新完整轨迹；完整自身轨迹/稳定性/恢复保护及随后PPO价值属于接续验证，不把同状态成绩当AUC。

5项本机CPU测试通过、无跳过：实际shared/concat梯度有限差分，所有K特征/评分排列一致性及K4原描述一致，多组共同排除及评价标签扰动不改变生成器，确定性监督拟合，严格预算前缀/真实仓库解码/唯一调用，完整合成采集-训练-评价、预测先保存、零重搜重训续跑、来源不变和预测/模型篡改拒绝。网络为真实NumPy实现，SGLA/图像由合成Observer替代；无正式GPU复现。CLI、Python/Bash语法及LF检查通过，Stage-1～111 Python未改。正式服务器结果待运行。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE112_NESTED_SHARED_SCORING_BUDGET.md)

### 0.95 2026-11-15 Stage-112 评分适配与候选数量结果（已完成，接近q0但净收益未建立）

收到并逐字节归档`stage112_results_20261115T073439Z_1008077.txt`，17,427,407字节，SHA256`fc81f5dd269359cdd93b150e70f4eee84e30a2461c9d97d9b8b3ea4351b88599`。decision指纹`4622ef840a90ccfc658ff91b86652ac460a1371a3da9579e37b08425d8f3803f`，plan指纹`375afca111d43d4132810a86a9941040f49544c06f66bf7c09d218c9e95d4590`；Stage-111来源一致。L20／UUID bd2283f0-25e4-2734-53c1-e3a243ca482c，runtime差异空、fixed来源误差0。4新生成包、32评分器，每个1,440样本/240更新，共7,680评分优化；7,680fold状态、84,468唯一搜索、591,276块、92,160逻辑搜索，另7,680模板初始化。零SGLA/PPO更新、完整轨迹与确认推理；不重跑。

generated4的q0/Hann/oldPPO/concat/shared/oracle IoU为0.631911800／0.626374013／0.546084471／0.627471226／0.631973853／0.668758410。预定shared−Hann +0.005599841，16胜12负、中位+0.002110119、最小留一+0.004017027；对oldPPO+0.085889382（25胜3负、最小留一+0.069152769），对concat+0.004502627（15胜13负、最小留一+0.002578188）。两网络都做新监督拟合及共享标准化，不能将全部恢复只归于结构，参数2,113/8,356也不相同。

关键限制：shared对q0仅+0.000062053，11胜17负、中位−0.001935351、最小留一−0.001725864；最大获益0308+0.048335812，去该组转负。训练种子2312/2313在generated4为0.634195786/0.629751920，对q0+0.002283986/−0.002159880，不能后验选2312。历史12组shared−q0−0.001638，扩展16组+0.001337；matched+0.000935，pairwise−0.000811，净新增价值未稳。

固定四布局q0/Hann/oldPPO/concat/shared/oracle为0.631911800／0.632988945／0.632668080／0.633453279／0.632728161／0.650361797。新generated4供给最佳较fixed4+0.018396613，但shared实际较fixed4/shared低0.000754308，13胜15负；仍未转化生成供给进展。新外层生成器14拟合/7校准与Stage-110的18/9不同，旧PPO0.479921→本轮0.546084不是PPO训练进展，不跨候选集归因。

严重替换有明确修复：oldPPO185/1,920状态q0≥0.5却选<0.2；shared仅2/3,840决策（两训练seed），宏比例9.8958%→0.0558%，分母不同不能直接当同口径185→2。shared选择非q0索引1,997/3,840=52.01%，不是索引上永远q0，但没有实际候选坐标不能判定都改变物理输出，也不能把使用率当收益。shared好框选择3,016次，q0同双seed重复计数3,028次；shared漏好92、q0漏80，小定位增益与损伤抵消。

generated2/4/6/8的oracle为0.646771023／0.668758410／0.683774474／0.691895102，shared为0.629812399／0.631973853／0.630023221／0.627226492。6−4均差−0.001950632（9胜19负）、8−4−0.004747361（4胜24负）。对应搜索观测耗时14.35/25.81/37.93/49.48ms，排除图像/模板/提议/评分，非端到端。K2/6/8均用K4训练模型，关系特征随集合变化，因此只说明此模型的数量迁移未获益，不能宣布四最优或更多候选无价值。候选前缀的oracle/调用/计时单调性逐状态成立。

正例0089／来源2172／pairwise696 q0=0.669118，两训练seed均选0.867925；0307／来源2172／pairwise56 q0=0.366692，选0.510993/0.675395。严重负例0176／来源2171／pairwise668 q0=0.761108，两训练seed选0.761108/0；0085／来源2171／pairwise1453 q0=0.535996，选0.535996/0。32模型训练MSE均下降，concat拟合误差更低却选择更差，不能只凭拟合MSE加训练或宣称校准。

本机核对decision自指纹、七表SHA/行数、四外层及14/7/7身份与生成行数、从Stage-110采样表重建320上下文、32模型匹配输入/样本/更新/参数账本、7,680采集及1,920评价身份、59,520决策算术和共同集合，216汇总/486配对全部重算，独立宏平均核对24全组策略；34,597产物名/SHA格式完整。另fixed的q0/旧PPO/oracle共5,760项逐状态与Stage-110归档表相符。实际服务器manifest、模型系数、NPZ/收据未取得，隔离仅为回传支持表与固定设计一致性，未独立复演训练/概率或完整check-results。

接续优先冻结当前生成器与两训练seed共享评分器，在完整自身轨迹中比较q0、fixed4/shared、generated4/shared，检查稳定与恢复保护及小错放大；暂不扩大K或后验挑seed。在同架构下验证PPO后继价值另作匹配目标对照，当前既没证明PPO必需，也不能因当帧接近q0否定长期研究。本次只归档与复核，未建立Stage-113/新训练；最新完整AUC仍Stage-103，确认集封存。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE112_RESULTS_REVIEW.md)

### 0.96 2026-11-15 Stage-113 冻结生成器/共享评分器的完整自身轨迹（准备记录，结果见0.97）

用户要求继续下一步，现有Stage-112固定状态适配尚未证明完整收益。本轮已实现`tools/stage113_frozen_closed_loop.py`、`tools/run_stage113_closed_loop.sh`与`tools/test_stage113_frozen_closed_loop.py`，不修改Stage-1～112 Python。五臂为native、fixed_hann、generated_hann、fixed_shared、generated_shared；native/Hann每视频1条，shared两训练种子2312/2313各1条。28开发源组去重为28视频，共196完整轨迹，历史12/扩展16分列；28确认组继续封存。

绑定Stage-112 decision `4622ef840a90ccfc658ff91b86652ac460a1371a3da9579e37b08425d8f3803f`，复用四外层生成包与fixed4/generated4共享评分器各8份，共20模型文件，不重新拟合/标准化、无新PPO或SGLA训练。每个评价组使用自己的外层留出fold。旧40个来源视频/种子单元只定位数据和来源，不重复计作40个视频；native所有控制均在当前卡新跑。generated_shared−fixed_shared是“生成＋匹配评分”整体比较，不能称纯生成器效应；两Hann臂补同一选择规则的输入比较。

每条轨迹仅首帧GT初始化，完整推进自己的实际选中框。生成器读取本臂最近最多五框、自己的上一帧4×24特征、选框和可信写入资格，不导入旧轨迹历史。资格位按Stage-103同一公开响应/几何规则重新计算，保留memory_write字段仅为预测输入兼容；没有外观记忆重排/新模板更新。第一跟踪帧统一固定布局，第二帧起预测，短历史沿原online_input零填充；原训练采样起于frame≥5，早期曝光差异必须保留。每视频七条轨迹落盘后才做GT评价；单轨迹未提交则从首帧重跑。

按既有帧表28视频29,047跟踪帧、七遍203,329帧次，逻辑搜索上限726,175。帧内完全相同浮点父框去重，executed掩码审计物理head/block；每搜索7块，模板各轨迹一次。NPZ保留原始响应/几何图、父框/候选/特征/生成状态与选择，196份NPZ及196份receipt全部SHA绑定。报告计数为唯一已提交产物，失败重试实际计算另计。计时排除图像/模板/写盘，原生也有审计特征，不声称部署速度。

统计先视频内平均两训练种子，再28视频等权，bootstrap seed23130/20,000。主比较generated_shared−fixed_shared，并将对native、generated_hann列为必要对照，三AUC Bonferroni家族区间；另报名义区间、历史/扩展与六项逐种子差。PROMISING要求三对照均≥+0.5AUC、家族下界>0、两seed各自正；对native/fixed_shared低IoU及最长失跟不增；原生可靠帧差CI下界≥−1；两个seed均无原生/同seed固定稳定视频损伤或参考恢复丢失；≥4视频有改变裁剪且实际非q0输出。否则REVIEW，不把保护失败藏在均值里。恢复用既有3低/3好及15帧匹配规则，逐事件保留，保护汇总为任一seed受损，丢失恢复计数为两个seed总数。

默认根`output/stage113/frozen_generated_shared_closed_loop_v1`。服务器激活sglatrack-test并同步代码后：`mkdir -p output/stage113`，再`nohup bash tools/run_stage113_closed_loop.sh > output/stage113/run.log 2>&1 < /dev/null &`。可用STAGE113_ROOT/STAGE113_GPU及来源根/CHECKPOINT覆盖。首次支持L20/L40并绑定UUID及软件，同根续跑必须同运行身份，换卡新root；同身份native框须与Stage-103误差0，首次换卡仅容许GPU/UUID差异并报告漂移。入口自动核验与打包，`--pack-only`仅核验打包完整结果；不自动commit/push。

预计八表：model_scope20、policy_summary15、paired最多270、video_summary140、seed_video196、scope196、recovery逐事件、seed_effects6，另decision。收回自动生成的stage113_results TXT即可进行复核，完整manifest/模型来源/NPZ收据仍保留服务器。先核对来源与两seed、整体/队列、供给与漏选、稳定/恢复事件，再决定是否分析首次分叉；不扩K、不选seed、不默认重训PPO。

本机七项CPU测试8.113秒全部通过，无跳过：五臂自身历史/冷启动/预算及GT隔离、过去响应和资格位输入、改变选择影响后续裁剪、恢复匹配、硬件首次绑定/续跑拒绝、冻结模型加载/泄漏拒绝、两队列14轨迹完整流程/先落盘后标签/禁止新训练/零重复推理续跑/篡改拒绝。生成/评分/解码/统计为实际仓库实现，SGLA与图像为合成Observer，不能称GPU精度复现。Python语法/CLI、Bash语法及LF通过。以上为运行前准备记录；服务器随后已完成，结果见0.97。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE113_FROZEN_SHARED_CLOSED_LOOP.md)

### 0.97 2026-11-15 Stage-113 完整自身轨迹结果（已完成，生成方案明显退化）

收到`stage113_results_20261115T121348Z_1071811.txt`，235,656字节、SHA256 `13aa51d094a9c8edd970c7be114d3dcb184e8c5104e1494ef752a7cb5197e938`，逐字节归档results并加-text保护。decision指纹`11cada1ae2be6ad882acb3f6a55e4a0073ea44dc9957b70dfe1c69a799d93d0c`，plan `cef8b54972f44135c581e587a841cdc5dcbf0e57abf63119d8bc304a7059efa5`。来源Stage-112相符；L20／UUID bd2283f0-25e4-2734-53c1-e3a243ca482c，运行身份差异空、native框误差0。28组196完整轨迹、203,329帧次、725,283唯一搜索、5,076,981块、726,175逻辑搜索；20份冻结模型、0新模型/训练/确认推理，不重跑。

native／fixed_hann／generated_hann／fixed_shared／generated_shared的完整AUC为63.9605／64.5947／46.2415／64.1309／53.5418，低IoU13.20%／11.91%／39.29%／13.43%／29.04%。主差生成共享−固定共享−10.5890，名义CI[−18.0703,−3.7164]、家族CI[−19.8596,−2.3016]、7胜21负、中位−1.0350，全部留一均差均负[−11.8059,−8.8041]。对native−10.4187、5胜23负，家族上界−1.6552；对generated_hann+7.3003但14胜14负且家族跨零。generated_hann−fixed_hann−18.3532、7胜21负，生成损伤也发生于Hann控制，不可只归咎共享评分。

fixed_shared对native仅+0.1703、CI[−5.5484,+5.8882]、14胜14负，对fixed_hann−0.4639。原生可靠帧AUC native75.0192、fixed_hann72.5438、fixed_shared70.9015、generated_shared60.4414；生成共享对原生−14.5778、2胜26负，CI[−22.3740,−7.5949]。固定Hann均值最高也有保护损伤，不能直接作为合格最终方案。

训练seed2312/2313：fixed_shared66.4678/61.7939，generated_shared57.1878/49.8959；生成−固定−9.2800/−11.8980、对原生−6.7728/−14.0646，不能挑2312采用。历史12组native/fixed_shared/generated_shared63.4512/63.1842/53.9451，主差−9.2391、5胜7负；扩展16组64.3425/64.8408/53.2394，−11.6015、2胜14负。损伤不是单一旧0304或仅历史组问题。

11门槛仅nontrivial_usage通过，其他10项全部未过。生成共享非q0物理输出24,792/58,094帧次42.68%，28视频均实际干预，58,038帧有生成裁剪变化。原生12稳定视频新增5失败：0070/0080/0174/0304/0308；固定共享新增0003/0080/0304，fixed_hann新增0080，generated_hann新增0080/0239/0304。相对同seed fixed_shared稳定轨迹，生成共享新败6不同视频0016/0070/0169/0174/0304/0308（2312四条、2313三条，0070重合）。新增失败含短事件，不能都称长期失跟。

生成共享丢原生恢复14/26评价（13参考事件×2seed），涉及0076/0089/0169/0226/0307/0348/0349；丢同seed fixed_shared恢复18/24，涉及0080/0089/0176/0307/0348/0349。固定共享丢原生5/26，fixed_hann2/13，generated_hann12/13；比较保留分母，不把两个参考的32次丢失相加成32个独立事件。未匹配为15帧时限内未满足3好帧，不代表以后永不恢复。

完整自身状态上，fixed_shared候选最佳/实际IoU0.668424/0.650694，好框可用84.06%；generated_shared0.569442/0.544003、69.34%；generated_hann最佳0.491144、59.66%。生成自身历史的供给和选择差距均恶化，不能将Stage-112旧来源历史的0.668758供给直接移植成当前闭环能力。当前生成历史冻结后的oracle上界也低于固定实际，但oracle改选会改变未来，不能当oracle完整政策AUC或唯一因果分摊。短历史frame2启用与训练frame≥5、旧策略训练状态支持差异都只是待检验假设，现实遮挡/相似物原因未证实。

关键视频：0080固定共享66.6071→生成7.8234、双seed约88%低IoU且末尾未恢复；0176固定67.2012→生成14.6974，固定的6次参考恢复全部丢失。0089−43.7974、0304−35.6315、0307−34.5567，最差五条贡献总体−8.0455，其余23条仍均值−3.0965。0304固定2312/2313为72.3478/22.1394，生成18.0759/5.1484；固定本身也有单seed持续失跟。0204生成72.6746/72.3747有救援，但固定72.3489/72.5971也有，不独归生成。0076生成77.4092/34.6956、0226生成54.2320/0.9217均强种子分岔；0226对很差的固定5.3111虽+22.2657，仍低于native55.0614。0308平均改善却2313新增末尾未恢复，说明不能只看均值。

本机核对decision自指纹、八表SHA/行数（20/15/270/140/196/196/102/6）、Stage-112来源/运行身份/20模型/外层身份、196→140→15汇总、270配对bootstrap/方向/留一、六种子差、11门槛、调用与392产物身份/SHA格式、102恢复参考数量/时限/丢失计数；28native逐视频指标与Stage-103相符。服务器manifest/系数/NPZ收据未取得，未独立复演完整check_results或逐帧网络；没有本轮精确首败帧，不能沿用旧轮次555等值。

接续优先在现有196轨迹上统一定位首次q0好却错选、首次候选均差、持续失跟及丢失恢复前的供给/输入支持，先零新GPU审计，再按需要预定共同父框/后继分支。不要扩K、挑seed、改原模型或立即追加PPO训练，确认28组继续封存。本次只归档复核，Stage-1～113 Python冻结、未新建Stage-114、未自动commit/push。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE113_RESULTS_REVIEW.md)

### 0.98 2026-11-15 Stage-114 全量缓存错选/供给链诊断（准备记录，结果见0.99）

本轮新增`tools/stage114_cached_failure_chain.py`、`tools/run_stage114_failure_chain.sh`和`tools/test_stage114_cached_failure_chain.py`。绑定Stage-113 decision `11cada1ae2be6ad882acb3f6a55e4a0073ea44dc9957b70dfe1c69a799d93d0c`，覆盖全部28开发源组/196轨迹/203,329帧次、两个评分训练种子，复用四生成包和16共享评分器；Stage-1～113 Python冻结。零图像读取/网络搜索/块调用/训练/新模型/新完整轨迹/确认推理，当前L20或L40分配不影响这项CPU诊断。COMPLETE仅表示诊断执行，无方法采用门槛。

先严格校验源报告/模型/缓存，再从原始NPZ重建父框、特征、原生解码、生成输入、共享评分、动作与自身历史，重现Stage-113 seed_video196、video_summary140、policy_summary15、paired270、recovery102。固定good≥0.5、low<0.2：harmful_q0为q0好却选低；beneficial反向；missed_good/严重漏选区分0.5/0.2；all_low表示全部候选低，连续3帧单列起点。逐轨迹报告首次物理替换、有害q0、严重漏选、all_low/持续all_low、失败及≥32帧长失败。32为诊断长度，不更改原3低/3好事件规则；无效GT分列，物理使用率分母为全部帧，其余质量比例为有效GT帧。

全部非native对照自身视频native，生成Hann另对fixed_hann，生成shared另对同seed fixed_shared，共252前缀。直到首次输出分叉当帧，须核对过去框、q0父框和q0输出完全一致，history_latest仅下一帧允许不同。每统一案例另做一次改提交为当前已观测q0的几何影子，沿用观测并重算资格、下一帧预测输入/父框/裁剪；实际下一帧父框须精确重建。只报告裁剪是否改变/父框像素差，不生成替代未来框、IoU或恢复结论。

完整147失败事件及102参考恢复，前后16帧局部先后、删失标记与所有未匹配恢复均保留，不只选0304；顺序不当唯一因果。诊断记录q0/选中/最佳IoU、真实裁剪覆盖、中心误差、宽高比、raw/Hann峰值、评分差、生成均值/范围。cold第1帧、short_history第2～4帧、from_frame5单列；生成均值模型标准化输入最大值/RMS、评分标准化±10裁剪比例、四维GT状态在条件范围内的经验覆盖仅作描述，不宣称校准OOD概率或闭环90%保证。未读图像，不能断言遮挡等环境原因。

九表：policy_summary15、seed_video196、cases按统一角色、failure_events147、recovery102、prefix252、support_summary按轨迹/阶段/状态、scope196、chain_excerpt去重上下文。另196诊断NPZ及196receipt绑定设计/job/GT指纹，来源只读，已提交诊断复用。decision报告旧725,283搜索的缓存审计和零新搜索。自动TXT打包decision及前八表，chain_excerpt和诊断数组留服务器，保留完整上游来源。

默认根`output/stage114/cached_generated_failure_chain_v1`。同步代码并使用sglatrack-test后：`mkdir -p output/stage114`，再`nohup bash tools/run_stage114_failure_chain.sh > output/stage114/run.log 2>&1 < /dev/null &`。不需要checkpoint或GPU参数；可覆盖STAGE114_ROOT以及STAGE76/89/93/112/113_ROOT。`--pack-only`核验并打包完整结果，不自动commit/push。CPU来源校验仍需原完整服务器产物，不把本机只有TXT说成能直接完成正式诊断。

四项本机测试全部通过，无跳过：分类/无效GT/连续供给不足；替换标签不改动作/模型支持/几何影子；首次分叉索引、相同过去q0及删失顺序；两队列14完整合成轨迹的真实失败/恢复/末尾删失、原五表精确复现、零新图像/网络/训练、来源/模型不变、续跑复用、来源/输出/缓存篡改拒绝。实际仓库算法用于生成/特征/评分/解码/事件，图像网络由合成Observer替代。CLI、Python/Bash语法、LF通过。以上为准备记录；正式196轨迹缓存诊断随后已完成，见0.99，最新完整成绩仍Stage-113。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE114_CACHED_FAILURE_CHAIN.md)

### 0.99 2026-11-15 Stage-114 错选与供给链结果（已完成，尚无新方法收益）

收到`stage114_results_20261115T134707Z_1170301.txt`，1,234,082字节、SHA256 `d7e8f13ccaae31a4a253abb6c2b7b905c738d0b8f24f7b70999e2ef4461aaa82`，逐字节归档results并加-text。decision指纹`cb1665a027fbb6e7db9c5394eb6d14969134c946d3001552ce41c65ff90ae282`，plan `adf563bd15776559b513d9feae373cb62467f39371c69fd2bf2c20e05f962321`；来源Stage-113及完整运行身份相符。196轨迹/203,329帧次、旧725,283搜索，零新图像/搜索/块/训练/模型/完整轨迹/确认推理，COMPLETE只表示诊断完成。九表15/196/1038/147/102/252/855/196/10065行，前八表回传齐全，chain_excerpt按协议留服务器。

主要定位：generated_shared45失败事件，20个起点有≥0.5好框却漏选（其中17好q0被换<0.2），25个无好框（其中14个四框均<0.2），35为≥32帧长事件，其中17个起点有好框。generated_hann26事件中16个同类好q0替换，fixed_shared38事件中12有框漏选/4好q0替换。不同臂单/双种子分母不同，事件不是独立视频。生成共享19/56轨迹共28次严格有害q0替换；36轨迹有失败，其中16在首次失败前/当时有严格替换、18从未有该标签、2首次失败后才有。不能因标签罕见否认风险，也不能把全部供给坏帧归于28次动作。

宏全低候选比例fixed_shared13.1454%→generated_shared28.3756%，实际低IoU13.4298%→29.0353%；原始全低帧10115→18961（同为双种子58,094帧次）。有害q0宏比例0.0505%→0.0883%，不是直接用19/28除总帧所得；实际使用率宏45.50%与前轮24,792/58,094=42.68%是视频等权/帧池化区别。范围及质量随自身状态恶化，没有新AUC。

明确链条：0089/2312第305帧q0=0.644914却选q3=0，306持续全低；0089/2313第471帧q0=0.538922却选q2=0，473首次全低、476持续全低；0304/2313第114帧q0=0.669838却选0，115持续全低；0307/2313第328帧q0=0.848485却选0.089506，329持续全低；0176/2313第129帧0.706004→0，130持续全低。这些帧即各自首次失败/长失败起点。17个此类失跟起点坏框Hann全低于好q0、最大覆盖均1且GT均在条件范围内；不能一律怪预测范围未覆盖，不能由此直接采用Hann，因为生成Hann也失败。

0089/2312在305帧好q0/坏q3评分0.362129/0.404520，Hann0.628524/0.159601；0307/2313好/坏评分只差0.011329。19个首个严格替换案例的单次q0提交几何影子全部改变下一帧裁剪，父框最大坐标/尺寸像素差的案例中位1290.72。0089/305影子差533.28，0304/114为1791.36，0307/328为1723.18；不是目标真实位移。只有替代几何，尚未观测替代未来输出/IoU，不能称已证明保留q0避免失跟。

保留反例：0080/2312、2313首次失败140/142，q0仅0.497648/0.305596却都选0，141/143持续全低；两条从无严格harmful_q0标签，阈值不等于真实风险界线，不后验改定义。0304/2312在458有0.652104→0.199324替换，471才持续失败且当帧q0已最佳但低；不能说471当帧漏选或458唯一原因。0176/2312在158起点全低，到164却已有q3=0.712851而选q0=0，说明供给不足与恢复漏选交替。20有框漏选起点余3个均保留坏q0却漏好q3：0091/2313/91为0.022764对0.608696、0147/2312/1362为0对0.644、0235/2313/894为0对0.559193，不能永远保q0。0307/2312在372最佳q0=0.345841却选0.091756、宽/高比2.90/3.76；0226/2313在43最佳0.360582却选0，44持续全低；仅保护≥0.5的q0会漏风险，扩大规则是否损伤真实救援仍需检验。

丢失参考恢复：生成共享对native14/26未匹配，对同seed fixed_shared18/24未匹配，在各自参考恢复帧全部候选<0.2。分别只有5/14、8/18最大覆盖≥90%，8/14、10/18覆盖0；范围未含GT12/14、15/18。与45失败起点全部最大覆盖≥90%（43个为1）形成状态差别，不能把两个参考相加成32独立事件或宣称恢复支线必有效。

早期与支持：生成共享第2～4帧168帧次，严格有害替换0、选择IoU0.838802、范围覆盖96.43%；frame≥5选择0.542434、覆盖74.47%，28次严格替换全在后段。生成Hann早期84帧次也零严格替换，后段覆盖65.22%。196轨迹均无frame<5首败，但生成共享53/56首次输出差异在frame<5，当帧对native平均仅−0.001791、无严格替换；未排除早期小差异改变路径。共享评分标准化帧内最大值的宏均值3.7355→41.3348，0089/2313整段均值1340.8891是大尾部且可能在失跟后；17严格替换起点只有5超10，不能用简单超界概括全部。评分±10裁剪宏比例0.2364%→0.4613%，无概率校准结论。

本机核对decision自指纹/来源身份、八CSV SHA/行数、196身份/比例/首次位置/调用、15宏均值、855支持行重构单位均值、Stage-113逐轨迹指标/物理使用率桥接、147事件计数/起点、102原恢复精确对应、252前缀身份/下一帧时序、1038统一角色和575去重快照的重复/评分/IoU/标签一致性、392产物身份/SHA格式。案例窗口去重帧数重建10065，与未回传长表声明一致，未验证该长表内容/SHA。服务器manifest/模型/NPZ未取得，未独立执行完整check_results或几何影子/网络；本次只归档复核。

接续优先预定共同起点的实际后继分支，比较原动作与一次q0提交后继续同一冻结策略，同时纳入中等q0被换低和救援对照，再设计公开的质量冲突/几何提交约束。不直接按GT上界采用门控、不扩K/选seed/追加PPO；28确认组仍封存。Stage-1～114 Python冻结，未启动Stage-115或新训练，未自动commit/push。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE114_RESULTS_REVIEW.md)

### 0.100 2026-11-15 Stage-115 一次q0动作的匹配后继实验（准备记录，结果见0.101）

Stage-114只观测了原轨迹和替代提交的几何影子，尚不能证明保留q0避免后续失跟。本轮固定对比同一历史/同一新观测下的原policy与一次q0动作，检验实际未来，并同时保留成功替换坏q0的反面控制。不加训PPO/评分器、不扩K、不按个别视频挑点，28确认源组继续封存。

实现`tools/stage115_q0_continuation.py`、运行器`tools/run_stage115_q0_continuation.sh`、五项测试`tools/test_stage115_q0_continuation.py`及固定协议。绑定Stage-114 decision `cb1665a027fbb6e7db9c5394eb6d14969134c946d3001552ce41c65ff90ae282`与原Stage-113/112完整来源；复用四个外层生成包及对应八个generated4共享评分器，两评分种子2312/2313，SGLA、模板、解码和标准化冻结。

在28开发视频×2种子＝56条generated_shared轨迹各选三个角色的首次出现：好q0≥0.5被换至<0.2、中等0.2≤q0<0.5被换至<0.2、坏q0<0.2成功被替换至≥0.5；均要求实际框与q0不同。168条支持记录保留匹配总数、首次帧与完整32帧资格，末尾不足不缩窗/另挑。受支持去重起点数N由服务器完整缓存计算、首次图像推理前封存，目前未知，不能预填猜测数量。GT用于条件诊断选点，绝不是线上资格或独立确认。

从起点前最近至多五个来源框和上一帧4×24特征/选择/资格恢复共同状态。起点四搜索只实际观测一次，两分支共享；policy沿用当前GPU冻结评分，q0_once仅起点强制q0，之后31帧原规则各自滚动，不再借来源未来。改动同时一致提交框、上一帧所选响应和重算可信资格，是完整一次动作，不是中心/尺寸/资格的纯因子隔离。双方NPZ及收据落盘后才计算这对结果。

主读数`future_mean_iou`排除起点，只平均后31帧，另报当帧/末帧、后继低IoU、供给/漏好和窗口事件。每角色与all_unique_anchors分别按同视频种子内起点→视频内有支持种子→视频等权平均，历史/扩展队列分列。后继IoU差给视频层10,000次条件bootstrap及胜平负，不是全视频AUC或方法采用门槛，不能因起点已按GT选出就将当帧改进作新证据。

首次L20/L40均可，绑定GPU/UUID与原软件；换卡允许单独运行，但同根续跑必须相同身份，否则换`STAGE115_ROOT`。同Stage-113完整运行身份时policy32帧除计时外所有来源字段必须精确复现；换卡时报告框/候选/特征误差、动作数及旧/新角色支持，原封存起点不按新结果重选。记录两分支的首次框/父框/裁剪分叉；来源32帧差可能含反馈放大。

账本：N公共点+2N短分支，64N分支帧曝光；最多252N新物理搜索、1,764N搜索块，3N模板初始化分开报。分支首行new_executed=False，公共点只收费一次，精确重复浮点父框去重；6N个NPZ/receipt，已提交缓存续跑复用，故障重试不计入唯一产物账本。九表sampling_scope/anchors/scope/branches/contrasts/summary/paired/bridge/branch_frames；自动TXT打包decision及前八表，逐帧表和缓存留服务器。

默认`output/stage115/matched_one_q0_continuation_v1`，需要原数据/checkpoint及Stage-76/89/93/112/113/114完整服务器产物。在sglatrack-test同步代码后执行`mkdir -p output/stage115`，再`nohup bash tools/run_stage115_q0_continuation.sh > output/stage115/run.log 2>&1 < /dev/null &`；`--pack-only`核验打包，不自动commit/push。首次换卡不需先询问或改代码，同根中途换卡要新路径。

五项本机测试全部通过，无跳过：三角色及末尾窗口支持；共同点观测、单次动作/后继分叉、GT换标签不改动作、policy精确复现与调用掩码；排除起点/层次平均；首次换卡及续跑身份；两队列四上下文八短分支完整合成流程、落盘先于成对评价、真实调用账本、来源模型不变、零重搜续跑及篡改拒绝。生成/评分/解码/统计用实际仓库实现，图像网络由合成Observer替代。Python/CLI/Bash语法及LF通过，Stage-1～114 Python未改。以上为准备时本机检查；正式GPU分支随后已回传，见0.101，最新完整成绩仍Stage-113。 [完整协议](docs/experiments/06_sgla_uav_tracking/STAGE115_MATCHED_Q0_CONTINUATION.md)

### 0.101 2026-11-15 Stage-115 一次q0选择的实际后继结果（已完成，不能统一保q0）

收到`stage115_results_20261115T154035Z_1212926.txt`，103,609字节，SHA256 `cd55003ee016f9833d6e841aa1b9c27be2444615a8f0552cae4f614f6636fb18`，逐字节归档results并加-text。decision指纹`141baa4bae77280cf89cf063665b8b4fa03545ba3ae3c98e52ce0137f0b90edc`，plan `696f9b78cd8c8c02b2a010040ab6c2bfe5f1899f424dcb90cfb3008124e37a62`，来源Stage-114相符。状态COMPLETE_MATCHED_ONE_Q0_ACTION_CONTINUATION表示诊断完成，不是新方法PASS。

56条generated_shared来源/28开发视频统一扫描，168角色支持行：132无该角色、36有首次帧，2个末尾不足窗口，实际34起点/18视频/28条种子轨迹、68条32帧匹配分支/2176帧次。好q0被换低19起点/13视频，中等q0被换低7/5，原成功救援8/6；角色视频有交叉。原匹配帧总数28/10/10，非全部逐帧分支。未运行的2个首次角色为0089/2312中等1357（来源1380）和0308/2313救援433（来源460），不缩窗或后验另挑。历史队列11起点/7视频，扩展23/11，均已消耗开发证据，确认28组仍封存。

L20 UUID `bd2283f0-25e4-2734-53c1-e3a243ca482c`，完整运行身份与Stage-113一致，runtime_differences=[]。34个policy控制桥接框/候选/特征差及动作变化均0，服务器代码要求除计时外全部字段精确复现。全部起点角色保留，两分支首框都在起点不同、父框与裁剪均在下一帧首次不同。实际8568新搜索/59976搜索块，34×4公共点＋68×124后继，各公共点只计一次；另102模板初始化。新训练/模型/完整轨迹/确认推理均0，权重冻结。

主要差值q0_once−policy，均排除起点、只看后31帧，先种子内起点→视频内有支持种子→视频等权：好q0组IoU0.097310→0.315195、+0.217885，条件CI[+0.119802,+0.310545]，10胜1平2负，低IoU−35.4839个百分点；中等组0.014681→0.175691、+0.161010，CI[+0.042648,+0.279372]，4/0/1，低IoU−36.7742个百分点；原救援0.634385→0.304051、−0.330334，CI[−0.613240,−0.080428]，0/0/6，低IoU+42.2043个百分点。全体混合+0.080210，CI[−0.045078,+0.189579]，11/1/6；GT角色抽样的混合均值不是任意部署帧收益，更不是AUC。

好q0组后继好候选比例16.75%→42.43%，中等1.29%→8.71%，原救援80.65%→41.13%，支持选择/提交反馈确实改变后继供给。历史/扩展开发队列方向一致：好q0+0.283692/+0.176755，中等+0.178656/+0.134541，救援−0.456765/−0.267119；扩展中等仅2视频，区间跨零。19好q0起点为15改善/2平/2下降，7中等6改善/1下降，8救援1微增/1平/6下降；勿把视频胜负改成逐起点胜负。所有区间均条件于已消耗源组/冻结模型/GT选点，不是独立确认。

正例：0089/2313/471后继均值0→0.577912、末帧0→0.636943，低IoU31→0帧、好供给2→31帧；0304/2312/458为0.121921→0.722728、末帧0→0.925926，好供给0→27帧；0174/2313/745为0→0.728244、末帧0→0.837838；0307/2313/328为0→0.388667、末帧0→0.416298，改善但未稳定达到≥0.5。0080/2312/140为0→0.539730但末帧仍0，只是窗口改善。

重要反例：0070/2312/25当帧修正+0.756105，未来0.526815→0.160370、末帧0.573144→0；0085/2313/1446当帧修正+0.565077，未来0.110581→0。0176/2313/129起点好q0=0.706004，两分支未来均0，0349/2313/85也均0；0304/2313/114仅0.000769→0.022444且末帧相同。好q0组处理后宏低IoU仍49.50%，19窗口末帧11个<0.2、10个0，仅7个≥0.5。只纠正一次当前错框不足以保证后续可靠，没有逐帧表不得杜撰再次失效的动作。

救援代价：0091/2312/91未来0.704031→0.001837、末帧0.711230→0；0237/2312/524未来0.874768→0、末帧0.855919→0，必须允许必要替换。原当帧救援也不全持续：0089/2313/719原当帧0.907193但两条未来均0；0070/2312/79强制q0未来微增0.000437，不后验删除。这是一次完整动作（框、上一选择响应、可信资格一起一致反馈），没有分离纯中心/尺寸或环境遮挡/相似物原因。

九表168/34/68/68/34/24/12/34/2176行，前八表回传齐，branch_frames及102NPZ/102receipt按协议留服务器。本机已核对decision自指纹/来源与运行身份、八CSV SHA/行数、168支持/34起点/68分支身份和31帧比例算术、逐对差/24宏均值/12bootstrap和胜负、独立层次平均、34零漂移桥接/调用账本/204产物名及SHA格式，以及26个在Stage-114有同帧快照的q0/实际IoU和动作。未取得manifest/模型/NPZ，不称独立执行完整check_results或网络，未验证未回传逐帧内容/SHA。

新增证据是局部实际动作/反馈的后继作用，不是已可上线的q0保护。接续优先统一审计公开可用的q0/替代响应与外观证据、几何冲突、过去可靠性和预测不确定性，区分保留获益/保留有害/救援必需/两动作均坏；若训练风险或后继效用，应在fit源组另采代表性匹配分支并纳入救援，守住源组留出/同曝光控制/冻结后完整轨迹保护。当前34个GT选点不能训练后自验，不后验调间隔、统一保q0或立即加训旧PPO。本次只归档复核，Stage-1～115 Python冻结，未启动Stage-116，最新完整成绩仍Stage-113。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE115_RESULTS_REVIEW.md)

### 0.102 2026-11-16 Stage-116 在线证据与匹配后继收益审计（准备记录，结果见0.103）

接续Stage-115的局部收益和救援损伤，先检验现有公开证据是否能区分保护机会，再开发风险约束；不直接统一回q0、加训旧PPO或在34个GT选点上训练后自验。实现`tools/stage116_online_evidence.py`、运行器`tools/run_stage116_online_evidence.sh`、四项测试及固定协议。Stage-1～115 Python冻结，已有未提交的Stage-115归档/复核文档改动保留，不自动commit/push。

绑定Stage-115 decision `141baa4bae77280cf89cf063665b8b4fa03545ba3ae3c98e52ce0137f0b90edc`及Stage-113/112冻结指纹。第一部分全56条generated_shared/28开发视频/58,094状态做单帧影子规则；第二部分全部34起点/68条实际分支，公共点证据决定是否采用已测q0_once后31帧。没有新SGLA推理、图像解码、模型/训练/完整轨迹/确认；冻结生成器/评分器的数值重放是CPU计算。28确认组仍封存，本机缺正式完整来源，服务器审计待运行。

在线API白名单仅当前四框、4×24特征、冻结分数/选择、预测均值/半径/激活、过去至多五预测框及冻结标准化；不接受GT、未来标签、角色、视频身份或来源未来帧。九项固定方向证据：q0对原选择的Hann峰/原峰/PSR优势、低熵优势、历史速度连续性、历史中位log尺寸稳定性、预测半径兼容性、去除精确重复框的几何佐证、原共享评分优势；另4量描述中心/尺寸冲突、半径、标准化超界比例。没有新外观描述子，不声称已测外观相似度。预测兼容与多框一致可共同出错，不是独立可靠性概率。

固定五规则：policy不改；q0_all在所有物理替换时保q0；hann_pair在Hann优势>1e−12保护；continuity_pair在历史外推中心距离优势>1e−12保护；corroborated要求Hann/原峰/中心连续性三项同时为正。平局沿用原选择，序号不同但框相同不算物理替换。无阈值搜索，不直接重选四候选Hann最大框。五规则全量报告保护率、当前影子IoU、拦截坏替换及阻断救援数量，不能写成规则自身完整AUC或失跟预防率。

匹配部分加三项同率期望控制及两动作future_oracle上界，共九策略。同率保护比例先在同视频/评分种子的全部可用起点固定，再按角色分组，不根据子组标签重算。34起点仅28条种子轨迹，多数上下文单起点；单点或全保/全不保时同率控制必等原规则，paired明确rate_informative_contexts/anchors（概率介于0和1）以示实际区分空间，不将必然平局作强证据。oracle使用后继GT只作两动作上界。

后继标签按两分支未来平均：两者<0.2优先both_low；否则正差q0_better/负差policy_better/平差tie。连续差仍保留，both_low非零差也纳入九证据固定方向的后继符号加权AUROC；只排除平局，无正/负类返回空，不后验翻方向或挑特征。策略后继均值/低IoU/保护率/实际增益/上界遗憾，按种子内起点→视频内种子→视频等权，历史/扩展分列；三候选规则对policy、q0_all、各自同率控制全部报告视频层10,000次条件bootstrap，seed23160，不作多比较校正或方法采用PASS。没有新独立评价证据。

输出56诊断NPZ/56receipt。九表预计population_scope280、population_summary15、anchor_evidence34、evidence_summary195、discrimination27、anchor_decisions306、policy_summary108、paired108、video_effects756；自动TXT含decision及除anchor_decisions/video_effects之外七表，明细/缓存留服务器。来源/代码/产物SHA与上下文/帧账本绑定，完整缓存续跑复用，来源和结果篡改拒绝。

服务器sglatrack-test同步本轮代码后：`mkdir -p output/stage116`，再`nohup bash tools/run_stage116_online_evidence.sh > output/stage116/run.log 2>&1 < /dev/null &`。无需checkpoint或GPU参数，当前卡L20/L40不影响这次CPU缓存审计；原软件/来源仍需完整。默认`output/stage116/cached_online_q0_evidence_v1`；可覆盖STAGE116_ROOT及Stage-76/89/93/112/113/115_ROOT，Stage-115用过另设路径时务必沿用。完成自动TXT，`--pack-only`核验/打包。

四项本机测试：权重/平局/缺类AUROC与四标签；固定比较/无物理替换/同率先于角色分组；白名单与GT替换不改证据/决定、非法半径；两队列四上下文真实生成/评分/解码和匹配缓存完整合成流程，零新图像/网络/训练、模型/来源不变、复用和篡改拒绝。Python/CLI/Bash语法及LF检查；合成图像网络不是正式SGLA GPU复现。以上为准备记录，正式审计随后已回传，见0.103，完整开发成绩仍Stage-113。 [完整协议与命令](docs/experiments/06_sgla_uav_tracking/STAGE116_CACHED_ONLINE_EVIDENCE.md)

### 0.103 2026-11-16 Stage-116 在线证据审计结果（已完成，判据尚未建立）

收到`stage116_results_20261116T055314Z_1421093.txt`，127,386字节、SHA256 `e37a3066312577349ced1931f8338afc8d3ae761f4e977447abcadd2c1b71376`，逐字节归档results并加-text。decision指纹`e15637be797024fd188b10dd2f02783ba6b51f381b5e9e81c35b687ea71d2075`，plan `a9403d492f3448ca332573154c3878a916045829296028083beeddac34675685`，来源Stage-115固定指纹相符。COMPLETE_CACHED_ONLINE_Q0_EVIDENCE仅表示诊断完成。

全56条generated_shared/28开发视频/58,094状态和原34起点完成；新增图像/SGLA搜索/块/训练/模型/完整轨迹/确认推理均0。九表280/15/34/195/27/306/108/108/756行；七表按计划回传，anchor_decisions/video_effects与56NPZ/56receipt留服务器。34点是原GT角色选出的已消耗开发证据，确认28组仍封存。

全量单帧影子：原policy宏当前IoU0.544003、物理替换24792帧。q0_all保护24792、拦严格有害28/28、阻断当帧救援10/10；Hann保护23863、拦28/28、阻9/10；连续性13627、拦27/28、阻9/10；联合9875、拦18/28、阻3/10。Hann覆盖96.25%原替换，近统一回退，不能将拦全部错误称可靠识错。当前均值仅0.545167/0.545165/0.544988/0.544698；影子未改变来源未来，不是完整AUC或实际防失跟率。宏保护比例45.50/44.17/24.80/17.33%，不等于原始帧数池化比例。

真实匹配后31帧宏IoU：policy0.180024，q0_all0.260234，Hann0.261311，连续性0.299838，联合0.260356，上界0.346628。Hann−policy+0.081287、条件CI[−0.051184,+0.189807]，11胜1平6负；连续性+0.119814、CI[+0.032915,+0.201963]，12/2/4；联合+0.080332、CI[−0.010703,+0.163154]，10/5/3。区间未校多比较，不能据此宣布方法通过。连续性未来低IoU仍53.14%，两动作GT上界也46.91%。历史/扩展连续性对原+0.121205/+0.118929，非独立外部验证。

Hann保护33/34起点，仅放行0226/2312/337的救援，后继0.894673而非q0的0.855912；对q0_all仅+0.001077、1胜17平。连续性保护32/34，只额外放行0076/2312/638（避免0.068593→0.047231的小损伤）和0091/2312/91（保0.704031而非q0的0.001837大救援），对q0_all+0.039604、2胜16平、CI[0,+0.118219]。其余32点完全相同，存在具体进展，尚非广泛判别。联合保护19/34，宏保护54.17%不等于19/34池化率。

救援约束仍未过：原8救援起点/6视频后继0.634385，Hann/连续性/联合仅0.310512/0.421084/0.458562；连续性对原−0.213302、0胜1平5负。好q0角色中Hann/连续性均全保，未来0.315195，仍未排除0070/2312/25等保好当前框却伤未来；联合该角色仅0.238259。不能用混合均值掩盖救援与选择反例。

同率控制：Hann/连续性informative_contexts/anchors均0/0，各原上下文全保或全不保，同率期望必等自身；18视频全平不能证明或否定选择价值。联合3上下文/6点：0089/2312正确保护305（收益+.221684）、放行1301（−.026680）；0235/2313错放697（+.023599）、错保719（−.180175）；0237/2312错放413（+.398613）、错保524（−.874768）。联合对同率−0.018791、CI[−0.056993,+0.003449]、1胜15平2负；历史/扩展−0.045478/−0.001809。小支持下未显示同率筛选增值，不作普遍不可能性结论。

后继标签q0_better16/policy_better6/both_low12/tie0；both_low是两分支未来均值<0.2，其中3个差值平局，故符号AUROC仍排除3个，剩31点/17视频、正22负9。九项固定方向加权AUROC：Hann.705693、原峰.529538、PSR.668099、熵.586466、连续性.776584、尺寸稳定.757250、预测兼容.830290、几何佐证.588614、原共享评分.359828。预测兼容历史/扩展.900000/.788506，但历史仅10非平点/2负例；均为描述性线索，无概率校准/新拟合/独立测试，不在此翻方向或挑阈值。

预测兼容在16个q0_better全正，但6个policy_better有3个正、12个both_low有11个正；连续性在policy_better有5/6正。0070/2312/25的Hann/连续性/预测优势全正，未来仍0.526815→0.160370；0174/2313/745原峰差−.056使联合错过0→.728244；0237/2312/524联合三优势全正却毁救援，预测优势−9.9827是另一待验证线索，未据此追加门控。没有新外观描述子或环境遮挡/相似物归因。

本机已核对自指纹/来源、七CSV SHA/行数、56上下文帧账本和保护计数/15宏汇总、34点与Stage-115当前/后继/低IoU精确桥接、固定规则/四标签/195汇总/27 AUC/108策略和bootstrap/同率支持；AUC另用成对比较复算。重建未回传306/756明细并按原格式序列化，SHA均与decision一致。全56上下文当前/q0均值和计数对Stage-114逐轨迹一致，26起点旧快照Hann/原峰/评分差相符，112产物名/SHA格式相符。没有manifest/模型/NPZ，不称独立全帧提取、网络或完整check_results。

接续把连续性保留为固定控制，把预测兼容等作为待检验输入，在fit源组另采代表性、无GT选点的匹配分支，覆盖普通替换/必要救援/两动作都坏，并扩大同上下文的可辨支持。源组隔离拟合/校准与冻结评价，比较同率、统一q0、固定连续性，再做连续部署完整轨迹；不在34点上继续调阈值或直接加训旧PPO。本次仅归档复核，Stage-1～116 Python冻结，未启动Stage-117，无新完整AUC或采用结论。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE116_RESULTS_REVIEW.md)

### 0.104 2026-11-16 Stage-117 无GT值选点匹配q0后继实验（准备记录，结果见0.105）

Stage-116的34点按GT条件选出，Hann/连续性同率无可辨支持。本轮补充剩余普通替换状态的实际后继，不在34点上调阈值。实现`tools/stage117_sampled_q0.py`、运行器`tools/run_stage117_sampled_q0.sh`、五项测试及固定协议。已有Stage-116归档/复核文档改动保留，Stage-1～116 Python不改，不自动commit/push。

绑定Stage-116 decision `e15637be797024fd188b10dd2f02783ba6b51f381b5e9e81c35b687ea71d2075`及Stage-115/113/112固定指纹，使用56条generated_shared/28开发源组、评分种子2312/2313，所有生成器、评分器、SGLA、解码和首帧模板冻结，28确认源组继续封存。抽样函数只接受frame_id、physical、13维公开证据与已用起点身份，不读GT/IoU/失败/有效GT位/后继标签。

母体为原实际输出与q0不同、frame≥5、可完整观测32帧、未与同视频任一种子Stage-115已用起点帧号相同的状态。只排除确切旧起点，不排除所有相邻/重叠窗口。它是剩余物理替换状态，不是全部视频帧、新独立数据或新确认；救援标签事后列出，不保证抽到多少，不按结果补抽。

两套独立PCG64流（sampler_seed23170，SHA按上下文/流导出种子）：uniform每上下文均匀8点，少则全取、权重1/实际抽数；stratified按来源Hann/连续性/预测兼容性优势>1e−12形成8层，位1/2/4，每非空层1点，权重层数/母体数。交集只运行一次，保留两套membership/权重，不合并同权报告。504条抽样记录保留空层/空上下文、排除数/母体/ID/概率；N推理前封存，尚未知，去重上限896、两套membership总数亦≤896。

每点复用Stage-115已实测point/collect/audit：恢复原前缀最近五框与上一响应/选择/资格，共同新观测一次四候选；policy沿原选择，q0_once仅起点强制q0，后31帧各自滚动。框、选择响应、资格一致反馈，双方分支/收据保存后评价，不借用来源未来。不是纯中心/尺寸因子或连续部署规则。

规则沿用policy/q0_all/Hann/连续性/联合佐证，新增forecast_pair（预测兼容优势>1e−12保q0），只在当前框物理不同才干预。主比较预定uniform/all的forecast_pair−continuity_pair，四候选规则还对原policy、q0_all和各自同率控制，未调阈值或新拟合。各研究上下文内先算分析权重下的保护率，再给同期望率控制；uniform可解释同数量随机保护期望，stratified仅同总体加权期望率，不声称每次加权数量恰等。加future_oracle共11策略，明确informative_contexts/anchors。

指标只看后31帧均值，单列当帧/末帧/低IoU、实际增益/上界遗憾；按研究内权重→同视频种子→视频等权，历史/扩展开发分列，视频层10,000次条件bootstrap、seed23171、不校多比较、不判方法PASS。支持表事后列好q0被换低/中等被换低/救援/其他及正平负差/both_low。新后继不是新源组；若以后训练门控，还须核对上游生成/评分与训练组嵌套，不能将这些OOF行任意混折声称严格无泄漏。本轮不训练也不设新fit/held集。

首次GPU L20/L40均可单独绑定；完整身份同来源时policy32帧除计时外精确复现。首次换卡仍固定原抽样身份/层/权重，报告公开证据/层/来源输出和动作漂移、干预是否无实际变化，不重选或删点；同根续跑必须相同GPU/UUID/软件，否则新STAGE117_ROOT。软件/权重/配置变更拒绝。

N公共点、2N短分支、64N帧次，最多252N新搜索/1764N搜索块，最大225792搜索/1580544块；3N模板初始化另计，公共点只计费一次、重复父框去重。6N缓存/收据，已提交续跑复用，故障重试额外成本不在唯一产物账本。十一表sampling_scope/anchors/scope/branches/samples/sample_decisions/support/policy_summary/paired/bridge/branch_frames；自动TXT含decision及sampling_scope/samples/support/policy_summary/paired/scope/bridge七表，其余明细留服务器。

服务器sglatrack-test同步后：`mkdir -p output/stage117`，再`nohup bash tools/run_stage117_sampled_q0.sh > output/stage117/run.log 2>&1 < /dev/null &`。默认`output/stage117/gt_free_sampled_q0_continuation_v1`，支持STAGE117_GPU、CHECKPOINT及七个来源根Stage-76/89/93/112/113/115/116。原图像/checkpoint/完整来源仍需；完成自动TXT，`--pack-only`核验打包。不要把上限说成实际已完成数量或估算运行时长。

五项本机测试全部通过：确定性均匀/分层抽样、空池/完整窗口/剔除/交集/独立权重；GT/收益/有效位不改抽样、同视频跨种子排除；加权同率/主对照算术；首次换卡/续跑身份；两队列四上下文真实生成/评分/解码及匹配后继合成流程、收据先标签、控制精确、物理调用账本/模型来源不变/零重搜续跑和篡改拒绝。Python/CLI/Bash语法及LF检查；合成Observer不等于正式GPU结果。以上为准备记录，正式GPU测量随后已回传，见0.105，完整成绩仍Stage-113。 [完整协议与命令](docs/experiments/06_sgla_uav_tracking/STAGE117_GT_FREE_SAMPLED_Q0.md)

### 0.105 2026-11-16 Stage-117 无GT值抽样结果（已完成，主比较未建立收益）

收到`stage117_results_20261116T071645Z_1437692.txt`，1,499,252字节，SHA256 `c377f533b2337e3b881f69e75dcdb358a1144aebe264a827a84934191ab9946b`，逐字节归档results并加-text。decision `59f645ac852adc35080cf94dbce9dd8708b4118ee27d8a6bad879ee3e476d3cc`，plan `7deb1239b1b8cbc5da83623c98a934876925da28a836c75aedaaf9f8b635034c`，来源Stage-116固定指纹相符。COMPLETE_GT_FREE_SAMPLED_Q0_CONTINUATION只表示固定测量完成。

56上下文/28开发视频中，完整窗口且frame≥5的物理替换24055帧，剔除同视频跨种子已用起点42，剩24013、各上下文37～1768。uniform每上下文8点，共448；stratified309非空符号层各1点；交集13，去重744（历史323/扩展421），757是membership曝光而非独立起点。两研究权重分开，不能择优混报。28确认源组仍未用。

L20 UUID `bd2283f0-25e4-2734-53c1-e3a243ca482c`，完整身份同Stage-113，runtime_differences=[]。744个policy桥接框/候选/特征差及动作变化全0，公开证据差0、来源/当前层全同；744均实际改起点框，父框/裁剪首次差异均下一帧。1488条32帧分支/47616帧次，744公共点。187488新搜索/1312416块达到N上限，2232模板初始化另计；新训练/模型/完整轨迹/确认0。2232NPZ/2232receipt留服务器。

主比较uniform/all forecast_pair−continuity_pair为−0.0019301829，条件CI[−0.0064611739,+0.0015889400]，13胜1平14负；历史/扩展−0.003646/−0.000643，均跨零。stratified对应−0.000237、CI[−0.004683,+0.003671]、15胜13负；历史−0.003419、扩展+.002149，均跨零，不能替代主结果。Stage-116选点上的预测排序线索没有建立本轮普通状态零阈值规则优势，不等于预测信息普遍无用。

后31帧宏IoU uniform：policy0.570277、q0_all0.568618、Hann0.568607、连续性0.570739、联合0.570935、预测0.568809、两动作上界0.576528；stratified：0.566558/0.568417/0.568609/0.569603/0.569079/0.569366/0.573539。没有新完整AUC，policy是生成共享策略而非原生SGLA。普通剩余母体上的一次动作上界对原仅+.006251/+.006981，不是其他动作/长后继/罕见触发或完整方法的上界。

uniform预测−原−.001468、CI[−.006536,+.003179]；对同率−.001061、CI[−.003164,+.000657]、11胜1平16负，有效53上下文/424点。连续性对原+.000462、对同率+.000854（CI[−.001295,+.003242]，15/1/12），联合对原+.000658、对同率+.000509（CI[−.002239,+.003448]），均未建立稳定优势。两者同率有效支持均56/448，已非Stage-116的不可辨0支持。分层预测对同率+.002046、CI[−.001077,+.006162]、17胜11负，56/309支持；均值方向与uniform相反，不能择优宣布选择价值。

支持：uniform448全other，正209/平85/负154，both_low112；分层309中2个good_q0_lost、307other，正110/平89/负110，both_low110；两研究均没有medium_q0_lost或q0_rescue。分层计数不是加权总体比例，交集不能当新样本。严格救援覆盖限制不是单纯抽样偶然：来源10个q0<.2且原选择≥.5的帧中，8个支持Stage-115旧救援起点已剔除，0308/2313/433末尾不足32帧不合格，剩余合格母体最多1个。不能据本轮没有救援断言不损伤恢复；也不应靠盲增均匀配额追已排除的救援。该上界只针对严格当帧定义，非全部参考恢复。

当前两框都好也可大伤后继：0204/2313/661当前原/q0=.734891/.761866，未来.754091→.052623（−.701468），预测优势+.013575触发保护、连续性−.031385放行；0070/2313/69当前.776819/.794227，未来.726336→.176925（−.549410），预测+.013646而连续性−.014955。主比较探索性视频贡献0204−.042785、0070−.035770，被0085+.015524、0235+.011752等部分抵消。不能只按当前好框判断，也不据这些数后验加阈值。

分层两严重替换反例：0070/2313/106原当前0、q0=.865747，但未来.400171→0；0089/2312/1303原0、q0=.8，未来0→.724713。连续性两者都大正，预测优势−.031599/+.556775，保留条件线索和反例，不只报有利窗口。它们可能邻近/重叠旧窗口，不是独立新源组。

十一表504/744/1488/1488/744/8327/30/66/90/744/47616行；七表按计划回传，另外anchors/branches/sample_decisions/branch_frames留服务器。本机核对自指纹/来源/完整身份、七CSV SHA/行数、504身份/种子/概率/层质量/剔除/窗口、744去重/交集/独立权重/符号层、标签和31帧比例、所有规则、30支持/66均值/90bootstrap/主结果和同率支持；独立层→种子→视频权重、744零桥接/1488账本/4464产物名SHA格式相符。由账本重建744 anchors、由samples重建8327 sample_decisions（按顺序求和语义），其SHA均与声明一致；统计跨平台差约1e−17，容差核对。未取得manifest/完整母体/模型/NPZ，不称独立重放全母体随机抽样、全帧特征、网络或check_results；未独立验证未回传branches/branch_frames内容SHA。

接续应明确普通状态误干预成本与必要救援压力目标，保留普通无GT评估和救援压力队列分别解释；若学条件风险/后继效用，先解决上游提供者与源组划分嵌套。不要继续只按34点调阈值、直接部署预测规则、加训旧PPO，或用更多已排除救援的均匀抽样替代恢复保护。本次只归档复核，Stage-1～117 Python冻结，未启动Stage-118；最新完整成绩仍Stage-113，无新AUC、稳定/恢复保护通过或独立确认结论。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE117_RESULTS_REVIEW.md)

### 0.106 2026-11-16 Stage-118 提供者隔离提交收益学习（准备记录，结果见0.107）

用户要求继续下一步。沿Stage-117结果推进学习提交机制，检验相同公开证据与容量下，后继监督能否超过当帧监督，同时保留普通误干预成本与必要救援压力。实现`tools/stage118_nested_commit.py`、运行器`tools/run_stage118_nested_commit.sh`、六项测试及固定协议；保留原Stage-117归档／复核改动，Stage-1～117 Python冻结，未自动commit/push。

冻结Stage-117 decision `59f645ac852adc35080cf94dbce9dd8708b4118ee27d8a6bad879ee3e476d3cc`，连同Stage-116/115/113/112的manifest、完整缓存／模型、表和代码SHA核验。无图像、GPU、SGLA搜索或新完整轨迹。**有新的CPU监督拟合**，不能说无训练：28包内56岭头＋56常数控制=112小模型，0优化器步、0SGLA更新。L20/L40均可，无GPU或CHECKPOINT参数；首次运行绑定Python/NumPy环境，部分续跑环境变化另设根。

关键嵌套划分：Stage-112每提供者train21/held7。对组g，只从其同一provider-held7中的另外6组拟合，不混用其他三个提供者生成的训练行，因为这些提供者可能在g上训练过。每头96均匀例=6视频×2评分种子×8点，g全视频／种子标签不进拟合或标准化，pressure所有标签从所有拟合排除。28组各有一个独立留组模型包；6个拟合源组较少，限制学习能力，不能把失败扩大为所有时序学习无效。这里只保证指定上游生成／评分模型与提交头的组隔离，仍是已消耗开发，不称全项目独立确认。

13维输入沿Stage-116，白名单是Hann/raw/PSR/熵/中心连续性/尺寸稳定/预测兼容/候选一致/共享分数九优势与center_conflict/log_size_conflict/radius_max/scorer_clip_fraction。myopic目标=当前q0−policy IoU；future目标=已测一次q0提交对原policy的后31帧平均IoU差。相同训练例和视频→种子→点权重、训练内标准化、clip±10、ridge1、截距不惩罚，正规方程求解；各目标另有同拟合标签均值常数控制。分数>1e−12保q0，否则原选择。不搜索阈值、超参或checkpoint；得分不是校准概率，不是PPO训练。

评价复用778去重起点=Stage-117的744＋旧34压力；791个membership=uniform448＋stratified309＋pressure34（普通交集13不重复测）。主训练仅uniform，分层保留总体层质量权重；uniform与stratified有交集，不能称相互独立测试集。压力34全部只评，不与普通混合；all/historical/expanded，压力三类角色各报。原严格救援主要在旧压力集中，不能用普通支持0掩盖误伤。

13策略=旧6规则＋myopic/future及两常数＋两同率期望＋future_oracle。主比较ordinary_uniform/all/all的future−myopic后31帧IoU；另对policy、q0_all、continuity_pair、future_constant、future同率。按研究／视频／种子原权重先算同率，再做角色切片，报告可辨支持；policy仍是generated_shared，不是native。报告低IoU、保护率、实现收益、有害成本、漏益、遗憾、损失<-0.1起点数／最差收益、中位／最差视频／最小留一差、胜平负与视频层10k条件bootstrap(seed23180)，不重训bootstrap，未经多重校正，不称无条件确认区间。

28 JSON模型包＋28预测NPZ＋28收据=84产物。预测仅公开x和四分数／元数据，全部保存后才评价；断点复用、hash锁定。九表fold_scope/model_scope/fit_metrics/sample_predictions/policy_decisions/policy_summary/paired/support/video_effects；112模型作用域、56训练指标、778预测、10283逐决策。自动TXT含decision与七表，逐决策和逐视频表留服务器。COMPLETE只表示计算完成，无采用gate、无新完整AUC。若普通改善仍伤原救援不能通过；后续还需完整自身轨迹及封存确认。

六项本机测试通过：上游分区／训练排除／混折拒绝；待评组、其他上游折和全部压力标签替换不改训练包与预测，训练标签改变会改签名；已知信号岭解／相同容量；独立权重、同率期望、压力分离与遗憾分解；CSV归属／标签范围／权重拒绝；六组合成完整流程、先保存预测、零图像／SGLA／上游拟合、来源不变、零重拟合续跑／篡改拒绝。另对归档实际744＋34行做输入schema/范围/身份检查，778独立ID、791membership、448拟合候选相符；未在缺完整来源的本机拟合正式模型。

服务器同步后：`mkdir -p output/stage118`，再`nohup bash tools/run_stage118_nested_commit.sh > output/stage118/run.log 2>&1 < /dev/null &`。默认根`output/stage118/provider_isolated_commit_learning_v1`；支持STAGE112/113/115/116/117_ROOT和STAGE118_ROOT覆盖。`--pack-only`核验再打包；完成TXT在结果根。来源完整核验可读取旧NPZ，但无读图／网络推理。以上为准备时记录，正式结果现已回传，见0.107；最新完整方法成绩仍Stage-113。 [完整协议与运行命令](docs/experiments/06_sgla_uav_tracking/STAGE118_PROVIDER_ISOLATED_COMMIT_LEARNING.md)

### 0.107 2026-11-16 Stage-118 提交收益学习结果（已完成，future主比较更差，仍伤救援）

收到`stage118_results_20261116T081210Z_1492336.txt`，676662字节，SHA256 `924efabcfa5b1a6cc784cef84ce95176b2d6d3856b0ba5dda89d598b1a5f060f`，逐字节归档results并加-text。decision `8efa6dcf197a9249a3b034f5a70eba0f5c80b5eeeb4cbcccb2b2ee216ecede64`，plan `68d5b3f34dec69a5eed0d80a60be9d4114071217ee9507271034d3db2591a4be`，来源Stage-117固定指纹正确。COMPLETE_PROVIDER_ISOLATED_COMMIT_LEARNING仅表示固定CPU计算完成。

新增56岭头＋56常数=112模型，28包／28预测NPZ／28receipt=84产物。上游4提供者train21/held7；每待评组g仅用同provider-held7其余6组96个uniform例，g全视频/种子和所有压力点均不拟合/归一化。778去重点=744普通＋34压力；791membership=uniform448＋stratified309＋pressure34（普通交集13）。无新图像/GPU搜索/SGLA更新/完整轨迹/确认推理；有CPU拟合，不称无训练。

预定uniform/all future−myopic后31帧IoU差−.0030969871，CI[−.0063986451,−.0003312782]，8胜1平19负；中位−.0007624、最差视频−.0297047、最小留一−.0037248。历史/扩展−.002504/−.003541，前者跨零后者全负。均匀后继IoU policy .570277、q0_all .568618、连续性 .570739、myopic .569844、future .566747、future常数 .567057、future同率 .567975、上界 .576528。future对原−.003530（CI全负）、连续性−.003992（CI跨零）、常数−.000310、同率−.001228（后两跨零）；同率有效41上下文/328点。不能把主要负结果写成“有提高但尚不显著”。

分层future−myopic+.000599，CI[−.007098,+.009721]，12胜1平15负；对原+.001491、同率+.001369，区间均跨零，最小留一主差−.003061。分层辅助结果不替代主研究或择优采用，普通无严格救援支持不变。

均匀保护率future41.29%／myopic75.45%／连续性58.48%。future有害成本.005067比myopic .005570略低，但漏益.004715远高于.001115、遗憾.009781高于.006684；净效用−.003530，低IoU26.36%高于原25.94%。future损失<−.1起点5个（myopic7、统一q0为9），最差仍−.701468，不因数量少就称安全。0204/2313/661原/q0后继.754091/.052623，future预测+.002905仍错放；0070/2313/69真实−.549410、future预测+.000245但myopic拒绝。0147/2312/1354真实收益+.490282、future预测−.002274而漏益，0308/2313/409及411也漏+.246765/+.298039；不同错误并存，不能后验调阈值只修一例。

压力34/18 future后继.215059，低于myopic .238222、连续性.299838；好q0被换低19/13为.228408对原.097310、连续性.315195，中等7/5为.115915对原.014681。原救援8/6：policy .634385、连续性.421084、myopic .452033、future .508706；future对原−.125679，CI[−.357766,−.002150]，1胜2平3负，保护未成立。future虽避开部分大伤，0091/2312/91仍将后继.704031变.001837（−.702193），模型只预测+.000561。旧救援future同率+.102327、区间全正，但仅3上下文/3救援点；压力全部同率+.025206、3胜15平0负、区间下界0，不能凭稀疏子组外推普通或完整保护。

固定模型附加诊断：myopic/future平均训练MSE .00262307/.00337650，自身常数 .00279451/.00355403，平均误差相对减少6.14%/5.00%；均匀留组MSE .00337390/.00412854，自身常数 .00301167/.00362069，恶化12.03%/14.03%。这是平均MSE比，不是平均逐模型R²，也不跨目标直接比较MSE。uniform只有4个收益>+.1、9个<−.1、中位0，无严格救援；训练内区分有限与留组损失同时存在，当前特征/线性容量/96例支持的贡献仍待分清，不归因为所有时序方向无效。

九表28/112/56/778/10283/234/108/45/2548行。已核对自指纹/来源/零调用、七CSV原行尾SHA、28个train21/held7/fit6分区及96例/112模型容量、778输入标签权重角色与归档Stage-117/116逐项零差；用独立加权增广最小二乘重建固定岭计算，778四分数最大差4.17e−16、56训练指标差5.56e−17。重算234摘要/108 bootstrap/45支持及主结果，并独立权重聚合，误差≤2.23e−16。10283 policy_decisions按来源顺序求和重建，SHA与声明完全相符。仅固定计算复核，无新正式拟合/调参。未取得manifest/模型JSON/预测NPZ/receipt，未验证84文件内容SHA或完整check_results；2548逐视频表原件未回传，重建序列化SHA不同，未独立核实其原表内容，不能声称全量字节一致。

下一步先检验现有公开证据/训练支持为何分不开大错放与大漏益，审计固定模型覆盖、错误近邻、特征冲突和留组误差，再决定目标身份/短时外观或新隔离成对训练支持；新增信号需单独消融/同率控制。不要直接部署future、加训PPO，或在这批压力反例上调阈值自证。本次只归档复核，Stage-1～118 Python冻结，未启动Stage-119。最新完整成绩仍Stage-113，无新AUC、稳定/恢复保护通过或独立确认。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE118_RESULTS_REVIEW.md)

### 0.108 2026-11-16 Stage-119 固定提交模型支持诊断（准备记录，结果见0.109）

用户要求继续下一步。沿Stage-118均匀future负结果和留组误差增加，先检验大错放／漏益的训练覆盖、近邻标签冲突和固定分数来源，再决定新增身份／短时外观信号或隔离的匹配训练支持。实现`tools/stage119_commit_support.py`、`tools/run_stage119_commit_support.sh`、六项测试与固定协议；保留前次Stage-118归档／复核改动，Stage-1～118 Python冻结，不自动commit/push。

绑定Stage-118 decision `8efa6dcf197a9249a3b034f5a70eba0f5c80b5eeeb4cbcccb2b2ee216ecede64`，完整check_results及manifest/decision/九表/84模型与预测收据和递归来源代码SHA。只需STAGE118_ROOT，无GPU／CHECKPOINT／图像入口；仍需服务器完整来源，非仅TXT即可过正式核验。本机无正式Stage-118 manifest／模型缓存，尚未执行正式778点诊断。

复用28冻结包，原每头同provider-held7其余6组／96例及g和pressure排除不变。检查原归一化／常数／正规方程和预测NPZ，复算2688训练＋778评价=3466行四个旧CPU分数；不求新β，不改／保存模型，无新训练、SGLA调用或完整轨迹。新增计算包括解正规矩阵以审计旧标签贡献，不能说零计算。778去重点、uniform448/stratified309交13、pressure34，791membership独立权重，确认组未用。

覆盖距离=原训练标准化但未clip的13维RMS；邻居身份只由public X和UID排序，为每个训练源组选最近1点，六组全输出，局部固定取3组。参照为96训练例各排除自己整个视频（跨种子／帧），从另外5组取最近3组平均距离的95%分位；待评距离与它比较并报连续差、最大标准化幅度、clip比例。训练5组库与评价6组库不同，不称95%覆盖保证／线上拒绝条件；常数维仍按原scale=1。小组合成测试k=min(3,训练组数−1)，正式6组k=3。

确定近邻后才附当前／后继标签，报近邻future均值／范围／正负混合和±.1材料性混合；不能据局部相反标签证明表示不可分。fit_support给每96例材料性正负的点／源组数、当帧严格救援及clip支持；压力标签不补入训练。±.1是原描述尺度，仅分类审计，不调决策。

对778点全部作特征14项（13维clip值×β＋截距）和6训练源组的有符号核权重×原标签分解，myopic/future各项和须复现冻结分数，核权重和须为1。固定归一化、Z/W/A，既不删组重训，也不是唯一因果归因；不能将最大负项直接说成环境或训练组导致失跟。逐点摘要列最近3组ID/标签、数值最大／最小训练组和特征贡献。

focus自动并集：future接受且真实差<−.1、future拒绝且真实差>+.1、全部旧pressure严格救援（含有利／不利／平局），按UID去重并保留membership，不只手选0204／0091。GT只作报告，不回流距离、邻居、原模型或阈值。三研究和all/historical/expanded分开；generalization对各自目标MSE／常数比及真实动作效用，support_summary六子集含空支持，按原权重→种子→视频宏平均；子集重叠、支持视频不同，不当独立或因果比较。

十表scope28、fit_quality56、fit_support28、sample_audit778、nearest_groups4668、score_sources4668、feature_terms10892、generalization18、support_summary54；focus数量由固定规则决定。TXT含decision＋scope/fit_quality/fit_support/generalization/support_summary/focus/sample_audit七表，完整近邻／组分数／特征项留服务器。无新模型NPZ缓存；已完成核验复用，部分重算确定性审计，不重训。COMPLETE只表示诊断完成，无新AUC／采用gate／能力保护通过。

六项本机测试通过：近邻身份与标签／分数无关、每源组唯一／同视频跨种子整组排除／UID平局；fit-only参照独立计算与未截断距离；冻结正规方程／核权重和／训练组及特征分解精确、错系数拒绝；独立研究权重、空支持和压力分离；大幅超范围／非有限几何；六组合成模型包审计全流程、禁止新拟合／图像／SGLA／上游训练、源文件不变、确定性续跑／篡改拒绝。Python AST/CLI、Bash -n、LF/无BOM检查通过。合成测试不当正式统计结果。

服务器同步后：`mkdir -p output/stage119`，再`nohup bash tools/run_stage119_commit_support.sh > output/stage119/run.log 2>&1 < /dev/null &`。默认来源`output/stage118/provider_isolated_commit_learning_v1`，结果根`output/stage119/fixed_commit_support_diagnostic_v1`，可覆盖STAGE118_ROOT/STAGE119_ROOT，CPU线程缺省1；结束自动TXT，`--pack-only`核验打包。以上为准备记录，正式结果已回传，见0.109；最新完整成绩仍Stage-113。 [完整协议与运行命令](docs/experiments/06_sgla_uav_tracking/STAGE119_FIXED_COMMIT_SUPPORT_DIAGNOSTIC.md)

### 0.109 2026-11-16 Stage-119 固定提交支持诊断结果（已完成，普通严重错判并非都分布外）

收到`stage119_results_20261116T091459Z_1515349.txt`，1164623字节，SHA256 `a65b12ec6caf6612440da395af9c6df6ff04d4e5a9bddf859cea8665f8281f03`，逐字节归档results并设-text。decision `81cc50e2b8a74be60274f9a1f02b793176125e3fcaae21e1dd8a30e9bbbaf5ad`，plan `935392f9f2881fbcf28dc4850a7f63792ca7ad5aa5b44228557973dc4d66e7e5`，来源Stage-118指纹相符。COMPLETE_FIXED_COMMIT_SUPPORT_DIAGNOSTIC只表示固定审计完成。

28冻结包／各96例作用域不变，778去重点（普通448/309交13＋旧压力34），3466行旧CPU分数复算。28组保存预测重放差全0，正规方程残差最大≤6.94e−18，组／特征分解最大差1.39e−16。新训练／模型／图像／GPU搜索／完整轨迹／确认推理均0。确认28源组仍封存，最新完整AUC仍Stage-113。

训练每头96例只有0～3个后继收益>+.1、0～4个<−.1，涉及0～2／0～3源组；全部头严格当帧救援0、训练clip0。当前13维未clip训练标准化RMS，取最近3不同源组，与训练内排除整视频后经验p95比较：均匀31/448超参照、宏6.92%；分层51/309（原始16.50%），宏10.26%；压力26/34（原始76.47%），宏73.61%；旧8救援全超参照。参照库5训练组／测试库6组且非IID，不称95%覆盖或线上拒绝保证。

均匀5错放＋3漏益全部未超参照、未clip；均匀全部13个材料性大效应（4正9负）也全未超参照。分层4严重错判2个超，压力7严重错判6个超。故单靠“超参照才拒绝”不能挡住这8个均匀严重错判；本轮并未实际部署拒绝策略，也不否定其他拒绝机制。

近邻正负混合的宏比例均匀63.39%、分层63.94%、压力50.00%，但778点没有任何近邻集合同时存在>+.1和<−.1；697/778点的三近邻无任一材料性大效应，均匀8严重错误／分层4严重错误全部如此。更多是相近公开表示下邻居只有微小效应、待评点却有大差异的经验线索，不能把微小符号冲突夸成已证明大效果不可分或目标身份不可识别。

0204/2313/661真实后继差−.701468、future+.002905，距离.2177对参照1.7553，近邻收益+.012984/+.012340/−.025380；0070/2313/69真实−.549410、预测+.000245，距离.3664/2.0602，近邻−.002253/−.002201/+.010739。0147/2312/1354真实+.490282、预测−.002274，距离.2995/.8177，近邻−.008150/+.003612/−.011291。旧救援0091/2312/91真实−.702193、预测+.000561，距离1.7303/.9291，近邻0/+.009432/+.003362。这些例子尚不能独立区分身份信息不足、动态支持缺失、线性容量／正则或后继不稳定。

固定分数分解：0204的score_advantage项+.003904、Hann−.001219，训练组0307+.002536／0070−.001959；0070特征绝对贡献和.012565，抵消后+.000245。0147的共享分数优势项−.001121；0308/409、411截距均−.007748，仍漏+.246765/+.298039收益。是固定A/归一化的代数恒等式，不是删组重训或因果环境归因，不按最大项删训练视频／改阈值自证。

距离长尾须说明：均匀原始中位.3626、p95=1.9761、最大51696.83，分层中位.4382／p95=7.0515；摘要平均577.96／675.86被极端值拉高。最高几个点来自0089的radius_max归一化幅度约186378.30，11个距离>100点来自0089或0226；该未clip诊断尺度敏感，不能据均值说普通点普遍很远，也不在看结果后删特征。压力距离中位2.9833、p95=35.2452。门控实际clip输入与诊断未clip距离区分。

Stage-118学习结果复算一致：均匀myopic/future留组MSE比各自常数高12.03%/14.03%，训练相对常数平均误差减少6.14%/5.00%；28头平均非零标签符号准确率59.65%/49.66%，非平衡类别不把后者直接称随机。focus26 UID＝错放9＋漏益10＋旧救援8−交集1，15超参照／11未超；无新增方法／能力保护通过。

十表28/56/28/778/4668/4668/10892/18/54/26行，按计划回传七CSV，其余近邻／训练组贡献／特征项留服务器。核对自指纹、来源、七CSV原行尾SHA／行数／零调用、28原fit6/96例作用域与模型指纹、778原公开输入标签分数权重逐字段零差。根据固定96例重建旧岭计算与矩阵，以独立按组最近点／整视频排除p95核对全部近邻身份、56训练／28支持／26focus及18泛化／54支持摘要，并独立权重宏聚合。大尺度几何最大绝对差2.92e−11、参照差4.45e−16，其余主项约1e−16，以容差验证。不调参、不发布替代模型。

未取得manifest／模型JSON／预测NPZ或收据及三完整贡献表，未核验这些实际内容SHA或完整check_results；没有读图，不能认定遮挡、相似干扰、光照或相机运动的现实原因。接续优先有对照地扩关键后继训练支持或检验身份／短时外观信息增量，保持提供者／源组隔离和普通／救援分开评价，不将旧压力加入训练再用同点验证。不要只做分布外拒绝、调零阈值或直接扩大PPO。本次只归档复核，Stage-1～119 Python冻结，未启动Stage-120。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE119_RESULTS_REVIEW.md)

### 0.110 2026-11-16 Stage-120 支持×外观提交消融（准备记录，结果见0.111）

用户要求继续下一步。根据Stage-119覆盖稀缺与普通严重错判不全分布外的诊断，本轮直接检验训练支持扩展及外观信息增量，而不继续只作距离审计。实现`tools/stage120_appearance.py`、`tools/stage120_commit_factorial.py`、`tools/run_stage120_commit_factorial.sh`及七项测试和固定协议。保留前次Stage-119归档／复核改动，Stage-1～119 Python冻结，不自动commit/push。

绑定Stage-119 decision `81cc50e2b8a74be60274f9a1f02b793176125e3fcaae21e1dd8a30e9bbbaf5ad`以及原118/117/115/113/112固定指纹、模型、公共点／轨迹／预测收据和递归来源／代码SHA。旧76/89/93仅读图像根与条目，不初始化CUDA。需要服务器原RGB与完整缓存，不需GPU／checkpoint；CPU有新拟合和图像解码，不能称无训练／无读图。本机没有正式旧模型／图像来源，正式实验待用户服务器运行。

仍评价778去重点＝普通744（uniform448/stratified309交13）＋旧压力34，791membership原权重，三研究及队列分开，28确认组封存。每个待评组g仅使用同上游provider-held7其余6组；g的所有帧／种子／普通与压力标签排除，其他提供者组不混入。

支持因子：uniform原96例，视频→种子→例等权；expanded加入同一允许6组的全部旧压力点（含有利／有害／平局），不加分层独有点。有压力的训练上下文将uniform／pressure各分一半质量，各池内等权；无压力仍uniform全质量，视频和种子总质量不变。此处理是补样＋固定重加权整体，不能单归因样本数或当真实频率。压力训练仅用其他视频，待评g压力仍整组留出；因旧压力曾看过且按GT选，不称新确认。model_scope报补点数、质量和三类角色支持。

外观因子：原13维＋24维RGB线索。两参考为首帧GT目标ROI及此前30帧内最近一个通过旧memory_write条件的预测ROI，后者可能错误；严格不使用当前／未来作参考，不用后续GT修正。q0／原policy框在同一原RGB图取ROI。每参考×描述（RGB每通道8-bin开方归一化直方图、16×16双线性灰度去均值L2归一化NCC）给q0/policy相似度、差与两有效位，共20维；加两覆盖率、recent存在位／年龄共24。可见率<.5、平坦灰度／无近期参考明确mask置零，不丢样本。NumPy纯计算无Pillow依赖，checked_image仍核验原SHA／尺寸。这是轻量颜色纹理线索，不称已验证语义身份特征或重识别能力。

四主组uniform_relative／uniform_appearance／expanded_relative／expanded_appearance；另uniform_shuffled／expanded_shuffled同容量控制。打乱只在允许训练视频内联合置换24维，跨其种子／池，按UID和PCG64固定seed23201，不读标签／待评组；真实／打乱用相同例／权重／真实训练X归一化，外观头38参数相同，评价均当前真实外观，报实际值改变率。保留视频边际，不当唯一因果随机化。原13维头14参数，外观37维输入38参数。全部新头同后31帧差目标、ridge1、截距不罚、clip10和零收益阈值，不调参／加训PPO。

uniform_relative及uniform_constant精确复用Stage-118原future／常数和模型绑定，其他5岭头＋1扩展常数每组新拟合，共140岭头＋28常数＝168新CPU模型，28新包；另复用28旧岭头／28旧常数。原模型不改，新预测只收公开base/appearance和冻结头，不收评价标签，全部保存后才汇总。

预定主比较uniform/all/all expanded_appearance−expanded_relative后31帧IoU；另固定支持在两表示下的作用、原支持外观增量、2×2交互、各自真实对打乱、最终头对policy/q0_all/连续性/旧myopic/扩展常数/同率。19策略＝原4控制＋6头＋2常数＋6同率＋两动作GT上界。各研究原权重→种子→视频，pressure三角色，保留损伤／漏益／遗憾／大损伤数及最坏点、视频胜平负／留一／10k条件bootstrap(seed23200)，未经多重校正／重训bootstrap，非新确认。只改一次动作的旧匹配后继，不代替连续门控完整自身轨迹与恢复保护，不能报新AUC或采用通过。

56外观NPZ＋receipt＝112，28包＋28预测NPZ／receipt＝84，总196产物。56上下文按使用帧去重，图像解码上限56＋2×778＝1612，实际任务manifest固定；跨种子同文件可重复，image_reads是已提交缓存账本，失败重试额外未计。续跑不重训／解码已缓存图，但来源字节SHA仍核验。十表appearance_scope/evidence/model_scope/fit_metrics/sample_predictions前五行数56/778/224/224/778，policy_decisions15029；其余policy_summary/paired/support/video_effects。TXT含decision及除逐策略／逐视频外的八CSV，两完整表留服务器。

七项本机测试通过：颜色纹理同目标优势、平坦／部分ROI mask与当前／未来参考拒绝；未来GT／写入／框不影响过去参考；扩展只同提供者、测试整组及其他提供者标签／外观替换不改拟合；视频内打乱／同容量／同归一化；2×2交互和独立研究权重／同率；原公共点／轨迹收据匹配与图像绑定；六组合成像素＋拟合全流程、旧基线保持、预测先标签、零SGLA／上游训练、源文件不变、零重训／读图续跑及篡改拒绝。Python 3.10语法、CLI、Bash -n与LF/无BOM检查；不把合成结果当正式科学结果。

服务器同步后：`mkdir -p output/stage120`，再`nohup bash tools/run_stage120_commit_factorial.sh > output/stage120/run.log 2>&1 < /dev/null &`。默认根`output/stage120/support_appearance_commit_factorial_v1`，支持STAGE76/89/93/112/113/115/117/118/119_ROOT和STAGE120_ROOT覆盖。结束自动TXT，`--pack-only`核验打包。以上为准备记录，正式结果现已回传，见0.111，不重复运行；最新完整评价仍Stage-113。 [完整协议与运行命令](docs/experiments/06_sgla_uav_tracking/STAGE120_SUPPORT_APPEARANCE_COMMIT_FACTORIAL.md)

### 0.111 2026-11-16 Stage-120 支持×外观提交结果（已完成，外观次要信号与救援损伤并存）

回传`stage120_results_20261116T105024Z_1547979.txt`已逐字节归档并设-text，1,179,310字节，SHA256 `7bb4ecae8f5386a7bc96313a50c72f2f61158ef3736312ec2e2e7e4d8d997db2`。状态`COMPLETE_SUPPORT_APPEARANCE_COMMIT_FACTORIAL`，decision `3c559807c1d88ab1a961dd586777f07b4adc9d91a4e3c06e2dc73376b1c96bc5`、plan `fd9a1cfb164b2e7cf16d7912f8b0f5a6142c8cba7ad7d5f21e19face7812b581`；来源Stage-119指纹相符。原件＋八CSV齐全，不重复索取或重跑。

范围：28开发源组、56种子上下文、778去重点、791 membership，普通均匀448／分层309交集13、旧压力34。168个新CPU模型=140岭头＋28常数，另28旧岭／28旧常数复用；196文件产物，实际提交账本1484次RGB解码。SGLA更新／GPU搜索／新完整轨迹／确认推理均0。它是一次q0提交后恢复冻结生成／共享评分策略的后31帧条件IoU，不能称完整AUC、持续门控轨迹或新确认；最新完整仍Stage-113。

每个待评源组仍只从同provider-held7其余6组拟合，测试整视频／压力／两种子标签排除。原支持96例；扩展100～107例，压力4～11例、权重1/6～1/3、严格救援0～4。扩展同时改变支持与普通／压力池权重，不能仅归因补样或真实线上事件分布。新增24维=颜色／NCC匹配＋覆盖／有效位／参考年龄，共37维；722/778有近期参考，56缺失，不删点。当前处理没有区分图像内容与有效位增量，不当可靠身份识别。

均匀主比较EA−ER（扩展外观−扩展13维）+0.001960、CI[−0.001724,+0.006073]，16胜3平9负，中位+0.000160、最差−0.026846、最小留一+0.000709。主比较未建立收益。原支持UA−UR（外观−13维）+0.004521、CI[+0.001010,+0.008959]、16/5/7，UA−US（真实−打乱）+0.004214、CI[+0.000812,+0.008698]、13/5/10，是预定次要正向线索；多比较未校正，不能升级为主要成功。分层UA−UR为−0.000352、UA−US−0.001119，区间均跨零，没有同方向支持。

EA−UA均匀−0.001443、ER−UR+0.001118、交互−0.002561，区间均跨零，扩展处理没有建立互补价值。均匀原policy／连续性／UR／UA／US／ER／EA／ES后继IoU为0.570277／0.570739／0.566747／0.571269／0.567055／0.567865／0.569825／0.568787。EA对policy−0.000452、对自身同率+0.000569，均跨零。事后另算UA对policy+0.000992、对连续性+0.000530、对自身同率+0.001512，均跨零；这三个是探索性补充，不是原paired预定对比。尚未建立优于实际策略的可靠选择价值。

普通均匀UR／UA／ER／EA接管率41.29%／40.40%／65.18%／54.24%，严重有害起点5／2／7／6；UA有害成本0.001793低于UR0.005067，EA0.004705。0070/2313/69与0204/2313/661真实后继差−0.549410／−0.701468：UA将旧正分改负，挡住损伤；但EA在0204又转为+0.001073并接管。0147/2312/1354真实+0.490282，UA和EA补到；0308/2313/409真实+0.246765，仍漏。不能据这些例子调阈值或筛训练视频。

**旧8个严格救援点，EA全部接管q0**，后继与统一q0相同0.304051，原policy0.634385；差−0.330334、CI[−0.591862,−0.080428]，6视频全负。UA旧救援0.479839，也未保护；事后对policy−0.154547、区间全负。角色按当帧条件定义，不等于每点后继差必负。EA旧压力总体0.229294高于policy0.180024，但低于ER0.238861和连续性0.299838；不能用损伤组改善掩盖必要救援失败。

核验：八CSV原行尾SHA／行数、decision自封印、旧778点公共字段／标签／分数／权重与Stage-118精确一致。独立重建视频→种子→池权重、SHA-PCG64打乱和训练归一化，以增广最小二乘重建140新岭头／常数与旧基线，全部八分数最大差2.22e−15、训练损失差2.78e−17、旧预测误差0；重算342摘要／252区间／45支持，独立聚合最大差2.22e−16，确认15029逐策略／3724逐视频行数。56上下文参考时间／相似度差值与掩码算术、1484帧集合账本相符。

未取得manifest、原图、完整可信历史或196文件实际字节和两完整表，不能称重提外观、核验全部产物内容SHA或运行完整check_results。数据表只支持行为与统计判断，不能认定某例遮挡／干扰物等现实成因。

后续保留“原支持＋外观”作为表示增量线索，优先隔离图像内容与覆盖／有效位／年龄，并同时检验普通状态、必要救援及同率价值；扩展压力＋重加权不可当已验证组件直接继承。不要直接采用EA／UA、挑seed、调阈值或加训PPO。当前仍无可靠新提交规则、完整方法收益或能力保护通过；28确认组继续封存。本次仅归档复核，Stage-1～120 Python冻结，Stage-121未实施；未自动commit/push。 [完整复核与原件](docs/experiments/06_sgla_uav_tracking/STAGE120_RESULTS_REVIEW.md)

### 0.112 2026-11-18 Stage-121 单帧备用搜索状态（已完成，保护未通过）

本轮冻结生成器、评分器、SGLATrack 和模板。上一帧最高分框与标准框 q0 分差 ≤0.05 且中心距离至少为两框较大的面积平方根时，用 q0 作为下一帧第 4 个搜索槽位的备用父框，保持四次预算。L40 四臂同卡重测 196 条，不复用 L20 对照。实际搜索 725,395 次，生成共享与备用各 232,208 次。零新模型、训练更新、确认推理。原生与固定共享逐种子 AUC 和 Stage-113 一致，生成共享有 17/56 条 AUC 变化，本轮只用 L40 作为直接对照。

生成共享／备用 AUC 53.4103／54.0502，差 +0.6399，95% CI [−0.3953,+2.3317]，5 胜 18 平 5 负，两个种子 −0.4079／+1.6876。低 IoU 和最长失跟占比下降 0.823／1.135 个百分点，区间均跨零。备用对原生／固定共享仍低 9.9103／10.0807 点，新增失败视频 5／6，丢匹配恢复 13/26／17/24；对生成共享无新增稳定视频失败，丢恢复 1/20。开发保护未通过。

58,094 个备用组帧次中实际换入 358 次，18 视频、29 轨迹，备用槽位被选 73 次。0174／2313 两次换入、均未获选，AUC 24.3043→66.4236，去掉 0174 后主差 −0.1164。0089／2312 45.7660→35.2726，未匹配参考恢复 805 帧。后续优先利用缓存对齐实际触发、首次输出分叉和后继，隔离备用框的直接供给与更换候选后对其他槽位评分的影响。整组诊断覆盖全部轨迹，两个视频作为相反案例。没有新 Stage-122 或新阈值设置，确认集继续封存。

原始附件已逐字节归档。5 个 CSV 的 SHA-256 与报告一致，从 196 条轨迹重算策略、配对、双种子、留一及保护统计，表内浮点差为 0；另核对 392 个产物哈希条目的命名与格式。服务器 manifest、NPZ 不在回传件中。 [完整结果与原始附件](docs/experiments/06_sgla_uav_tracking/STAGE121_RESULTS_REVIEW.md)

### 0.113 Stage-122 备用搜索的候选供给与评分重排（已完成）

Stage-121 L40 的 28 视频／56 对轨迹、358 次实际替换全部列账；80 次前史和前三槽观测一致，278 次不一致。80 次中输出不变 56，改选备用槽 11，原第四框被替代后选其余框 8，两组均选第四但箱体不同 3，两组均从相同前三框竞争却改选 2。仅在共同前三框中最高分换框共 4 次，其中另 2 次最终选了备用框。19 条轨迹出现首次输出分叉，其中 18 条发生在匹配替换点。80 个匹配点补出 9 个原集合没有的好框（IoU ≥ 0.5），仅 0349／2312 第 73 帧选中，未观察到匹配点因移除原第四框而丢失集合中唯一好框。

0174／2313 第 746 帧 q0→q2，两输出 IoU 均 0，备用框 0.7020 未获选。单独换入平均尺度差可复现前三框翻转，恢复该项可逆转。747 帧起前三槽裁剪分叉，后 31 帧两臂 IoU 都为 0；全程 +42.1193 AUC 不能直接归因此单帧或声称立刻恢复。0089／2312 第 802 帧 q2→q0，二者 IoU 均 0，备用框 0.6657 未获选。单项换入均不足，分别恢复平均中心距离或 Hann 相对排序可撤销前三框翻转，后 31 帧实际轨迹 0.6331→0.0021。两例支持候选间关系评分可改变输出和后继搜索，改善与损伤分开解释。

回传五 CSV 的字节 SHA、56 对 AUC 与 Stage-121 逐轨迹、358 点分类与汇总均核对。附件未含服务器 manifest、模型或 NPZ，本机未独立重算逐帧网络分数。零新模型训练、图像读取、搜索与确认集推理。接续可先隔离备用框直接供给与已有候选评分的变化，比较固定原三框分数和按新集合重算的方案，并保留完整恢复对照。尚未启动下一轮。 [完整复核与原件](docs/experiments/06_sgla_uav_tracking/STAGE122_RESULTS_REVIEW.md)

### 0.114 Stage-123 固定前三框分数（已完成）

Stage-122 全部 80 匹配点（18 视频），相同四框和备用框分数。只替换前三框分数为原集合里的分数，改选 5 点，新增好框采用 1/9→3/9（0176／2313／130：0→0.7130，0237／2312／414：0→0.7967）。其余 3 次换框当帧 IoU 均为 0→0。视频配对差 +0.041936，区间 [0,+0.106004]、2 胜 16 平 0 负。原对照 80 点只选中 1 个好框，零丢失没有广泛保护证据。

0174／2313／746 从 q2 回到 q0，0089／2312／802 从 q0 回到 q2，两次仍漏选备用框。固定分数依赖原第 4 候选上下文，只是机制对照。后续的候选自身及历史参照控制已在 Stage-124 完成，结果见 0.115。完整效果仍以 Stage-121 为准。

已归档并核对四张 CSV 字节哈希、80 个身份与分数、160 个唯一最大分数及全部汇总。附件缺服务器 manifest 和 NPZ。零新图像、GPU 搜索、训练、完整轨迹及确认集推理。 [完整复核与原件](docs/experiments/06_sgla_uav_tracking/STAGE123_RESULTS_REVIEW.md)

### 0.115 Stage-124 候选独立评分（已完成）

7,561 个去重状态、28 来源、112 个 CPU 小模型。两个输入条件共用同一上游折内其他 6 个训练来源，1,544～1,693 状态／头，评价来源各自留出。屏蔽 6 个同帧关系及其平方，保留候选自身响应和历史参照。普通状态 7,168 个，两个种子 14,336 次评价。独立对同批关系重训当帧 IoU +0.001098、CI [−0.001091,+0.003306]，19 胜 9 负、两种子 +0.000203／+0.001992。对原记录 −0.002650、名义 CI [−0.005146,−0.000465]。对关系评分好框增选 78 丢 54，对原记录增 34 丢 103。

358 备用触发对原记录 +0.046293（名义 CI [+0.001732,+0.114896]）；对关系重训 +0.015578、区间跨零、最小留一负。9 个新增好框的关系／独立／原记录采用分别 4／5／2 次（两模型种子），独立较关系的种子差是 −1／+2。普通原记录救援 6 次缓存观察对应三个视频时刻，两重训组对每种子均只保住 2/6，漏掉 0091 与 0235。原生参考恢复时刻独立较关系 −0.005664，0 胜 6 平 4 负。暂不进入全局替换。若继续，优先以相同触发范围对照原评分、重训关系与独立评分，保证有普通状态和已知救援的压力验证。

原始结果已逐字节归档，五表哈希、逐点动作及汇总均复核。附件缺服务端 manifest、模型和 NPZ。Stage-124 零新 GPU、图像、完整轨迹或确认推理。 [完整复核和原件](docs/experiments/06_sgla_uav_tracking/STAGE124_RESULTS_REVIEW.md)

### 0.116 Stage-125 备用搜索触发后的短程闭环（已完成）

358 个实际触发点中 346 个有完整后 31 帧、17 个视频、12 个片尾截断。三种评分起点共用历史和四候选；下一帧恢复原评分并沿各自轨迹。L40 同运行身份，1,730 个逻辑分支／925 个实际续程／579 条新分支／71,796 次新搜索。新增训练、整段轨迹及确认集推理均为 0。六表 SHA、当帧选择和后继汇总已复核。 [结果复核及原件](docs/experiments/06_sgla_uav_tracking/STAGE125_RESULTS_REVIEW.md)

独立对原评分后 31 帧视频等权 IoU +0.005629、95% 名义 CI [−0.000185,+0.015083]，9 胜 6 平 2 负；对关系 +0.012156、[+0.000180,+0.031509]，5 胜 6 平 6 负。两个种子的单独区间均跨零。去掉 0089 和 0348 后对原评分 +0.000150。独立新增持续好窗 0／3 个且保住原评分的 4 个窗；严格救援只 2 个起点，0349 一次当帧好框丢失。仅 13/346 个起点有好候选。0237／414 的当帧 IoU 0.797 未延续；同视频 525 仅变下一帧备用请求即可让同动作续程从 0.874 跌到 0。当前结果保留作条件诊断，尚不足以采用持续在线评分替换。新一轮设计可把候选可用性、当帧提交与下一帧备用触发的效果分别验证。

### 0.117 Stage-126 当帧提交与下一帧备用搜索（已完成）

L40 同身份，346 个起点／17 视频／824 个已选动作／1,648 个逻辑格。实际续程 1,220 条，复用 925 条、新增 295 条和 36,580 次搜索。四表哈希、Stage-125 自然结果、全部效应与配对统计已复核。零新训练、完整轨迹或确认集推理。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE126_FACTORIAL_CONTINUATION.md)

两路均禁止下一帧备用时，独立评分所选框的后 31 帧视频等权差 +0.030208［+0.000489，+0.082522］；自然总差 +0.005629，区间跨零，净请求贡献 −0.024580。交互 −0.027621，去掉 0089、0237、0348 后约 −0.001176，禁用下的提交差也只剩 +0.000298。396 次实际备用更换中，33 个起点动作后继均值改善、43 个下降、320 个持平；持续好窗新增与丢失各 2 个。0237／414 开启备用后仍有同样好的候选，却从选中 IoU 0.813 变成选到 0；0237／525 从候选全差变为找到并选中 IoU 0.883 的好框。后续若检验评分与搜索的结合，须区分实际补框和保留候选上的选择变化。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE126_RESULTS_REVIEW.md)

### 0.118 Stage-127 搜索覆盖与输入尺度（已完成）

346 点，1,384 槽，零新图像／搜索／训练。333 个无好输出点中 54（16.2%）四裁剪无交集，3（0.9%）部分覆盖，276（82.9%）覆盖≥90%；275/276 完全覆盖。覆盖充分的无好输出点中，226 的最佳短边小于 4 像素、243 小于 8 像素。充分覆盖最小裁剪的组内中位数为原帧目标短边 41、裁剪边长 5,712、补边 93.8%、缩放短边 1.97。13 个好输出点对应 28／156／0%／52.55。0226 占 276 中 115 个，均属消耗过的触发状态。三表、四槽 IoU、采样身份与汇总复核一致。现有证据首先定位搜索尺度和补边，具体膨胀来源及缩小后能否恢复需同帧对照。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE127_RESULTS_REVIEW.md)

### 0.119 Stage-128 保留覆盖的裁剪缩放（已完成）

同身份 L40，346 个同帧起点／265 个受干预起点／288 次搜索。q0／q1／q2／q3 改 17／243／5／23 个裁剪，补边中位数 93.8%→43.8%，改动裁剪边长中位 5,712→1,904 像素。各槽位目标覆盖率相同。226 个原目标过小／覆盖充分点最佳输入短边中位 1.49→4.43，199 个仍低于 8 像素，全部原路和改后最高 IoU <0.2。全体新增、丢失好候选与好选框皆 0，最高 IoU 视频等权 +0.001931，区间［−0.000517，+0.005209］，选中框 +0.000075、区间跨零。

极小目标组中只有 42 点的最近历史包含至少一个按当前画面长边复算会超限的提交框。需同时看 q1 生成放大与反馈，不能把超限一概归到历史。画面长边级缩放已检验，尚无恢复证据。四表、原 Stage-127 数值和所有配对统计已复核。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE128_RESULTS_REVIEW.md)

### 0.120 Stage-129 粗搜与四倍局部放大（已完成）

同身份 L40，346 同帧起点／265 启用／1,384 控制＋530 局部＋265 真值居中搜索，零新训练或完整视频。全部好候选 13→15（增 8、失 6），好选框 3→6（增 4、失 1）。失去的 6 个好候选全是原 q3。最佳 IoU 视频等权差 −0.013085、区间跨零；选中 +0.026468［+0.000284,+0.060899］，实际启用组区间仍跨零。226 个过小点补 6 个好候选、3 个好选框，第一和第二局部区域各独占 3 个好候选。冻结原 Stage-128 同卡对照逐槽一致。

真值居中同尺度 265 点均完整覆盖，54 个有好框；这 54 点有 38 个实际局部覆盖不足、8 个实际覆盖充分但定位仍差、8 个两路都好。实际局部覆盖≥90% 共63点，实际局部好框共9点。不能统一把放大行为放到两个后备槽位，也不能用真值居中结果直接代替实际选址。若续做，现有同帧缓存可组合检验保留 q3、只开一个局部位的候选代价与重评分；选址可用性仍是独立问题。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE129_RESULTS_REVIEW.md)

### 0.121 Stage-130 保留 q3 的单槽放大与重评分（已完成）

346 点／17 视频、1,730 点×条件、265 启用点，零新搜索、读图、训练或完整视频。缩边／双局部／第一／第二／投票的好候选点为 13／15／18／18／18，最终好选框 3／6／5／4／4。三个单槽条件均新增 5 个有好候选起点、丢失 0 个。第一位置保住全部原好选框，新增 0080／2312／163、165。第二和投票各丢同一个原好选框：0349／2312／73，q3 IoU 0.611 仍在，局部 q2 高分取代它，最终 IoU 0。

三种单槽相对缩边的最高 IoU 差 +0.007818／+0.036761／+0.022136；选中框差 +0.002378／+0.032185／−0.001215，三项选中区间均跨零。投票 31 次转第二位置，这些点在第一、第二之间的好候选有无均一致。第二位置均值收益集中，删 0237、0085 后描述性差约 +0.00251。三表、两历史基线、全部局部候选拼装及 24 项配对统计已复核。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE130_RESULTS_REVIEW.md)

### 0.122 Stage-131 单槽放大整段闭环（已完成）

同 L40，28 视频／196 轨迹，新 112 条、复用 84 条、464,416 次新搜索。原生／备用／缩边／放大 AUC 63.9605／54.0502／51.4511／51.6007。放大对缩边 +0.1497［−0.2536，+0.7702］，3 胜21 平4 负，种子分别 −0.1186、+0.4179，新增稳定失败 0，丢 2 次匹配恢复。相对 Stage-121 备用 −2.4495、丢 6 次恢复；相对原生 −12.3598、5 个稳定视频新增失败、丢 15 次恢复。缩边对原备用 −2.5991［−5.6202，−0.1519］。开发判据全未通过。

实际放大 227 帧次／23 轨迹／15 视频，放大槽获选 32 次。0176／2313 AUC 13.616→28.147、另一种子略降，去掉 0176 平均 −0.1131。丢失缩边恢复为 0076／2312／1485 和 0085／2313／1478，两个种子轨迹分别 7 和 1 次放大但无局部槽获选，需在服务器缓存中核查非放大槽重排与备用请求的首次分叉。五表、84 对照、40 配对及保护汇总已复核。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE131_RESULTS_REVIEW.md)

### 0.123 Stage-132 搜索裁剪与轨迹首次分叉审计（已完成）

28 视频、种子 2312／2313、112 对缓存轨迹、189 条事件记录。原备用对缩边在匹配状态出现 887 帧次裁剪改变、好候选整体消失 0、漏选 1、同框但备用请求改变 3，最终丢失 5 次匹配恢复。缩边对放大为 67／0／0／0，丢失 2 次。0070／2312 第 31 帧只改 q1，保留好 q0（0.7047）却选 q1（0.0733），下一帧最佳 IoU 0.0829。0174／2313 第 740 帧同框但下一帧预测输入变了。

放大对照中，0076／2312 第 956 帧原 q2 被替换后降分，输出改选 q0，第 1485 帧仍有好 q1（0.9194）却漏选。0085／2313 第 1454 帧只替换 q2，保留槽位首名 q3→q0，最终也从原 q2 改选 q0，第 1478 帧放大组最佳 IoU 0.0487。两个首次换框点两侧所选 IoU 均为 0。接续可固定首次分叉前的历史，用同起点短程续跑检查候选替换、选框写回与备用请求各自的作用。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE132_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE132_FIRST_DIVERGENCE.md)

### 0.124 Stage-133 原生单搜索失跟诊断（已完成）

同协议的原版 LETrack 为 62.0056，原生 SGLA 为 63.9605，配对差 −1.9549 点［−7.53，+3.57］。当前继续以 SGLA 为研究对象，LETrack 用于外部对照。Stage-133 复用 28 视频的原生轨迹、29,047 跟踪帧、19 个失跟事件和 121 个去重邻近帧。全部起点的目标中心在有效裁剪内，17 个完整覆盖，另两个覆盖约 87.72% 与 90.16%。11 个仍有好格点框，Hann 与原始响应两套前五峰合并保留 6 个。其他 8 个充分覆盖却无好框，输入真值短边 23.33–32.00 像素。

原始最高峰救回 0 个失跟起点。所有帧由低转好 47，反向 271。固定状态 AUC 为 63.5387，相对 63.9605 差 −0.4218［−0.7501，−0.1245］，7 胜4 平17 负。全部低 IoU 帧中有 2,501 个裁剪无交集且无好框，这个情况在起点未出现。接续优先做同一原生状态、同裁剪的候选一次改选和等预算短程续跑，检查首次错框是否会被后续轨迹放大。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE133_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE133_NATIVE_FAILURE_AUDIT.md)

### 0.125 Stage-134 原生候选改选与短程续跑（已完成）

固定 Stage-133 的 19 个失跟起点，每条开发视频最多另取一个满足连续 33 帧 IoU ≥ 0.50 的原生稳定窗口。起点共用一次前向，分为原生输出、去重后的 Hann 第二峰、原始和 Hann 各前五峰并集内的真值上限、稠密格点真值上限。换框只发生在起点，随后各自接回原生单搜索 32 帧。优先看新增或丢失的连续三帧好框、后继平均 IoU 和稳定窗口退化。真值上限用于定位问题，完整视频效果仍需后续评估。

同 L40 完成 19 个失跟窗口、28 个稳定窗口、6,063 次物理搜索。原生轨迹与缓存最大坐标差为 0。失跟窗口内，原生／Hann 第二峰／峰池真值／稠密真值分别有 6／8／8／9 个出现连续三帧好框，后 3 组增减分别为 +3／−1、+3／−1、+5／−2。后续 32 帧视频等权平均 IoU 相对原生分别变化 +7.44、+7.51、+15.26 个百分点。后两个真值组只优化起点 IoU。

第二峰在稳定窗口带来 4 次新增低 IoU，0169／556 和 0174／1088 后续 32 帧全部低 IoU。0204／664 能靠第二峰恢复，0091／92 和 0176／162 只由稠密组新增恢复。0089／1159 即使当帧稠密框更准，仍丢失原生恢复。建议先在固定候选、固定预算下验证可在线计算的改选条件，比较新增恢复和稳定损害。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE134_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE134_NATIVE_PEAK_CONTINUATION.md)

### 0.126 Stage-135 同裁剪第二峰改选条件（已完成）

复用 Stage-134 的 19 个失跟窗口和 28 个稳定窗口。原生与强制第二峰为对照。三条固定规则依次要求 Hann 相对分差 ≤0.20、两框 IoU ≤0.20、第二峰与历史外推框的 IoU 优势 ≥0.10。外推用前两框的中心位移并限幅，保留上一框尺寸。只按当帧证据选择起点，后续 32 帧从已算的对应分支评分。无新增搜索和训练。

三条规则在 19 个失跟窗口分别改选 10／9／6 次，均新增 0016／605、0204／664、0348／538 的恢复，并保住原生 6 次。后续视频等权平均 IoU 差分别为 +8.13／+8.17／+7.81 个百分点，28 个稳定窗口都未触发。稳定窗口中不同第二峰的最小相对分差为 40.54%，单独限制 20% 分差即可挡住强制第二峰带来的 4 个低 IoU 窗口。历史条件减少改选，仍未证明有独立收益。下一步冻结分差单条件与完整条件，做整段视频同预算闭环对照，衡量重复触发和长时失跟风险。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE135_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE135_CACHED_PEAK_GATE.md)

### 0.127 Stage-136 同裁剪第二峰整段闭环（已完成）

同 L40、28 个开发视频、84 条完整轨迹、87,141 次搜索。原生／仅分差／完整规则 AUC 63.9605／61.0823／63.6637，低 IoU 13.20%／17.45%／13.53%，未恢复事件 6／9／6。两规则相对原生 AUC 差 −2.8782／−0.2968，95% 视频配对区间均跨零。原生稳定的 12 个视频中，分别 2／1 个新增失跟。0204 两条规则都在 664 一次改选救回，整段 +18.63 点。0304 原生无失跟，两条规则新增 1,176／640 帧持续失跟，整段 −50.34／−28.43 点。

接续 Stage-137 保持原生历史、同裁剪和同一 L6 特征，从 L7–L12 各自执行一个独立块。优先看 Stage-133 原选块满网格都无好框的八个起点，其他失跟点和稳定帧作对照。先判断跨块是否补出了好框，再决定是否做难帧补算的同预算闭环。Stage-136 两条改选规则保留为对照，暂不作为补算分支的默认写回。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE136_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE136_PEAK_GATE_CLOSED_LOOP.md)

### 0.128 Stage-137 同裁剪独立候选块供给诊断（已完成）

Stage-133／134 固定起点 8 无好网格／11 有好网格失跟／28 稳定。47 个原生父框同裁剪，L1–L6 算一次，L7–L12 分别直接接同一 L6 特征和原 CENTER/Hann 标准解码。主比较其他块能否直接给出 IoU ≥0.50 框，另记仅其他块稠密网格含好框、选择器第二名和最大 Hann 峰选择。实际 47 预处理裁剪、282 候选块／头、282 前缀块、47 选择器。无写回、训练和新整段轨迹。服务器执行 `bash tools/run_stage137_shared_crop_layers.sh` 并回传 `stage137_results_*.txt`。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE137_SHARED_CROP_LAYER_SUPPLY.md)

首次权重摘要调用在首个窗口前中断，修复后在 `output/stage137/shared_crop_layers_v2` 同 L40 完成 47 起点，原选框与 Stage-134 最大差为 0。主分析 0／8 有标准好框，0016／605 只有 L11 网格 IoU 0.5221，标准输出 0.4541。原网格已有好框的失跟组 8／11 从其他块获得标准好框，相对原块有限峰池额外 3 个，即 0085／1541、0089／1159、0237／242。原选择器 47 次全选 L9；取跨块最大峰找回 4 个起点，但破坏 3 个稳定起点的好框。若继续补块，先比较跨块框与同图框的同起点短续跑，衡量额外候选带来的恢复和风险。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE137_RESULTS_REVIEW.md)

### 0.129 Stage-138 跨块候选写回与短程续跑（已完成）

固定 0085／1541、0089／1159、0237／242 三个跨块新增好框及 0080／645、0147／1113、0349／408 三个最高峰受损稳定点。同 L40 完成五组 30 条短分支、960 次后续搜索，与 Stage-134 的原块控制最大坐标误差为 0。跨块最优组在 3 个失跟起点全有当帧好框，恢复总数 2，与原生及原块有限峰池相同。后续平均 IoU 比原生／有限峰池变化 −1.82／−1.02 个百分点。0089 早恢复但复发，0237 起点 IoU 0.511，接下来 32 帧均 0。三个稳定误选点续跑无新增低 IoU。当前优先查失跟后裁剪覆盖／响应供给，直接做动态补块触发的依据不足。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE138_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE138_CROSS_LAYER_CONTINUATION.md)

### 0.130 Stage-139 写回后的裁剪与响应诊断（已完成）

同 L40 重放 240 个分支帧／200 次物理搜索，原输出框误差 0。960 个后继分支帧覆盖均为 100%，304 个低 IoU 帧均在覆盖内。0237／243 原生与跨块输出均 0，原生网格最佳 0.5087、跨块 0.3205。0089 跨块从 1165 复发，输出 0.0920、稠密 0.7478、有限池 0.4816，1166 有限池恢复好框但仍漏选。优先对 0237 同帧交叉原生与跨块父框的中心和尺寸，定位影响供给的裁剪因素。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE139_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE139_POST_WRITE_DIAGNOSTIC.md)

### 0.131 Stage-140 搜索父框中心与尺寸交叉（已完成）

同 L40 完成 11 帧四组 44 搜索，Stage-139 原始组最大框差 0。0237／243 完整覆盖，稠密最佳原中心原尺寸／跨中心原尺寸／原中心跨尺寸／跨中心跨尺寸为 0.5087／0.4679／0.3739／0.3205，恢复原尺寸与中心的增量分别为 0.1474／0.0534，交互 −0.0126。四组实际输出均 0。244–250 四组稠密均未达 0.50，实际输出全 0。0089／1165–1166 八个交叉都有稠密好框却没选中。0349／409 单换跨块中心时覆盖 88.52%、标准输出 0、稠密最佳 0.7976。局部尺寸效应不足以作为在线规则，后续须把可见上下文、输入尺度与候选排序分开检验。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE140_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE140_CROP_CENTER_SIZE_CROSS.md)

### 0.132 Stage-141 固定裁剪尺寸的外圈背景干预（已完成）

同 L40 完成 11 帧／22 对中心／88 次搜索，两种填充均改变 22 对像素哈希，原始框与 Stage-140 最大差为 0。0237／243 目标完整保留，大裁剪全网格最佳相对小裁剪 −0.1411，均值／边缘填充对大裁剪回升 0.0239／0.0384，未达好框。244–250 两种填充对大裁剪平均 −0.0106／+0.0060，四组均无好框。0237 全 64 标准输出 0。0089 全 16 次仍有好格点框，标准输出均低于 0.50。后续可分开验证采样与网格位置，并把峰值排序保留为独立对照。 [结果及原件](docs/experiments/06_sgla_uav_tracking/STAGE141_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE141_CROP_CONTEXT_PROBE.md)

### 0.133 Stage-142 等候选预算的全网格提取（结果已复核）

28 视频 29,047 原生缓存帧，包含 19 个审计失跟起点，另有 240 个分支回放帧。固定裁剪、响应与原生框，共用原始响应／Hann 排名合并和框去重。每帧两组候选数相同，最多 8 个。全帧好框覆盖由 24,272 增至 24,953。原生 IoU < 0.20 且网格存在 IoU ≥ 0.50 好框的 281 帧中，等预算召回由 162 增至 179，新增 18、丢失 1。按视频等权为 +8.69 个百分点，95% 配对区间 [+0.77, +22.70]。19 个失跟起点由 5 增至 8。原始回传、来源指纹、汇总及区间已核对。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE142_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE142_DENSE_CANDIDATE_RECALL.md)

0089 的原生失跟起点 1159 候选池最佳 IoU 从 0.4698 到 0.6135。跨块分支第 1165 帧的两种候选池最佳均为 0.4816，第 1166 帧均为 0.5546，原记录输出为 0.1042。下一轮可冻结候选池，按视频源组验证小型 IoU 质量评分器相对原生 Hann 首选和池内原始响应首选的收益，在相同预算下检查原稳定窗口。0237 原响应网格缺好框的问题仍需单独处理。当前结果仅度量固定状态的候选召回，整段 AUC 和恢复收益仍待验证。

### 0.134 Stage-143 固定候选池的质量评分（结果已复核）

使用 Stage-142 两个冻结候选池和按视频源组留出的 9 维岭回归 IoU 评分。29,047 个原生状态按源组等权的平均 IoU 为原生 Hann 0.649779、局部峰质量评分 0.649844、全网格质量评分 0.648239，两项相对 Hann 配对区间均跨零。161 个两池都有好框且原生 IoU <0.20 的帧，池内原始响应选中 47 个，质量评分局部峰／全网格选中 8／7 个。全网格独有好框的 18 帧，质量评分选中 0 个。19 个原生失跟起点上，质量评分局部峰／全网格选中 4／3 个，池内原始响应 0 个。0089／1159 全网格好框 IoU 0.6135，质量评分选中 0.4698。跨块最优分支／1166 好框 IoU 0.5546，质量评分仍选原生 0.1042。645 个事先定义的原生稳定窗口中，两池质量评分均无 IoU <0.20 的帧，原始响应有 185 帧且 14 个窗口出现连续至少 3 帧。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE143_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE143_FIXED_POOL_QUALITY.md)

下一步优先检验哪些可观测证据能区分值得接受的改选与有害改选。保持同池、同预算和按源组留出，设置同帧难例的成对排序对照，分别报告失跟起点、全网格新增机会和原稳定窗口。当前结果属固定状态诊断，尚无整段闭环收益证据。

### 0.135 Stage-144 同帧成对排序（结果已复核）

固定两套候选池、9 维特征、28 个留出源组，共核对 29,287 条原生及分支读数。普通排序在 18 个全网格独有机会中选中 0147／1111，IoU 0.5131，回归为 0.2204。19 个失跟起点上普通排序与回归的选择完全一致，局部峰／全网格均为 4／3 个好框。29,047 个原生状态的源组等权平均 IoU，Hann 为 0.649779，普通排序两池为 0.649778／0.647374，难例加权为 0.586731／0.557621。普通排序两池的 645 个稳定窗口均无连续低 IoU；难例加权局部峰／全网格分别出现 68／4 个，且有 2,152／5,306 个稳定帧跌破 IoU 0.50。0089／1159、1166 的好框仍未选中。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE144_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE144_PAIRWISE_CANDIDATE_RANKING.md)

继续保留逐框 IoU 回归作对照。接续优先查好框与高响应错框的可分性，先在同一裁剪与候选池内比较响应／几何证据和候选相对首帧模板或可靠历史的外观证据，再决定是否改排序模型。阶段 91–93 提供历史外观参照，其完整闭环曾丢失匹配恢复。Stage-144 是已用开发数据上的固定状态诊断，整段 AUC 和恢复收益仍待验证。

### 0.136 Stage-145 同池外观证据诊断（结果已复核）

共 4,754 个状态、452 个配对和 110 个稳定窗口，CPU 读取图像 7,171 次，无新增搜索或写回。全网格独有好框的 18 帧中，首帧灰度相似度把好框排到原评分错框之前 8 次，直接全池选中 3 次。近期颜色在 19 个失跟起点选中 6 个，保留原评分的 3 个并新增 0085／1541、0348／538、0349／96。相对于回归，近期颜色的失跟起点配对 IoU 差区间跨零，整体诊断样本也无明确收益。全网格首帧灰度／近期颜色／近期灰度直接选框分别在 31／2／0 个稳定窗口出现连续严重低 IoU，跌到 IoU 0.20 以下的帧为 517／17／10，原评分为 0。0089／1159、1166 的好框仍未选中，1166 的该分支缺合格历史参考。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE145_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE145_FIXED_POOL_APPEARANCE.md)

接续可保持全网格候选与逐框回归，检验外观只在评分冲突时辅助改选。比较相同触发条件下的纯响应／几何与加入近期颜色或首帧灰度相似度的结果。按视频源组留出校准，逐项检查新增好框、既有救回保留和稳定窗口损伤。现有结果属于固定状态，整段跟踪收益仍待闭环验证。

### 0.137 Stage-146 候选冲突时的外观辅助选择（结果已复核）

复用 Stage-145 的 4,754 个状态和 110 个原稳定窗口。4,514 个原生状态中 31 个出现可改选候选，2 个有新增好框可选。281 个原生严重低 IoU 且响应网格有好框的状态中，池内真值最优有 179 个好框，默认回归选中 7 个。触发组合为回归分差 ≤0.15 且框间 IoU ≤0.25，6 个状态通过但无新增好框进入改选集合。19 个失跟起点有 2 个状态通过，均无可补的好框。18 个全网格独有机会没有任何状态通过。240 个分支状态有 4 个新增好框机会通过共同过滤，三组仍均未改选。整体读数与默认回归一致，稳定窗口连续低 IoU 均为 0。0089／1159、1166 池内有好框但均在共同触发阶段被排除。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE146_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE146_CONFLICT_APPEARANCE.md)

接续优先拆分回归分差和框间重叠两个条件，记录每个漏选好框分别被哪个条件挡住。用冻结候选池做覆盖审计，同时计数各条件允许的稳定帧误改选机会，再决定是否扩大外观比较范围。Stage-146 三臂无实际改选，无法据此区分纯响应／几何、近期颜色和首帧灰度的改选效果。

### 0.138 Stage-147 触发条件覆盖审计（结果已复核）

复用 Stage-146 的 4,754 个状态，核对 33,085 个备选和 96 种条件。172 个原生严重低 IoU 补救状态的 184 个好候选中，172 个仅被回归分差挡住、7 个仅被重叠挡住、5 个被两者挡住。只把分差从 0.15 放宽到 0.25，放行 41 个补救状态，稳定帧严重错框暴露由 6 增至 137。只把重叠上限从 0.25 放宽到 0.40，放行 4 个补救状态，其中 2 个是失跟起点，严重错框暴露为 8 帧，低于好框阈值的错框暴露为 97 帧。取消两个几何筛选后，两种外观均可用时覆盖 168／172 个原生补救状态。0089／1159 的 IoU 0.5108 候选只受重叠限制，最佳 0.6135 候选两项都超阈。1166 的 0.5546 好框只受分差限制且缺近期外观。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE147_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE147_TRIGGER_COVERAGE.md)

接续可把分差 0.15、重叠上限 0.40 作为覆盖诊断后的固定开发对照，验证外观辅助能否实际选中新增进入比较的起点好框，另报稳定帧误改选。大分差候选需要另外验证默认框不可靠的证据。当前统计为候选进入比较的机会，尚无此条件下的实际改选或闭环收益。

### 0.139 Stage-148 放宽框间重叠后的外观辅助选择（结果已复核）

同样本的框间 IoU 上限从 0.25 改为 0.40，分差保持 0.15。281 个原生严重低 IoU 且网格存在好框的状态中，4 个新增好框进入比较。纯响应／几何实际选中 1 个，两种外观各选中 0 个。19 个失跟起点有 2 个新增好框（0085／1541、0089／1159），三臂均将其排为各自池内最佳备选，但合成分差均为负，现有 3 个回归救回保留。按源组重新校准后，4,514 个原生状态中三臂分别改选 5／4／4 帧，取得 3／2／2 个好框。三臂共同选中的是 0003 的 139、144 两帧，两帧原生 Hann 已正确。110 个采样稳定窗口均未发生实际改选。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE148_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE148_OVERLAP_APPEARANCE.md)

接续可固定当前候选集合，分离“允许备选进入比较”和“是否接受备选”对回归分差与原始响应差的使用。现有两个起点好框的回归和响应分差均劣于默认框，外观差幅度不足以抵消。检查同集合上的成分消融和独立接受证据，并报告稳定帧误改选。整段跟踪收益仍待闭环评价。

### 0.140 Stage-149 准入后的接受分数消融（结果已复核）

保持 Stage-148 的 4,754 个状态、默认框和准入集合。仅用首帧灰度的 28 个留出折均选 0.025。在 4,514 个原生状态中改选 14 帧，4 帧从低于 0.50 改为好框，已有好框无损失。19 个失跟起点由 3 个好框变为 4 个，新增 0089／1159（0.4698→0.5108），0085／1541 因分差 0.0214 未采纳，0169／881 从 0.2893 降到 0.0696。110 个采样稳定窗口无实际改选。全原生诊断样本平均 IoU 差 +0.000507［−0.000029，+0.001321］，起点平均差 −0.011166［−0.041184，+0.007685］。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE149_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE149_SCORE_ABLATION.md)

首帧灰度零门槛选中 0085／1541 和 0089／1159，同时丢失 0204／664 原起点好框，并让 36 个稳定帧跌破 0.50。接续可冻结已校准的首帧灰度单项规则，对实际改选状态做同起点短程续跑，检查 0089 的小幅当帧改善能否持续及 0169 等误改选是否扩大。整段收益仍待闭环评价。

### 0.141 Stage-150 校准灰度改选的同起点续跑（结果已复核）

从 Stage-149 的全部 14 个校准改选状态开始，两路分别写入 Stage-143 回归框和灰度备选框，之后各自承接预测，用同一个冻结 SGLA 和回归评分器。L40 同卡运行 714 次后继搜索。起始帧新增 4 个好框，逐起点平均差 +0.149051。后继完整 8 帧（12 起点／7 源组）和 32 帧（7 起点／6 源组）平均差 −0.012955／−0.045662，32 帧连续好框新增 1、丢失 2。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE150_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE150_GRAY_WRITE_CONTINUATION.md)

0089／1159 当帧 0.4698→0.5108，后 32 帧 0.5880→0.0620。0085／690 当帧 0→0.2896，后 32 帧 0→0.7516，同视频 687 帧写回没有恢复。0169／881 两组都出现连续严重低 IoU，灰度组早 1 帧。0348／362 当帧小幅改善，灰度组随后丢失回归组的连续好框。短程均值对个别源组敏感，此结果用于已触发状态下的单次写回诊断。 [服务器原始回传](docs/experiments/06_sgla_uav_tracking/results/stage150_results_20261120T045650Z_603337.txt)

### 0.142 Stage-151 写回框中心与宽高交叉对照（结果已复核）

Stage-150 的 14 个改选点四路交叉，同 L40 共 1,428 次后继搜索，原两组 714 个后继框完全一致。完整 8／32 帧窗口只换中心相对回归 +0.068707／+0.045123，只换宽高 −0.066419／−0.088469，写回完整灰度框 −0.012955／−0.045662。32 帧仅 7 起点／6 源组，去掉 0085 后只换中心的均值差转负。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE151_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE151_WRITE_GEOMETRY_CROSS.md)

0089／1159 替换宽高使次帧裁剪边长 66→104，目标仍完全覆盖，全网格最佳 IoU 0.7275→0.4044，后 32 帧均值 0.5880→0.0288。0085／690 只换中心即恢复，次帧 0→0.7084，后 32 帧 0→0.7539。0169／881 是宽高有益的反例，单换宽高后 32 帧 0.0310→0.4227。范围限于校准规则已触发的开发起点。 [服务器原始回传](docs/experiments/06_sgla_uav_tracking/results/stage151_results_20261120T051054Z_606941.txt)

### 0.143 Stage-152 只换搜索中心的整段闭环（结果已复核）

L40 同卡完成 28 个开发视频、112 条轨迹、116,188 次搜索。原生／回归／完整灰度／只换中心 AUC 为 63.9605／60.6450／56.6701／55.4456，低 IoU 为 13.20%／18.53%／23.85%／25.68%。只换中心相对完整灰度 −1.2245［−4.7955，+1.9302］，相对回归 −5.1994［−10.0169，−1.2002］，相对原生 −8.5149［−16.0811，−1.2946］。只换中心在 2 个回归稳定视频产生新失跟，15 次失跟中有 14 次未恢复。4,514 个原生采样选择及 14 次影子改选复现。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE152_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE152_CENTER_CLOSED_LOOP.md)

0089 在两条灰度轨迹均于 1159 帧首次改选，整段只换中心相对完整灰度 +10.3451 AUC。0085 两路在 216 帧已触发，690 帧均未触发，不能用冻结起点的恢复替代其整段结果。0080／0304 相对原生的失跟在回归组已存在，0016／0169 是只换中心相对回归新增的失跟。当前未支持采用整段灰度写回规则。 [服务器原始回传](docs/experiments/06_sgla_uav_tracking/results/stage152_results_20261120T064318Z_611857.txt)

### 0.144 Stage-153 难帧串行加算一个 Transformer 块（结果已复核）

同 L40 完成 28 视频、56 条轨迹、58,094 次搜索。两组各 29,047 帧均选 L9，0 个难帧、0 次加算，两组 AUC 均为 63.9605、低 IoU 13.20%、失跟 19 次且 6 次未恢复。两组逐帧预测框一致。原生与自适应的每视频中位耗时均值为 6.056／4.981 毫秒，在相同调用数下无法归因于深度变化。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE153_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE153_ADAPTIVE_DEPTH.md)

门槛规则以过去分差的第 25 百分位为比较基准，当前分差须严格更低。Stage-137 的 47 个点均记录前两名分差 1.0，支持检查选择器 Sigmoid 分数的饱和现象。Stage-153 逐帧分差与门槛留在服务器轨迹缓存中。整段运行未触及加算路径，实际串联收益待验证。

### 0.145 Stage-154 强制加算一块的同裁剪对照（结果已复核）

同 L40 完成 19 个失跟起点和 28 个稳定起点，3,102 次搜索，47 个起点均实际执行 L9 → L10。原生分支对 Stage-134 框坐标最大差为 0。19 个失跟起点当帧新增好框 1 个，后继 8／32 帧平均 IoU 差 +0.006420／−0.009831，32 帧连续好框新增／丢失为 1／1，新增连续严重低 IoU 1 次。原网格有好框的 11 点后 32 帧平均差 +0.001616，无好框的 8 点 −0.041814。28 个原生稳定起点均保留当帧好框，没有新连续严重低 IoU，后 32 帧平均差 −0.010346。 [结果复核](docs/experiments/06_sgla_uav_tracking/STAGE154_RESULTS_REVIEW.md) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE154_FORCED_DEPTH_CONTINUATION.md)

0085／204 当帧 0→0.5147，连续好框 223→205。0348／367 原生从 383 帧开始连续跟准，加算组后继 32 帧 IoU 全为 0。0348／538 新增连续好框，32 帧均值与原生接近。结果为使用冻结权重的一次串行加算干预，反映已使用开发起点上的局部效果。 [服务器原始回传](docs/experiments/06_sgla_uav_tracking/results/stage154_results_20261120T121217Z_707147.txt)

### 0.146 Stage-155 只训练串行路径的 L10（结果已复核）

同 L40，28 个源组两折，1,192 个视图，每折适配 L10 副本 192 步。47 个起点共 4,653 次评估搜索及 94 次追加块。19 个失跟点上，适配相对原 L10 加算新增 1 个当帧好框，后 8／32 帧视频均值差 +0.014948／+0.000080，连续好框恢复无新增或丢失。相对原生新增 2 个当帧好框，后 32 帧均值 −0.009751，连续恢复新增和丢失各 1 次。28 个稳定点相对原生的后 32 帧均值 −0.011581，无新连续低 IoU。0237／242 适配当帧 IoU 0.576，后 32 帧均为 0。0348／367 的原生恢复在两种加算中均丢失。 [原始结果](docs/experiments/06_sgla_uav_tracking/results/stage155_results_20261120T134722Z_730065.txt) · [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE155_ADAPT_SERIAL_L10.md)

### 0.147 Stage-156 动态加算触发离线验证（已完成）

将 Stage-137 原生 L9 峰与 Stage-155 续跑按 47 点匹配，门槛和两变量收益模型按源组留出。以适配 L10 为例，按低峰值／峰偏离中心加算 16／12 次，后 32 帧相对原生 −0.004147／−0.003142，均低于同预算随机期望。收益模型只在 5 个稳定点触发，后 32 帧差 +0.000041，19 个失跟点未改。16 次预算内按后续真值保护性选择，适配／冻结的事后参考为 +0.002628／+0.003090。优先检验何时保留原生框以及写回对下一帧的影响。基于当前原生响应触发需要在部署成本中计入追加 head。 [结果及复现](docs/experiments/06_sgla_uav_tracking/STAGE156_DEPTH_ROUTING_AUDIT.md)

### 0.148 Stage-157 固定加算预算下的写回对照（GPU 已完成）

同 L40 在 16 个触发点采集双头，后继 1,024 次。直接写回、峰值增益、峰值加框间 IoU ≥ 0.30 的后 32 帧相对原生为 −0.004147／−0.004907／−0.005794。两条实际规则均接受 0348／367 的坏写回并拒绝 0085／1541 的正向写回，连续好框新增／丢失 0／1。真值保护参考后 32 帧相对原生 +0.002035。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE157_DEPTH_WRITEBACK.md)

### 0.149 Stage-158 等预算六选一与短程续跑（已完成）

同 L40 复用 47 个起点，9,024 次后继搜索。原生／留出固定／当帧事后／保护性后续事后的 32 帧视频均值 IoU 为 0.582533／0.578013／0.598246／0.613905，连续恢复新增／丢失 0／0、0／0、3／1、3／0。保护性事后对固定 +0.035893，24 胜4 平，最差留一 +0.023121。0016、0204、0348 的新增恢复分别由 L11、L8、L11 提供。尚未训练在线可用选择器。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE158_LAYER_ROUTING.md)

### 0.150 Stage-159 六选一选层监督对照（已完成）

同 L40 完成 328 点、62,976 次续跑和四个模型。原生、当帧监督、后续监督、事后保护上界的后 32 帧视频平均 IoU 0.603926／0.603175／0.605847／0.615646。后续监督换层 247 次，新增恢复 2 次、丢失 0 次、新增连续低 IoU 2 次，0176／163 获益、0226／174 恶化。后续相对原生 13 胜15 负，最差留一仅 +0.000170。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE159_LAYER_SELECTOR.md)

### 0.151 Stage-160 原生层保留门槛（已完成）

328 点复用 Stage-159，零 GPU。校准后换层从 247 降至 68，后 32 帧相对原生 −0.000673、相对 Stage-159 −0.002594，新增连续恢复和新增连续低 IoU 均从 2 降至 0。0176／163 的好换层与 0226／174 的坏换层均被拒绝。事后参考保留 110 次，后 32 帧相对原生 +0.005491、新增恢复 2 次。下一轮建议沿用六路缓存，比较仅用预测平均收益和同时预测连续退化幅度的否决，按源组留出校准，对比匹配换层次数下的恢复保留与新增低 IoU。六路缓存共有 33 个严重有害候选，集中于 14 个起点／10 个视频，连续退化幅度适合作为第一轮监督。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE160_LAYER_VETO.md)

### 0.152 Stage-161 同预算的退化预测（已完成）

328 点复用缓存，两折共换层 68 次。加退化预测相对 Stage-160 后 32 帧 +0.000515，相对原生 −0.000158，最差留一相对 Stage-160 −0.000029，新增／丢失恢复和新增低 IoU 均为零。10 个起点改决策，4 改善、5 变差、1 持平。0176／163 在该折 135 个候选中从第 68 降到第 113，0226／174 从第 91 降到第 135。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE161_LAYER_DOWNSIDE.md)

### 0.153 Stage-162 历史特征消融（已完成）

四组比较 4 帧和最多 16 帧统计、趋势特征有无。只重拟合退化头，固定 Stage-161 的 64／4 次换层及候选层。正式消融的 16 帧趋势组后 32 帧平均 IoU 为 0.605447，四帧基础组为 0.603767，原生为 0.603926。16 帧趋势组新增一次持续恢复（0176／163，L7），新增持续低 IoU 和丢失恢复均为零。0176 的提议从第 113 升到第 62，第二折的 0348／540 仍排第 90。剔除贡献最大的 0176 后，16 帧趋势组相对四帧基础组为 −0.000112。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE162_HISTORY_ABLATION.md)

### 0.154 Stage-163 收益预测的历史特征消融（已完成）

固定 Stage-159 提议层、Stage-162 的 16 帧趋势退化量及各折 64、4 次换层。正式报告指纹为 `6a8f4361ef426d140e946d52a47dd5872cfc6250b8f169d5282328962fb33974`。四帧收益对照后 32 帧平均 IoU 为 0.605447，长历史趋势收益组为 0.605180，相对差 −0.000267，剔除 0176 后 −0.000277。0176／163 的 L7 排名从第 62 升到第 21，两组均恢复。两折分别替换 7、2 个名额，两组均无新增持续低 IoU 和丢失恢复。未达到在线验证门槛。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE163_HISTORY_GAIN.md)

### 0.155 Stage-164 L6 局部匹配收益特征（已完成）

328 个起点均完成 L6 采集，L40 运行和原归档投影逐点一致。两组投影对照后 32 帧平均 IoU 均为 0.605447，局部匹配为 0.603588，差 −0.001860，剔除 0176 后差 −0.000098。0176／163 的 L7 从第 62 降至第 126，局部组失去对照恢复，无新增持续低 IoU 或丢失原生恢复。正式报告指纹为 `4574f92e613bf14106b3193306d2c375bf58b8fe2e12e2ed58e79454e0d97234`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE164_L6_LOCAL_MATCH_GAIN.md)

### 0.156 Stage-165 候选层重提议（已完成）

四帧对照、同起点重提议、重提议并重排起点的后 32 帧平均 IoU 分别为 0.605447、0.605413、0.605342。重排后相对四帧对照 −0.000105，剔除 0176 后 −0.000109。0176 的 L7 从第 62 降到第 64，仍进入配额并恢复。33 个提议层发生变化，同起点实际只换了 1 个层。338 个 IoU 相差至少 0.01 的层对中，预测收益减退化的排序准确率为 50.3%。正式报告指纹为 `dc02d44ae9215a66c9a8cd34a7d5ec9db7bbbd7ddca0d43f2ed244d9325eec9d`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE165_LAYER_REPROPOSAL.md)

### 0.157 Stage-166 同帧层间排序（已完成）

固定 68 个起点。带状态的成对排序组后 32 帧平均 IoU 为 0.605587，比四帧对照高 0.000140，剔除 0176 后高 0.000224。层间排序按视频等权准确率 57.4%，对照为 47.4% 和 52.8%。合并 338 个层对的准确率为 50.6%，原收益分数为 51.2%。0176 第 163 帧改选 L8，仍恢复。无新增持续低 IoU 或丢失原生恢复，达到完整视频验证门槛。结果指纹为 `c85ff5d3f9b591261088f160c9af73b3bff4b76a0a8909947f1867925dc318e3`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE166_PAIRWISE_LAYER_RANKING.md)

### 0.158 Stage-167 在线完整视频对照（已完成）

28 个开发视频、84 条轨迹的 AUC 为原生 63.961、原提议 62.383、成对排序 62.598。成对排序较原生低 1.363 点，剔除 0176 后低 1.406 点，较原提议高 0.215 点，剔除 0304 后低 0.569 点。两组换层策略在 0304 和 0080 引入失跟，成对排序在 0089、0176 丢失原提议组的恢复。原提议和成对排序分别换层 2784、2813 次。开发门槛未通过。报告指纹为 `ce4232a7f7a3774866fd526c89ebf267d42c5b0078fcc3188e931d4f2a464cbb`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE167_ONLINE_LAYER_RANKING.md)

### 0.159 Stage-168 同状态续跑（已完成）

冻结 6 个窗口，完成 17 个起点、85 条分支，持续组按自身历史复现原轨迹，最大框偏差为零。0304 的原提议历史在第 530 帧单换 L11 后第 554 帧失跟，原生 L9 和单换成对 L7 的分支未失跟。第 523 帧单换 L11 则未失跟。0089 的成对历史在第 1182 帧单次干预并恢复原生后第 1194 帧恢复，两条持续组在对应窗口内均未恢复。0080 失跟和 0176 后段恢复窗口前 32 帧均无换层。报告指纹为 `2bb16b118a44a9bb6790c98a37bb9c67e1b2ed3601991e1a09dd461ff94f7153`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE168_MATCHED_LAYER_ROLLOUTS.md)

### 0.160 Stage-169 定点阻断（已完成）

三组配对、六条续跑完成，对照精确复现。0304 只否决第 530 帧，原第 554 帧失跟消失，第 554 帧 IoU 为 0.7588，到第 651 帧仍未失跟，开回门控后触发 19 次。0089 保留第 1182 帧动作，暂停第 1183 至 1198 帧换层，两个策略各拦下 5 次并在第 1194 帧恢复，到第 1310 帧无再次失跟。第 1199 帧恢复门控后没有实际换层。报告指纹为 `024b4c0990a0a7a66f3607d0fb518c9605260c0578ffc5f4a7b83cca2681009e`。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE169_TARGETED_LAYER_VETO.md)

### 0.161 Stage-170 固定冷却完整视频对照（已完成）

28 个开发视频、五组。原生、原提议和成对排序为对照，后两组各加入固定 16 帧冷却。规则只由过去是否实际换层触发，被拦提议不重置计时。每组复用 Stage-167 的门槛及权重，核对首个有效否决前的精确复现和每帧原调用预算。同设备复用旧 84 条轨迹，仅新跑 56 条；运行身份不同则五组同卡重跑。报告 AUC、低 IoU、原稳定视频新增失跟、丢失恢复、实际换层和否决，含剔除 0176 或 0304 的对照。服务器执行 `bash tools/run_stage170_layer_cooldown.sh`，支持 L20／L40 和总体进度条。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE170_FULL_VIDEO_COOLDOWN.md)

实际在 L20 新跑 140 条完整轨迹，五组各 29,047 次搜索。原生、原提议、成对排序、原提议加冷却、成对排序加冷却 AUC 分别为 63.961、62.383、62.598、63.051、63.779。成对排序加冷却比自身对照 +1.181 点（配对区间 −1.862 至 +5.194），剔除 0176 后 −0.476 点。0304 的失跟由 1091 帧提前到 554 帧，0080 仍有新增失跟。两组冷却均未通过开发门槛。已核对回传的决定指纹、七张表的哈希、140 条汇总记录与调用预算。原始 GPU 轨迹留服务器。

### 0.162 Stage-171 同起点换层与冷却交叉续跑（已完成）

同 L20 完成 6 个起点、24 条续跑、8,476 次搜索，9 条自然对照逐帧复现 Stage-170。0304 第 11 帧沿用 L9 的两组都在 554 帧失跟，换到 L7 的两组都撑过第 586 帧，虽然换层后的前 32 帧略降。第 526 帧再干预的四组都在 554 帧失跟。0176 第 140 帧只有 L12 加冷却避开持续失跟。0089 第 1182 帧用 L7 后开放换层未恢复，加冷却在 1194 帧恢复。已核对六张表哈希，按逐帧回传复算 27 行窗口效应。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE171_MATCHED_COOLDOWN.md)

### 0.163 Stage-172 跨视频换层与续跑时长诊断（已完成）

在其余 25 个开发视频的 Stage-170 冷却轨迹中，每视频从实际换层和被冷却拦截两类各取至多一个起点。固定哈希顺序，要求各起点至少留 128 帧。L20 完成 17 个视频、33 个起点、132 条续跑和 17,028 次搜索，33 个自然分支的框误差为零。按视频等权，128 帧起点换层的平均 IoU 效应在后续开放换层时为 +0.00241，在延续冷却时为 +0.00025。起点换层的两项对照在 32 和 128 帧之间均无两端绝对值至少 0.01 的反向案例。后续开放时有 5 例仅在 128 帧达到该幅度。四组各有 5 个未恢复事件。33 个起点中 23 个提议层与排序层不同。 [实验及运行命令](docs/experiments/06_sgla_uav_tracking/STAGE172_COOLDOWN_HORIZONS.md)

### 0.164 Stage-173 门控评估层与实际执行层的对齐诊断（已完成）

同 L20 完成 39 个起点、117 条续跑和 15,093 次搜索。39 个起点的存档层、门控分及原动作框全部精确复现，26 个提议层与排序层不同。每起点执行原生、提议、排序三个分支，随后各自反馈预测框并共用原生选层续跑 128 帧。结果指纹 `10e874a80e41905eb159898d875800822b83339639c387403fb9ebbd78bad85f`。已核验七张表的哈希，并由起点级数据复算 27 行汇总和 18 行配对结果。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE173_GATE_LAYER_ALIGNMENT.md)

33 个取样起点来自 17 个视频。沿用原阈值，让排序层自行通过门控后，放行从 33 降至 13 次。按平均 IoU 差 0.01 计数，32 帧有害放行由 2 降至 0，三档窗口内均无有益动作被挡。对原门控加排序层，32／64／128 帧视频等权 IoU 差为 +0.000175／+0.000322／+0.000624。128 帧未校正区间为 [+0.0000044, +0.001320]。三种规则各有 5 个未恢复事件。排序层自身的 32 帧退化预测绝对误差为 0.00363，直接借用提议层预测为 0.00190，尚无风险预测改善。

6 个定点中，0304 第 526 帧提议和排序均选 L11，门控仍通过。相对 L9，L11 的 32 帧与 128 帧平均 IoU 分别低 0.13756 和 0.62183，第 554 帧失跟且本窗口未恢复。0176 第 51 帧则因对齐而挡掉排序 L11，错过 128 帧 +0.01600 的局部收益。整个取样只覆盖原规则已放行的机会，续跑使用原生选层。

下一轮建议先扩大有后效标注的起点，纳入原规则放行和拒绝的在线状态。在源视频组隔离下比较现有 32 帧退化评分与加入 128 帧严重退化或未恢复失跟监督的评分，按各折相同放行配额衡量严重失跟拦截与有益换层保留。0304 等既有定点用于复查，留出评估负责判断是否进入完整视频对照。

### 0.165 Stage-174 同放行预算的长窗退化监督（已完成）

同 L20 完成 106 个起点、212 条续跑、27,348 次搜索，全部有效、存档原动作框误差为零。25 个源组的 100 个主起点含原规则放行 25、冷却否决 26、风险门控拒绝 49，另有 6 个定点。三组各放行 40 点（两折 24／16），严重有害放行与新增未恢复放行均为 0。冻结 32 帧评分、32 帧重拟合、128 帧重拟合有益放行分别为 0／1／1，128 帧按源组等权的平均 IoU 差相对原生续跑为 −0.000331／+0.000114／+0.000114。两个重拟合精确选中同一批 40 点。结果指纹 `eba621e11a0f96763b14af9fe588f18dbec74fea2a99af6f8577d004b7306a17`。已核验八张表哈希、放行集合、各折配额、源组隔离和策略主指标。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE174_LONG_HORIZON_RISK.md)

主样本没有 32 或 128 帧平均 IoU 下降达到 0.05 的起点。唯一严重事件在 0348 第 438 帧，换 L12 后多了未恢复失跟，完整 128 帧平均 IoU 差仍为 +0.000655，均值退化标签因此为零，三组均未放行。留出该源组的折没有严重事件训练样本。0304 第 526 帧三组仍放行 L11，128 帧平均 IoU 差 −0.62183，长窗模型只预估 0.00051 退化。`supports_closed_loop_pilot=false`。

下一轮建议用冻结轨迹的失跟／恢复事件作离线抽样索引，在事件前选取同状态起点并配普通窗口。对照回放判断换层是否额外造成失跟或阻断恢复，再核对这些事件是否被全窗均值掩盖。待多个源视频组有可重复的动作相关事件后，将新增未恢复失跟和丢失恢复直接作为监督，继续同配额留出对照。

### 0.166 Stage-175 失跟与恢复事件附近的同状态续跑（已完成）

从 Stage-170 固定冷却加排序轨迹的失跟和恢复前 1～4 帧、5～16 帧各选一个起点。同一视频、同原门控状态、同续跑长度配普通窗口。片尾保留至少 32 帧，主分析要求完整 128 帧。排除 Stage-173／174 的重复起点，单独记录此前诊断过严重退化的四个源组。两组从同一历史分别执行原生层和固定排序层，后续共用原生选层策略，评估沿用起点前共同的 IoU 与事件状态。主事件风险为新增未恢复失跟或超出 32 帧宽限的丢失恢复，同时核对平均 IoU 风险标签漏标。若主分析的两折均有事件风险且合计覆盖至少三个源组，则支持后续事件监督试验。

同 L20 完成 139 个起点、278 条续跑、33,858 次搜索。全部窗口有效，归档原动作框和门控分误差为零。72 个事件起点含失跟前 46 个及恢复前 26 个，匹配普通窗口 67 个。主分析含 12 个源组的 41 个完整事件窗口，严重事件风险仅 1 例，来自 0307 第 499 帧。L9 在第 504 帧恢复，改选 L10 后这一失跟未恢复，128 帧平均 IoU 差 −0.64958。完整新源组的 36 对匹配窗口里，事件组和普通组各 1 例，按源组等权差为零。普通窗口的 0169 第 649 帧换 L10 后第 653 帧起持续失跟，128 帧平均 IoU 差 −0.62766。全部 5 个事件风险案例的整窗平均 IoU 降幅均至少 0.05。Stage-174 的 0348 均值漏标仍是此前证据，此轮未再出现。

换层也可阻止失跟。主分析中 0204 第 663 帧从 L9 换到 L11，避免第 664 帧起的未恢复事件，128 帧平均 IoU 差 +0.77126。不同起点及状态的效果不可直接外推为完整视频收益。`supports_event_supervision_pilot=false`。回传指纹 `33214a0995ab082783b6016db51369d3baed77410a1d21b9b9e376c1edebc5c8`。已核验六张表，并复算 16 行总结、67 个匹配和 4 行配对统计。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE175_EVENT_RISK.md)

下一轮建议在有害、有益和普通起点上做固定起点 L7–L12 六层交叉续跑。比较同一状态下各层的事件风险与恢复，确认有害的排序层是否可由其他非原生层替代，并核验有益替代在多少源组重复。用受保护的事后上界检查候选是否提供可用空间，再决定增加事件采样还是训练评分器。

### 0.167 Stage-176 同起点六层续跑与事件保护（已完成）

从 Stage-175 固定选取五个严重风险起点、三个避免未恢复或挽回恢复的起点，另每源组按哈希抽一个普通起点。加入 Stage-173 的 0304 第 526 帧和 Stage-174 的 0348 第 438 帧，去重后 26 点覆盖 16 个源组。分别强制执行 L7–L12，再统一沿用原生选层，各自反馈预测框。L20 完成 156 个分支、19,602 次逻辑搜索，复用 Stage-175 的 6,018 次搜索，新跑 13,584 次。26 点原动作框与门控分误差均为零。

七个有害排序起点中，五个有受保护的非原生替代，但 0307 第 499 帧与 0304 第 526 帧只有 L9 达保护条件。0307 的 L7 可在期末恢复，前 32 帧平均 IoU 仍下降 0.1266。0204 第 663 帧五个非原生层均受保护且避免持续失跟，说明这里的关键是得到换层机会。16 个普通起点无达标有益替代，受保护事后上界平均只比原生高 0.00234。非定点源组的完整 128 帧子集有 14 点／11 组，2 组排序有害、1 组有安全替代、1 组有保护下的实际收益。仅事后拦住有害排序、其余继续排序的反事实均值为 +0.03456，六层受保护上界为 +0.03733，两者均相对原生续跑。上述反事实使用真实后效，仅作机制分析。已核验六张结果表并复算 26 条上界、8 个分组与 32 个汇总对照。结果指纹 `7eb638828bb53591223c9d30beb5d38a3c23d0a26aee3d1a07f3ca700108ac57`。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE176_LAYER_EVENT_SWEEP.md)

下一轮建议先审计原门控提议原生层的决策状态。Stage-175 的三个挽救起点均提议原生，而五个有害起点原规则均未放行（四个 `gate_rejected`、一个 `native_proposal`）。固定每帧可见的原生选择置信度与历史状态，另采未用于设计的源组中的原生提议起点，比较高疑难状态和常规对照的受保护收益与新增严重风险。在匹配预算下检验能否打开少数有益机会，保留原生回退。旧定点仅用于复查。

### 0.168 Stage-177 原生提议状态下的换层审计（已完成）

同 L20 完成 243 次扫描（239 个开发起点和 4 个定点）与 9,030 次续跑搜索，共 35 个去重的双分支起点。12 个 Stage-176 未纳入的开发源组中 9 个满足每组两点的配额，各规则各选 18 点。归档原动作框与门控分误差均为零。四组均无有益换层、无新增失跟或丢失恢复。35 个续跑点的 128 帧平均 IoU 差范围为 −0.004266 至 +0.005294，33 个满足保护条件，收益均未达 0.01。

243 个扫描点原生均为 L9，选择器 Sigmoid 输出前两名分差均精确为 1.0。分差排序与均匀对照选中同一批 18 点，联合排序与历史响应选中同一批 18 点。相对原生 128 帧平均 IoU，均匀／分差为 +0.000382，历史／联合为 −0.000276；联合相对均匀 −0.000658，名义区间 [−0.00189,+0.000594]。`supports_reopening_pilot=false`。已核验九张表，并复算四种选择、配额、决策及汇总。回传指纹 `a149550250865cb0f22fe2eff6d6e99da80360dbaefc60a44c1237303695f624`。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE177_NATIVE_PROPOSAL_AUDIT.md)

下一轮建议先从同一批状态读取选择器末层 `fc2` 的 Sigmoid 前 logits，不修改现有路由和推理。核验各层分数是否有分散度、能否在已有同状态分支的真实后效上区分有益和有害动作。四个定点仍单列，若分数有差异但效果标签支持不足，再决定是否扩充状态采集。当前结果没有支持以 Sigmoid 输出分差触发换层。

### 0.169 Stage-178 选择器 Sigmoid 前分数审计（已完成）

L20 完成 265 个单帧原生前向，框、父框、层和 Hann 峰对齐归档。Stage-177 的 243 个状态中，Sigmoid 前分差 243 个不同值，范围 102.57～155.27，Sigmoid 后全为 1.0。全部 265 个状态的原始最高分均为 L9，只出现三种完整层排名。对 265 个六维 logits 中心化做 SVD，第一主成分解释 99.948% 方差，逐维标准化后仍为 99.947%。原生分差与排序候选相对原生的分数缺口相关系数 0.996。该描述性复算表明原始分数变化主要沿同一方向。

Stage-177 的 35 个留出续跑仍无达标有益或有害标签。按源组等权，原始分差与连续 IoU 后效的一致率为 46.3%（9 个源组、51 对）。Stage-176 的完整窗口里，跨起点的受保护收益一致率为 100%，支持仅 2 个源组、3 对；有害事件为 60%，支持 5 个源组、7 对。同起点按非原生候选的原始分区分有益或有害，一致率分别为 35.4% 和 35.0%（22 对／4 组、28 对／5 组）。这些诊断未支持使用原始分直接触发或选层。回传指纹 `941317bf0105565e62e01ee709be16c10ea595cfc6d7ac56403b3d7e9cb72630`。已核验七张结果表并复算 12 行分布和 48 行关联。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE178_SELECTOR_LOGITS_AUDIT.md)

下一轮建议用同一缓存按源组留出，在拟合组估计并移除 logits 的共同变化方向，固定比较原始分、残差分与均匀对照。分别检查跨时机和同帧层间的关联，继续独立报告标签支持，避免把残差的样本内变化解释为在线收益。若去除共同变化后仍缺少跨组一致性，则转查当前帧的目标匹配证据，保留原生默认选层。

### 0.170 Stage-179 选择器共同分量与残差诊断（已完成）

本机离线复用 Stage-178 的 265 点／26 源组，逐组留出，拟合组等权估计均值和第一主成分，比较原始分、中心化分、残差分与均匀。未拟合后效标签，新增搜索为零。完整窗口的 Stage-177 留出排序一致率由 46.3% 升至 48.1%，低于均匀 50%。Stage-176 有害换层的跨起点一致率由 60% 升至 90%，仅 5 个源组／7 对；排除定点源组剩 2 对，原始与残差结果一致。同帧候选层连续收益的一致率由 53.4% 降至 51.4%。

在 23 个完整六层窗口，各规则强制选择一个非原生层的条件性均值，均匀／原始／中心化／残差为 +0.00461／−0.01320／+0.00909／−0.00374。非定点源组的残差均值 +0.03523，剔除 0204 后约为零。结果指纹 `3f126d466ad37e39cb8c09102bc756148ce69153468f18137950a3645a0cc4fa`，本机输出 `output/stage179/selector_residual_audit_v1/stage179_results.txt`。已验证留出隔离和原始分数复算，并核验重读生成结果。 [实验和结果](docs/experiments/06_sgla_uav_tracking/STAGE179_SELECTOR_RESIDUAL_AUDIT.md)

下一轮转向当前帧的目标匹配信号。优先在同一裁剪、同一历史的已有回放起点上读取原生 CENTER 响应的主次峰差、峰位置与回归框的一致性，检查这些信号在留出源组是否能辨识“保持原生会失跟、备选能恢复”与“原生稳定、换层有害”。先用归档后效验证区分度，原生选层仍作默认。

### 0.171 Stage-180 当前帧 CENTER 匹配证据审计（已完成）

同 L20 完成 265 次原生单帧前向，原生框和 Hann 峰归档误差均为零，重算主峰框最大误差 3.05e−5 像素。沿用 61 条排序候选标签及 26 个候选集合，关联只用完整 128 帧。Stage-176 的 23 个完整窗口内，主次峰竞争识别存在受保护有益候选的一致率为 87.5%（4 个源组／6 对），识别排序层的挽救为 100%（2 组／3 对）。排除已知定点源组后，有益替代仅 0204 第 663 帧一个正例（1 对）。主次峰与排序层伤害的一致率 40%，联合分为 10%（5 组／7 对）。0089 第 1177 帧主次峰竞争高达 0.704，排序层仍阻断恢复（128 帧 −0.6354）。

Stage-177 的 35 个留出窗口没有达标有益或有害标签。连续 128 帧收益排序，主次峰、回归下降、近邻框分歧、联合分的一致率分别为 43.5%、44.4%、50.0%、42.6%（9 组／51 对）。原生端的峰竞争可提示备选供给，现有证据不足以触发在线换层。已核验七张结果表，按上游回传复算标签、分布及 95 行关联。结果指纹 `47a3ca149e922fc3f25ce11fb976c57ad0752aa99bbc3a9810db42c8e25be3f2`。 [实验协议](docs/experiments/06_sgla_uav_tracking/STAGE180_CENTER_MATCH_AUDIT.md)

下一轮建议在有归档后效的窗口里固定同一裁剪与 L6 积累状态，只计算原生层和既定备选层的当前 CENTER 图。直接比较备选是否减少峰竞争、回归分歧并支持一致的目标位置，先按源组留出对比原生端信号与候选相对信号。0089 第 1177 帧作为有疑难但原备选有害的核对点，0204 第 663 帧作为可挽救核对点，均与留出统计分开读。若候选相对证据在保护标签上有支持，再进入等额额外块调用的验证。

### 0.172 Stage-181 同裁剪双层 CENTER 对照（已完成）

同 L20 完成 61 个同裁剪、同 L6 的双层对照，每点增加一个块与一次 CENTER 头。原生响应图和预测框、备选预测框与 Hann 峰的最大归档误差均为零。58 个窗口有完整 128 帧后效。核验回传包决策与五张表的哈希，使用 Stage-180 已封存标签复算 61 条后效与 165 行关联。结果指纹 `a9863249c809b407f0dfd4281a14d8c86248f0a14d72d5397e5cfc96d6bd641a`。

Stage-177 的 35 个留出窗口缺少达标挽救／伤害，128 帧平均 IoU 差值在 −0.00427 至 +0.00529 之间。9 个源组／51 对中，原生峰竞争、相对联合分、Hann 峰增益、父框 IoU 增益的一致率为 43.5%、44.4%、38.9%、55.6%，均匀基准为 50.0%。Stage-176 排除既有定点源组后，14 个窗口只有 1 个挽救和 2 个伤害。0089:1177 换 L7 后 Hann 峰升高、主次峰竞争减弱，128 帧平均 IoU 却下降 0.6354。0204:663 换 L11 后 Hann 峰降低、峰竞争增强，128 帧平均 IoU 提高 0.7713。现有结果不足以开启在线换层。 [实验和结果](docs/experiments/06_sgla_uav_tracking/STAGE181_PAIRED_CENTER_AUDIT.md)

下一轮建议预先固定独立源组与取样规则，用因果可见的历史峰值下降、框位移或尺度突变富集疑难窗口，并从同源视频抽取稳定对照。以同一原生历史和裁剪为起点，分别执行原生层和 L7–L12 中各个非原生层的一帧干预，随后恢复原生路由并记录 128 帧后效。先按源组报告安全挽救是否存在、伤害发生率和当前排序覆盖了多少可挽救起点。若独立源组中有重复出现的安全挽救，再验证可用信号能否选中它们。真值用于续跑评价，原确认集继续封存。

### 0.173 Stage-182 疑难与稳定窗口六层续跑（已完成）

9 个源组完成 18 个有效窗口、108 个分支和 13,932 次跟踪调用，同 L20 下原生框、Hann 峰和路由复算误差均为零。六张回传表的哈希、起点排除、分组汇总和逐层结果均通过核验。决策指纹 `47cb1378551c0b083f6787e35cba1b95d73ff8c51ed3870a771a163a532767fa`。

疑难窗口 0/9、低波动窗口 1/9 存在安全挽救，既有排序选中唯一挽救。0091:61 原生第 92 帧失跟，L8 当帧 IoU 为 0.621，低于 L9 的 0.702，但 128 帧平均 IoU 提升 0.5332。0091:95 已继承同一次失跟，六层均无恢复。其余 16 窗原生后效平均 IoU 均高于 0.63，剔除 0091 后两组最优安全层平均提升约 0.002。90 个非原生比较有 50 个平均 IoU 差值为负，按预设条件没有新增窗末失跟或恢复损失。该实验未达到留出选层信号验证条件。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE182_LAYER_OPPORTUNITY.md)

下一步检验搜索状态反馈是否压缩换层的挽救空间。用逐帧裁剪覆盖、目标输入尺度、各层稠密网格最佳 IoU 与实际解码 IoU，定位裁剪退化、好框供给消失或解码错选的状态变化。对变化前后的相邻状态固定同一父框做原生层与备选层续跑，并另设恢复较早预测父框的裁剪对照。按独立事件和源组计数。现有 Stage-170 原生记录共 16 个视频的 19 个失跟事件，Stage-182 候选源组里只有 0016 和 0091 的两个事件。独立事件不足时先补充开发轨迹。真值指标用于离线诊断和评价，状态取样间隔由实际变化确定，原确认集继续封存。

### 0.174 Stage-183 搜索反馈与六层换层机制对照（已完成）

使用 Stage-170 原生的 19 个失跟事件，覆盖 16 个源组，其中仅 1 个同时属于 Stage-176 未扫描源组且事件未被 Stage-182 窗口重叠。扫描范围由最后连续好框与失跟确认确定，逐帧记录目标可见覆盖及尺度、六层的全网格最佳和常规输出。优先在好框供给变化的相邻帧，交叉对比当前父框、只恢复较早中心、只恢复较早宽高、恢复完整父框。继续在当前和完整恢复两组各六层提交起点输出，续跑原生至多 128 帧，按源组汇总。真值只用于离线诊断和评价。

同 L20 完成 399 个扫描状态、38 个对照点、912 条裁剪与层的交叉记录和 456 条续跑对照，40,938 次实际后继调用。已核验回传哈希，并复算标记、锚点、交叉指标、调用数和源组结果。决策指纹 `2879714ac87a3a7b5cc0e0f750b02026469561fe4d591342c5fbbc33a027bed3`。

变化后 19 点均有稠密好框，其中 14 点的原生 L9 网格有好框，18 点原裁剪完整覆盖可见目标。只恢复中心／只恢复尺寸／恢复完整父框使 9／7／12 点恢复常规好框；完整父框组受保护挽救从 2 个源组增至 4 个，完整 128 帧对照从 1 个增至 3 个，新增长窗挽救在 0091 和 0204。0204:665 原生 L9 常规 IoU 为 0、稠密为 0.9434，只恢复中心的 L9 常规 IoU 为 0.750，只恢复宽高仍为 0。0091:93 完整恢复父框后 L7 挽救，L9 仍无当帧好框。恢复父框组在 0307 与 0348 仍出现事件伤害。唯一同时避开既有扫描和事件窗口的 0016 只支持较短续跑，当前父框就有可挽救层。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE183_SEARCH_FEEDBACK.md)

下一步固定裁剪尺寸和层，交叉重组原中心和恢复中心的 CENTER 图及坐标映射，量化仅改映射、仅改网络图与完整中心修正的影响。继续做相同的写回续跑和稳定性对照，机制结论按独立源组复验。

### 0.175 Stage-184 固定尺寸和层的搜索中心分解（已完成）

绑定 Stage-183 的 19 个常规好框供给变化后状态，另从 Stage-182 的 9 个低波动窗口筛选稳定对照。稳定候选要求原生起点 IoU 至少 0.5、128 帧标注有效且无失跟。固定搜索尺寸、层和 Hann 窗，交叉当前与较早中心生成的 CENTER 响应、宽高、偏移图，以及解码中心。记录四组当帧结果与边际效应及交互，失跟状态续跑六层、稳定对照续跑 L9。后效均比较同一层下的当前父框分支。

同 L20 完成 19 点、8 个稳定窗口和 43,986 次额外后继调用。回传哈希、Stage-183 实际裁剪组合的 IoU 与格点、因子效应和汇总均已核验。决策指纹 `5391afd90afa6a9199066b71330294f2257311df747b450210152e33ee3ce2e9`。

L9 只改映射／只换图／完整修正的当帧平均 IoU 差为 +0.2172／+0.0040／+0.2050。15 个完整窗口中，按 14 个源组等权的后效为 +0.0247／−0.0743／+0.1446，交互 +0.1942，完整修正有 3 个安全挽救源组且无事件伤害。剔除贡献最大组后完整修正 +0.0961。8 个稳定窗口均无事件伤害。唯一此前未覆盖的 0016 只有 33 帧后效。0204:665 只改映射便可挽救。0091:93 只改映射与完整修正的起点 IoU 为 0.5268／0.5247，后效为 +0.0002／+0.6927，两组起点框高为 14.90／17.43 像素。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE184_CENTER_MAPPING.md)

下一步以 Stage-184 同层续跑为参照，交换只改映射与完整修正的起点框中心和宽高，用交叉续跑核对第一处裁剪、响应与选框分岔。按事件和源组统计能否复现完整修正的挽救，并检查稳定对照。独立长窗复验仍待补充。

### 0.176 Stage-185 写回框中心与尺寸对照（已完成）

绑定 Stage-184 L9 的 19 个变化点、8 个稳定控制。基于只改映射和完整中心修正的实际写回框，交换中心或宽高并沿原生路径续跑。保留原生当前裁剪的 L9 分支，用其衡量受保护挽救和事件伤害。同运行环境核对三个归档分支并复用完全相同起点的续跑。逐帧定位首个取整裁剪、中心、尺寸与输出分歧，对所有组的第一后续帧及候选首个响应变化重测原生 CENTER 图，核对网格好框、Hann 格点及输出。

同 L20 完成 27 点、6,594 次续跑调用及 137 次响应复测。七张表的哈希、Stage-184 共同分支、交互效应、首个响应分歧与汇总均已核验。决策指纹 `7101aaae1c4b38f14094df8f752546e9a55adb6d92f165cb3c68bad772a44027`。

15 个完整窗口中，按 14 个源组等权，只换中心／只换宽高／完整修正相对只改映射的后效为 +0.0894／+0.0298／+0.1199，复现完整修正的挽救源组为 3／1／3，其中 0204 的只改映射组已可挽救。0091 和 0176 只换中心新增挽救。0091:93 当帧 IoU 同为 0.5268；下一帧同 67 像素边长，取整左上角 `(146,343)→(147,342)`，Hann 格点 148→136，输出 IoU 0.0303→0.6146，续跑 +0.6972。只换宽高为 −0.0001。0348:361 只换中心延续只改映射的恢复损失，较原生 −0.4949；只换尺寸保住恢复。8 个稳定窗口均无事件伤害。唯一此前未覆盖的 0016 只有 33 帧后效，只换中心同样挽救。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE185_WRITEBACK_COMPONENTS.md)

下一步以实际连续中心与取整裁剪的变化作配对，固定同帧同层同尺寸，对照取整不变的亚像素移动、只跨一个坐标轴的取整边界、实际两轴变化和同幅反向裁剪。用网格供给、常规选峰和原生续跑区分空间量化敏感性与方向作用。保留 0348 的尺寸保护例和稳定对照，并按独立源组报告。

### 0.177 Stage-186 搜索裁剪取整与移动方向对照（已完成）

复用 Stage-185 的 19 个变化点和 8 个稳定窗口，以其只改映射分支为基准。第一后续帧固定模板、L9 和裁剪尺寸，对照窗口内亚像素移动、单轴跨界、原观测方向、等幅反向浮点中心及等幅反向整数像素。0348:361 单列尺寸保护分支。报告真实输出、统一坐标映射后的输出、Hann 选峰、网格上限和至多 128 帧后效。保存同环境的复现核对并复用完全相同的续跑，最后按源组及是否跨界分报。

同 L20 完成 217 次探针、16,350 次新增续跑。六张回传表、Stage-185 共享分支的格点、当帧 IoU 与后效，方向配对和源组汇总均已核验。决策指纹 `318e1a59f5322d3458f59c5b3fca8bb1240d7d648db7301e8ebbdedda0cb5741`。

27 个同窗亚像素对照的三张图和 Hann 格点完全相同；0176 与 0089 仅浮点回映变化仍分别出现长窗挽救和恢复损失。15 个完整变化窗内，实际／等幅反向像素分别有 7／11 个改选 Hann 格点；统一坐标映射下实际方向第一帧优势 +0.0972，恢复各自原映射为 −0.0048。相对等幅反向像素的 128 帧优势 +0.0427，中位数 +0.0008，去掉最大贡献源组约 −0.0002。8 个稳定窗口均无事件伤害。0091:94 的原裁剪 IoU 0.030，水平／垂直单轴、实际、反向像素分别约 0.674／0.674／0.615／0.643，全部挽救。0348 的尺寸分支保留恢复。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE186_CROP_QUANTIZATION.md)

下一步在同一候选池与搜索预算下，统一图像坐标并合并同取整窗口，检验跨相邻裁剪支持是否能选出更稳的候选。用既有续跑对照单裁剪和最高响应，报告不同源组、稳定窗口和恢复损失。仍需独立开发复验，确认集封存。

### 0.178 Stage-187 相邻裁剪预测框一致性（已完成）

沿用 Stage-186 的 19 个变化点和 8 个稳定窗口。在当前及等尺寸邻域分支中按取整裁剪去重，每组保留浮点中心最接近基准的候选。当前、最高 Hann 和跨裁剪一致性共用候选池。一致性为与其他不同裁剪实际输出框的平均图像坐标 IoU，遇平分优先当前。核验每个输出在其已记录续跑中的首帧完全一致。续跑指标来自原归档，无新增搜索和训练。

已核验 27 窗、134 个去重裁剪、81 个选择，按 Stage-186 后效和原图框坐标复算排序及六张结果表。决策指纹 `a187e3bb24c84c0127675b54a0d510271d75ddd40401b6036695ab879fba9781`。15 个完整变化窗有 3 个安全挽救机会，当前、最高 Hann、一致性分别选中 1／2／2 个，事件伤害为 1／0／1。8 个稳定窗均无伤害。相对最高 Hann，一致性当帧 +0.0199，128 帧均值 −0.0326，5 胜 1 平 8 负。唯一此前未覆盖的 0016 短窗有挽救候选，三法均漏选。去重没有抹掉任何窗口的全部安全挽救。

0176 唯一挽救分支的邻域框平均 IoU 为 0.385，失败的反向分支为 0.703，两种规则均选失败分支。0348:361 一致性选择邻域重叠 0.877 的框，当帧 IoU 0.205，高于最高 Hann 的 0.197，后效却相对原生下降 0.493；最高 Hann 相对原生只降 0.012，保住恢复。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE187_CROP_CONSENSUS.md)

下一步先补独立开发源组的失跟和稳定窗口，固定现有三条规则和候选池。验证此前多帧的外观匹配或位置轨迹是否给高度一致但会失跟的框提供额外可辨证据。按源组报告安全挽救供给、信号选择能力和稳定性，确认集继续封存。

### 0.179 Stage-188 补充开发源组与历史位置对照（已完成）

Stage-79 封存名册外有 40 个历史开发源组，Stage-93／170 的近期开发对照覆盖 28 个。剩余 12 个候选源组为 `0001、0002、0090、0107、0148、0149、0173、0199、0205、0236、0303、0331`。这些源组在早期实验或旧记录中已有接触，结果只作为开发补证。名册的 28 个确认源组仍封存。

每组先跑原生轨迹，然后比较首个具有完整后效的原生失跟起点与同视频可用稳定窗口。相同去重裁剪池内复用当前、最高 Hann 和裁剪一致性，并加最近 16 个预测框的中位速度外推。位移候选依据当前父框与上一个原生预测中心生成，和 Stage-187 的写回状态来源有别。各候选在测试帧写回，之后恢复原生选层和各自的搜索状态续跑 128 帧。

L20 实际运行了 10 个源组、11,470 帧。`0001`／`0002` 未入选。3 个原生失跟中，`0107:161`、`0173:502` 进入完整事件窗，`0331` 仅进入稳定对照。事件窗 2、稳定窗 9、去重裁剪 43，四条规则都没有新增事件伤害。两个事件窗的池内安全挽救均为零。历史位置相对最高 Hann 的后效为事件窗 −0.00317、稳定窗 −0.00020。`0107:161` 当帧最好 IoU 仅 0.198；`0173:502` 的最高 Hann 和历史位置当帧分别选到 0.860 和 0.837，仍未形成安全挽救。回传的决策和六张表均通过哈希核对，选择分数、汇总和配对已复算。服务器轨迹缓存尚未逐帧核查。决策指纹 `1c030e7fcec60b73aeefe900ed8310e10ebdabd86b5c68e574131c4eb022e8d8`。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE188_TEMPORAL_SUPPORT.md)

下一步从 Stage-188 服务器缓存检查 `0173:502` 当帧纠正后的再次低 IoU、恢复及持续时间，用 `0107:161` 作当帧候选不足对照。先查明 `0331` 未进入完整事件窗的资格原因，再决定是否报告截短轨迹。根据结果选择排查写回后的搜索状态或候选供给。

### 0.180 Stage-189 当帧选框后的轨迹持续性审计（已完成）

复用 Stage-188 原生轨迹、候选续跑与同版标注。已核对回传决策、代码和四张表哈希，并复算两窗 10 分支的 1,290 条逐帧记录。决策指纹 `6fd02d8652fd71a15952e2a5eee677bf081966a810286bd5b2652db4271d25a0`。

`0173:502` 的最高 Hann、历史位置、反向像素分支当帧 IoU 分别 0.860、0.837、0.855，均只维持一帧。503 帧 IoU 降至 0.091～0.094，中心误差升至 0.96～0.97 个真值尺度，宽为真值的 0.96～0.97 倍，高为 1.14～1.15 倍。505 帧确认失跟，到 630 帧窗末无恢复。相较原生失跟推迟一帧。`0107:161` 当帧最好 IoU 0.198，确有尺寸压缩和中心偏差，整窗无 IoU ≥ 0.5。`0331` 在 338 帧失跟，391 帧结束，只剩 53 个后续帧，未达 128 帧要求。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE189_CONTINUATION_AUDIT.md)

下一步在 `0173:503` 复测原生与三个高 IoU 写回状态的搜索裁剪、真值覆盖、CENTER 网格最佳 IoU 和 Hann 实际读出。用 `0107:161` 作对照。需要这些中间观测才能把中心跳偏归到搜索覆盖、网格供给或读出，确认集仍封存。

### 0.181 Stage-190 失跟首帧的裁剪与 CENTER 读出对照（已完成）

绑定 Stage-188 原始缓存和 Stage-189 决策，完成 6 次同运行身份的 L9 搜索，所有归档框精确复现。已核验决策和 CSV 哈希，指纹 `6567d6ed4f07724d05ab21094ab7fd777b72fda3c6c582b1ab0c018baaeea9b9`。`0173:503` 四种写回裁剪均全覆盖，网格最佳 IoU 0.851～0.866、首个好框的 Hann 排名第 2～3，实际仅 0.091～0.094。`0107:161` 两种裁剪覆盖 90.1%～91.2%，网格最佳 0.582～0.588、首个好框排名第 6～8，实际仅 0.193～0.198。六次的原始响应和 Hann 最高格点一致，单纯去掉 Hann 无法纠正这六次选框。两处失跟均存在漏选，`0107` 也有不完整覆盖。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE190_SUCCESSOR_READOUT.md)

下一步固定这两个起点的父框与搜索输入，把原生最高峰和真值挑出的网格最佳框分别写回，按原生续跑 128 帧。先确认现有好框能否带来持续收益，再决定验证候选重排或检查后继搜索状态。以真值挑框只作离线诊断，确认集继续封存。

### 0.182 Stage-191 网格好框一次写回对照（已完成）

绑定 Stage-190 的六种匹配输入，完成 6 次锚点和 1,536 次续跑搜索。全部匹配 Stage-190，Hann 在归档重叠帧的框误差均为零。已核验回传决策、五张 CSV 和 1,548 条逐帧记录，指纹 `f033641b48174b1b8a42fb945ad947cc5d1dd082db58be5b44d7986bf4733db3`。六对属于两次失跟事件，真值选框只作离线上界。

`0173:503` 四条最佳框次帧都失跟，三条在 547 恢复至 631 无复发，后 128 帧对各自 Hann 的增益 +0.559／+0.533／+0.558。`observed` 未恢复，−0.00061。该分支 546／547 帧 IoU 0.047／0，其余三条为 0.28～0.30／0.63～0.68。`0107:161` 两条均仅保持含写回帧在内的 5 帧好框，166 失跟后至 289 无恢复，后效增量 +0.00807／+0.00560。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE191_GRID_WRITEBACK.md)

下一步用 0173 的第 546～547 帧做成败分支匹配，检查覆盖、稠密网格与响应排序，再交换恢复与未恢复分支的缓存父框验证恢复能否转移。结果用于恢复机制诊断，确认集继续封存。

### 0.183 Stage-192 跨开发源组的网格候选与写回收益（已完成）

按用户最新要求，优先扩展跨视频证据。绑定 Stage-170 的 28 个近期开发源组和 Stage-188 的 10 个补充开发源组。依据原生轨迹，在读取候选之前每组取第一个有完整 128 帧后效的失跟起点，以及不重叠的稳定窗（若有）。稳定窗锚点及后续 128 帧均要求原生 IoU ≥ 0.5。有失跟窗时选最近的稳定起点，否则选合格起点中位位置。未入选源组保留资格计数。

各窗在同一原生网格对比 Hann 与真值最佳框的一次写回。38 个源组有 22 次原生失跟，17 次符合完整窗条件，按源组取到 16 个失跟窗和 36 个稳定窗。52 次锚点搜索、5,888 次新增续跑，Hann 归档框误差为零。已核验决策和五张表哈希，复算分组汇总。指纹 `0d39f60a743eb0d683f9b5a14693286574e6b55db82d7f5e128cefbe15905c41`。

失跟窗 10/16 有好框却漏选。真值写回受保护挽救 `0091`、`0173`、`0176`、`0204`，其中 `0176` 当帧最佳 IoU 仅 0.343。后 128 帧总体 +0.1343，去掉最佳源组 +0.0916。排除 `0173`、`0107` 后 14 窗仍挽救 3 个源组，后效 +0.1133，另剔除最佳组 +0.0624。`0348:367` 阻断已有恢复，−0.5276。`0085` 首好框排第 2 仍新增失败。36 个稳定窗无新增失跟，15 个后效下降，均差 −0.0063；`0307:527` 当帧 0.799→0.868，后效 −0.2232。这些是开发窗口的一次写回证据，确认集仍封存。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE192_DEVELOPMENT_GRID_RESCUE.md)

下一步检验可部署候选。固定裁剪及搜索预算，比较 Hann 首选与经空间分离的第二响应峰，核对能否复现 `0091`、`0173`、`0204` 的挽救，并记录对 `0085` 与稳定窗口的伤害。先确定真实候选的后效空间，再决定是否拟合在线选择规则。

### 0.184 Stage-193 同裁剪分离第二峰的一次写回（已完成）

在 Stage-192 的 16 个失跟窗和 36 个稳定窗上完成 52 次锚点与 5,888 次新增续跑。沿用 Stage-28 和 Stage-136 的局部峰与候选框抑制规则。锚点和真值后效复现，已核验决策、四张 CSV 及分组汇总，指纹 `b2e425200d456c40ad82b4e0e42255a70bbdc02dab415679fcbfc07a05372a17`。第二峰挽救 `0173`、`0204`，漏掉 `0091`、`0176`，`0348` 丢掉原生恢复。失跟窗均差 +0.0518，排除先导个案后 +0.0202，再去掉最佳组 −0.0378。稳定窗 5 个严重事件伤害、7 个新增失跟，均差 −0.1229。回传表复算旧 20% 分差门槛，局部保住两个挽救，拦下本轮全部严重事件伤害和新增失跟；Stage-136 已显示同门槛在全视频仍损伤稳定轨迹。 [实验结果](docs/experiments/06_sgla_uav_tracking/STAGE193_SECOND_PEAK_REPLAY.md)

用户随后重申必须同时改进 SGLA 的 Transformer 骨干与选层，下一阶段方案扩展为目标条件的时序特征适配、任务质量驱动的选层及 CENTER 质量评分。原生推理执行 L1–L6 后由 `MLP(x[:,:,0])` 在 L7–L12 中选一个独立块，单纯重排冻结块的既有尝试未通过完整视频稳定性检验。新的骨干适配、路由和头改动均须可训练并在正常入口消融，结构尚未实现。原生、仅适配、仅路由、仅头和组合分别比较全视频 AUC、持续失跟、稳定损伤与开销，保持确认集封存。原生 SGLA 在完整 UAV123 的早期复现为 66.5143，见 Stage-3 第18节，VisDrone 28组的 63.9605 属于另一评估集。

### 0.185 Stage-194 同预算内部模块消融（已完成，未建立采用收益）

38个历史开发源组，两折19 fit／19 held、两种子、9个训练组，36次正式训练、9个smoke、798条全视频、684条held读出、1824采样状态。回传 `stage194_results_20261125T083202Z_816395.txt` 已原字节归档，SHA为 `829ba345ea2e2bc63050c747cf3ecba009bbb10448f742ca4533c1505482db52`；决策指纹 `aecfc2112422a0cf5f1af3daa1ddc61bcce9f54fc4c3f2ae94f762429a9dd547`，plan指纹 `69ce5be3cf224c940d203698d07049760a4018e9928dee31270babdd9f8a27e0`。11表哈希通过，12 policy、115 paired、3 factor全部复算最大误差0。每折／种子各组draw哈希一致，256更新、1024例，梯度范围及冻结记录一致。服务器3207个原始产物只有哈希清单，本机未逐个重验模型和NPZ。

原生与固定L9 AUC均64.4184514。仅Q 64.7794047（+0.36095，CI[−2.2907,3.7465]，8胜14负16平），仅A 63.1884277，仅R 58.2466101，AQ 62.4349995，RQ 56.9212393，AR 56.5513408，ARQ 56.1598745。current适配62.7787463，current三模块55.7202066。预注册主比较ARQ−currentARQ +0.4396679，CI[−2.5792,3.2955]，26胜12负，剔除最佳组−0.29436；A−currentA +0.4096814，CI亦跨零。

R四背景因素效应均负，平均−6.73555。含current的20个R模型15个整段固定一个层，另4个主层比例≥94%，全部没有选过L9。两折fit均匀样本最佳固定层都是L7，合并即时IoU L7 .6666083、L9 .6572459；完整固定L7仅58.4363487，较L9 −5.9821028。该对照支持即时原生状态标签不能直接决定自身闭环最佳路径，尚未拆清瞬时目标与状态分布偏移各自责任。

仅Q的0176平均+49.48694，剔除该组其余37组−0.9667762；新增0080短失败及0304未恢复失败（0304由72.2871降至43.8454）。仅A的0304平均−50.0883、0307−27.9346。Q不能按小幅宏均值提高采用，目标条件与三模块协同也尚未成立。first/final loss为不同batch，不能据此判收敛。

下一步建议Stage-195复用模型／缓存，冻结新模块，审计全fit／held池的初始化与训练后MSE、每层常数控制、跨层共同难度与中心化残差、成对排序及相对L9 regret，再根据结果确定相对效用或自身状态训练；不能直接重复Stage-158～170的收益标签路线或未经依据加入记忆、PPO。确认集仍封存，尚未实现Stage-195。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE194_RESULTS_REVIEW.md)

### 0.186 Stage-195 冻结路由目标与优化审计（已完成）

回传 `stage195_results_20261125T090931Z_1082854.txt` 已原字节归档，SHA256 `08a26fd13c1a400ac0141c40edb8ac2f72574aa07f5dbc48c2cb22ba18e6b977`；决策指纹 `a282dbfb2ca856e5a2fe4009356a10e35a4784a011837fa8cf2381bd1622b764`，plan `a4ebb55b99ec61c17489113af42dd8c3ef6c6e602b24a62161e6347653a8a2a5`。9表哈希相符，100误差汇总、760视频误差、14策略汇总、1064视频策略、16配对、7296预测、76固定路径、1024训练history和4scope齐全。按float32预测均值重算，误差汇总最大1.11e-16、视频策略2.22e-16，其余0。627个服务器来源哈希与Stage-194清单一致，本机未取得原始模型／NPZ。CPU动作与原GPU完全一致，分解误差2.797e-17。0新增训练／图像／骨干／确认组。

4模型在实际1024次抽样上的初始／训练后／抽样均值常数MSE为 .107595/.093958/.096592、.105884/.088348/.090313、.098019/.059910/.061966、.097654/.062207/.064813。模型确实学习并超过常数，不能称完全未拟合；99.149%／99.345%／99.970%／99.973%的MSE下降来自共同均值误差。held标签共同分量方差占比两折97.495%／98.343%；held相对零残差skill为−2.381%、−1.703%、+.311%、+.929%。分解不构成共同项导致退化的因果证明。

held平均IoU：R .655652，L9 .657246，fit全池均值选层 .666608，fit预测均值和实际抽样均值均 .655514，oracle .697494。R对预测均值仅+.000137613，CI[−.000017353,+.000421516]，2胜1负35平；对全池均值−.0109563，名义CI负。全fit池偏L7，而训练抽样和预测均值都偏fold0 L7／fold1 L10，支持主要学到抽样平均分支偏好。76条原有R轨迹69条固定（38 L7、31 L10），有控制的38条L7（19源组双seed）逐帧相同；L10无独立固定控制。完整AUC仍引用Stage-194，不能把缓存IoU当新AUC。

下一步建议Stage-196同架构同输入同draws同预算，对比绝对MSE与 ||P(p−y)||²/6（P=I−11ᵀ/6），两组同设备重训且保留原Sigmoid和参数，先检验held差异误差／排序／对常数增量，再决定完整闭环。已有158～170差值路线需区分，本次针对共同项机制的因果对照，不预设有效或新颖；尚未实现。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE195_RESULTS_REVIEW.md)

### 0.187 Stage-196 同预算absolute／relative路由损失（已完成）

原回传 `stage196_results_20261125T094605Z_1113564.txt` 已归档，SHA256 `088948bcf075f61efc7cad32f705f870e8388e1c809148f339635ca0ef2f4d85`，决策 `b2769871bf752da66efd6d5a6bb1f2f211affd3e18b0105113e27429eff9bc91`，plan `6453511ab517f010f0bc79bee7222adc8f3e8e2be881bcb799956e6e37626121`。9表哈希相符，8模型、200误差汇总、1520视频误差、28策略汇总、2128视频策略、40配对、14592预测、2048history、152bridge。复算误差汇总／视频策略≤2.22e-16，其余0；初始、draw、256更新、1024例、4614参数均匹配，A/Q关闭。163来源哈希与Stage-194一致，624模型／预测引用无本机原始字节复核。

held IoU absolute .6558270836，relative .6543384867，主差−.00148859696，名义CI[−.00311981516,−.000201583585]，7胜8负23平，两seed均负。native L9 .6572458709、fit全池均值 .6666083461、预测均值 .6555144089、oracle .6974937197，均为缓存IoU不是AUC。relative对自身预测均值−.00117592228，CI跨零，无逐帧增量。

训练相对MSE由.00147850161降至.00143986795，held反而.00155395258→.00158268857（31组变差），差值加权排序held54.5243%→48.8184%，下降5.7059个百分点，30组变差。动作更动态但held421/3648次改变后整体更差，fold0两relative仍全L7。不能把去共同项作为已成立修复；也不能认定所有相对学习无效。absolute262/1024步裁剪，relative0，损失干预同时改变梯度尺度／裁剪行为。新CPU absolute对旧来源20/7296动作不同，max分数差5.07e-5，主比较仍两个新CPU组，原路由／预聚合误差0。

下一步建议Stage-197检验新增路由接口的空间布局可辨识性，同参数同预算比较保留token顺序与逐样本无序／排序读取，加现有全局统计及固定均值控制。全局矩统计对顺序不敏感，但token含位置编码，不能声称全部位置丢失；固定全局置换可逆，不能当丢失布局对照。需区别Stage-164旧4×4匹配16维失败路线。该方案现已由Stage-197实现，见0.188；不进入本relative组的全视频采用验证。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE196_RESULTS_REVIEW.md)

### 0.188 Stage-197 显式token顺序的同参数读取（实现协议，结果见0.189）

用户授权继续后，新增`stage197_spatial_router.py`、`stage197_spatial_router_model.py`、`test_stage197_spatial_router.py`和`run_stage197_spatial_router.sh`。理论变量为全局矩统计的排列不变性，不预设它已经造成失效；token仍有位置编码与上下文，原SGLA的`x[:,:,0]`保留顺序，当前全局统计是项目新增接口。须区别Stage-164旧4×4局部匹配失败路线。

保留g=[模板均值／std、搜索均值／std]768维，增加每token共享192→1线性投影，w初始化首通道单位向量、b=0。ordered保留320个值的原排列，unordered每样本模板64／搜索256分别排序，两组均1088→6＋Sigmoid，6727参数、初始参数哈希相同；两组独立学习。global_stats只有768→6、4614参数，为同次重训的原接口控制，容量不同只作次要参照。固定全局置换可逆，不能用作无序控制。

复用Stage-194原生父框L6特征与六分支真实IoU、38开发源组19／19两折、2132／2133两seed，三组12模型，每模型256×4同draw、AdamW／lr1e-5／wd1e-4／clip1／absolute MSE，3072步、21888输出。0新图像／骨干／完整轨迹／确认组。绑定194／195／196已复核清单与原代码，source196指纹b2769871bf752da66efd6d5a6bb1f2f211affd3e18b0105113e27429eff9bc91。global_stats另与196 absolute作复现桥接，原阶段不修改。

10表：model_scope、input_audit、error_summary、error_video、policy_summary、policy_video、paired、frame_predictions、training_history、source_bridge。主比较held ordered−unordered mean_iou；另看排序、相对误差及超过fit标签／预测固定均值的增量。先源组内平均种子再bootstrap，名义探索区间，不能称新AUC／外部验证。若两组同优于global_stats，不能独立归因空间；未过同参数与常数控制，不直接上完整采用验证。保存模型／预测receipt、完整哈希、独立manifest，支持复用，单一总体进度条及100%终点。

真实缓存只在服务器，本机7项CPU测试通过，包含38组合成数据的12模型／3072步端到端、排序不变性、投影梯度、全局控制与196 absolute训练一致、模型及缓存复用／篡改拒绝、CUDA禁止初始化。Bash `-n`、Python语法、文档链接及`git diff --check`通过，不能预报真实结论。服务器在`sglatrack-test`和`~/workspace/paper`运行`bash tools/run_stage197_spatial_router.sh`；默认root=`output/stage197/spatial_routing_interface_v1`，源root可分别设置STAGE194_ROOT／STAGE195_ROOT／STAGE196_ROOT。后台命令、检验与结果包见[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE197_SPATIAL_ROUTING_INTERFACE.md)。完成回传`stage197_results_<UTC>_<PID>.txt`即可；不代commit／push。

### 0.189 Stage-197 空间顺序对照结果（已完成，REVIEW）

回传`stage197_results_20261125T102154Z_1143642.txt`已按原字节归档，SHA256 b824c702af4a80f15ecebd418f4b55f3cdf3cfa8309c780368b1645220cc18b6；决策4b0f88097a9f748eb12d58a53e458b6e53d77d0016da7a001f3b8c7ea026b7b0，本机重算决策指纹及10表哈希相符。12scope／38input／300误差汇总／2280视频误差／42策略汇总／3192视频策略／72paired／21888预测／3072history／152bridge齐全。全部汇总从逐状态复算，最大差2.22e-16；draw次数、fit/held分离、同6727参数及初始化、梯度和256步均核验。163来源引用与194清单相符，936新模型／NPZ哈希引用无本机原文件核验。global_stats全部7296分数／选择／真值与196 absolute完全相同。计划指纹43a49f31db5532f7d7a934cdc6aad18f3b9e3a1869dd0c244956db10d137fb53只由报告引用，未收到manifest正文。

held即时IoU ordered .6592832300，unordered .6591530844，global .6558270836，固定L9 .6572458709，fit标签均值常数 .6666083461，ordered预测均值常数 .6592483744。主差+.00013014564、名义CI[−.00010678033,+.00039024377]、6胜4负28平，两个seed为0／+.0002602913，未通过协议条件；held排序有小幅信号55.5597%对55.3064%（+.253365个百分点、名义CI正），相对误差−1.29528e-6。不能概括为空间完全无价值，也不能称已证明动态选层有效。

ordered对自身预测均值常数仅+.0000348556，CI跨零；对fit标签常数−.007325116，CI负。ordered−global总+.003456146 = 固定预测均值常数差+.003733965 + 动态增量差−.000277819。三个折／seed单元三组动作全相同，差异集中fold1／2133：ordered以L8为主829/912，unordered860/912，global以L10为主869/912。新增两组的fit预测均值常数在该单元都已选L8，global仍L10。ordered/unordered held仅83/3648动作不同，主要改善是固定偏好变化，不是跨种子可靠的逐帧识别。梯度clip次数450／446／262，接口变动也影响优化。0新图像／骨干／完整轨迹／确认组，不进入当前组全视频采用验证。

下一步建议Stage-198复用预测，以fit源组等权均值μ_y、μ_p构造s_t=μ_y+(p_t−μ_p)，检查校准后残差是否超过固定μ_y，再比较视频内同动作频率、状态无对应的精确期望Σ_lπ_lȳ_l。这与196去行共同项不同，专门分开列均值偏好和状态×层交互；不增门槛／缩放参数／训练或推理。held真值仅评价无对应期望，不改变在线规则。若仍无状态对应增量，记录有效交互未学会再决定新任务条件表征／监督，不能直接加记忆／更深模型。该建议已由Stage-198完成，见0.190。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE197_RESULTS_REVIEW.md)

### 0.190 Stage-198 fit均值校准与状态对应诊断（真实缓存已本机完成）

用户授权后新增`stage198_routing_calibration.py`、`test_stage198_routing_calibration.py`、`run_stage198_routing_calibration.sh`及固定协议。真实来源为197原包，SHA b824c702af4a80f15ecebd418f4b55f3cdf3cfa8309c780368b1645220cc18b6；只需完整分数表，故本机直接完成，非合成研究结果，服务器不需要补跑。先保存协议再运行；协议SHA 3e6013e254e94718ce7dd30f2611fe8c872e996e04c9903d0544defde66c698c，保留原文不随结果修改。

μ_y及μ_p仅从fit视频等权均值得到，p及μ_p保留197 float32约定，s=μ_y+p−μ_p用float64。12模型72列偏移，无新梯度更新。比较raw／raw_redecoded／calibrated／fit_label_mean／fit_prediction_mean／L9／oracle；新策略并列规则保留L9，raw动作原样保留并报告重解码差。无对应期望Σ_lπ_lȳ_l保留视频内动作频率，实际差等于选择指示与各层IoU总体协方差之和，held标签只作评价，不进入决策。

运行输出`output/stage198/fit_mean_state_correspondence_v1`，决策447f9394ced978f2c3586b28b52ca89327d1c2cb05533b7990147f4732d93e34，plan 11d884a49bc372bdb81da4463e13be7849807c482fe9a7070273f37cae17b0c3。9表：12scope／72calibration／3192policy_video／42policy_summary／82paired／1824error_video／24error_summary／21888frames／456bridge。完整包`stage198_results_447f9394ced9.txt`已归档，SHA 0f3bcebd12370081e9e8a2319bb12458a537a1cdffe095491c422f2a2024ae77，含manifest与9表。0新训练／网络前向／图像／完整视频／确认组，无Torch导入，未复核原PT／特征NPZ字节。

held原ordered／unordered／global IoU .6592832300／.6591530844／.6558270836；校准后三者与固定fit标签均值全为.6666083461，两个seed相同。全部10944 held模型状态都选L7，最小分差.000340958，非并列规则造成。fit侧ordered14/3648、unordered6/3648改选L10/L11，不能说全部21888状态都固定。calibrated对fit常数0，状态对应差仅2.78e-17浮点误差。raw状态对应增量ordered+.0000755572、CI[−.000137230,+.000329902]，另两组同样跨零。oracle即时.69749372及状态对应+.02779945只为事后空间，未证实可学习／闭环持续。

5项本地测试通过，含源篡改、fit/held隔离、精确期望与协方差、合成端到端、禁止Torch和产物复用；真实9表、code与协议哈希通过，全部分数独立复算差0，逐视频七策略全48循环移位的期望差≤5.55e-16，协方差恒等差≤4.44e-16，raw重解码动作差0。Bash语法通过，保持总体进度条。结果不是新AUC或可采用固定L7，194已知其完整轨迹受损。

下一步建议Stage-199固定fit均值为先验，直接训练带符号的层间状态残差，目标P(y−μ_y)、分数μ_y+P fθ；与同参数现有回归控制及固定μ_y比较，fit统计可用性、初始决策和draw／预算均匹配。区别196去行共同项与198后处理，不预设残差参数化有效／新颖；先检验held状态对应增量再决定完整轨迹。199现已完成，结果见0.192；不调本轮缩放／门槛。 [结果与复现](docs/experiments/06_sgla_uav_tracking/STAGE198_RESULTS_REVIEW.md)

### 0.191 Stage-199 固定fit先验与中心化层间收益训练（实现协议，结果见0.192）

用户授权后新增`stage199_residual_model.py`、`stage199_residual_routing.py`、`test_stage199_residual_routing.py`、`run_stage199_residual_routing.sh`和运行前协议。198后处理没有held动态增量，本轮改为训练：rθ=W[g−ḡ;(F−F̄)w]，F̄、ḡ、μ仅由fit源组等权确定且冻结。ḡ必须为E[g]，不替换成g(E[F])。无偏置投影192→1＋1088→6，共6720参数；原6727中7个中心化后冗余偏置三组一致删除，不拿旧组充同容量对照。

absolute_sigmoid输出sigmoid(logitμ+r)，absolute_additive输出μ+r，均优化绝对MSE；interaction_residual输出μ+Pr、优化mean((Pr−P(y−μ))²)。P=I−11ᵀ/6。初始w=e0、W=0，三组完整初始参数／buffer哈希相同，初始决策都固定μ；Sigmoid重构分数有float32误差，单报。前两组比较输出形式，后两组在同参数下argmax等价，检验训练中删除共同项的影响。原198旧分数不继续调缩放／门槛。

源为完整Stage-194特征／标签／receipt及`stage198_results_447f9394ced9.txt`，198包SHA 0f3bcebd12370081e9e8a2319bb12458a537a1cdffe095491c422f2a2024ae77，决策447f9394ced978f2c3586b28b52ca89327d1c2cb05533b7990147f4732d93e34。核验198九表与manifest、旧代码、全部真值与194原生父框六层标签一致，并保持原38源组／19-19两折／2132-2133／256×4／AdamW 1e-5 wd1e-4 clip1，同draw哈希。只训练小路由CPU，0图像／原骨干／CENTER／完整轨迹／确认集。

12模型3072步21888预测。10表含model_scope、fit_summary（2折统计）、source_bridge（38组标签对齐）、error_summary、error_video、policy_summary、policy_video、paired、frame_predictions、training_history。共同主要看held residual对prior、状态无对应期望、absolute_additive的IoU，附交互误差／zero-residual skill／排序、种子和留一源组。固定prior／fit预测均值／L9／oracle均保留。交互MSE代数等同相对MSE，能量基准另列，不能当独立指标。fit raw均值及加性分数均值1e-5数值容差，Sigmoid非线性均值漂移单报。

6项本地CPU测试通过（同初始化／6720参数、中心化和held隔离、投影与梯度恒等式、训练／buffer／复用、来源／缓存／模型篡改、38组合成12模型完整流程及禁止CUDA）；新增均值约束后端到端再次通过。实现时本机没有194特征NPZ，仅完成合成验证；服务器真实训练结果现已复核，见0.192。旧1～198 Python及结果冻结。

服务器`sglatrack-test`／`~/workspace/paper`执行`bash tools/run_stage199_residual_routing.sh`；默认root=`output/stage199/fit_centered_interaction_v1`，仅可改STAGE194_ROOT和STAGE198_BUNDLE来源路径，STAGE199_ROOT独立输出。原生checkpoint／GPU都不需要。总体进度条＋后台定时更新，完成结果包包含manifest／decision／10表。运行与后台命令见[协议](docs/experiments/06_sgla_uav_tracking/STAGE199_FIT_CENTERED_INTERACTION.md)。若通过缓存条件，再考虑正常自身轨迹；未通过不加训练步或后验门槛。

### 0.192 Stage-199 固定先验残差训练结果（已完成，REVIEW）

原回传`stage199_results_20261125T113420Z_1204543.txt`已归档，SHA 12fe41d3e3643e937bbabe922f0253f53788185fb89a82095e9182ecf0ade8df。决策6e98e01801e9f63f34d1a6d97ffb769a34d45566e698e608817805dfa41f10a5，plan 0b73e5c12d53685cae7228f94baea461231cf915503f9f3416fda1d5afbd1e06，二者指纹、10表、代码及原协议SHA相符，原协议不修改。12scope／2fit_summary／38bridge／18error_summary／1368error_video／30policy_summary／2280policy_video／66paired／21888frames／3072history齐全。按帧重建控制、draw及计数和所有统计，最大差4.44e-16；全循环动作移位独立复算无对应期望差≤5.55e-16。初始哈希／选择／6720参数／1088输入／draw匹配；梯度非零，scope报告buffer不变，fit raw均值<8e-10，加性均值误差<1.5e-7，直接／预聚合误差0。163原194引用相符，940新模型／NPZ引用无本机物理文件核验。

held IoU Sigmoid .66646213、additive .66140912、residual .66138895、固定μ/L7 .66660835、L9 .65724587。残差三主比较：对prior−.00521940（CI[−.01280567,+.00161716]、12胜24负2平）；状态对应+.000246217（CI[−.000570557,+.000958276]）；对additive−.0000201672（CI跨零）。三项均未通过正向要求。两个种子residual .66033892/.66243898均低于prior。不进当前组完整视频采用，不加步／缩放／门槛。

训练已生效：residual实际draw相对固定μ基准MSE下降18.37%～28.85%；fit排序77.749%、held54.909%，低于prior59.295%；fit交互宏skill+21.65%、held−4.49%。held1812/3648选择非prior，六层均被选择，不能解释成没有动态或没有梯度。clip次数Sigmoid2／additive584／residual8，属于同配方下损失干预，不能认定唯一优化病因。0图像／骨干／完整轨迹／确认组，仍为缓存IoU，不是AUC。

探索性视频分量分解：残差fit视频均值误差.000267407→.0000534381，视频内变化.001207604→.001135497；约74.795%总下降来自视频均值分量，视频内变化只降5.97%。held均值.000361556→.000383227、视频内.001207604→.001218822都变差。支持跨视频迁移未建立及检验视频内监督，不证明直接记忆视频编号、环境成因或增加容量必有效。

下一步建议Stage-200同视频两状态差分监督，使用E||e_t−e_s||²/2=E||e−E[e|g]||²去视频均值误差；1/2来自恒等式，不用held调权。同参数点监督控制必须同次用相同状态对重训，每更新2对共4状态引用，保持1024引用／256步及相同fit统计；不能以本轮旧点监督替代匹配控制。先看heldIoU、状态对应增量、双seed与源组一致性再决定闭环。不新增在线历史输入或开确认集。200已完成，真实结果见0.194。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE199_RESULTS_REVIEW.md)

### 0.193 Stage-200 同视频差分与匹配点监督（固定实现，真实结果见0.194）

用户授权并要求继续后，新增`stage200_pair_common.py`、`stage200_within_video.py`、`test_stage200_within_video.py`、`run_stage200_within_video.sh`与运行前协议。唯一监督干预：e=P(score−y)，point优化mean(e²)，difference优化mean((e_t−e_s)²)/2。同视频t/s独立均匀、有放回，自配对保留，系数1/2由条件方差恒等式确定。没有帧间隔选择／失败窗口／新门槛／在线历史模块。

两组均用原199 interaction_residual架构：有符号层间残差、fit均值先验和中心化ordered输入，6720参数、同初始权重／buffer／决策、同完整fit统计。38开发源组19／19两折、2132／2133种子、原AdamW lr1e-5 wd1e-4 clip1。每步2对共4个状态引用，每模型256步／512对／1024引用；8模型2048更新8192训练引用、4096模型对引用；独立抽样日程4套共2048行，由两组共用。模型8×1824=14592评价输出。point必须本轮匹配重训，不拿旧199代控制；新按对抽样日程不要求等于旧逐状态draw哈希。

来源只需完整194特征／标签目录与仓库内199原包。199 SHA 12fe41d3e3643e937bbabe922f0253f53788185fb89a82095e9182ecf0ade8df，决策6e98e01801e9f63f34d1a6d97ffb769a34d45566e698e608817805dfa41f10a5，plan 0b73e5c12d53685cae7228f94baea461231cf915503f9f3416fda1d5afbd1e06。核验199 manifest／10表／代码及原协议，194 NPZ与receipt按原清单验证，全部真值和fit先验对齐。0新图像／原骨干／CENTER／完整轨迹／确认源组，CPU小模型训练，不用SGLA checkpoint或GPU。

12表为model_scope、fit_summary、source_bridge、pair_schedule、pair_support、error_summary、error_video、policy_summary、policy_video、paired、frame_predictions、training_history。日程记录每对g、索引／帧号、自配对，支持表记每视频引用、唯一状态和自配对数，逐状态计数来自本轮对。视频误差分解字段`video_decomposition_error`独立于原MSE恒等式字段。主要判断held差分对point、prior及无状态对应期望IoU；附视频内误差、排序、双seed／留一源组，不自动采用，也不称新AUC。

7项本地测试通过：真实199包身份、配对预算／范围、自配对、穷举状态对方差恒等式、视频级偏移与梯度抵消、6720初始先验、两组训练／buffer／复用／篡改、38组合成完整8模型2048步和12表、计数重建、禁止CUDA及结果复用。合成特征小维数，实际192维特征未在本机训练。本机缺194真实NPZ，真实200待服务器执行，旧1～199 Python和证据冻结。

服务器`sglatrack-test`／`~/workspace/paper`执行`bash tools/run_stage200_within_video.sh`；默认root=`output/stage200/matched_within_video_v1`，来源可设置STAGE194_ROOT／STAGE199_BUNDLE，独立输出STAGE200_ROOT。单一总体进度条及后台定时刷新，结果自动包为`stage200_results_<UTC>_<PID>.txt`（decision／manifest／12表）。不重跑198／199。 [协议、验证标准和后台命令](docs/experiments/06_sgla_uav_tracking/STAGE200_WITHIN_VIDEO_DIFFERENCE.md)

### 0.194 Stage-200 真实同视频差分结果（复核完成，未采用）

原回传`stage200_results_20261125T140231Z_1329509.txt`已按原字节存results，SHA 00a487b9552785b6a4eb9f2a64a7477591e4ddbed32b5d348df362026a8d8838。决策b9b4aafb6a648a38d54347e76dd1e2138f8ad9b188cfbea51c962bb59b30f2dc，计划8792d5489f5d1711191ea202e4c2d550bbc310fa17f72615a308fd68f437522f。38历史开发组、两折两seed、8模型2048更新8192训练引用4096模型配对、14592缓存输出，0新图像／骨干／完整轨迹／确认组。原协议与1～200代码保持冻结；本机无原L6 NPZ／PT，628条物理产物仅核对声明。

决策／manifest／12表／代码与协议哈希均相符；163来源引用与已审199相同。全部14592逐状态、2048抽样对、76支持行和引用次数重建一致；误差表重算差0，视频策略≤4.44e-16、汇总≤1.73e-17、38配对≤2.78e-17；48循环动作移位的无对应期望独立验证≤4.44e-16。两组同初始／6720参数／1088输入／抽样／fit统计，history与梯度记账相符；点目标下降19.31%～22.68%、差分目标下降11.66%～12.19%，不同目标下降率不能直接比较。

held point／difference／固定fit先验IoU .66061429／.66165173／.66660835。difference−prior −.00495661，CI[−.01048812,−.00003010]、12胜24负2平；difference−point +.00103745，CI[−.00203838,+.00428195]、19胜18负1平。差分状态对应增量+.00086284，CI[+.00015161,+.00161365]、21胜14负3平、最小留一+.00065229；两seed +.00097471／+.00075096。这个预设比较有名义正信号，但相对point的对应增量差区间仍跨零，不能称差分已优于点监督。差分held约50.16%状态偏离L7、六层均有选择，不是未换层。

差分fit排序67.11%→held56.16%，仍低于held固定先验59.30%；point为77.31%→54.93%。差分held视频平均偏好MSE .00037253、视频内变化MSE .00120883，均高于零残差基准.00036156／.00120760；与point的总MSE、视频内MSE和排序差区间均跨零。描述性恒等分解：相对prior −.00495661 = 选择比例无对应期望负项−.00581945 + 状态对应正项+.00086284；负项尚未证明由均值漂移唯一造成。不能把缓存IoU当AUC，不能据先验优势采用固定L7（194完整轨迹曾−5.98 AUC）。

下一步建议Stage-201分数基准可识别性诊断。差分目标对每视频时间恒定的层偏置不变，只约束变化，整体fit均值并不能固定每个新视频的基准。先用现有point／difference预测缓存，同规则比较原分数／prior、全视频预测均值去除（未来信息，只作离线参考）、严格过去预测均值去除（首状态prior、不挑新窗口／λ）；不使用held真值校准。共同查看对prior与自身无对应期望的收益、双seed／源组一致性和因果估计差距；离线也不改善则降低均值校准假设优先级，仅未来参考改善则不可部署，过去估计稳定改善后才预注册闭环。201已在本机完成，结果见0.195，无需服务器补跑。 [详细复核及建模依据](docs/experiments/06_sgla_uav_tracking/STAGE200_RESULTS_REVIEW.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage200_review_audit_20261125.json)

### 0.195 Stage-201 分数基准可识别性诊断（真实缓存本机完成，无需服务器补跑）

用户授权继续后实现`stage201_score_baseline.py`、6项测试、单一总体进度条shell与运行前协议，随后在本机复用真实200文本包完成，不需要原L6 NPZ／PT或GPU。8原模型、38历史开发源组、两fold／两seed、14592模型状态；0新训练／模型前向／骨干／图像／完整轨迹／确认源组。固定三种分数：raw、μ+p−视频全均值（含未来，仅离线参考）、首状态μ而后μ+p−严格过去均值（每模型／视频独立重置，不挑窗口／权重）；另报prior、L9、oracle。

决策7984550b318cc7f5c0cf7c793adb91ffadde792870c6b099ab848cc2bbdeeccf，计划1ed6c2cc8f778bd08e3d60e102c7f5c260bbda1da151196c71fd0b6199b73ba7。完整结果`stage201_results_7984550b318c.txt`原字节存results，SHA acdc3d9b4d5e6cf051b03d26a37813eaa1ee3982ca424880e190e59840b27fd8。9表source_audit8／calibration_video304／policy_video1824／policy_summary24／seed_summary48／paired38／error_video1216／error_summary16／frame_predictions14592。200来源12表／代码／原协议和201哈希均核验；逐状态分数／动作／IoU、过去范围、视频／汇总／种子及38配对bootstrap重算差0；循环无对应期望误差≤5.55e-16，误差分解≤1.74e-18，离线均值恢复误差≤9.99e-16。6项测试含未来扰动／前缀不变、视频偏置不变、held标签隔离、完整合成报告／复用／篡改拒绝、禁止Torch与图像依赖。实际进度306／306。

差分held raw／offline／causal IoU .66165173／.66079708／.65945720，prior .66660835；对应增量+.00086284／+.00037464／+.00005178。causal对prior −.00715115，CI[−.01152665,−.00327181]、9胜22负7平；对raw −.00219453、CI跨零；自身对应增量CI[−.00059900,+.00065810]、17胜14负7平。offline对raw −.00085465、CI跨零，对prior −.00581127、区间为负。两seed校正后IoU均下降；causal对应增量+.00048148／−.00037792，不挑seed。

point raw／offline／causal为.66061429／.66181104／.66074537；offline对应增量+.00128162名义区间为正，但对raw IoU区间跨零、仍低于prior，不能替代差分主结果。差分离线去均值把视频偏好MSE .00037253归还prior .00036156，视频内MSE .00120883保持不变，IoU仍不升；causal视频内MSE .00121547更高。降低“只需校正均值”的优先级，不宣称所有校准无效或均值为唯一病因。均为原生父框抽样即时IoU，不是新AUC／自身历史闭环，offline含未来，不部署。原物理PT／特征NPZ未在本机重新核验。

下一步建议Stage-202直接定位遗憾监督，同输入／6720参数／fit先验／初始化／配对／AdamW／256步1024引用，同次重训point MSE与结构化损失。令l*=argmax y，Δ_l=y_l*−y_l；L=max_l(s_l+Δ_l)−s_l*，对精确argmax的实际定位遗憾有Δ_a≤L，保留1e-8近最大值规则则≤L+1e-8。间隔由真实IoU差确定，无新温度／窗口／手调margin。标准代价敏感损失本身不称新理论；检验其对决策的对齐价值，不预认定MSE是根因。先held对匹配MSE／prior／自身无对应期望及源组／双seed一致性，再闭环；缓存增益不能直接当AUC或采用固定L7。Stage-202已实现，见0.196，待服务器真实训练，确认集封存。 [结果及完整推导](docs/experiments/06_sgla_uav_tracking/STAGE201_RESULTS_REVIEW.md) · [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE201_SCORE_BASELINE_IDENTIFICATION.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage201_review_audit_20261125.json)

### 0.196 Stage-202 定位遗憾上界监督（固定实现，真实结果见0.197）

用户授权下一轮后新增`stage202_regret_common.py`、`stage202_regret_routing.py`、`test_stage202_regret_routing.py`、`run_stage202_regret_routing.sh`及固定协议。唯一训练干预为目标函数：point=mean(P(s−y)²)；regret_margin=mean(max_l[s_l−s_l*+y_l*−y_l])，l*取当前精度标签的精确argmax。精确分数argmax的实际定位遗憾不超过该损失，继承L9的1e-8近最大值规则后上界加1e-8；真值最优项精确0、无手调margin／温度／组合权重。该标准代价敏感损失不单独宣称原创理论，不预先认定MSE为唯一病因。

两组原199 interaction_residual架构／6720参数／1088输入，μ+P r、fit中心化ordered输入、同初始权重与buffer、同配对日程。沿用38开发组19／19两折、2132／2133、原AdamW及256更新／512状态对／1024引用；8模型2048更新8192训练引用4096模型对引用、4套日程2048行、14592评价输出。两组同次重训MSE，不能拿旧200作唯一匹配控制。日程沿用200同视频独立有放回抽样，自配对保留；两个新目标均按状态监督，不优化视频差分。

来源为服务器194原始特征／标签目录和仓库已审200原字节包，后者SHA 00a487b9552785b6a4eb9f2a64a7477591e4ddbed32b5d348df362026a8d8838，决策b9b4aafb6a648a38d54347e76dd1e2138f8ad9b188cfbea51c962bb59b30f2dc、计划8792d5489f5d1711191ea202e4c2d550bbc310fa17f72615a308fd68f437522f。200指纹／12表／冻结代码和协议、194每个NPZ／receipt和标签／fit先验均核验。不读取图像、不运行骨干／CENTER、不生成新完整轨迹、不打开确认组；只训练CPU小路由器。

13表：model_scope8／fit_summary2／source_bridge38／pair_schedule2048／pair_support76／error_summary12／error_video912／policy_summary20／policy_video1520／paired40／frame_predictions14592／training_history2048／seed_summary40。支持指标附定位遗憾上界及违反量、双seed、层选择分布／平均层号。`mean_selected_layer`不是实际执行块数或延迟，actual_transformer_calls_measured=False；缓存阶段无法证明动态加深的速度／精度收益。三共同主比较regret held对point、prior、自身无对应期望，再看源组／seed一致性，满足后才另立自身轨迹验证。所有区间为历史开发集名义区间，不能当确认集结果或新AUC。

6项本地测试全部通过（Python3.12.4／Torch2.7.1+cpu／NumPy1.26.4）：真实200来源及抽样复现／缓存篡改、float64遗憾上界及近并列规则、margin梯度与MSE逐元素／梯度一致、真实6720参数初始化、两组匹配训练／恢复／buffer／旧200 MSE最终模型digest精确相同、38组合成完整8模型2048步／13表／引用重建／上界检查／禁止CUDA／产物复用与篡改拒绝。Bash只读语法检查、Python编译通过。本地合成token尺寸小，实际320×192仅单模型初始化检查，不冒充真实训练。本机没有真实194 NPZ；真实202等待服务器运行，1～201代码和证据未修改。

服务器`sglatrack-test`、`~/workspace/paper`执行`bash tools/run_stage202_regret_routing.sh`，默认root `output/stage202/matched_regret_margin_v1`，可用STAGE194_ROOT／STAGE200_BUNDLE指定来源，STAGE202_ROOT指定独立输出。后台nohup及tail命令见[固定协议](docs/experiments/06_sgla_uav_tracking/STAGE202_REGRET_MARGIN_ROUTING.md)。单一总体进度条＋日志定时刷新；完成回传`stage202_results_<UTC>_<PID>.txt`（decision／manifest／13表）。[本地验证记录](docs/experiments/06_sgla_uav_tracking/results/stage202_local_validation_20261125.json)。未代commit／push。

### 0.197 Stage-202 真实定位遗憾监督结果（复核完成，未采用）

原回传`stage202_results_20261125T154651Z_1417349.txt`按原字节归档，SHA 8f0b46df691d9c9a1027dc96af56bf699431f470b58027451742f536371f3562。决策5048f7fe1c427f22b2f3e8c61533abe8b5cfeb70d362536a6e96acaa976602a4，计划715da640896f6884dc930515174d62815d839d0520d7c9c9b48ee188a7e47616。38历史开发源组／2fold／2seed／8模型2048更新8192训练引用4096模型配对／14592预测，0新图像／骨干／完整轨迹／确认组。原协议和1～202代码冻结。本机仍无原PT／特征NPZ，628物理产物仅核对声明。

决策／manifest／13表／代码／原协议哈希相符，163来源引用与200相同。抽样／引用／逐状态／误差表复算差0，视频策略≤4.44e-16、汇总≤1.73e-17、40配对≤1.81e-17、seed汇总≤1.76e-17。循环无对应期望验证≤4.44e-16，上界违反量0。四个point最终模型哈希与全部导出分数和200 point精确一致，匹配控制复现。

held point／regret／prior IoU .66061429／.66104727／.66660835。regret−point +.00043298，CI[−.00116842,+.00209279]、18胜19负1平；对prior −.00556108，CI[−.01238980,+.00069025]、15胜21负2平；对应增量+.00057849，CI[−.00037627,+.00149258]、22胜12负4平。三个共同主要比较均未成立。regret两seed IoU .66083385／.66126069，对point +.00113729／−.00027132，对应−.00018515／+.00134214，不挑种子。57.92% held状态偏离L7、六层均使用；平均层号8.3026不等于实际执行块数，本轮成本未测。

regret fit IoU .678679／排序76.02%，held .661047／54.01%；point对应 .679220／77.31%、.660614／54.93%。regret目标按实际抽样相对初始化下降19.71%～24.68%，fit全池上界.06258479优于point .06536856，但held .08373497差于point .08173666和prior .08028782。有效训练没有迁移为可靠held收益，尚未唯一定位特征／优化／泛化原因。

探索性发现：按裁剪前梯度范数>既有阈值1计，regret 1022／1024（99.80%）、point 4／1024（0.39%）；中位范数9.48993／.14661，最大16.44853／1.15178。202作为固定优化规则下的目标比较仍有效，但进一步归因需分离目标与裁剪交互。频繁裁剪不等于没学，也未证明它造成泛化下降。另核对held每组3648模型状态中778多重最优／430六层全零；regret全零状态仅贡献2.18%上界总和，评价时取最佳最优层锚点仅少2.95%上界，没有据此重训或认定它为主因。

下一步建议Stage-203损失×梯度裁剪2×2匹配：point／regret各配clip=1和仅记范数不裁剪，四组同次重训，保持结构／先验／初始／样本／AdamW／256步1024引用；16模型4096更新16384引用29184缓存预测，0骨干／图像。检查每目标裁剪开关效应及差中之差、实际更新尺度、fit和held、相对prior与自身无对应期望、源组／双seed一致性；非有限值即记失败停止，不临时改阈值。无裁剪也仅降低fit而held不升，则降低优化裁剪假设优先级，转查表征／支持，不继续扫学习率与阈值。不能从缓存结果称新AUC或动态加深成功。203已实现，见0.198，待服务器真实训练，确认集封存。 [完整结果复核](docs/experiments/06_sgla_uav_tracking/STAGE202_RESULTS_REVIEW.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage202_review_audit_20261126.json)

### 0.198 Stage-203 损失目标×梯度裁剪匹配四组（固定实现，真实结果见0.199）

用户授权继续后新增`stage203_clipping_common.py`、`stage203_loss_clipping.py`、`test_stage203_loss_clipping.py`、`run_stage203_loss_clipping.sh`和运行前协议。四组point_clip、point_no_clip、regret_clip、regret_no_clip；完整复用202两损失，不改温度／margin／loss权重／网络。开启裁剪保留原torch clip_grad_norm_阈值1及epsilon，关闭时只读计算范数，梯度值不变。记录处理前后全局范数、实际缩放、实际裁剪次数、超继承阈值次数、参数更新L2与相对L2。

同原199残差μ+P r／6720参数／1088输入、fit统计／初始化／buffer／同源组抽样日程／AdamW lr1e-5 wd1e-4／256步512对1024引用，四组同次重训。38历史开发组、原19／19两fold、2132／2133、每视频48原生父框状态。16模型4096更新16384训练引用8192模型对引用，4套日程仍2048行；16×1824=29184缓存预测，真实完整进度35144工作单元。0新图像／骨干／CENTER／完整轨迹／确认源组，平均层编号不当执行块数或延迟。

来源只需服务器194完整NPZ／receipt目录及仓库内202原字节包，SHA 8f0b46df691d9c9a1027dc96af56bf699431f470b58027451742f536371f3562、决策5048f7fe1c427f22b2f3e8c61533abe8b5cfeb70d362536a6e96acaa976602a4、plan715da640896f6884dc930515174d62815d839d0520d7c9c9b48ee188a7e47616。验证202指纹／13表／代码／原协议、194每文件及标签／fit先验。不加载202旧PT代替新控制。开启裁剪的8个新模型额外与202旧权重哈希声明／逐状态分数和动作比对，输出control_replay；库版本漂移如实报告，以本轮同次4组为因子控制，不假称逐bit重现。

14表：scope16／fit_summary2／source_bridge38／pair_schedule2048／pair_support76／error_summary24／error_video1824／policy_summary40／policy_video3040／paired92／frame_predictions29184／training_history4096／seed_summary80／control_replay8。统计／模型／预测及receipt预计1252条物理产物引用。成功决策status REVIEW_LOSS_BY_GRADIENT_CLIPPING，不自动采用。

五项主要held即时IoU比较：regret_no_clip−regret_clip；regret_no_clip−point_no_clip；(regret_no_clip−regret_clip)−(point_no_clip−point_clip)；regret_no_clip−prior；regret_no_clip的状态对应增量。前三项是优化机制诊断、后两项是实用条件，不用单个正比较宣称成功或要求交互一定为正才承认实用收益。源组等权／先平均种子／原seed24194 bootstrap20000名义区间，fit／held分开；仍是历史开发暴露，不开确认集。不裁剪仅改善fit而held不升，则降低裁剪假设优先级，转查表征／数据支持，不继续扫阈值。

loss／梯度／参数／Adam状态／参数范数／更新量非有限时封存diagnostics/nonfinite_failure.json（计划／组／fold／seed／step／阶段），立即停止且没有完整成功决策；该路径不自动重试，不补换阈值或把剩余组当完整结果。普通中断可复用哈希／metadata一致的已完成模型和预测；篡改／失败标记拒绝。

9项本地CPU测试通过（Python3.12.4／Torch2.7.1+cpu／NumPy1.26.4）：原202包和抽样／缓存身份；遗憾上界；已知数值交互方向／92配对；四组真实参数初始化；loss及梯度继承；关闭裁剪梯度逐bit不变、开启与原torch一致；开启两损失的完整训练最终digest都与原202精确一致；非有限loss封存并阻止重试；38组合成16模型4096更新／14表／1216预测（每视频仅2合成状态）／训练计数／5主要比较／复用篡改拒绝／禁止CUDA。Bash -n、Python编译、LF／导出清单与冻结来源哈希均通过。本机没有真实194特征，未启动真实203，不把合成测试当研究结果。[本地验证记录](docs/experiments/06_sgla_uav_tracking/results/stage203_local_validation_20261126.json)

服务器`sglatrack-test`、`~/workspace/paper`运行`bash tools/run_stage203_loss_clipping.sh`。默认来源194=output/stage194/module_ablation_v1，202包为仓库已审原包，输出output/stage203/loss_clipping_factorial_v1；可设置STAGE194_ROOT／STAGE202_BUNDLE／STAGE203_ROOT。后台nohup／tail命令见[协议](docs/experiments/06_sgla_uav_tracking/STAGE203_LOSS_CLIPPING_FACTORIAL.md)。单一总体进度条；完成回传stage203_results_<UTC>_<PID>.txt（decision／manifest／14表），异常回传失败JSON和日志。未代commit／push。

### 0.199 Stage-203 真实损失×裁剪结果（复核完成，未采用）

原包stage203_results_20261125T162346Z_1448196.txt按原字节归档，SHA 1fc37a4c5aa9e21cd385771b22b393158da0330aeefcaa7376b9c58197887fee。决策e5f2e4bd5c58d3728cc80b1726b8590bcc22ec80a024fa74ec4e4d694055568e，计划7d8f3a19278b77d522fece421f957f294a5457cb3a42499d4feef21f94a76079。38历史开发源组／2fold／2seed／16模型4096更新16384训练引用8192模型对引用／29184缓存预测，0新图像／骨干／完整轨迹／确认组。1252个物理PT／NPZ／receipt本机不可得，只核对声明；不称重新核验权重文件。

决策／manifest／14表／代码与原协议哈希相符，163来源引用与202一致。抽样／计数／逐状态／误差表重算0；视频策略≤4.44e-16、汇总≤1.11e-16、92配对≤2.78e-17、seed≤1.76e-17；循环无对应期望≤4.44e-16，上界违反0。8个开启裁剪控制与202最终hash／所有分数／动作精确一致，control_replay复算0。4096步梯度处理／裁剪／参数更新记账一致，关闭分支scale恒1、前后norm相同、0实际裁剪。

held point clip／off IoU .66061429／.66064190；regret clip／off .66104727／.66096011；prior .66660835。regret off−clip −.00008716，CI[−.00043852,+.00029151]、12胜20负6平；无裁剪regret−point +.00031820、CI跨零、17胜20负1平；差中之差−.00011478、CI跨零、10胜22负6平；off regret对prior −.00564824、CI跨零；对应+.00048108、CI跨零。五主要比较均未成立。point off−clip +.00002762、CI跨零、4胜2负32平。regret off−clip两seed +.00021667／−.00039100，对应−.00007895／+.00104111，不挑seed。

实际裁剪point clip4／off0、regret clip1022／off0（每组1024更新）；off regret仍有1023次原始norm>1。regret处理后平均梯度.998047→9.441478，但平均参数更新L2 .000214380→.000214596，仅+0.1008%，不能将梯度范数当参数步长，也未证明每步更新方向相同。regret fit损失上界clip／off .06258479／.06264468、held .08373497／.08368473；取消裁剪没有稳定fit或held选择改善。降低裁剪瓶颈假设优先级，不继续扫阈值／学习率。尚未唯一定位表征、容量、数据支持或标签病因，所有指标仍是原生父框即时IoU，不是新AUC；层号不当实际执行块数。

下一步建议Stage-204同参数目标—搜索关系输入对照，固定point MSE／原预算，不再追加损失。现有g=[meanZ,stdZ,meanX,stdX]经线性读出，原始token另有192→1投影；输入和联合骨干本来已有真模板，不能称没有目标条件。拟三组：原统计拼接；正确关系g_rel=[meanZ⊙meanX,|meanZ−meanX|,stdZ⊙stdX,|stdZ−stdX|]；相同运算下用预先固定fit参考的错配关系。4×192仍768、ordered token仍320、总参数仍6720。错配只改新增关系描述，不改SGLA真正模板／原token／标签，不从held未来或held真值选参考；不是有模板／无模板控制。三组共同fit-only尺度规则和参考映射须运行前固定。新描述的训练／直接计算／预聚合评价接口必须一致，不能只替换缓存pooled而重算仍走旧pool。

检验正确关系是否优于匹配原输入、错配关系、prior并有跨源／双seed对应收益；两种关系一起改善不能归于正确目标匹配，只有fit好也不算泛化。关系运算本身不是原创理论，表征缺失尚非既定病因。仍冻结骨干做路由表征假设，有缓存增量后才另立自身轨迹验证，不凭prior优采用L7或新AUC。204已完成，真实结果见0.201；确认集封存，1～204代码／原协议／证据冻结。 [完整复核与建议](docs/experiments/06_sgla_uav_tracking/STAGE203_RESULTS_REVIEW.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage203_review_audit_20261126.json)

### 0.200 Stage-204 同参数目标—搜索关系输入（固定实现记录；真实结果见0.201）

新增stage204_relation_model.py、stage204_relation_common.py、stage204_target_relation.py、test_stage204_target_relation.py、run_stage204_target_relation.sh及固定协议。三组concat、matched_relation、mismatched_relation；令a/b为模板均值／总体std，c/d为搜索均值／std，原输入[a,b,c,d]，关系[a*c,abs(a-c),b*d,abs(b-d)]。4×192仍768、320个原ordered token投影不变，总6720参数、1088输入。原联合骨干已有目标条件，不称从零补入模板；此类关系运算本身不宣称原创理论。

错配规则在运行前固定：fit按排序循环取下一fit源，held按SHA256(源组名)模fit数量取fit源，参考a/b为该fit视频全部缓存模板统计均值float32。参考不用标签、不取held未来，fit参考均不同于自身。仅新增关系描述用错配，SGLA真正模板／原token／定位标签不变；不是有模板／无模板消融。每视频映射和reference哈希导出。

三组描述均按自身fit源等权总体均值／方差中心化和标准差缩放，零方差分母1，无held校准／调权；原输入也同规则标准化，故不要求复现203未标准化控制权重。ordered token只做原fit中心化。固定buffer因输入不同而不同；initial_sha256改为仅对可训练参数内容哈希以验证同初始化，最终状态hash含buffer。训练、直接和预聚合路径均实现同一关系及参考；接口一致性检查保留。

38历史开发组19／19两fold、2132／2133、每视频48原生父框状态。三组同次重训point MSE、clip1、原AdamW lr1e-5 wd1e-4、256步512对1024引用，沿用原同视频有放回配对／自配对日程。12模型3072更新12288训练引用6144模型对引用，日程仍2048行，21888评价输出；14表scope12／fit_summary6／source_bridge38／pair_schedule2048／pair_support76／error_summary18／error_video1368／policy_summary30／policy_video2280／paired66／frame_predictions21888／training_history3072／seed_summary60／descriptor_audit228。统计／模型／预测及receipt预计948条物理产物。

来源为194原NPZ／receipt目录和仓库已审203包，SHA1fc37a4c5aa9e21cd385771b22b393158da0330aeefcaa7376b9c58197887fee、决策e5f2e4bd5c58d3728cc80b1726b8590bcc22ec80a024fa74ec4e4d694055568e、plan7d8f3a19278b77d522fece421f957f294a5457cb3a42499d4feef21f94a76079。来源指纹／14表／代码／协议、194每文件和逐状态标签／fit先验均核验。0新图像／骨干／CENTER／完整轨迹／确认集，仍是即时IoU而非新AUC，层号不当块数或速度。

四主要held比较：matched对concat、对mismatched、对prior、自身无对应期望。源组等权／先平均seed／原bootstrap24194共20000次名义区间。正确与错配一起改善不能归因匹配关系；fit好但held不升不是泛化。通过缓存条件才另立自身轨迹；不凭固定L7的缓存优势采用。沿用非有限loss／梯度／参数／Adam状态失败JSON封存及停止、普通中断哈希复用／篡改拒绝。

5项CPU测试通过（Python3.12.4／Torch2.7.1+cpu／NumPy1.26.4）：真实203来源与抽样／缓存身份；继承上界；三组同可训练初始化和真实6720参数；fit参考／held标签及未来描述隔离、直接／预聚合关系一致；三组训练／恢复、38组合成12模型3072步／14表／912预测（每视频2状态）／引用重建／4主要比较／篡改拒绝／禁止CUDA。Bash -n、Python语法、LF与14表导出清单、旧来源hash均通过。真实194特征本机缺失，未启动真实204。 [本地验证记录](docs/experiments/06_sgla_uav_tracking/results/stage204_local_validation_20261126.json)

服务器sglatrack-test、~/workspace/paper运行bash tools/run_stage204_target_relation.sh，默认输出output/stage204/target_search_relation_v1，来源194默认module_ablation_v1及仓库203原包；可设置STAGE194_ROOT／STAGE203_BUNDLE／STAGE204_ROOT。单一总体进度条，后台nohup／tail见[协议](docs/experiments/06_sgla_uav_tracking/STAGE204_TARGET_SEARCH_RELATION.md)。完成回传stage204_results_<UTC>_<PID>.txt（decision／manifest／14表）。未代commit／push。

### 0.201 Stage-204 真实结果：关系表征有线索，整体选层收益未建立

收到stage204_results_20261126T011205Z_1583527.txt，按原字节归档，SHA256为3d1a89cad966001838cc8145917bd12d08b11030737081cf5e159a4b54a48811。状态REVIEW_TARGET_SEARCH_RELATION，decision 3cc164919abc8af737a21520e9e53b9775c5bf055ddfb6d0eda1d43a0ee36935，plan 603de9cf615c4d318c1bc54de29de745c76827a7319d5593b4d5f1c79828ff5c。指纹／14表／代码及原协议哈希相符，163条194来源引用与203一致。

38历史开发源组、两fold各19 fit／19 held、两seed、三输入12模型，共3072更新、12288训练引用、6144模型配对引用、2048独立日程行、21888模型评价状态。6720参数、可训练参数初始化、point MSE／clip1／预算匹配。全部日程、支持、逐状态误差和上界、策略、seed、66配对从导出重算；最大差≤4.44e-16。48次循环动作移位独立核验无对应期望；228参考映射按规则重建且均来自fit、不借自身，跨组哈希相符。948物理产物不在本机，未加载PT或重算原NPZ描述／尺度，只核验声明与导出。0新图像／骨干／CENTER／完整轨迹／确认组。

held平均IoU：concat .66036904、正确关系 .66127519、错配 .66074865、prior .66660835。正确关系四项主要差值：对prior −.00533315、CI[−.01148243,+.00004140]；对自身无对应期望+.00159258、CI[+.00041757,+.00289660]；对concat +.00090616、CI[−.00104494,+.00285417]；对错配+.00052654、CI[−.00325716,+.00432923]。仅自身对应一项有名义正区间；不是全部无效，也不是方法通过。两seed正确对concat +.00108482／+.00072749，对错配+.00262742／−.00157433。自身对应+.00136829／+.00181686均正，最小留一总体+.00123136。

concat自身对应也为正+.00150170；正确对concat对应差+.00009088、CI跨零，不能归因正确匹配。正确对错配对应差+.00130349、名义区间为正，只为支持性信号，总体IoU主要比较仍不成立。正确对concat held层间MSE −.0000542575、区间为负；加权排序+3.112个百分点、区间[+.953,+5.480]。held正确MSE .00159380仍差于prior .00156916，排序57.173%仍低于prior59.295%。约95.4%的MSE改善来自视频均值分量，视频内改善区间跨零；这是描述性分解，不证明唯一病因。

正确组3648动作L7～L12分别1817／587／498／384／249／113。描述性分解：实际−prior −.00533315＝无对应期望−prior −.00692573＋时机对应+.00159258；不能把缓存层号当实际执行成本，不能称AUC。38组已历史暴露、区间名义，非独立确认；不采用当前路由，不因prior即时优而采用已知完整轨迹受损的固定L7。

下一步建议Stage-205：严格fit内源组交叉拟合残差收缩。每外fold19 fit逐源组留一，用其余18组同预算拟合并预测该组；三输入×两seed×两外fold×19内折＝228个额外CPU小模型。内折先验、mean_F、描述均值／尺度、错配参考池全部从18组重算，不能复用含内留出组的外层统计。令P为六层去均值，内折OOF残差h=P(s−μ)、目标z=P(y−μ)，源组等权估计α=clip_[0,1](E⟨h,z⟩/E||h||²)，分母0时0。α仅fit内部估计并锁定，再对冻结204外层分数评价μ_outer+αP(s_outer−μ_outer)。不扫网格、不用held标签校准；最小MSE不保证argmax IoU，必须实际验证。

三输入都做同规则收缩并保留α=1／0，核验IoU、状态对应和非先验动作数；若全部退回L7，即使恢复.666608也只是固定先验保护，不是动态收益。228内折训练为额外校准成本，单列，不能宣称总训练预算与原模型相同；外层204预测可冻结复用。不同于198固定均值校准、201逐视频去均值，不使用held未来均值。若仍无非固定增量，降低单独校准假设优先级，重审数据支持／骨干，不持续扫系数。以上为结果复核时的建议；205现已实现，见0.202，待服务器真实运行。确认集封存，1～204代码／原协议／结果冻结。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE204_RESULTS_REVIEW.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage204_review_audit_20261126.json)

### 0.202 Stage-205 fit内源组留一残差可信度（固定实现记录；真实结果见0.203）

新增tools/stage205_shrinkage_common.py、stage205_crossfit_shrinkage.py、test_stage205_crossfit_shrinkage.py、run_stage205_crossfit_shrinkage.sh及固定协议。继承204三种输入，检验其小幅状态对应信号能否在降低不可靠残差幅度后保留；不预设校准必然修复选层。不同于198固定层均值、201视频去均值，不使用外held未来统计。

来源固定204原包SHA 3d1a89cad966001838cc8145917bd12d08b11030737081cf5e159a4b54a48811，decision 3cc164919abc8af737a21520e9e53b9775c5bf055ddfb6d0eda1d43a0ee36935、plan 603de9cf615c4d318c1bc54de29de745c76827a7319d5593b4d5f1c79828ff5c，以及194原生父框特征／六层标签。14来源表、指纹、代码与协议核验；原204 PT不必复制，外层分数由原包精确恢复float32，动作核对后复用。

每外fold19 fit逐源组留一，内层18 fit／1校准。provider只含外fit19组；内留出组不参与mean_tokens、描述均值／标准差、先验或错配参考，全部每内折重算。三输入同6720参数／1088输入／可训练初始化、point MSE／clip1、原AdamW lr1e-5 wd1e-4，256更新、每步两对同视频独立有放回状态，自配对保留。两外fold×19内折×两seed×三输入＝228额外CPU模型，58368更新、233472训练引用、116736模型对引用；三输入共享38912行日程，1368行支持。内层预测10944状态，外held复用10944状态。明确新增校准成本，不能声称与原raw总预算完全相同。

P为六层去均值，内层OOF h=P(s−μ_inner)、z=P(y−μ_inner)，μ_inner取实际float32模型prior。源组等权、组内状态等权估计α=clip_[0,1](E⟨h,z⟩/E||h||²)，分母0则α=0。每外fold／seed／输入一个α共12，带OOF分数hash的sealed JSON全部发布后才评价外held。外层μ_outer+αP(s_outer−μ_outer)用float64统计，α1显式保留204原float32分数值，α0返回原float64 prior；ties沿用L9差≤1e-8优先，否则首个最大。raw与prior须逐源组复现204（1e-12指标容差）。未最优化IoU，平方误差最小不保证选择成功。

五主要比较固定为matched shrunk对raw／prior／自身无对应期望／concat shrunk／wrong shrunk。报告两seed、非先验比例、六层计数、对应、MSE／排序。若全部固定L7，即使IoU回到.666608也不是动态收益；不得据此采用已知完整轨迹受损的固定L7。沿用38历史开发源、先平均seed再源组bootstrap20000次seed24194、名义95%区间；不称独立确认／完整AUC／真实执行块数或速度。

16表：source_bridge38、inner_scope228、inner_statistics114、inner_schedule38912、inner_support1368、descriptor_audit2166、training_history58368、inner_predictions10944、calibration12、frame_predictions10944、policy_video1140、policy_summary15、error_video684、error_summary9、paired22、seed_summary30。114统计／228模型／228预测各有receipt，12 sealed系数，共1152内部文件绑定decision。使用冻结204训练／缓存引擎，其schema保留，但元数据绑定205计划及严格内折；普通中断复用已提交小模型，未提交小模型从头重训，非有限诊断封存后拒绝继续。0新图像／骨干／CENTER／外模型重训／完整轨迹／确认集，单一总体进度条含复用计数。

4项本机CPU测试全部通过、无跳过（129.699秒，Python3.12.4／Torch2.7.1+cpu／NumPy1.26.4）：真实204来源与10944 held分数恢复、篡改拒绝；解析二次风险／源组等权／负值与上界／零分母／近并列α端点；严格内折先验／mean_tokens／描述尺度／参考不含内留出组；完整38组合成228模型58368步、内外各456状态、12系数先封存、1152内部文件／16表／五主比较，正常中断后两模型复用、冻结源不变、输出复算与篡改／非有限阻断。未调用CUDA。Python AST／CLI、Bash -n、LF、16表打包清单和旧源代码／协议哈希通过。[本地验证记录](docs/experiments/06_sgla_uav_tracking/results/stage205_local_validation_20261126.json)

服务器sglatrack-test、~/workspace/paper：mkdir -p output/stage205，然后nohup bash tools/run_stage205_crossfit_shrinkage.sh > output/stage205/run.log 2>&1 < /dev/null &；tail -f查看。CPU-only无需指定L20或L40。默认output/stage205/fit_source_crossfit_shrinkage_v1，支持STAGE194_ROOT／STAGE204_BUNDLE／STAGE205_ROOT，--check-results仅复核并打包。自动产出stage205_results_<UTC>_<PID>.txt（decision／manifest／16表）。本机没有真实194特征，尚未真实训练，不承诺收益，未代commit／push。无非固定增量时降低单独校准假设优先级，转审数据支持／骨干，避免扫系数；确认集封存。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE205_CROSSFIT_RESIDUAL_SHRINKAGE.md)

### 0.203 Stage-205 真实结果：近固定先验保护，不是有效动态选层

原包stage205_results_20261126T013956Z_1605754.txt按原字节归档，SHA256 0f342f0b40ad5f877045b55c0e19f39255848978418031a63f09ac7961e8972d。decision 074e680f182d8752a1796ffb7ece24458d17d0994f82c3235723a2e36655da7c、plan d9df9fb9490b763a3d5da152a7733be95362ad8eeaef624128202fc809471d93，状态REVIEW_CROSSFIT_RESIDUAL_SHRINKAGE。指纹／16表／代码／原协议哈希相符；163条来源引用声明与204一致。

228内层模型、58368更新、233472训练引用、116736模型对引用、38912日程、1368支持，内外各10944状态，12系数／1152内部物理文件声明。重算12系数与全部逐状态分数误差0，策略／汇总／22配对／seed最大差≤4.44e-16，五主要对比与decision相符；76个严格内折日程／支持、2166 donor映射和228训练历史预算／峰值核查通过，独立48次循环移位无对应期望差≤4.44e-16。本机无原特征NPZ／PT，不重算实际描述或模型，未独立观测系数文件发布时间。0新图像／骨干／外模型重训／完整轨迹／确认组。

concat raw→shrunk IoU .66036904→.66661928；正确关系 .66127519→.66662117；错配 .66074865→.66339913；prior .66660835。正确五主差：对raw +.00534597，CI[−.00001956,+.01149160]；对prior +.00001282，CI[−.00011489,+.00011698]；自身对应+.00002418，CI[−.00006335,+.00010583]；对concat shrunk +.00000188，CI[−.00010491,+.00009413]；对wrong shrunk +.00322204，CI[+.00072251,+.00603783]。仅错配一项名义正区间；不能四舍五入把对raw下界变成通过。

正确3648动作从raw L7的1817次变为3615次，非L7仅33次（L8 10、L9 23，其他0），非先验.9046%；concat仅11非L7、wrong488。正确seed2132／2133 shrunk IoU .66663399／.66660835，对应+.00004836／约0，后者全部固定L7。全部12 α中4个0、无1：fold0 concat .310005／.244168、matched .470337／.248421、wrong .146236／.356705；fold1 concat／matched两seed全0，wrong .088010／.011995。fold1两正确未截断系数为负，按预定规则归0，并非后验挑值。

正确层间MSE .00159380→.00156583，prior .00156916；加权排序57.173%→59.433%，prior59.295%。补充MSE／排序对raw与prior的源组区间均跨零。内层校准风险下降是所选解析目标性质，不是独立评价。此结果更接近固定偏好保护，未证明动态收益；也不证明动态选层不可能或表征缺失为唯一病因。38组已有历史暴露，名义区间、即时IoU不称新AUC；不采用缓存先验L7，194完整固定L7已受损。

接续建议Stage-206：停止单独调α／均值，直接做SGLA内部可学习空间目标—搜索交互适配的最小对照。与194全局均值scale／shift、164固定局部匹配区别；逐token、零输出初始化。单搜索／CENTER／反馈沿原生，原生路由在未适配L6上先定层，避免同时改路由。对照原生、同参数同训练预算的搜索自交互、正确模板条件、错配模板条件，统一定位监督；参数相等不冒称耗时相等。先验初始精确复现／梯度／fit-held定位，再冻结模型评完整开发轨迹及稳定／恢复保护；确认集封存。已有SGLA本就联合模板与搜索，不能称首次引入模板；若wrong也改善仅一般适配。以上为结果复核时建议，206现已按独立固定协议实现，见0.204，待GPU运行。1～205代码／原协议／证据冻结。 [完整复核](docs/experiments/06_sgla_uav_tracking/STAGE205_RESULTS_REVIEW.md) · [机器复核](docs/experiments/06_sgla_uav_tracking/results/stage205_review_audit_20261126.json)

### 0.204 Stage-206 原生路由固定的空间目标条件适配（已实现，待GPU运行）

新增tools/stage206_spatial_model.py、stage206_spatial_common.py、stage206_spatial_adaptation.py、test_stage206_spatial_adaptation.py、run_stage206_spatial_adaptation.sh及固定协议。停止单独调α，检验空间表征改动；不预设空间缺失为病因。与194全局均值scale／shift、164固定局部读出区别，不称首次给SGLA引入模板，也不是新动态路由。

F6=[Z64×192,X256×192]，无可训练仿射LN，全宽无bias Q/K/V/O，单头softmax(Q LN(X) (K LN(R))ᵀ/√192)，X'=X+O(A V LN(R))，Z不动。每组147456参数。O零初始化，初始精确原函数，三输入同seed参数完全相同；参考分别为搜索16×16经2×2均值池化为8×8、正确Z、另一fit源48状态模板token按位置均值。三组均64参考token、256×64注意力矩阵。错配fit排序循环下一组、held SHA256模19选fit；仅适配参考换，不换原SGLA模板，held不参与参考统计。

L6 hook先保存未适配F6的channel0，返回适配F6'；MLP pre-hook替换输入，原MLP只调用一次且分数／权重不改，候选块使用F6'。线上核对hook次数及MLP所选层与实际层。每帧在自身状态内保留原生决策，不要求完整轨迹选层序列与baseline一致。训练同样在未适配F6原生选层，只对所选候选块原定位2GIoU+5L1+focal反传，无六层GT择优；CENTER和全部原生参数／buffer冻结。

源为194原生父框特征／target及已审205原包（SHA0f342f0b40ad5f877045b55c0e19f39255848978418031a63f09ac7961e8972d、decision074e680f182d8752a1796ffb7ece24458d17d0994f82c3235723a2e36655da7c）。沿原76／89／93／121／170／188／191入口核38组／标注／帧清单、原checkpoint和sealed config，确认集封存。38组19/19两fold、2132/2133，三组同19 fit抽样日程、同参数／初值／AdamW lr1e-5 wd1e-4 clip1、256更新batch4，12模型3072更新12288状态引用；每例一个原生所选候选，实际分组batch调用另记。不是与194六层监督等计算预算，更不以参数等同耗时。

模型全部拟合后固定，执行每模型38×48状态的fit／held单选层读出，共21888状态、456行readout；历史native_box只作缓存参考，记录cached_native_layer_differences。完整视频全新同卡原生38条＋三组两seed228条＝266条自身轨迹。首帧初始化后不GT重置，原裁剪／CENTER／Hann／反馈保持。三主要AUC对比为matched对native／search_control／mismatched，先平均seed、源组bootstrap20000次seed24194，名义区间；另报新增失败、恢复丢失、好帧损失与两seed，固定REVIEW不自动采用。

9表scope_summary12／training_history3072／reference_audit76／smoke_summary3／readout_video456／policy_summary4／paired17／seed_video_summary266／routing1596。新适配权重、参考、读出和完整轨迹NPZ/receipt及sealed来源receipt绑定decision；4次公共路径zero smoke、训练、读出和7倍完整开发帧数使用一个总体进度条。首次L20或L40均可，新同卡原生控制；GPU UUID与软件／配置／模型哈希封存，同root续跑不能换GPU身份。普通中断复用已提交模型／视频，未提交模型重训；非有限诊断阻止继续。原source206 finally显式remove NativeObserver.handles，不调用不存在的close。

3项CPU测试全部通过、无跳过，75.333秒（Python3.12.4／Torch2.7.1+cpu／NumPy1.26.4，Pillow使用已有cp312缓存依赖）：真实205来源哈希／代码／协议及错配held隔离；独立空间公式、三组初值一致／零残差精确保持模板行、实际配置147456参数、非零适配后原MLP输入与分数保持／hook清理；真实仓库CENTER与定位loss、小型线性骨干／38组合成视频完整12模型3072步、266条两帧轨迹、912读出状态、3主比较、模型复用／冻结／篡改及非有限阻断。没有运行真实SGLA Transformer或GPU／真实数据，不能称正式精度复现。AST／CLI／Bash -n、LF与九表导出清单通过。 [本地验证记录](docs/experiments/06_sgla_uav_tracking/results/stage206_local_validation_20261126.json)

服务器sglatrack-test、~/workspace/paper，mkdir -p output/stage206，然后nohup bash tools/run_stage206_spatial_adaptation.sh > output/stage206/run.log 2>&1 < /dev/null &，tail -f查看总体进度。默认output/stage206/native_routed_spatial_adaptation_v1、GPU0，可设STAGE206_ROOT／STAGE206_GPU／STAGE194_ROOT／STAGE205_BUNDLE／CHECKPOINT及七旧根。需要原图像和GPU；--check-results只复核／打包。完成自动stage206_results_<UTC>_<PID>.txt（decision／manifest／9表）。本机无真实194缓存，待服务器GPU训练，不承诺收益，未commit／push。 [固定协议](docs/experiments/06_sgla_uav_tracking/STAGE206_NATIVE_ROUTED_SPATIAL_ADAPTATION.md)

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
