# Stage-57：按拟合组相对质量选择末帧保持

日期：2026-10-30。用户已报告`REVIEW_QUALITY_SELECTIVE_PRESERVATION_PILOT`，五份结果已齐，详见[Stage-57结果复核](STAGE57_RESULTS_REVIEW.md)。以下保留预定设计及历史复现入口，不重跑已完成阶段。核心仍为原生SGLA上的无人机单目标跟踪，使用现有VisDrone-SOT。Stage-1～56源码及固定结果保持冻结。

## 本轮要回答的问题

[Stage-56复核](STAGE56_RESULTS_REVIEW.md)显示，首末各半保持未超过完整末帧保持，且0176／0307／0204的收益在种子间取舍。原生IoU≥0.5代表参考合格，却不表示它比当前预测更优。本轮检验：**在训练中，当原生参考已经不如当前预测时，允许当前模型偏离它，能否保留有效修正及恢复收益？**

本轮回到较强的完整末帧控制，不继续搜索首末权重或分支长度。没有预先证明参考质量冲突是Stage-56损伤原因；这是新的可检验训练假设。按相对质量选择教师是可借鉴的一般思路，本轮具体差异是将它用于SGLA自身短分支的末帧状态，并配条件等数量控制与完整自身轨迹检验，不宣称绝对原创。

## 三个固定匹配臂

三组均从原始SGLA头初始化，使用相同GT普通监督Lsup。对于每条两帧短分支，先保留原资格条件：原生两帧IoU均≥0.5，参考末帧中心在当前学生末帧实际可用裁剪内。该集合记作eligible。仅在fit组内，用相同的官方IoU口径比较参考末帧q_ref与当前预测q_student。

| 臂 | 末帧保持掩码 | 目的 |
|---|---|---|
| `endpoint_control` | 全部eligible | 精确复现Stage-56完整末帧控制 |
| `count_matched` | 从本批eligible中均匀选择k项；k等于本臂当前状态上满足相对质量规则的数量 | 控制仅减少保持数量的作用 |
| `quality_select` | eligible且q_ref > q_student + 1e-6 | 检验选择较优原生参考的价值 |

保持损失沿用图像坐标下的中心差及log尺寸比、1e-6死区；每项系数为1，固定分母是本批两条起点分支数，不按选中数重归一化。普通监督完全保留。相对质量掩码detach，IoU不求导；未选中项的保持梯度为0，空选择批次继续普通监督，不重抽样。

每个优化步有原生／fit-only donor两个预测起点，因而只有两个末帧保持项。随机控制预先生成独立优先数，固定seed+100×fold+571000，并从eligible取最小优先数的k项。三个臂均计算质量与随机影子掩码，以及完整／实际保持项的梯度审计。k=0或k等于eligible数时，质量与随机无法区分；必须报告混合批次及掩码分歧，不能将全开／全关当成选择机制。

**等数量是本臂当前同批输入上的条件匹配，不保证两个训练臂全程实际保留数量相同。** 各臂输出会改变自身后续输入和可用裁剪，完整运行会产生不同的k轨迹。scope和history记录实际数量、质量分歧与质量差，解释结果时同时报告。比较对象是两种完整训练策略，不是全程固定状态上的纯掩码效应。

## 范围、来源及梯度

输入是已经完成的Stage-56根目录。prepare严格核验Stage-56的manifest、smoke、12个模型、72条新轨迹、8张报告的SHA，并从轨迹重算132行逐种子／72行逐视频／6行策略表及实际推理预算；继续沿上游验证Stage-54及更早来源。Stage-55／56结果选择的具体失败帧不重新加权到训练中。

沿用12个已消费源组、2折每折6 fit／6 held-out、种子2132／2133、每视频48个预定均匀时刻、256步、每步一个双起点样本；原始初始化、AdamW、学习率1e-5、weight decay 1e-4、clip 1及抽样序列不变。两条原生参考和两条当前训练头分支均从同一对预测父框出发，当前头第二帧用自己第一帧的真实输出。

GT只在所有短分支已推进后附加，用于fit普通监督、资格及质量标签。actor接口仅接收当前图像、首帧模板框和预测父框；评价时首帧GT初始化后完全自身推进，**没有在线GT门控、挑头或隐式重置**。

只训练原头size／offset；原生骨干、独立候选、选择器、响应参数和BN缓冲冻结。新增保持项仅对末帧几何直接求导，父框、裁剪、取整及argmax仍截断，不是BPTT。官方IoU的离散比较可能产生质量平局，scope单独记录。IoU较优不保证梯度相容或长期更优，本轮没有把这种关系当作既定事实。

## 预算与复现

正式3臂×2折×2种子＝12个新头、72条新完整OOF，另复用原生／旧几何／旧静态60条上下文轨迹。每模型1,024个普通监督视图、1,024次原生完整前向、1,024次学生完整前向、1,024次仅头重算。训练合计24,576次完整前向、172,032个block、12,288次仅头调用；全部三臂的影子计算流程相同，另报告实际推理调用。3个8步smoke头和短推理单独计费。

正式完整末帧控制须复现Stage-56相同折种子的最终权重摘要、监督计数、抽样指纹；完整OOF的boxes／previous／peak／layer须逐元素一致。不符合时先排查环境或实现，不放宽校验后解释效果。随机优先数指纹、累计选择计数和AdamW状态写入checkpoint，按第128／256步提交；重跑同一train命令续训，完整视频缓存复用，未提交视频重跑。

## 报告与预定判断

主要比较`quality_select_minus_endpoint_control`，必要控制为`quality_select_minus_count_matched`，另报count_matched−endpoint_control。先视频内平均两个种子，再对视频5,000次条件bootstrap，种子2157；两条预定AUC对比各报95%区间和97.5%家族区间（0.0125／0.9875）。

积极开发条件：主要AUC均值≥+0.5，两条家族区间下界均>0，主要两个种子均值均>0；相对原生宏AUC不下降、低IoU不恶化、可靠帧AUC至少−0.5、原稳定视频无新增失效。还需自身后继实际裁剪改变，两个选择臂各模型实际选中项、选中项梯度及掩码分歧均非零，且确有被拒绝的平局／较差参考，否则不能建立选择机制。

满足标记`PROMISING_QUALITY_SELECTIVE_PRESERVATION_PILOT`，否则`REVIEW_QUALITY_SELECTIVE_PRESERVATION_PILOT`。无选择暴露是有效的未支持结果，不因它更换种子或阈值。保留所有固定视频，重点报告0176／0226／0304／0307／0204及原10条获益队列的幅度，不能用少数高分种子或失效事件减少替代完整轨迹保持。

scope新增eligible、较优／平局／较差参考数、随机选中数、实际选中数、选中却不较优数、漏选较优数、掩码分歧数、混合批次数、eligible及实际选中参考IoU差之和。由这些量可复核条件等数量与质量选择；累计参考质量差不是闭环AUC提升证据。history同时记录完整和实际保持损失及梯度。

这是已消费12组上的开发检验；GT质量门控是训练工具，不是已经学得在线可靠性估计。没有独立泛化、压缩、部署加速或绝对新颖性结论。不根据本轮结果继续搜索阈值、权重、长度、种子或checkpoint。

## 服务器执行

本机验证已完成：真实CPU Torch／OpenCV的5项测试全部通过，170.601秒、无跳过。覆盖质量／平局／无资格及非法标签、条件等数量掩码、真实CENTER头目标与梯度、未选中和空掩码的零梯度、后续GT不改变actor、完整Stage-51→56来源生成与Stage-57三臂12模型／72轨迹合成流程、fit隔离、4个控制精确复现、AdamW断点恢复、缓存复用与来源表／结果表／checkpoint篡改拒绝。5文件AST、无需Torch的CLI帮助及维护检查通过。它们不代表服务器精度结果。

以下为已经由用户完成的历史复现命令，不是下一轮指令。输入是Stage-56，输出为新目录，不重跑已完成的Stage-54～56。

```bash
cd ~/workspace/paper
conda activate sglatrack-test
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE56_ROOT="$PWD/output/stage56/first_transition_preservation_v1"
export STAGE57_ROOT="$PWD/output/stage57/quality_selective_endpoint_v1"

python -B tracking/stage57_quality_experiment.py --mode prepare \
  --input "$STAGE56_ROOT" --output_dir "$STAGE57_ROOT"
python -B tracking/stage57_quality_experiment.py --mode smoke --input "$STAGE57_ROOT"
python -B tracking/stage57_quality_experiment.py --mode train --input "$STAGE57_ROOT" --seed 2132
python -B tracking/stage57_quality_experiment.py --mode train --input "$STAGE57_ROOT" --seed 2133
python -B tracking/stage57_quality_experiment.py --mode evaluate --input "$STAGE57_ROOT"
python -B tracking/stage57_quality_summary.py --input "$STAGE57_ROOT"
```

以下五份材料已收齐，不重复索取；查看历史结果使用cat：

```bash
cat "$STAGE57_ROOT/analysis/stage57_decision.json"
cat "$STAGE57_ROOT/analysis/stage57_policy_summary.csv"
cat "$STAGE57_ROOT/analysis/stage57_scope_summary.csv"
cat "$STAGE57_ROOT/analysis/stage57_paired.csv"
cat "$STAGE57_ROOT/analysis/stage57_focus.csv"
```

另有retention_summary、video_summary、seed_video_summary及events。Git同步不是产物备份，保留服务器所有上游manifest、收据、模型与轨迹。本助手未登录服务器或执行本轮GPU实验，未代用户commit／push。
