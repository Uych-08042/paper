# Stage-58：完整末帧保持的梯度方向与长度控制

日期：2026-10-30。核心为原生SGLA上的无人机单目标跟踪，使用现有VisDrone-SOT。Stage-1～57实验Python及服务器结果冻结。本轮实现已准备，5项真实CPU测试通过（219.308秒、无跳过）；服务器GPU结果尚未产生。

## 从Stage-57继续的问题

[Stage-57结果复核](STAGE57_RESULTS_REVIEW.md)中，相对质量门控正确执行，但只保留20.52%的eligible；相对完整保持AUC−3.6057、两个种子均负。0204双种子无失效值得保留，0176／0307恢复丢失与0304早期失效也必须解释。当前IoU较高和梯度兼容／未来有用不能等同。

本轮检验：**保留全部原合格末帧项，只处理其与普通监督相反的梯度分量，是否比删参考更能保留适配收益与恢复？** 梯度冲突还没有被证明是此前损伤的原因，本轮同时记录其实际出现情况和完整闭环效果。

借鉴[Yu等，Gradient Surgery for Multi-Task Learning，NeurIPS 2020](https://arxiv.org/abs/2001.06782)的投影操作。此处固定普通监督方向，只改原生末帧保持梯度，应用于SGLA自身短分支几何适配，并加入同长度控制；不等同于完整对称PCGrad，也不把已有方法的存在作为否定新增价值的理由。

## 三个固定臂

记gS为本批普通GT监督对原头size／offset参数的梯度，gP为全部eligible末帧保持项的聚合梯度。普通监督仍使用既有raw峰值回归路径，保持项仍通过原生Hann峰值解码；本轮不改变监督格点或响应分支。

当内积gS·gP<0且两个向量非零时，计算：

```text
gP_projected = gP − (gS·gP / ||gS||²) gS
alpha = ||gP_projected|| / ||gP||
```

否则gP_projected=gP，alpha=1；gP=0或gS=0均为合法情形。投影系数和梯度向量全部detach，不求高阶导数。内积及范数用float64审计，实际参数与应用梯度保持原float32。没有梯度阈值扫描。

| 臂 | 提交给clip／AdamW的梯度 | 用途 |
|---|---|---|
| `endpoint_control` | 原始Lsup+Pend的完整backward | 精确复现Stage-57完整控制 |
| `norm_matched` | gS + alpha·gP | 只改保持梯度长度，保留原方向 |
| `projected_anchor` | gS + gP_projected | 检验去除反向分量的新增作用 |

两个处理臂的**影子长度匹配发生在各自当前批状态**，不保证两个真实训练臂整程梯度长度或损失相同。投影还会改变总梯度长度、裁剪量及AdamW状态，因此是完整优化策略比较，不是相同实际步长的纯角度因果分解。

资格保持不变：参考原生两帧IoU均≥0.5，且参考末帧中心位于学生末帧实际可用裁剪内。全部eligible仍进入固定分母的原中心差＋log尺寸比保持损失，系数1，不用当前IoU删除项，不按幸存数归一化。完全反向共线时投影梯度可为0，这是向量处理的合法结果，需在报告中保留。

## 梯度与优化器边界

投影保证的是给定批次、给定参数上的欧氏一阶方向关系，**不保证实际AdamW位移或未来跟踪都改善**。动量、逐参数预条件及weight decay均可能改变实际位移方向。本轮每步计算并累计`−gS·(theta_after−theta_before)`及其为负的次数，不把它与原生保持梯度内积混为一谈。CPU测试需展示存在“投影后不反向、AdamW位移仍反向”的实例。

三臂均计算gS、gP、影子投影／长度控制，并执行完整原始标量backward；只有两个处理臂在保持梯度确实改变的批次随后覆盖几何参数grad。不需要修改的批次直接保留原始backward结果，避免拆分再相加引入舍入差；控制臂始终保留原加法顺序，以复现原控制。投影臂一般没有等价固定标量目标，history里的`full_scalar_loss`是原始标量审计值，不能称为实际被最小化的新损失。

保持项直接梯度仍仅到末帧size／offset，普通监督覆盖两个访问帧；父框、实际裁剪、取整和argmax都截断，不是BPTT。骨干、独立候选块、选择器、响应及BN缓冲冻结。GT只在fit组用于普通监督、原资格和诊断；在线actor始终只有当前图像、首帧模板及自身预测父框。

## 来源、预算与记录

输入为完整Stage-57输出。prepare核验manifest、smoke、12个来源模型、72条来源轨迹及8张表SHA，并从轨迹复核132行逐种子／72行逐视频／6行宏策略及调用数；继续验证上游Stage-56／54及更早依赖。不将具体0204／0304失败帧重加权到fit，也不使用held-out结果选择参数或checkpoint。

沿用12个已消费源组、2折各6 fit／6 held-out、种子2132／2133、每视频48个均匀时刻、256步、原生／fit-only donor两个预测起点。每条两帧学生分支的第二帧使用当前头第一帧输出。原始初始化、抽样、AdamW（lr1e-5、weight decay1e-4）、clip1及更新数保持一致。

正式3臂×2折×2种子＝12个新头、72条新完整OOF；另复用原生／旧几何／旧静态60条上下文轨迹。正式训练24,576次完整前向／172,032个block及12,288次仅头重算；每批只有一次头重算路径，不为分离gS额外重放头。3个8步smoke头及短推理单独计费。梯度拆分与投影有训练开销；前向预算一致不表示全部训练FLOPs或墙钟完全相同，推理不增加模块或调用。

完整控制须复现Stage-57相同折种子的最终权重摘要、普通监督计数和抽样指纹，OOF的boxes／previous／peak／layer逐元素相同。失败时停止解释，不放宽数值容差。检查点保留AdamW状态、抽样与来源、累计诊断及history，128／256步提交；重跑同一train命令续训，复用已提交完整视频缓存。

scope／history记录：

- 原合格项及参考当帧较优／平局／较差数，均只作诊断；实际原合格项全部保留。
- gS／gP／投影／同长度／实际保持梯度范数总和，原始／投影／实际内积、余弦和去除分量。
- 冲突批次、非平凡投影批次、实际修改批次；全合格参考都较优仍出现冲突的批次，以及包含非较优参考的冲突批次。后者为聚合批次关系，不是逐参考的因果归因。
- 投影后反向残差与影子范数相对误差（容差1e-6），实际AdamW监督方向反向次数及内积总和。

history在既定保存步写入累计计数和标量，不保存全部逐步梯度向量。零范数时余弦诊断记0，范数比记1。非平凡投影定义为去除量超过原保持梯度范数的1e-6；这只是数值曝光判断。

## 预定判读

主要`projected_anchor_minus_endpoint_control`，必要方向控制`projected_anchor_minus_norm_matched`。先视频内平均两种子，视频级5,000次bootstrap、种子2158；两条预定AUC对比各报告名义95%和97.5%家族区间。

积极开发条件沿用：主要AUC≥+0.5、两条家族区间下界>0、主要两个种子均值>0；对原生宏AUC不下降、低IoU不恶化、可靠帧至少−0.5、原稳定视频无新增失效。另需自身后继裁剪改变，且两个处理臂的每个模型至少5%优化步发生非平凡投影（正式至少13／256步）、实际修改非零。曝光不足照常报告REVIEW，不更换阈值或种子。

通过为`PROMISING_PROJECTED_ENDPOINT_GRADIENT_PILOT`，否则`REVIEW_PROJECTED_ENDPOINT_GRADIENT_PILOT`。保留全部固定视频及原10条获益队列，关注0176／0226／0304／0307／0204的两个种子、恢复与末尾未恢复；失效事件少可能只是恢复少，不能单独表示更稳定。

这是消费过的开发OOF，bootstrap不代表独立外部确认或重训不确定性。若投影曝光充分但闭环不改善，可否定当前固定投影配方的改进主张，不能据此认定所有梯度冲突处理无价值；若两个控制同样改善，也不能把收益归为方向修正。结果产生后不搜索权重、投影阈值、长度、种子或checkpoint。

## 服务器执行

本机5项CPU Torch／OpenCV测试全部通过，219.308秒、无跳过。覆盖真实CENTER小通道头及原生解码、仅一次头重算、未来GT不改变actor、末帧直接梯度、投影与同长度数学性质／零向量／非法值、无操作批次原backward逐位保持、AdamW位移反向反例、完整Stage-51→57合成来源、fit隔离、12模型／72条OOF、四个完整控制精确复现、控制与投影臂断点恢复及AdamW状态一致、缓存复用和来源／结果／checkpoint篡改拒绝。5文件AST、纯NumPy CLI帮助及维护检查通过。CPU合成流程不代表服务器精度结果。助手未登录服务器或执行GPU实验，未代用户commit／push。同步新增5个Stage-58 Python文件及本协议，保留Stage-57及全部上游产物，在原环境逐条执行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE57_ROOT="$PWD/output/stage57/quality_selective_endpoint_v1"
export STAGE58_ROOT="$PWD/output/stage58/projected_endpoint_gradient_v1"

python -B tracking/stage58_gradient_experiment.py --mode prepare --input "$STAGE57_ROOT" --output_dir "$STAGE58_ROOT"
python -B tracking/stage58_gradient_experiment.py --mode smoke --input "$STAGE58_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage58_gradient_experiment.py --mode train --input "$STAGE58_ROOT" --checkpoint "$CHECKPOINT" --gpu 0 --seed 2132
python -B tracking/stage58_gradient_experiment.py --mode train --input "$STAGE58_ROOT" --checkpoint "$CHECKPOINT" --gpu 0 --seed 2133
python -B tracking/stage58_gradient_experiment.py --mode evaluate --input "$STAGE58_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage58_gradient_summary.py --input "$STAGE58_ROOT"
```

优先用cat查看并提供以下五份材料，长表可附件；完整逐视频／逐种子／retention／events也会生成并保留：

```bash
cat "$STAGE58_ROOT/analysis/stage58_decision.json"
cat "$STAGE58_ROOT/analysis/stage58_policy_summary.csv"
cat "$STAGE58_ROOT/analysis/stage58_scope_summary.csv"
cat "$STAGE58_ROOT/analysis/stage58_paired.csv"
cat "$STAGE58_ROOT/analysis/stage58_focus.csv"
```

Git同步不是结果备份，不删除被新manifest绑定的上游模型、收据、缓存或轨迹。
