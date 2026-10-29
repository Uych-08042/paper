# Stage-54：短程自身状态与原生末帧保持

Stage-53说明固定采样状态、单Hann格点的几何接近未能保住完整轨迹。本轮把训练样本改为**当前训练头实际推进的两帧短分支**，检验在后继自身状态上保持原生短分支末帧结果，能否兼顾适配收益和长期稳定性。

实现及3项真实CPU测试已完成，服务器实验待运行。Stage-1～53实验Python源码和既有结果保持冻结。主任务继续为VisDrone-SOT单目标跟踪，原生SGLA骨干、候选、选择器、响应参数及BN缓冲冻结；只更新原size／offset参数。推理仍为原生SGLA的一条路径。

## 对照

| 组 | 内容 | 来源 |
|---|---|---|
| native | 原始SGLA | 复用12条完整轨迹 |
| geometry_control | 固定状态普通几何适配 | 复用Stage-52的24条轨迹 |
| static_anchor | 固定状态、正确fit状态几何保持 | 复用Stage-53 anchor_reliable的24条轨迹 |
| rollout_supervised | 当前训练头产生两帧自身状态，两帧均普通监督 | 新训练；预定匹配控制 |
| rollout_preserve | 同上，增加对原生短分支末帧的保持 | 新训练；预定主要候选 |

主要对比固定为`rollout_preserve_minus_rollout_supervised`。两条新策略的初始化、帧与起点抽样、更新数和前向预算一致；随着参数不同，自身裁剪、归一化标签与回归有效数可以不同。旧静态模型的状态／帧安排不同，只用于背景比较，不能据差值唯一归因于自身状态训练。

## 两帧分支如何形成

1. 沿用Stage-51均匀时刻及相应原生／fit-only donor预测父框，固定两个起点。每折仅读取6个fit视频的缓存；不使用互补折held-out轨迹生成训练起点。丢弃无法容纳后续一帧的视频末尾时刻，实际每视频47个起点、每折282个样本对。
2. 原始SGLA和当前训练头分别从**相同的预测父框**推进两帧，各自把第一帧实际输出作为第二帧父框。每条分支仍使用原始首帧模板；未把当前GT框放入起点或中途重置。
3. 每个优化步都用当前参数重新生成自身短分支，不复用固定donor或早期训练头的后继状态。通过原生采样、选层、Hann选点、解码和裁剪执行，记录真实头输入及父框。
4. 两种新组都生成原生参考和自身分支，之后才附加fit标签、普通监督与资格。GT不参与动作和状态推进。

第一个原生起点分支的两帧输出与冻结原生完整轨迹核对。两条分支的第二帧父框差、实际取整裁剪变化、资格数量都记录；不能把仅有浮点父框变化直接描述为新的图像暴露。

## 训练目标与梯度边界

两种新组都在自身访问的全部帧上使用原有focal＋2×GIoU＋5×L1监督，GT按实际取整裁剪坐标归一化，出裁剪状态仍保留。普通回归仍取raw最高响应格点。

末帧保持目标来自**另一条原生短分支的末帧图像坐标框**。它不是在学生当前裁剪上重新计算的同状态教师图。只有下列条件均满足时，保持项生效：

- 原生分支两帧的官方框IoU均≥0.5；资格仅用于fit训练。
- 原生末帧框中心仍落在学生末帧的可用裁剪区域，避免以该约束要求学生从没有目标的裁剪内找回目标。

用学生当次真实actor选择的Hann格点重算可导几何框，按原生float32尺度运算、float64父框映射和margin=10图像裁剪解码。重算与actor框误差≤0.001像素；响应重放误差≤1e-6。逐帧以batch-one重算头，匹配原生数值路径。

```text
E_end = mean(max(abs((center_student - center_native_end)
                     / size_native_end) - 1e-6, 0))
      + mean(max(abs(log(size_student / size_native_end)) - 1e-6, 0))

L_supervised = mean(ordinary_loss_on_all_own_frames)
L_preserve   = L_supervised + 1.0 * mean(qualified * E_end)
```

末帧误差按全部两个起点平均，未满足资格者贡献0，不按当前有效数重新放大。系数1.0、长度2、资格阈值0.5及死区均固定，不做搜索。

**梯度在帧间状态处截断。** 原生图像裁剪、取整、argmax、第一帧输出父框及参考目标均detach。末帧保持项直接训练末帧几何输出，第一帧另有普通监督；共享几何参数在两帧使用。它是实时自身状态上的截断末帧保持，不是可导图像裁剪或完整跨帧反向传播，不能据此预先宣称长时状态误差已被约束。

## 匹配预算与来源

- 仍为2折、种子2132／2133、原始SGLA头初始化、256次更新、AdamW学习率1e-5／权重衰减1e-4／梯度裁剪1。
- 每步改为1个时刻样本对×2个预测起点×2帧，仍为4个监督视图；每模型总1,024视图。两条新策略共享完整抽样指纹；旧静态参考的帧安排与本轮不同。
- 每模型原生参考1,024次完整前向，自身分支1,024次完整前向，梯度重算1,024次仅头前向。全8个正式模型共**16,384次完整前向、114,688个block调用、8,192次仅头前向**。两组都计算保持项的影子梯度用于审计，控制组不会将它加进优化损失。
- 原生模板初始化仅预处理，不执行网络前向；预处理成本仍存在。中断后的未提交工作重跑会额外花费计算，报告预算对应已提交优化步。
- 新增8个正式头、48条完整OOF轨迹，复用60条参考轨迹。smoke的2个短训头及3条32帧前缀另列，不计入正式预算。
- 正式每次完整调用均核对原生6个前缀块＋1个候选、一次selector和一次head。仅头梯度重算没有backbone梯度。响应、BN和原始网络恢复后SHA继续核对。
- prepare验证完整Stage-53及来源链的模型、缓存和表格，并从轨迹重算其108条种子—视频指标。只在fit缓存读取父框；原先缓存特征不替代本轮实时后继特征。

## 服务器执行

同步仓库后，从项目目录重设变量。无需单独collect，实时分支生成包含在每个训练步内。

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export STAGE53_ROOT="$PWD/output/stage53/native_geometry_preservation_v1"
export STAGE54_ROOT="$PWD/output/stage54/short_rollout_endpoint_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage54_rollout.py -v
python -B tracking/stage54_rollout_experiment.py \
  --mode prepare --input "$STAGE53_ROOT" --output_dir "$STAGE54_ROOT"
cat "$STAGE54_ROOT/preparation/stage54_anchor_pools.csv"
python -B tracking/stage54_rollout_experiment.py \
  --mode smoke --input "$STAGE54_ROOT" --gpu 0
```

出现`PASS_SHORT_ROLLOUT_SMOKE`后：

```bash
python -B tracking/stage54_rollout_experiment.py \
  --mode train --seed 2132 --input "$STAGE54_ROOT" --gpu 0
python -B tracking/stage54_rollout_experiment.py \
  --mode train --seed 2133 --input "$STAGE54_ROOT" --gpu 0
python -B tracking/stage54_rollout_experiment.py \
  --mode evaluate --input "$STAGE54_ROOT" --gpu 0
python -B tracking/stage54_rollout_summary.py --input "$STAGE54_ROOT"

cat "$STAGE54_ROOT/analysis/stage54_scope_summary.csv"
cat "$STAGE54_ROOT/analysis/stage54_policy_summary.csv"
cat "$STAGE54_ROOT/analysis/stage54_retention_summary.csv"
cat "$STAGE54_ROOT/analysis/stage54_focus.csv"
```

先提供这四张表，分别8、5、5、25行。中断后重跑同一root／seed，按已提交参数和AdamW状态续训；完整评价按轨迹缓存复用。Stage-51～53及其上游产物需保留。

## 判读与输出

首先看scope的两类网络调用、仅头调用、资格数量、响应／解码误差，以及实际后继裁剪变化。再看预定主比较、原生保持、固定10条获益队列和0304／0307／0176／0226／0204五个案例。`control_failure_auc`沿用冻结geometry_control两个种子任一IoU<0.2帧的并集，仅作离线描述。

积极开发条件固定为：主要均值≥+0.5 AUC、名义95% CI下界>0、两个种子主要均值均>0；对原生宏AUC不下降、低IoU不恶化、可靠帧均值≥−0.5、原稳定视频无新增失效；各正式模型确有后继取整裁剪变化。满足时为`PROMISING_SHORT_ROLLOUT_ENDPOINT_PILOT`，否则为`REVIEW_SHORT_ROLLOUT_ENDPOINT_PILOT`。−0.5是实用容限，不是零损害证明。

种子先在视频内平均，再做5,000次视频bootstrap；两个新组对native的AUC另列家族区间，其他区间为名义读数。当前源组已用于设计，重叠拟合下区间不包含重训练不确定性。两帧末帧保持、完整长时收益、独立验证分别判断。

| 文件 | 用途 |
|---|---|
| `preparation/stage54_anchor_pools.csv` | 两折fit起点数和指纹 |
| `analysis/stage54_scope_summary.csv` | 调用、资格、后继状态／裁剪变化、解码及梯度 |
| `analysis/stage54_policy_summary.csv` | 五组完整轨迹宏指标 |
| `analysis/stage54_retention_summary.csv` | 来源10条获益队列的收益保持和全体最差损伤 |
| `analysis/stage54_paired.csv` | 新策略主要／次级对比及区间 |
| `analysis/stage54_focus.csv` | 五个固定视频的逐种子与恢复读数 |
| `analysis/stage54_video_summary.csv`、`stage54_seed_video_summary.csv`、`stage54_events.csv` | 完整逐视频、种子和事件 |
| `analysis/stage54_decision.json` | 状态、预算、来源及结果哈希、解释边界 |

## 本机验证

3项真实CPU测试通过（92.571秒，无跳过）。使用仓库实际CENTER小通道实例及原生解码／调用hook，验证：第二帧父框等于第一帧实际输出；修改后续GT不改变actor输出；保持项对size／offset有梯度且响应无梯度；末帧反传不进入第一帧解码图；临时头装载异常可恢复原网络。完整合成流程先生成Stage-51～53来源，再执行Stage-54 prepare／smoke／训练／OOF／汇总，核对fit隔离、末尾样本剔除、AdamW断点恢复一致、实时前向预算、48条新评价、缓存复用和来源／checkpoint／结果篡改拒绝。5个文件AST及纯NumPy帮助命令通过。CPU实现验证不代替服务器精度实验。
