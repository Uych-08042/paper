# Stage-53：原生SGLA的训练期几何保持

Stage-52的geometry_only相对原生10胜2负、中位数+2.6419，但0304／0307合计−56.0919抵消了10条正收益的+58.8606。本轮固定原生SGLATrack、训练状态和几何更新范围，检验：**在原生已能正确定位的fit状态上约束几何漂移，能否保留多数适配收益并减少长期失跟？**

实现及3项真实CPU测试已完成，服务器实验待运行。Stage-1～52实验Python源码与已有结果不修改。使用VisDrone-SOT、原生SGLA及首帧目标框初始化；本轮为训练期输出保持原型，推理结构和原生跟踪流程保持一致。

## 五组对照与数据来源

| 组 | 训练内容 | 来源 |
|---|---|---|
| native | 无适配 | 复用原生轨迹 |
| geometry_control | 普通监督，仅更新size＋offset | 按字节复用Stage-52 geometry_only |
| joint | 普通监督，更新三个头分支 | 按字节复用Stage-51 state_mix，由Stage-52来源链绑定 |
| anchor_all | geometry_control训练配置＋所有状态的原生几何保持 | 新训练 |
| anchor_reliable | 同上，仅在原生定位正确的fit状态保持；预定主要候选 | 新训练 |

两种新组都从**原始SGLA预测头**初始化，只训练原size／offset卷积塔及对应BN仿射参数；ctr参数和所有BN缓冲冻结。与geometry_control共享2折、种子2132／2133、256步、每步2个样本对×2视图、学习率1e-5、AdamW权重衰减1e-4、梯度裁剪1。初始化、参数范围、特征文件、抽样指纹及普通监督计数核对来源控制。

沿用Stage-51完整fit-only状态池：每个均匀时刻使用原生预测父框视图和固定fit内donor预测父框视图。全部出裁剪状态仍保留普通监督；不筛掉困难样本，不加入held-out坏帧，不变更混合比例。没有新donor轨迹或特征采集。

## 保持约束的定义

每个固定状态已有原始SGLA的响应、size、offset缓存。用原生Hann加权响应的最高格点，解码该状态的原生几何框。与实际裁剪坐标中的fit GT计算连续IoU，固定定义：

`q_i = 1[IoU(native_Hann_box_i, GT_crop_i) >= 0.5]`

此处是不经图像边缘裁剪、未经整数取整的连续裁剪坐标IoU，用于训练资格定义；它不是完整视频的官方整数框AUC，也不是在线可靠性检测器。原生框不正确时，anchor_reliable仍接受普通监督，只关闭保持约束。

在同一冻结格点取新头的size和offset，定义每视图保持误差：

```text
E_i = mean(max(abs(log(size_student / size_native)) - 1e-6, 0))
    + mean(max(abs((offset_student - offset_native)
                  / (feature_side * size_native)) - 1e-6, 0))
```

第一项约束相对宽高漂移；第二项是以原生预测宽高归一化的中心变化，格点坐标相消。1e-6数值死区避免缓存浮点噪声触发L1梯度。缓存教师、资格和归一化尺度全部detach；约束直接作用于可训练size／offset，而不是对已冻结的响应图加常数损失。

权重按**整折fit池**计算，不按当前batch或held-out结果计算：

```text
p_fit = mean(q_i)  # 两种选中视图均计入
w_all = 1
w_reliable = q_i / p_fit
L = L_ordinary + 1.0 * mean(w_i * E_i)
```

整折平均保持权重都为1，帮助区分资格选择和简单减少约束量。抽样后实际权重和梯度范数仍可能不同，scope表报告实际均值。每折必须同时存在正确和不正确教师状态，否则prepare拒绝进行该对比。门槛0.5、系数1.0及死区不搜索；它们是本轮固定假设，不能在结果出来后调整。

普通监督沿用focal＋2×GIoU＋5×L1，框来自raw argmax；保持项作用于冻结原生Hann格点。该格点更贴近原生输出选择，但单格点约束不保证整个状态分布或长时轨迹被保护。缓存教师不需要额外网络前向；新增梯度审计会增加训练计算，相同步数不是相同FLOPs。

本轮与早期Stage-35的保留思路有联系，已有失败结果继续保留。这里具体检验原生头的几何更新、固定自身状态暴露、实际Hann格点以及平均权重匹配的全状态／正确状态对照，不据这些差异预先宣布新方法成立。

## 来源与执行核对

- prepare要求完整Stage-52 manifest、smoke、8个模型、48条新轨迹和汇总，通过来源链校验模型／特征／表格／缓存哈希，并从轨迹重新计算84条Stage-52种子—视频指标。
- 资格比例与保持权重只从相应折6个fit视频的已有特征计算。`preparation/stage53_anchor_support.csv`列两折视图总数、有效数、比例、归一化系数及门控指纹。正式每折288个样本对、576个选中视图。
- 全模型非选中参数与BN缓冲SHA、固定特征上的响应误差、实际几何与保持项梯度都核对；保持项最大梯度必须非零。
- 新增8个正式头和48条完整OOF轨迹，复用12条native、24条geometry_control及24条joint轨迹。smoke另含2个8步短训头、原生和两种新组各32帧前缀，不计入正式预算。
- 评价沿各自预测继续，每帧6个prefix block＋1个候选block、一次selector与一次head。资格、GT和教师缓存均不进入推理控制；首帧之后GT仅用于评价。

## 服务器命令

同步仓库后执行，从项目目录重设变量：

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export STAGE52_ROOT="$PWD/output/stage52/fixed_state_branch_adaptation_v1"
export STAGE53_ROOT="$PWD/output/stage53/native_geometry_preservation_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage53_geometry.py -v
python -B tracking/stage53_geometry_experiment.py \
  --mode prepare --input "$STAGE52_ROOT" --output_dir "$STAGE53_ROOT"
cat "$STAGE53_ROOT/preparation/stage53_anchor_support.csv"
python -B tracking/stage53_geometry_experiment.py \
  --mode smoke --input "$STAGE53_ROOT" --gpu 0
```

出现`PASS_GEOMETRY_PRESERVATION_SMOKE`后：

```bash
python -B tracking/stage53_geometry_experiment.py \
  --mode train --seed 2132 --input "$STAGE53_ROOT" --gpu 0
python -B tracking/stage53_geometry_experiment.py \
  --mode train --seed 2133 --input "$STAGE53_ROOT" --gpu 0
python -B tracking/stage53_geometry_experiment.py \
  --mode evaluate --input "$STAGE53_ROOT" --gpu 0
python -B tracking/stage53_geometry_summary.py --input "$STAGE53_ROOT"

cat "$STAGE53_ROOT/analysis/stage53_scope_summary.csv"
cat "$STAGE53_ROOT/analysis/stage53_policy_summary.csv"
cat "$STAGE53_ROOT/analysis/stage53_retention_summary.csv"
cat "$STAGE53_ROOT/analysis/stage53_focus.csv"
```

先提供这四张表，分别8、5、5、25行。prepare和smoke失败时不继续正式训练；在相同代码、root和seed上重复命令可断点恢复，已提交轨迹按缓存复用。Stage-51／52及其依赖产物需保留；无需重跑旧采集和训练。

## 主要比较与解释

主要对比预定为`anchor_reliable_minus_geometry_control`，另列anchor_all对控制、reliable对all，以及两个新组对native的AUC、低IoU、最长失效、原生可靠帧与control失败队列读数。种子先在视频内平均，再按视频bootstrap 5,000次。额外家族区间仅覆盖两个新组对native的AUC，其他区间为名义读数。

保留Stage-52 geometry_control优于原生的固定离线视频队列，报告这些视频中有多少仍优于原生、平均收益和相对控制变化；同时保留全12视频的最差损伤。该队列在来源中有10条，不作为训练选择或视频名称触发规则。focus仍固定0304／0307及0176／0226／0204，列逐种子AUC、首次失效、末尾未恢复与恢复时刻。

`control_failure_auc`使用冻结geometry_control两种子任一IoU<0.2的帧并集；它与旧joint失败队列不同，仅用于离线描述。各自事件起点不同，恢复次数变化不能直接代表共同起点的因果恢复概率。

开发pilot的积极条件固定为：主要均值≥+0.5、名义CI下界>0、两个种子主要均值均>0；对原生宏AUC不下降、低IoU不恶化、可靠帧均值不低于−0.5、原稳定视频无新增失效；相对anchor_all平均AUC不下降。满足时为`PROMISING_NATIVE_GEOMETRY_PRESERVATION_PILOT`，否则为`REVIEW_NATIVE_GEOMETRY_PRESERVATION_PILOT`。另单列reliable−all的CI下界是否>0，区分原型改善与资格选择的新增证据。−0.5是开发实用容限，不是无损证明。

当前12个源组已用于设计，bootstrap条件于固定重叠拟合，不能建立外部泛化或重新训练稳定性。保持成功、小规模原型收益、后续独立验证分别报告；不会自动将PROMISING等同于新方法或部署收益。

| 文件 | 用途 |
|---|---|
| `preparation/stage53_anchor_support.csv` | 两折fit-only资格与归一化尺度 |
| `analysis/stage53_scope_summary.csv` | 冻结／梯度／匹配抽样、实际保持权重与误差 |
| `analysis/stage53_policy_summary.csv` | 五组宏指标与来源 |
| `analysis/stage53_retention_summary.csv` | 固定10条获益视频的收益保留及全体最差损伤 |
| `analysis/stage53_paired.csv` | 预定配对读数与区间 |
| `analysis/stage53_focus.csv` | 固定五视频的逐种子结果与恢复信息 |
| `analysis/stage53_video_summary.csv`、`stage53_seed_video_summary.csv`、`stage53_events.csv` | 完整逐视频、种子及事件 |
| `analysis/stage53_decision.json` | 状态、来源与输出哈希、固定门槛和解释边界 |

## 本机验证

3项CPU测试通过（53.814秒，无跳过）：连续坐标IoU、Hann／raw格点区别、fit内资格和匹配平均权重；真实仓库CENTER小通道实例中size与offset的保持项梯度非零、ctr无梯度、BN缓冲冻结、错误视图不进入有条件保持。完整合成流程先生成Stage-51和Stage-52来源，再执行Stage-53 prepare／smoke／训练／OOF／汇总，验证fit池不读held-out缓存、断点参数及AdamW恢复一致、48条新轨迹预算、缓存复用、原网络冻结和来源表／checkpoint／结果篡改拒绝。纯NumPy命令帮助及5个文件AST检查通过。CPU合成验证不代替服务器精度实验。
