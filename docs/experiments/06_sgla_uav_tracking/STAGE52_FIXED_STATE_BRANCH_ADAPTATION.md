# Stage-52：固定状态池下的响应／几何分支适配

本轮延续原生SGLATrack无人机单目标跟踪。Stage-51已表明实际状态暴露充分，混合比重复原生状态控制有部分收益，但0304／0307的共同适配损伤仍在。因此固定其状态来源、普通监督、抽样与步数，检验：**限制原CENTER头的更新范围，能否保住原生成功轨迹，同时保留对困难状态的改善？**

用户已报告服务器完成，状态为`COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC`，scope／policy／focus／paired／factor五张表已收到；结果解释见[Stage-52结果复核](STAGE52_RESULTS_REVIEW.md)。几何组对原生10胜2负，但0304／0307严重损伤抵消多数收益，宏均值与因子区间跨零。本机3项真实CPU测试已通过，未独立重验服务器完整产物。以下预定协议与复现命令保留；Stage-1～52实验Python源码与既有结果保持冻结，本轮无需重新训练。

## 四组对照

| 组 | 允许更新的参数 | 来源 |
|---|---|---|
| native | 无 | 复用原生SGLA完整轨迹 |
| response_only | `conv1_ctr`～`conv5_ctr`的卷积与BN仿射参数 | 新训练 |
| geometry_only | `conv1_size`～`conv5_size`、`conv1_offset`～`conv5_offset`的卷积与BN仿射参数 | 新训练 |
| joint | 上述三个卷积塔的参数 | 按字节复用Stage-51 `state_mix`模型与完整轨迹 |

真实CENTER头的三座卷积塔彼此独立，没有公共可训练层。代码检查子模块与参数归属，遇到未知公共模块直接拒绝执行。所有分支均为eval模式；全部BN运行均值、方差和计数缓冲保持冻结。未选中的参数不进入AdamW，不能被权重衰减间接修改。模板、backbone、候选层、selector、原生Hann窗口与解码流程均保留。

三个训练组均从原始SGLA预测头初始化，不从Stage-51联合适配权重继续训练。新组沿用相应折／种子的Stage-51 `state_mix`两视图：原生预测父框视图＋固定fit内监督模型的预测父框视图。旧±1像素视图不进入本轮优化；保留Stage-51全部出裁剪状态，不事后过滤或改变混合比例。

沿用2折、种子2132／2133、256次更新、每步2个样本对、每对2视图、学习率1e-5、AdamW权重衰减1e-4、梯度裁剪1。初始化SHA、特征／标签文件绑定、完整抽样序列指纹、监督／回归／中心计数逐一与已有joint模型核对，避免把数据或预算变化误当作分支效果。

训练损失仍为既有普通focal＋GIoU＋L1。需要明确其梯度路径：框回归位置来自raw响应的**硬argmax**，不对位置选择反传；因此response_only的实际更新来自focal项，geometry_only中的focal项为常数。保留相同损失公式不等于两组获得相同梯度。各组可训练参数数目与共同裁剪的梯度范数也不同，这是本轮干预的一部分；相同步数不是相同训练FLOPs。

## 来源、冻结与调用核对

- `prepare`要求Stage-51采集、smoke、8个模型、48条新OOF轨迹及汇总均已完成。验证manifest／run／decision、模型／特征／轨迹／表格哈希，并从轨迹重算84条种子—视频指标与已提交seed表比对。
- 新训练仅调用Stage-51相应折的fit池，复用其固定donor状态；不把互补折held-out轨迹回填训练。held-out仍只用于smoke检查与完整评价。
- 检查非选中参数及全部BN缓冲的SHA，在断点恢复、保存和最终装载时验证。共同固定特征上的冻结输出图前后最大误差须≤1e-6；同时要求选中组存在非零梯度与实际权重改变，未选中组无梯度。
- smoke固定fold0／seed2132，两个新组各8步。检查原生32帧前缀复现，以及两个新头都能由原生tracker运行；通过状态为`PASS_BRANCH_SCOPE_SMOKE`。
- 正式新增**8个头、48条完整OOF轨迹**。复用12条原生及24条joint轨迹；无新donor轨迹或特征采集。smoke的2个短训头及3条短前缀单列，不计入8个正式模型／48条正式轨迹。
- 每个正式评价帧核对原生6个prefix block＋1个候选block、一次selector与一次head调用。评价各组沿自身预测继续；首帧初始化之后，GT只用于指标计算。

固定特征下某个输出塔保持不变，不意味着自身完整轨迹的响应、选择位置或宽高保持不变：状态已经可以随另一分支的输出变化。四组是训练参数范围的因子对照；joint学到的参数不等于把两个单独训练结果拼接，不能把交互量当作纯模块耦合或整条轨迹的唯一因果归因。

## 服务器执行

同步仓库后，从项目目录设置变量，避免沿用指向`~/workspace/output`的过期路径。原有Stage-51及其依赖结果目录需保留。

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export STAGE51_ROOT="$PWD/output/stage51/fit_only_state_source_v1"
export STAGE52_ROOT="$PWD/output/stage52/fixed_state_branch_adaptation_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage52_branch.py -v
python -B tracking/stage52_branch_experiment.py \
  --mode prepare --input "$STAGE51_ROOT" --output_dir "$STAGE52_ROOT"
python -B tracking/stage52_branch_experiment.py \
  --mode smoke --input "$STAGE52_ROOT" --gpu 0
```

出现`PASS_BRANCH_SCOPE_SMOKE`后执行正式训练与评价：

```bash
python -B tracking/stage52_branch_experiment.py \
  --mode train --seed 2132 --input "$STAGE52_ROOT" --gpu 0
python -B tracking/stage52_branch_experiment.py \
  --mode train --seed 2133 --input "$STAGE52_ROOT" --gpu 0
python -B tracking/stage52_branch_experiment.py \
  --mode evaluate --input "$STAGE52_ROOT" --gpu 0
python -B tracking/stage52_branch_summary.py --input "$STAGE52_ROOT"

cat "$STAGE52_ROOT/analysis/stage52_scope_summary.csv"
cat "$STAGE52_ROOT/analysis/stage52_policy_summary.csv"
cat "$STAGE52_ROOT/analysis/stage52_focus.csv"
```

前两表分别8行和4行，focus固定5个视频×4组共20行；先提供这三张小表即可。中断后重跑相同root／seed命令，使用已提交参数和AdamW状态恢复。完整评价按轨迹复用缓存。prepare后不改代码或配置；不覆盖已提交结果。

## 结果解释与后续空间

先确认scope审计通过，再看全12视频宏AUC、低IoU、原生可靠帧AUC及新增失效。种子先在视频内平均，再以视频为单位做5,000次条件bootstrap。两个“新组减native”的AUC比较额外给出Bonferroni百分位家族区间（每个97.5%，分位数0.0125／0.9875）；其他区间为名义探索读数。两折拟合有重叠，区间不包含重新训练的不确定性，当前源组已被用于设计。

focus固定保留0304／0307的共同损伤，以及0176／0226／0204的恢复取舍；提供逐种子AUC、首次失效、末尾未恢复与实际恢复时刻。不得只展示有利视频或种子。`joint_failure_auc`使用冻结Stage-51 joint两个种子任一IoU<0.2的帧并集；它是离线结果定义队列，与旧`source_supervised_failure_auc`不同。事件计数来自各自轨迹，恢复次数差异不是同起点恢复概率的因果比较。

| 结果模式 | 可推进的具体问题 |
|---|---|
| 某个更新范围能减轻0304／0307损伤，且保留困难视频改善 | 将其作为后续低损伤适配方案的基础，再检验新增保护机制及独立组迁移 |
| 一个范围保护原生成功轨迹，另一个改善困难视频 | 研究训练期能力保持与选择性更新的取舍，以固定范围基线建立新增价值 |
| 两个范围仍有共同损伤 | 当前固定范围不足；结合全部状态与事件判断需要改善训练分布、目标定义或输出保持的哪个环节 |

本轮完成状态固定为`COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC`，不自动选赢家或宣布新方法通过。它直接缩小下一步训练机制的设计空间；不要求分支冻结本身完全没有先例。任何后续方案都应说明相对本轮简单对照新增了什么价值，并单独验证完整轨迹与能力保持。

| 文件 | 用途 |
|---|---|
| `analysis/stage52_scope_summary.csv` | 参数规模、冻结SHA／输出图、两组梯度及匹配抽样审计 |
| `analysis/stage52_policy_summary.csv` | 四组全视频指标及复用／新训练来源 |
| `analysis/stage52_paired.csv` | 五类预定对比、可靠帧与失败队列、区间 |
| `analysis/stage52_factor_summary.csv` | 更新范围的顺序平均效应及交互量 |
| `analysis/stage52_focus.csv` | 五个固定关注视频的逐种子与恢复信息 |
| `analysis/stage52_video_summary.csv`、`stage52_seed_video_summary.csv` | 全部逐视频／逐种子指标 |
| `analysis/stage52_events.csv` | 失效、恢复和视频末尾截断；无事件时为空表 |
| `analysis/stage52_diagnostic.json` | 来源、预算、输出哈希与解释边界 |

## 本机验证记录

3项CPU测试通过（35.652秒，无跳过）。使用仓库原`conv`和`CenterPredictor`定义的小通道实例验证独立塔、BN缓冲、硬argmax梯度及冻结输出；以真实Torch卷积、原解码与native hook构成的合成流程，先生成完整Stage-51来源，再完成Stage-52 prepare／smoke／两种子训练／OOF／summary。核对fit池不读held-out缓存、初始化／抽样／标签计数匹配、断点AdamW恢复后权重与不中断一致、48条评价预算、缓存复用、原网络冻结及来源表／checkpoint／结果篡改拒绝。CPU合成流程验证实现，不代替服务器真实数据精度。
