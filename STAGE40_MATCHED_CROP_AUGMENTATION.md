# Stage-40：匹配父框扰动增强训练对照

Stage-39 的主要额外损失为 −0.00645，名义95%区间 [−0.02121,+0.00274]，四项门槛全部未通过。不能继续把“六层压缩必然放大父框误差”当作已证实的动机。两份微调学生出现正的扰动损失，只是一个需要干预验证的线索，不能直接归因于微调。

本轮检验较窄的问题：**普通监督训练加入相同幅度的预测父框扰动，能否改善紧凑学生的完整视频结果，并保住原本可靠的定位？**

这是训练机制的开发筛选。平移增强早已用于跟踪训练，[TransT（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Transformer_Tracking_CVPR_2021_paper.pdf)明确采用普通平移和亮度增强；[SLT（ECCV 2022）](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5099_ECCV_2022_paper.php)已经研究帧级训练与序列跟踪之间的分布和目标不一致。因此本轮的增强本身不作为新方法创新点，也不称为序列训练或压缩特有机制。

## 两组只改变第二个训练裁剪

从同一原始 SGLA 权重重新初始化物理六层学生，删除 L7–L12 和选择器，只训练 L5–L6；head、norm、前四层和嵌入冻结，保持 eval 模式。

| 项目 | clean | augmented |
|---|---|---|
| 每个采样帧的第一视图 | 冻结 SGLA 预测父框 | 同一个父框 |
| 第二视图 | 再次使用同一个父框 | 按固定规则扰动同一个父框 |
| 损失 | 两个普通 GT 定位损失的均值 | 两个普通 GT 定位损失的均值 |
| 教师蒸馏、额外一致性损失 | 无 | 无 |
| 训练前向次数、更新次数、采样帧 | 匹配 | 匹配 |

两组同折同 seed 的初始化、采样顺序、优化器、更新数和两次模型前向严格匹配。clean 两次相同损失取均值，与单次普通监督具有相同目标，不因重复视图把梯度乘二。

第二视图扰动固定为水平／垂直正负 `0.15×sqrt(w×h)` 平移，以及中心不变的0.8／1.25倍宽高缩放。六种扰动在整段抽样中近似均衡分配，顺序由预定 seed 固定，不根据成绩或 GT 选择方向。幅度沿用 Stage-39，不做强度搜索。

GT 标签必须按**实际整数裁剪原点和实际裁剪边长**重新映射到每个视图，不沿用原裁剪的 target，也不裁掉框超出搜索区域的部分。出界时沿用现有 GT 损失的规则，不丢弃困难视图。训练父框来自原 SGLA 缓存，训练中不会按学生输出继续推进；这次只改变静态训练状态覆盖，不再重复 Stage-20 的学生轨迹训练。

## 数据与固定预算

复用原8个已消耗 fit 开发组、原 Stage-35 的4折划分与固定训练帧。每折6视频训练、2视频 OOF；easy/hard 采样各50%，权重只在该折训练视频内计算。

- seeds=2110／2111，两组均从原始 Fast 初始化，不继续训练旧学生。
- 每模型256次 AdamW 更新，batch=2个基础帧，每帧2个视图，lr=1e-4，weight decay=1e-4，梯度裁剪1。
- 4折×2seed×2组＝16个新模型。
- 不早停、不选择最好 checkpoint；只评价固定最后一步。
- 两个 seed、所有16个模型及收据先冻结，再允许任何完整 OOF 评价。

此前的微调模型仅用于形成假设。本轮重新训练 clean 对照，避免把训练预算变化当增强收益。OOF 训练集相互重叠，数据又已经参与设计，所有区间仍是条件于固定拟合的开发证据，不能当外部验证。

## 两类评价

**主要评价：完整视频自己的预测轨迹。** 两组、两seed各跑8视频，共32条新完整轨迹。每个视频只能用未训练过该视频的折模型。GT 只用于首框初始化，未来 GT 在完整轨迹结束后计分；原 Fast／SGLA 基线从 Stage-35 校验复用。

**补充评价：原 Stage-39 固定锚点的扰动压力。** 同样的8视频×3锚点×4模型×7条件×20帧＝13,440个分支帧，共96个模型／锚点包。每个包内7条短分支只在初始父框干预一次，随后使用各自预测。起点使用同一个冻结 SGLA 预测父框，非 GT。第一帧直接效应不计入主要压力损失，报告第2至20帧的 clean-minus-perturbed 平均 IoU 损失。

补充评价的幅度和方向在训练中出现过，因此测量的是对未参与该折训练的视频的迁移，不是对未知扰动类型的泛化，也不代表自然失败发生率。

## 预定通过规则

先在每视频内平均两seed，再做5000次视频配对 bootstrap，seed2119。帧、锚点和重复seed不是独立视频。主要对比为 `augmented − clean` 的完整视频 AUC（0–100点）。

主要精度条件同时要求：均值≥+0.3 AUC点、名义95%下界>0、至少6／8视频正收益、去掉任一视频后的均值仍>0、两seed各自平均收益均>0。

保留条件同时要求：

1. 原 Fast 可靠帧（IoU≥0.5）上，augmented−Fast 的条件 AUC 区间下界≥−0.5点、最差单视频差≥−1点。每视频至少10可靠帧，至少4个支持视频；这是实际容忍界，不能说完全无损。
2. 对原 Fast 稳定视频，任一 seed 都不能新引入持续失跟；对同seed的 clean 稳定视频也不能引入新失跟。至少存在一个原 Fast 稳定视频。
3. augmented 的视频宏平均低 IoU 帧占比不高于 clean，也不高于原 Fast；整体 AUC 均值不低于原 Fast。

满足精度与保留条件才为 `PASS_MATCHED_AUGMENTATION_DEVELOPMENT_SCREEN`；支持不足为 `INSUFFICIENT_RETENTION_SUPPORT`；其余为 `NO_MATCHED_AUGMENTATION_GAIN_ESTABLISHED`。

压力损失另报告 `D(augmented)−D(clean)`，负值较好。只有其视频配对区间上界<0才支持压力损失下降；只有它和主要开发门槛同时通过，`augmentation_stability_mechanism_supported` 才为 true。不能只凭压力表变好就宣布完整视频方法通过。

这些条件不包含“超过 SGLA”的要求；相对 SGLA 的结果会完整报告。即使开发筛选通过，也只是明确普通增强对照是否值得保留，`novel_method_established` 始终为 false。

## 服务器执行

通过 Git 同步新增文件后执行，无需下载新数据：

```bash
cd ~/workspace/paper
export STAGE39_ROOT="$PWD/output/stage39/matched_parent_perturbation_v1"
export STAGE40_ROOT="$PWD/output/stage40/matched_crop_augmentation_v1"

ls -l "$STAGE39_ROOT/analysis/stage39_decision.json"
python tracking/test_stage40_crop_augmentation.py

python tracking/stage40_crop_experiment.py \
  --input "$STAGE39_ROOT" --mode prepare \
  --output_dir "$STAGE40_ROOT"

python tracking/stage40_crop_experiment.py \
  --input "$STAGE40_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

cat "$STAGE40_ROOT/smoke/stage40_smoke.json"
```

smoke 使用首折、seed2110，两组各8次更新；随后在首折第一个 held-out 视频的第一固定锚点，各做7条4帧分支。`PASS_MATCHED_CROP_SMOKE` 仅表示梯度、匹配预算、坐标、冻结状态和执行检查通过。

权重查找沿用 `--checkpoint` → `CHECKPOINT` 环境变量 → 原 tracker 配置，并核对原 SGLA 哈希。路径变量会在代码中预检查；不要让旧值指向 `~/workspace/output/...`。

smoke 通过后，两seed依次训练：

```bash
for SEED in 2110 2111; do
  python tracking/stage40_crop_experiment.py \
    --input "$STAGE40_ROOT" --mode train --seed "$SEED" \
    --parameter deit_distilled --gpu 0 || break
done

python tracking/stage40_crop_experiment.py \
  --input "$STAGE40_ROOT" --mode status

python tracking/stage40_crop_experiment.py \
  --input "$STAGE40_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage40_crop_summary.py --input "$STAGE40_ROOT"
```

完整评价要求32／32条整视频轨迹、96／96个压力分支包。可在 evaluate 加 `--sequence <原8条之一>` 分视频运行，最终仍要求全部齐全。

## 中断与结果

每64次更新及最后一步原子保存参数、Adam状态和哈希收据。训练中断后重跑同seed同目录，从最近提交步恢复；尚未提交的步重算。完整模型经哈希审计后直接复用。评价的提交单位是整视频轨迹，或一个模型／锚点的7条完整分支；当前未完成单位重算，完成单位复用。

不要在同一输出目录并发 collector。普通 Ctrl+C 会释放锁；强制终止留下 `collect.lock` 时，应确认其中 PID 的原进程已退出，再处理残留锁。

结果优先查看或作为附件提供：

```bash
cat "$STAGE40_ROOT/analysis/stage40_decision.json"
cat "$STAGE40_ROOT/analysis/stage40_policy_summary.csv"
cat "$STAGE40_ROOT/analysis/stage40_video_influence.csv"
```

`stage40_decision.json` 保持紧凑，完整来源哈希另放 `stage40_evidence.json`，由决策文件记录其哈希。其他表包括逐seed／逐视频结果、全部配对对比、稳定性、压力锚点和压力视频表。重跑汇总会核对来源、模型、收据、结果和表格，拒绝混用、缺失或篡改。

本轮物理六层、单头推理，不引入额外部署模块；但没有专门计时实验，不能由此报告速度提升。失败后保留固定结果，不在这些结果上搜索更新数、损失权重、扰动强度、fold或 checkpoint。
