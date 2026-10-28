# Stage-34：单路径学生的空间身份表征与困难负样本机制验证

Stage-31～33 没有建立冻结特征后外接候选排序器的纠错收益。Stage-34 直接训练像素前向中的空间 token 表征，检验在压缩为单路径时，模板条件的目标身份约束是否能改善实际跟踪。

这是一轮新的结构/监督机制实验，不把 L5–L6 微调本身当成创新，也不预先宣称空间对比损失就是新方法。关键证据来自匹配 2×2 对照及未使用过的源组完整视频检查。

## 结构与匹配对照

学生物理保留 L1–L6，删除独立 L7–L12 分支及原选层 MLP。六层后只调用一次原 CENTER 定位头，另加一个小型模板条件身份读出：

1. 从当前联合前向的模板 token 中，按首帧已给定目标区域池化 query。模板图像始终来自初始化，不用后续预测更新模板。
2. 将 query 与每个 search token 投影到 32 维，计算归一化身份相似度图。
3. 用带可学习增益的 logit 残差修正定位响应，再用原尺寸/偏移图解码。增益初始为零，响应与 Fast 初始化严格相同，同时保留增益的梯度。

query 所用的模板 token 会在当前联合编码中与 search 交互，因此它不是与当前图像无关的永久向量。模型没有上一帧对象描述子银行，也没有 GT 纠正或事件触发。

| 组别 | 视觉表征 | 训练监督 |
|---|---|---|
| `readout_standard` | 六层全部冻结，只训练身份读出 | 原定位监督 |
| `readout_contrast` | 同上 | 定位监督＋空间身份约束 |
| `representation_standard` | 同时训练 L5–L6 与身份读出 | 原定位监督 |
| `representation_contrast` | 同上；预定主方法 | 定位监督＋空间身份约束 |

四组结构相同、初始权重相同、同一 seed 的采样顺序和更新预算相同。原定位头、patch embedding、位置编码、L1–L4 和最终 norm 均冻结；模型保持 eval 模式，不更新 BN 或开启随机 dropout。

空间身份约束使用训练 GT 区域内 token 作为正位置，并始终包含原 CENTER 监督采用的量化中心格，避免小目标产生冲突标签。在目标框 1.5 倍区域及正位置一格邻域外，按当前响应选最多 8 个困难空间负位置。对比项固定权重 0.25、margin 0.2、temperature 0.1。负位置是几何负样本，不自动等于经过语义确认的干扰物；框内也不等于目标完全可见。

所有训练组均计算定位/身份读出及矿选操作，只有 contrast 两组把对比项计入优化目标。读出冻结组不能把该项传给原表征，表征训练组可以；smoke 显式核对这条梯度路径。

## 数据与固定训练预算

- 训练仍用原 Stage-31 的 8 个 fit 视频；它们是已使用过的开发数据。
- 训练裁剪来自已冻结的原始 SGLA 预测父框，不用 GT 重置父框，也不把这种状态暴露称为 on-policy。
- 每视频最多 64 个固定样本，先取最多 32 个原生 IoU<0.5 的困难样本，再取容易样本，剩余额度从可用样本补足。
- 训练标签必须在实际裁剪可用区域中覆盖≥0.9，且目标中心在裁剪中；还需留出至少一个空间负位置。此筛选只用于训练监督，检查时不筛帧。
- 至少需要 32 个困难训练样本、来自至少 4 个视频。采样概率在易/难两组之间各 50%，组内按视频、帧均衡。
- 从 VisDrone 原训练目录的剩余源组中，固定 seed=2103 选 **6 个新 checking 视频**。排除 Stage-27 的全部诊断组，以及 Stage-31 的全部 14 组，包括那 6 个旧 checking 组。
- 不读取被排除旧 checking 的标注内容。新 checking 的标注仅在建表阶段审计几何/首框；训练不读取新 checking 图像或标注。
- 组数不足就停止，不缩小检查集、不复用旧失败案例。
- 固定种子 **2103、2104、2105**，每组 **512 次更新，batch size=2**，AdamW 学习率 1e-4。没有 tune、早停、最佳 epoch 或 seed 选择。

共 12 个学生训练。训练样本按原始轨迹采样；检查则从首帧独立运行完整视频，全部使用各模型自己的输出生成后续裁剪。

## 服务器步骤

通过仓库同步新增文件，使用现有 `sglatrack-test` 环境中的 PyTorch、torchvision、OpenCV。无需新数据集下载。

```bash
export STAGE34_ROOT="$PWD/output/stage34/spatial_identity_student_v1"

python tracking/test_stage34_identity_representation.py

python tracking/stage34_identity_experiment.py \
  --mode prepare \
  --input "${STAGE31_ROOT:?请设置原Stage31目录}" \
  --stage33_dir "${STAGE33_ROOT:?请设置原Stage33目录}/full_v1" \
  --output_dir "$STAGE34_ROOT"
```

prepare 无需 GPU。它先验证已完成的 Stage-33 来源和原 fit 缓存，再检查训练支持量、剩余源组数，审计选中的新 checking 全视频图像与标注，最后冻结 `manifest.json`。原文件不会被改写。

先进行短训练 smoke：

```bash
python tracking/stage34_identity_experiment.py \
  --mode smoke --input "${STAGE34_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?请设置原始SGLA权重}" \
  --parameter deit_distilled --gpu 0

cat "$STAGE34_ROOT/smoke/stage34_training_decision.json"
```

smoke 使用 seed2103、四组各 8 次更新，不读取新 checking。只有 `PASS_SPATIAL_IDENTITY_SMOKE` 后完整训练入口才放行。其通过只说明真实前向、参数冻结、原模型保持、物理六层、参数量减少和对比梯度通路成立。

完整训练在单 GPU 上依次执行：

```bash
for SEED in 2103 2104 2105; do
  python tracking/stage34_identity_experiment.py \
    --mode train --input "${STAGE34_ROOT:?}" \
    --checkpoint "${CHECKPOINT:?}" \
    --parameter deit_distilled --gpu 0 --seed "$SEED" || break
done
```

每个 seed 完成四组后输出 `COMPLETE_MATCHED_TRAINING`。三个 seed 全部完成并冻结后，才能打开新 checking 结果：

```bash
python tracking/stage34_identity_experiment.py \
  --mode evaluate --input "${STAGE34_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage34_identity_summary.py --input "${STAGE34_ROOT:?}"

cat "$STAGE34_ROOT/analysis/stage34_decision.json"
cat "$STAGE34_ROOT/analysis/stage34_policy_summary.csv"
cat "$STAGE34_ROOT/analysis/stage34_paired.csv"
```

检查包含原始 SGLA、未训练 Fast，以及 12 个训练学生，共 **6×14=84 条完整轨迹**。每种方法重新初始化自身状态；未来 GT 在该条完整轨迹生成后才读取。方法执行顺序按视频轮换。不会提前查看 checking 选择训练轮次或模型。

## 固定判定

先在每个视频内平均三个 seed，再对六个源组做配对 bootstrap。三个 seed 不是额外的独立视频。

预定主方法 `representation_contrast` 同时满足：

1. 相对 `representation_standard` 的宏平均 AUC 增益≥0.3 点，95% 区间下界>0，且三个 seed 的平均增益均>0；
2. 交互效应 `(representation_contrast−representation_standard)−(readout_contrast−readout_standard)` 的 95% 区间下界>0；
3. 相对 `readout_contrast` 的 AUC 平均差>0；
4. 相对未训练 `fast_initial` 的 AUC 增益≥0.3 点，95% 区间下界>0，避免只优于被训练损坏的对照；
5. 相对 Fast 初始化，宏平均低 IoU 帧比例不增加；至少有一个原本稳定的视频，且三个 seed 均不为原本稳定视频引入新的持续低 IoU 事件。

全部满足才报告 `PASS_SPATIAL_REPRESENTATION_PILOT`。否则是 `NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED`。原 SGLA 作为质量参照完整报告；该机制筛查通过仍不表示已经追平 SGLA、建立最佳精度/资源权衡或完成独立外部验证。

## 恢复与资源解释

同一训练命令可在中断后重跑。每 128 次更新保存 trainable 参数、优化器、固定 step 和历史，并写 SHA256 完成回执。恢复会核对所有已完成 checkpoint；没有回执的当前孤立保存只能从上一个完成点重算。不会覆盖已提交的 checkpoint。

相同 seed 下四组使用预生成的相同采样序列，恢复后不会换样本顺序。已完成四组会核验并复用。不要同时在一张 GPU 上启动多个训练进程。

保存的是可训练状态，冻结部分由原权重重建；本阶段尚未打包独立部署程序。物理学生参数量在 smoke/训练报告中统计。原 SGLA 常规执行的是六层前缀加一个候选层，并非每帧执行全部十二层，所以删除六个候选模块不等于推理时延减半。

检查中的时间只覆盖带同步的 crop/model/decode，排除读图、标签和初始化，不是完整无人机端 FPS。检查进程同时保留原模型及对照学生，不能用其进程显存当作独立部署占用。没有真实设备上的独立部署基准前，不宣称速度或能耗收益。

一旦检查结果被读取，这六组即成为已使用开发数据。后续不得在其上调整损失权重、融合强度、训练步数、采样配额或选择 seed；先保留本轮固定结论，再决定是否有新的机制值得验证。
