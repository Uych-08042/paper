# Stage-33：困难状态均衡的完整候选描述子留一验证

Stage-32 显示，九维身份排序器的全部训练候选对准确率为 93.50%，但原生低 IoU 困难状态只有 43.83%；近期身份和其他候选线索在这些状态中也经常偏向错误对象。因此，下一步先检验池化描述子是否包含这些标量摘要没有利用的信息。

这是固定预算的开发可行性实验。候选关联和干扰物记忆已有 [KeepTrack](https://openaccess.thecvf.com/content/ICCV2021/html/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.html) 与 [DAM4SAM](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html) 等工作。本实验不把简单投影或候选关联命名为新的压缩方法。

## 范围与隔离

- 只读取原 Stage-31 的 **8 个 fit 视频**，使用已保存的原生 SGLA 轨迹。原 6 个 checking 视频的图像、标注、缓存和决定文件均不打开。
- 这些 fit 视频已经参与过开发分析。本轮的按视频留一，仅保证被留出视频不进入当前探针的参数拟合、归一化和采样分布；不能消除先前方法设计使用这些开发数据的选择影响。
- 缓存保存的是从 search token 按候选区域池化并归一化的描述子。**它不包含完整 token 网格。** 实验失败不能推出完整 SGLA 特征中不存在身份信息。
- 不重跑 SGLA 主干推理，不修改原主干、预测头、候选集合、原型更新规则或原跟踪轨迹。只用 NumPy 训练很小的探针，在已有候选上评估同状态排序。
- 初始身份锚点来自已经允许使用的首帧框。未来 GT 只用于离线监督及结果统计，不进入评分函数。

## 三个固定对照

| 名称 | 输入与训练 |
|---|---|
| `scalar_balanced` | 原九维特征；重新在每个训练折内拟合，使用新的易/难均衡目标。 |
| `descriptor_fixed` | 首帧锚点与完整候选描述子经过同一个固定随机投影和 tanh；只训练关系评分头。 |
| `descriptor_learned` | 与 fixed 相同的初始投影、评分头、配对样本和采样序列；同时训练投影。预定主模型。 |

两个描述子模型的评分输入为：投影后锚点与候选的逐元素乘积、绝对差，以及当前相对响应和加权响应。它们不读取可变的近期/可信/其他候选记忆，也不读取视频编号、帧编号或候选 GT 排名。

共享投影为 `tanh(xW)`，默认 16 维。测试原描述子的可学习部分不等于对 SGLA 主干做压缩。九维模型与描述子模型的容量和输入不同，fixed 投影对照用于进一步判断训练投影是否优于仅增加固定的向量关系输入。

另外报告原响应排名和初始余弦两个未训练参照。原 Stage-31 在全部 8 个 fit 视频上训练的排序权重**不作为 OOF 预测或控制输入**。

## 数据平衡与固定预算

- 正候选 IoU≥0.5，负候选 IoU<0.1，中间候选不构成排序训练对。
- 有正负候选对且原生候选 IoU<0.5，为困难训练状态；原生候选 IoU≥0.5，为容易状态。
- 采样权重首先给容易/困难两组各 50% 质量；每组内视频等权，再对视频中的帧等权，最后帧内正负候选对等权。每个 minibatch 按此固定概率采样，不要求每批恰好各一半。
- **8 折按视频留一**：每次用其余 7 个源组训练。投影、评分头、标量归一化、配对权重全部只来自这 7 个视频。
- **2 个种子：2100、2101**。每个模型固定 **600 次更新、每次 128 个候选对**，Adam 学习率 0.003；投影 16 维，固定正则与裁剪。共 8×2×3＝48 个小模型。
- 不进行验证集搜索、早停、训练轮次延长或选择最佳 seed。每个模型只使用固定预算结束时的参数。
- 同一视频先对两个 seed 的指标取平均，再对视频做描述性 bootstrap；seed、帧和候选对不会增加独立视频数。

## 错误历史压力测试

对每个存在上一帧第二候选的状态，固定把“近期对象描述子”替换为上一帧响应排名第二的候选，并将其余上一帧候选作为其他对象。当前裁剪、框、速度、可信原型和响应保持不变。

这个替换对全部可用状态实施，不按当前 GT 或失效时刻触发；上一帧候选的 GT 标签只在评分完成后标识“替换对象确为负候选”的子组。它不是选择一个 GT 已知的最佳或最坏候选来输入模型。

描述子模型不读取这些可变历史特征，因此分数不变是**设计保证的输入隔离性质**，不是学到的历史鲁棒性，也不能证明错误裁剪后仍能跟踪。压力测试主要展示九维模型在相同候选与框条件下对历史改变的依赖。

## 固定开发筛查规则

只在 `full` 完成全部 48 个模型后进行以下筛查：

1. 困难正负候选对至少覆盖 6 个 fit 源组、100 个帧；不足时报告 `INSUFFICIENT_HARD_SUPPORT`。
2. 主模型困难候选对的视频宏平均准确率≥0.60。
3. 主模型相对重新训练的 `scalar_balanced`，困难准确率至少增加 0.05，视频配对描述性 95% 区间下界>0，且两个 seed 的平均差均>0。
4. 在原生候选 IoU<0.5、但保留集合存在正确候选的状态中，主模型 top-1 正确率≥0.20。
5. 在原生候选已正确的状态中，主模型保留正确输出的宏平均比例≥0.99。

前四项提供描述子信号证据，第五项是同状态保留正确输出的保护条件。信号满足但保护条件不满足时，报告 `SIGNAL_WITH_SAFETY_LOSS`。全部满足才是 `PASS_POOLED_DESCRIPTOR_SCREEN`；否则报告 `NO_DESCRIPTOR_GAIN_ESTABLISHED`。

学习投影相对固定随机投影的配对结果单独报告，用于检验投影训练本身的作用，不通过时不能声称学到的投影机制已有证据。

这些规则在开发数据已经被查看后设定，属于固定开发筛查，不是独立确认性检验。通过也只说明值得另设新源组闭环实验，尚无闭环 AUC、泛化、速度或压缩收益。

## 服务器执行

通过仓库同步新增文件，仍使用原环境。无需 GPU、原 checkpoint 或数据下载；原 fit 标注需要保留，以执行原缓存完整性审计。

```bash
export STAGE33_ROOT="$PWD/output/stage33/pooled_descriptor_oof_v1"

python tracking/test_stage33_descriptor_probe.py

python tracking/stage33_descriptor_probe.py \
  --input "${STAGE31_ROOT:?请设置原Stage31根目录}" \
  --mode audit \
  --output_dir "$STAGE33_ROOT/audit_v1"
```

审计状态为 `READY_DESCRIPTOR_PROBE` 后，执行一次工程 smoke：

```bash
python tracking/stage33_descriptor_probe.py \
  --input "${STAGE31_ROOT:?}" \
  --mode smoke \
  --output_dir "$STAGE33_ROOT/smoke_v1"
```

smoke 固定为第一个留出视频、一个 seed、三个模型各 12 次更新。`PASS_DESCRIPTOR_SMOKE` 只检查梯度、冻结投影、头更新和输入隔离，不评价方法收益。

完整实验：

```bash
python tracking/stage33_descriptor_probe.py \
  --input "${STAGE31_ROOT:?}" \
  --mode full \
  --output_dir "$STAGE33_ROOT/full_v1"
```

同一命令可重跑。已完成的小模型用 NPZ、完成回执和 SHA256 绑定后复用；中断时尚未完成的单个模型重新执行其固定预算。当前版本不是模型内部 update 级断点恢复。

如果在写入完成回执前中断，只有元数据属于当前计划的孤立 checkpoint 才会重训覆盖。已有完成回执但哈希不匹配时会拒绝，不能把改动后的参数当成已完成训练继续使用。

输出目录必须位于原 Stage-31 目录之外。不同 mode 使用不同目录，旧冻结文件保持原样。

## 结果文件

```bash
cat "$STAGE33_ROOT/audit_v1/stage33_support.csv"
cat "$STAGE33_ROOT/smoke_v1/stage33_decision.json"

cat "$STAGE33_ROOT/full_v1/stage33_decision.json"
cat "$STAGE33_ROOT/full_v1/stage33_paired.csv"
```

- `stage33_metric_summary.csv`：全部、困难、低 IoU、可纠正、原生正确及历史替换子组的指标。只读 `nominal` 结果判断预定主筛查。
- `stage33_seed_video_metrics.csv` 和 `stage33_video_metrics.csv`：seed/视频原始结果和 seed 平均后的视频结果，核对有效支持量。
- `stage33_integrity.csv`：投影是否改变、数据梯度、头更新、历史替换分数差。
- `stage33_training_history.csv`：固定训练预算中的损失和梯度，不用于挑选训练轮次。
- `checkpoints/`：各训练折的小模型及其完成回执。没有在全部 fit 上重训的可部署模型。
- `oof_predictions/`：留出视频上的评分及来源元数据；评分未更新原跟踪器状态。
- `plan.json`：原输入、代码、设计、划分和预算指纹。输入与代码在结束时复核。

没有产生新跟踪轨迹，不能把单帧 IoU 代理量或候选对准确率换算为策略 AUC。若未通过，只能收束这个固定投影/配对目标的可行性，不能宣称所有身份表征、完整 token、候选关联或强化学习均无效。
