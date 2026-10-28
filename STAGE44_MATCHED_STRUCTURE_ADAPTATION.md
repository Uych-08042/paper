# Stage-44：压缩结构 × 相同适配预算的基础对照

Stage-43 的历史恢复没有通过预定门槛。本轮回到一个先决问题：**在已固定的普通监督适配预算下，删除原生 SGLA 候选结构后，是否仍存在稳定的完整闭环性能缺口？**

此前只适配六层模型再与未适配 SGLA 比较，混合了结构和适配两种变化。Stage-44 补全保留结构的适配对照。它是方法设计前的基础验证，不是新压缩算法。

## 四个固定角点

| 方法 | 结构 | 适配 | 来源 |
|---|---|---|---|
| six_initial | 删除 L7–L12 与选择器，六层单路径 | 无 | 冻结原始 Fast |
| retained_initial | 原生 SGLA，六个前缀块＋一个独立候选块 | 无 | 冻结原始 SGLA |
| six_adapted | 六层单路径 | 仅 L5–L6 | 原样复用 Stage-40 clean 最终 checkpoint |
| retained_adapted | 原生 SGLA 候选结构 | 仅 L5–L6 | 本轮新增训练 |

不固定为 L9，不训练新的选择器，不串行执行十二层。保留结构保存全部候选和原选择器，但每次只执行一个候选。选择器权重冻结，L5–L6 适配后其输入可以改变，硬选层结果也可能改变。比较对象是整个“独立候选＋原生选择”结构，不能把结果简化为纯深度或参数数量的因果效果。

## 训练与数据控制

- 同一批已消耗的 8 个开发视频，原 4 折 OOF，每折 6 fit / 2 held-out，种子 2110、2111；不访问原 checking/tune/validation 划分。
- 新增 4 折 × 2 种子＝8 个 retained 模型；复用 8 个六层 clean 模型，不重训、不挑选其 checkpoint。
- 两个结构的可训练参数完全相同，仅 L5–L6。其余前缀、候选、选择器、归一化及预测头权重冻结，均处于 eval 模式。
- 新 retained 模型从原 SGLA 权重初始化；共享可训练前缀与六层模型的初始参数逐项相等。
- 完全沿用 Stage-40 clean 的样本抽样、已冻结 SGLA 父框、实际取整裁剪和 GT 坐标映射。每个样本两张相同 clean view，普通监督损失取均值。
- 每模型 256 个 AdamW 更新，batch_size=2，LR=1e-4，weight_decay=1e-4，梯度裁剪=1。运行时从冻结 Stage-40 设计核对这些配置及更新数。
- 不添加 KD、对比损失、恢复模块或新的训练状态滚动。
- 相同更新数、图像/裁剪、可训练参数及头调用数；保留结构每个裁剪执行 7 个块，六层模型执行 6 个块，因此训练 FLOPs 不相同，这是结构处理的一部分。

所有 8 个新模型都冻结后才允许完整 OOF 推理。新增 16 条完整视频轨迹，各自使用自身预测决定下一帧裁剪。其余三个角点从已验证的旧缓存读取并绑定哈希；初始角点在两个种子下相同，不是重复的独立视频。

## 预定判读

主指标为 `retained_adapted − six_adapted` 的视频宏平均 AUC。先平均每个视频的两个种子，再做 5000 次视频 bootstrap，种子 2123。

首先检查保留结构的适配对照是否有效：

- retained_adapted 相对 retained_initial 的平均 AUC 差值不低于 -0.5 点，95% CI 下界不低于 -1 点。
- 宏平均低 IoU(<0.2) 比例不增加。
- 原 retained_initial 无持续低 IoU 事件的视频/种子，不引入新事件。

不满足时输出 **MATCHED_RETAINED_CONTROL_DEGRADED**。这表示当前保留结构对照未通过保护条件，不能把结果当作六层充分或压缩无损的证明。

对照有效后，只有同时满足下列条件，才输出 **DEVELOPMENT_COMPRESSION_GAP_SUPPORTED**：

- 适配后结构缺口至少 +0.5 AUC 点，95% CI 下界>0。
- 至少 5/8 视频为正，去掉任一视频后的平均仍为正，两个种子各自平均差值均为正。
- retained_adapted 的宏平均低 IoU 比例不高于 six_adapted。

否则输出 **NO_MATCHED_COMPRESSION_GAP_ESTABLISHED**。本轮不使用“方法 PASS”。初始结构差值、两个结构的适配效应、结构×适配交互及逐视频影响均报告，不能在观察结果后改选主比较。

即便通过，也只建立该数据和适配预算下的开发线索。不能外推到充分训练后的模型最优能力，不能证明无人机特有机制、部署加速或新的压缩方法成立。

## 服务器指令

通过仓库同步新文件，无需脚本包。使用包含 manifest.json 的 Stage-43 根目录，不是 analysis 子目录。

```bash
cd ~/workspace/paper
export STAGE43_ROOT="$PWD/output/stage43/online_history_recovery_v1"
export STAGE44_ROOT="$PWD/output/stage44/matched_structure_adaptation_v1"

python tracking/test_stage44_matched_structure.py

python tracking/stage44_matched_experiment.py \
  --input "$STAGE43_ROOT" --mode prepare \
  --output_dir "$STAGE44_ROOT"

python tracking/stage44_matched_experiment.py \
  --input "$STAGE44_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

cat "$STAGE44_ROOT/smoke/stage44_smoke.json"
```

smoke 在第一折的 fit 视频上检查短轨迹、梯度、初始化与原生 SGLA 输出复现，不对 held-out 视频作模型选择。`PASS_MATCHED_STRUCTURE_SMOKE` 不代表结构缺口成立。

通过后固定两个种子依次运行：

```bash
for seed in 2110 2111; do
  python tracking/stage44_matched_experiment.py \
    --input "$STAGE44_ROOT" --mode train \
    --seed "$seed" --parameter deit_distilled --gpu 0 || break
done

python tracking/stage44_matched_experiment.py \
  --input "$STAGE44_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage44_matched_summary.py --input "$STAGE44_ROOT"

cat "$STAGE44_ROOT/analysis/stage44_decision.json"
cat "$STAGE44_ROOT/analysis/stage44_policy_summary.csv"
cat "$STAGE44_ROOT/analysis/stage44_paired.csv"
cat "$STAGE44_ROOT/analysis/stage44_video_influence.csv"
```

两个种子所有模型未完成时，evaluate 会拒绝运行。权重路径解析顺序为 `--checkpoint`、环境变量 CHECKPOINT、原 tracker 配置，并核对冻结原权重哈希。失效的旧 CHECKPOINT 环境变量需要纠正或 unset，不能改用学生权重。

## 中断与结果保护

```bash
python tracking/stage44_matched_experiment.py --input "$STAGE44_ROOT" --mode status
```

训练每 64 更新及最终更新原子保存参数、AdamW 状态与收据；重跑同一训练指令从最近提交点继续，未提交更新重算。完整推理以视频为提交单位，已完成视频核验复用，未完成视频从首帧开始。可使用 `evaluate --sequence 视频名` 分视频运行，汇总仍必须等待全部 16 条新轨迹完成。

旧 Stage-40/43、所有源模型、结果和表格均绑定哈希。主 decision 保持紧凑，完整账本在 `stage44_evidence.json`。不修改旧结果、不重新选择六层 checkpoint；不在此次结果上搜索层、学习率、训练步数或折划分。

## 方法边界

[SGLA 原文，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_Similarity-Guided_Layer-Adaptive_Vision_Transformer_for_UAV_Tracking_CVPR_2025_paper.html) 的核心是保留具有互补表示的候选层以权衡跟踪精度与效率。本轮仅在当前实现和受控适配预算下检验进一步删除该结构后的性能缺口，不复现论文的完整训练规模，也不把普通监督适配作为新方法。
