# Stage-35B：普通监督学生的事后只读审计

Stage-35 主方法未建立收益，可靠帧保留也失败。与此同时，普通监督六层学生的开发 OOF 平均 AUC=68.6324，高于 Fast=65.2745 和 SGLA=67.3924。

这一步核实该信号是否跨 seed、跨视频成立，以及是否通过牺牲原本可靠的帧获得收益。**不训练、不生成新预测、不修改 Stage-35 门槛或原决策。** 看到结果后关注普通监督属于事后分析，不能将它重新称为预先指定的获胜主方法。

## 执行

通过 Git 同步三个新增文件，仍在现有仓库根目录、现有 Python 环境中运行；无需 GPU、原始 checkpoint 参数或新数据下载。

```bash
export STAGE35_ROOT="${STAGE35_ROOT:-$PWD/output/stage35/preserve_correct_oof_v1}"

python tracking/test_stage35_supervised_audit.py

python tracking/stage35_supervised_audit.py --input "${STAGE35_ROOT:?}"

export STAGE35B_DIR="$STAGE35_ROOT/analysis/supervised_baseline_audit_v1"
cat "$STAGE35B_DIR/stage35b_audit.json"
cat "$STAGE35B_DIR/stage35b_paired.csv"
cat "$STAGE35B_DIR/stage35b_seed_summary.csv"
cat "$STAGE35B_DIR/stage35b_video_influence.csv"
cat "$STAGE35B_DIR/stage35b_stability.csv"
```

成功状态仅为 `COMPLETE_POSTHOC_SUPERVISED_AUDIT`，表示审计完整，**不表示实验通过**。原 `NO_PRESERVE_CORRECT_GAIN_ESTABLISHED` 保持不变。没有训练/推理任务可续跑；中断后重跑同一条审计命令，完整且相同的结果可以复用。

## 输入完整性

脚本核对冻结 Stage-35 manifest、原 decision、OOF run、两个训练完成报告、全部 40 个最终 checkpoint 的 SHA256，以及全部 96 个原轨迹缓存的 SHA256、元数据和折归属。

重新核对完整视频的帧索引、预测父框连续性、IoU、GT 标注哈希和每帧一次主定位头记录。随后从缓存重新计算指标，与原 `stage35_policy_summary.csv` 的宏平均 AUC、低 IoU 比例和计时数值对齐。来源缺失、混入、修改或缓存不完整都会停止审计。

只读取这 8 个已消费 fit 源组的标注与原缓存，不读取旧 checking 图像/标注、不打开额外数据。checkpoint 仅用于字节哈希核对，不导入模型进行前向。

所有新输出都位于 `analysis/supervised_baseline_audit_v1/`；原脚本、训练目录、manifest、预测缓存和原分析文件不会被改写。已有新审计报告若与当前来源或审计代码不一致，会拒绝覆盖。

## 输出和含义

- `stage35b_paired.csv`：普通监督相对 Fast、SGLA 及其他四个训练组的 AUC 配对区间；相对两条基线的低 IoU 帧占比和失败事件数差异；以及基线定义可靠/失败帧队列的条件 AUC。
- `stage35b_seed_summary.csv`：两个 seed 分别相对 Fast、SGLA 的视频宏平均差异和 W/T/L；不得挑选较好的 seed。
- `stage35b_video_influence.csv`：每个视频的 seed 平均 AUC 差异、对总宏平均的贡献、正收益份额、两个 seed 的单视频差异，以及去掉该视频后的均值变化。
- `stage35b_stability.csv`：每条基线原本稳定的视频中，是否有任一普通监督 seed 引入持续失跟；可靠帧上的均值差异和最差 seed 差异。首个失跟帧是跟踪轨迹中的编号，首帧初始化不计入。
- `stage35b_seed_video_summary.csv`、`stage35b_video_summary.csv`：逐 seed/视频及先在视频内平均 seeds 后的完整可核对指标。
- `stage35b_audit.json`：两条基线的主要配对结果、正向 seed 数、收益集中程度、删一视频均值范围、稳定视频变化与完整来源绑定。

差异统一为 `supervised − reference`：AUC 正值有利；低 IoU 占比和失败数负值有利。CSV 中 W/T/L 按各指标的有利方向统计，并记录 `favorable_direction`；低 IoU 占比使用 0–1 单位，乘 100 才是百分点。AUC 已是 0–100 单位。

先在同一视频内平均 seeds，再对 8 个源组做 5000 次配对 bootstrap，固定分析 seed2113。两个训练 seeds、帧数和折数都不能扩充独立视频数；折的训练集相互重叠，区间仅描述给定拟合模型下的视频不确定性，不包含重新训练的不确定性。

可靠帧以 Fast 或 SGLA 原轨迹 IoU≥0.5 固定；SGLA 失败帧以其 IoU<0.5 固定。各视频的条件队列至少 10 帧才计分；支持不足显示空值并报告实际支持视频数，不自动视为通过。条件队列是结果分层诊断，不是线上触发规则。

正收益份额的分母为所有正向视频差异之和，而非可能正负抵消的净收益。去掉单个视频后的均值变化只用于检查影响程度，**不允许据此删除视频或重新训练选折**。

## 结果如何使用

如果普通监督的优势主要来自一个视频、seed 分歧明显或引入可靠帧损伤，则保留为尚未稳定的开发信号。如果优势跨视频、跨 seed 较一致，再把它冻结成更强的压缩基线，另行预定新的方法与独立数据验证。

本次没有新增采用门槛、确认性显著性判定或方法成功结论。普通监督微调本身也不是创新证明；带同步的原计时不构成部署加速或能耗结论。
