# Stage-38：固定学生响应的 size／offset 闭环消融

Stage-37 的教师几何臂平均 AUC 最高，但相对学生的配对区间跨零、视频胜负为4／4，收益集中在少数案例。本轮只回答：该变化更依赖预测宽高 size、子格中心偏移 offset，还是两者配合。

使用原8个 OOF 视频、原4折、seed2110／2111 的冻结 supervised 学生。不训练、不换 checkpoint、不搜索层、阈值或损失权重，不增加 checking 视频。

## 四臂与复用

| 臂 | 响应图 | size | offset | 完整轨迹来源 |
|---|---|---|---|---|
| `student` | 学生 | 学生 | 学生 | 直接复用 Stage-37 |
| `teacher_size` | 学生 | SGLA | 学生 | 本轮新增 |
| `teacher_offset` | 学生 | 学生 | SGLA | 本轮新增 |
| `teacher_geometry` | 学生 | SGLA | SGLA | 直接复用 Stage-37 |

另外读取原 `sgla` 轨迹作为参照，不重新推理。SGLA 保留原生层选择，不固定为 L9。

完整实验只新增 **8视频×2seed×2臂＝32条轨迹**；直接读取32条旧端点轨迹和8条共同 SGLA 参照。既有预测不复制、不改写。

“固定学生响应”指四臂始终使用**当前自身裁剪上计算的学生响应**。不同轨迹的裁剪可能不同，响应数值不要求相等。每条轨迹都从第一帧开始，混合输出作为自己的下一帧父框，不重置到另一臂、旧缓存或 GT。

窗口、argmax、坐标映射和边界裁剪沿用原解码器。size 与 offset 在学生响应选中的同一格取值。offset 不是完整中心坐标，响应峰仍决定粗网格位置。

## 服务器命令

通过 Git 同步新增文件后，在仓库根目录依次执行：

```bash
export STAGE37_ROOT="${STAGE37_ROOT:-$PWD/output/stage37/frozen_crossed_closed_loop_v1}"
export STAGE38_ROOT="$PWD/output/stage38/fixed_response_geometry_v1"

python tracking/test_stage38_geometry.py

python tracking/stage38_geometry_experiment.py \
  --input "$STAGE37_ROOT" --mode prepare \
  --output_dir "$STAGE38_ROOT"

python tracking/stage38_geometry_experiment.py \
  --input "$STAGE38_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

python tracking/stage38_geometry_experiment.py \
  --input "$STAGE38_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage38_geometry_summary.py --input "$STAGE38_ROOT"
```

权重查找顺序为 `--checkpoint`、已有的 `CHECKPOINT` 环境变量、仓库 tracker 配置。核对的是原始 SGLA 权重，学生按原折和 seed 自动恢复。需要显式指定时，在两条 GPU 命令中加入 `--checkpoint /你的原始权重路径.pth.tar`。

prepare、status、summary 不加载网络或 CUDA；旧权重、图片、标注和结果仍需存在，以验证来源。测试使用 CPU 合成小模型，不调用 GPU。

## smoke 与冻结检查

smoke 固定使用原 fit 列表第一条视频的前64个跟踪帧，运行四臂、两个seed，共8条短轨迹。两个端点会重新执行，用于确认新实现能复现原 Stage-37 的学生／全部教师几何输出；全量阶段只运行两个新增臂。

要求：

- 端点框和父框最大差≤`0.001`像素，原评价 IoU 差≤`1e-12`。
- 原始密集解码与执行框最大差≤`0.001`像素。
- 每帧两个模型各一次 head 调用，读取同一个搜索张量；模型与搜索输入不被修改。
- 所有父框等于该臂自己的上一帧输出；未来标注不参与裁剪、选峰或状态更新。
- 运行时、8个学生权重摘要、SGLA摘要和窗口与原 Stage-37 一致。

状态 `PASS_FROZEN_SIZE_OFFSET_SMOKE` 只表示执行和端点复现通过，不表示新方法或跟踪收益通过。

## 中断与进度

重复同一条 smoke／evaluate 命令即可。已提交的完整轨迹审计后复用；当前未完成轨迹从第一帧重新运行。NPZ与收据之间中断时，重跑会先检查数据再补发收据。

```bash
python tracking/stage38_geometry_experiment.py \
  --input "$STAGE38_ROOT" --mode status
```

全量进度应最终为 `32/32` 新轨迹，旧对照不计入新推理数。可以用 `evaluate --sequence <原8条之一>` 分次运行；最终汇总仍要求所有32条齐全。不要在同一个输出目录并发启动多个 collector。

## 先查看三行重点表

完成后贴回：

```bash
cat "$STAGE38_ROOT/analysis/stage38_policy_summary.csv"
cat "$STAGE38_ROOT/analysis/stage38_focus.csv"
cat "$STAGE38_ROOT/analysis/stage38_factor_summary.csv"
```

`focus.csv` 只保留 size-only、offset-only、全部几何三行：相对学生的 AUC 差、区间、两seed均值、低IoU变化、可靠片段变化、新增失跟视频数，以及单视频收益占比和留一视频均值。

原稳定视频按基线 seed／视频判断，新增失跟再合并为不同视频数；不是将两个seed当作额外视频。完整逐seed记录仍保留。

可靠片段每视频至少需10帧，且两个seed都需有支持；否则相应差异留空，并报告 `reliable_supported_videos`。缺失不代表零损伤或通过稳定性检查。

## 预先固定的分析

两个并列主要比较为：

1. `teacher_size − student` 的完整视频宏平均 AUC。
2. `teacher_offset − student` 的完整视频宏平均 AUC。

先在每个视频内平均两个seed，再做5000次视频配对 bootstrap，seed2117。主要比较同时报告普通95%区间和按两项比较的 Bonferroni 口径补充区间（百分位1.25%／98.75%）；它们仍是固定已训练模型条件下的描述结果。

其余比较包括全部几何、两种单分量替换之差、全部几何减去各单分量，以及与原 SGLA 的配对差。可靠片段由基线 IoU≥0.5 离线定义，不能变成在线触发规则。

令四条完整轨迹的 AUC 分别为 B（学生）、S（仅换size）、O（仅换offset）、G（两者都换），另报告：

- size 的两顺序平均增量：`((S−B)+(G−O))/2`。
- offset 的两顺序平均增量：`((O−B)+(G−S))/2`。
- 交互：`G−S−O+B`。

前两项严格相加为 `G−B`。这是包含裁剪反馈的轨迹级干预记账，不是唯一的物理因果分配，也不代表两个模块已分别训练。

除低 IoU 比例和失跟事件数，还报告最长失跟事件占视频长度的比例，避免把“事件变少但持续更久”误判为恢复改善。事件定义沿用原3帧确认／3帧恢复规则；最长事件可以包含事件期间的中间 IoU 帧，不能等同于连续全低 IoU 段。

AUC 用0–100点，比例用0–1。留一视频结果只检查收益集中程度，不用于删视频或重训。

## 输出与结论边界

| 文件 | 用途 |
|---|---|
| `manifest.json` | 原来源、八视频、折、模型、四臂和代码指纹 |
| `smoke/stage38_smoke.json` | 8条短轨迹与4条端点复现记录 |
| `full/traces/*.npz`、`*.json` | 32条新增完整轨迹、原始两来源图和提交收据 |
| `analysis/stage38_policy_summary.csv` | 四臂＋SGLA参照宏平均结果 |
| `analysis/stage38_focus.csv` | 三行重点结果 |
| `analysis/stage38_factor_summary.csv` | 两顺序平均增量及交互 |
| `analysis/stage38_paired.csv` | 全部预定比较 |
| `analysis/stage38_seed_video_summary.csv`、`stage38_video_summary.csv` | 逐seed与seed平均后的逐视频结果 |
| `analysis/stage38_stability.csv` | 新增失跟、可靠片段与事件时长变化 |
| `analysis/stage38_video_influence.csv` | 逐视频贡献与留一均值 |
| `analysis/stage38_source_controls.csv` | 32条旧端点和8条共同参照的实际来源路径／哈希 |
| `analysis/stage38_diagnostic.json` | 完整性、来源、输出指纹与记账闭合误差 |

状态 `COMPLETE_FROZEN_SIZE_OFFSET_DIAGNOSTIC` 仅表示实验完整，不自动选择获胜配置或启动训练。

这些8个开发组以及原几何结果已经参与设计，OOF训练集相互重叠，不能作为独立外部验证。收益如果只集中在个别视频，或同时损伤可靠片段，应保留这一边界。教师在线参与的本轮结果不能证明独立紧凑学生能学到收益，也不能用于报告压缩或速度提升。

本轮仅新增 Stage-38 文件。测试实际运行 CPU 小模型的 Stage-37 完整来源链和 Stage-38 新轨迹，覆盖通道独立性、自己的裁剪父框、GT隔离、端点复现、来源只读、32条新轨迹覆盖、缺失结果拒绝汇总、复用和篡改拒绝。
