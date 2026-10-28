# Stage-41：增强收益与轨迹损伤的缓存审计

Stage-40 的完整视频 AUC 差为 −0.6179 点，区间跨零，两个 seed 的均值方向不同；固定压力损失却减少0.00826 IoU。下一步需要区分压力指标的差值变化、绝对精度变化和完整轨迹的收益／损伤发生位置。

本轮只读现有 Stage-40 结果，使用全部8视频、两seed、32条完整轨迹和96个压力包。不训练、不调用网络、不解码图片，不改变原结果，也不消耗新的 checking 组。它是对已消耗结果的事后定位，不设置新方法通过门槛。

## 核心问题一：压力损失下降来自哪一部分

令同一个模型无扰动分支的后续平均 IoU 为 C，有扰动分支为 P，压力损失为 `D=C−P`。模型从 clean 换为 augmented 后，有精确等式：

`D_aug−D_clean = (C_aug−C_clean) − (P_aug−P_clean)`

因此，压力损失减少的贡献可以写成：

`−ΔD = −ΔC + ΔP`

例如 clean 模型的 C=0.8、P=0.7，增强模型的 C=0.6、P=0.65。压力损失从0.1降到−0.05，但有扰动绝对 IoU 也下降0.05。不能把差值变好等同于绝对耐扰能力更强。

审计在每个视频／seed／锚点上记录 C、P、ΔC、ΔP、ΔD，再先平均视频内锚点和seed。第1帧仍排除，沿用原第2至20帧。数值闭合误差必须≤1e-10。

`reduced_loss_without_absolute_perturbed_gain` 表示 ΔD<0 但 ΔP≤0，是一个离线记账提示，不是在线触发器。若 ΔC<0、ΔP>0，应同时保留两种贡献，不能把全部压力改善归到其中一种。

## 核心问题二：完整视频 AUC 差发生在哪些帧

原评价是21个阈值的严格 `IoU>threshold` 成功率均值。逐帧定义同口径分数，再将每帧 augmented−clean 的分数差分配到五个互斥、穷尽的队列：

| 队列 | clean IoU | augmented IoU |
|---|---|---|
| both_reliable | ≥0.5 | ≥0.5 |
| rescued_low_to_reliable | <0.2 | ≥0.5 |
| harmed_reliable_to_low | ≥0.5 | <0.2 |
| both_low | <0.2 | <0.2 |
| other | 以上之外 | 以上之外 |

每组贡献为该组逐帧分数差之和除以**完整视频帧数**，单位为 AUC 点。五组贡献严格相加等于完整视频 AUC 差。另列条件均值，避免把大贡献误解为较差的条件表现；空队列的条件均值留空，不填零。

这些队列用两条轨迹的 GT 评价结果定义，属于结果分层，不是可部署判断规则。双方都可靠时也可能有持续的定位精度差；双方都低 IoU 时的贡献也不强制为零。

## 时间定位与自己的裁剪历史

每个视频、每个seed都记录两组全部失跟事件，沿用原连续3帧 IoU<0.2确认、连续3帧 IoU≥0.5恢复的规则。事件时长不等同于连续低 IoU 帧数，因此两者分别记录。

另外构造下列观察锚点，审计前10帧、锚点帧、后30帧：

- clean 的第一次失跟与 augmented 的第一次失跟（存在时）。
- 第一次连续3帧 IoU 差≥0.2的增强收益段，或≤−0.2的损伤段（存在时）。
- 原视频1／4、1／2、3／4时点作为固定时间参照。

所有视频都处理，不只查看 `uav0000089` 和 `uav0000348`。完成后才按视频平均 AUC 差列出最大收益和最大损伤案例；这两个案例使用了已观察结果，不能称为预选或独立验证。

几何量由每条缓存自己的 `previous` 和 `boxes` 计算：裁剪对 GT 的覆盖、输入中目标边长、原图目标边长、父框／输出中心误差、父框尺度比、输出宽高比、相邻帧中心误差向量的变化。中心误差按当前 GT 几何平均边长归一化。时间点与源缓存一致：初始化图索引0，跟踪 `frame_id=1` 对应第二张图，不直接等于文件名中的帧号。

GT 覆盖不表示可见性。中心误差跳变不表示响应峰必然换到某个目标。整视频缓存没有响应图，本轮无法拆分响应／size／offset，也不能做同裁剪反事实；时间先后只用于描述，不直接等同于裁剪反馈的因果归因。

## 运行

通过 Git 同步三个新增文件，在服务器仓库根目录执行：

```bash
cd ~/workspace/paper
export STAGE40_ROOT="$PWD/output/stage40/matched_crop_augmentation_v1"
export STAGE41_ROOT="$PWD/output/stage41/cached_crop_transfer_audit_v1"

python tracking/test_stage41_transfer_audit.py

python tracking/stage41_transfer_audit.py \
  --input "$STAGE40_ROOT" \
  --output_dir "$STAGE41_ROOT"
```

审计命令只需要 Python 和 NumPy，不需要 CUDA、模型加载或额外数据下载。源权重、图片、轨迹、收据、决策和证据文件仍须存在，用于来源哈希核对；不会进行图像解码。测试中的 CPU PyTorch 只用于生成合成的完整 Stage-40 来源，实际审计不调用训练或推理。

执行前验证 Stage-40 完整报告、证据文件、模型绑定、32条完整轨迹和96个压力包。两项主要均值和两组绝对完整视频 AUC 必须复现原报告。不会调用原汇总函数或重写旧文件。

中断后重跑相同命令；未完成审计在相同冻结配置下重新计算，已完成审计校验后直接复用。输出目录必须独立于 Stage-40 来源树。代码／配置／来源改变时拒绝覆盖原审计结果，需使用新输出目录。

## 结果文件

优先提供以下三个文件，均可作为附件：

```text
stage41_diagnostic.json
stage41_video_summary.csv
stage41_anchor_summary.csv
```

| 文件 | 内容 |
|---|---|
| stage41_diagnostic.json | 简短完成报告、最大收益／损伤案例、seed一致性、压力记账提示与探索性区间 |
| stage41_video_summary.csv | 8行；两seed平均后的完整视频、压力绝对精度、损失差与AUC帧队列贡献 |
| stage41_seed_video_summary.csv | 16行；每视频每seed的同口径结果 |
| stage41_auc_partition.csv | 五个互斥队列的帧数、完整视频AUC贡献和条件差 |
| stage41_event_summary.csv | 两组全部失跟事件、持续时间与起点几何；无事件时保留表头 |
| stage41_anchor_summary.csv | 每个观察锚点／组别的前、当帧、后三段几何和IoU概况 |
| stage41_anchor_timeline.csv | 观察窗口内的逐帧几何和IoU差 |
| stage41_stress_accounting.csv | 每视频／seed／锚点的压力损失精确拆分 |
| stage41_paired.csv | 视频内平均seed后，8个视频的探索性配对区间 |
| stage41_evidence.json | 独立保存完整来源哈希，避免简短诊断文件膨胀 |
| plan.json | 冻结本轮配置、代码和来源身份 |

状态 `COMPLETE_CACHED_TRANSFER_AUDIT` 仅表示数据与记账完整。源 Stage-40 的失败保持不变，本轮不产生方法 PASS，也不自动选择下一种模块、阈值或训练权重。

bootstrap 为5000次、seed2120，先平均视频内seed与压力锚点，再以视频为单位抽样。全部区间都是事后探索、条件于已有重叠 OOF 拟合；窗口、帧或seed不能充当更多独立视频。本轮也不建立压缩特有失稳、无人机场景特异性、速度或外部泛化结论。
