# Stage-37：冻结响应／几何四臂闭环实验

Stage-36C 在 0089 的两个已消耗失跟起点观察到：窗口加权响应峰从 136 跳到 69，换格项构成约 94%／97% 的中心误差变化投影；正确目标仍在固定 top-5 中，SGLA 在同一学生裁剪上仍能定位。单帧替换尚不能说明这种差异能否沿真实轨迹持续。

本轮固定使用原 Stage-35 的 **8个 OOF 视频、4折、seed2110／2111 的已训练 supervised 学生**。不再训练或选择 checkpoint，不增加视频，不使用原 checking 数据。

## 四臂设计

| 臂 | score_map | offset_map＋size_map |
|---|---|---|
| `student` | 学生 | 学生 |
| `teacher_response` | 原 SGLA | 学生 |
| `teacher_geometry` | 学生 | 原 SGLA |
| `sgla` | 原 SGLA | 原 SGLA |

窗口、argmax、坐标映射和边界裁剪均沿用原解码器。SGLA 仍执行自身原生层选择，不固定成 L9。

每条轨迹从第一帧初始化，之后仅用**该臂上一帧自己的预测**创建裁剪；两个模型读取该帧同一个搜索张量。混合后的预测直接成为下一帧父框，不在下一帧重置到原记录、另一臂或 GT。

为保留同状态来源审计，**四臂每帧均运行学生和 SGLA 两个冻结模型**，然后按表选择通道。这样记录的时间包含两模型与完整性检查，不能用于声称单模型速度、压缩率或部署收益。教师参与只是解码机制干预，最终紧凑模型能否学到该能力仍待验证。

每个视频只收集一次共同的 `sgla` 轨迹，另收集两个 seed 各三条学生相关轨迹：共 **7条／视频，56条完整轨迹**。汇总里的教师配对复用同一条轨迹，不增加独立视频数。

## 服务器命令

通过 Git 同步新增文件，在仓库根目录依次执行：

```bash
export STAGE36_ROOT="${STAGE36_ROOT:-$PWD/output/stage36/same_crop_mechanism_v1}"
export STAGE37_ROOT="$PWD/output/stage37/frozen_crossed_closed_loop_v1"

python tracking/test_stage37_crossed.py

python tracking/stage37_crossed_experiment.py \
  --input "$STAGE36_ROOT" --mode prepare \
  --output_dir "$STAGE37_ROOT"

python tracking/stage37_crossed_experiment.py \
  --input "$STAGE37_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

python tracking/stage37_crossed_experiment.py \
  --input "$STAGE37_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage37_crossed_summary.py --input "$STAGE37_ROOT"
```

GPU 命令优先使用 `--checkpoint`，其次使用已有的 `CHECKPOINT` 环境变量，均未提供时使用仓库 tracker 配置，并核对原指纹。如果需要指定原权重，在 smoke 和 evaluate 两条命令中均加入：

```bash
--checkpoint /你的原始SGLA权重路径.pth.tar
```

因此不依赖必须预设的 `CHECKPOINT` 环境变量。不要传入训练后的学生文件代替原始权重；学生会自动按原 OOF 折和 seed 从冻结训练记录恢复。

prepare、status 和 summary 不初始化网络或 CUDA；它们仍需保留旧权重、图片、标注和缓存，以验证来源哈希。测试使用 CPU 小模型，不调用服务器 GPU。

## smoke 与原生复现

smoke 固定使用原 fit 列表第一条视频的前64个跟踪帧和全部7条轨迹，不选择更容易通过的序列。

必须同时满足：

- 每条轨迹前后父框连续，来源模型冻结，两个模型未修改共享搜索输入。
- 每个模型每帧一次 head 调用；交叉密集解码与原生解码最大框差不超过 `0.001` 像素。
- 原生 `student` 两 seed 与 `sgla` 逐帧复现 Stage-35 原记录：框和父框最大差不超过 `0.001` 像素，原评价 IoU 差不超过 `1e-12`。
- 三张输出图按既定来源组成，沿用原窗口；未来 GT 不参与裁剪、选峰或父框更新。

成功状态为 `PASS_FROZEN_CROSSED_SMOKE`，仅代表执行完整性。全量阶段还会再次验证三个原生控制在整个视频上复现旧结果。任何复现失败都应先查实现，不放宽容差来生成结果。

## 中断、复用与进度

重新执行同一条 smoke 或 evaluate 命令即可。以**一条完整视频轨迹**为提交单位：已经完成并通过哈希审计的轨迹复用；当前未完成的那一条从第一帧重新运行，不是逐帧 checkpoint 续接。

每条 NPZ 另有密封 JSON 收据。如果写完 NPZ、尚未发布收据时中断，重跑会检查其连续性、解码和来源，再补发收据。已经提交的文件被修改时拒绝复用。

```bash
python tracking/stage37_crossed_experiment.py \
  --input "$STAGE37_ROOT" --mode status
```

可以用 `evaluate --sequence <原8条之一>` 分次收集，仍固定相同计划。完整汇总要求全部56条轨迹，不对未收齐的视频子集发布收益比较。不要在一个输出目录同时启动多个 collector；原有写锁会拒绝并发写入。

## 汇总与解读

主要比较预先固定为 **teacher_response − student 的完整视频宏平均 AUC**；同时报告几何替换、响应与几何替换之差、响应替换与 SGLA 之差，以及响应×几何交互。

所有比较先在同一视频内平均两个 seed，再对8个视频进行配对 bootstrap：5000次，seed2116。AUC 用0–100点，低 IoU 比例用0–1。两个 seed、帧和重复教师行均不是额外独立视频。

保留以下稳定性信息：

- 相对同 seed 学生及原 SGLA 的完整视频 AUC、低 IoU 比例和失跟事件数。
- 基线 IoU≥0.5 的可靠片段上，交叉臂的 AUC 变化；这些 GT 分层仅用于离线评价。
- 原来没有失跟事件的视频，交叉后是否引入失跟；逐 seed、逐视频保留。
- 主要收益的逐视频贡献与留一视频均值，检查收益是否由单个案例主导。留一结果不用于丢弃视频或重训。

首先贴回：

```bash
cat "$STAGE37_ROOT/analysis/stage37_policy_summary.csv"
cat "$STAGE37_ROOT/analysis/stage37_paired.csv"
cat "$STAGE37_ROOT/analysis/stage37_stability.csv"
```

输出文件：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 原8视频、OOF折、模型来源、四臂、规则及代码指纹 |
| `smoke/stage37_smoke.json` | 前64帧执行审计 |
| `full/traces/*.npz`、`*.json` | 全视频轨迹、同状态原始三张图和提交收据 |
| `analysis/stage37_policy_summary.csv` | 四臂整体表现 |
| `analysis/stage37_seed_video_summary.csv` | seed／视频／臂结果；教师配对复用 |
| `analysis/stage37_video_summary.csv` | seed平均后的逐视频结果 |
| `analysis/stage37_paired.csv` | 完整视频配对差、可靠片段差及交互 |
| `analysis/stage37_stability.csv` | 新增失跟、可靠片段损伤与低 IoU 变化 |
| `analysis/stage37_video_influence.csv` | 主要差异的逐视频贡献 |
| `analysis/stage37_native_parity.csv` | 24条原生控制与旧预测的复现差异 |
| `analysis/stage37_diagnostic.json` | 完整性、来源与输出指纹 |

`COMPLETE_FROZEN_CROSSED_DIAGNOSTIC` 只表示实验完整，不是新方法通过。这里不自动启动后续训练。

如果响应替换在完整轨迹上稳定获益，且可靠片段和原稳定视频未明显受损，才有理由研究如何把这种空间判别能力保留到独立运行的紧凑学生。若替换只救回个别案例、带来新的失跟或受几何强交互限制，应保留这些边界。

这8个开发组已经多次参与方法构思；OOF训练集也相互重叠。区间是固定已训练模型条件下的描述结果，未计入重训练不确定性或多重比较调整，不能作为外部泛化证明。几何交换是解码级干预，不能把效果唯一归给已独立训练的某个模块。

## 测试与冻结

新增文件独立于 Stage-35／36 源代码和输出树。来源、代码、图片、旧缓存或 checkpoint 改变时拒绝静默继续；新结果绑定自身收据和表格哈希。

测试覆盖四个通道端点的隔离、按视频而非 seed 计算区间，以及实际 PyTorch CPU 小模型的8视频完整流程：原生预测一致、自己的父框连续、未来标签替换不改变预测、smoke门槛、部分结果拒绝汇总、完整复用、收据中断恢复和模型／结果篡改拒绝。CPU合成数据测试不是服务器真实追踪结果。
