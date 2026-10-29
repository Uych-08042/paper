# Stage-46B：搜索偏移的缓存时序审计

日期：2026-10-29。状态：代码与本地 CPU 夹具验证完成，实际服务器缓存待审计。

## 1. 为什么做这一轮

用户提供的 Stage-46 结果为 `REVIEW_CAMERA_SEARCH_PILOT`：原生 SGLA AUC 63.4512，普通速度63.6449，仅相机补偿64.8664，相机加速度60.8608。仅相机补偿的平均收益主要由三个案例决定：

| 视频 | 仅相机补偿相对原生 AUC | 本轮要解释的现象 |
| --- | ---: | --- |
| `uav0000176_00000_s` | +48.8111 | 普通速度也改善，相机加速度却没有保留收益 |
| `uav0000204_00000_s` | +18.5254 | 仅相机补偿消除了低 IoU 帧，普通速度没有做到 |
| `uav0000304_00253_s` | −50.1173 | 原生稳定，补偿和速度均导致持续失败 |

其余9个视频的仅相机补偿平均增益约−0.0262点。所有帧均通过背景拟合门槛，仍有大幅跟踪损伤。因此需要定位偏移、输出变化和失败的先后顺序，进一步区分数值拟合通过与搜索中心修正有效。

这是事后选定的开发案例诊断。它不重新选择主比较，不为某种策略设置新的采用门槛，也不增加训练或网络推理。

## 2. 输入与冻结边界

入口：`tracking/stage46_shift_audit.py`。默认输出：原 Stage-46 根目录下的新子目录 `analysis/search_shift_audit_v1/`。

脚本先严格读取 Stage-46 manifest、run、完成的 decision、全部48个 NPZ及其receipt，并继续核对 Stage-27 的冻结来源。原生复现、逐帧状态、真实调用预算仍由原 Stage-46 审计器检查。缺失、混合、NPZ字节改变或decision中的SHA不一致均报错；不重跑来源来掩盖错误。

随后只分析上述三个视频的四条已有轨迹。只需 Python 与 NumPy；不导入 Torch/OpenCV，不打开图像或 checkpoint，不重新估计光流，不改变跟踪输出。标签来自冻结来源，全部用于预测完成后的离线测量。

保留 Stage-46 的源码、manifest、run、smoke、trace、receipt 和 analysis，不修改既有结果或其指纹。本轮只写专用的新目录。输出也记录自己的配置、来源与代码SHA；重复命令核验并复用，已有报告或表内容不同则拒绝覆盖。

## 3. 具体分析

### 搜索偏移与实际裁剪分开记录

原生 `sample_target` 对裁剪原点取整，但 `map_box_back` 使用浮点父框中心。即使裁剪图像相同，映射回原图的框仍可能不同。本轮分别报告：

- 首次实际施加中心偏移、首次相对本组上一预测改变整数裁剪。
- 相对原生首次父框变化、整数裁剪变化、浮点输出变化与官方截断框变化。
- 首次选层变化、响应最大值幅度变化。
- 首个低 IoU 帧、首次失败起点/确认帧、首次持续至少30帧的事件起点。

浮点差异的描述阈值固定为1e-9，不改变旧实验的复现容差。整数裁剪用原几何函数重建，包含ceil边长、round原点和历史padding约定。

### 失败与固定时刻的前后窗口

沿用原事件定义：连续3帧IoU<0.2确认失败，起点回溯到其中第一帧；连续3帧IoU≥0.5恢复。持续30帧的事件可能夹杂中间IoU，不等于连续30帧低IoU。

每组取首次输出变化、首次整数裁剪变化、首次失败和首次长事件，再加入全视频1/4、1/2、3/4处的固定时刻。相同owner/frame合并锚点标签，每个锚点统一观察全部四条自身轨迹的前10帧、当帧、后30帧；窗口遇到视频边界截断并记录实际样本数，空窗口不伪造均值。固定时刻不保证成功。

记录IoU、裁剪覆盖率、父框/输出中心误差、偏移及相对速度、相机与相对速度的方向关系、拟合内点和残差、选层及响应幅度。背景矩阵映射前一帧GT中心后的误差也是离线代理，包含目标相对运动，不能当作相机真值误差或在线触发条件。

四组使用各自已有预测、遮罩与背景矩阵。相同帧比较并非相同父框输入，也没有建立单帧或长期因果增益。

### 中心误差变化的精确记账

使用记录中的裁剪边长 `s`、搜索父框中心 `p`、最终输出中心 `o`，定义：

```text
decoder_origin = p - s/2
remainder = o - decoder_origin
error = o - GT_center

Δerror = Δown_previous_center + Δapplied_shift
       - Δs/2 + Δremainder - ΔGT_center
```

输出各项的x/y向量和沿总误差变化方向的有符号投影，核对总和。`remainder` 包含模型输出、解码算术及最终边界裁剪，不能拆称为响应头贡献；各项大小不能相加解释百分比，有符号投影也不是唯一因果归因。

**Stage-46没有保存响应峰位置、热图、size/offset图或Top-K候选。** `raw_peak_amplitude`只表示最大值幅度，本轮不能判定是否切换响应峰、是否仍有正确目标候选或是否发生身份混淆。若确实需要这些证据，再单独设计相同父框的响应重放。

## 4. 服务器命令

通过已有 Git 流程同步新增脚本、测试和文档后，在项目根目录运行：

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
export STAGE46_ROOT="$PWD/output/stage46/sgla_camera_search_v1"
export STAGE46B_DIR="$STAGE46_ROOT/analysis/search_shift_audit_v1"

python -B tracking/test_stage46_shift_audit.py

python -B tracking/stage46_shift_audit.py \
  --input "${STAGE46_ROOT:?}"

cat "$STAGE46B_DIR/stage46b_case_summary.csv"
cat "$STAGE46B_DIR/stage46b_onsets.csv"
```

无需设置 `CHECKPOINT` 或 GPU。完整结束状态为 `COMPLETE_CACHED_SEARCH_SHIFT_AUDIT`，表示诊断完成，不代表新策略通过。

## 5. 输出与阅读顺序

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 本轮固定配置、来源和代码SHA |
| `stage46b_diagnostic.json` | 状态、审计范围、表哈希、向量核对误差和解释边界 |
| `stage46b_case_summary.csv` | 三视频×四方法的全视频指标与首次变化时刻，优先阅读 |
| `stage46b_onsets.csv` | 全部失败事件的起点、时长、低IoU帧数及起点几何/运动值，优先阅读 |
| `stage46b_anchors.csv` | owner、锚点帧和合并后的类型标签 |
| `stage46b_phase_summary.csv` | 各锚点下四组自身轨迹前/当/后的描述均值 |
| `stage46b_timeline.csv` | 所有锚点窗口的去重逐帧记录，带多重窗口归属 |
| `stage46b_transitions.csv` | 窗口内各帧相对前一帧的中心误差向量记账 |

帧号对应冻结视频中初始化图像为0的图像索引，跟踪帧从1开始；`source_file`给出实际文件，避免与图像文件名混淆。窗口相互重叠，不能把窗口或帧视为独立视频。

本地测试使用原生sampler/map-back函数定义与合成冻结轨迹，验证取整差异、失败索引、边界窗口、精确向量记账、48轨迹来源校验、重复运行和篡改拒绝。实际服务器NPZ并未在本机运行；下一步依据服务器输出解释三个案例，再固定后续机制。
