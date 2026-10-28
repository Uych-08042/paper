# Stage-36C：失跟起点的响应峰与中心跳变审计

Stage-36B 的同裁剪平均结果偏向宽高变化，但 0089 的两个失跟起点出现了另一种现象：父框中心仍接近 GT，当前输出中心却突然偏离。窗口平均的宽高效应不能解释这一帧发生了什么。

本轮固定检查原 Stage-36 计划中已有的两个 `first_failure`：

| 视频 | 学生 | 相邻帧 |
|---|---|---|
| `uav0000089_00920_s` | `supervised_seed2110` | 723 → 724 |
| `uav0000089_00920_s` | `supervised_seed2111` | 640 → 641 |

每个起点只读取已有的前后各两帧，共十张不同帧图像对应的缓存。两次事件来自同一视频，不能视为两个独立视频。

**只使用 NumPy 和既有缓存，不训练、不执行模型前向、不生成新预测。** 先回答中心跳变由哪些解码分量构成，再决定是否值得提出新的训练机制。

## 服务器运行

通过 Git 同步新增脚本、测试和本说明，在服务器仓库根目录执行：

```bash
export STAGE36_ROOT="${STAGE36_ROOT:-$PWD/output/stage36/same_crop_mechanism_v1}"

python tracking/test_stage36_onset_jump.py

python tracking/stage36_onset_jump_audit.py \
  --input "${STAGE36_ROOT:?}"

export STAGE36C_DIR="$STAGE36_ROOT/analysis/onset_jump_audit_v1"
cat "$STAGE36C_DIR/stage36c_events.csv"
```

终端会打印两行事件简表。交流结果时粘贴这两行或 `stage36c_events.csv` 即可，完整 JSON 中的大量来源哈希保留用于审计。

本轮不需要 GPU 或 `CHECKPOINT` 环境变量。既有来源文件、权重和图像仍需保留，以通过前序来源哈希校验；不会加载网络权重或解码图像。输入必须是已完成 Stage-36 和 Stage-36B 的原目录。

## 分解的具体含义

原浮点父框解码中心可以写为：

`中心 = 父框中心 + 像素步长 × (响应峰网格坐标 + 所选 offset − 网格边长/2)`。

这里的响应峰来自原响应乘原窗口后的 argmax；像素步长沿用原搜索尺寸和 resize factor。保留原 float32 解码、边界裁剪以及原生框绑定产生的数值差异。

对前一帧到当前帧的中心变化，记录九个二维向量：

| 分量 | 含义 |
|---|---|
| `parent_translation` | 父框中心平移 |
| `crop_scale` | 搜索裁剪映射的像素步长变化 |
| `response_grid` | 响应峰从旧格移动到新格 |
| `offset_field` | 固定旧格、新格位置时，offset 图在相邻帧间的变化，取两位置平均 |
| `offset_resampling` | 固定前后两张 offset 图时，换格带来的 offset 取值变化，取两张图平均 |
| `float32_arithmetic` | 理想 float64 公式与原 float32 解码的数值差异变化 |
| `boundary_clipping` | 图像边界裁剪造成的中心变化 |
| `native_decode` | 与缓存中原生输出绑定的剩余数值差异变化 |
| `gt_motion` | 负的 GT 中心位移，仅用于离线误差记账 |

前八项之和等于原生输出中心的位移；加入第九项后，等于预测中心相对 GT 的**误差向量变化**。逐个 x/y 坐标的闭合误差须不超过 `1e-9` 像素。

将 offset 分成两项，是为了区分“图在固定格发生变化”和“响应峰换格后读取了图上的另一处”。采用两种交换顺序平均的分配，二者严格相加为所选 offset 的总变化。跨帧的相同网格坐标不一定对应同一物理像素；该分解是记账约定，不是唯一因果归因。

宽高不会进入未裁剪框的中心公式，但可能通过边界裁剪改变最终框中心，也会影响下一帧裁剪和 IoU。本轮保留宽比、高比和裁剪差异，不能据此否定前序发现的平均宽高效应。

## 简表怎么读

优先看 `stage36c_events.csv`：

- `previous_iou`、`onset_iou`：前一帧与起点帧的原评价 IoU。
- `from_peak_cell`、`to_peak_cell`、`peak_changed`：是否发生峰值换格；完整 x/y 坐标在逐帧表中。
- `previous_center_error_gt_units`、`onset_center_error_gt_units`：两帧各自的中心误差大小。
- `error_jump_gt_units`：两帧误差向量之差的长度，**不是两帧误差长度之差**。
- `grid_along_error_gt_units`、`offset_field_along_error_gt_units`、`offset_resampling_along_error_gt_units`：各项沿误差变化方向的有符号投影。
- `parent_and_scale_along_error_gt_units`：父框平移与裁剪步长两项合并；数值解码三项也单独合并报告。
- `previous_weighted_top5_gt_recall`、`onset_weighted_top5_gt_recall`：原固定局部峰、NMS 和 top-5 规则下是否保留 GT 匹配框。
- `onset_dense_best_gt_iou`：所有密集格中最高 GT IoU，是事后定位上限。
- `sgla_same_crop_iou`、`sgla_same_crop_top5_gt_recall`：SGLA 在**该学生同一裁剪**上的已有缓存结果。

每条相邻帧转换的所有分量统一除以**当前帧** `sqrt(GT宽×GT高)`。有符号投影之和等于 `error_jump_gt_units`，正负项可能抵消，单项也可能大于总量。各分量的向量长度不能直接相加。若误差变化接近零，有符号投影留空，避免除以零。

“正投影”表示沿观测到的误差变化方向，不自动表示该分量降低了 IoU；GT 自身变化包含目标、相机及图像坐标变化，不能当作无人机或目标的独立运动测量。

## 输出与证据边界

新增文件全部位于 `analysis/onset_jump_audit_v1/`：

| 文件 | 内容 |
|---|---|
| `stage36c_events.csv` | 两个原生学生起点的简表 |
| `stage36c_frame_states.csv` | 十个帧位置 × 四种父框 × 四个模型，共 160 个已有视图 |
| `stage36c_candidates.csv` | 同一视图的 raw / weighted 固定 top-5 候选、分数、框和 GT IoU |
| `stage36c_transitions.csv` | 连续帧转换、所选格和 offset、记账闭合误差 |
| `stage36c_components.csv` | 每条转换的九项 x/y 位移、长度及有符号投影 |
| `stage36c_same_crop_onset.csv` | 两个起点上的四种父框 × 四个模型，便于同裁剪比较 |
| `stage36c_diagnostic.json` | 完整性、配置、来源和输出指纹 |

`COMPLETE_CACHED_ONSET_AUDIT` 只表示审计完整且记账一致，不是新方法通过。不会统计新的显著性或采用门槛。两次起点均由已消耗的开发结果选出；同裁剪反事实不是另一条真实闭环轨迹。GT 覆盖率不等于可见性，非 GT 框也不能直接标成语义干扰物。

本轮主要区分三种后续假设：峰格切换、固定格 offset 变化、裁剪历史变化。候选保留只说明原规则下是否存在几何匹配框；不能直接证明可部署的身份识别器能够选对，也不据此自动启动 PPO 或蒸馏训练。

## 冻结、复用和中断

运行前核对原 Stage-36 来源链和 Stage-36B 完整报告、代码及表格哈希；计算结束再次核对。原阶段文件不改写，新报告绑定本轮代码、配置和六张表格哈希。

完成后重跑会审计并复用结果；来源、配置、代码或新表格被修改时拒绝静默覆盖。若中断时新报告尚未完整发布，重跑同一命令重新计算 CPU 统计即可。若原冻结缓存缺少规定帧或起点不匹配，直接报错，不新增 GPU 回放或另选事件。

测试覆盖峰格与 offset 分离、空间重采样与跨帧图变化分离、裁剪/GT/数值分量闭合、零变化处理、完整缓存流程、来源不变、重复复用、篡改拒绝，以及缺少帧或非原定起点的拒绝。
