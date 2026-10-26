# Stage-36B：缓存中的中心偏移与宽高分解

Stage-36 已说明部分变化与定位几何有关，并观察到学生在基线裁剪上改善、在自身父框上恶化的情况。本轮进一步分开响应选位置、子格中心 offset、宽高 size，并检查原定起点前后的几何状态。

**完全使用现有缓存，只有 CPU 解码与统计；不训练、不执行模型前向、不重新生成裁剪或预测。** 四个案例、帧、起点和窗口均沿用 Stage-36，不根据本轮结果重新选取。

## 服务器命令

通过仓库同步三个新增文件，在原仓库根目录执行：

```bash
export STAGE36_ROOT="${STAGE36_ROOT:-$PWD/output/stage36/same_crop_mechanism_v1}"

python tracking/test_stage36_geometry_audit.py

python tracking/stage36_geometry_audit.py --input "${STAGE36_ROOT:?}"

export STAGE36B_DIR="$STAGE36_ROOT/analysis/geometry_factor_audit_v1"
cat "$STAGE36B_DIR/stage36b_geometry_focus.csv"
```

终端会直接打印八行 offset/size/geometry/full 简表。完整 JSON 包含大量来源哈希，保留用于核对即可，交流结果时无需整份粘贴。

本轮不需要 `CHECKPOINT` 参数或 GPU。既有权重文件、图像和缓存仍需存在，以通过旧阶段的来源哈希审计，但不会加载神经网络、解码图像或调用 CUDA。

成功状态 `COMPLETE_CACHED_GEOMETRY_AUDIT` 仅表示分析完整、计算一致，**不表示一种新方法通过**。原 Stage-35、35B、36 结果不改写。

## 八种组合

对每个原定帧、四种原裁剪父框、每条基线与每个学生，固定网格和窗口，计算响应 R、offset O、size Z 的全部 2×2×2 组合。

| 位串 R/O/Z | 响应来自 | offset 来自 | size 来自 |
|---|---|---|---|
| 000 | 基线 | 基线 | 基线 |
| 100 | 学生 | 基线 | 基线 |
| 010 | 基线 | 学生 | 基线 |
| 001 | 基线 | 基线 | 学生 |
| 110 | 学生 | 学生 | 基线 |
| 101 | 学生 | 基线 | 学生 |
| 011 | 基线 | 学生 | 学生 |
| 111 | 学生 | 学生 | 学生 |

响应乘原窗口后选定一个格；offset 只提供这个格内的中心修正；size 提供该格的宽高。因此，**offset 不是完整中心位置**，响应换格也能改变中心。

000、100、011、111 必须逐帧重现 Stage-36 原始 `stage36_hybrids.csv` 的四个 IoU 端点，误差≤1e-12。单点解码沿用原 float32 运算、浮点父框映射和边界裁剪。

原生端点仍使用来源模型的原解码框。新增混合组合只有在“所选格、offset、size 与某个来源原生输入完全一致”时，才绑定该原生框；不使用 GT 修正框，也不将近似相等的通道吸附到来源值。原有四端点的绑定规则保持不变。

## 如何阅读分解

`geometry_focus.csv` 保留**基线自身父框**，先按视频/基线分别平均两 seeds，给出最便于判断的八行结果；保留 `sampled_frames` 和 `seed_repeats`，两 seeds 不增加独立视频数。

最先查看：

- `mean_geometry_gain`：原 Stage-36 的“只换学生 offset＋size”差异，即 011−000。
- `mean_offset_given_reference_response`：固定基线响应，对先换 offset 和后换 offset 两种顺序的增量取平均。
- `mean_size_given_reference_response`：同样固定基线响应，对 size 的两种交换顺序取平均。

后二者之和严格等于 `mean_geometry_gain`。这样可以看清 0016 的几何损伤和 0089/0090 的同裁剪几何改善，更偏向子格中心偏移、宽高，还是二者交互。

同时报告固定学生响应时的两项分配，以及所有六种 R/O/Z 交换顺序平均后的三项增量。六序平均的三项之和严格等于完整差异 111−000；另列两两和三阶交互。**这些是交互效应的记账约定，不是唯一的物理因果分配。**

所有 IoU 和差异均用 0–1 单位，不能当作 0–100 的 AUC 点。窗口均值不替代完整视频结果，不进行新的显著性或采用门槛判断。

## 边界裁剪与取整

宽高改变时，即使原始中心未变，图像边界裁剪也可能移动最终框中心。因此记录：

- 裁剪前和裁剪后的中心误差、宽比、高比。
- 边界裁剪引起的最大坐标变化。
- 为匹配来源原生解码而产生的微小数值绑定差异。

中心误差以 `sqrt(GT宽×GT高)` 归一化。宽比、高比分别除以对应 GT 尺寸。

主要结果仍使用原评价的预测框截断取整规则；另报**不取整预测框的浮点 IoU**及相应分解，作为量化敏感性检查。如果两者给出的方向不同，应先检查边界/取整，不能直接归因为学习机制。没有任何输出用于替换原轨迹。

## 起点前后时间线

沿用原计划中的所有起点：

- `first_sustained_discrepancy`：原定的首次同方向持续差异。
- `first_failure`：原定的首次持续低 IoU 事件。

持续差异起点使用原对应的学生/基线；学生失跟起点与两条基线分别比较，基线失跟起点与两个学生分别比较。每一对都检查基线父框与学生自身父框，分别运行两种已有输出的离线几何统计。

`timeline.csv` 的 `relative_frame=0` 是原起点，负值为之前，正值为之后。`timeline_summary.csv` 按 before、anchor、after 分别平均，并报告实际帧数，不补齐视频边界外的帧。

关键列包括：

- `coverage`：GT 在实际可用裁剪内的覆盖率，不等于可见性。
- `parent_center_error_gt_units`：生成当前裁剪的父框中心相对当前 GT 的误差。
- `parent_side_ratio`：父框面积平方根／当前 GT 面积平方根。
- `center_error_gt_units`、`width_ratio`、`height_ratio`：当前输出的几何偏差。
- 对应的 `unclipped_*` 列：区分原始预测与边界裁剪。
- `projected_gt_side_px`：输入投影尺寸，不是原图目标分辨率。

首次“持续差异”可能是学生更好，也可能更差，必须结合 `direction` 解读，不能把所有起点都叫失跟起点。窗口重叠时会保留归属，不能将重复帧或 before/after 均值视作独立样本或因果时间顺序证明。

## 输出

全部新文件位于 `analysis/geometry_factor_audit_v1/`：

| 文件 | 用途 |
|---|---|
| `stage36b_geometry_focus.csv` | 基线自身父框、双 seed 均值的简表，首先查看 |
| `stage36b_factor_summary.csv` | 各案例、父框、基线和 seed 的完整分解 |
| `stage36b_phase_factor_summary.csv` | 原固定时间控制点、原起点帧分别统计 |
| `stage36b_factorial_frames.csv` | 所有原定帧的八端点、交互与六序平均分解 |
| `stage36b_corner_metrics.csv` | 原起点及固定控制点上八种组合的框和几何指标 |
| `stage36b_timeline.csv` | 原起点前后逐帧状态 |
| `stage36b_timeline_summary.csv` | 原起点前后分段统计 |
| `stage36b_anchor_points.csv` | 起点帧本身，便于与审阅图对齐 |
| `stage36b_diagnostic.json` | 完整性、端点误差、分解闭合误差和来源指纹 |

若原计划没有任何起点，时间线文件可能不生成，报告会明确 `timeline_rows=0`；不能将缺少起点视作成功。

可按案例和原生父框查看时间线：

```bash
awk -F, 'NR==1 || ($1=="uav0000089_00920_s" && $9==$10)' \
  "$STAGE36B_DIR/stage36b_timeline_summary.csv"
```

这里第 9、10 列分别是 `parent_method`、`evaluator`，用于保留每种模型自己的裁剪路径。逐 seed 结果全部保留，不选择表现更好的 seed。

## 冻结、复用与测试

分析前核对原计划、原完整诊断、四个缓存、全部原表格/图片哈希，并沿用旧缓存的父框、解码、GT 和调用数审计。计算结束后再次核对来源；新报告绑定新代码、配置和输出表格哈希。

重复命令会审计后复用完整结果。输入、代码或新输出被修改时拒绝静默覆盖。中断后重跑同一命令即可，不存在训练 checkpoint 续跑。没有新增数据、超参数搜索、GT 输出纠正或方法 PASS/FAIL。

测试覆盖：单点与原密集解码逐元素一致；单因素响应/offset/size 的隔离；恒等干预；交互分解闭合；边界裁剪影响；缓存与旧端点一致；原数据不变和篡改拒绝。均为 NumPy CPU 测试，不调用模型或 CUDA。
