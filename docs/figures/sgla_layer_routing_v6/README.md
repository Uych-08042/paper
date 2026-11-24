# 时序感知选层框架与技术路线

| 版本 | 高清 PNG | 可编辑 SVG |
| --- | --- | --- |
| 中文 | [中文图](sgla_temporal_layer_framework_zh_v6.png) | [中文 SVG](sgla_temporal_layer_framework_zh_v6.svg) |
| 英文 | [英文图](sgla_temporal_layer_framework_en_v6.png) | [英文 SVG](sgla_temporal_layer_framework_en_v6.svg) |

## 图注

**Temporal-aware layer selection for UAV-view single-object tracking.** (a) A single search crop, centered using the preceding prediction, is processed with the fixed first-frame template by the frozen SGLA prefix (L1–L6). The resulting visual projection is combined with causal history to encode the short and long temporal states. (b) A shared risk gate uses predicted gain and downside to decide if the tracker can replace the native layer. When allowed, the pairwise ranker selects one of the nonnative candidates, and the selected independent block is followed by the original CENTER head and Hann window. The decoded box, response peak and executed switch are fed back for the next frame. The dashed box denotes the 16-frame cooldown under evaluation. The photos are illustrative crops of one real UAV image. The boxes are schematic annotations.

**无人机视角单目标跟踪中的时序感知选层。** (a) 用上一帧预测框确定当前帧的单个搜索裁剪，与固定首帧模板一起经过 SGLA 的 L1–L6。视觉投影与历史预测框及响应峰值组成短、长两种时序状态。(b) 共享风险门控判断是否允许换层，通过后用成对排序器从非原生候选中选一个执行，随后经原生 CENTER 头和 Hann 窗解码。预测框、响应峰值和实际换层状态写回。虚线框为 Stage-170 的 16 帧冷却试验，完整视频收益待验证。照片来自同一张真实无人机图像，裁剪和加框用于示意。

## 创新点的建模方法

**主要方法贡献是用同一搜索状态下的后续轨迹差异监督选层。** SGLA 的当前输出框会决定下一帧裁剪，因此层的评价需要覆盖后续跟踪。模型将选层写成两步决策，即是否替换原生层，以及允许替换后选哪一层。

### 1. 在固定执行预算下构造时序状态

以首帧给定框构造固定模板 $z_0$。对当前帧 $I_t$，用本策略的上一帧预测框 $\hat b_{t-1}$ 构造一个搜索裁剪。模板和搜索区域经过冻结的 L1–L6，得到 token 表示 $F_t$。将模板 token 的均值、最大值和搜索 token 的 $2\times2$ 池化拼接，再用固定随机矩阵得到 16 维视觉向量 $v_t$。

令 $h_j$ 表示第 $j$ 帧经 Hann 加权的响应峰值。历史状态使用已提交的 $(\hat b_j,h_j)$。用峰值的最新值、均值、标准差和首尾变化，以及中心位移和尺寸变化各自的最新值、均值构成 8 维基础统计。中心位移用前一预测框面积的平方根归一化。尺寸变化取宽高对数差的绝对值均值。长历史分支另加峰值连续下降比例、最新位移相对历史中位数的偏差、尺寸变化标准差。

| 图中符号 | 定义 |
| --- | --- |
| $x_4(t)\in\mathbb R^{24}$ | $v_t$ 与最近 4 帧的 8 维基础统计，用于收益预测和成对排序 |
| $x_{16}(t)\in\mathbb R^{27}$ | $v_t$ 与最近最多 16 帧的 11 维统计，用于退化预测 |
| $\ell_0$ | 当前帧由 SGLA 原生选择器给出的层 |
| $p_t$ | 由收益模型在六层中给出的原提议 |
| $\ell_r,\ell_t$ | 成对排序给出的非原生层，以及最终实际执行层 |

### 2. 用同状态续跑构造层间偏好

离线训练样本在同一帧共享模板、父框与搜索裁剪。起点分别执行 L7–L12 中的一个候选块，获得六种首步输出。每个分支随后统一用原生 SGLA 续跑 32 帧，各自用自己的预测框裁剪。层 $\ell$ 的监督值为这些后续帧的平均 IoU $U_t(\ell)$，相对原生层的变化为 $\Delta U_t(\ell)=U_t(\ell)-U_t(\ell_0)$。未来图像和标注用于离线构造监督。

成对排序只比较同帧的非原生候选。某层相对原生分支新增连续三帧 IoU $<0.2$，或丢失原生分支已有的连续三帧 IoU $\ge 0.5$，则记为发生退化。若一对中只有一层发生退化，则优先另一层。两层都通过时，仅在 $U_t$ 相差至少 0.01 时给出偏好。对优选层 $i$ 和较差层 $j$，使用线性层分数 $q_\ell(x_4)$ 拟合带 $L_2$ 正则的逻辑排序损失

$$
\mathcal L_{\mathrm{rank}}
=\sum_{(t,i,j)} w_{tij}\log\!\left(1+\exp\!\left[-\bigl(q_i(x_4(t))-q_j(x_4(t))\bigr)\right]\right)
+\tfrac{\lambda}{2}\|\theta\|_2^2 .
$$

权重在有监督样本的训练源组之间等分，再在组内按起点和层对分配。每折只使用训练源组拟合。推理时从非原生候选中取层分最高者作为 $\ell_r$。

### 3. 用风险门控与冷却控制干预时机

收益模型用 $x_4(t)$ 拟合各层的 $\Delta U_t(\ell)$，退化模型用 $x_{16}(t)$ 拟合 $\max(0,-\Delta U_t(\ell))$。两者均使用按源组加权的岭回归。若收益原始输出为 $f_g$，则原提议 $p_t=\arg\max_\ell f_g(\ell)$，图中 $g(p_t)=f_g(p_t)-f_g(\ell_0)$ 表示相对原生层的预测收益。预测退化量 $r(p_t)$ 在推理时裁为非负。

历史达到四帧、$p_t\ne\ell_0$ 且 $s_t=g(p_t)-r(p_t)>\tau$ 时，才允许用成对排序结果 $\ell_r$ 替换 $\ell_0$。$\tau$ 用外层训练源组内的逐源组留出预测校准。门控评估的候选是收益模型给出的 $p_t$，门控通过后由成对排序决定最终执行层。

Stage-170 新增一个时间条件。若上一次实际换层发生在 $t_{\mathrm{last}}$，则随后 16 帧沿用原生层，从第 17 帧起重新允许换层。首次换层前冷却条件默认满足，只有实际执行了非原生层才更新时间。该时间戳供冷却判断使用。最终每帧仍执行一次裁剪、六个前缀块、一个独立候选块和一次预测头。调用次数用于核对计算预算。

## 技术路线与证据

1. **配对生成监督。** 在 VisDrone-SOT 的相同起点逐层执行，再统一用原生 SGLA 续跑，记录后 32 帧 IoU 和连续高、低 IoU 段。训练和留出按视频源组分开。 [Stage-159](../../experiments/06_sgla_uav_tracking/STAGE159_LAYER_SELECTOR.md)
2. **固定起点验证选层。** Stage-166 固定 68 个换层起点，成对排序相对四帧对照提高 0.014 个 IoU 点，剔除 0176 后提高 0.022 个点。这个比较隔离了选哪一层的影响。 [Stage-166](../../experiments/06_sgla_uav_tracking/STAGE166_PAIRWISE_LAYER_RANKING.md)
3. **完整轨迹验证并回溯失跟。** Stage-167 成对排序的 AUC 为 62.598，原生为 63.961。Stage-168／169 使用同状态分支和定点阻断验证，0304 避开单次换层后观察窗内保持跟踪，0089 暂停后在窗口内恢复。 [Stage-167](../../experiments/06_sgla_uav_tracking/STAGE167_ONLINE_LAYER_RANKING.md) · [Stage-169](../../experiments/06_sgla_uav_tracking/STAGE169_TARGETED_LAYER_VETO.md)
4. **用统一冷却规则验证完整轨迹。** Stage-170 将固定 16 帧冷却用于全部 28 个开发视频，与原生 SGLA 及各自的无冷却对照比较 AUC、持续低 IoU、恢复和实际换层次数。正式完整视频结果待验证。 [Stage-170](../../experiments/06_sgla_uav_tracking/STAGE170_FULL_VIDEO_COOLDOWN.md)

## 图片来源与导出

实景素材沿用 v5 已归档的 *visdrone Dataset*（lab / Roboflow Universe，2022），原出处标注 CC BY 4.0。中文和英文图都采用相同图片，框线和高亮的 L8 是架构示意。 [原图出处及逐裁剪记录](../sgla_search_state_v5/README.md)

SVG 自包含 JPEG，可编辑文字、框和连线。PNG 为 3680 × 1494，带 300 dpi 元数据。

~~~powershell
python -B docs/figures/sgla_layer_routing_v6/render_layer_framework.py --rasterize
~~~
