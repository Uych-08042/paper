# SGLA 无人机跟踪：初始论文方法框架图

日期：2026-11-01，Asia/Hong_Kong。依据当前已完成的 Stage-54～60 研究绘制，用于论文结构讨论，暂用描述性标题“基于自身历史的 SGLA 搜索状态选择”。本次只新增图与说明，实验代码、权重与封存结果保持原样。

## 文件

| 用途 | 中文 | 英文 |
|---|---|---|
| 可编辑矢量图，文字和形状可分别修改 | [中文 SVG](sgla_search_state_framework_zh_v1.svg) | [英文 SVG](sgla_search_state_framework_en_v1.svg) |
| 高清预览，4320 × 2540 像素，300 dpi | [中文 PNG](sgla_search_state_framework_zh_v1.png) | [英文 PNG](sgla_search_state_framework_en_v1.png) |

[绘图源文件](render_method_framework.py)直接生成矢量形状和文字，不依赖生成式图片模型。图中车辆、道路、候选框均为示意，未使用真实数据帧或暗示具体样本表现。PNG 使用本机字体渲染；SVG 保留可编辑文字，跨系统可优先使用 PNG 保持外观。

## 图的核心叙事

目标是改善 **SGLA 无人机单目标跟踪中“预测框→下一帧裁剪→预测框”的反馈稳定性**。当前框不仅是输出，还决定下一帧的搜索位置、范围及目标分辨率。中心或尺寸变化可能改变响应竞争，进一步影响能否避免或摆脱持续失跟。

上半图展示在线闭环：从最近的已接受预测中提取当前／滞后中心和当前／历史尺寸，组成四个搜索父框；对每个父框分别裁剪并运行同一个固定的 SGLA 跟踪器，评分后接受选中候选的原始预测框，再将它写入历史。

青色实线突出目前最值得继续验证的“自身历史候选＋Hann 响应选择”。橙色虚线展示已测试的即时／8 帧收益学习策略；这两种评分与响应评分是独立对比策略，没有在图中引入新的融合、可信门控或安全回退模块。

下半图只解释学习支线的离线监督：在 fit 视频的自身轨迹上均匀抽取锚点，从同一历史构造四个父框，各自推进 8 帧。仅首帧改变父框，其后均为普通 E 自身预测。预测产生后才用 fit GT 计算相对 current 分支的即时／8 帧平均 IoU 差，用相同特征和回归设置分别拟合评分器。

## 符号及实现对应

- `c(t−1)` 为最新已接受框中心；`c(t−2)` 为前一个历史中心，历史不足时复用最新中心。`s(t−1)` 为最新框的宽高。
- `s(hist)` 是最近至多 5 个已接受框宽高的逐维 log 中位数再取 exp；历史可包含标准首帧初始化框。当前帧尚未接受的探测框不进入该统计。
- 四候选编号保持 Stage-60 原顺序：0 current、1 history_size、2 lag_center、3 lag_center_history_size。图中为突出中心×尺寸组合而排成 2 × 2。
- `Mᵏ` 为第 k 次探测的原始响应；响应分数使用原生 float32 Hann 乘积的峰值。图中分支与评分框描述的是同一次 Hann 加权，不额外重复加权。
- E 是既有两帧末帧保持训练得到的几何适配跟踪器，即 Stage-54 的 rollout_preserve、Stage-58 的 endpoint_control，Stage-60 中权重固定。它保留原生 SGLA 骨干、候选结构、选择器与响应分支；先前只对 size／offset 做几何适配。E 的 current 完整轨迹与原生 native 参考是不同策略。
- 每次搜索执行 6 个前缀块，加 L7～L12 中 **1 个独立候选块**。高亮 L9 只是结构示意，不表示本方法固定选择 L9，也不表示串行执行 12 层。
- 所有候选使用同一固定首帧模板。搜索裁剪不同，搜索特征逐候选分别计算；每帧 4 次完整搜索的成本保留在图中，未画成共享搜索特征或 1 次前向。
- 学习评分使用 24 维特征，先减同状态 current 特征，再用本折 fit RMS 逐维缩放。无截距岭回归 λ=0.01，current 预测收益恒为 0。另一个候选必须高于 current 的分数 1e−6；并列按固定候选顺序处理。
- `θ₁ / θ₈` 是两种监督目标的评分参数，连接上、下半图的同名学习分支。实际各有折／种子对应模型，图中为统一示意。
- 8 帧标签是一次父框干预后普通推进的相对回报；完整在线策略会反复选择，因此该标签不能直接等同于连续选择策略的最优价值。

## 当前证据如何决定图的主次

| 实验依据 | 支持图中的什么内容 | 仍需补足什么 |
|---|---|---|
| [Stage-59 复核](../../experiments/06_sgla_uav_tracking/STAGE59_RESULTS_REVIEW.md) | 中心与尺寸可交互，特定组合改变同一冻结头的后继表现；据此组织四种组合候选 | 当时为条件诊断，不是完整在线方法收益 |
| [Stage-60 协议](../../experiments/06_sgla_uav_tracking/STAGE60_CAUSAL_SEARCH_STATE_VALUE.md) | 因果自身历史、四次独立探测、固定 E、两种匹配标签与线性评分器 | 这是当前实现，不替后续方案预设新增模块 |
| [Stage-60 复核](../../experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md) | response−current 宏 AUC +3.7427，9 胜 3 负；主线值得继续发展 | 95% CI [−1.0196,+9.5075] 跨零，且有 4 倍搜索预算；还需同预算普通多裁剪控制或减少探测成本 |
| 同一 Stage-60 复核 | 已运行后继收益支线，可在图中保留其监督结构 | future−response AUC −3.5585，家族区间全负；当前线性后继学习没有建立同预算新增价值 |

因此，初稿的新增价值假设是：**围绕搜索输入的中心—尺寸兼容性，用因果自身历史形成候选，并在真实闭环中选择合适的下一步输入状态**。图没有将未来价值学习画成已经优于响应选择的最终贡献，也没有宣称独立泛化、部署加速或压缩收益。全部现有完整评价仍是反复使用的 12 个开发源组。

## 建议图注

**中文。** 基于自身历史的 SGLA 搜索状态选择框架。在线阶段从已接受预测提取两种中心和两种尺寸，组合成四个搜索父框，并使用同一个固定的 SGLA 跟踪器分别预测。实线以 Hann 响应选择候选，只有选中输出进入后续历史。虚线表示学习评分的探索替代策略：使用匹配起点的短分支构造即时及 8 帧相对收益标签，离线拟合线性评分器。GT 和未来图像仅用于 fit 标签；在线仅使用首帧初始化、历史预测和当前图像。四候选策略每帧运行四次完整搜索。

**English.** History-guided search-state selection for SGLA-based UAV tracking. At each frame, two centers and two sizes from accepted predictions form four search parents. The same fixed SGLA-based tracker processes each crop separately. The solid route selects the candidate with the highest Hann-weighted response peak; only its decoded output updates the accepted history. The dashed route denotes an exploratory learned alternative. Matched-start branches provide immediate and eight-frame returns relative to the current-parent branch for offline scorer fitting. Ground truth and future images are restricted to fit-split labels. Online inference uses first-frame initialization, accepted history and the current image. Each four-candidate policy requires four complete searches per frame.

## 修改与再导出

在仓库根目录运行；只写本图文件，不启动任何跟踪实验：

```powershell
# 仅 SVG：Python 标准库即可。
python -B docs/figures/sgla_search_state_v1/render_method_framework.py

# 同时生成高清 PNG：需要 Pillow，以及本机 Windows 的 Arial / 微软雅黑 / Cambria 字体。
python -B docs/figures/sgla_search_state_v1/render_method_framework.py --rasterize
```

当前源文件集中保存中英标签、布局、箭头、配色和示意图形，便于下一轮将保留的选择器、同预算对照或减预算方案落实后修改。下一轮实验尚未由这次画图自动立项。
