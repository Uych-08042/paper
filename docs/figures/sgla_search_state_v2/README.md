# 论文方法框架图 v2：总览＋关键模块细节

日期：2026-11-01。按用户要求，以正式顶会论文的方法图为版式模板重排 [v1](../sgla_search_state_v1/README.md)，保留中英文可编辑矢量图和高清预览。本次只修改图示组织，不改变研究方法、实验代码或结果。

| 文件 | 英文论文版 | 中文讨论版 |
|---|---|---|
| 可编辑 SVG | [英文 SVG](sgla_search_state_framework_en_v2.svg) | [中文 SVG](sgla_search_state_framework_zh_v2.svg) |
| 高清 PNG（4000 × 2120，300 dpi） | [英文 PNG](sgla_search_state_framework_en_v2.png) | [中文 PNG](sgla_search_state_framework_zh_v2.png) |

## 具体模板与取舍

主要参考 **OSTrack，ECCV 2022，Fig. 3（论文第 5 页）**：[正式论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820332.pdf)，[作者项目](https://github.com/botaoye/OSTrack)。该图将整体输入—网络—预测流程与编码层关键模块细节分开展示。本图借鉴这种“总览＋局部展开”的组织方式，模块、文字、候选图形和响应图均按当前研究重新绘制。

对应调整：

- **(a) 在线总览**：压缩为一条主要数据流，保留因果反馈环；当前帧／历史、候选构造、固定 SGLA、候选证据、评分、输出依次排列。
- **(b) 关键模块展开**：将“中心 × 尺寸”候选画成真正的 2 × 2 组合矩阵，避免把四个候选仅列成文字清单。
- **(c) 学习监督展开**：用四条短分支、相对收益标签和线性评分器表示离线训练，和在线路径区分。
- 原生保留结构使用灰色，历史候选使用蓝色，响应选择使用绿色，学习替代策略使用橙色虚线；减少大色块和重复说明。
- 移除大标题、实验阶段清单和图内结果讨论；结果判断放在图注与研究说明中。

研究内容与 v1 相同：E 是已有几何适配基线，Stage-60 中固定。骨干为六个顺序前缀块＋L7～L12 中一个独立候选，L9 高亮仅作示意。四次候选搜索使用相同权重与固定首帧模板，搜索特征分别计算。图中响应图、车辆和框均为合成示意，不能当作某个样本的真实结果。

## 图注初稿

**Figure. Overview of history-guided search-state selection for SGLA-based UAV tracking.** (a) Accepted predictions provide causal search-parent proposals. Each parent defines a separate crop processed by the same fixed SGLA-based tracker. Candidate response maps and decoded boxes are scored by either Hann-weighted response or an alternative learned utility scorer; only the selected decoded output updates the accepted history. (b) Latest and lagged centers are crossed with latest and historical sizes to form four proposals. Historical size is the componentwise log-median of up to five accepted boxes. (c) On fit videos, matched-start branches change the parent only at the first step, followed by eight-frame ordinary E continuations. Immediate or mean continuation IoU gains relative to the current-parent branch supervise separate linear scorers. Future frames and GT are used only for offline fit labels; online GT is limited to standard first-frame initialization. Four proposals require four complete inference passes per frame.

当前证据仍为开发验证：response−current AUC +3.7427，但区间跨零；future−response AUC −3.5585，尚未建立同预算新增价值。因此橙色分支标为替代策略，不表述为已优于响应的最终贡献。完整数值与局限见 [Stage-60 结果复核](../../experiments/06_sgla_uav_tracking/STAGE60_RESULTS_REVIEW.md)；实现细节与符号说明见 [v1 说明](../sgla_search_state_v1/README.md)。

## 再生成

绘图源文件复用 v1 中的 SVG／PNG 基础绘图器，因此两份源文件应共同保留。不会改写 v1 图或任何实验产物。

```powershell
python -B docs/figures/sgla_search_state_v2/render_method_framework.py --rasterize
```

生成 SVG 只用 Python 标准库；PNG 需要 Pillow 和 Windows Arial／微软雅黑／Cambria 字体。[本版绘图源文件](render_method_framework.py)。
