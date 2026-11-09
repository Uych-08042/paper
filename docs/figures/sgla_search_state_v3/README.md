# SGLA 四候选 PPO 跟踪框架图 v3

| 格式 | 英文论文版 | 中文讨论版 |
|---|---|---|
| 可编辑 SVG | [英文 SVG](sgla_ppo_framework_en_v3.svg) | [中文 SVG](sgla_ppo_framework_zh_v3.svg) |
| 高清 PNG，3280 × 1452 | [英文 PNG](sgla_ppo_framework_en_v3.png) | [中文 PNG](sgla_ppo_framework_zh_v3.png) |

版式参考 Ziang Cao 等的 *Towards Real-World Visual Tracking With Temporal Contexts*（TCTrack++），IEEE Transactions on Pattern Analysis and Machine Intelligence，45(12)，15834–15849，2023。参考 Fig. 2 的分区方式，以及用不同箭头区分帧内处理与跨帧传递的做法。新版重新绘制了场景、裁剪、候选与响应图，内容依据本项目实现。 [IEEE Xplore](https://ieeexplore.ieee.org/document/10225683/) · [作者稿第 3 页，Fig. 2](https://arxiv.org/pdf/2308.10330v1#page=3)

## 双栏通栏图注

**History-guided candidate selection with PPO.** (a) Latest and lagged centers, c(t−1) and c(t−2), are crossed with the latest size s(t−1) and historical size s(hist). The latter is the exponentiated componentwise median of log sizes over up to five predicted boxes. Each of the four crops is processed independently by the same frozen SGLA with first-frame template z₀. (b) The actor scores four concatenated, fit-normalized 64-D descriptors. Identical decoded boxes retain the lowest action index. The highest-probability action selects the output, which updates the next frame's search history (green dashed arrow). (c) During fitting, sampled actions advance the tracker. The reward is the selected box's IoU with ground truth g(t). A separate critic and generalized advantage estimation (GAE) supply the advantage and value targets. The PPO ratio ρ compares updated and behavior action probabilities. The objective combines the clipped surrogate with value loss and entropy regularization. At inference, ground truth is used for first-frame initialization. Images and responses are schematic.

**历史引导的 PPO 候选选择。** （a）最新与滞后中心分别和最新、历史尺寸组合。历史尺寸由最近至多五个预测框的宽高在对数域逐分量取中位数，再映回原尺度。四个裁剪分别经过同一冻结 SGLA，并共用首帧模板。（b）四个 64 维描述拼接并用拟合集统计量标准化。完全相同的解码框保留最小动作索引。PPO 选择概率最高的合法候选，绿色虚线表示选中框写回历史以构造下一帧裁剪。（c）拟合时按策略采样动作并推进跟踪，奖励为选中框与真实框的 IoU。独立 critic 和 GAE 生成优势及价值目标，ρ 为更新后与采样时的动作概率比。裁剪代理目标结合价值损失和熵正则。推理时真实框用于首帧初始化。场景与响应图均为示意。

## 实验依据

在线分区对应 Stage-74 训练、Stage-76 与 Stage-83 重测的原尺度四候选 PPO。训练图展开了实际采样闭环、IoU 奖励、GAE 与裁剪策略目标。评价时 critic 不参与动作计算，冻结 actor 直接选 q0、q1、q2 或 q3。 [Stage-74 协议](../../experiments/06_sgla_uav_tracking/STAGE74_DIRECT_FOUR_CANDIDATE_PPO.md)

当前图定位为**开发原型**。原尺度 PPO 在 12 个开发视频上的 AUC 为 67.2356，原生 SGLA 为 63.4512。PPO 相对 Q 和既有 future 的配对区间均跨零，并在两个原生稳定视频上出现失跟。Stage-83 的短时中心保留把 AUC 降至 65.6062，五项门槛均未通过，因此只在实验讨论中记录。28 组确认集仍待方法定型。 [Stage-76 结果](../../experiments/06_sgla_uav_tracking/STAGE76_RESULTS_REVIEW.md) · [Stage-83 结果](../../experiments/06_sgla_uav_tracking/STAGE83_RESULTS_REVIEW.md)

## 再生成

从仓库根目录执行下列命令。SVG 使用 Python 标准库，PNG 导出需要 Pillow 和 Windows 字体。绘图器复用 v1 的矢量基本图形。

```powershell
python -B docs/figures/sgla_search_state_v3/render_method_framework.py --rasterize
```

[绘图源文件](render_method_framework.py) · [上一版框架图](../sgla_search_state_v2/README.md)
