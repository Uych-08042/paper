# 历史条件搜索与后继提交框架 v4

更新至2026-11-16，内容依据Stage-113完整跟踪主干与Stage-120离线提交研究。重新绘制全部图形，参考用户所附两张框架图的主流程／细节分区、浅色模块、场景堆叠和底部时间轴组织。未复用原论文截图中的照片、模块或文字，也未将其扩散、语言、多模态和生成解码器机制移植到本项目。

| 格式 | 英文版 | 中文版 |
| --- | --- | --- |
| 高清PNG，4320×2400 | [英文PNG](sgla_history_search_framework_en_v4.png) | [中文PNG](sgla_history_search_framework_zh_v4.png) |
| 可编辑SVG | [英文SVG](sgla_history_search_framework_en_v4.svg) | [中文SVG](sgla_history_search_framework_zh_v4.svg) |

## 图中各部分

- **(a) 历史条件搜索。** 最近至多五个预测框、上一帧所选响应峰及可信资格组成32维输入。冻结的离线岭模型输出中心／对数尺寸均值及条件误差尺度，校准系数α缩放固定的16对正反高斯扰动，构成32个假设状态。标准父框＋均值父框＋32个状态父框形成34个备选项；按固定SGLA搜索系数转换成裁剪，保留标准父框并贪心选择另外三个区域。34项可能重复，四是当前预算，不是最优数量结论。场景是四维状态，图中散点只画中心的二维示意，不表示实际分布、轨迹或区间覆盖保证。
- **(b) 冻结SGLA与共享评分。** 当前图像按四个父框裁剪／缩放，与固定首帧模板输入同一SGLA。六个前缀块后，原生选择器只执行L7～L12中的一个独立块，图中突出L9仅为示意；不是串行执行十二层。CENTER头原生解码后得到q0～q3及响应／几何证据。64→32 Tanh→1 Sigmoid评分器逐候选共享参数，监督拟合当前IoU，argmax选择q*。标签GT只用于离线拟合，在线不提供后续GT。q0是标准搜索的当前输出，不是上一帧框。此处没有PPO、扩散模型或在线模板更新。
- **(c) 独立虚线分区。** Stage-120在已保存同状态q0／q*及实际匹配后继上检验提交。原13维证据加24维RGB直方图／NCC、覆盖／有效位／参考年龄，共37维；参考为首帧GT初始化及此前30帧内最近符合旧可信规则的预测。岭头预测q0一次动作相对原policy的后31帧平均IoU差，正分接管q0，否则保持q*。图中>0是对固定>1e−12判据的视觉简写。它仍是离线交叉拟合与匹配动作评价，尚未部署为持续门控完成整段自身轨迹，故不接入(b)的实线反馈。
- **(d) 时序反馈。** 主干选中框进入自身历史，上一帧响应、选择及资格状态也更新，影响下一帧生成。下方时间条重复t表示同一帧内的不同处理步骤，不是多次时间推进。

实线主干已完成完整开发轨迹运行，但未通过收益／保护门槛；“已完成”不表示方法采用。Stage-120只出现原支持外观的局部次要信号，主比较仍跨零且必要救援受损。图不展示成功结论、速度收益或确认集结果。场景、目标框、响应热图及突出层均为人工示意。

## 英文图注草稿

**History-conditioned search with a separately evaluated future-gain commit head.** (a) Past predictions and response cues condition a learned mean and error scale. Thirty-two fixed antithetic Gaussian state scenarios, together with the standard and mean proposals, form a pool of 34 possibly repeated search options. A geometric marginal-support criterion retains the standard option and allocates three additional searches. (b) Each crop is processed by the same frozen SGLA with its original first-frame template. A shared quality regressor scores 64-D candidate descriptors and selects the output, which updates the next-frame prediction history and response state. (c) A separate offline matched-action study augments 13-D relative evidence with 24-D appearance and reference cues to predict the subsequent 31-frame benefit of one q0 override. This head has not been validated in continuous gated rollouts. (d) The schematic temporal strip illustrates how selected predictions affect later searches. All images, response maps, scenario projections, and the highlighted layer are illustrative. Ground truth after initialization is used only for offline supervision and evaluation.

## 再生成

SVG生成只需Python标准库；PNG通过本目录PowerShell脚本与Windows自带System.Drawing绘制，不需Pillow、浏览器、网络或GPU。英文使用Arial，中文使用Microsoft YaHei；SVG保留文本和路径以便排版修改。PNG以2倍像素、300dpi元数据导出；SVG为推荐的论文排版源。

```powershell
python -B docs/figures/sgla_search_state_v4/render_method_framework.py --rasterize
```

[绘图源文件](render_method_framework.py) · [PNG导出器](rasterize_svg.ps1) · [Stage-113协议](../../experiments/06_sgla_uav_tracking/STAGE113_FROZEN_SHARED_CLOSED_LOOP.md) · [Stage-120协议](../../experiments/06_sgla_uav_tracking/STAGE120_SUPPORT_APPEARANCE_COMMIT_FACTORIAL.md) · [Stage-120结果](../../experiments/06_sgla_uav_tracking/STAGE120_RESULTS_REVIEW.md) · [历史PPO图v3](../sgla_search_state_v3/README.md)

本次仅新增框架图与说明，不改实验代码、冻结模型或结果。
