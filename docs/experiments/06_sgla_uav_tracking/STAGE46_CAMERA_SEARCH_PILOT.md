# Stage-46：原生 SGLA 的相机运动补偿搜索原型

日期：2026-10-29。状态：用户已报告服务器48条完整轨迹完成，结果为 `REVIEW_CAMERA_SEARCH_PILOT`。

本轮原生/普通速度/仅相机补偿/相机加速度的AUC分别为63.4512、63.6449、64.8664、60.8608。仅相机补偿在`0176`和`0204`上大幅改善，在原稳定的`0304`上大幅退化，尚无稳定收益结论。后续为独立新增的 [Stage-46B 缓存时序审计](STAGE46B_SEARCH_SHIFT_AUDIT.md)，原实现与协议保留。

## 1. 本轮问题与改进点

无人机图像中的目标位移包含背景整体运动和目标相对运动。直接用自身预测的图像位移外推，可能混合两种运动。本轮检验：**从相邻帧估计背景变换，再预测搜索中心，能否给原生 SGLA 带来跟踪收益？**

主要基线是原始 SGLATrack；首帧模板、原生前缀、选择器、独立候选块、预测头和 Hann 解码均保持原样。代码直接调用原生 `tracker.track()`，只在调用前设置用于本帧裁剪的搜索父框。网络输出直接成为本组下一帧的历史预测。

这是相机补偿思想在现有 SGLA 管线上的小原型。已有工作中的运动补偿和运动预测为实现与对照提供基础；本轮以实际新增价值决定是否值得继续完善。

Stage-31 的简单历史候选关联未建立收益，Stage-43 的恢复规则也未建立收益。Stage-43 使用六层学生、自身历史运动和失效警报触发的额外搜索，没有相机补偿。本轮保留完整原生 SGLA，在每帧前向前修改搜索中心，用普通速度预测和仅背景补偿分别作对照。

Stage-27 的多数背景运动代理值较低，且许多失效起点的目标仍在裁剪内。因此不预设相机运动是所有失效的主因；同时测量估计质量和背景位移相对目标尺度，辨别本批视频是否提供了足够运动支持。

## 2. 固定四组与预算

| 方法 | 本帧搜索中心 |
| --- | --- |
| `sgla` | 上一帧本组预测框中心；原生基线 |
| `velocity` | 上一帧中心 + 自身图像位移的指数平滑速度 |
| `camera_only` | 用本帧背景变换映射上一帧中心；估计不可靠时保持上一帧中心 |
| `camera_velocity` | 背景变换后的中心 + 映射到当前坐标系的目标相对运动速度；估计不可靠时退回本组普通速度预测 |

四组均使用各自预测的宽高、原搜索倍率、首帧模板和原网络。只改中心，不扩大裁剪，不重排候选、不替换网络输出、不用另一组父框或 GT 重置。每帧实际执行六个前缀块、一个原生选中的候选块、一次选择器和一个预测头；通过模块 hooks 核对实际调用次数。

设上一帧自身中心为 `c`，背景变换为 `M(x)=Ax+b`，普通速度为 `v`，相对速度为 `r`：

```text
velocity:         proposed_center = c + v
camera_only:      proposed_center = M(c)
camera_velocity:  proposed_center = M(c) + A*r
```

网络输出新中心 `c_new` 后，按固定系数 0.5 更新：

```text
v_new = 0.5*v + 0.5*(c_new-c)
r_new = 0.5*A*r + 0.5*(c_new-M(c))
```

背景估计失败时，相对速度清零；原生组始终保持原父框。其他组的中心位移上限固定为上一帧预测框等效边长 `sqrt(w*h)` 的 0.5 倍，再将中心限制在当前图像内。该限制来自固定的保守原型设计，不根据结果搜索。

## 3. 背景估计与因果输入

只使用上一帧图像、当前图像和本组上一帧预测框。首帧给定的初始化框允许使用；之后没有 GT 进入估计器、控制器或跟踪器。

1. RGB 转灰度，最长边最多 640 像素，记录实际横纵缩放比例。
2. 在上一帧屏蔽以自身预测框为中心、宽高各放大2倍的区域；从其余位置最多提取500个角点。
3. 金字塔 LK 前后向追踪，前后向误差≤1.5个缩小图像像素，至少30条有效轨迹。
4. `estimateAffinePartial2D` 做 RANSAC：阈值2像素；至少20个内点、内点比例≥0.6、内点残差中位数≤1.5像素，并覆盖4×4网格中的至少4格。
5. 相邻帧拟合尺度在[0.9,1.1]、旋转绝对值≤10度、图像中心位移≤对角线的15%，才接受估计。失败时返回单位变换并记录原因。
6. 用实际横纵缩放比例将完整变换还原到原图坐标，正确处理缩放取整；相对速度也随线性部分映射。

**不能复用 Stage-27 缓存中的 motion 作为在线输入**：那个诊断曾使用 GT 遮罩。本轮四组都重新从图像估计，遮罩只来自自身预测。非相机组也计算影子估计，便于比较算法和分析背景支持；它不影响原生输出。

估计到的是主导背景的部分仿射运动，并非经过相机真值验证的位姿。动态背景、视差、遮挡和滚动快门都可能使这个近似失效。

接口依据：[OpenCV LK 文档](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)、[OpenCV 几何变换估计文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)。

## 4. 数据与结果角色

- 复用完整 Stage-27 的24个已消耗开发源组，按 `seed=2127` 与视频名的 SHA 固定顺序选12个完整视频，不按AUC、事件或运动量筛选。
- 不训练新模型，不下载检测数据；继续使用现有 VisDrone-SOT、原 SGLA 权重和 `sglatrack-test` 环境。
- 共48条完整视频轨迹。四种方法按视频轮换执行顺序，每次相机估计按视频/帧固定随机种子，保证断点重跑不受先前视频影响。
- 这12个视频之前已经影响研究判断，不是新的 checking，也不是外部验证。源前缀仍是源组代理，不能当作物理采集身份的严格证明。
- 只有新产生的自身预测决定状态。来源审计会读取冻结标签来核对旧轨迹；本轮 actor 的 API 只接收图像记录和首帧框。对本轮预测的 GT 评价在该视频预测完成后进行，标签不进入控制器。

## 5. 服务器命令

通过现有 Git 流程同步本轮新增文件后，在服务器运行。无需脚本包；本地改动不会自动提交推送。

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
export PYTHONDONTWRITEBYTECODE=1
export STAGE27_ROOT="$PWD/output/stage27/uav_failure_diagnostic_v1"
export STAGE46_ROOT="$PWD/output/stage46/sgla_camera_search_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
test -f "$CHECKPOINT"

python -B tracking/test_stage46_motion.py

python -B tracking/stage46_motion_experiment.py --mode prepare \
  --input "$STAGE27_ROOT/cache" --output_dir "$STAGE46_ROOT"

cat "$STAGE46_ROOT/manifest.json"

python -B tracking/stage46_motion_experiment.py --mode smoke \
  --input "$STAGE46_ROOT" --checkpoint "$CHECKPOINT" \
  --parameter deit_distilled --gpu 0

cat "$STAGE46_ROOT/smoke/stage46_smoke_decision.json"
```

`prepare` 接受 Stage-27 根目录或其 `cache`，核对完整的原始24条轨迹、来源和源码哈希，再固定本轮计划。新输出必须独立且为空，或已有完全相同的 manifest。权重、配置和原始运行环境需要与 Stage-27 一致。

smoke 使用首个选中视频最多64个跟踪帧、四种方法，检查原生框/响应/选层复现、连续状态和真实网络调用。通过条件只说明工程运行正确；静态或低运动 smoke 不等于方法有效。合成图像单元测试覆盖非零运动、缩放坐标与估计失败回退。

通过 smoke 后运行全部视频并汇总：

```bash
python -B tracking/stage46_motion_experiment.py --mode evaluate \
  --input "$STAGE46_ROOT" --checkpoint "$CHECKPOINT" \
  --parameter deit_distilled --gpu 0

python -B tracking/stage46_motion_summary.py --input "$STAGE46_ROOT"

cat "$STAGE46_ROOT/analysis/stage46_decision.json"
cat "$STAGE46_ROOT/analysis/stage46_policy_summary.csv"
cat "$STAGE46_ROOT/analysis/stage46_paired.csv"
cat "$STAGE46_ROOT/analysis/stage46_video_influence.csv"
```

断点后重复原命令即可。已提交的视频按 NPZ 原始字节 SHA 与 receipt 核验后复用，未提交的整段视频重跑。不要并发写同一 root；异常退出遗留 `collect.lock` 时，确认其中 PID 已退出再删除锁。`--sequence` 可运行一个已选中的完整视频；汇总需要全部48条结果。

```bash
python -B tracking/stage46_motion_experiment.py --mode status --input "$STAGE46_ROOT"
```

不应重跑或改写 Stage-27；它是本轮的冻结来源。若原生复现失败，先排查权重、配置、版本与路径，保留原容差。

## 6. 预先确定的分析

主比较为 `camera_velocity−velocity`，回答相机补偿在普通运动预测上是否有新增价值；同时报告相对原生 SGLA 的完整视频收益。`camera_only−sgla`、`velocity−sgla` 与四组交互作为组件对照。交互描述整个闭环策略，不能唯一归因于某一帧或某个层。

指标沿用原跟踪协议：视频宏平均 AUC、IoU<0.2比例、失败事件、最长失败持续比例、原生 IoU≥0.5片段的 AUC。后者是离线可靠片段诊断，不是在线开关。先算各视频的差，再按12个视频重采样5000次，种子2128。报告区间、输赢、去掉任一视频后的均值和逐视频影响。

逐视频表另报搜索父框中心误差、相对本组上一帧位置的中心误差减少量，以及实际裁剪的 GT 覆盖率。这些全部在预测完成后计算，用于解释中心预测是否改善，不能进入在线控制。原生复现同时要求浮点框误差在原容差内，且官方截断为整数后的框完全相同，避免微小浮点差异偷偷改变基线 AUC。

背景支持按原生轨迹定义：有效拟合至少50%的帧，且有效拟合的背景中心位移中位数≥上一帧预测目标边长的0.05倍；至少4个视频满足才认为本批有基本运动支持。该规则不用于重新挑视频，也不进入控制器。

实践参考为主比较和相对原生 AUC 都至少+0.5点，平均低IoU比例不增加，且至少有一个原生稳定视频可供检查、没有为原稳定视频引入失败。满足时标记 `PROMISING_CAMERA_SEARCH_PILOT`；支持不足为 `LIMITED_BACKGROUND_MOTION_SUPPORT`；其他结果为 `REVIEW_CAMERA_SEARCH_PILOT`。这些标签用于开发决策，不是新方法认证。区间较宽时保留不确定性，不能仅因跨零就否定整个方向。

若补偿有价值，下一步研究局部运动可靠性与不同运动条件的独立迁移；若普通速度预测更有用，先保留简单方案，再分析补偿错误；若两者都损伤正常跟踪，重点检查错误历史和裁剪偏移。只在当前开发证据上完善解释；使用新增检查视频前，先固定后续设计和样本方案。

时间分别记录网络与带背景估计的整个管线，后者包括灰度处理、LK/RANSAC、控制器与网络/解码，排除读取/哈希图像和首10帧。原生组也有影子运动计算，其管线时间不能当作未经仪表化的原生部署速度。实际前向数与耗时分别报告。

## 7. 本地验证与保留

8项本地 CPU 测试通过：已执行真实 OpenCV 光流/仿射估计、带缩放取整的坐标还原、背景遮罩、低纹理/尺寸变化回退、速度坐标与位移上限、实际原生 backbone `forward_test` 定义的6+1调用检查，以及四组采集/续跑/汇总与文件篡改拒绝。全流程测试使用合成图像和跟踪器夹具，不代表已加载完整 SGLA 权重运行 GPU。

保留 manifest、run、smoke、所有逐视频 NPZ/receipt 与 analysis 表。Stage-1～44 的原始代码、权重、哈希与结果不变。Stage-45 检测支线继续暂不执行。
