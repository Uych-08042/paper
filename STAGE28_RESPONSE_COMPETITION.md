# Stage-28：冻结 SGLA 的空间响应竞争审计

这一步回答：Stage-27 的失效发生时，真实目标是否仍被 SGLA 表示为可定位的候选，只是在空间竞争或原始窗口加权后没有成为最终输出？它同时检查输入中目标变小与预测框膨胀的时间关系。

本轮使用原始 SGLA 权重和 Stage-27 已记录的预测状态。没有训练、PPO、修改窗口、GT 重置跟踪或新策略性能比较。两个已经人工查看的车辆错跟案例是动机；采样覆盖全部事件，不仅选这两个案例。

## 1. 固定设计

- 输入必须是完整、未混入其他实验的 Stage-27 原始 SGLA 缓存。
- 对全部失效事件，取 onset 前 10 帧至后 30 帧，包含端点，遇到视频边界截断。
- 每条视频另取跟踪时段的 1/4、1/2、3/4 位置作为固定时间对照，不按成功与否筛选。它们不一定是稳定跟踪帧。
- 按视频和帧号去重。当前 24 视频、11 事件的设计最多需要 523 次额外单帧前向；实际数量由 `--plan_only` 输出。
- 原始响应图与原始 Hann 窗口乘积分别排名。原始预测仍由原实现产生。
- 空间候选：3×3 局部峰、分数降序、相同分数按格点行优先；对解码后的框做连续坐标 IoU>0.5 的 NMS，最多保留 5 个，不足 5 个不补齐。
- GT 匹配：沿用 Stage-27 的整数预测框、包含边界像素的 IoU 定义，IoU≥0.5；IoU<0.1 记作非目标框，中间区间保留为不确定。
- 同时检查全部响应格点，区分“存在真目标解码框但未成为局部峰/未进入前五”与“所有格点均无法定位真目标”。

`plan.json` 绑定原始运行、manifest、全部轨迹文件哈希、采样设计和采集代码。`run.json` 另绑定权重、运行配置与实际窗口。恢复时重新核对，不覆盖不一致的缓存。

## 2. 重放如何保持原轨迹

每个视频只用原始第一帧及初始框初始化模板。每个审计帧的裁剪父框直接读取 Stage-27 的 `previous` 预测框；随后运行原始前缀、选择器、候选层和预测头。

每帧核对原始选层、裁剪缩放、响应峰值、输出框和官方整数框 IoU。另将所有 CENTER 格点按原始浮点父框中心解码，检查加权最大值处的框与原实现一致。框误差上限为 0.001 原图像素，响应峰值误差上限为 1e-5。用于 GT 统计的原始加权 top-1 框直接保留原解码结果，避免 NumPy 舍入改变整数框 IoU。

原始响应 argmax 和其他候选的框只用于离线诊断，均不写回后续帧。GT 只用于初始化及前向结束后的匹配；中途不用于裁剪、选层或选框。

## 3. 仓库同步与服务器运行

按已有 Git 流程提交、推送本轮文件，在服务器拉取同一版本。不需要下载脚本包。以下命令在服务器仓库根目录执行；如之前的 checkpoint 路径不同，保留实际路径。

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
export STAGE27_ROOT="$PWD/output/stage27/uav_failure_diagnostic_v1"
export STAGE28_ROOT="$PWD/output/stage28/response_competition_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/test_stage28_response.py

python tracking/stage28_response_collect.py \
  --input "${STAGE27_ROOT:?}/cache" \
  --plan_only \
  --output_dir "${STAGE28_ROOT:?}"
```

先检查计划输出应为 24 视频、11 事件，且额外重放帧数不超过 523。`--plan_only` 会核对原始轨迹和数据文件，不初始化 CUDA、不生成预测。

先重放已经查看过的一个案例，确认工程一致性：

```bash
python tracking/stage28_response_collect.py \
  --input "${STAGE27_ROOT:?}/cache" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --sequence uav0000173_00781_s \
  --output_dir "${STAGE28_ROOT:?}"

python tracking/stage28_response_summary.py \
  --input "${STAGE28_ROOT:?}" \
  --allow_incomplete --render
```

没有重放不一致后，去掉单视频限制完成同一计划：

```bash
python tracking/stage28_response_collect.py \
  --input "${STAGE27_ROOT:?}/cache" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --output_dir "${STAGE28_ROOT:?}"

python tracking/stage28_response_summary.py \
  --input "${STAGE28_ROOT:?}" --render
```

`--render` 使用 CPU，需要 Pillow 和 OpenCV。省略该选项仍会导出完整数值报告。

完成的视频缓存会核验并跳过；中断在视频内部时，该视频重新重放。同一输出目录只运行一个采集进程。正常退出会释放 `collect.lock`；如果进程被强制终止，先确认没有采集进程在写入，再处理遗留锁。不要删除原始 Stage-27 文件或通过重建其指纹绕过一致性错误。

## 4. 看哪些结果

完整报告位于 `$STAGE28_ROOT/analysis`，不完整报告位于 `analysis_pilot`。

| 文件 | 内容 |
| --- | --- |
| `stage28_response_diagnostic.json` | 完整性、重放误差、指纹、覆盖范围 |
| `stage28_onset_responses.csv` | 每个失效起点的原始/加权 top-1 IoU、前五候选召回、全部格点中真目标排名及分数间隔 |
| `stage28_onset_observations.csv` | 各观察类型的事件数，不自动赋予语义失效原因 |
| `stage28_frame_responses.csv` | 所有去重审计帧及事件/对照成员关系 |
| `stage28_candidate_boxes.csv` | 原图坐标候选框、格点、排名、分数、GT IoU |
| `stage28_phase_summary.csv` | 起点、起点前后、固定时间对照等分组的 source-group 宏平均 |
| `stage28_source_raw_size_summary.csv` | 全部原始 Stage-27 帧的原图 GT 尺寸、父框与当前 GT 尺度比；即使 pilot 也覆盖全部原始视频 |
| `stage28_scale_temporal.csv` | 固定事件窗口内，投影尺寸变小、覆盖率下降、父框尺度增大的时间关系 |
| `index.html`、`heatmaps/*.png` | onset−1、onset、onset+1、onset+10 的实际裁剪视图、原始和加权热图 |

热图左侧是实际裁剪的展示缩放，绿框是 GT、红框是原始预测。热图中绿边格点表示该格点解码框与 GT 的 IoU≥0.5；数字为候选排名。每张响应图独立归一化颜色，不能把颜色跨图比较为概率。

优先回传：

```bash
cat "$STAGE28_ROOT/analysis/stage28_response_diagnostic.json"
cat "$STAGE28_ROOT/analysis/stage28_onset_observations.csv"
cat "$STAGE28_ROOT/analysis/stage28_onset_responses.csv"
```

并查看两个已知案例及其余事件的热图。

## 5. 如何决定后续方法

| 观察 | 后续需要验证的机制 |
| --- | --- |
| 真目标仍在加权前五、原始 top-1 跟错 | 候选身份关联或目标与非目标判别是否能改变空间竞争 |
| 原始 top-1 匹配 GT，加窗后不匹配 | 空间先验与运动变化的冲突；需另设实验，不能直接把去窗后的诊断当成完整跟踪增益 |
| 真目标框只存在于全部格点、未进入局部峰或前五 | 需要区分响应判别、候选生成与回归质量，不直接假设候选关联即可解决 |
| 裁剪内没有格点能解码出 GT 匹配框 | 当前候选关联缺乏可用输入，应审查定位表示、回归或遮挡/可见性 |
| 先错跟、后父框膨胀或 GT 离开裁剪 | 反馈放大是后续损失的候选机制，不能据投影尺寸小就认定原图小目标是起因 |

这轮不设“新方法通过”门槛，因为尚未训练或评估新方法。先报告全部 11 个事件的分布，再确定一个主导且可干预的机制，以及对应训练对照和闭环小规模验证门槛。单个成功案例或 GT 辅助候选选择不能代替这一过程。

额外限制：非目标解码框可能来自错误回归，不自动等于真实干扰物；GT 几何有效也不自动代表目标可见。事件窗口是依据原始结果选取的诊断样本，不能作为独立外部验证。分组统计按 source-group 聚合后做 5000 次 bootstrap，单组不报区间；各时间分组可能重叠，不能当作互相独立的试验。

## 6. 本地验证范围

11 项测试覆盖采样计划、浮点父框映射、原始解码对照、窗口排名反转、局部峰漏选、NMS 与相同分数、缓存防混用、GT 标签不影响前向输出、汇总命令和热图输出。缺少 PyTorch 或 Pillow 时对应测试会明确跳过。实际 GPU 重放一致性必须由服务器数据检验；本地合成测试不构成任何真实失效机制结果。
