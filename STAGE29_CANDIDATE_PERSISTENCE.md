# Stage-29：一次空间候选纠正能否产生持续改善

Stage-28 已完成 24 视频、522 帧、11 个失效起点的响应审计。三个起点的加权前五中存在准确目标框：`uav0000173_00781_s`、`uav0000204_00000_s`、`uav0000235_00001_s`。本轮验证一次正确空间选择的后续价值，帮助决定是否开发身份保持模块。

这是一轮 GT 辅助、冻结原始 SGLA 的开发机制实验。它没有训练身份模块或 PPO，不构成新方法、完整视频 AUC、外部泛化或速度结果。

## 固定设计

1. 覆盖 Stage-28 全部 11 个事件。用同一套原始加权响应局部峰与框 NMS 规则产生最多五个候选；候选不补齐，也不追加 GT 框。
2. 为每个候选独立初始化原始首帧模板，恢复失效起点之前已记录的预测父框，重放该起点的原始输出。
3. rank 1 是原始首选，不替换输出；rank 2 及之后仅在该起点将对应候选框写入状态一次。
4. 各分支随后使用自身预测裁剪连续运行 40 帧。继续执行原始 SGLA 的选择器与候选层，不固定 L9，不采用 GT 更新、再纠正或候选重排。
5. 每个事件的 rank 1 分支须逐帧复现 Stage-27 的原始框、选层、裁剪缩放、响应峰值与整数框 IoU。替换分支则核验 `previous[t+1] == boxes[t]`，并且未来输出均等于实际模型解码结果。

根据用户提供的 Stage-28 候选数，本轮应有 **33 个分支，1320 帧后续跟踪和 33 次起点重放**，另有模板和原始网络一致性核验开销。计划会从真实缓存重新计算这些数字。

每条分支保存 onset 到 onset+40，共 41 个时间点；所有主指标排除 onset。不会因视频边界缩短 40 帧，若数据不够会直接停止。

## 主分析与门槛

主分析案例按 Stage-28 的起点数据预先固定：可用裁剪覆盖率≥0.9、原始输出 IoU<0.2、加权前五至少一个候选 IoU≥0.5。必须恰好有三个事件，且来自三个 source group。其余事件同样采集和报告。

每个主分析案例的“纠正候选”取**起点 GT IoU 最高**的已有候选；相同 IoU 按原响应排名打破平局。它在后续轨迹产生之前固定，不能用后续回报最高的分支代替。

进入下一轮开发的门槛保持为：三个主分析事件中至少两个同时满足：

- 起点之后第 1–40 帧平均 IoU，相对原始首选分支提高至少 **0.05**；
- 同一 40 帧内的 IoU<0.2 帧数减少。

门槛不依赖被直接纠正的起点 IoU。它是小规模可行性规则，不是统计显著性检验；40 帧不是 40 个独立视频。

额外报告前 20 帧、后 20 帧的平均 IoU、局部窗口 AUC、覆盖率、连续三帧 IoU≥0.5 的恢复起点，以及恢复后连续三帧 IoU<0.2 的再次失效起点。恢复/再失效的时间均相对被干预帧计算，最早计入的未来帧为 +1。

枚举全部候选还能给出“候选均匀随机选择”的精确期望，以及排除原首选后的均匀随机期望；候选只有一个时后者为空。这些只作辅助比较，不参与主门槛。

## 仓库同步与服务器命令

按现有 Git 流程同步本轮新增文件。原始 Stage-27、Stage-28 的脚本和数据保持冻结。

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"

export STAGE28_ROOT="$PWD/output/stage28/response_competition_v1"
export STAGE29_ROOT="$PWD/output/stage29/spatial_candidate_persistence_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/test_stage29_candidate.py

python tracking/stage29_candidate_collect.py \
  --input "${STAGE28_ROOT:?}" \
  --plan_only \
  --output_dir "${STAGE29_ROOT:?}"

cat "$STAGE29_ROOT/stage29_plan_summary.csv"
```

如果已有 `CHECKPOINT` 指向其他实际位置，使用原来的路径；内容必须与 Stage-27 的原始权重哈希一致。CPU 计划会重新核验全部 Stage-28 缓存、完整报告及其 Stage-27 来源，不初始化 CUDA。

确认计划为 11 事件、33 分支、3 个主分析事件后，先运行一个工程案例：

```bash
python tracking/stage29_candidate_collect.py \
  --input "${STAGE28_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --sequence uav0000173_00781_s \
  --output_dir "${STAGE29_ROOT:?}"

python tracking/stage29_candidate_summary.py \
  --input "${STAGE29_ROOT:?}" --allow_incomplete
```

工程案例须通过原轨迹重放和分支状态连续性核验。它只会给出 `PILOT_INCOMPLETE`，不能据单个事件认定本轮通过或失败。

完成同一冻结计划：

```bash
python tracking/stage29_candidate_collect.py \
  --input "${STAGE28_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --output_dir "${STAGE29_ROOT:?}"

python tracking/stage29_candidate_summary.py \
  --input "${STAGE29_ROOT:?}"

cat "$STAGE29_ROOT/analysis/stage29_decision.json"
cat "$STAGE29_ROOT/analysis/stage29_primary_events.csv"
cat "$STAGE29_ROOT/analysis/stage29_event_summary.csv"
```

同一目录只运行一个采集进程。完成的事件缓存会审计并复用；中断在一个事件内部时，该事件的全部候选重新采集。正常退出会释放 `collect.lock`；强制终止后的遗留锁应先确认无进程写入再处理。不要删除来源数据或修改其指纹绕过校验。

## 输出与解释

| 文件 | 内容 |
| --- | --- |
| `plan.json`、`stage29_plan_summary.csv` | 固定事件、全部候选及起点标签、主分析资格、H40 与代码/来源指纹 |
| `run.json` | 原始模型运行身份、一次替换与原生延续协议 |
| `branches/*__stage29_candidate.npz` | 每个候选分支的框、真实模型输出、父框、裁剪、选层、IoU 和替换掩码 |
| `analysis/stage29_decision.json` | 完整性、三事件门槛结果、原轨迹重放误差 |
| `analysis/stage29_primary_events.csv` | 三个预先固定的主分析案例 |
| `analysis/stage29_event_summary.csv` | 全部事件的原始首选与起点 GT 候选的后续结果 |
| `analysis/stage29_branch_summary.csv` | 所有候选分支的局部窗口指标 |
| `analysis/stage29_step_effects.csv` | 包含 onset 的逐帧 IoU、覆盖率、父框尺度比和实际选层；标明 onset 不属于主指标 |

不完整报告写入 `analysis_pilot`，完整报告写入 `analysis`。其他事件若没有 GT 匹配候选，纠正候选和相关差值为空，仍保留全部分支结果，不把它们混入三事件门槛。

- **通过**：说明在这三个已知开发事件中，一次可用候选选择具有持续改善空间。下一关需要不使用 GT 的轻量身份判别，并在完整闭环视频中检验其收益及误纠正代价。
- **未通过**：检查是候选无法持续保住身份，还是很快再次漂移；不能仅凭原始 top-5 上限继续扩大训练。

本轮的触发时刻和纠正标签均使用已知失效与 GT，只能支撑该固定模型、这些开发状态下的一次输出干预结论。它尚未验证在线触发、遮挡原因、压缩性能或新设备运行效率。

## 验证范围

CPU 合成测试覆盖：原始首选精确重放、候选分支自身状态连续性、只替换一次、未来 GT 标签不改变预测、当前帧不计主指标、恢复/再失效时间、0.05 与低 IoU 帧数的双门槛、不完整数据不能通过、H40 不截断、输入报告/指纹防混用、原始缓存保持不变、命令行计划与汇总。实际原始 SGLA 的 GPU 重放仍由服务器数据检验。
