# Stage-75 四候选 PPO 失跟诊断与共同起点续跑

2026-11-07 固定方案，2026-11-08 收到 L20 结果。Stage-74 的四选一 PPO 对匹配监督平均提高 4.0722 AUC 点，0003 出现两次末尾短失败，0304 的 2172 仍从第 1090 帧失跟至终点。本轮在这些起点区分候选不足、选择错误和历史分岔，并测量改选一次的后续影响。 [Stage-74 结果复核](STAGE74_RESULTS_REVIEW.md) [Stage-75 L20 结果复核](STAGE75_L20_RESULTS_REVIEW.md)

## 固定起点

帧号从首个跟踪帧开始计数，初始化帧为 0。D 组使用既有首次失败起点，E 组使用相同事件前 8 帧。S01 使用 0304 的稳定种子，在失败种子的同一帧对照。0176、0307 使用各 PPO 种子的首次确认恢复起始帧，0204 使用 q0 轨迹的首次失败帧。载入完整缓存时重算对应事件，检查表中的参考帧。

| 起点 | 视频 | PPO 种子 | 帧 | 续跑帧数，含起点 |
|---|---|---:|---:|---:|
| D01 | 0003 | 2171 | 138 | 7 |
| D02 | 0003 | 2172 | 140 | 5 |
| D03 | 0304 | 2172 | 1090 | 32 |
| E01 | 0003 | 2171 | 130 | 15 |
| E02 | 0003 | 2172 | 132 | 13 |
| E03 | 0304 | 2172 | 1082 | 32 |
| S01 | 0304 | 2171 | 1090 | 32 |
| R01 | 0176 | 2171 | 388 | 32 |
| R02 | 0176 | 2172 | 321 | 32 |
| R03 | 0307 | 2171 | 474 | 32 |
| R04 | 0307 | 2172 | 471 | 32 |
| R05 | 0204 | 2171 | 664 | 32 |
| R06 | 0204 | 2172 | 664 | 32 |

起点和窗口是在已见 Stage-74 结果上固定的开发案例。13 个窗口可在同一视频内重叠。8 帧提前量是本轮选定的对照距离，各起点最多续跑 32 帧并在视频末尾截断。

## 缓存对照

每个起点从 q0、Q、future、双种子 PPO 和双种子四选一监督的真实轨迹中取同一帧，得到 7 个历史输入。对每个输入并列记录四候选 IoU、原轨迹动作、q0、缓存 Q 推荐、同折同种子 PPO 与监督重评分、以及当帧最高 IoU 的合法候选。重评分使用 Stage-74 既有标准化与权重，两套评分器在同一输入上各自输出四个动作概率。完全相同的输出框使用最小索引代表。

观察起点前最多 32 帧与续跑窗口内的原轨迹。用 IoU ≥ 0.5 定义良好候选，用 IoU < 0.2 标记低质量输出。分开记录四框均低、四框无良好候选、良好 q0 被换成低框、其他良好候选漏选，以及选中良好框。对照轨迹记录首个实际输出、搜索父框、裁剪内容及候选框差异，同时标出焦点窗口内 PPO 无良好候选而参考轨迹有良好候选的帧。这样的可用性差异用于定位历史分岔，具有共享起点的续跑用于检验单次动作的后效。

## 共同起点续跑

每例以 PPO 实际历史、原始首帧模板和冻结 SGLA V1 重新生成一次共享观测。校验原轨迹的裁剪、候选、特征、动作概率和输出。逐个选取起点的不同公开框作为输出，之后用同一折、同一种子的冻结 PPO 从各自预测历史继续。先核验原动作分支和 Stage-74 对应窗口逐帧一致，再采集其他动作分支。GT 用于事件复核和离线统计。

主要比较是每例强制 q0 相对原 PPO 动作的**后续帧平均 IoU 差**和**窗口低 IoU 帧数差**。表中也给出其他不同候选相对原动作的同样比较、当帧差、含起点平均差和末帧 IoU。若某例 q0 与原动作等价，比较为零。0003 的 5／7 帧结果按短窗口解释。若起点缺少后续帧，后续均值留空。

各候选接受相同后续控制器。一次改选有收益时，该结论适用于指定父框和短窗口。判断稳定保护还要看 R01～R06 对同样改选的代价。原生跟踪报告和旧四选一完整成绩用于背景对照，本轮的完成状态表示缓存及短续跑均已审计。

## 执行预算与证据

| 项目 | 预算 |
|---|---:|
| 视频及预定起点 | 5 个视频，13 个起点 |
| 锚点缓存重评分上下文 | 91 个 |
| 续跑分支上限 | 52 条，含原动作分支 |
| 共享观测 | 每例 4 次 SGLA 搜索 |
| 续跑搜索 | 每例 `4 × 合法不同框数 × (窗口长度 − 1)` |
| 总搜索及块调用上限 | 5092 次搜索，35644 次块调用 |
| 训练更新、完整新轨迹、在线恢复支线 | 0 |

使用完整的 Stage-74 封存运行及其上游绑定，包括模型、收据、NPZ 和 17 张结果表。阶段输入默认位于 `output/stage74/direct_four_candidate_ppo_v1`。新根为 `output/stage75/four_choice_failure_replay_v1`。缓存部分可以先在 NumPy 环境执行。续跑使用与来源匹配的 L40 软件、SGLA 配置和模型身份，物理 GPU UUID 可以在新运行开始时重新绑定。中断后使用同一绑定恢复已提交的分支。

默认结果包有 `stage75_diagnostic.json`、`stage75_anchors.csv`、`stage75_same_state.csv`、`stage75_divergence.csv` 和 `stage75_branches.csv`。完整缓存窗口、逐帧分支及各收据留在新根。正式搜索账按收据覆盖的共享观测和短分支计数。已完成的复核可直接重读收据和结果表。

## 服务器命令

在服务器仓库根目录及 `sglatrack-test` 环境中，先查看当前作业可见的 CUDA 逻辑编号。

```bash
python -B - <<'PY'
import torch
for index in range(torch.cuda.device_count()):
    device = torch.cuda.get_device_properties(index)
    print(index, device.name, getattr(device, "uuid", "unavailable"))
PY
```

例如输出 `1 NVIDIA L40 ...` 时，用下面的命令启动。将 `1` 替换为本次作业中 L40 的逻辑编号。若 `STAGE75_ROOT` 已有 `run.json`，续跑选用其中 `runtime.gpu_uuid` 对应的 L40。

```bash
mkdir -p output/stage75
STAGE75_GPU=1 nohup bash tools/run_stage75_ppo_failure.sh > output/stage75/run_stage75_l40.log 2>&1 < /dev/null &
echo $!
```

```bash
tail -f output/stage75/run_stage75_l40.log
```

脚本按顺序验证来源、输出缓存诊断、运行短分支并用 `cat` 生成 `docs/experiments/06_sgla_uav_tracking/results/stage75_results_*.txt`。可通过 `STAGE74_ROOT`、`STAGE75_ROOT`、`CHECKPOINT`、`STAGE75_GPU` 设置本机路径及设备。回传该 TXT 即可复核本轮。

若启动报 `requires NVIDIA L40, got NVIDIA L20`，原入口已经生成缓存诊断并停在设备检查。当前只有 L20 时，使用下方的独立入口。 [L20 服务器命令](#l20-服务器命令)

若先在 CPU 上检查缓存，可使用同一新根执行以下入口。缓存诊断完成后，完整脚本会继续到 L40 短续跑。

```bash
python -B tools/stage75_ppo_failure_experiment.py --mode cached \
  --input output/stage74/direct_four_candidate_ppo_v1 \
  --output_dir output/stage75/four_choice_failure_replay_v1
cat output/stage75/four_choice_failure_replay_v1/analysis/stage75_cached.json
```

## L20 服务器命令

服务器当次分到 L20，使用独立入口。L20 版读取同一份封存的 Stage-74、同样的 13 个起点、原始首帧模板和缓存 PPO 历史。每例先在 L20 上重算共享观测和冻结 PPO 的确定性动作，随后在同一张卡上运行该基准及所有不同输出框的续跑。比较仍使用后续平均 IoU 和窗口低 IoU 帧数，上限仍为 52 条短分支、5092 次搜索。启动检查要求原 SGLA 权重、配置和来源软件版本。

L40 缓存用于检查起点历史及裁剪，并记录两张卡之间的候选框、动作及基准续跑差异。`stage75_l20_device_bridge.csv` 记录这些差异，`stage75_l20_branches.csv` 的增益以 L20 重算的 PPO 分支为基准。若某起点出现动作或轨迹分歧，该例结论适用于相同缓存历史下的 L20 短窗口。原 L40 根和 Stage-74 输入各自独立保存。

同步新增脚本后，在服务器仓库根目录和 `sglatrack-test` 环境执行。默认使用 CUDA 逻辑编号 `0`，可用 `STAGE75_L20_GPU` 指定当前作业的 L20 编号。

```bash
mkdir -p output/stage75
nohup bash tools/run_stage75_l20_replay.sh > output/stage75/run_stage75_l20.log 2>&1 < /dev/null &
echo $!
```

```bash
tail -f output/stage75/run_stage75_l20.log
```

默认根为 `output/stage75/four_choice_failure_replay_l20_v1`，可用 `STAGE75_L20_ROOT` 指定新路径。部分运行以 `run.json` 中的 L20 UUID 绑定续跑，设备换卡后应指定新的输出目录。完成后回传 `docs/experiments/06_sgla_uav_tracking/results/stage75_l20_results_*.txt`。结果包包含 L20 诊断、跨卡差异、分支摘要及三张 L40 缓存对照表。

## 本机验证

3 项回归测试通过。测试使用小型真实 CENTER 的预测缓存和合成的上游训练记录，覆盖来源核验、原动作复现、中断恢复、GT 隔离与预算。13 个参考事件及窗口长度已与 Stage-74 的服务器回传表逐项核对。正式 L40 运行待执行。

L20 入口通过 2 项本机回归测试，覆盖 GPU 类型和 UUID 绑定、模拟数值漂移后的同卡基准、重复框掩码、裁剪一致性及收据续跑。服务器 L20 已完成 52 条短分支并回传。 [Stage-75 L20 结果复核](STAGE75_L20_RESULTS_REVIEW.md)
