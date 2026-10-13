# Stage-5：周期 L9 闭环基线

Stage-4F 表明，在逐视频因果预算下，当前 history-only 和 fast-only 门控都不能稳定超过
相同深算次数的周期策略。fast-only 还需要重复执行 tracking head，在延迟上被简单周期
L9 明显支配。因此本阶段不再验证 learned gate，而是把周期 L9 放入真实闭环轨迹。

本实验回答：冻结轨迹中观察到的 L9 单帧收益，能否在策略自身不断更新搜索区域的条件下
转化为长期跟踪收益。

## 1. 策略

- `fast`：所有帧停在 L6；
- `sgla`：原始选择器；当前检查点实际选择 L9；
- `periodic_l9_10`：每 10 个决策运行一次 L9；
- `periodic_l9_20`：每 5 个决策运行一次 L9；
- `periodic_l9_30`：任意前缀恰好使用 `floor(0.3 * N)` 次 L9；
- `periodic_l9_50`：每 2 个决策运行一次 L9。

每个策略维护自己的 tracker state。周期策略在非深算帧只执行 L6 head，在深算帧只执行
L9 head，不会像 fast-only gate 那样重复执行两个 head。

## 2. 单序列检查

```bash
export PERIODIC_SANITY="$PWD/output/test/tracking_results/sglatrack/deit_distilled/stage5_periodic_l9/uav123_sanity"

python tracking/stage3_closed_loop_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --sequence 0 \
  --gpu 0 \
  --policies \
    fast sgla \
    periodic_l9_10 periodic_l9_20 periodic_l9_30 periodic_l9_50 \
  --output_dir "$PERIODIC_SANITY" \
  --latency_warmup_frames 20 \
  --verify_sgla \
  --overwrite

python tracking/stage3_closed_loop_summary.py \
  --input "$PERIODIC_SANITY" \
  --stage2_input "$OUTPUT_DIR"
```

检查输出应包含 6 个 CSV，每个 3085 行。`blocks/f` 应约为 0、1、0.1、0.2、0.3、0.5，
前 20 个 tracked frames 会正常更新状态，但不进入延迟均值。周期策略延迟应位于 Fast 与
SGLA 之间；单序列仍可能受 GPU 频率波动影响，最终延迟以全量均值和独立 benchmark 为准。

## 3. UAV123 全量实验

使用全新目录，并在相同预热协议下同时运行 Fast、SGLA 和四档周期策略。不要与旧的
Stage-3 延迟 CSV 混合。

```bash
export PERIODIC_DIR="$PWD/output/test/tracking_results/sglatrack/deit_distilled/stage5_periodic_l9/uav123"

python tracking/stage3_closed_loop_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --gpu 0 \
  --policies \
    fast sgla \
    periodic_l9_10 periodic_l9_20 periodic_l9_30 periodic_l9_50 \
  --output_dir "$PERIODIC_DIR" \
  --latency_warmup_frames 20

python tracking/stage3_closed_loop_summary.py \
  --input "$PERIODIC_DIR" \
  --stage2_input "$OUTPUT_DIR"
```

多 GPU 时，每个进程使用相同输出目录并设置不同的 `--gpu`、`--num_shards` 和
`--shard_id`。所有 shard 结束后再执行 summary。

## 4. 判读

1. 首先检查 AUC 是否随 10%、20%、30%、50% 预算大体单调增加；
2. 比较周期策略相对真实闭环 `fast` 的增益，而不是 Stage-2 counterfactual Fast；
3. 比较 `periodic_l9_50` 与 `sgla` 的 AUC 差距及实测延迟；
4. 使用 sequence summary 检查收益是否由少数视频贡献；
5. 任何后续 learned gate 都必须在相同 `blocks/f` 和延迟下超过该周期基线。

若周期 L9 能稳定修复 Fast 的累计漂移，后续方法应围绕“何时刷新”建模；若周期策略也
无效，则冻结轨迹中的单帧 oracle 收益不能转化为闭环收益，应停止该计算分配方向。

