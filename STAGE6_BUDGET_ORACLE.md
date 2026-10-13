# Stage-6：序列级预算 Oracle

Stage-5 表明，30% 与 50% 周期 L9 在总体精度上没有显著差异，但两者在不同序列上的
胜负差异明显。本阶段先回答一个比训练新路由器更基础的问题：若能为每个序列选择不同的
完整闭环预算，在相同的全数据集候选块预算下，理论上还能获得多少精度。

该实验完全复用 Stage-5 的闭环 CSV，不需要 GPU，也不重新运行 tracker。

## 1. 优化问题

对序列 `i` 从以下策略中选择一个：

```text
Fast, Periodic-10, Periodic-20, Periodic-30, Periodic-50, SGLA
```

Oracle 最大化官方序列等权 AUC：

```text
maximize  mean_i AUC(i, selected_policy_i)
```

同时约束所有 tracked frames 上的候选块执行总次数：

```text
sum_i candidate_blocks(i, selected_policy_i)
    <= budget * sum_i tracked_frames(i)
```

实现采用精确的多选背包动态规划。长序列会消耗更多计算预算，但每个序列在 AUC 中仍然
等权，与 UAV123 的评测协议一致。

## 2. 运行

```bash
export STAGE6_DIR="$PERIODIC_DIR/stage6_budget_oracle"

python tracking/stage6_budget_oracle.py \
  --input "$PERIODIC_DIR/stage3_closed_loop_sequence_summary.csv" \
  --output_dir "$STAGE6_DIR" \
  --budgets 0 10 20 30 50 100
```

该命令输出：

- `stage6_fixed_policy_summary.csv`：应复现 Stage-5 各固定策略 AUC；
- `stage6_budget_oracle.csv`：各总预算下的 Oracle AUC、最优固定基线和额外 headroom；
- `stage6_budget_oracle_assignments.csv`：每个预算下 123 个序列的策略分配。

## 3. 完整性检查

1. 0% 预算只能选择 Fast，AUC 应为 `64.4089`；
2. 每一行 `candidate_blocks_per_frame` 不得超过名义预算；
3. `gain_vs_best_fixed_feasible` 不得为负；
4. 每个预算应恰好输出 123 个 assignment；
5. 100% Oracle 可以混合所有策略，因此可能高于固定 SGLA，但它仍然只是 GT 上限。

## 4. 结果判读

重点查看 30% 预算：

- 若相对固定 Periodic-30 仍有明显 headroom，说明不同序列确实需要不同刷新率，可继续做
  粗粒度预算控制器；
- 若 headroom 很小，则当前差异难以转化为方法收益，应保留 Periodic-30 并转向网络状态
  下的预算调整；
- assignment 的策略分布用于判断收益来自 Fast/低预算分流，还是少数困难序列获得高预算。

所有 Oracle 行必须标记 `uses_ground_truth=1`。它只能作为可学习性和方法上限证据，不能
作为可部署方法与现有 tracker 比较。
