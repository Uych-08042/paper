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

## 5. 时间迁移审计

完整序列 Oracle 可能利用了事后 GT 选择。训练控制器前，先检查序列前段显示出的策略
偏好能否迁移到不重叠的后段。该诊断分别使用前 25% 和前 50% tracked frames 的 GT AUC
进行预算分配，再在剩余帧上评价所选策略。

```bash
export STAGE6_STABILITY_DIR="$STAGE6_DIR/temporal_stability"

python tracking/stage6_temporal_stability.py \
  --input "$PERIODIC_DIR" \
  --output_dir "$STAGE6_STABILITY_DIR" \
  --fractions 0.25 0.5 \
  --budgets 10 20 30 50 \
  --bootstrap_samples 5000 \
  --seed 2026
```

重点查看 30% 预算的：

- `gain_vs_best_fixed_late` 及其序列 bootstrap 区间；
- `recovery_of_late_oracle_headroom`；
- `assignment_exact_match_rate`；
- 前段分配与后段 Oracle 的策略分布。

该实验仍使用前段 GT，且直接评价各策略既有的完整闭环轨迹。因此它不是实际的在线切换
结果，而是一项必要条件检查：若连前段 GT 都无法带来后段提升，普通因果特征更不可能稳定
实现完整序列 Oracle；若迁移明显成立，再进入 GOT-10k 上的早期特征预算控制器训练。

## 6. 无 GT 的 OOF 因果特征路由

时间迁移审计表明应优先验证稀疏预算。下一步只使用前段 Fast 轨迹中的预测框，提取运动、
尺度变化、连续框重叠、加速度和抖动等 33 个特征。目标标签是各策略在不重叠后段相对
Fast 的 AUC 收益。所有预测均采用序列级 5 折 OOF Ridge；正则系数只在每个外层训练折
内部选择，测试序列不会参与特征标准化、参数选择或拟合。

主实验先固定 10% 共享候选块预算：

```bash
export STAGE6_ROUTER_DIR="$STAGE6_DIR/causal_router_full_v1"

python tracking/stage6_causal_router.py \
  --input "$PERIODIC_DIR" \
  --output_dir "$STAGE6_ROUTER_DIR" \
  --fractions 0.25 0.5 \
  --budgets 10 \
  --folds 5 \
  --inner_folds 4 \
  --ridge_alphas 0.01 0.1 1 10 100 1000 \
  --bootstrap_samples 5000 \
  --seed 2026
```

该命令不需要 GPU，并输出：

- `stage6_causal_router_summary.csv`：OOF 路由与各对照的总体结果；
- `stage6_causal_router_assignments.csv`：每个预算下的序列策略分配；
- `stage6_causal_router_predictions.csv`：每个序列/策略的 OOF 预测收益与真实后段收益；
- `stage6_causal_router_features.csv`：只含 Fast 前段轨迹的可审计特征；
- `stage6_causal_router_folds.csv`：外层折测试误差与所选 alpha；
- `stage6_causal_router_alpha_selection.csv`：内层 OOF 的 alpha 选择结果。

`mean_prior_auc` 是只使用外层训练集平均策略收益、不使用轨迹特征的强对照；
`early_gt_transfer_auc` 是前段 GT 分配上界；`late_gt_oracle_auc` 是后段事后上界。只有当
10% 预算下 `gain_vs_best_fixed` 的 95% 区间完全大于 0，且路由相对 `mean_prior_auc`
也有稳定收益，才值得实现真正的中途切换闭环。否则保留 Periodic-L9 作为视觉策略，将
自适应重点放到网络状态和系统预算上。

虽然输入特征严格因果且预测为 OOF，当前后段结果仍来自每个策略从首帧独立运行得到的
既有闭环轨迹，并且背包求解使用已知后段长度计算成本。因此这仍是离线迁移诊断，不能
直接作为可部署控制器结果。
