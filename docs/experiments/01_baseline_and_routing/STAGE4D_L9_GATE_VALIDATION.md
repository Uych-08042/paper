# Stage-4D：Fast/L9 单动作收益门控

Stage-4C 的诊断已经确认两件事：

1. 原 SGLA selector 在当前 checkpoint、UAV123 和 GOT-10k pilot 上都实际选择 L9；
2. 六输出 Ridge 几乎也总选择 L9，但其超参数和 AUROC 标签仍按“六层中任一层有益”定义。

因此本阶段不再同时学习“是否深算”和“选哪一层”，而只验证一个可证伪问题：

```text
在运行 L9 之前，能否预测 delta_L9 = IoU_L9 - IoU_Fast？
```

脚本只读取已审计的 Stage-4B NPZ，不修改 tracker，不重新运行 GPU 特征采集。
输出仍是冻结轨迹上的 current-state 诊断，不是最终闭环 tracking AUC。

## 1. 四种输入边界

| 版本 | 当前帧输入 | 决策位置 | 深算帧额外代价 |
|---|---|---|---|
| `history_only` | 上一帧状态、Fast 框和动作 | 当前帧视觉计算前 | L9 block |
| `token_only` | 历史 + L6 池化 token | Fast head 前 | L9 block |
| `fast_only` | 历史 + 当前 Fast score/box | Fast head 后 | L9 block + 第二次 head |
| `token_fast` | token + Fast score/box | Fast head 后 | pooling + L9 block + 第二次 head |

原 selector 特征不再加入，因为 Stage-4C 已测得它对 Ridge 的系数范数约为
`1.16e-26`，且 `token_selector` 与 `token_only` 的预测数值相同。

## 2. 三类简单监督目标

- `signed`：直接回归 `delta_L9`，与最终 IoU 收益完全一致；
- `positive`：回归 `max(delta_L9, 0)`，强调可获得的正收益幅度；
- `benefit`：用 Ridge 线性概率基线预测 `delta_L9 > epsilon`，默认测试
  `epsilon = 0, 0.005, 0.01`。

每个模型的 Ridge alpha 只在训练视频内部的视频级 calibration split 上选择，
随后在全部 30 个训练视频上重拟合，并只评估 10 个未见验证视频。

## 3. 运行

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

export STAGE4_DIR="$PWD/output/stage4"
export PILOT_CACHE="$STAGE4_DIR/pilot_cache_v1"
export L9_GATE_DIR="$STAGE4_DIR/l9_gate_pilot_v1"

python -B -m unittest discover -s tracking -p 'test_stage4*.py' -v

python tracking/stage4_l9_gate_train.py \
  --input "$PILOT_CACHE" \
  --output_dir "$L9_GATE_DIR" \
  --variants history_only token_only fast_only token_fast \
  --objectives signed positive benefit \
  --benefit_epsilons 0 0.005 0.01 \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.0001 0.001 0.01 0.1 1 10 100 1000 \
  --calibration_fraction 0.2 \
  --budgets 0 10 20 30 50 100 \
  --profile_dirs \
    "$STAGE4_DIR/profile_seq0" \
    "$STAGE4_DIR/profile_seq1" \
    "$STAGE4_DIR/profile_seq2" \
  --seed 2026
```

若目录已经存在，换一个新目录；只有明确重算同一实验时才追加 `--overwrite`。

## 4. 输出

```text
l9_gate_pilot_v1/
  stage4_l9_gate_alpha_selection.csv
  stage4_l9_gate_summary.csv
  stage4_l9_gate_budget.csv
  stage4_l9_gate_val_predictions.csv
  stage4_l9_gate_<variant>__<objective>.npz
  stage4_l9_gate_experiment.json
```

重点查看：

```bash
cat "$L9_GATE_DIR/stage4_l9_gate_summary.csv"
cat "$L9_GATE_DIR/stage4_l9_gate_budget.csv"
cat "$L9_GATE_DIR/stage4_l9_gate_alpha_selection.csv"
```

`budget.csv` 的 `global_score_exact_topk_offline_diagnostic` 使用整个验证 horizon
的预测分数排序，但不读取验证标签来选帧。它是严格同预算排序诊断，不是可直接部署的
在线预算控制器。

## 5. 判读顺序

1. `passes_constant_calibration_baseline` 必须优先看。若为 0，模型在 calibration
   上连训练均值常数都没有超过，不应进入闭环。
2. 看 `l9_benefit_auroc`、`score_delta_pearson`，确认学到的是 L9 的真实收益，
   不再使用“任一候选层有益”的错位标签。
3. 在 10%、20%、30% 预算看 `lift_over_random_l9`。它必须稳定为正；只看
   `gain_vs_fast` 会把“L9 本来平均就有益”误当成路由能力。
4. 看 `l9_oracle_budget_cap_gain` 与 `oracle_cap_recovery`，区分候选动作上限不足
   与排序器不足。
5. 若多个版本接近，优先 `history_only` 或 `token_only`。`fast_only` 和
   `token_fast` 在深算帧需要第二次 head，必须用更大的准确率收益补偿延迟。
6. `natural_gate_gain_vs_fixed_l9 > 0` 表示模型能够跳过部分有害 L9 帧，才有机会
   在少算的同时超过当前全帧 SGLA/L9。

Pilot 的积极信号建议至少满足：calibration 超过常数、验证 AUROC 明显高于 0.5、
10%-30% 三档均优于随机 L9。若这些条件仍不满足，就应停止扩大当前线性逐帧路由器，
转向时序窗口、闭环稳定性或候选框融合，而不是直接增加模型复杂度。
