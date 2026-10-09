# SGLATrack 第三阶段离线路由验证

## 1. 已知结论

Stage-2 已证明：

- 原 selector 在 UAV123 的 112,455 个跟踪帧上全部选择 L9。
- 原 SGLA 等价于固定 L9，AUC 为 `66.5143`。
- Best-of-6 候选层 Oracle AUC 为 `69.7411`。
- Best candidate 在 `80.87%` 的有效帧上优于 Fast。
- 当前 selector 的最佳候选命中率仅为 `27.55%`。

因此 Stage-3 验证：

> 在执行 L7-L12 之前，仅使用当前 L6 可观测信息，能否预测应当选择 Fast 或某一个候选 block？

动作空间为：

```text
Fast, L7, L8, L9, L10, L11, L12
```

## 2. 实验边界

Stage-3 直接读取已有 `*_stage2_candidates.csv`，不重新读取图像、不加载 checkpoint，也不需要重新执行 GPU 跟踪。

训练标签使用 GT IoU：

```text
utility_layer_k = iou_layer_k - iou_fast
```

预测特征严格限制为执行候选 block 前可获得的信息：

- L6 selector 的六维 sigmoid 输出。
- 六维输出的 mean、std、max、margin 和归一化 entropy。
- 与上一帧相比的 selector 输出变化。
- 当前 Fast bbox 相对上一帧 tracker state 的运动与尺度变化。
- 当前 Fast bbox 相对上一帧 Fast bbox 的变化。

以下字段绝不会进入模型特征：

- GT bbox 或当前帧 GT IoU。
- `iou_layer*`、`delta_layer*`。
- 候选层 bbox。
- `response_peak_layer*`。
- 当前帧 SGLA、confidence 或 ensemble 输出。

候选 IoU 只作为训练目标和测试评价值。

## 3. 严格序列级 OOF

脚本默认执行 5-fold out-of-fold 验证：

1. 以完整序列为最小分组单位。
2. 同一序列的所有帧只能位于一个 fold。
3. 每轮使用四个 fold 训练，剩余 fold 测试。
4. 汇总五轮测试预测，保证每个结果都来自未见过该序列的模型。

禁止随机拆分帧。相邻视频帧高度相关，随机帧划分会产生严重泄漏。

## 4. 两种验证模型

### Ridge

多输出 Ridge 同时预测六个候选层相对 Fast 的 utility。它速度快、无新增依赖，用于回答线性信号是否存在。

### MLP

小型 MLP 使用 Smooth-L1 utility 回归与辅助 7-action 分类损失。每个外层 fold 会从训练序列中再划分一部分序列用于 early stopping。测试序列从不参与标准化、训练或 early stopping。

MLP 只是可预测性探针，不是最终论文模型。

## 5. Best-of-7

Stage-2 的 Best-of-6 每帧必须选择一个候选层。Stage-3 另外计算：

```text
Best-of-7 = max(Fast, L7, L8, L9, L10, L11, L12)
```

若所有候选层都不优于 Fast，Oracle 会选择 Fast。因此 Best-of-7 AUC 必须不低于 Stage-2 的 `69.7411`。

## 6. 上传文件

只需要上传新增文件：

```text
tracking/stage3_router_validate.py
STAGE3_ROUTER_VALIDATION.md
```

不需要上传 Stage-2 CSV。服务器原有的 123 个序列 CSV 会被直接读取。

## 7. 语法和输入检查

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

python -m py_compile tracking/stage3_router_validate.py

export OUTPUT_DIR="/home/u25600003160604/workspace/paper/output/test/tracking_results/sglatrack/deit_distilled/stage2_candidate_validation/uav123"

find "$OUTPUT_DIR" -maxdepth 1 -name '*_stage2_candidates.csv' | wc -l
```

最后一条命令应输出 `123`。

## 8. 先运行 Ridge

```bash
python tracking/stage3_router_validate.py \
  --input "$OUTPUT_DIR" \
  --model ridge \
  --folds 5 \
  --seed 2026 \
  --budgets 0 10 20 30 50 100
```

该实验只进行矩阵运算，通常不需要 GPU。

正常启动信息应为：

```text
Loaded 112578 frames, 112455 valid decisions, 123 sequences from 123 CSV file(s).
Features: 28 causal pre-candidate values; model=ridge; folds=5
```

每个 fold 都会打印训练和测试的帧数及序列数。如果检测到同一序列同时出现在训练和测试中，脚本会直接报错。

## 9. 再运行 MLP

Ridge 完整通过后运行：

```bash
python tracking/stage3_router_validate.py \
  --input "$OUTPUT_DIR" \
  --model mlp \
  --device cuda \
  --folds 5 \
  --seed 2026 \
  --epochs 40 \
  --patience 6 \
  --batch_size 4096 \
  --budgets 0 10 20 30 50 100
```

若不使用 GPU，将 `--device cuda` 改为 `--device cpu`。这里的 GPU 仅训练小型表格 MLP，不会重新运行 tracker。

## 10. 输出文件

Ridge 输出：

```text
uav123_stage3_ridge_overall.csv
uav123_stage3_ridge_budget.csv
uav123_stage3_ridge_folds.csv
uav123_stage3_ridge_oof_predictions.csv
```

MLP 使用相同文件名格式，将 `ridge` 替换为 `mlp`，因此两组结果不会相互覆盖。

### overall

主要字段：

| 字段 | 含义 |
|---|---|
| `best_action_auc` | Best-of-7 理论上界 |
| `best_action_deep_ratio` | Best-of-7 选择候选 block 的帧比例 |
| `oof_layer_hit_rate` | OOF 模型选层达到 Best-of-6 IoU 的比例，并列最佳计为命中 |
| `oof_benefit_auroc` | 预测“是否存在优于 Fast 的候选层”的 AUROC |
| `oof_always_deep_auc` | 每帧执行模型预测候选层，不做 Fast gate |
| `oof_positive_gate_auc` | 仅在预测最大 utility 大于 0 时执行候选层 |

### budget

| 字段 | 含义 |
|---|---|
| `router_auc` | 模型同时负责选帧和选层的真实 OOF AUC |
| `predicted_gate_oracle_layer_auc` | 保留模型选帧，但把选层替换成 Oracle |
| `oracle_gate_predicted_layer_auc` | 使用 Oracle 选帧，但保留模型预测层 |
| `oracle_budget_auc` | 同预算下选帧和选层均使用 GT 的上界 |
| `oracle_recovery_ratio` | 模型恢复同预算 Oracle 增益的比例 |
| `selected_layer_hit_rate` | 被路由为深路径的帧中，模型选层命中率 |
| `action_hit_rate` | 模型最终 Fast/候选动作达到 Best-of-7 的比例 |

分解结果用于定位瓶颈：

- `predicted_gate_oracle_layer_auc` 明显高于 `router_auc`：主要问题是选层。
- `oracle_gate_predicted_layer_auc` 明显高于 `router_auc`：主要问题是选帧或 utility 排序。
- 两者都低：当前低成本特征缺少足够预测信息。

## 11. 结果判读

优先检查：

1. `best_action_auc` 是否明显高于 `69.7411`。
2. OOF benefit AUROC 是否显著高于随机值 `0.5`。
3. OOF layer hit 是否超过原 selector 的 `27.55%`。
4. 10%-50% 预算下 `router_auc` 是否超过 Fast `66.2912` 和原 SGLA `66.5143`。
5. Ridge 与 MLP 的差距是否表明需要非线性决策器。

可将 AUROC `0.55-0.65` 视为存在初步信号，超过 `0.65` 视为较强信号；这只是课题筛选经验值，不是论文指标标准。最终是否成立应以 OOF 路由 AUC 和 Oracle recovery 为准。

## 12. 研究限制

Stage-3 仍使用 Stage-2 的共同 SGLA 控制轨迹。它验证的是离线动作可预测性，不是新路由器自己的闭环跟踪性能。

如果 OOF 结果成立，下一步必须把预测器接入 tracker，让其预测动作真正更新下一帧状态，并分别测量 Fast 与候选路径的 CUDA latency。最终论文还应在训练数据上学习路由器，再在 UAV123 和其他 UAV 数据集上进行跨数据集测试，不能把 UAV123 OOF 当作最终泛化结果。
