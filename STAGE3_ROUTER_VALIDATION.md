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

## 13. UAV123 实际结果

| 指标 | Ridge | MLP |
|---|---:|---:|
| Benefit AUROC | 0.4937 | 0.5088 |
| Candidate-layer hit | 27.76% | 28.50% |
| Always-deep AUC | 66.5006 | 66.2873 |
| Predicted-positive AUC | 66.5349 | 66.3883 |
| 最佳 budget AUC | 66.5573 (50%) | 66.4887 (30%) |

共同上界为：

```text
Fast                     66.2912
SGLA                     66.5143
Best-of-6 candidates     69.7411
Best-of-7 Fast+L7-L12    69.8518
```

结论：H1 仍然成立，因为候选层 Oracle 相对 Fast 有 `+3.56` AUC 的空间；当前形式的 H2 不成立。Ridge 和 MLP 的 benefit AUROC 都接近随机值 `0.5`，而选层命中率只比原 selector 的 `27.55%` 高 `0.21` 和 `0.95` 个百分点。当前 28 个聚合特征不足以可靠预测额外计算收益或最佳候选层，不应继续以调参作为主要实验。

## 14. Stage-3B 候选子集 Oracle

在增加新特征或重新运行 tracker 前，先使用现有 Stage-2 CSV 确定需要计算多少个候选 block。`stage3_subset_oracle.py` 会枚举 L7-L12 的全部 `63` 个非空子集，并分别计算：

- 强制从子集中选择候选层的 Best-of-K AUC。
- 允许保留 Fast 的 Best-of-(K+1) AUC。
- 相对完整六候选 Oracle 的增益恢复率。
- 每层 leave-one-out AUC 损失和唯一贡献帧比例。

运行命令：

```bash
python -m py_compile tracking/stage3_subset_oracle.py

python tracking/stage3_subset_oracle.py \
  --input "$OUTPUT_DIR"
```

输出文件：

```text
uav123_stage3_subset_summary.csv
uav123_stage3_subset_all.csv
uav123_stage3_subset_best_by_k.csv
uav123_stage3_subset_leave_one_out.csv
```

优先查看 `best_by_k.csv`。如果 2-3 个候选 block 已恢复大部分六候选上界，下一步应针对该小候选集设计候选后 agreement selector；如果必须使用接近 6 个 block 才能获得上界，则需要先重新训练具有可分辨路由监督的 selector，而不是继续扩展当前表格模型。

## 15. Stage-3B 实际结果

| Block 数 | 最佳 Fast-inclusive 子集 | AUC | 完整上界恢复率 |
|---:|---|---:|---:|
| 1 | L11 | 67.6364 | 37.78% |
| 2 | L7, L11 | 68.5265 | 62.78% |
| 3 | L7, L9, L12 | 69.0960 | 78.77% |
| 4 | L7, L9, L11, L12 | 69.4804 | 89.57% |
| 5 | L7, L8, L9, L11, L12 | 69.6859 | 95.34% |
| 6 | L7-L12 | 69.8518 | 100.00% |

4-block 子集是较明确的计算量与上界折中点。Leave-one-out 结果显示，L10 和 L8 的边际 AUC 贡献最小，但每层仍分别在约 `9.8%-14.7%` 的有效帧上是唯一最佳动作，不能依据全局固定 AUC 将任一层视为完全冗余。

## 16. Stage-3C 候选后选择

该实验使用已有的 candidate response peak 和候选框，不执行模型推理。它回答：当 K 个候选 block 已经执行后，能否通过可观测置信度和候选框一致性选择或融合出更好的结果？

测试方法包括：

- 原 selector probability 在候选子集内选层。
- response peak 选层。
- 候选框 IoU consensus medoid。
- selector、peak、consensus 的等权选择。
- 三者权重在训练序列上搜索、测试序列上应用的 5-fold OOF 选择。
- bbox mean、median、peak-weighted 和 consensus-weighted 融合。
- 对应候选子集的 GT Oracle，仅作为上界。

运行：

```bash
python -m py_compile tracking/stage3_post_candidate_validate.py

python tracking/stage3_post_candidate_validate.py \
  --input "$OUTPUT_DIR" \
  --folds 5 \
  --seed 2026 \
  --weight_step 0.1
```

输出：

```text
uav123_stage3_post_candidate_summary.csv
uav123_stage3_post_candidate_oof_weights.csv
```

注意：候选子集来自同一 UAV123 数据集的探索性 Oracle 分析。OOF 权重本身没有看到测试序列，但候选子集的确定仍使用了全数据，因此结果用于判断信号是否存在，不能直接作为最终无泄漏论文结果。正式实验必须在训练集确定候选子集和权重，再在独立测试集评估。

## 17. Stage-3C 实际结果与闭环验证

可观测选层仍然失败：含 L9 的 OOF weighted selector 在约 `97%-100%` 帧上继续选择 L9，AUC 没有超过原 SGLA。真正有效的是候选框的鲁棒融合：

```text
Method                         AUC       gain vs SGLA
3-block bbox median          67.3137       +0.7994
4-block bbox median          67.4989       +0.9846
6-block bbox median          67.4998       +0.9855
```

4-block `L7,L9,L11,L12` 与 6-block median 只差 `0.00084` AUC，因此没有证据支持额外执行 L8 和 L10。

这些结果来自共同 SGLA 轨迹。必须让 median4 输出更新下一帧 tracker state，才能得到真实闭环结果。新增验证脚本只运行指定策略所需的候选 block，不会为了 median4 额外执行 L8/L10。

先在单序列独立目录进行一致性检查：

```bash
export CLOSED_LOOP_SANITY="/home/u25600003160604/workspace/paper/output/test/tracking_results/sglatrack/deit_distilled/stage3_closed_loop_validation/uav123_sanity"

python tracking/stage3_closed_loop_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --sequence 0 \
  --gpu 0 \
  --policies sgla median4 median6 \
  --output_dir "$CLOSED_LOOP_SANITY" \
  --verify_sgla \
  --overwrite

python tracking/stage3_closed_loop_summary.py \
  --input "$CLOSED_LOOP_SANITY" \
  --stage2_input "$OUTPUT_DIR"
```

确认 SGLA 单序列 AUC 与 Stage-2 对应序列一致后，运行完整 median4：

```bash
python tracking/stage3_closed_loop_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --gpu 0 \
  --policies median4
```

多 GPU 可使用 `--num_shards` 和 `--shard_id`，所有 shard 必须写入相同默认输出目录。完成后：

```bash
export CLOSED_LOOP_DIR="/home/u25600003160604/workspace/paper/output/test/tracking_results/sglatrack/deit_distilled/stage3_closed_loop_validation/uav123"

python tracking/stage3_closed_loop_summary.py \
  --input "$CLOSED_LOOP_DIR" \
  --stage2_input "$OUTPUT_DIR"
```

`model_decode_latency_ms` 只统计 GPU forward、bbox decode 和融合，并在每帧前后同步 CUDA。它适合策略间相对比较，不等同于包含图像读取、crop 和预处理的端到端 FPS。
