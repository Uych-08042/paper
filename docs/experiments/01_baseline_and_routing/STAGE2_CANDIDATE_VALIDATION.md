# SGLATrack 第二阶段候选层验证

## 1. 本阶段解决什么问题

第一阶段已经得到：

- Fast AUC：`66.2912`
- 原 SGLA AUC：`66.5143`
- 顺序 Full AUC：`63.7242`
- 约 31.78% 的有效帧满足 `IoU_full > IoU_fast`
- 30% 帧预算 Oracle 达到 `67.8364`

顺序 Full 会把 L7-L12 串联执行，但原 SGLATrack 的训练和推理机制是：先执行 L1-L6，再从 L7-L12 中选择一个 block，使该 block 直接作用于同一个 L6 特征。因此，顺序 Full 不是训练一致的路径。

本阶段固定 L1-L6，并执行：

```text
L6 -> L7
L6 -> L8
L6 -> L9
L6 -> L10
L6 -> L11
L6 -> L12
```

六个候选 block 都接收完全相同的 L6 tokens。实验回答：

1. 哪个候选 block 的固定 AUC 最好。
2. 原 MLP selector 是否选中了当前帧最好的候选 block。
3. Best-of-6 相对原 SGLA 还有多大上界。
4. 不使用 GT 的响应峰值选择和均值集成能否超过原 SGLA。
5. 只在 10%/20%/30%/50% 帧上启用候选路径时，Oracle 上界是多少。

和第一阶段一样，每帧结束后只使用原 SGLA 选择结果更新 tracker 状态。其他方法都是同一 crop、同一历史状态上的反事实结果，不是各自独立闭环轨迹。

## 2. 新增文件

- `tracking/stage2_candidate_validate.py`：运行 Fast、六个独立候选层、原 SGLA selector、响应峰值选择、均值集成和 Best-of-6。
- `tracking/stage2_candidate_oracle.py`：计算方法 AUC、候选层分布、selector regret 和帧预算 Oracle。
- `STAGE2_CANDIDATE_VALIDATION.md`：本说明。

原模型、训练代码、tracker 和第一阶段文件均未修改。

## 3. 上传服务器后先做语法检查

进入已经跑通第一阶段的仓库和 Conda 环境：

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

python -m py_compile \
  tracking/stage2_candidate_validate.py \
  tracking/stage2_candidate_oracle.py
```

继续使用第一阶段已经修正的：

- `lib/test/evaluation/local.py`
- checkpoint 路径
- UAV123 路径

## 4. 必须先跑单序列 smoke test

```bash
python tracking/stage2_candidate_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --sequence 0 \
  --gpu 0 \
  --verify_sgla \
  --overwrite
```

正常输出应包含：

```text
Candidates: L6 base + independent L7, L8, L9, L10, L11, L12
[done] uav_bike1: 3085 frames -> .../uav_bike1_stage2_candidates.csv
```

`--verify_sgla` 会在第一个跟踪帧上检查：脚本根据 MLP 选出的候选 block 输出，必须与未修改的原始 `network.forward()` 数值一致。若报 `Standalone SGLA path differs`，不要继续全量实验。

默认输出目录为：

```text
<results_path>/sglatrack/deit_distilled/stage2_candidate_validation/uav123/
```

脚本启动时会打印实际绝对路径。

## 5. 检查单序列输出

```bash
OUTPUT_DIR=/home/u25600003160604/workspace/paper/output/test/tracking_results/sglatrack/deit_distilled/stage2_candidate_validation/uav123

head -n 2 "$OUTPUT_DIR/uav_bike1_stage2_candidates.csv"
```

核心字段如下：

| 字段 | 含义 |
|---|---|
| `iou_fast` | L1-L6 后直接进入 head |
| `iou_layer7` ... `iou_layer12` | 六个 block 分别独立作用于同一个 L6 特征后的 IoU |
| `iou_sgla` | 原 MLP selector 选中候选层的 IoU |
| `iou_confidence_select` | 选择 Hann-windowed response peak 最大候选层的 IoU |
| `iou_ensemble_mean` | 对六个候选输出的 score/size/offset map 求均值后解码的 IoU |
| `iou_best_candidate` | 使用 GT 选择六个候选中 IoU 最大者，仅作为 Oracle |
| `sgla_selected_layer` | 原 selector 选择的 block，范围 7-12 |
| `confidence_selected_layer` | response peak 选择的 block |
| `best_candidate_layer` | GT Oracle 选择的 block |
| `selector_hit` | 原 selector 的 IoU 是否达到 Best-of-6；并列最佳也计为命中 |
| `selector_regret` | `iou_best_candidate - iou_sgla` |
| `best_candidate_tie_count` | 当前帧达到相同最佳 IoU 的候选层数量 |
| `selector_prob_layer*` | 原 MLP 对各候选 block 的 sigmoid 输出 |
| `response_peak_layer*` | 各候选层经过 Hann window 后的最大响应 |
| `selector_max/margin/entropy` | 后续预测困难度可使用的无 GT 特征 |
| `bbox_*` | 各方法和各候选层的 xywh 预测框 |

## 6. 先分析单序列

```bash
python tracking/stage2_candidate_oracle.py \
  --input "$OUTPUT_DIR/uav_bike1_stage2_candidates.csv" \
  --budgets 0 10 20 30 50 100
```

这一步用于检查分析脚本能够完整运行，不要用一个序列的结果下研究结论。

## 7. 跑完整 UAV123

单 GPU：

```bash
python tracking/stage2_candidate_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --gpu 0 \
  --verify_sgla
```

已有的 `*_stage2_candidates.csv` 默认跳过，因此任务中断后可直接重复运行。只有要重算已有序列时才添加 `--overwrite`。

四 GPU 分片：

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python tracking/stage2_candidate_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 0 --verify_sgla > logs/stage2_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python tracking/stage2_candidate_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 1 --verify_sgla > logs/stage2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python tracking/stage2_candidate_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 2 --verify_sgla > logs/stage2_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python tracking/stage2_candidate_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 3 --verify_sgla > logs/stage2_gpu3.log 2>&1 &
wait
```

确认输出 123 个序列：

```bash
find "$OUTPUT_DIR" -maxdepth 1 -name '*_stage2_candidates.csv' | wc -l
```

确认没有进程报错：

```bash
grep -HnE 'Traceback|RuntimeError|Error' logs/stage2_gpu*.log
```

若该命令没有输出，表示未发现这些错误关键字。

## 8. 全量汇总和预算实验

```bash
python tracking/stage2_candidate_oracle.py \
  --input "$OUTPUT_DIR" \
  --budgets 0 10 20 30 50 100
```

输出四个汇总文件：

```text
uav123_stage2_candidates_all.csv
uav123_stage2_method_summary.csv
uav123_stage2_budget_summary.csv
uav123_stage2_layer_distribution.csv
```

### method summary

`uav123_stage2_method_summary.csv` 包含：

- Fast、原 SGLA、固定 L7-L12、响应峰值选择、均值集成和 Best-of-6 的宏序列 AUC。
- 相对 Fast 的 AUC 增益。
- 原 selector 和响应峰值选择的 exact-best hit rate。
- `regret_mean/p50/p90/p95`。
- `near_best_001_rate`：所选候选与 Best-of-6 的 IoU 差不超过 0.01 的帧比例。

### layer distribution

`uav123_stage2_layer_distribution.csv` 包含每个候选层的：

- 固定层 AUC。
- 原 selector 选择比例。
- response peak 选择比例。
- 作为确定性代表层成为 Best-of-6 的比例，以及包含并列最佳的比例。

### budget summary

`uav123_stage2_budget_summary.csv` 对以下四种深路径分别计算精确帧预算 Oracle：

```text
sgla_selector
confidence_select
ensemble_mean
best_candidate_oracle
```

Oracle 使用 GT 按 `deep_iou - fast_iou` 从高到低选择规定比例的非初始化帧。因此它衡量“如果能够完美预测哪些帧值得启用该深路径”的上界，不是可部署方法。

`candidate_blocks_per_deep_frame` 表示深帧额外计算的候选 block 数：原 SGLA 为 1，response peak、ensemble 和 Best-of-6 都必须计算 6 个候选 block。`equivalent_candidate_blocks_per_frame` 用于避免只比较帧比例而忽略每个深帧的计算量。

## 9. 必做正确性检查

1. 单序列和全量运行的 `--verify_sgla` 必须通过。
2. 全量输出必须有 123 个序列 CSV。
3. `sgla_selector` AUC 应接近第一阶段的 `66.5143`，建议误差不超过 0.1 AUC。
4. `fast` AUC 应接近第一阶段的 `66.2912`。
5. 每个预算策略的 0% Oracle 必须等于 Fast，100% 必须等于该策略；分析脚本会自动断言。
6. `selector_regret` 和 `confidence_regret` 不应为负数，只有浮点误差量级例外。
7. `best_candidate_oracle` AUC 必须不低于六个固定候选层中的最高 AUC，但宏序列 AUC与逐帧最大值组合的差异仍应通过汇总脚本计算，不要手工平均。

## 10. 结果判读

优先比较下面四组数：

1. `best_candidate_oracle - sgla_selector`
2. 原 selector 的 `exact_best_hit_rate`、`near_best_001_rate` 和 `regret_mean`
3. `confidence_select`、`ensemble_mean` 是否超过 `sgla_selector`
4. 10%-30% `best_candidate_oracle` 或实际候选策略的预算 Oracle 是否明显超过 Fast/SGLA

结论可按以下逻辑判断：

- **Best-of-6 明显高于 SGLA**：候选 block 本身有互补性，当前 selector 仍有改进空间，值得训练新的层选择器。
- **Best-of-6 高，但所有无 GT 选择方法低**：上界存在，但当前可观测置信度不足，需要进入难度/收益预测实验。
- **response peak 或 ensemble 已超过 SGLA**：得到一个无需训练的强基线；后续学习方法至少应超过它。
- **某个固定层接近 Best-of-6**：动态选层价值较弱，应先与该固定层比较计算开销，再决定是否继续做 selector。
- **Best-of-6 仅比 Fast/SGLA 高很少**：候选层多样性不足。继续做复杂路由器的收益空间有限，应考虑重新训练候选层或改变深路径定义。

本阶段通过后，下一阶段才使用不含 GT 的当前帧和历史特征，预测 `delta_best_fast`、`selector_regret` 或“是否启用候选路径”，并进行严格的序列级 train/validation/test 划分。
