# SGLATrack 第一阶段三路径验证

## 1. 本阶段验证什么

本实验只回答两个问题：

1. 同一帧是否存在明显的计算需求差异，即 `Full` 相对 `Fast` 是否能提高 IoU。
2. 如果 Oracle 事先知道哪些帧最值得增加计算，10%/20%/30%/50% 的 Full 预算能恢复多少性能。

三条路径定义为：

- `Fast`：执行 Transformer block 1-6，然后进入原 box head。
- `SGLA`：执行 block 1-6，再执行原选择器选中的一个 block，然后进入原 box head。
- `Full`：顺序执行全部 12 个 block，然后进入原 box head。

验证脚本让三条路径使用完全相同的 template、search crop、预处理和上一帧状态。每帧结束后只使用 `SGLA` 的预测更新 tracker 状态；`Fast` 与 `Full` 是该帧上的反事实预测。因此本实验验证的是“这一帧增加计算是否有价值”，不是三条独立闭环轨迹的最终性能。

## 2. 文件范围

本实现不修改原始训练、模型和测试代码，只新增：

- `tracking/stage1_validate.py`：运行 Fast/SGLA/Full，逐帧计算 IoU 并写 CSV。
- `tracking/stage1_oracle.py`：合并序列 CSV，计算固定预算 Oracle AUC。
- `STAGE1_VALIDATION.md`：本说明。

`stage1_validate.py` 直接复用原 checkpoint、数据预处理、Hann window、box head、坐标映射和裁剪逻辑。SGLA 分支还可以与原 `network.forward()` 做数值一致性检查。

## 3. 上传服务器后的准备

进入仓库根目录，并激活你已经跑通 baseline 的环境：

```bash
cd /path/to/SGLATrack
conda activate <your_sglatrack_env>
```

确认下面两项仍然正确：

1. `lib/test/evaluation/local.py` 中的 `prj_dir`、`save_dir`、`results_path` 和 `uav123_path`。
2. `lib/test/parameter/sglatrack.py` 最终指向已经复现 AUC 66.51 的 checkpoint。

先做语法检查：

```bash
python -m py_compile tracking/stage1_validate.py tracking/stage1_oracle.py
```

## 4. 先跑一个序列

```bash
python tracking/stage1_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --sequence 0 \
  --gpu 0 \
  --verify_sgla \
  --overwrite
```

`--verify_sgla` 会在该序列第一个跟踪帧上，将验证脚本的 SGLA 分支与仓库原始 SGLA forward 比较。如果不一致，脚本会直接报错并给出最大绝对误差，不应继续全量实验。

正常输出应包含：

```text
Paths: Fast=L1-L6, SGLA=L1-L6+selected, Full=L1-L12
[done] <sequence>: <N> frames -> .../<sequence>_stage1.csv
```

默认输出目录是：

```text
<results_path>/sglatrack/deit_distilled/stage1_validation/uav123/
```

脚本启动时会打印服务器上的实际绝对路径。

## 5. 检查单序列 CSV

```bash
head -n 3 <实际输出目录>/<sequence>_stage1.csv
```

核心列为：

| 列 | 含义 |
|---|---|
| `dataset`, `sequence`, `frame_id` | 数据集、序列和从 0 开始的帧号 |
| `gt_valid` | 当前 GT 是否有效 |
| `iou_fast` | Fast 框与 GT 的 IoU |
| `iou_sgla` | 原 SGLA 框与 GT 的 IoU |
| `iou_full` | Full 框与 GT 的 IoU |
| `delta_full_fast` | `iou_full - iou_fast` |
| `sgla_selected_layer` | SGLA 选中的 1-based block 编号，通常为 7-12 |
| `gt_*` | 当前 GT 的 xywh |
| `bbox_fast_*`, `bbox_sgla_*`, `bbox_full_*` | 三条路径的 xywh 预测框 |

IoU 计算沿用仓库 `extract_results.py` 的 inclusive-pixel 定义。除初始化帧外，预测框会按原评测保存逻辑先截断为整数，因此 `iou_sgla` 计算出的 AUC可以直接和现有 baseline 对照。初始化帧按原评测逻辑令 IoU 为 1。

## 6. 跑完整 UAV123

单 GPU：

```bash
python tracking/stage1_validate.py \
  sglatrack deit_distilled \
  --dataset_name uav123 \
  --gpu 0
```

已经存在且完整的 `*_stage1.csv` 默认会跳过，所以中断后直接重复命令即可续跑。只有需要重新生成时才加 `--overwrite`。

四 GPU 可按序列分片并行。四个进程各自只看见一张 GPU，因此都使用 `--gpu 0`：

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python tracking/stage1_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 0 > logs/stage1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python tracking/stage1_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 1 > logs/stage1_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python tracking/stage1_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 2 > logs/stage1_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python tracking/stage1_validate.py sglatrack deit_distilled --dataset_name uav123 --gpu 0 --num_shards 4 --shard_id 3 > logs/stage1_gpu3.log 2>&1 &
wait
```

确认得到 123 个序列文件：

```bash
find <实际输出目录> -maxdepth 1 -name '*_stage1.csv' | wc -l
```

## 7. 运行 Oracle budget 实验

把 `<实际输出目录>` 替换为验证脚本打印的目录：

```bash
python tracking/stage1_oracle.py \
  --input <实际输出目录> \
  --budgets 0 10 20 30 50 100
```

产生两个文件：

```text
uav123_stage1_all.csv
uav123_oracle_budget.csv
```

`uav123_stage1_all.csv` 合并所有帧，并为每个预算新增：

```text
oracle_path_20   # init / fast / full
iou_oracle_20
```

`uav123_oracle_budget.csv` 每个预算一行，主要字段为：

| 列 | 含义 |
|---|---|
| `budget_percent` | 允许使用 Full 的跟踪帧比例 |
| `num_full` | 实际选择的 Full 帧数 |
| `actual_full_ratio` | 实际 Full 比例 |
| `oracle_auc` | 对每个序列先算成功曲线、再对 123 个序列平均的 AUC，单位为百分数 |
| `oracle_gain_vs_fast` | Oracle AUC 相对 Fast 的提升 |
| `recovery_ratio` | `(Oracle-Fast)/(Full-Fast)` |
| `fast_auc`, `sgla_auc`, `full_auc` | 同一 SGLA 控制轨迹上的三路径条件参考 AUC |
| `positive_delta_ratio` | 有效跟踪帧中 `Full > Fast` 的比例 |
| `delta_p50/p90/p95_valid` | `delta_full_fast` 的分位数 |

Oracle 对所有非初始化帧按 `delta_full_fast` 从高到低排序，再严格选取预算允许的帧数。它使用 GT，是理论上界，不是可部署策略。

## 8. 必做的正确性检查

1. 单序列 `--verify_sgla` 必须通过。
2. CSV 的帧数必须等于该序列图像帧数。
3. Oracle `0%` 的 AUC 必须等于 `fast_auc`。
4. Oracle `100%` 的 AUC 必须等于 `full_auc`。脚本也会自动断言这两项。
5. `sgla_auc` 应接近你已经得到的 66.51。若差异超过约 0.1 AUC，先检查是否使用了同一 checkpoint、同一 UAV123 路径和完整 123 个序列。

本脚本每帧执行三次推理，因此其整体 FPS不能作为 Fast、SGLA 或 Full 的独立速度。第一阶段只分析精度收益；单路径延迟应在后续实验中分别测量并执行 CUDA synchronize。

## 9. 如何判读结果

重点看 `Full-Fast` 的总差距和 Oracle 在小预算下恢复了多少差距：

- 若 `full_auc` 明显高于 `fast_auc`，且 20%-30% Oracle 已恢复大部分差距，说明额外计算集中在少量帧上，课题的自适应计算假设较强。
- 若 Full 平均提升不大，但 `delta_p90/p95` 较高，且中等预算 Oracle 超过 Fast 和 Full，说明 Full 只对部分帧有帮助，更需要准确的帧级决策器。
- 若多数 `delta_full_fast <= 0`，或者 10%-50% Oracle 曲线几乎不升，说明“困难帧增加深度”在当前 checkpoint 上缺少足够收益，应先调整路径定义或训练方式。
- `recovery_ratio` 只在 `full_auc > fast_auc` 时最直观。可将 20%-30% 预算恢复约 70% 以上视为很强的继续信号，40%-70% 视为中等信号；这是研究筛选经验值，不是论文指标标准。

第一阶段成立后，下一步才是用 SGLA 当前帧可观测特征预测 `delta_full_fast` 是否为正，并与置信度、entropy、temporal change 等难度指标比较。
