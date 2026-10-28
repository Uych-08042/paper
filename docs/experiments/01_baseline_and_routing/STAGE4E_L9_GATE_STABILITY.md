# Stage-4E：L9 门控的在线阈值与稳定性验证

Stage-4D 证明 L9 收益存在弱但可重复的排序信号，但其 `exact_topk` 使用了完整验证
horizon 的分数排序，不能直接作为在线门控器。本阶段不重新训练，也不运行 tracker，
只回答三个问题：

1. 用训练分数确定的固定阈值，能否在未见验证视频上维持预算？
2. 固定阈值策略是否仍优于同深算比例的随机 L9？
3. 该提升是否跨视频稳定，并显著强于保持时序结构的随机对齐？

## 1. 预先固定的比较对象

- 主模型：`history_only__signed`。它不需要当前帧 Fast head 或 token pooling，部署成本最低。
- 次模型：`history_only__positive`。它是 Stage-4D 中超过常数 calibration MSE 的历史模型。
- 信息上限：`fast_only__signed`、`fast_only__positive`。它们使用当前 Fast 输出，深算帧
  需要第二次 head，因此只作为可预测性对照，不作为默认部署候选。

本阶段不再根据验证结果搜索新模型或 alpha。

## 2. 固定阈值协议

对每个模型和预算 `B`：

1. 在全部训练缓存分数上取 top-`B%` 的分界值；
2. 保存一个固定数值阈值；
3. 验证时逐帧执行 `score > threshold`，不读取未来分数和 GT；
4. 报告验证集实际深算比例及其相对目标预算的漂移。

这与 Stage-4D 的验证集全局 exact-top-k 不同，已经满足离线部署阈值的因果边界。

稳定性检验包括：

- 以视频为单位重采样、同时保留 fast/sgla 两条轨迹的 bootstrap 95% 区间；
- 在每个“视频×轨迹”内部循环移动门控序列的单侧置换检验，保留深算簇和时序自相关，
  只破坏门控分数与当前 L9 收益的对齐关系。

## 3. 运行命令

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

export STAGE4_DIR="$PWD/output/stage4"
export PILOT_CACHE="$STAGE4_DIR/pilot_cache_v1"
export L9_GATE_DIR="$STAGE4_DIR/l9_gate_pilot_v1"
export L9_STABILITY_DIR="$STAGE4_DIR/l9_gate_stability_v1"

python -B -m unittest discover -s tracking -p 'test_stage4*.py' -v

python tracking/stage4_l9_gate_stability.py \
  --input "$PILOT_CACHE" \
  --router_dir "$L9_GATE_DIR" \
  --output_dir "$L9_STABILITY_DIR" \
  --models \
    history_only__signed \
    history_only__positive \
    fast_only__signed \
    fast_only__positive \
  --budgets 10 20 30 50 \
  --bootstrap_samples 2000 \
  --permutations 1000 \
  --seed 2026
```

本实验只读取 NPZ/CSV/模型文件，CPU 即可，不需要重新采集数据。

## 4. 输出文件

```text
l9_gate_stability_v1/
  stage4_l9_gate_score_stability.csv
  stage4_l9_gate_online_budget.csv
  stage4_l9_gate_online_groups.csv
  stage4_l9_gate_bootstrap.csv
  stage4_l9_gate_permutation.csv
  stage4_l9_gate_stability.json
```

- `score_stability.csv`：train/val 及 fast/sgla 轨迹上的 AUROC、AP、Pearson。
- `online_budget.csv`：训练阈值、验证实际预算、在线收益及相对 exact-top-k 的差距。
- `online_groups.csv`：每个视频和轨迹的收益、预算与随机基线。
- `bootstrap.csv`：视频级 95% 区间和收益为正的重采样概率。
- `permutation.csv`：循环移位零假设及单侧 p 值。

## 5. 进入闭环的条件

首先只判定预先指定的 `history_only__signed`：

1. 10%、20%、30% 的验证实际预算没有严重漂移；
2. 三档 `val_lift_over_random_l9` 均为正；
3. 至少 20% 或 30% 档的 bootstrap `lift_ci_low > 0`，或者在仅 10 个视频的 pilot
   下 `lift_probability_positive >= 0.9`；
4. 循环移位置换检验 `one_sided_p_value < 0.05`；
5. fast 和 sgla 两种冻结轨迹上的 AUROC 方向一致。

满足后只把一个主预算和一个敏感性预算接入闭环，例如 20% 与 30%。若固定阈值收益
消失或预算严重漂移，应先增加训练视频或改成在线分位数控制，不能用验证集 exact-top-k
结果替代在线策略。
