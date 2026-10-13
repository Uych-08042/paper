# Stage-4F：因果在线预算控制器

Stage-4E 表明，训练集固定阈值无法把目标预算稳定迁移到未见视频。例如，10% 训练预算
在验证集上变为约 19%--22%，20% 预算变为约 33%--46%。因此，本阶段不再调整固定阈值，
而是验证一个只依赖当前视频历史分数的在线控制器。

本阶段仍然读取冻结轨迹缓存，不运行 tracker、不重新训练模型，也不需要 GPU。

## 1. 预先固定的实验对象

- 主模型：`history_only__signed`
- 敏感性模型：`history_only__positive`
- 预算：10%、20%、30%、50%
- 主控制器：过去 32 帧分数的经验分位数，至少积累 10 帧历史
- 预算器：容量为 2 的 token bucket
- 成本基线：按目标预算运行、与分数无关的周期性 L9
- 排序基线：在每条轨迹上与在线控制器使用完全相同深算次数的周期性 L9

模型仅提供帧级排序分数。在线控制器在第 `t` 帧作决定时，只使用该轨迹第
`1,...,t-1` 帧的分数。token bucket 保证任意前缀内的深算次数不超过目标预算，并允许
最多约一个深算机会被延后，以便把 L9 分配给较高分帧。

## 2. 运行命令

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

export STAGE4_DIR="$PWD/output/stage4"
export PILOT_CACHE="$STAGE4_DIR/pilot_cache_v1"
export L9_GATE_DIR="$STAGE4_DIR/l9_gate_pilot_v1"
export L9_CONTROLLER_DIR="$STAGE4_DIR/l9_online_controller_v1"

python -B -m unittest discover -s tracking -p 'test_stage4*.py' -v

python tracking/stage4_l9_online_controller.py \
  --input "$PILOT_CACHE" \
  --router_dir "$L9_GATE_DIR" \
  --output_dir "$L9_CONTROLLER_DIR" \
  --models history_only__signed history_only__positive \
  --budgets 10 20 30 50 \
  --window 32 \
  --min_history 10 \
  --bucket_capacity 2 \
  --bootstrap_samples 2000 \
  --seed 2026
```

若目录中已有旧结果，确认不再需要后加 `--overwrite`。不要根据验证结果搜索大量
`window`、`min_history` 或 `bucket_capacity` 组合；上述配置是本阶段预先指定的主实验。

## 3. 输出文件

```text
l9_online_controller_v1/
  stage4_l9_online_controller_summary.csv
  stage4_l9_online_controller_groups.csv
  stage4_l9_online_controller_bootstrap.csv
  stage4_l9_online_controller.json
```

- `summary.csv`：在线分位数、目标预算周期基线和次数匹配周期基线的实际预算、收益、
  随机提升、exact-top-k 差距和估计延迟。
- `groups.csv`：每个视频、每条冻结轨迹的结果。
- `bootstrap.csv`：在线分位数策略本身的区间，以及相对周期基线的配对视频级区间。
- `json`：输入指纹、模型哈希、参数和输出路径，便于复现。

## 4. 判定规则

首先只判定 `history_only__signed`，`positive` 仅作敏感性检查。

进入闭环验证的最低条件：

1. 20% 或 30% 档的 `gain_vs_periodic_matched > 0`；
2. 对应 `gain_vs_periodic_matched_ci_low > 0`，或 pilot 数据较小时
   `gain_vs_periodic_matched_probability_positive >= 0.90`；
3. 实际深算比例接近目标预算，且明显不再出现 Stage-4E 的约两倍漂移；
4. 10%、20%、30% 三档的方向总体一致，而不是只有单个档位偶然为正。

若条件成立，只选择 20% 和 30% 中证据更强的一档进入完整闭环，并保留另一档作敏感性
实验。若条件不成立，应停止继续调阈值，扩大独立训练视频或引入新的低成本时序特征。
`fast_only` 在 Stage-4E 中证明当前帧 Fast 输出包含更多可预测信息，但其额外 head 延迟使它
不适合作为当前本地深算策略的默认部署方案。

