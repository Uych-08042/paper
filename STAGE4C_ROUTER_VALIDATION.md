# Stage-4C：候选收益路由器离线验证

本阶段直接读取 Stage-4B 已审计的 NPZ 缓存，不重新运行 SGLATrack，也不需要 GPU。
目标是在未见过的 GOT-10k 验证视频上回答：L6 的候选前信息能否预测 L7-L12
相对 Fast 的逐帧 IoU 收益。

## 1. 实验边界

- 训练和校准只使用 manifest 的 `train` 视频。
- `train` 内再按视频划分 fit/calibration；同一视频的 fast/sgla 两条轨迹不会被拆开。
- manifest 的 `val` 视频不参与 PCA、标准化、Ridge 参数或预算阈值选择。
- 输入只使用缓存中的因果字段。`gt`、`iou`、`delta` 只作为监督或最终评估。
- 每个“视频×轨迹”在损失和精度指标中权重相同，避免长视频支配结果。
- 预算比例按真实帧数计算，表示“最多允许该比例帧深算”。负预测收益的帧不会为凑满预算而强制深算。
- 输出是冻结轨迹上的 current-state mean IoU，不是闭环 tracking AUC。

三个输入版本：

| 版本 | 当前帧可用信息 | 部署含义 |
|---|---|---|
| `token_only` | L6 池化 token + 历史状态/动作 | 在 Fast head 前决策，最有速度潜力 |
| `token_selector` | `token_only` + 原 selector 输出 | 判断原 selector 信息是否有增益 |
| `fast_aware` | `token_only` + Fast score/box | 信息更充分，但深算帧需要第二次 head |

三者都先以训练集拟合的 randomized PCA 将 token 降到 64 维，再用六输出 Ridge
预测 `delta_L7 ... delta_L12`。

## 2. 运行命令

在服务器仓库根目录执行：

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

export STAGE4_DIR="$PWD/output/stage4"
export PILOT_CACHE="$STAGE4_DIR/pilot_cache_v1"
export ROUTER_DIR="$STAGE4_DIR/router_pilot_v1"

python -B -m unittest discover -s tracking -p 'test_stage4*.py' -v

python tracking/stage4_router_train.py \
  --input "$PILOT_CACHE" \
  --output_dir "$ROUTER_DIR" \
  --variants token_only token_selector fast_aware \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.0001 0.001 0.01 0.1 1 10 \
  --calibration_fraction 0.2 \
  --budgets 0 10 20 30 50 100 \
  --profile_dirs \
    "$STAGE4_DIR/profile_seq0" \
    "$STAGE4_DIR/profile_seq1" \
    "$STAGE4_DIR/profile_seq2" \
  --seed 2026
```

`ROUTER_DIR` 必须位于 `PILOT_CACHE` 外部，否则模型 NPZ 会被缓存完整性检查视为未知文件。
首次运行不要加 `--overwrite`；需要在同一目录重算时才加。

## 3. 输出文件

```text
router_pilot_v1/
  stage4_router_alpha_selection.csv
  stage4_router_summary.csv
  stage4_router_budget.csv
  stage4_router_val_predictions.csv
  stage4_router_token_only.npz
  stage4_router_token_selector.npz
  stage4_router_fast_aware.npz
  stage4_router_experiment.json
```

- `summary.csv`：未见 val 视频上的收益 AUROC、层命中率、回归误差，以及 Fast、原 SGLA selector、动作 Oracle 的 current-state mean IoU。
- `budget.csv`：各最大深算预算下的实际深算比例、IoU、相对 Fast/SGLA 的增益、regret 和估算延迟。
- `val_predictions.csv`：逐帧六维预测与真值，供误差分层和可视化。
- 三个模型 NPZ：PCA、标准化和 Ridge 参数，使用 `allow_pickle=False` 可读取。
- `experiment.json`：缓存、manifest、模型和 profile 指纹及实验警告。

快速查看：

```bash
cat "$ROUTER_DIR/stage4_router_summary.csv"
cat "$ROUTER_DIR/stage4_router_budget.csv"
cat "$ROUTER_DIR/stage4_router_alpha_selection.csv"
```

## 4. 结果判读顺序

1. 先看 `benefit_auroc`。接近 0.50 表示仍不能可靠识别“深算有益帧”；明显高于 0.50 才值得继续。
2. 再看 10%-30% 档的 `gain_vs_fast` 和 `gain_vs_sgla_selector`。低预算正增益比 100% 档更符合研究目标。
3. 比较三个版本。如果 `token_only` 接近或优于 `fast_aware`，后续优先部署前者；它不会产生 Fast head 后再深算的重复 head 成本。
4. 看 `candidate_layer_hit_rate` 和 `regret_mean`，区分问题来自“找不到有益帧”还是“有益帧上选错层”。
5. `estimated_latency_ms_excluding_router` 只使用 Stage-4A 分项估算，尚未包含 PCA/Ridge 的真实 GPU 延迟，不能作为最终 FPS。

当前 pilot 只有 30 个训练视频和 10 个验证视频。它适合做方向筛选，不适合作为论文最终泛化结果。
若 `token_only` 在 val 上显示稳定正信号，下一步才把导出的模型接入真正闭环并测量路由器端到端延迟。
