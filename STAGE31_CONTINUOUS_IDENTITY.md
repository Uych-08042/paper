# Stage-31：无 GT 的持续候选身份关联小规模验证

已建立本轮契约的运行若遇到 `relative_response` 约 1e-9 的回放差异，请沿用原契约并使用 [精度兼容入口](STAGE31_PRECISION_COMPAT.md) 采集、训练、评估和汇总。它保留原容差与原脚本，补充记录审计所用的源 float32 精度。

Stage-30 的两个关键帧都保留了 GT 匹配的第二候选。现在验证一个可部署接口的候选关联基线：只用当前图像、初始框和历史预测连续选框，检验固定 SGLA 特征能否支持持续身份保持。

这是可行性原型。候选关联已有 KeepTrack 等相关工作；本轮尚未提出或验证新的模型压缩方法。SGLA 主干、原始选层器和预测头全部冻结，原始 L7–L12 选择仍照常执行。

## 1. 数据划分与范围

- 数据来源为现有 VisDrone2019-SOT-train。
- 排除 Stage-27 已分析的全部 24 个 source-group 及其同源片段。
- 从其余源视频组固定随机选择 **8 个 fit、6 个 checking**，每组取一个完整视频，seed=2096。
- 选择依据只有源组和目录结构，不依据原始跟踪分数或失效场景。
- 如果剩余源组不足 14 个，建表阶段打印数量并停止，不自动混用旧案例或缩小检查集。
- 建表阶段对选中视频的全部图像进行解码与哈希，并审计标注几何。训练契约只存标注路径、哈希和首帧框，不嵌入未来 GT 数组。
- 训练只读 fit 标签。模型冻结后才运行 checking；checking 的未来 GT 只用于预测完成后的记录与评价，不进入选框、原型更新或裁剪。

`uavNNNNNNN` 前缀是保守源组代理，并非已经验证的物理视频身份。以上 checking 属于新的开发检查，原始模型预训练重叠尚未排除，不能称为外部测试。

## 2. 五个完整视频对照

| 方法 | 在线选框依据 |
| --- | --- |
| `sgla` | 完整保留原始加权首选输出 |
| `motion` | 与自身历史预测的常速度外推位置及尺度最接近 |
| `appearance` | 与首帧目标特征的余弦相似度最高 |
| `static_ranker` | 与完整关联器同样 fit 数据和优化器，只使用初始身份与当前响应 |
| `identity` | 使用初始身份、可靠原型、近期候选对应关系、运动与响应的 9 维排序器 |

所有方法从首帧独立运行完整视频，使用自己的输出产生下一帧裁剪。没有已知失效时刻触发、GT 纠正、事件窗口重置或检查集阈值搜索。每个 checking 视频轮换方法执行顺序，避免始终由同一方法先执行。

## 3. 特征、记忆与训练

每帧仍只执行原始六层前缀加原选择器选出的一个候选层，以及一个原预测头。非原始对照从同一前向的归一化 search token 中，按候选框与实际裁剪 patch 的相交面积池化特征，最多五个候选，不追加 GT 框。

非原始对照在初始化时额外运行一次首帧前向，从已给定的初始框提取永久身份锚点。该开销不属于逐帧主干；不会访问未来 GT。

在线状态包括永久初始锚点、可靠目标特征 EMA、上一帧被选候选及其他候选的描述子、预测框与速度。9 个输入特征为：初始身份相似度、可靠原型相似度、近期目标相似度、上一帧其他候选最大相似度、两者差值、运动代价、尺度代价、相对响应、加权响应。

所有在线状态由预测选择更新。可靠 EMA 的固定更新条件为初始相似度≥0.5 且近期目标对其他候选的相似度差≥0.02，EMA 系数 0.1；这些数值在 checking 前固定，不根据结果调节。

训练步骤：

1. 为 8 个 fit 视频各采集一条原始 SGLA 自身预测轨迹，缓存候选描述子与因果历史特征；不按 GT 更新轨迹或记忆。
2. 状态推进后，用 GT 标注已有候选：IoU≥0.5 为正，IoU<0.1 为负，中间区间忽略。没有正负候选对的帧不进入排序损失。
3. 使用按视频、帧均衡加权的 pairwise logistic loss，固定 L2=0.1，最多 50 次 Newton 迭代。只拟合 9 个排序权重与 fit 特征尺度，没有主干反向传播。
4. 同时拟合 `static_ranker`，使用完全相同的数据、损失和优化器，将全部历史与几何特征置零，检验完整模型是否超出单纯静态评分训练。

至少需要 100 个有候选对的 fit 帧、来自至少 4 个 fit 视频，否则停止。训练准确率是重代入诊断，不是留出结果。fit 状态来自原始策略，完整关联器检查时会产生不同状态分布；闭环 checking 正是对这一变化的检验。

## 4. 固定进入门槛

6 个 checking 视频必须完成全部五种方法，共 30 个文件。完整关联器同时满足以下条件，才得到 `PASS_GT_FREE_IDENTITY_PILOT`：

1. 相对原始 SGLA 的视频配对 AUC 差，5000 次 bootstrap 的 95% 区间下界>0；
2. 相对纯运动、纯外观、同训练预算静态排序器的 AUC 平均差均>0；
3. 平均低 IoU 帧比例不增加；
4. 原始 SGLA 没有持续低 IoU 事件的视频至少有一个，且完整关联器没有为这些视频引入新的事件。事件沿用连续三帧 IoU<0.2 的定义；
5. 完整关联器选择非原首选的比例≥0.5%，且至少发生在两个 checking 视频。

这是小规模开发门槛，不能代替大规模独立验证。简单对照的均值条件不表示已建立统计显著优势。不完整数据只能生成 `PILOT_INCOMPLETE`。

## 5. 服务器步骤

按 Git 流程同步新增文件后，在仓库根目录运行：

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"

export STAGE30_ROOT="$PWD/output/stage30/corrected_branch_response_v1"
export STAGE31_ROOT="$PWD/output/stage31/continuous_identity_pilot_v1"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

python tracking/test_stage31_identity.py

python tracking/stage31_identity_manifest.py \
  --source "${STAGE30_ROOT:?}" \
  --output_dir "${STAGE31_ROOT:?}"

cat "$STAGE31_ROOT/stage31_split.csv"
```

先完成工程 smoke，仅使用第一个 fit 视频最多 64 帧和固定的未训练 bootstrap 权重，不读取 checking 标签：

```bash
python tracking/stage31_identity_fit.py \
  --input "${STAGE31_ROOT:?}" --mode smoke \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

cat "$STAGE31_ROOT/smoke/stage31_smoke_decision.json"
```

预期 `PASS_CAUSAL_IDENTITY_SMOKE`。它只确认真实 SGLA 前向一致、特征有效及五种方法的状态连续性，不能证明关联收益。

smoke 通过后采集全部 fit 轨迹并在 CPU 上拟合冻结模型：

```bash
python tracking/stage31_identity_fit.py \
  --input "${STAGE31_ROOT:?}" --mode collect \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage31_identity_fit.py \
  --input "${STAGE31_ROOT:?}" --mode train
```

随后只进行一次冻结模型的完整 checking：

```bash
python tracking/stage31_identity_evaluate.py \
  --input "${STAGE31_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage31_identity_summary.py \
  --input "${STAGE31_ROOT:?}"

cat "$STAGE31_ROOT/analysis/stage31_decision.json"
cat "$STAGE31_ROOT/analysis/stage31_policy_summary.csv"
cat "$STAGE31_ROOT/analysis/stage31_paired.csv"
```

## 6. 恢复、耗时与产物

- 一个目录只运行一个采集/训练进程。已完成的完整视频缓存会审计后跳过；视频中途断开时该条轨迹重新运行。
- `collect` 可加 `--sequence <fit-video>`；`evaluate` 可加 `--sequence <checking-video>`。这不改变固定 split 或参数。
- 中途查看已有 checking 结果时给 summary 加 `--allow_incomplete`，报告写入 `analysis_pilot`；不据此修改模型。
- 正常退出会释放 `collect.lock`；强制终止遗留的锁应确认无进程写入后再处理。保留所有冻结 run、model、manifest，不能删除标识后混入旧缓存。
- `model.json` 固定两个排序器、fit 数据哈希与来源身份。模型存在时先审计并复用；编辑代码、数据或配置须使用新的契约目录。
- `fit_cache` 只存 fit 原始轨迹；`checking` 保存各方法独立完整轨迹。`analysis` 包含视频级结果、方法宏平均、配对差值及门槛结果。

耗时列是去掉前五帧后的、带同步和诊断的 crop/model/decode/association 墙钟时间，排除图像读取和 GT 记录。它用于观察原型开销，不等于端到端 FPS，也不建立模型压缩、加速或功耗结论。

若通过，再研究身份关系保持的压缩训练，并使用新的检查划分评价后续改变。若未通过，保留原门槛与负结果，不在这六个 checking 视频上继续选特征、阈值或训练步数。
