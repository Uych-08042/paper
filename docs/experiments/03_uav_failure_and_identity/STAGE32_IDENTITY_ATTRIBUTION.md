# Stage-32：冻结候选、特征与选框的只读归因

Stage-31 的持续身份关联在六个 checking 视频上未取得净收益：相对 SGLA 为 −0.1279 AUC 点，0 胜 / 4 平 / 2 负，改选比例仅 0.176%。训练候选对准确率 93.50% 不能说明它学会了纠正原始响应选错的困难帧。

本轮用已有缓存拆分候选覆盖、特征排序和改选结果。它是失败后的描述性审计，不建立新方法的通过门槛，不训练、不搜索参数、不解码图像、不调用 GPU，也不生成新轨迹。原 Stage-31 模型、契约、缓存和失败决定全部保留。

## 先运行 fit 归因

将新增脚本通过仓库同步到服务器。继续使用 `sglatrack-test` 环境，无需配置 GPU 或重新指定 checkpoint：

```bash
export STAGE32_ROOT="$PWD/output/stage32/frozen_identity_attribution_v1"

python tracking/test_stage32_identity_attribution.py

python tracking/stage32_identity_attribution.py \
  --input "${STAGE31_ROOT:?请设置已有Stage31根目录}" \
  --scope fit \
  --output_dir "$STAGE32_ROOT/fit_v1"
```

默认 `--scope fit` 只打开八个 fit 视频的缓存及标注，不打开 checking 文件。fit 中的 frozen identity 和 static ranker 已经在这些视频上训练过，因此这些分数是拟合后诊断，不能称为 OOF 或独立泛化结果。本轮只按视频聚合，不重新训练交叉验证模型。

## 补充解释已经失败的 checking

用户已经完成并查看本轮 checking。若同时解释六个视频上的错误改选和全视频差异，显式运行：

```bash
python tracking/stage32_identity_attribution.py \
  --input "${STAGE31_ROOT:?}" \
  --scope both \
  --output_dir "$STAGE32_ROOT/fit_and_checking_v1"
```

该模式要求原 `analysis/stage31_decision.json` 和全部 30 个 checking 缓存完整匹配，读取 fit 与 checking 后分开报告。不能用 checking 的结果选择新的特征、阈值、权重、训练步数或候选提取规则。后续若提出新模型，需要另设开发与评估安排。

两个 scope 使用不同输出目录；同一目录只允许重跑完全相同的来源、脚本和分析规格。

## 看哪些结果

以下示例读取包含 checking 的完整解释目录；只做 fit 时改为 `$STAGE32_ROOT/fit_v1`。

```bash
export STAGE32_ANALYSIS="$STAGE32_ROOT/fit_and_checking_v1"
cat "$STAGE32_ANALYSIS/stage32_attribution.json"
cat "$STAGE32_ANALYSIS/stage32_candidate_summary.csv"
cat "$STAGE32_ANALYSIS/stage32_choice_summary.csv"
cat "$STAGE32_ANALYSIS/stage32_closed_loop_attribution.csv"
```

| 文件 | 回答的问题 |
|---|---|
| `stage32_candidate_summary.csv` | 当前候选 0 选错时，Top-5 中是否还保留 IoU≥0.5 的候选？ |
| `stage32_video_candidates.csv` | 这些机会分布于哪些视频，是否集中在少数轨迹？ |
| `stage32_feature_summary.csv` | 正确与错误候选相比，初始外观、历史身份、运动、原响应及冻结模型分数的排序表现如何？ |
| `stage32_video_feature_pairs.csv` | 每个视频、每个困难条件的候选对数量与排序结果，便于核对支持量。 |
| `stage32_choice_summary.csv` | 同一组缓存状态上，各固定评分规则纠正、误改、漏改多少帧？ |
| `stage32_same_state_choices.csv` | 上述统计的视频级明细；这里的 IoU 增益是单帧代理量。 |
| `stage32_frame_errors.csv` | 冻结身份排序器每帧选择、候选覆盖，以及正确候选对原候选的九项特征差和分数贡献。 |
| `stage32_model_coefficients.csv` | 原模型的归一化权重与还原到特征单位的有效系数。 |
| `stage32_closed_loop_attribution.csv` | 已存在的 identity 与 SGLA 独立全视频结果、首次改选和框轨迹分叉；仅 both 模式输出。 |

`stage32_attribution.json` 中的 `fit_rank_accuracy_context` 会并列列出原响应排名、冻结静态排序器和冻结身份排序器在容易帧与困难帧上的表现，供解释 93.50% 的训练准确率。

权重表用于核对原评分公式；特征单位不同，不能只按有效系数的绝对值给特征排重要性。困难帧的实际分数贡献另见 `stage32_frame_errors.csv`。

## 固定定义与计算单位

- 正确候选：使用原协议计算的 IoU≥0.5；负候选：IoU<0.1。0.1≤IoU<0.5 的候选不参加正负候选对统计。
- 可排序纠正帧 `rank_recoverable_frames`：候选 0 的 IoU<0.5，且当前保留的 Top-5 中存在 IoU≥0.5 的候选。
- 低 IoU 可纠正帧 `low_iou_recoverable_frames`：候选 0 的 IoU<0.2，且当前保留的 Top-5 中存在 IoU≥0.5 的候选。
- 候选缺失：当前保留集合中没有 IoU≥0.5 的框。这不能进一步区分目标不可见、裁剪不覆盖、响应未保留或定位框本身不准确。
- 纠正：候选 0 的 IoU<0.2、选择结果 IoU≥0.5；误改：候选 0 的 IoU≥0.5、选择结果 IoU<0.2。其他变好/变坏分别列出，不混入这两类。
- 候选对统计先在一帧内平均，再在一个视频内平均，最后视频等权平均；候选数多的视频或连续长失败不会变成更多独立视频。
- `macro_pair_accuracy` 对并列分数计 0.5；`macro_strict_pair_accuracy` 对并列计 0，与原 fit 准确率的严格大于零约定一致。置信区间只是这些已使用视频的描述性视频 bootstrap。
- `initial_cos`、`trusted_cos`、`recent_cos`、`identity_margin`、两项响应按越大越好解释；上一帧其他候选相似度、运动与尺度代价按越小越好解释。方向不随观察结果反转，也不据此选择特征。

九维特征的单项排序表现差，不能证明原始描述子或新的交互模型中不存在身份信息。

## 两个必须分清的边界

**checking 的原始 SGLA 缓存没有记录 Top-5。** 它只有原输出，因此本轮将其候选覆盖标为未知，CSV 留空；不能把填充槽误算为“没有正确候选”。八个 fit 视频的原始 SGLA 轨迹保存过完整候选，可用于原轨迹候选归因。

**不同关联方法沿着自己的预测轨迹运行。** checking identity 缓存里的候选 0 是 identity 当前父框下的原生输出，不是独立 SGLA 全视频轨迹上同一帧的输出。`stage32_choice_summary.csv` 始终按轨迹分别报告同状态比较，不能据此推算新策略的闭环 AUC。真正已有的全视频差异只放在 `stage32_closed_loop_attribution.csv`。

所有 GT 只用于离线描述。最优当前候选是一个当前状态的回顾上界，没有新 rollout，也没有持续纠正策略。

## 冻结与输出

入口继续执行原 Stage-31 契约、模型、缓存元数据、标签、历史、特征和选框审计，并加载已记录的 float32 精度兼容信息。容差不放宽。源文件在读取前后校验 SHA256；模型/输入的哈希、分析代码、NumPy 版本和固定定义写入单独的 `run.json`。

输出目录必须位于 Stage-31 源目录之外；不能覆盖原 analysis、model、manifest 或 NPZ。正常完成状态是 `COMPLETE_READ_ONLY_ATTRIBUTION`，不自动宣布某个新方向通过或失败。

结果应帮助决定下一步需要验证的具体问题：候选机会少则先研究候选生成/定位；机会存在但这些线索排序无效则先验证可辨别的身份表征；若线索已有明显信息但冻结排序器未利用，才有依据研究新的关联模型。这些是待论证的问题，不是本审计自动确立的结论。
