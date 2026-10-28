# Stage-30：纠正分支首次失效帧的响应审计

Stage-29 的一次候选纠正在 `uav0000204` 持续有效，在 `uav0000173` 下一帧即低 IoU，在 `uav0000235` 保持两帧后再次低 IoU。这一轮检查后两例失效时，真实目标是否仍在空间候选中。

本轮只重放既有分支的少量状态，使用原始 SGLA，不训练、不再替换框、不生成新闭环轨迹，也不改动 Stage-29 的未通过结论。

## 1. 固定取样

从 Stage-29 已固定的三个主分析分支中，选取未来第一个 IoU<0.2 的帧及其前一帧。没有低 IoU 帧的分支取 +1、+3 作为固定对照。根据当前结果应为：

| 视频 | 全局帧号 | 相对纠正帧偏移 | 用途 |
| --- | ---: | ---: | --- |
| `uav0000173_00781_s` | 502 | 0 | 首次低 IoU 前的上下文；此帧发生过一次输出替换 |
| `uav0000173_00781_s` | 503 | +1 | 纠正分支首次低 IoU |
| `uav0000204_00000_s` | 665 | +1 | 纠正成功分支对照 |
| `uav0000204_00000_s` | 667 | +3 | 纠正成功分支对照 |
| `uav0000235_00001_s` | 897 | +2 | 首次低 IoU 前的上下文 |
| `uav0000235_00001_s` | 898 | +3 | 纠正分支首次低 IoU |

共 6 个单帧重放点，另有原始网络一致性核验开销。代码从已冻结的分支数组计算这些点，不按新响应结果改选帧。

每个点都从原始首帧初始化模板，然后读取 **Stage-29 对应候选分支的 `previous[branch, offset]`** 作为当前裁剪父框。不会用 Stage-27 的原始父框代替，也不会把当前 GT 当父框。

核验内容包括：父框、原始模型解码框、裁剪缩放、选层、响应峰值、整数框 IoU 与全部格点解码一致性。GPU 重放只读该状态；得到的候选不推进到下一帧。

## 2. offset 0 的特殊语义

`uav0000173` 的上下文帧是 502，即原来发生人工候选替换的帧。该帧有两个不同的框：

- `native_box`：原始预测头实际解码的首选框；
- `recorded_branch_box`：Stage-29 当时写入状态的候选框。

重放应复现前者。二者不相等是实验设计的一部分。CSV 分别保存 `model_top1_iou` 和 `recorded_branch_iou`，并标记 `output_was_overridden`。绘图中 GT 为绿色、原模型输出为红色，曾被写入的替换框为橙色。其余 5 个点没有输出替换，两个框应相同。

## 3. 运行步骤

按现有 Git 流程同步新增文件。Stage-27、28、29 的脚本与数据保持冻结。

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"

export STAGE29_ROOT="$PWD/output/stage29/spatial_candidate_persistence_v1"
export STAGE30_ROOT="$PWD/output/stage30/corrected_branch_response_v1"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

python tracking/test_stage30_branch_response.py

python tracking/stage30_branch_response_collect.py \
  --input "${STAGE29_ROOT:?}" \
  --plan_only \
  --output_dir "${STAGE30_ROOT:?}"

cat "$STAGE30_ROOT/stage30_plan_summary.csv"
```

CPU 计划核验全部 Stage-29 分支与完整决策、Stage-28 缓存及其 Stage-27 来源。应显示上表 6 个点。计划绑定来源、分支文件哈希、选点规则和采集代码，不初始化 CUDA。

计划一致后直接完成 6 个点：

```bash
python tracking/stage30_branch_response_collect.py \
  --input "${STAGE29_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?请设置原始SGLA权重路径}" \
  --parameter deit_distilled --gpu 0 \
  --output_dir "${STAGE30_ROOT:?}"

python tracking/stage30_branch_response_summary.py \
  --input "${STAGE30_ROOT:?}" --render
```

上面的变量设置优先保留已有 `CHECKPOINT`；未设置时使用仓库常用路径。如果实际权重位于其他位置，填写实际路径，内容须与 Stage-27 一致。

`--render` 使用 CPU，需要 Pillow 与 OpenCV。省略该参数可以生成相同数值报告，之后再补图。

单个点完成后原子保存；重启同一命令会核验并复用已有点。只运行一个采集进程。正常退出释放 `collect.lock`；强制终止留下的锁，应确认无进程写入后再处理。不要重建上游指纹来绕过不一致。

需要分视频工程检查时，采集命令可加 `--sequence uav0000173_00781_s`；对应汇总加 `--allow_incomplete`，写入 `analysis_pilot`。正式完成后省略这两个参数。

## 4. 输出与回传

完整报告位于 `$STAGE30_ROOT/analysis`：

| 文件 | 内容 |
| --- | --- |
| `stage30_response_diagnostic.json` | 完整性、上游 Stage-29 结论、重放误差、两个首次低 IoU 点的观察 |
| `stage30_failure_responses.csv` | 只包含 503 与 898 两个关键帧的全部指标 |
| `stage30_point_responses.csv` | 全部 6 个点，含上下文和成功对照 |
| `stage30_candidate_boxes.csv` | 原始/加权候选的排名、分数、原图框及 GT IoU |
| `index.html`、`figures/*.png` | 每个点的原图、实际分支裁剪、原始与加权响应图 |

优先回传：

```bash
cat "$STAGE30_ROOT/analysis/stage30_response_diagnostic.json"
cat "$STAGE30_ROOT/analysis/stage30_failure_responses.csv"
cat "$STAGE30_ROOT/analysis/stage30_point_responses.csv"
```

并查看两个 `first_low_iou` 点及各自前一帧的图。响应图绿色格点表示该格点解码框与 GT 的 IoU≥0.5；数字是候选排名。每张图独立归一化颜色，颜色不能跨图解释为概率。

## 5. 解释范围

- 真目标仍在加权候选中，但首选错误：结合原图确认是否切换了实例；支持进一步验证持续身份关联。
- 真目标只在原始候选或全部格点中存在：区分窗口加权、局部峰提取和定位表示。
- 裁剪内无任何准确解码框：当前候选重排缺少准确输入，需要审查回归质量、表示及可见性。

“非目标框”只表示 GT IoU<0.1，可能是错误回归，不能自动标成真实干扰物。GT 覆盖率完整也不代表目标可见。

本轮没有新的方法通过门槛。`COMPLETE_CORRECTED_BRANCH_RESPONSE_AUDIT` 只表示审计完成。两个失败帧是依据已消费的开发结果选取的，不是新验证集；结果不建立新方法增益、速度或外部泛化结论。
