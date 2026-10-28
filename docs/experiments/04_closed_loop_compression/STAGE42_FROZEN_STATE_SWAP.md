# Stage-42：冻结模型与初始预测父框的交叉干预

Stage-41 显示，Stage-40 的压力损失下降未伴随平均有扰动绝对精度上升；最大损伤视频在首次低 IoU 时两组都发生错误，随后恢复轨迹却明显分化。本轮直接检验：**在同一时间交换一次记录父框，后续差异更依赖当前模型还是初始空间状态？**

这是冻结模型的机制诊断，不训练新网络、不搜索增强强度，也不建立新方法 PASS。全部8个已消耗开发视频和两seed均纳入。

## 四种组合

对每个时点，恢复原 Stage-40 对应 OOF 折、seed 的两个冻结模型，运行2×2组合：

| 组合 | 当前模型 | 首个搜索裁剪的父框 |
|---|---|---|
| CC | clean | clean 原轨迹记录父框 |
| CA | clean | augmented 原轨迹记录父框 |
| AC | augmented | clean 原轨迹记录父框 |
| AA | augmented | augmented 原轨迹记录父框 |

父框始终是原预测，不使用 GT 框。clean 父框并不一定更好；例如在受益案例中 clean 可能早已失跟。程序记录两种父框的离线 GT IoU 和首个搜索区域覆盖，不按 GT 质量筛选、替换或丢弃时点。

**只交换初始父框一次。** 首帧输出后，每个分支都使用自己的上一帧输出作为下一父框，不再返回 donor 轨迹。每个分支使用原视频第一帧模板和首框初始化，保持原窗口、head、解码和边界处理。两种模型均为物理六层静态学生，没有另一个递归记忆状态。

CC 与 AA 两个对角分支必须复现原 Stage-40 对应时间段的框和评价 IoU。该检查用于验证模板、模型恢复和实现一致，不能直接假设新分支正确。

## 固定时点与预算

每视频、每seed分别取 clean 和 augmented 的第一次持续失跟起点（沿用3帧确认规则），以及该起点后第5、10帧。另取视频1／4、1／2、3／4时点作为固定时间对照。

相同视频／seed／帧的多个来源合并为一个时点，保留全部成员标签。重复所属不同失败组，不产生额外独立样本。没有失跟的模型不制造失跟时点；超出视频结尾的延迟时点在 manifest 中标记跳过。

每个时点最多产生41帧：offset=0是第一次输出，offset=1..40是后续40帧。视频结束时截断，明确标记 `censored`。全部预算最多144个未去重时点、576条短分支、23,616个跟踪模型帧；实际数量在 prepare 时打印并冻结，去重或视频结尾会减少预算。

失跟起点使用已消耗 GT 评价结果选取，属于事后离线干预时点，不是可部署的在线检测器；不需要等待原确认帧才能安排这个诊断实验，但不能因此宣称在线及时恢复。

## 判读指标

重点对比是 `AC−AA`：**保持 augmented 模型，改用 clean 记录父框**带来的后续平均 IoU 差。它测量本次初始状态干预的影响，不能把一次框交换称为恢复了整段不同历史。

同时报告：

- `CA−CC`：clean 模型接收 augmented 父框的变化。
- `AC−CC`：从同一个 clean 父框出发，两模型的差异。
- `AA−CA`：从同一个 augmented 父框出发，两模型的差异。
- `AA−CC`：两条原生分支的差异。
- 模型两顺序平均增量：`((AC−CC)+(AA−CA))/2`。
- 父框两顺序平均增量：`((CA−CC)+(AA−AC))/2`。
- 交互：`AA−AC−CA+CC`。

两个顺序平均增量严格相加为 `AA−CC`。这些量包含分支自主续接，条件于所选初始状态；不是对原始整段历史的唯一因果分配。

每分支记录首帧 IoU、可用后续均值、完整offset1..10均值、完整offset1..40均值、offset21..40均值、完整后续低IoU比例，以及第一次连续3帧可靠／其后首次连续3帧低IoU的偏移。可靠与低IoU阈值沿用0.5和0.2。

**不足40个后续帧时，40帧指标留空；不足10帧时，10帧指标也留空。** 可用后续均值另列，不能当成完整窗口指标。观察到的可靠／复发偏移只是离线 GT 评价，不能自动解释为身份找回或在线重置规则。

分组汇总分别列出两种失败 owner 的 lag0／5／10，以及固定时间对照。先在每视频每seed内部平均同组有效时点，再要求两seed都有完整后续支持，才将该视频纳入完整40帧的配对区间。缺少失跟或窗口不足的视频列在 `excluded_videos`，不填零；不足两视频不报告区间。

bootstrap5000次、seed2121。所有区间均为已消耗开发数据、固定重叠 OOF 拟合条件下的探索性结果，不能将帧、分支、时点或seed视为独立视频。

## 服务器运行

通过 Git 同步5个新增文件后执行。模型全部复用 Stage-40，无需再训练或下载数据。

```bash
cd ~/workspace/paper
export STAGE41_ROOT="$PWD/output/stage41/cached_crop_transfer_audit_v1"
export STAGE42_ROOT="$PWD/output/stage42/frozen_model_parent_swap_v1"

python tracking/test_stage42_state_swap.py

python tracking/stage42_state_experiment.py \
  --input "$STAGE41_ROOT" --mode prepare \
  --output_dir "$STAGE42_ROOT"

python tracking/stage42_state_experiment.py \
  --input "$STAGE42_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0
```

smoke 固定使用第一个具有足够帧数的固定时间对照时点，运行四种组合各4帧。`PASS_FROZEN_STATE_SWAP_SMOKE` 只表示原轨迹复现、四组合执行和状态连续性通过。

smoke 通过后执行：

```bash
python tracking/stage42_state_experiment.py \
  --input "$STAGE42_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage42_state_summary.py --input "$STAGE42_ROOT"
```

原始 SGLA 权重仍按 `--checkpoint`、`CHECKPOINT`、tracker 配置依次查找并核对哈希。16个学生全部按原计划恢复并核对参数摘要、折、seed、训练预算和收据。runtime／窗口须与原 Stage-40 一致。

prepare、status、summary 不加载网络或 CUDA；旧权重、原图、轨迹、收据和证据文件仍须存在用于来源核验。collector 需要 GPU，进行新的冻结模型推理，所有旧结果和代码保持只读。

## 中断与结果

重复同一命令即可继续。提交单位是一个时点的四条完整分支；未完成时点重新计算，已完成包审计后复用。NPZ与哈希收据之间中断时，重跑验证数组和原轨迹后可补收据。不在同一输出目录并发 collector。

```bash
python tracking/stage42_state_experiment.py \
  --input "$STAGE42_ROOT" --mode status
```

evaluate 可加 `--sequence <原8条之一>` 分视频运行。summary 要求所有计划内包齐全，不能用部分有利案例汇总。

完成后查看或以附件提供：

```bash
cat "$STAGE42_ROOT/analysis/stage42_diagnostic.json"
cat "$STAGE42_ROOT/analysis/stage42_group_summary.csv"
cat "$STAGE42_ROOT/analysis/stage42_task_summary.csv"
```

`stage42_branch_summary.csv` 另外保留四分支的所有窗口、可靠和复发指标；`stage42_video_effects.csv` 保留每视频组别的seed支持情况，便于核对组平均数。所有原始响应／size／offset图、父框、输出框和GT评价保存在 `full/branches/`，可支持后续有明确问题的诊断。完整来源哈希放在独立的 `stage42_evidence.json`。

状态 `COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC` 只表示计划内干预完整、原生对角复现和数值记账通过。结果若显示一次父框交换只能短暂改善，就不能宣布解决长期恢复；即使恢复持续，也只是当前冻结模型和选定状态下的干预结果，还需要可部署、GT无关的识别／恢复机制与新的开发验证。当前不建立完整视频 AUC、部署加速或新方法创新结论。
