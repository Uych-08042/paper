# Stage-36：获益与受损案例的同裁剪机制定位

Stage-35B 没有证明普通监督学生稳定超过 SGLA。其平均收益集中在少数视频，同时出现可靠定位下降。本轮只定位这些变化发生在哪里，不增加训练、不调参、不选择新策略。

## 案例与问题

固定使用已经消费过结果的四个开发案例：

| 视频 | 关注理由 |
|---|---|
| `uav0000085_00000_s` | 普通监督相对 Fast 大幅获益 |
| `uav0000090_00276_s` | 普通监督相对 SGLA 大幅获益 |
| `uav0000016_00000_s` | 原 Fast 稳定，学生定位下降并新增失跟 |
| `uav0000089_00920_s` | 两个学生 seed 都明显受损 |

这是事后选例，不能把四个案例作为新的独立验证。目标是区分：目标是否仍在裁剪内、可匹配目标的框是否仍被网络产生、响应排序是否改变、尺寸/偏移回归是否改变，以及换用另一条历史轨迹的父框时结果如何变化。

## 固定窗口

每个视频按以下规则生成窗口，不能看回放结果后移动窗口：

1. Fast、SGLA、普通监督 seed2110、seed2111 各自**首次持续失跟**的起点。失跟沿用原规则：IoU<0.2 连续 3 帧。
2. 两个学生分别相对 Fast、SGLA，首次同方向 IoU 差异≥0.2 且连续 3 帧的起点；取首次较好或较差中的较早者。
3. 每个起点取前 10、后 30 帧，超出视频范围则截断。
4. 保留视频 1/4、1/2、3/4 的固定时间点，不按成功与否过滤。窗口重叠只回放一次，但记录全部归属。

窗口选取使用已经看到的 GT 结果，仅用于离线定位问题。`frame_id` 与原缓存一致：0 为初始化，1 为首个跟踪帧；实际图像文件以 `manifest.frames[frame_id]` 为准。

## 4×4 交叉回放

对每个选定帧，取四条原轨迹各自已经预测的父框，形成四个裁剪。每个裁剪只预处理一次，**同一个 search tensor**依次交给四个评估模型：

| 裁剪父框来自 | 在这个相同裁剪上评估 |
|---|---|
| 原 Fast 轨迹 | Fast、SGLA、普通监督2110、普通监督2111 |
| 原 SGLA 轨迹 | 同上 |
| 普通监督2110轨迹 | 同上 |
| 普通监督2111轨迹 | 同上 |

所有模型使用相同首帧模板。学生是对应 OOF 折的原冻结 checkpoint，该折没有训练当前视频。SGLA 使用原选层逻辑，不强制指定某层。只加载四折×两 seed 的 8 个普通监督学生网络；为来源完整性，仍核对原 Stage-35 全部 40 个最终 checkpoint 的哈希。

矩阵对角线必须重现原 Stage-35 输出：框最大绝对误差≤0.001 像素，原整数框 IoU 必须在 1e-12 内一致；密集解码与原定位头解码的最大框误差≤0.001 像素。达不到就停止，不能通过放宽容差解释成机制变化。

每个裁剪保存输入 tensor 的 SHA256，并检查模型未修改这个共享输入。逐帧检查实际定位头调用数、裁剪父框、裁剪因子、原图尺寸，以及所有输出有限。网络参数/缓冲区在回放前后必须不变。

**回放结果不会成为下一帧父框。** 每个点重新读取原预测父框，不用 GT 重置父框，不修改输出、不延续分支。因此这是单状态对照，不是新跟踪策略，也不是新一轮完整视频 AUC。

## 两种离线诊断

### 响应排序与框回归

沿用 Stage-28 原规则：3×3 局部峰、最多 5 个候选、框 NMS IoU=0.5、GT 匹配 IoU≥0.5、非目标框 IoU<0.1。原始响应与乘原窗口后的响应都保留，不调整阈值。

同一裁剪上，对学生和每条基线做四种解码：

- 基线响应＋基线尺寸/偏移：原基线输出。
- **学生响应＋基线尺寸/偏移**：仅交换选位置的响应图。
- **基线响应＋学生尺寸/偏移**：仅交换定位几何图。
- 学生响应＋学生尺寸/偏移：原学生输出。

它们使用相同位置网格、裁剪映射与原窗口，不增加神经网络前向。`score_only_gain` 和 `geometry_only_gain` 便于辨认变化来源；两者之和一般不等于完整差异，另报交互项。这是解码器探针，不是证明两个模块具有独立因果效果，也不是可部署的新方法。

若 dense 网格有高 IoU 框、top-5 保留它但 top-1 选错，支持继续检查排序。若目标仍在裁剪中但 dense 网格没有匹配框，说明在当前图像/输入状态下，仅对现有候选重排不足；不能仅凭这一条就断定是回归头、分辨率或遮挡造成。

### 模型变化与父框变化

设 `I(模型, 父框来源)` 为同帧离线 IoU，则报告两条恒等式：

```text
学生自己的输出 − 基线自己的输出
 = [学生在基线父框上 − 基线在基线父框上]
 + [学生在自己父框上 − 学生在基线父框上]

 = [学生在自己父框上 − 基线在学生父框上]
 + [基线在学生父框上 − 基线在基线父框上]
```

这区分给定裁剪下的模型差异和给定模型下换父框的差异。两种分解可能不同，因为模型与裁剪有交互；不能把其中一项当成整段历史反馈的唯一因果归因。源轨迹差异仍由原 Stage-35 完整视频结果描述。

## 服务器步骤

通过仓库同步新增文件。复用现有 `sglatrack-test` 环境和数据，无需下载新数据。

```bash
export STAGE35_ROOT="${STAGE35_ROOT:-$PWD/output/stage35/preserve_correct_oof_v1}"
export STAGE36_ROOT="$PWD/output/stage36/same_crop_mechanism_v1"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
ls -lh "$CHECKPOINT"

python tracking/test_stage36_same_crop.py

# CPU：核对 Stage-35、Stage-35B，固定窗口和来源指纹。
python tracking/stage36_same_crop_collect.py \
  --input "${STAGE35_ROOT:?}" --output_dir "$STAGE36_ROOT" \
  --plan_only
```

plan-only 不初始化 GPU。它核对旧计划、原决策、完整 96 个缓存、40 个最终 checkpoint、Stage-35B 审计来源，以及所选图像的字节哈希。Stage-36 输出目录必须与 Stage-35 分开。原脚本、数据和结果不改写。

`CHECKPOINT` 必须指向原始 SGLA 权重。若上述默认路径不存在，用 `find "$PWD" -type f -name 'sglatrack_ep0297.pth.tar'` 定位文件，再把实际绝对路径赋给 `CHECKPOINT`。新终端或重新登录后需要重新设置环境变量；此类变量未设置的 Bash 提示发生在 Python 启动之前，恢复变量即可重跑原回放命令，无需重建已冻结计划。

先做一个受损案例的工程回放：

```bash
python tracking/stage36_same_crop_collect.py \
  --input "${STAGE35_ROOT:?}" --output_dir "${STAGE36_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?请设置原始SGLA权重}" \
  --parameter deit_distilled --gpu 0 \
  --sequence uav0000089_00920_s

python tracking/stage36_same_crop_summary.py --input "$STAGE36_ROOT"
cat "$STAGE36_ROOT/analysis_pilot/stage36_diagnostic.json"
```

`PILOT_REPLAY_ONLY` 只说明已有案例通过重现审计，不是方法收益通过。检查输出中的 own-parent/dense 解码误差后，完成原定四个案例：

```bash
python tracking/stage36_same_crop_collect.py \
  --input "${STAGE35_ROOT:?}" --output_dir "${STAGE36_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage36_same_crop_summary.py --input "$STAGE36_ROOT"

export STAGE36_ANALYSIS="$STAGE36_ROOT/analysis"
cat "$STAGE36_ANALYSIS/stage36_diagnostic.json"
cat "$STAGE36_ANALYSIS/stage36_parent_summary.csv"
# 先查看每条基线自身裁剪上的两种学生解码交换，保留表头。
awk -F, 'NR==1 || $2==$3' "$STAGE36_ANALYSIS/stage36_hybrid_summary.csv"
```

最终状态 `COMPLETE_SAME_CROP_DIAGNOSTIC` 仅表示四个案例完整、重现/来源审计通过。本轮不设置新的方法 PASS/FAIL。

## 输出

- `stage36_responses.csv`：逐帧、四种父框、四个模型的覆盖率、目标投影尺寸、中心/尺度误差、候选保留、分数间隔与观察标签。
- `stage36_same_crop_summary.csv`：每视频/父框/模型的选定窗口均值。
- `stage36_phase_summary.csv`：固定时间控制点与起点帧分别统计，避免只看混合窗口均值。
- `stage36_hybrids.csv`、`stage36_hybrid_summary.csv`：逐帧和窗口平均的响应/几何交换结果。
- `stage36_parent_decomposition.csv`、`stage36_parent_summary.csv`：两种父框交叉分解及恒等式误差。
- `figures/`：每个不同起点一张 4×4 图；行是裁剪来源、列是评估模型，包含 GT、原输出、其他固定 top-5 候选与窗口响应图。没有起点的视频展示固定时间点。

若先运行单案例，结果写入 `analysis_pilot/`；完整四例另写 `analysis/`。完整报告绑定缓存、表格和图片 SHA256，重复命令审计后复用；修改或缺失已完成结果会拒绝静默覆盖。

## 解读边界与中断

表中 IoU、差异均为 0–1，**不是 0–100 的 AUC 点**；窗口均值不能替代全视频成绩。窗口重叠、帧数和两 seeds 都不增加独立视频数，本轮不做新的显著性判断。

GT 覆盖率不代表可见性；输入投影尺寸不同于原图目标尺寸。高 dense GT IoU 只是事后定位上限；非目标框不自动等于语义干扰物。图片上的各响应图分别归一化，颜色不能直接比较绝对分数。

回放按视频原子保存。中断后重跑同一命令：已完整的视频缓存审计后复用，未完成视频重算，不从中间帧续跑。Ctrl+C 正常退出释放锁；异常终止遗留锁时，先核对锁记录 PID 并确认原进程结束，不要绕过仍在运行的任务。

本地测试使用小型 CPU PyTorch 模型验证重现、一次裁剪共享、解码交换、来源不变、缓存复用和篡改拒绝；不能替代服务器真实权重的 CUDA 重现审计。
