# Stage-4A/B: 评估口径、计算开销与独立训练数据

本次实现下一阶段的前两步，不改原模型结构、训练代码或 checkpoint。

- **Stage-4A**：统一离线融合的 IoU 口径，测量 Fast、SGLA 和额外 Fast head 的实际开销。
- **Stage-4B**：在 GOT-10k/LaSOT 的训练视频中固定划分视频级 train/val，采集冻结模型的候选前特征和收益标签。
- 尚未实现 Stage-4C 的新路由器训练、在线部署或边缘卸载。先用下面的小规模实验确认数据与计算条件，再进入训练。

## 1. 为什么先做这些

已有完整闭环结果为 Fast 64.4089、SGLA 66.5143、median4 64.7181、median6 66.1426。
当前证据不足以支持“多候选融合已优于 SGLA”，也不能把共享 SGLA 轨迹上的 Oracle 当成闭环性能上限。
下一步的问题是：**更充分的候选前特征，能否预测额外候选块的收益，并在自身轨迹上维持准确率与速度？**

检查代码时还发现，原离线融合评估直接用浮点预测框，而基线评估将预测框坐标截断为整数。
两者不适合直接比较。此次将离线融合改为相同口径：预测 xywh 向零截断、GT 保持浮点、inclusive-pixel IoU。
候选框之间用于一致性打分的 IoU 仍保持浮点，不改变选择规则。
此前的离线融合 AUC 需要重新汇总；Stage-1/2 候选预测及已有闭环 CSV 不需要因此重新采集。

## 2. 文件清单

| 文件 | 用途 |
|---|---|
| `tracking/validation_metrics.py` | 无 PyTorch 依赖的统一 IoU |
| `tracking/stage1_validate.py` | 原 `_iou_xywh` 改为调用统一实现，正常输入的数值口径不变 |
| `tracking/stage3_post_candidate_validate.py` | 修正融合框评估，输出 `iou_protocol` |
| `tracking/stage4_runtime.py` | 冻结网络的分路径计算，支持显式 checkpoint |
| `tracking/stage4_benchmark.py` | CUDA 同步计时和原生 SGLA 耗时对照 |
| `tracking/stage4_manifest.py` | 生成固定视频划分与连续片段清单 |
| `tracking/stage4_collect.py` | 采集 SGLA/Fast 各自轨迹的特征与标签 |
| `tracking/stage4_common.py` | 指纹、序列布局、原子保存、缓存校验 |
| `tracking/stage4_check_cache.py` | CPU 检查完整性和状态/标签一致性 |
| `tracking/test_stage4.py` | CPU 合成数据测试，不需要模型权重 |

## 3. 服务器准备

以下命令在服务器仓库根目录、已跑通的 `sglatrack-test` 环境中运行。
新脚本沿用原验证入口的兼容处理，不必再次修改 `lib/`。仅加载你信任来源的 checkpoint；原验证入口对旧 checkpoint 使用完整反序列化。

```bash
cd /home/u25600003160604/workspace/paper
conda activate sglatrack-test

export OUTPUT_DIR="$PWD/output/test/tracking_results/sglatrack/deit_distilled/stage2_candidate_validation/uav123"
export STAGE4_DIR="$PWD/output/stage4"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B -m unittest discover -s tracking -p test_stage4.py -v
```

CPU 测试通过不代表 CUDA 模型已经验证。新 GPU 入口先用小样本运行。
退出终端后环境变量会消失，重新登录需要重新设置。

## 4. Stage-4A-1: 重算离线融合

使用新目录保存结果，保留原汇总方便对照，不覆盖旧实验记录。

```bash
python tracking/stage3_post_candidate_validate.py \
  --input "$OUTPUT_DIR" \
  --output_dir "$STAGE4_DIR/post_candidate_metric_fixed" \
  --folds 5 --seed 2026 --weight_step 0.1
```

检查新 `uav123_stage3_post_candidate_summary.csv`：

- `iou_protocol` 应为 `sgla_int_prediction_inclusive_v1`。
- Fast、固定候选及选择策略读取原 CSV IoU，原则上不变；融合策略需要以本次重算值为准。
- 仍然是共享轨迹诊断，不是闭环 AUC，不据此宣布最终提升。

## 5. Stage-4A-2: 测量延迟

```bash
python tracking/stage4_benchmark.py \
  --parameter deit_distilled --checkpoint "$CHECKPOINT" \
  --dataset_name uav123 --sequence 0 --gpu 0 \
  --frames 50 --repeats 3 --warmup 10 \
  --output_dir "$STAGE4_DIR/profile_seq0"
```

脚本自动核对重建的 SGLA 路径和原网络输出。输出：

```text
profile_seq0/
  benchmark_raw.csv
  benchmark_summary.csv
  benchmark_metadata.json
```

汇总列：`scope,operation,samples,mean_ms,p50_ms,p95_ms`。

| operation | 含义 |
|---|---|
| `native_sgla` | 原 `tracker.track`，计时包含其内部状态更新 |
| `sgla` | 分解重建的 SGLA 单候选路径，结果对齐原网络 |
| `fast` | L1-L6 + Fast head + 解码 |
| `fast_then_sgla` | Fast head/解码后，再计算原选择器和一个候选及其 head/解码 |
| `prefix_L1_L6` | patch embedding 与前六层 |
| `norm_fast_head` | Fast 的 norm 与预测 head |
| `pool_all_channels_grid4` | 全通道 token 池化的候选前特征成本 |
| `block_L7` 到 `block_L12` | 独立候选块，含输入 clone，不含 head |

还输出裁剪预处理、选择器、Fast 解码的分项耗时。分项有独立同步开销，不能直接相加当作端到端速度。
整体路径包含裁剪、预处理和解码，不包含读图。每个测量样本前后 CUDA 同步，各路径使用相同 SGLA 状态，原生跟踪调用的状态修改会在计时外恢复。
这里的 `fast` 是计算成本对照，不是 Fast 自身轨迹的性能评估。没有计算新路由器成本，也没有声称模型级 FLOPs 降幅等于实际加速。

建议在 GPU 无其他任务时，再测序列 1、2，分别使用 `profile_seq1`、`profile_seq2` 新目录；短测仅用于可行性诊断。
若 Fast 相比原生 SGLA 已几乎没有耗时优势，后续“Fast head + 新路由器”很可能无法通过跳层获得显著加速，需要考虑仅 token 特征的门控。
如果有余量，则在路由器实现后测量真实混合路径，不能仅凭此处差值宣布达到速度目标。

## 6. Stage-4B-1: 准备训练源

**不要使用 UAV123 构造训练缓存。** 当前 UAV123 已用于较多探索，后续正式泛化评估还需要未参与选择策略的测试集。
本次 val 是路由器拟合之外的视频，不代表冻结的 SGLATrack 主干从未见过这些训练数据。

二选一：

- GOT-10k：`--root` 指向含完整 `list.txt` 的官方 **train** 目录。脚本再按仓库 `got10k_train_split.txt` 过滤。
- LaSOT：`--root` 指向含 `airplane/airplane-1/` 等目录的根。只采样仓库 `lasot_train_split.txt` 中的视频。

GOT-10k 布局：

```text
train/list.txt
train/GOT-10k_Train_000001/groundtruth.txt
train/GOT-10k_Train_000001/absence.label
train/GOT-10k_Train_000001/cover.label
train/GOT-10k_Train_000001/00000001.jpg
```

LaSOT 每个视频包含 `groundtruth.txt`、`full_occlusion.txt`、`out_of_view.txt`、`img/00000001.jpg`。
脚本会检查标注长度和片段内图片，缺失训练视频可以跳过；已选视频缺图则报错，不静默填充。

### 低存储模式

Stage-4 不要求服务器保存完整 GOT-10k 或 LaSOT。优先保留较多视频的较短片段，
因为相邻帧高度相关，视频多样性通常比单个视频的超长片段更有价值。

根目录可以只含准备参与训练的序列。将这些序列名写入文本文件后使用
`--sequence_file`；脚本会将名称与仓库的正式训练划分交叉检查，因此不会接受
GOT-10k test 视频或 LaSOT 非训练视频。此模式下 GOT-10k 不需要完整 `list.txt`。

```bash
mkdir -p "$STAGE4_DIR"

python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
split = Path("lib/train/data_specs/got10k_train_split.txt")
allowed = {
    "GOT-10k_Train_{:06d}".format(int(line) + 1)
    for line in split.read_text().splitlines()
    if line.strip()
}
installed = sorted(
    path.name for path in root.glob("GOT-10k_Train_*") if path.is_dir()
)
eligible = [name for name in installed if name in allowed]
held_out = [name for name in installed if name not in allowed]
output = Path(os.environ["STAGE4_DIR"]) / "got10k_installed_train.txt"
output.write_text("\n".join(eligible) + "\n")
print("installed={}, eligible={}, held_out={}".format(
    len(installed), len(eligible), len(held_out)
))
print("sequence file: {}".format(output))
PY

python tracking/stage4_manifest.py \
  --dataset got10k_train --root "$TRAIN_ROOT" \
  --sequence_file "$STAGE4_DIR/got10k_installed_train.txt" \
  --train_sequences 30 --val_sequences 10 --clip_frames 256 \
  --seed 2026 --output "$STAGE4_DIR/subset_manifest.json"
```

不能直接把所有已安装目录写入 `--sequence_file`，因为其中可能含有仓库预留的
GOT-10k 验证序列；上面的交集步骤会将它们排除，但不会删除任何数据。

至少需要安装 `train_sequences + val_sequences` 个可用视频。先保留所选视频的完整
标注文件和完整图像目录以生成 manifest；manifest 固定后，采集只读取每个条目
`[start, stop)` 范围内的连续图像。若需要进一步裁剪磁盘占用，可在另一台存储充足的
机器上按 manifest 制作片段包，再将片段包传到服务器。不要手工修改 manifest，
因为根路径、标注哈希和划分都包含在实验指纹中。

## 7. Stage-4B-2: 先做 5/2 视频冒烟实验

下面的 GOT-10k 路径只是示例，必须替换成服务器真实训练目录；目前只确认过你的 UAV123 路径。

```bash
export TRAIN_ROOT="/home/u25600003160604/workspace/datasets/GOT10K/train"

python tracking/stage4_manifest.py \
  --dataset got10k_train --root "$TRAIN_ROOT" \
  --train_sequences 5 --val_sequences 2 --clip_frames 128 \
  --seed 2026 --output "$STAGE4_DIR/smoke_manifest.json"

python tracking/stage4_collect.py \
  --manifest "$STAGE4_DIR/smoke_manifest.json" \
  --parameter deit_distilled --checkpoint "$CHECKPOINT" --gpu 0 \
  --trajectories sgla fast --pool_grid 4 --verify_sgla \
  --output_dir "$STAGE4_DIR/smoke_cache"

python tracking/stage4_check_cache.py --input "$STAGE4_DIR/smoke_cache"
```

使用 LaSOT 时，清单命令改成 `--dataset lasot_train --root /实际路径/LaSOT`，其余命令不变。
每视频只抽一个连续片段；随机种子固定，首帧须有有效可见目标。视频短于 128 帧时使用整个视频。
只在片段首帧用 GT 初始化，之后逐帧传播预测，失败也不使用 GT 重置。
当前帧不可见/无有效 GT 时保留其状态与特征，但 `valid_gt=False`、`iou=-1`、`delta=NaN`，训练时必须排除这些标签。

正常输出大致为：

```text
[done] GOT-10k_Train_...: 127 decisions -> .../train/...__sgla.npz
[done] GOT-10k_Train_...: 127 decisions -> .../train/...__fast.npz
Newly completed video-policy clips: 14
PASS: disjoint video splits, complete clips, finite causal features, state continuity, IoU/delta labels
split  trajectory  videos  decisions  valid
train  fast             5        635  <实际有效数>
train  sgla             5        635  <实际有效数>
val    fast             2        254  <实际有效数>
val    sgla             2        254  <实际有效数>
```

上述帧数假设七个视频都至少有 128 帧；短视频会更少。有效标签数不能预先保证。
完整检查要求清单内两种轨迹全部存在。仅运行一个 shard 或一个序列后检查，会报告缺少文件，这是正常的完整性约束。

## 8. 缓存格式与因果边界

```text
smoke_cache/
  run.json
  cache_summary.csv
  train/<sequence>__sgla.npz
  train/<sequence>__fast.npz
  val/<sequence>__sgla.npz
  val/<sequence>__fast.npz
```

每个 NPZ 一个视频片段的一种轨迹，用 `np.load(path, allow_pickle=False)` 读取，不需要解压到 CSV。
令 N 为跟踪帧数，不包含初始化帧；C 为实际 embedding 通道数；S 为 search token 网格边长。

| 字段 | 形状与用途 |
|---|---|
| `token_features` | `[N,(2+g*g)*C]`，FP16；template mean/max + search 的 g×g 全通道平均池化 |
| `fast_score_map` | `[N,1,S,S]`，FP16；Fast 原始 score map，未乘 Hann window |
| `selector_prob` | `[N,6]`，原选择器输出；不是强制和为 1 的概率分布 |
| `previous_state`, `previous_fast_box` | `[N,4]`，上一跟踪状态和上一帧 Fast 预测 |
| `previous_action` | `[N]`，上一步执行动作，片段第一个决策为 -1 |
| `fast_box`, `image_hw`, `resize_factor` | 当前 Fast 结果、原图尺寸与裁剪尺度，可用于因果几何特征 |
| `action_boxes` | `[N,7,4]`，同一当前裁剪上 Fast/L7-L12 框，仅用于标签/诊断 |
| `iou` | `[N,7]`，动作順序固定为 Fast,L7,L8,L9,L10,L11,L12 |
| `delta` | `[N,6]`，每个候选 IoU 减去当前 Fast IoU |
| `gt`, `valid_gt` | `[N,4]`、`[N]`，仅监督和有效标签屏蔽 |
| `selected_layer`, `executed_action`, `frame_id` | 当前原选择层、真正推进状态的动作、原视频 0-based 帧号，用于核验 |
| `metadata_json` | Unicode 标量 JSON，包括视频清单条目、初始框、轨迹、实验指纹 |

动作编码为 Fast=0、L7=1、...、L12=6。SGLA 轨迹执行原选择器所选候选，Fast 轨迹始终执行动作 0。
六个候选都从同一份 L6 tokens 独立计算，**不是顺序执行 L7 到 L12**。
采集会执行全部候选以取得监督标签，其耗时不能作为在线部署耗时。

`run.json` 显式记录 `feature_fields` 和 `label_fields`。后续训练必须白名单读取 feature_fields，不能把整个 NPZ 拼成输入。
候选框、当前 GT/IoU/delta、未来帧信息都不得进入路由器。
`fast_box` 和 Fast score map 虽然是候选前特征，但需先执行 Fast head，成本已经在 Stage-4A 单独测量。
Fast 与 SGLA 同视频的全部轨迹必须属于同一 split，不能随机按帧拆分。

可选 `--save_full_tokens` 额外保存 `[N,Lz+Lx,C]` 的 FP16 原始 L6 tokens，方便后续训练空间特征编码器，但更占磁盘。
默认 tiny、320 tokens、192 channels 时，仅完整 tokens 的未压缩成本约 120 KiB/决策帧；默认池化特征约 6.75 KiB/决策帧。
这些仅为对应形状的估算，实际 shape 写入 run.json；不需要在代码中硬编码通道数。

## 9. 恢复、分片与扩展

重复相同采集命令会检查并跳过已完成 NPZ；只有完整写入后才会原子替换目标文件。
`--overwrite` 只覆盖同一实验配置的片段。清单、权重、特征设置、采集代码或相关运行环境变化后，使用新目录，不能混合缓存。
`run.json` 记录 checkpoint/config、PyTorch/CUDA/GPU 型号、清单和采集脚本指纹；不会哈希所有原图或整个仓库。生成清单后请保持数据和模型源代码不变。
进程异常退出可能留下 `.tmp` 或 `run.lock`；确认没有运行中的采集进程后再人工清理残留，不能在其他进程运行时删除锁。
不要把较大的 `.npz` 缓存提交到仓库；代码与小型汇总足够用于交流。

小实验通过后，扩到 80/20 视频、每视频最多 512 帧：

```bash
python tracking/stage4_manifest.py \
  --dataset got10k_train --root "$TRAIN_ROOT" \
  --train_sequences 80 --val_sequences 20 --clip_frames 512 \
  --seed 2026 --output "$STAGE4_DIR/pilot_manifest.json"

python tracking/stage4_collect.py \
  --manifest "$STAGE4_DIR/pilot_manifest.json" \
  --parameter deit_distilled --checkpoint "$CHECKPOINT" --gpu 0 \
  --trajectories sgla fast --pool_grid 4 --verify_sgla \
  --output_dir "$STAGE4_DIR/pilot_cache"

python tracking/stage4_check_cache.py --input "$STAGE4_DIR/pilot_cache"
```

这是一个新的 pilot 划分；不要把 smoke 缓存合并进去，视频可能在两个清单中角色不同。
pilot 是初步可学习性验证规模，不是最终充分的训练集。
多 GPU 时，对相同命令分别加 `--num_shards 2 --shard_id 0 --gpu 0` 和 `--num_shards 2 --shard_id 1 --gpu 1`。
输出目录相同，视频按清单分片；必须保持其余参数一致，不要同时启动重复 shard。当前配置指纹要求 GPU 型号一致，异构 GPU 请分别采集并另行评估数值一致性。

## 10. 这一轮需要返回哪些结果

先运行小规模实验，返回：

1. 修正后的 `uav123_stage3_post_candidate_summary.csv` 中融合方法结果。
2. `profile_seq0/benchmark_summary.csv`，最好同时有另两段视频的结果。
3. `stage4_check_cache.py` 完整终端汇总，及 `smoke_cache/cache_summary.csv`。
4. `run.json` 中数据源、embedding/search 尺寸、所用 checkpoint 和 GPU 型号。

暂时不需要上传大 NPZ。仅有 UAV123 而没有训练数据时，先完成 Stage-4A，不拿 UAV123 代替训练源。
Stage-4B 的检查通过说明缓存可用于下一步实验，不说明收益已经可预测。
后续 Stage-4C 应从六维收益回归和 Fast=0 的动作选择出发，在固定 val 视频上校准，再做真正闭环与硬件延迟验证；训练可学习性、闭环泛化和端到端加速是三个独立验收项。
