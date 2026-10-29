# Stage-45：尺度、特征一致性与状态年龄辅助的视频检测小规模实验

日期：2026-10-29。状态：**暂不执行的检测支线**。

用户已重申核心目标是以 SGLA 为基础开展无人机跟踪研究。本方案改变了任务和主基线，因此不再是当前下一轮实验。下面保留当时的设计、实现和命令作为记录；当前无需按本文下载数据、建立检测环境或启动训练。已有本地 CPU 检查不代表已完成服务器 GPU 实验。当前研究范围见 [SGLA 无人机跟踪研究范围](../../research/SGLA_UAV_TRACKING_RESEARCH_SCOPE.md)。

## 1. 这轮回答什么问题

当视频中目标很小、历史特征可能发生错位时，给现有时序融合提供简单、可观测的可靠性线索，是否比只增加相同数量的可训练参数更有价值？

基线采用作者发布的 YOLOFT VID 检测器。本轮只考察三种线索的组合是否值得继续发展。门控与时序融合已有研究基础；本轮的新增价值需要通过对照体现，不以没有先例作为立项条件。

服务器目前只有 VisDrone-SOT。它的单目标标注不能当作完整的检测标注。本轮先补齐与官方 VID 权重类别和接口匹配的 **XS-VID v2**；后续再用 VisDrone-VID 评价无人机场景。XS-VID 包含其他来源场景，不能直接称为无人机专项验证。

## 2. 固定设计

| 项目 | 设定 |
| --- | --- |
| 官方代码 | [YOLOFT](https://github.com/gjhhust/YOLOFT/tree/9e51ac319baf0d3793694e770e049585648bae42)，固定提交 `9e51ac319baf0d3793694e770e049585648bae42` |
| 权重 | 官方 `vid.pt`；SHA256 `214a4fcb277ac527ee7c83718fc28d7dac8464f7fb82bb8a6794049206a1caa4` |
| 原生臂 | `native`：完全冻结的官方检测器 |
| 容量对照 | `capacity`：当前特征与时序残差的六个局部统计量作为门控输入 |
| 信号臂 | `signals`：前三个共用统计量，加上一帧预测尺度、同位置历史一致性、距状态重置的年龄 |
| 可训练部分 | 三个逐位置 MLP：`6→16→1`，每个 129 参数，共 **387 参数**；原检测器参数和 BN 统计全部冻结 |
| 融合方式 | 在三个官方 MSTFv1 输出处，将原时序残差乘以 `2*sigmoid(logit)`；最后层零初始化，初始增益恰为 1 |
| 状态 | 每个完整视频从空历史开始；只读取本臂历史与上一帧预测；新视频清空历史；不使用 GT 重置 |
| 开发划分 | 固定哈希顺序选取 8 个 train 视频用于训练、8 个 val_dev 视频用于检查；训练排除所有 val_dev 视频 ID 和同名图像；检查图像字节跨组选重 |
| 训练预算 | 单种子 2125；两个臂相同的 128 次更新；每次连续 8 帧，前 2 帧建立历史，后 6 帧监督；帧间截断梯度；batch=1 |
| 优化 | AdamW，学习率 0.001，weight decay=0，梯度裁剪 1；每 16 步原子保存，仅使用第 128 步权重 |
| 监督 | 使用官方检测损失；两臂相同；不训练额外分割辅助头；忽略标注不作正例，并屏蔽忽略区域内的纯背景分类损失 |
| 输入与输出 | 原始规范化的 1024×1024 图像，FP32；多标签 NMS：confidence=0.001、IoU=0.7、max_det=300；保留原 0～6 类别槽 |

三个信号的准确含义：

- **尺度**：上一帧置信度≥0.25 的预测框，在当前特征网格对应位置写入归一化的 `log(1+sqrt(w*h))`。预测缺失处为 0。它不是当前 GT 尺度，也没有借助 GT 匹配。
- **一致性**：官方 FDCN 编码后的当前特征与上一状态的同位置余弦相似度。它是历史可靠性的代理，**不是测得的光流对齐误差**。
- **年龄**：距本视频/训练片段状态重置的帧数，6 帧饱和；训练监督帧覆盖饱和值。它不是逐目标可见性、逐目标记忆寿命或遮挡标签。

两种适配臂参数量、帧抽样、初始化、损失、更新次数完全相同，且都计算两组输入特征。`signals` 用三个结构信号替换容量对照的后三个统计量，比较的是输入设计；不能单独归因到某一个信号。门控改变送往后续网络的融合输出，保留当前模块原生记忆更新；其他层状态可随本臂轨迹变化。

**公开权重曾使用 XS-VID 训练数据，val_dev 也来自这一训练分区。**这是一轮带预训练重合的开发原型，不是独立外部泛化检验。视频 ID 也只是源组代理，元数据不能保证不同 ID 来自不同原始采集。

## 3. 服务器：代码、独立环境与数据

先通过现有仓库流程将本次新增代码同步到服务器。这次修改尚未自动提交或推送。下面在服务器执行；无需下载本项目脚本压缩包，也不使用旧 SGLA `CHECKPOINT`。

```bash
cd /home/u25600003160604/workspace/paper
test -f tracking/stage45_detection_experiment.py
export PAPER_ROOT="$PWD"
export YOLOFT_ROOT="$HOME/workspace/YOLOFT-stage45"

git clone https://github.com/gjhhust/YOLOFT.git "$YOLOFT_ROOT"
git -C "$YOLOFT_ROOT" checkout --detach 9e51ac319baf0d3793694e770e049585648bae42

conda create -n uav-det45 python=3.10 -y
conda activate uav-det45
cd "$YOLOFT_ROOT"
python -m pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -c config/environment/cu118.txt
python -m pip install setuptools==72.1.0 wheel==0.44.0
```

接着按作者工具下载数据与发布权重。数据可用 ModelScope 镜像；权重使用 Hugging Face。工具会验证发布文件、解压数据并创建官方目录，不需要手动拼接 VisDrone-SOT 文件夹。

```bash
python tools/xsvid/download_release.py \
  --hub ms --component dataset --destination "$HOME/workspace/datasets"
python tools/xsvid/download_release.py \
  --hub hf --component model --destination "$HOME/workspace/models"

export XSVID_ROOT="$HOME/workspace/datasets/XS-VID-v2"
export VID_WEIGHTS="$HOME/workspace/models/YOLOFT-XSVID-v2-weights/vid.pt"
```

若数据镜像不可用，可将数据命令的 `--hub ms` 换成 `--hub hf`，输出目录保持相同。模型包还包含 MOT/SOT 权重，本轮只读取 `vid.pt`。

YOLOFT 需要编译官方 CUDA 算子。按作者说明使用 **CUDA 11.8 开发工具链和 GCC/G++ 11**；GPU 驱动版本不能代替 `nvcc`。下面的 CUDA 路径应对应服务器实际安装位置；若没有该工具链，先解决这一环境依赖，再运行 smoke。

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=gcc-11 CXX=g++-11 MAX_JOBS=4
nvcc --version
"$CC" --version
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"

cd "$YOLOFT_ROOT"
python -m pip install --no-build-isolation --no-deps ./ultralytics/nn/modules/ops_dcnv3
python -m pip install --no-build-isolation --no-deps ./ultralytics/nn/modules/alt_cuda_corr_sparse
python -c "import DCNv3, alt_cuda_sparse_corr; print('CUDA operators imported')"
export YOLO_AUTOINSTALL=false
```

旧 `sglatrack-test` 环境、SOT 数据和 Stage-1～44 结果保持原样。不要运行官方 `train_vid.py` 代替下列入口：它的默认训练/验证配方面向官方复现，不是本轮的 train/val_dev 开发协议。

## 4. 准备与 smoke

```bash
cd "$PAPER_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export STAGE45_ROOT="$PAPER_ROOT/output/stage45/xsvid_temporal_signal_v1"

python -B tracking/test_stage45_detection.py --upstream "$YOLOFT_ROOT"

python -B tracking/stage45_detection_experiment.py --mode prepare \
  --data_root "$XSVID_ROOT" --yoloft_root "$YOLOFT_ROOT" \
  --weights "$VID_WEIGHTS" --output_dir "$STAGE45_ROOT"

python -B tracking/stage45_detection_experiment.py \
  --mode smoke --input "$STAGE45_ROOT" --gpu 0

cat "$STAGE45_ROOT/smoke/decision.json"
```

`prepare` 固定源码、权重、标注、顺序和所选图像哈希，输出必须为空或已有完全相同的 manifest。只读取 `train.json` 与 `val_dev.json`；本轮入口不会评估官方 test。选中视频不足、帧索引不连续、类别不符合官方发布格式或跨组图像重复时会给出具体错误。

smoke 检查完整的官方网络：两个臂的恒等初始化应逐帧复现原始解码，两个臂都能获得梯度并更新门控，训练后首帧仍保持原生状态重置行为，检测器权重/BN 不变。**smoke 通过只说明执行正确。**其临时更新不用于正式训练。

## 5. 两臂训练与一次完整检查

```bash
python -B tracking/stage45_detection_experiment.py \
  --mode train --arm capacity --input "$STAGE45_ROOT" --gpu 0
python -B tracking/stage45_detection_experiment.py \
  --mode train --arm signals --input "$STAGE45_ROOT" --gpu 0

python -B tracking/stage45_detection_experiment.py \
  --mode evaluate --input "$STAGE45_ROOT" --gpu 0

python -B tracking/stage45_detection_summary.py --input "$STAGE45_ROOT"

cat "$STAGE45_ROOT/analysis/stage45_decision.json"
cat "$STAGE45_ROOT/analysis/stage45_policy_summary.csv"
cat "$STAGE45_ROOT/analysis/stage45_paired.csv"
cat "$STAGE45_ROOT/analysis/stage45_video_summary.csv"
```

总计两个小模型、24 条完整视频轨迹。每帧恰好一次检测头调用，无额外检测器、GT 重置或未来帧输入。训练状态只来自本次连续片段中的自身历史；完整检查只来自本视频的自身历史。

中断后重复同一条命令：训练读取最后一个已提交的固定步 checkpoint；检查跳过已提交视频并从头重做未提交的视频。checkpoint 和结果都绑定原始文件 SHA，不能把读得开的被修改文件当成原文件继续。每个输出目录只运行一个任务；意外退出遗留 `run.lock` 时先确认其中 PID 已退出，再删除该锁文件。不要在冻结计划后更新实验源码或切换上游版本。

```bash
python -B tracking/stage45_detection_experiment.py \
  --mode status --input "$STAGE45_ROOT"
```

## 6. 如何解释结果并继续发展

主要比较是 **signals−capacity 的极小目标 AP（AP_es）**，同时看 signals−native 与整体 AP。整体 AP、AP_es、12～20 和20～32像素尺度 AP 采用固定提交中的官方 XS-VID COCO 评估器，保留原类别和 crowd/ignore 标注。这里的尺度基于规范化图像上的框面积，不能混写成无人机原始传感器分辨率。

配对区间按视频 ID 重采样 256 次；每次重复整段视频的逐图像匹配记录，再调用原评估器重新累积 pooled AP。它不是平均每视频 AP，也不把帧当成独立样本。单种子、预训练重合和源组代理仍限制解释范围。

补充指标为 score≥0.25、同类 IoU≥0.5 的一对一匹配召回、误报和连续漏检。补充匹配将预测框面积至少 50% 落入 ignore/crowd 区域的未匹配预测排除误报；这一附加规则与官方 AP 分开。只有实际存在 `track_id` 时才报告连续漏检，标注断开或目标消失时终止该段。≥5 帧漏检是离线描述，不进入控制器。

本轮采用 AP_es +1 点、整体 AP 相对普通适配和原生模型都不低于 -0.3 点作为**值得继续投入的实践参考**；至少 4 个视频、200 个极小目标标注才有基本尺度支持。小样本区间跨零时保留效应、支持量和不确定性，不把整个方向判死。`PROMISING_DEVELOPMENT_SIGNAL` 是开发结果标签，不是正式方法认证；其他结果输出 `REVIEW_DEVELOPMENT_SIGNAL_RESULT`。

后续可以按证据发展：

1. 若信号臂比容量对照稳定改善，下一轮拆分三个信号，检查收益集中在尺度变化、历史不一致还是短期状态建立阶段，再规划独立 VisDrone-VID 检查。
2. 若两个适配臂都改善而差别小，说明当前证据更支持普通轻量适配；可进一步研究信号定义是否过粗，尤其是将同位置一致性发展为对齐后的可靠性。当前年龄只标记早期状态建立，长期退化研究需要发展局部更新年龄或其他可观测的记忆可靠性信号。
3. 若 AP 改善但连续漏检或误报恶化，定位受影响的视频和门控变化，研究如何避免历史噪声扩散。
4. 若尺度支持不足或区间过宽，在查看新增结果前固定补充数据来源和样本规模；不要在当前 checking 结果上反复挑阈值、步数、最好 checkpoint 或最好种子。

时间记录为 FP32 网络、门控特征计算和 NMS 的同步耗时，排除每视频前8帧、图像读取/转换和结果写盘；提供 median/p95。这里只用于成本观察，不是机载时延、能耗或部署加速结论。

## 7. 本地检查与保留内容

本地以 CPU Torch 2.7.1 对固定提交中的真实 MSTFv1/FDCN/Conv3D 定义和 CPU DCNv3 参考实现进行检查；未下载模型/数据、未执行完整 GPU 检测。六项测试覆盖信号来源、387 参数、官方融合恒等复现与梯度、精确 CPU 续训与文件篡改拒绝、划分隔离、三臂汇总完整性、官方 AP 重采样与真正复制图像的评估一致性、空预测和连续漏检。`video_summary.csv` 的逐视频 AP 是影响诊断，主表始终使用全部检查图像的 pooled AP。

保留 `manifest.json`、smoke、全部已提交 checkpoint/receipt、`evaluation/run.json`、三个臂的逐视频预测/receipt、analysis 表。原 SGLA 所有冻结结果继续保留。官方数据和权重不提交到本项目 Git。

来源：[作者运行说明](https://github.com/gjhhust/YOLOFT/blob/9e51ac319baf0d3793694e770e049585648bae42/docs/QUICKSTART.md)、[官方融合实现](https://github.com/gjhhust/YOLOFT/blob/9e51ac319baf0d3793694e770e049585648bae42/ultralytics/nn/modules/block.py)、[官方评估器](https://github.com/gjhhust/YOLOFT/blob/9e51ac319baf0d3793694e770e049585648bae42/ultralytics/data/cocoeval_xs_vid.py)。
