# Stage-50：共同预测父框 × 冻结预测头重放

状态：实现及4项CPU测试通过；服务器CUDA运行待执行。主任务保持原生SGLA上的无人机单目标跟踪。

## 本轮要解决的问题

[Stage-49](STAGE49_RESULTS_REVIEW.md)把现象分成了当帧中心错误、尺寸失真、恢复与复发。Stage-50固定这些历史状态，在同一父框、图像、原生特征和选层下比较已有预测头，区分当前头输出差异与输入历史差异。

同时核对三个位置：普通回归使用的raw最高响应格点、实际执行的Hann加权最高响应格点、额外尺寸约束使用的GT中心格点。记录各处响应、size、offset及解码框；不能仅因三者不同就认定训练设计有害。

所有模型与历史均复用Stage-48。原生骨干、选择器、候选块与各预测头保持冻结；不训练、不滚动分支。父框来自各模型已记录的上一帧预测，GT仅用于首帧模板初始化及离线评价。

## 固定输入与预算

| 案例 | 视频简称 | 固定帧 | 历史父框来源 | 关注问题 |
|---|---|---|---|---|
| C01 | 0304 | 552,553,554,555,556 | native、rank2132、shape2133 | 早期中心错误 |
| C02 | 0304 | 1076,1078,1081,1083,1085 | native、组合2132、组合2133 | 晚期尺寸失真 |
| C03 | 0204 | 662,663,664,665,666 | native、组合2132、组合2133 | 准确父框下的输出跳变 |
| C04 | 0226 | 328,329,332,338,339 | native、supervised2132、rank2133、组合2132 | 恢复分岔 |
| C05 | 0176 | 238,239,240,241,242 | native、组合2132、组合2133 | 获益种子的恢复窗口 |
| C06 | 0307 | 470,471,474,486,487 | native、组合2132、组合2133 | 恢复后复发 |

这里“组合”指`rank_shape`。每个输入比较9个头：native，以及当前视频所在held-out折的supervised／rank／shape／rank_shape × 两种子2132、2133。不存在根据当前帧结果挑选checkpoint或种子。

总计95个父框／帧输入。每个输入只执行一次原生SGLA完整前向；另外8个头直接接收其**原始GPU头输入张量**，不从CPU重建或改变布局。检查各头执行前后共享特征字节不变，BN保持eval。

- 95次完整前向，760次仅预测头前向，合计855次实际头调用。
- 原生每次执行6个前缀块及1个独立候选块，合计665次骨干块调用。
- smoke只做C01的15个输入：15次完整前向＋120次仅头前向，合计135次头调用。正式采集复用这批缓存。
- 每个输入的9头共享特征；不同历史、不同帧仍分别提取特征。这不是部署计算量或速度试验。

每个“历史×自身模型”的对角线须复现源轨迹：浮点框最大误差≤0.001、raw峰幅误差≤0.00001、官方整数框与IoU完全相同、原生选层相同。完整前向与头调用均用真实模块hook计数。smoke通过只说明执行与重放一致性。

Stage-50先核验Stage-48的全部108条来源轨迹、正式模型与源结果表，再绑定已完成的Stage-49报告、源码和8张表。固定锚点须与Stage-49事件记录吻合。旧阶段源码和缓存不改写。

## 服务器执行

先按既有Git流程同步新代码。在仓库根目录设置路径，避免沿用指向`~/workspace/output`的过期变量：

```bash
cd ~/workspace/paper
conda activate sglatrack-test
export STAGE48_ROOT="$PWD/output/stage48/native_head_stability_v1"
export STAGE50_ROOT="$PWD/output/stage50/shared_parent_frozen_heads_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage50_crossed.py
python -B tracking/stage50_crossed_collect.py \
  --mode prepare --input "$STAGE48_ROOT" --output_dir "$STAGE50_ROOT"
python -B tracking/stage50_crossed_collect.py \
  --mode smoke --input "$STAGE50_ROOT" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0
cat "$STAGE50_ROOT/smoke/stage50_smoke.json"
```

出现`PASS_SHARED_PARENT_HEAD_SMOKE`后执行：

```bash
python -B tracking/stage50_crossed_collect.py \
  --mode collect --input "$STAGE50_ROOT" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0
python -B tracking/stage50_crossed_summary.py --input "$STAGE50_ROOT"

cat "$STAGE50_ROOT/analysis/stage50_crossed_matrix.csv"
cat "$STAGE50_ROOT/analysis/stage50_focus.csv"
```

完成状态为`COMPLETE_SHARED_PARENT_HEAD_DIAGNOSTIC`。中断后重跑相同采集命令；已提交缓存重新核验后复用。保持同一来源、代码、权重与输出root；验证错误应先定位，不通过扩大容差或改旧报告绕过。

## 输出与后续用途

| 文件 | 内容 |
|---|---|
| `stage50_crossed_matrix.csv` | 25行关注输入，每行包含9个头的当帧IoU，比较同父框上的头差异 |
| `stage50_focus.csv` | 同25行的自身模型结果：中心误差、宽高、raw／Hann／GT格点、Top-5保留与dense上界 |
| `stage50_responses.csv` | 855行完整头响应摘要，包含全95个输入 |
| `stage50_cell_positions.csv` | raw／Hann／有效GT位置的响应、size、offset及各自解码结果，最多2565行 |
| `stage50_diagnostic.json` | 来源指纹、全部缓存／表哈希、调用预算和最大重放误差 |

先提供两张25行表即可；其余长文件按需附加。GT中心在实际裁剪之外时不填GT格点，不以边界格点冒充有效GT位置。尺寸图对应的物理宽高在图像裁剪前计算，实际输出框另按原解码／裁边／整数IoU口径评价。

读数用于具体决定下一轮训练对照：

- 若执行格点的尺寸明显失真，而GT格点尺寸较好，可比较执行位置对齐的回归与既有raw位置回归，固定其余预算。
- 若相同输入下某些头可选中目标，另一些选错，可围绕实际Hann排序和几何一致性组织训练对照。
- 若主要差异来自历史父框，或失败历史上各头都难以产生合适候选，可推进纳入自身预测状态的匹配训练，区分候选可用性与恢复能力。

这些是事后选择的开发窗口；行、帧、头和种子不构成独立验证样本。共同父框下的一次正确输出不能证明后续恢复；完整恢复仍需新的闭环对照。当前不产生新方法采用门槛、全视频AUC收益或速度结论。

## 本机验证

4项测试通过。纯NumPy部分核查已完成Stage-49的来源绑定、锚点与95／855预算；真实CPU Torch、OpenCV和小型CENTER头部分覆盖实际模块hook、原解码函数、共同输入、smoke／正式采集／汇总／缓存复用、自身模型对角线、raw与Hann位置不同、GT出裁剪，以及缓存／结果篡改拒绝。使用合成轨迹与小模型，不是服务器SGLA精度复现。Stage-1～49实验Python源码保持不变。
