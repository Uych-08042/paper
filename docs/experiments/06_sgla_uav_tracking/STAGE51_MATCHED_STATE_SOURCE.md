# Stage-51：拟合组预测状态的匹配训练对照

状态：实现与CPU验证完成，服务器训练待执行。主任务继续采用原生SGLA的首帧框初始化单目标跟踪。

## 问题与固定对照

[Stage-50](STAGE50_RESULTS_REVIEW.md)观察到，多数恢复分岔随历史父框对应的输入变化；也存在同一输入上原生头可定位、适配头失败的例外。下一步检验：训练只覆盖原生轨迹附近的状态，是否遗漏了适配模型自身会遇到的中心／尺度组合。

本轮先固定普通监督，比较训练状态来源。保持原骨干、选择器、独立候选块、头结构与全部BN缓冲；仅训练原CENTER头参数。

| 臂 | 每个同帧训练对的视图0 | 视图1 |
|---|---|---|
| `native_replay` | 原生预测父框 | 同一个原生父框的重复视图 |
| `state_mix` | 同一个原生预测父框 | 当前外折fit组的固定监督模型预测父框 |

两臂均从原始SGLA头初始化，使用相同帧抽样、随机顺序、两视图数、AdamW、256次更新和普通`2*GIoU+5*L1+focal`监督。学习率1e-5、weight decay1e-4、梯度裁剪1；每步2个帧对、共4个视图。回归仍使用既有raw最高响应框，推理仍使用原生Hann解码；不同时改变损失位置、附加排序或尺寸项。

两个相同原生视图使控制臂与处理臂的原生监督权重直接匹配。Stage-48的±1像素视图只用于来源审计与分布参考，不进入两臂的优化；因此对比不混入取消像素扰动的效果。

这是一次固定、离线的预测状态混合，不是对新模型迭代滚动的在线训练。控制臂也重新拟合；既有Stage-48 supervised作为单独冻结参考，不冒充本轮匹配控制。

## 状态来源与折隔离

复用Stage-48两折，每折6个fit／6个held-out源组。每折固定使用该折的`supervised_seed2132`生成状态，取既定256步checkpoint。选择在看新结果前固定，不按种子、成绩或事件挑选模型。

关键方向：模型若在A组训练、在B组评价，本轮用它在A组滚动形成训练状态。不能拿“在B组训练、对A组做OOF评价”的互补折模型轨迹填入A组训练池。checkpoint与receipt的fit／held-out名单、fold、seed、arm、步数和SHA都要匹配。

各fit视频从首帧初始化后完整滚动，控制器只接收图像及自身预测，不用后续GT。随后沿用每视频48个均匀时刻，读取其上一帧预测父框。使用原生冻结骨干在这些父框上提取特征，GT在父框固定、前向完成后用于生成训练标签。原生视图复用Stage-48冻结特征，控制臂在同一batch中使用两份。

保留全部均匀样本，不按失败时刻、IoU、GT覆盖或尺寸过滤。GT中心出裁剪时沿用普通损失的零正例热图；GT框与裁剪完全不相交时不做框回归。该处理不意味着头能找回已经离开裁剪的目标。两臂样本视图总数相等，实际正例与框回归次数随状态变化，单独记录。

`prepare`还绑定完整Stage-50结果及旧来源。来源审计可读取既有开发结果，优化器只消费当前外折fit组样本。原checking标签不进入本轮流程。

## 预算与暴露检查

- 2个既有监督头，各在自己的6个fit视频上生成状态：12条新增完整轨迹，共13,525次原生头调用。
- 12×48＝576次新增原生特征前向；复用576个已有原生特征视图。既有平移视图仅只读审计和统计，不进入优化。
- 2折×2种子2132／2133×2臂＝8个新正式头；每个256步、每步4视图，即1,024次视图监督。
- 12视频×2种子×2臂＝48条新增完整OOF轨迹，共54,100次头调用。
- smoke另在fold0、seed2132对两臂各训练8步，并做原生重放及两条学习头的32帧推理；smoke模型不进入正式评价。

新状态生成、特征采集和评价的每次原生推理仍执行L1–L6及一个独立候选块。原生参考12条及旧supervised参考24条轨迹只读复用。缓存成功提交后不重复运行；训练保存头权重及AdamW状态，支持断点恢复。

每折至少16个均匀样本、覆盖至少2个fit视频，其新状态的实际取整裁剪须同时不同于原生与旧平移参考，才进入训练。这个门槛只检查实际输入发生变化，不按GT表现挑选样本；训练仍使用所有样本。暴露不足时保留缓存及报告，不通过调整阈值伪造处理。

状态表还记录GT覆盖、出裁剪中心、搜索图中目标像素尺寸、父框中心与宽高误差。若固定fit内模型几乎不产生困难状态，本轮结论应限定为这批状态的结果。

## 服务器命令

先按既有Git流程同步新增文件，再执行：

```bash
cd ~/workspace/paper
conda activate sglatrack-test
export STAGE50_ROOT="$PWD/output/stage50/shared_parent_frozen_heads_v1"
export STAGE51_ROOT="$PWD/output/stage51/fit_only_state_source_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage51_state_source.py
python -B tracking/stage51_state_experiment.py \
  --mode prepare --input "$STAGE50_ROOT" --output_dir "$STAGE51_ROOT"
python -B tracking/stage51_state_experiment.py \
  --mode collect --input "$STAGE51_ROOT" --gpu 0
cat "$STAGE51_ROOT/collection/stage51_state_support.csv"
python -B tracking/stage51_state_experiment.py \
  --mode smoke --input "$STAGE51_ROOT" --gpu 0
```

出现`PASS_STATE_SOURCE_SMOKE`后执行正式训练及评价：

```bash
python -B tracking/stage51_state_experiment.py \
  --mode train --input "$STAGE51_ROOT" --seed 2132 --gpu 0
python -B tracking/stage51_state_experiment.py \
  --mode train --input "$STAGE51_ROOT" --seed 2133 --gpu 0
python -B tracking/stage51_state_experiment.py \
  --mode evaluate --input "$STAGE51_ROOT" --gpu 0
python -B tracking/stage51_state_summary.py --input "$STAGE51_ROOT"

cat "$STAGE51_ROOT/analysis/stage51_policy_summary.csv"
cat "$STAGE51_ROOT/analysis/stage51_paired.csv"
```

相同root和命令可用于恢复中断。不改旧阶段源码、权重、划分和已提交结果；新阶段的配置及代码也在prepare后冻结。

## 评价与输出

主要比较`state_mix_minus_native_replay`全视频宏AUC，先在视频内平均种子，再对视频做5,000次条件bootstrap。另列原生及旧监督头参考、低IoU比例、原生可靠帧AUC、尺寸误差、最长失效比例、恢复／未恢复／复发事件。事件时间线保留视频末尾截断，不把截断事件称为短暂已恢复。

`source_supervised_failure_auc`使用旧监督头两个种子中任一个IoU<0.2的帧并集，同一视频所有方法共用该离线掩码。它和可靠帧队列都只用于描述，不用于在线触发。各方法自身事件集合不同，恢复次数的差别不是共同起点的因果收益。

pilot的积极信号预先固定为：主要均值至少+0.5 AUC且名义CI下界>0、两种子均为正、整体均值不低于原生，同时原生可靠帧均值不低于−0.5、低IoU均值不恶化、原稳定视频无新增失效。满足时为`PROMISING_FIT_STATE_SOURCE_PILOT`，否则为`REVIEW_FIT_STATE_SOURCE_PILOT`。保护条件是开发筛查，−0.5是实用容限，不是零损害证明。

| 文件 | 用途 |
|---|---|
| `collection/stage51_state_support.csv` | 两折的实际输入变化、覆盖及目标分辨率小表 |
| `collection/stage51_state_exposure.csv` | 全均匀时刻三类状态的几何分布 |
| `analysis/stage51_policy_summary.csv` | 4种方法的宏指标与复用／新训练来源 |
| `analysis/stage51_paired.csv` | 配对AUC、低IoU、保持与失败队列读数 |
| `analysis/stage51_seed_video_summary.csv`、`stage51_video_summary.csv` | 逐种子／逐视频结果 |
| `analysis/stage51_events.csv` | 自身轨迹失效、恢复、复发与截断事件；无事件时不生成 |
| `analysis/stage51_video_influence.csv` | 主比较的逐视频贡献与留一结果 |
| `analysis/stage51_decision.json` | 状态、来源与结果哈希、预算和解释边界 |

先提供support、policy和paired三张小表；其余长结果按需使用文件形式。当前12个源组已被用于设计，OOF仍为开发证据。本轮可检验这批固定状态的新增价值，不能据此直接宣称在线状态学习、外部泛化、压缩或部署加速。

## 本机验证

4项CPU测试覆盖：完整Stage-50来源与篡改拒绝；fit生成方向与互补折泄漏防护；普通损失只随对应视图变化、出裁剪状态仍计入监督；真实小型Torch头、原解码与模块hook组成的采集—两臂训练—OOF—汇总流程。验证了带优化器状态的中断续训与不中断权重一致、训练样本不读取held-out缓存、完整缓存复用、48条评价预算、原网络保持冻结，以及报告／checkpoint篡改拒绝。合成数据与CPU小模型验证不代替服务器精度实验。
