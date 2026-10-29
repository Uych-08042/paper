# Stage-48：原生 SGLA 预测头的排序与尺寸稳定性

## 问题与依据

主任务继续是 VisDrone-SOT、首帧框初始化、原生 SGLATrack。Stage-47 已由用户在服务器完成400次固定历史的独立重放，提供两条可分开检验的线索：

- `0304:555` 的同一 `camera_only` 历史上，不偏移 IoU=0.7703；约0.22像素的速度提议使实际裁剪改变，IoU降至0，GT仍在候选中但由第1降至第2。单改浮点映射的离线探针没有造成这次损伤。
- `0304:1087` 的 `camera_velocity` 历史上，中心误差仅约0.065个GT边长单位，宽度却达到GT的4.22倍；把GT尺寸放到预测中心的离线上界IoU约0.8524。这里需要检验尺寸校准，不能全部归因于中心选错。

`0176:158` 所有重放均无IoU≥0.5的dense候选，说明排序约束不可能覆盖所有状态；`0204:664` 的单帧相机修正也不复现完整相机策略的收益，说明历史不可忽略。以上都是已消费案例的机制证据，不是待复现的全视频方法成绩。

本轮检验：**固定原生特征与结构，仅训练预测头，能否通过较大的正确／错误位置分数间隔和相对尺寸监督，在自身闭环中改善跟踪，同时保护原本可靠的视频？** 新增价值由匹配对照与完整轨迹判断。训练期排序、增强和尺寸监督有既有基础，当前不预先宣称新方法成立。

## 冻结设计

| 项目 | 固定设置 |
|---|---|
| 数据 | Stage-46原来的12个已消费开发源组，不增加检测数据 |
| 划分 | 按源组固定2折，每折6 fit／6 held-out；每视频只由未见其标签的模型评估 |
| 种子 | 2132、2133；划分种子2131 |
| 训练对象 | 原始CENTER预测头参数；骨干、选择器、独立候选块不训练，头内BN运行缓冲也冻结 |
| 视图 | 原生预测父框；同一父框沿预定方向平移±1原图像素，宽高不变 |
| 采样 | 每视频均匀48个时刻；方向由视频名与时刻哈希决定，不按失败选帧 |
| 更新 | 256步、batch=2帧对，即每步4个特征视图；AdamW，LR=1e-5，WD=1e-4，clip=1 |
| 保存 | 第128与256步提交检查点，仅固定第256步用于评估 |
| 规模 | 4训练臂×2折×2种子＝16个正式模型；12视频×4臂×2种子＝96条新完整轨迹 |

所有臂采用相同初始化、样本顺序、两视图、标签、优化器和更新次数。共同首个样本含fit集合中可用的排序／尺寸支持，用于检查梯度；不会从held-out结果挑选。初始权重一律是原SGLA，不能传入Stage-35／40学生权重。

| 训练臂 | 损失 |
|---|---|
| `supervised` | 普通两视图监督：2×GIoU＋5×L1＋focal |
| `rank` | 普通监督＋0.1×排序项 |
| `shape` | 普通监督＋0.25×尺寸项 |
| `rank_shape` | 普通监督＋两个新增项 |

排序项：以**冻结原生预测**的dense框与GT的IoU定义正位置（≥0.5）和负位置（<0.1）。在当前头的Hann加权响应上，使用 `softplus(0.2 + max_negative_log_score - max_positive_log_score)`。只对分数分支求梯度；无正负支持的视图跳过该项。固定分配不能保证训练后的几何仍匹配GT，因此另报实际候选保留率。每折至少16个支持帧对、至少2个支持视频才训练。

尺寸项：在按**真实取整裁剪**计算的GT中心格点，约束预测物理宽高与GT宽高的对数差，SmoothL1 beta=0.1；同一帧两视图均受支持时，再加0.25倍物理尺寸一致性项。它不假设相邻帧尺度不变，不在推理时读取GT或钳制尺寸。

特征缓存直接通过原始预测头的pre-hook取得输入，避免另写一套骨干。两视图可能产生不同原生选层，缓存保留其真实输入；同一缓存供四臂共享。±1像素是浮点父框提议，真实裁剪采用原来的取整规则，不承诺有效像素差始终恰好为1。

这与Stage-40的普通裁剪增强、Stage-44的L5–L6前缀适配不同：本轮保留完整原生结构，新增项对照是**同样两视图的普通预测头监督**。不重跑六层压缩或运动恢复模块。

## 执行与证据隔离

先校验Stage-47及全部旧来源，再采集12个视频的特征、冻结标签和receipt。缓存采集可读取所有这12个开发视频的GT；**每个折的优化仅加载其6个fit视频的数组**，另一折的标签不进入该模型的训练。旧Stage-31等原checking标签不加载。这12个视频都已参与设计，OOF仍是开发证据。

smoke在第1折训练4个头各8步，并执行原始模型与4个短训头各32帧的实际原生跟踪；检查原生重放、梯度作用位置、参数变化、BN缓冲不变、来源与调用预算。它只证明执行，不建立精度收益。

两种子所有16个最终模型冻结后才允许完整评估。每条新轨迹从首帧框开始，后续裁剪只用自身预测。每图像仍调用一次原生头、L1–L6加一个原选择器选择的独立候选块；保留并审计真实块调用。没有搜索分支、相机补偿或额外推理头。特征采集、训练与离线探针的调用另属实验开销，不据此声称加速。

另在held-out视频的共同冻结父框上，执行缓存特征的两视图预测头探针；原生探针直接复用采集的原始图。输出未扰动与扰动后的**绝对IoU**、两者差、当前候选保留率、保留时仍选出低IoU框的比例、GT中心格点尺寸误差。它们是单状态诊断，不能当作完整轨迹AUC。

## 服务器执行

同步本次代码后，在仓库根目录执行。保留所有旧manifest、receipt和结果路径，不修改冻结的Stage-1～47源码。

```bash
cd ~/workspace/paper
export STAGE47_ROOT="$PWD/output/stage47/matched_search_response_v1"
export STAGE48_ROOT="$PWD/output/stage48/native_head_stability_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage48_stability.py
python -B tracking/stage48_stability_experiment.py \
  --mode prepare --input "$STAGE47_ROOT" --output_dir "$STAGE48_ROOT"
python -B tracking/stage48_stability_experiment.py \
  --mode collect --input "$STAGE48_ROOT" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0
python -B tracking/stage48_stability_experiment.py \
  --mode smoke --input "$STAGE48_ROOT" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0

cat "$STAGE48_ROOT/features/stage48_features.json"
cat "$STAGE48_ROOT/smoke/stage48_smoke.json"
```

确认输出为 `PASS_HEAD_STABILITY_SMOKE` 后继续。特征报告的 `support` 同时报每视频支持量和冻结原生排序错误视图数量，支持量不等于可纠正失败数量。

```bash
for seed in 2132 2133; do
  python -B tracking/stage48_stability_experiment.py \
    --mode train --input "$STAGE48_ROOT" --seed "$seed" \
    --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0 || break
done

python -B tracking/stage48_stability_experiment.py \
  --mode evaluate --input "$STAGE48_ROOT" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0
python -B tracking/stage48_stability_summary.py --input "$STAGE48_ROOT"

cat "$STAGE48_ROOT/analysis/stage48_decision.json"
cat "$STAGE48_ROOT/analysis/stage48_policy_summary.csv"
cat "$STAGE48_ROOT/analysis/stage48_paired.csv"
cat "$STAGE48_ROOT/analysis/stage48_probe_policy_summary.csv"
```

同一命令可续跑；训练恢复已提交的头／Adam状态，完整推理按视频与方法复用已提交缓存，未提交轨迹从头重跑。文件缺失或已提交内容变化时拒绝混用，不能通过改fingerprint或删receipt绕开。正式评估要求两种子的16个模型齐全，训练循环中断后不能跳过缺失模型。

## 结果解释

主比较是 `rank_shape - supervised` 的完整视频AUC；同时比较原生、单排序、单尺寸及两因素交互。种子先在视频内平均，再对12视频bootstrap 5000次，种子2134，报告均值、中位数、95%区间、胜负及留一视频均值。区间条件于固定重叠OOF拟合，不涵盖重新训练的不确定性。

开发继续条件：组合臂较普通监督平均至少+0.5 AUC点、较原生平均不低；低IoU宏均值不劣于原生、任一种子不在原稳定视频引入失败、原可靠片段AUC均值至少保持在原生−0.5点以上。满足则 `PROMISING_NATIVE_HEAD_STABILITY_PILOT`，否则 `REVIEW_NATIVE_HEAD_STABILITY_PILOT`。这是继续研究的实用筛选；跨零区间和个例改善应如实报告，不自动推导外部验证或放弃整个方向。

重点判断：新增项是否超出普通两视图监督、排序改善是否来自实际选择而不是候选消失、尺寸改善是否伴随闭环保留，以及扰动损失缩小是否确有绝对扰动表现提升。不能仅凭某项平均变好便宣称全部机制成立。

## 本机验证状态

新增5个Python文件；6项CPU测试使用实际既有跟踪损失、小型可训练头和合成图像，覆盖源文件变更拒绝、分组与fit数组隔离、梯度分流／空支持、特征回放、排除原生观察器hook的头复制、断点恢复、16模型／96轨迹夹具、重复汇总与checkpoint篡改拒绝。维护清单与文档链接检查通过。它不替代服务器原始SGLA GPU smoke与正式实验。本机未持有完整Stage-16～47服务器产物。
