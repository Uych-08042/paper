# Stage-55：共同父框、冻结头与32帧后继诊断

Stage-54六表已齐：末帧保持对匹配普通监督+0.7265 AUC、4胜8负、区间跨零；对旧静态保持12胜0负。下一步聚焦0204高分适配分支丢失、0304种子分岔及0307恢复改善，区分当帧几何差异和后续历史放大。本轮只使用既有模型进行诊断，新增训练为0。

用户已报告`COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`，focus／crossed_matrix／branch_contrasts三表已收到，详见[结果复核](STAGE55_RESULTS_REVIEW.md)。0204／0304的大幅当帧差主要随进入窗口的状态变化；0307恢复附近存在共同起点上放大的模型收益，也有起点依赖的损伤。3项真实CPU测试通过（90.476秒，无跳过），本机未独立重验服务器完整缓存。以下保留冻结协议及历史复现命令，不重跑已完成采集。Stage-1～55实验Python与结果保持冻结。核心任务仍为以原生SGLA开展无人机单目标跟踪研究。

## 固定锚点

| 案例 | 视频 | 开始帧 | 冻结来源事件 | 目的 |
|---|---|---:|---|---|
| C01 | uav0000204_00000_s | 664 | rollout_preserve_seed2132首次失效 | 定位普通监督无失效高分分支丢失 |
| C02 | uav0000304_00253_s | 554 | rollout_supervised_seed2132首次失效 | 比较早期分岔与保持的种子差异 |
| C03 | uav0000304_00253_s | 1078 | rollout_preserve_seed2132首次失效 | 检查推迟之后仍存在的晚期损伤 |
| C04 | uav0000307_04531_s | 389 | rollout_preserve_seed2132首次失效 | 比较相近起点之后的失效发展 |
| C05 | uav0000307_04531_s | 471 | rollout_preserve_seed2132首次恢复 | 比较恢复点及其后继稳定性 |

prepare从冻结轨迹重算事件并逐项核对锚点，不能静默改为邻近帧。每个锚点仅读取−1、0、+1三个当帧状态；32帧分支固定从0偏移开始，包含开始帧至开始帧+31。全部为已消费结果上选出的机制案例，不是新验证视频。

## 5种历史×5个冻结头

历史与评价头都固定为：原生native，以及两个种子2132／2133各自的rollout_supervised、rollout_preserve。所有适配头均来自该视频所在held-out折，不重新训练或选择checkpoint。初始化始终使用原始SGLA权重。

**当帧共同输入。** 从来源历史读取当前帧前的预测父框；一次完整原生前向生成一个实际选层特征，原生头和另外4个冻结头使用同一个GPU张量。记录raw／Hann格点、响应、dense候选、图像坐标输出框、中心误差、宽高比和覆盖。响应参数及缓冲均未更新，故相同特征上的响应误差必须≤1e-6；各头的几何仍可能不同。

各历史的自身头重放必须复现对应来源框，误差≤0.001像素，官方取整框完全相同。原生实际输出与dense解码也核对。这一步以相同输入比较当前头能力，再比较不同来源父框，防止把历史差异全部算到头上。

**匹配起点32帧分支。** 每个历史父框分别由5个头独立推进32帧，使用原生public tracker、选层、采样、Hann与裁剪。第二帧开始仅使用该分支自己的上一步实际输出；不再逐帧重置为来源父框。原始首帧模板在每条分支开始时重新初始化，后续GT不进入actor。没有输出校正、教师接管、在线重检或GT父框。

每条分支的第一帧与当帧共享特征探针核对，包含响应、size／offset图和输出框；自身头×自身历史的完整32帧还必须复现冻结完整轨迹。单次网络调用必须仍为6个前缀块＋1个候选、1次selector与1次head。共同父框下后续响应／选层可能因自身几何造成的新输入而改变，不能误称为响应权重被修改。

## 预算和缓存

- 当帧输入：5案例×3帧×5历史＝75次完整前向，额外4头共300次仅头前向；得到375个头输出。
- 分支：5案例×5历史×5头＝125条，每条32帧，共4,000次完整前向。第一帧也执行原生路径，用于独立核对共享特征探针，没有省略或双计为零成本。
- 合计：**4,075次完整网络前向、300次仅头前向、28,525个block调用、4,075次selector、4,375次head**。模板初始化只做预处理，不额外执行网络；预处理与图像读取成本仍存在，不做部署时延结论。
- smoke覆盖C01的全部15个当帧输入及native历史上的5条完整分支，即175次完整前向＋60次仅头前向。其缓存直接用于完整collect，总预算包含smoke，不额外重复。
- 缓存按单个输入／单条32帧分支提交并附身份和SHA收据。中断后复用已提交数据；未提交的整条分支重跑，实际重跑计算可能超出已提交预算。不会续接半条分支的未保存状态。
- 每个case结束核对原始网络、4个冻结头及BN状态；最终汇总核对全部输入／分支原生调用和来源绑定。

prepare校验完整Stage-54及上游来源，重算108条种子—视频、60条视频和5条宏汇总指标，并绑定模型、收据、8张表、manifest与run／decision。既有Stage-51～54产物及上游依赖均须保留。

## 模型／父框的记账与后继含义

对每个案例和种子，S代表普通短分支头，P代表末帧保持头；hS、hP是来源中两种历史的预测起点。报告当帧IoU、32帧平均IoU、末帧IoU三种读数：

```text
自身差 = value(P, hP) - value(S, hS)
      = [value(P, hS) - value(S, hS)]
      + [value(P, hP) - value(P, hS)]
```

另外计算另一条分解路径，核对最大残差≤1e-9。当帧分解是固定输入上的精确记账；32帧的“模型替换差”包含该头引起的后续自身状态演化，不能解释为每一帧都固定输入后的纯头差，也不是整个训练过程的唯一因果归因。

共同起点的P／S分支另报告首个浮点框、实际取整裁剪、响应图发生差异的帧。三者分别核对，避免把很小的直接几何变化等同于立即出现显著IoU损伤。

窗口内平均IoU不是完整视频AUC。局部失效／恢复计数在窗口边界重置，窗口开始可能左截断；`window_unrecovered_events`表示32帧末尾尚未恢复，不等于持续到原视频末尾。没有统计方法采用门槛，成功状态只代表完整诊断与一致性检查通过。

## 服务器执行

同步新代码后：

```bash
cd ~/workspace/paper
conda activate sglatrack-test
export STAGE54_ROOT="$PWD/output/stage54/short_rollout_endpoint_v1"
export STAGE55_ROOT="$PWD/output/stage55/matched_parent_continuation_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/stage55_matched_collect.py \
  --mode prepare --input "$STAGE54_ROOT" --output_dir "$STAGE55_ROOT"
cat "$STAGE55_ROOT/preparation/stage55_cases.csv"
python -B tracking/stage55_matched_collect.py \
  --mode smoke --input "$STAGE55_ROOT" --gpu 0
```

出现`PASS_MATCHED_BRANCH_SMOKE`后：

```bash
python -B tracking/stage55_matched_collect.py \
  --mode collect --input "$STAGE55_ROOT" --gpu 0
python -B tracking/stage55_matched_summary.py --input "$STAGE55_ROOT"

cat "$STAGE55_ROOT/analysis/stage55_focus.csv"
cat "$STAGE55_ROOT/analysis/stage55_crossed_matrix.csv"
cat "$STAGE55_ROOT/analysis/stage55_branch_contrasts.csv"
```

成功状态为`COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC`。以上三张表已收到，分别10、25、50行，不重复索取。完整轨迹和逐帧几何保存在本轮目录，初步判读无需粘贴4,000行timeline。

| 结果文件 | 用途 |
|---|---|
| `stage55_focus.csv` | 两种子在自身／共同起点上的当帧与32帧分解，10行 |
| `stage55_crossed_matrix.csv` | 每个开始帧5历史×5头IoU矩阵，25行 |
| `stage55_branch_contrasts.csv` | 共同起点P／S均值、末帧及首次分岔，50行 |
| `stage55_point_details.csv` | 三帧全部头的覆盖、选点、候选和几何，375行 |
| `stage55_rollout_summary.csv` | 125条分支各自局部指标与边界状态 |
| `stage55_timelines.csv` | 4,000个分支帧的中心、宽高、覆盖和IoU |
| `stage55_diagnostic.json` | 完整预算、重放／冻结／分解检查及来源／结果SHA |

## 本机验证范围

3项真实CPU测试通过（90.476秒，无跳过）。测试使用仓库实际CENTER小通道实例及原生采样／解码／计数hook，覆盖固定事件及预算、缓存身份和篡改拒绝、共享响应一致、真实预测作为后继父框、后续GT不改变actor输出、错误父框拒绝。完整测试从合成Stage-51～54来源开始生成真实训练／OOF产物，再执行本轮prepare／smoke／collect／summary，验证原生自身对角线、smoke及中断后缓存复用、130次完整／120次仅头的小样本精确预算、分解残差和来源表／checkpoint／新缓存／结果表篡改拒绝。CPU实现检查不替代服务器机制结果。
