# Stage-216：共同当前状态的几何与真实后继定位质量对照

2026-11-28。依据215：coverage关闭尺度后AUC+17.8019，但只中心仍低于none且新增稳定失跟。因此不扫尺度上限、不直接采用中心更新，检验损伤是否在共同状态的一步决策中已经出现。

## 固定设计

原215包SHA256 `275f66a26c14458045ca5af41a1831ac2537242fb77223df19a0cf6566067f87`，需完整215同型号重放物理缓存、214冻结更新器、208适配器、原始checkpoint和七旧来源目录。38开发源、两折、2132/2133、四mask、SGLA/CENTER/选层/裁剪倍数均固定；0训练、0新完整轨迹、确认组封存。来源代码/协议/数据/模型/receipt/NPZ哈希检查不放宽。

四种当前状态来源：215 coverage__none（nn规则）、mm、coverage__center、coverage__joint。nn/mm称“提供状态风格”，不是214模型实际训练样本，不能把OOF轨迹的误差称为训练误差。每源沿用既有最多48个均匀时刻，但从1到tracked_frames−1取样，以保证t+1可用；相同序列各seed/来源/动作使用同一时刻。当前均为48×38=1824个时刻，乘2seed×4来源=14592个状态。

先按来源parent重建当前帧t的首帧模板、frame_id、raw/适配读出与F6特征，输出/层/调用须精确复现缓存。每个冻结forecast/coverage更新器只在当前共同特征上推理一次；将none/center/scale/joint映射为四个下一父框，固定物理边界。每个动作独立重置模板和状态，在同一真实t+1图像执行一次同裁剪raw+adapted读出；不让一个分支影响另一个分支。两个家族的none须一致。

对nn、center、joint来源，还核对应动作的下一父框及t+1适配框/层精确复现来源；mm只核当前，因为本实验没有“直接以适配框更新”的额外动作。真值只在动作和全部输出确定后计算标签，不传给更新器或用于选时刻。缺失或漂移即拒绝，不以近似IoU替代。

每状态1次当前+8次后继搜索，共131328次新搜索；每搜索8block/2head/1selector/1crop；每状态2updater。首折固定首源各抽第一个时刻作串行校验，完整双spawn结果与其全数组一致，共16条来源/seed短检查、144次额外搜索。总体进度131473含最终封存一步，默认双worker。模板初始化另外执行，不包含在搜索block账本，不声明推理加速或独立FPS。

## 指标和主比较

下一父框几何误差为：中心坐标L1误差均值除以GT几何平均侧长，加绝对log侧长误差。它仅是诊断组合量，不称原214 SmoothL1训练损失；另独立报告中心L2误差、log侧长误差。

同时计算完整覆盖、GT名义网络侧长、dense最佳IoU及标准输出IoU；每动作相对同状态/同家族none的几何收益与输出收益配对。报告几何改善比例、几何改善且输出变差占全部状态的比例，以及存在几何改善状态的源组数及其条件输出均值。条件子集是事后标签分层，不解释为独立因果效应。

**唯一主要比较**：在nn/mm两种提供状态风格上，coverage只尺度−none的下一帧输出IoU，先每来源/源组内平均seed及时刻，再等权平均两来源，最后38源bootstrap20000次。其他来源/家族/动作与几何误差均为支持比较。它检查损伤是否在长闭环之前已出现，不预设一定下降。

不把不同状态来源差异直接称为分布漂移的因果贡献；不把一步IoU当完整AUC，也不报告恢复门槛已通过。全部来源是已消费开发数据，没有训练或确认集打开。若几何改善和定位改善稳定不一致，可为任务质量监督小原型提供标签；若自身状态上更差，应进一步区分预测质量和状态分布依赖。两种解释可能并存。

10表：selection1824、model_scope12（8更新器/4适配器）、source_replay14592、cells116736、policy_summary32、paired49（主比较1）、alignment24、calls14592、smoke16、execution2。mm的successor_checked=False、successor_exact留空，表示没有可比对角动作，不是复现失败。保持REVIEW，无自动采用。

## 服务器运行

```bash
mkdir -p output/stage216
LOG="output/stage216/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage216_successor_quality.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage216/run.pid
tail -f "$LOG"
```

默认输出 `output/stage216/common_state_successor_v1`，变量STAGE216_ROOT/GPU/WORKERS。继承STAGE214_ROOT、STAGE208_ROOT及STAGE215_REPLAY_ROOT，215默认same_model_gpu_replay_v1，不继承早期STAGE215_ROOT旧目录；允许同型号UUID独立新绑定，后续续跑与每个worker仍须精确当前绑定，不改来源。若再次换卡，设新STAGE216_ROOT完整重做本诊断，不混用既有缓存。后台结束自动合并manifest/decision及10CSV为TXT，不提交/推送Git。

```bash
python -B tools/stage216_successor_quality.py --check-results \
  --output-dir output/stage216/common_state_successor_v1
```

本机CPU合成检查不是实际GPU/Transformer或像素裁剪验证，不产生真实实验指标。
