# Stage-222 固定留出状态定位证据审计

日期：2026-11-29。当前实现待真实服务器缓存运行；不预设主要病因或方法收益。

目的：为“提高当前空间目标证据的可靠性，并研究它如何控制状态更新”确定一个最小模块改造入口。遵守220停止规则，不再增加该父框修正器的容量或历史长度。本轮仅描述证据结构，不能识别失跟最早原因，也不声称因果贡献、在线救回或新AUC。

## 固定样本与来源

38个历史开发源，两fold、2132/2133；只读Stage-216预先固定的每源48个held时刻，不依据本轮GT重选点。主分析coverage__none/mm共7296状态，coverage__center/joint另7296状态只作支持。两种seed下的原生历史可能相同，不作独立视频样本。确认集封存，0训练/新图像/GPU/完整轨迹。

Stage-221物理decision必须精确等已审a6980275…，manifest、九表及source_arrays哈希通过；不依赖当前缺失的221桌面TXT。215/216物理decision、manifest须等仓库已审结果包，核相同entries/folds/search_factor与来源指纹。GT及帧清单从七个原来源加载并核原有hash，仅用于离线诊断；不读取图像。215全轨迹NPZ及receipt逐个绑定，216当前raw/适配输出逐元素复现；221当前GT有效性及E三分量逐点重算。fit提供轨迹缺dense证据，本轮不纳入、不填充。

搜索尺寸从已绑定runtime TEST取256，倍数绑定原配置。覆盖重新调用原sample_target整数窗口支持函数，排除padding并保留旧右/下边界行为，须精确复现缓存；名义GT网络侧长也须复现。dense_best_iou只来自同一adapted分支的已存网格最大IoU。**不能据此声称原生raw网格有好框或原生CENTER选错。** 无法独立重建dense图，因为215只存dense最大值；必须保留这一核验边界。

## 四类互斥主描述

固定raw当前IoU<0.2为严重偏离，按下列优先顺序分类，好框为IoU>=0.5：

1. adapted_good：适配标准输出已好，不论是否完整覆盖，提示存在可用当前输出。
2. not_fully_covered：适配输出差且目标未完全位于实际可见裁剪内。部分可见不等于毫无目标信息。
3. covered_no_good_grid：完整覆盖、适配输出差、适配dense-best<0.5。
4. covered_readout_miss：完整覆盖、适配输出差、适配dense-best>=0.5。

raw>=0.2记raw_not_severe；当前GT无效保留原行、连续指标留空，不静默删点。主四类计数之和必须等有效raw严重偏离数。0.2/0.5沿用历史定义，不扫描阈值。连续raw/适配/dense IoU、差值、中心误差、目标名义网络侧长同时报告；all/raw_severe/raw_good三个固定层次。

每源内合并两seed与主两来源的条件计数，先算该源raw严重偏离中的四类比例，再跨有支持源等权；分母0源列入unsupported，不能填0。四个primary行是一项预定组成描述，非四项优效性假设；给出固定源bootstrap名义边际区间，不作联合区间或显著性采用门槛。源内重复记录不是独立样本。各单来源同样描述；不按支持分析改主结果。

## 输出、终点与执行

七表：selection1824/cells14592/group_metrics304/policy_summary15/primary4/source_replay304/execution1。另封存manifest、decision与source_arrays。总体305检查进度；CPU NumPy计算，不加载模型，不要求GPU型号或UUID。只可续用完整且哈希一致结果，部分失败可在同目录按相同输入重算；保留writer_lock，不自动清锁。

终点：报告各类跨源支持及规模，为一个模块原型提供依据。已有好输出优先检验反馈利用；覆盖缺口优先输入支持；覆盖却无好网格优先空间表征/回归；漏读优先目标身份与定位质量。混合结果如实报告，不用“最多的一类”自动证明因果或新增阈值；不无限拆分诊断。后续改造须固定状态定位对照，然后单独验证反馈与完整轨迹能力保护。

同步代码后，在服务器已激活环境、仓库根目录运行：

```bash
bash tools/run_stage222_localization_evidence.sh
```

继承STAGE221_ROOT、STAGE215_REPLAY_ROOT、STAGE216_ROOT及七来源变量。默认输出output/stage222/cached_localization_evidence_v1。后台命令：

```bash
mkdir -p output/stage222
LOG="output/stage222/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage222_localization_evidence.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage222/run.pid
echo "$LOG"
tail -f "$LOG"
```

正常结束自动打包七表及decision/manifest为TXT，位置在实验输出目录。Ctrl+C退出tail不会停止nohup。真实缓存不在本机，本地验证只检查合成几何、分层/加权、封存缓存工作流及篡改拒绝；不是服务器诊断结果。
