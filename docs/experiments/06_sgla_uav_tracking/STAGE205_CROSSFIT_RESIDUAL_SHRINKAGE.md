# Stage-205：训练源组内部交叉拟合的残差可信度

固定日期：2026-11-26。状态目标：`REVIEW_CROSSFIT_RESIDUAL_SHRINKAGE`。这是开发缓存上的校准试验，不是完整视频采用门槛。

## 问题与依据

Stage-204正确关系的held平均IoU为0.661275，对自身无对应期望有+0.001593的小幅信号，但仍低于固定fit先验0.666608；对原输入、错配关系的总体增量区间均跨零。分数残差在跨源组时可能不够可靠。检验在**训练源组内部的样本外预测**上估计残差应受信任的幅度，能否保留状态对应而减少不可靠动作。它不是已被证明的失败病因，也不保证平方误差改善会提高argmax选择的IoU。

来源固定为Stage-194原生父框L6特征／六层IoU标签和已复核Stage-204原样TXT：SHA256 `3d1a89cad966001838cc8145917bd12d08b11030737081cf5e159a4b54a48811`，decision `3cc164919abc8af737a21520e9e53b9775c5bf055ddfb6d0eda1d43a0ee36935`，plan `603de9cf615c4d318c1bc54de29de745c76827a7319d5593b4d5f1c79828ff5c`。1～204代码、协议和产物不修改。

## 严格嵌套与匹配条件

沿用38历史开发源组、19 fit／19 held的两外折、2132／2133。每外折的19 fit源组逐一留出，仅其余18组进入一次内层拟合。内层provider只含这19个外fit源组，完全排除外held；被留出的内层组不参与训练、先验、token均值、描述均值／标准差或错配参考池。全部统计每内折重算；错配参考为18组排序循环下一组或对内留出组按SHA256取18组中的一组，遵循204规则。

三输入`concat`、`matched_relation`、`mismatched_relation`均执行同规则。复用冻结204模型／损失／优化器／缓存和检查点引擎：实际布局320×192、6720可训练参数、1088输入，统一可训练参数初始化，内折统计buffer因输入不同而不同。point层间MSE、clip1、AdamW lr1e-5／wd1e-4，256更新、每步2对同视频独立有放回状态、保留自配对，每模型1024状态引用／512对。相同内折seed三输入抽样精确一致，抽样种子沿用`seed+100*outer_fold`；内留出组变化导致可抽样的18组变化，不要求不同内折抽样相同。

总共228个额外CPU模型＝2外折×19内留一×2seed×3输入，58368更新、233472训练状态引用、116736模型配对引用。三输入共享日程38912行，1368条源组支持计数。内层每模型只预测其内留出组的48状态，共10944次模型状态预测；不做额外outer模型拟合。

训练引擎仍使用204的receipt schema，所有内层metadata绑定新的205计划指纹、18组fit／1组内留出、抽样哈希、初始化哈希；路径位于205独立根下，绝不写204。普通中断复用已提交小模型和缓存，未提交的一个小模型从头重算，不声称逐步Adam续训；非有限训练诊断会阻止继续，须先复核原因。

## 闭式风险估计与锁定

P表示六层去均值。每个内层留出组保存预测s、内层模型实际float32先验μ和六层标签y，令`h=P(s−μ)`、`z=P(y−μ)`。在该外fit的19个源组等权、各组内状态等权下，最小化`E||αh−z||²`：

`α = clip_[0,1]( E⟨h,z⟩ / E||h||² )`；分母严格为0时α=0。

这是有界标量二次风险的解析解，不搜系数网格，不拟合视频身份均值，不使用外held标签。每个外fold／seed／输入单独一个α，共12个系数；全部带OOF分数哈希的sealed文件发布后，才执行外held策略评价。内层OOF用于估计α，其收缩后校准误差不是新独立性能评价。训练源组间互有开发暴露，不能承诺分布无关泛化。

外层直接复用204已保存float32预测s与fit先验μ，计算`μ+αP(s−μ)`。统计变换用float64，α=1显式返回原始float32分数的精确值，避免近并列动作变化；α=0精确返回204的float64先验控制。中间α为新的缓存统计策略，未来若进入运行时需另验精度。动作沿用原诊断规则：最大分数，L9与最大值差≤1e-8时优先L9，否则首个最大；不根据标签改动作。raw与prior两个控制须逐源组精确复现204（指标容差1e-12）。

与Stage-198固定层均值校准和201视频去均值不同，本轮从内层样本外误差估计残差幅度，不用外held视频未来统计。228内层训练是**新增校准成本**，单列，不宣称与raw完全相同总训练预算。

## 评价、输出与边界

只评价外held，原生父框即时IoU，不是完整跟踪AUC；源组等权，先平均两seed后对38源组做20000次bootstrap，沿用seed24194及名义95%区间。五项主要比较预先固定：正确关系shrunk对raw、对prior、对自身无对应期望、对concat shrunk、对错配shrunk。支持报告两seed、状态对应、层计数、非先验动作比例、层间误差和加权排序；不能只挑有利seed／表格。

三输入均保留shrunk／raw／prior_only／native_L9／oracle；oracle仅标签诊断上界。独立期望为该视频实际动作频率与六层平均标签的内积，状态对应增量为实际IoU减该期望。若全部退回固定L7，即使达到0.666608也仅为先验保护，不能称动态选层有效；已知194固定L7完整轨迹受损，不能因即时控制占优而采用。

16表：source_bridge38、inner_scope228、inner_statistics114、inner_schedule38912、inner_support1368、descriptor_audit2166、training_history58368、inner_predictions10944、calibration12、frame_predictions10944、policy_video1140、policy_summary15、error_video684、error_summary9、paired22、seed_summary30。另有内层统计114 NPZ、模型228 PT、预测228 NPZ及各receipt，12 sealed系数JSON；共1152个物理内部文件绑定在decision中。原始194来源引用另记账。本轮0新图像读取、骨干／CENTER推理、外模型重训、完整轨迹、确认源组；平均层号不代表实际Transformer执行量或速度。

单一总体进度条覆盖来源核验、缓存pool、内层训练、内留出预测、系数锁定及外held评价；终端原位显示，后台日志约10秒一条总体快照。决策固定为REVIEW，不自动部署或开启确认集。若仍无保留动态动作的增量，降低单独校准假设优先级，重新评估训练支持／骨干学习，不继续扫系数。

## 服务器运行

在`sglatrack-test`环境、`~/workspace/paper`同步代码后：

```bash
mkdir -p output/stage205
nohup bash tools/run_stage205_crossfit_shrinkage.sh > output/stage205/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage205/run.log
```

CPU-only，可在分配L20或L40时运行；本轮不绑定GPU型号。默认输入`output/stage194/module_ablation_v1`和仓库204原样TXT，输出`output/stage205/fit_source_crossfit_shrinkage_v1`。可覆盖STAGE194_ROOT／STAGE204_BUNDLE／STAGE205_ROOT。不需要204服务器PT或预测NPZ，外层分数从已核验TXT恢复；194原特征／标签NPZ及receipt必须保留。

完成后脚本自动校验并打包`stage205_results_<UTC>_<PID>.txt`（decision／manifest／16表）。回传该文件。仅复核／重新打包使用：

```bash
bash tools/run_stage205_crossfit_shrinkage.sh --check-results
```

本机没有194真实特征。本机合成测试验证真实CPU训练和隔离／收据／复算流程，不预报服务器精度或α。
