# Stage-207：同预算原生／自身父框状态训练

固定日期：2026-11-26。目标状态`REVIEW_PAIRED_PROVIDER_STATE_TRAINING`，38历史开发源组，两fold各19 fit／19 held、seed2132／2133，确认集封存。

## 问题和假设

206正确空间模块在原生父框held状态IoU +0.008320，但完整轨迹AUC −2.2103、稳定与恢复保护恶化。本轮固定模块、原生路由、CENTER、损失和学生训练预算，只检验训练父框来源是否影响自身轨迹效果。不再调α／注意力宽度／残差倍率，也不预设分布错配是唯一病因。与51几何头state_mix、54末帧保持不同，本轮直接干预206空间模块的训练状态来源。

来源为已审206包SHA `c2a3aae104ad211ad6b11358f3c8d43fd39f58fce2419890e2bf1a0c192478a5`、decision `0efb6e79719b9604b6438178738f038fe97f54e2037293fd89bb96dcda285385`，原206目录中的四个正确模板provider PT／receipt、194原生父框特征、既有七阶段数据入口。1～206代码、原协议和产物不改。CPU可解析原包，本轮真实启动对旧代码仍逐字节严格检查，不把本地CRLF审计规则用于放宽服务器实验。

## 冻结provider与配对采集

每个外fold／seed只加载该fold的206 matched_spatial模型，严格核对fit／held、seed、plan、scope和模型状态SHA。固定provider仅在自己的19个fit源组跑完整自身轨迹，从GT首帧框初始化，此后只反馈自身预测。既不使用206 held轨迹训练，也不借另一fold OOF路径；另一fold的provider可能使用本外held标签，会产生间接泄漏。

按194预先固定的每视频最多48个uniform帧，捕获provider执行时、适配器修改前的L6输出和真实父框、resize_factor、图像尺寸。原始捕获hook先注册，206适配hook随后注册，不能把已适配token作为学生输入再适配一次。每帧核对一个裁剪、一个前缀、一个原生MLP及实际所选候选；完整父框链和7块／1头计数沿旧审计。

采集actor只接收首帧GT框，不接收后续GT；完整轨迹生成后才附加样本帧真值与定位target。定位target沿194的原`executed_crop_roi`浮点回映，不在本轮改变监督坐标约定。

## 公共支持：采用实际取整裁剪

支持与训练target分开计算：由实际resize_factor反推整数crop_sz，使用原sample_target的Python round获得左上角，并严格按原实现剔除constant padding，包括其现有右／下边界`image_size−1`切片行为。要求GT框正面积且完整位于未padding的实际输入范围内。没有新增IoU阈值／边距／样本质量分数，也不宣称几何覆盖保证无遮挡可见性。

同一帧的原生父框与provider自身父框均满足条件时，才进入公共配对支持。导出逐帧native_covered／own_covered／paired以及源组支持数、父框／token改变数。所有76个fit上下文采完并预检后，才开始任何学生训练。

任一fit源组公共支持为0：写`stage207_support_decision.json`，状态`INSUFFICIENT_PAIRED_SUPPORT`，保留该源组并停止学生训练，不删源、不从held补样、不后验改变覆盖门槛。脚本仍打包预检结果，回传即可；这不是完整方法评价。若均非空，状态为`SUPPORT_READY`，以原子封存的支持及缓存继续。

## 三个学生条件和训练预算

| 学生 | 训练输入 | 用途 |
| --- | --- | --- |
| native_all | 194原生父框全部原样支持 | 206正确空间模型复现控制 |
| native_pair | 公共支持同帧原生父框 | 隔离支持筛选效应 |
| own_pair | 公共支持同帧自身父框 | 状态来源干预 |

三者都使用206 matched_spatial结构、147456参数、相同seed零输出初始化，从头训练256步／batch4。两fold×两seed×三条件＝12学生、3072更新、12288训练状态引用。AdamW lr1e-5／wd1e-4、clip1，原2GIoU+5L1+focal定位目标；原生权重及buffer、原MLP规则、CENTER和Hann冻结。每例只执行适配前L6上原生选中的一个候选；不同输入可能引起原生规则选层不同，不强制固定L9，也不学习新路由。

native_pair和own_pair的源组、同帧列表、抽样索引和参数初始化完全相同；native_all遵循原194全部支持抽样。每组源组等概率、组内状态有放回，少量配对帧不被伪装为大量独立样本。分别报告有效帧和训练引用数。所选候选样本相同预算，实际batch分组调用数可能因原生决策不同而不同，不主张等FLOPs或等时延。

复用206冻结训练引擎，内部receipt schema保留，但新207 plan指纹、不同conditions子目录隔离三条件。表中training_condition为三条件，arm=matched_spatial表示实际结构。普通中断复用已提交完整模型／视频，未提交学生从头拟合；不声称逐Adam步续训。嵌套条件的非有限失败JSON也会阻止继续。

native_all若与206同一runtime（含UUID），模型state SHA必须精确复现206。新GPU允许独立新run，但需先在当前卡按194原父框重放全部1824缓存状态，保持父框、target、真值不变，重算原生L6 token／输出／所选层，三学生都使用当前卡的原生数据；不能把旧卡原生特征与新卡自身特征混为纯父框变化。source_replay表记录差异与是否重算，新GPU只记录模型权重差，不混用旧成绩作主要对照。provider生成和可选原生重放成本单列，不把继续训练provider与从头训练控制比较。

## 评价、成本与统计

每个冻结学生在194原生父框全池上做fit／held读出，共21888状态，用于和206同入口对照；这是原生父框诊断，不是新自身状态held诊断。配对支持的效果以完整视频作为主要结果，不用训练loss下降代替泛化。

完整评价同卡新跑：38原生＋76冻结provider控制＋228学生＝342条OOF轨迹；另有76条fit-provider采集轨迹，不合并当作held样本。以本来源帧数计，采集81034跟踪帧，评价364653帧，总445687帧。额外provider成本和学生训练成本分开记账。

主要检验为own_pair−native_pair的38源组平均AUC；另报对native_all、原生、冻结provider、以及native_pair−native_all、native_all−provider。先平均两个seed再源组bootstrap20000次，seed24194、名义95%区间；另外检查新增失跟、最长失败、原生恢复丢失、好帧损失和两seed。确认集不开放，所有结果仍是历史开发证据。

完整12表：source_replay38、provider_scope4、support76、sampling3648、scope_summary12、training_history3072、control_replay4、readout_video456、policy_summary5、paired30、seed_video_summary342、routing2052。预检不足只产出source_replay／provider_scope／support／sampling四表、manifest和support_decision。内部保存76份采集NPZ及receipt、12学生、456读出、342评价及receipt，来源／smoke／支持封存记录；新GPU另有38份原生重放NPZ及receipt。最终固定REVIEW，不自动采用。

支持外缺目标裁剪未作为定位正例训练，本实验不能证明有新恢复能力。若own_pair没有改善完整轨迹或保护仍受损，降低单独状态来源假设优先级，重新评估定位监督及反馈目标，不持续扩大同一条路线。

## 总体进度和后台运行

一个总体进度条覆盖2次smoke、fit-provider完整轨迹、76组支持检查、3072学生更新、21888读出和342完整评价；新GPU再计1824原生重放状态。终端原位显示，后台约10秒一条快照。预检不足会在对应阶段停止，不显示伪100%完成。

在服务器sglatrack-test中，同步代码后：

```bash
cd ~/workspace/paper
mkdir -p output/stage207
nohup bash tools/run_stage207_state_training.sh > output/stage207/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage207/run.log
```

默认结果根`output/stage207/paired_provider_state_training_v1`，GPU0。可覆盖STAGE207_ROOT／STAGE207_GPU、STAGE194_ROOT／STAGE206_ROOT／STAGE206_BUNDLE、CHECKPOINT和原七数据源目录。须保留206四个matched模型PT及receipt；仅有结果TXT不够生成自身状态。首次L20或L40均可，新同卡重测全部主要控制；同输出根续跑身份必须一致。

完成或支持预检不足时均自动生成`stage207_results_<UTC>_<PID>.txt`，回传该文件。`--check-results`复核已完成结果或已封存的支持不足结果并重新打包。本机无真实194缓存和GPU，CPU合成测试不是正式精度复现。
