# Stage-63：共同历史下的一次候选选择与32帧后继

日期：2026-11-03，Asia/Hong_Kong。开始前已读取WORK_HANDOFF第0节和[Stage-62结果复核](STAGE62_RESULTS_REVIEW.md)。本协议及代码在读取本轮服务器后继结果前固定。

**结果更新（2026-11-03）：服务器已完成12例／48分支／6000搜索，diagnostic及5张CSV已齐，见[Stage-63结果复核](STAGE63_RESULTS_REVIEW.md)及WORK_HANDOFF第0.18节。下文保留结果前方案与历史运行入口，代码按结果冻结，本轮不重跑。**

## 研究问题与可发展的贡献

Stage-62说明，原生0226和E0307在失败附近没有良好已解码候选，但更早和恢复期间存在少量漏选机会。尚未知道：**同一个真实预测历史下，只改变一次接受的候选框，能否改善随后跟踪；当帧更好的框，是否也有更好的后继。**

本轮把“保留q0、避免局部选择破坏有效状态”的想法落实为一次干预原型，并实际展开全部四候选的短后继。它为后续GT-free保守选择／恢复机会保护提供直接对照，允许正、负、零效果出现；暂不拟合新评分器或宣称完整在线方法成立。

基线仍为原始SGLA V1，另保留固定E头检验适配后的情形。原始权重、模板、骨干、原生候选块、选择器及头均不训练。E沿用既有size／offset适配权重，来源折和种子不重新选择。任务仍是VisDrone-SOT单目标跟踪。

## 固定12个病例

读取完整Stage-62封存诊断、8表、代码哈希和上游Stage-61 L40来源；不重写或重跑旧summary。病例来自5个已消耗开发视频，属于定向机制实验，不是12条独立视频或新测试集。

| 病例 | 来源政策 | 锚点规则 | 目的 |
|---|---|---|---|
| R01 | 原生0304 recrop | 首次q0≥0.5却所选<0.2；现有记录555 | 明确即时损伤能否被一次q0保留修复 |
| R02 | 原生0226 history | 首次存在良好候选却漏选；现有记录15 | 早期漏选的后继价值 |
| R03 | E0307／2132 history | 首次良好q0被丢失；现有记录42 | 失效前的状态保护 |
| R04 | E0307／2133 history | 同上；现有记录41 | 对应另一种子 |
| R05 | E0307／2132 history | 441～501中唯一存在良好候选却漏选的帧 | 恢复附近稀少机会的价值 |
| N01 | 原生0226 history | 首次失败起点且四框均<0.5；现有记录115 | 无良好解码候选的负例 |
| N02 | E0307／2132 history | 同上；现有记录392 | 可见但输出差的负例 |
| N03 | E0307／2133 history | 同上；现有记录395 | 对应另一种子 |
| N04 | E0304／2132 history | 同上；现有记录1093 | 更晚退化及后续覆盖损伤 |
| P01 | 原生0176 history | 固定162，要求q0≥0.5 | 已有较好状态的保护 |
| P02 | 原生0204 history | 固定664，要求q0≥0.5 | 避免原current失效的正例 |
| P03 | E0304／2133 history | 固定554，要求q0≥0.5 | 高收益种子的已有状态 |

所有首次条件均从服务器原候选缓存按原整数IoU协议重新定位，并封入新manifest；不是根据Stage-63分支结果选点。R05具体帧由prepare自动解析，必须恰有1个符合条件，否则拒绝，不猜帧号或追加扫描。其他条件不满足、没有完整32帧或来源混合时也拒绝，不临时换病例。

## 干预与后继安排

对每个病例，读取源轨迹在锚点之前的实际预测历史，沿用长度5的因果历史窗口。原首帧图像与首帧框只负责固定模板初始化；锚点不是GT重置。

1. 在相同进入历史下，执行一次原政策的四个原生搜索，保存每个公共输出框、裁剪、原24项特征、分数和完整score／size／offset响应图。四次调用共享给后面的四条分支，实际计费一次。
2. 分别接受q0、q1、q2、q3的公共输出框作为锚点输出。**只改变本次接受哪个框**；不平滑输出、不替换GT尺寸、不改父框构造或Hann解码。
3. 每条分支从自身刚接受的框继续推进31帧，包含锚点共32帧。此后全部恢复该来源政策：history用原因果历史四候选、recrop用原当帧q0输出的四候选，均按原Hann峰值和原tie规则选择。每条分支每帧实际做四次网络搜索，不借用其他分支或旧轨迹的未来结果。
4. 接受原记录候选的分支必须逐帧精确复现来源32帧的输出、候选父框／框、特征、分数、选择、裁剪哈希、选层及调用数组。共享锚点四候选也须精确复现；不通过则停止，不放宽误差。

V1无在线模板／特征记忆更新，因此每条分支重新初始化相同首帧模板，并恢复源预测历史与正确逻辑帧号；源对角线检查同时验证这一恢复足以复现实际公共路径。不同病例可以有不同进入历史，**同一病例的四分支才是共同起点**。候选重复也照样执行、保留四条分支，并报告独特初始输出数量，不把重复框当作独立干预。

演员函数只接收图像、首帧框、已经发生的预测历史及固定配置，不接收GT、未来IoU或源视频身份评分特征。GT仅用于事后分析和已经公开的开发病例选择。

## 计算预算与L40身份

每例共享锚点4次搜索，四条31帧尾段共`4×31×4=496`次，合计500次；12例共：

- 48条短后继分支，1536个分支帧输出；没有新完整视频轨迹。
- 6000次实际新网络搜索、42000个原生块调用。每次搜索仍执行6个前缀块＋1个独立候选块。
- smoke仅运行第一例共享锚点和原选择分支，128次搜索全部复用于正式采集；不额外加算128次。已有receipt的缓存续跑不重复网络调用。

比较一条完整32帧候选政策时，逻辑预算为128次搜索；本轮四分支实际共享锚点，故不能把所有分支的128简单相加当实际新调用数。图像读取、哈希、重置和磁盘写入不构成部署计时，不给速度或能耗结论。

有receipt的共享锚点／分支不会重复推理；中断时未提交的尾段会从头重跑，失败尝试的额外开销不包含在完成缓存的6000次预算中。

继续使用NVIDIA L40和原软件／原权重／封存配置。prepare会观测本次L40 UUID并绑定；允许与Stage-61来源不同的物理L40 UUID，作为新阶段的显式差异记录，所有新对照须在同一已绑定设备上运行，且来源对角线仍须精确复现。其他型号、软件、权重或模型状态变化不自动接受。prepare之后换卡不能改写已有manifest，应保留原目录，另行明确新运行。启动沿用已测试的封存配置构建器，避免可变YAML影响运行。

## 预定分析

逐候选报告当帧、随后31帧均值、32帧均值、末帧IoU、低IoU帧数、良好帧数和首个连续3帧良好起点。最后一项是局部窗口描述，不当作完整视频无删失恢复率。

三类对比均相对同一病例原记录候选：

| 对比 | 选择依据 | 解读 |
|---|---|---|
| q0−recorded | 固定q0，无GT | 一次保守接受的实际后继效果 |
| immediate_best−recorded | 锚点四框的离线GT IoU | 当前帧最好框是否帮助后继，不作在线策略 |
| future_best−recorded | 看完四条32帧分支后取均值最好 | 事后条件空间，不作训练成绩或可部署方法 |

不把各病例拼成oracle轨迹、不计算新完整政策AUC、不按帧做显著性。多个锚点和两种子依赖同一视频，正负病例又已被开发结果选择，因此不给独立外部验证或新方法PASS。

新响应图可补充：公共框IoU、raw／Hann格点、dense最好IoU、≥0.5格点数、最好GT格点的Hann排名、Hann前5格点是否有良好输出、裁剪覆盖、GT输入尺寸、中心误差及宽高比例。**这里前5是五个网格格点，不做局部峰值NMS，与旧阶段的峰值top5定义不同。** dense解码与公共输出允许原0.001浮点框审计误差；整数IoU可能在边界跳变，两者分别报告，不偷偷互换。Stage-62原缓存没有完整图，本轮才新增采集。

## 产物、运行和回传

新增5个工具文件为`stage63_choice_common.py`、`stage63_choice_experiment.py`、`stage63_choice_summary.py`、`test_stage63_choice.py`及`run_stage63_choice.sh`，都位于tools。来源Stage-1～62源码、旧路径及结果保持冻结。

服务器同步这些文件后，在原`sglatrack-test`环境、获得L40的同一任务内执行：

```bash
cd ~/workspace/paper
export STAGE62_ROOT="$PWD/output/stage62/cached_candidate_selection_v1"
export STAGE63_ROOT="$PWD/output/stage63/matched_one_choice_continuation_v1"
export STAGE63_GPU=0
bash -n tools/run_stage63_choice.sh
bash tools/run_stage63_choice.sh
```

CHECKPOINT默认使用仓库原始`sglatrack_ep0297.pth.tar`，启动会核对哈希。脚本顺序执行prepare→smoke→collect→summary。也可手动顺序执行：

```bash
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
python -B tools/stage63_choice_experiment.py --mode prepare --input "$STAGE62_ROOT" --output_dir "$STAGE63_ROOT" --gpu 0
python -B tools/stage63_choice_experiment.py --mode smoke --input "$STAGE63_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage63_choice_experiment.py --mode collect --input "$STAGE63_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage63_choice_summary.py --input "$STAGE63_ROOT"
```

完成后状态为`COMPLETE_MATCHED_ONE_CHOICE_CONTINUATION_DIAGNOSTIC`。产生diagnostic和6张CSV：case_summary、branch_summary、contrasts、scope、dense_focus、timelines。脚本按用户偏好用cat合并前5张CSV及diagnostic，输出到：

```text
docs/experiments/06_sgla_uav_tracking/results/stage63_results_<UTC>_<pid>.txt
```

回传此TXT即可；逐帧表和响应图NPZ保留服务器。脚本不commit／push。查看单份结果可使用：

```bash
cat "$STAGE63_ROOT/analysis/stage63_diagnostic.json"
cat "$STAGE63_ROOT/analysis/stage63_case_summary.csv"
cat "$STAGE63_ROOT/analysis/stage63_contrasts.csv"
```

## 本机验证

4项测试全部通过，27.903秒、无跳过：首次／唯一条件规则及空支持拒绝；即时最好与后继最好不混用；真实小通道CENTER／公共解码、一次接受与自身历史推进、GT改变不改演员输出、来源对角线；Stage-59→60→61→L40→62→63完整小型流程、E载入、仅L40 UUID差异、精确记账、smoke复用、重复运行零新调用、旧文件逐字节不变及来源／缓存／结果篡改拒绝。小流程2例／8分支／3帧为72次搜索，smoke12次复用；不是服务器6000次结果。

上游Stage-58边界、骨干、图像和设备为fixture，不是独立真实GPU验证。准备时实际12例后继、dense候选情况和q0得失未知；现已收到服务器结果，见上方结果复核，不能把准备时CPU测试当作实际改善证据。

CLI help、4文件AST、脚本LF和维护检查通过。本机沙箱内Git Bash因MSYS权限不能启动，随后经自动审批执行沙箱外只读`bash -n`成功，Stage-63脚本语法检查通过；无需把旧Stage-62的本机Bash历史限制延续为本轮未完成项。
