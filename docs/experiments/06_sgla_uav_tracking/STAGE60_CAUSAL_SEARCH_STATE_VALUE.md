# Stage-60：SGLA自身历史候选与后继收益选择

日期：2026-10-31。主任务为原生SGLA无人机单目标跟踪，数据为现有VisDrone-SOT。输入完整Stage-59根目录，沿用其Stage-58来源中的完整末帧保持头E；不更新SGLA骨干、候选、选择器、响应或几何参数。仅拟合轻量线性选择器。服务器尚未运行本轮。

## 从Stage-59继续检验什么

[Stage-59结果](STAGE59_RESULTS_REVIEW.md)表明，中心与尺寸的特定组合能使冻结头从失败转为较好的后继表现，但当时使用两个保存模型的预测作为分量来源。本轮改用**单个跟踪器自己的近期预测**构造候选，并检验当帧质量和后继收益两种训练目标是否产生不同的完整轨迹。

新增价值的检验对象是输入状态的兼容性与选择目标，不预设统一放缩、平滑或运动外推有效。这里没有在线GT选框，也没有在线读取未来图像；“8帧后继”仅用于拟合视频的离线标签。

## 四个因果候选

保存最近至多5个已接受预测框，包含初始GT框时仅限标准首帧初始化。令当前父框为t−1预测，滞后中心为t−2预测中心；不足两框时复用当前中心。历史尺寸为这些框宽高的逐维log中位数再取exp，不含当前帧尚未接受的输出。

| 候选 | 中心 | 宽高 |
|---|---|---|
| current | 当前预测中心 | 当前预测宽高；整个框直接复制 |
| history_size | 当前预测中心 | 因果历史尺寸 |
| lag_center | 上一个历史中心 | 当前预测宽高 |
| lag_center_history_size | 上一个历史中心 | 因果历史尺寸 |

每次探测均走原始SGLA公共track路径、6前缀＋1独立候选、原Hann与CENTER解码。每个策略保持原首帧模板；探测替代框不会变成已接受历史。选中哪一个，就直接接受它的原始输出，不平滑输出宽高。只改变下一次搜索的父框；预测面积决定搜索范围，同时可能改变目标分辨率、padding和响应竞争。

## 四个完整视频策略

| 策略 | 选择规则 | 每帧完整搜索调用 |
|---|---|---:|
| current | 只运行当前框，完整复现来源E | 1 |
| response | 四候选Hann响应峰值最大 | 4 |
| immediate_value | 预测当帧相对收益最大 | 4 |
| future_value | 预测8帧均值相对收益最大 | 4 |

后三者候选、实际推理预算相同；两种学习选择器的输入、样本、归一化、模型容量和正则完全相同，只改变监督标签。保留current是较强完整末帧保持基线与复现控制，原生native只作为冻结来源参考。所有相同／近似相同候选也实际执行并记账，不以结果为由去重、换候选或调整预算。

## 拟合、标签与隔离

沿用12源组、两折6 fit／6 held-out、种子2132／2133。每个折与种子用对应E头在该折fit组重新生成完整自身轨迹，不能用互补折的OOF轨迹当拟合状态。每fit视频均匀选24个完整8帧窗口，不按失效、GT、Stage-59锚点或收益筛选。

每个窗口从该fit轨迹的自身历史构造四个起点，分别推进8帧：只在第一帧干预父框，随后用普通E头自身输出继续，不在分支中重复选择，也不注入其他轨迹或GT。current分支须复现同一fit来源的对应8帧。全部预测产生后才附加fit GT：

- immediate标签：当帧IoU相对current分支的差。
- future标签：8帧平均IoU相对current分支的差。

因此future标签是“一次干预后普通推进”的收益，不是连续多次在线选择的最优价值。本轮会用完整自身轨迹检验这种迁移是否成立，不能仅看fit回归误差。

24维白名单特征只用当前候选响应、预测、图像尺寸和已接受历史：raw／Hann峰值与间隔、响应熵／PSR、选点差与位置、候选与输出的中心／尺度变化、相对图像大小、padding，以及运动／尺寸历史与创新的交互项。Hann乘积先按原生float32形成，再转双精度算统计，避免改变近并列峰值的顺序。没有GT、coverage、视频名、帧号、其他模型状态或未来帧。

模型是无截距岭回归：特征先减去同状态current候选特征，再用**该折fit数据**的RMS缩放，正则λ=0.01。current预测收益恒为0。选其他候选需分数比current高1e-6，其他并列按固定候选顺序；不搜索系数、阈值、窗口、步数或checkpoint。两种目标各拟合4个小模型，共8个scorer，SGLA训练／新头数量均为0；选择器拟合仍属于监督训练。

## 来源与成本

prepare核验完整Stage-59诊断、8表／缓存及其SHA、对角线与调用审计，并沿原来源链验证Stage-58的模型／轨迹。新目录为`output/stage60/causal_search_state_value_v1/`；Stage-1～59实验Python和结果冻结。

在既有12视频共13,525跟踪帧下，正式预算如下；代码按冻结帧清单重新计算，不硬编码帧数：

- 24条新fit基线轨迹：27,050次搜索前向。
- 576个fit时刻×4候选×8帧：18,432次搜索前向；2,304条短分支。
- 四策略×12视频×2种子：96条新完整OOF轨迹，351,650次搜索前向。
- 合计397,132次完整搜索前向、2,779,924次block、397,132次head及397,132次selector搜索调用；没有额外仅头前向。首帧模板初始化不计入搜索口径，未单独测量其设备代价。

这轮主要成本是四候选完整推理，明显高于Stage-59小诊断；不声称部署加速或能耗改善。保存每视频实际call数及去掉前10帧后的仪表化pipeline中位数／P95，计时不含图像读取／哈希。各策略自己推进，虽然单帧调用数固定，轨迹和候选内容可以不同。

smoke先完成第一个fit视频及前2个窗口，这些提交缓存被正式collect复用。另用只在这部分fit数据上拟合的临时scorer检查一个held-out视频前32帧，四策略共416次额外搜索；临时scorer和短轨迹不进入正式拟合／评价。总计正式加额外smoke为397,548次搜索调用；模板初始化仍不计入该数。

按fit视频、fit窗口、完整OOF视频逐项提交缓存及收据。中断后重复同一命令复用已提交项；未提交窗口／整视频重跑。已有提交被修改、混入其他权重／折或代码变动时拒绝继续，不覆盖旧证据。

## 预定判读

主要对比future−immediate，全视频宏AUC；必要控制为future−response和future−current。先在视频内平均两个种子，再按视频bootstrap；上述3个AUC比较给出Bonferroni家族区间。固定重叠OOF拟合下的区间是条件开发描述，不能包含全部训练不确定性。

开发积极条件为：主要均值≥+0.5 AUC且两个种子主要均值均正；对current均值≥+0.5；三个必要AUC比较的家族下界均>0。同时要求：

- 不在current原稳定视频引入新失效；低IoU均值不比current高。
- 原生可靠帧的future−current AUC区间下界≥−1.0，作为固定实践容差。
- current原先优于native的视频仍全部优于native；报告收益幅度和最差视频。
- 至少4视频有实际预测改变，宏改变比例>1e-4；不把选择了同像素／同输出候选算成有效处理。

即使通过也只标记`PROMISING_CAUSAL_SEARCH_STATE_PILOT`；否则`REVIEW_CAUSAL_SEARCH_STATE_PILOT`。fit候选覆盖、正收益选项、回归误差、原生及current保护、恢复与再失效分开报告。可靠帧／current失败帧AUC单列支持视频数；current保护不等同于native能力已全部恢复。candidate不足、价值误判与反复选择的状态偏移是不同失败可能，不能在结果前假定哪一种已解决。

## 本机验证

5个新Python文件：common／runtime／experiment／summary与test。测试使用真实小通道CENTER头和原生公共解码；上游Stage-58来源边界由合成fixture提供，随后真实生成完整Stage-59诊断并通过Stage-60来源验证。它不重复跑全部Stage-51～58训练，旧冻结来源链的测试记录仍保留。

4项CPU测试全部通过（10.333秒、无跳过），覆盖：分量构造与current逐位复制；零基准动作价值与并列保守选择；源组隔离／缓存篡改拒绝；真实CENTER的四候选调用与仅接受自身历史；未来GT改变只影响离线标签；完整Stage-59→Stage-60 prepare／smoke／collect／fit／evaluate／summary；fit和评价缓存复用、held-out GT不改变本折scorer、来源／模型／结果表篡改拒绝。跳过真实Torch／OpenCV测试不算通过。CPU合成测试不代表GPU精度或方法效果。

## 服务器执行

同步5个Stage-60 Python文件与文档，保留Stage-59及完整上游模型／轨迹，在原环境逐条运行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE59_ROOT="$PWD/output/stage59/parent_component_continuation_v1"
export STAGE60_ROOT="$PWD/output/stage60/causal_search_state_value_v1"

python -B tracking/stage60_state_experiment.py --mode prepare --input "$STAGE59_ROOT" --output_dir "$STAGE60_ROOT"
python -B tracking/stage60_state_experiment.py --mode smoke --input "$STAGE60_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage60_state_experiment.py --mode collect --input "$STAGE60_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage60_state_experiment.py --mode fit --input "$STAGE60_ROOT"
python -B tracking/stage60_state_experiment.py --mode evaluate --input "$STAGE60_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage60_state_summary.py --input "$STAGE60_ROOT"
```

优先用cat提供以下五份材料，过长时作为文件附件：

```bash
cat "$STAGE60_ROOT/analysis/stage60_decision.json"
cat "$STAGE60_ROOT/analysis/stage60_policy_summary.csv"
cat "$STAGE60_ROOT/analysis/stage60_paired.csv"
cat "$STAGE60_ROOT/analysis/stage60_scope_summary.csv"
cat "$STAGE60_ROOT/analysis/stage60_fit_support.csv"
```

完整9表还包含video_summary／seed_video_summary／retention_summary／focus／events；这些及fit缓存、模型、轨迹、manifest和收据全部保留。当前设计由已有开发结果指导，尚无独立验证、压缩或部署收益结论。助手不代用户commit／push，也没有登录服务器运行GPU。
