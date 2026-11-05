# Stage-73：固定预算的actor／critic分别梯度裁剪对照

状态：服务器已完成，decision＋12表齐全，`REVIEW_SEPARATE_CLIP_PPO_PILOT`，见[结果复核](STAGE73_RESULTS_REVIEW.md)。PPO对原共同裁剪+0.5235，主区间跨零、两seed方向相反，保护未过。本页保留结果前处理／对照／预算及实现验证历史，接续[Stage-72诊断](STAGE72_RESULTS_REVIEW.md)和[Stage-71完整PPO结果](STAGE71_RESULTS_REVIEW.md)。旧代码、模型及结果保持原样。

## 1. 本轮要检验的具体问题

Stage-72发现，PPO与监督在相同缓存状态上的当帧IoU净差很小，PPO自身历史候选支持更弱；PPO概率多数近半、KL小，日志critic／actor梯度范数比约180～231。它提示优化尺度值得做实验，但没有证明critic或共同梯度裁剪是唯一病因，Adam自适应矩也会影响有效步长。

本轮检验：**在相同SGLA、初始化、种子、目标和训练次数下，将共同范数裁剪改为actor／critic分别裁剪，能否改善完整跟踪，且保护原稳定能力及恢复？** 本轮是训练机制对照；即使改善，也要再区分优化读数与闭环结果，不能仅凭“用了分别裁剪”就宣称新算法或创新成立。

保持SGLA V1无人机单目标跟踪、VisDrone-SOT、原checkpoint／CENTER头／独立候选和固定首帧模板。每次SGLA搜索执行6个前缀块＋1个独立候选块；每帧仍是既有四搜索，不增加重检测、模板更新或新动作。

## 2. 唯一处理及明确保持的条件

| 项目 | 共同裁剪对照 | 分别裁剪处理 |
|---|---|---|
| 来源 | Stage-71最终PPO／myopic包原样复用 | 新训练PPO／myopic包 |
| 梯度处理 | 反传后，actor＋critic全部参数共同norm clip1 | 同一次反传后，actor norm clip1、critic norm clip1 |
| 总梯度上界 | 共同范数≤1 | 两分支各≤1，合范数最多约√2 |
| Adam | 同一个Adam管理两独立分支，LR3e−4、epsilon1e−5 | 相同实现、相同LR／epsilon，不拆分优化器 |
| loss | 原PPO clipped／监督BCE＋0.5 value MSE−0.01 entropy | 直接复用原loss代码，系数不变 |
| 网络 | 独立actor／critic，各128→32 Tanh→1 | 相同架构，每臂8322参数 |
| 初始化／顺序 | 原2171／2172，封存视频顺序 | 初始参数／norm逐值核对原step0，复用完整job顺序 |
| 训练预算 | 原四遍fit、四epochs、mini-batch256、最终step | 完全相同，不挑checkpoint或扩训 |

分别裁剪的整体范数上界确实改变，这属于本轮处理，不称“总范数预算相同”。actor／critic本来就不共享参数；本轮也不改reward尺度、value系数、GAE定义、warm-start、记忆或阈值。

在线状态仍为原128维因果候选特征，复用本折Hann fit归一化，逐维clip±10；二元动作保留q0／接受冻结本折Q提议，无有效不同提议时强制0。奖励为实际选框IoU、gamma0.99／lambda0.95、eligible advantage标准化、PPO ratio clip0.2。评价为最终logit>0接受，零或负保留q0。

## 3. 状态曝光和公平比较的边界

新PPO实时收集完整fit视频，段内模型固定、段后更新，预测框形成后续输入；不使用缓存伪环境或GT失效重置。其myopic控制在同一状态矩阵／标签／GAE return／mini-batch排列上同步更新，使用相同分别裁剪，评价时各自滚动。

共同裁剪PPO原模型按字节复用，不再训练；它与新PPO各自的训练轨迹在更新后可以不同。**跨裁剪条件匹配的是初始化、视频曝光次数、随机种子规则和优化次数，不是逐帧状态、标签或梯度完全相同。** 因而主比较估计整个训练处理的最终效果，含后续状态分布变化。新／旧监督之间的交互同样包含对应PPO曝光变化，不是唯一梯度机制归因。

复用原共同裁剪模型可避免重复原训练，又保留已封存的正式基准。新root必须独立；只有合并TXT不足以运行，需服务器完整Stage-72及Stage-71和上游训练／评价来源链。

## 4. 完整评价、设备与精确预算

11种政策：五旧控制q0／Hann／quality／future／mixed_future；共同裁剪joint_ppo／joint_myopic／joint_rate；新分别裁剪ppo／myopic／ppo_rate。旧五各12条，其余六种各2seed×12条，共204条。全部政策在本轮绑定的L40上评价；旧132条必须精确复现Stage-71已存字段，decode误差沿原容差单独检查，不放宽旧对照。

只允许物理L40 UUID变化，软件、配置、checkpoint和模型身份不变；不复用旧卡数值代替当前同卡对照。来源bridge每折四搜索、共8，额外计费。PPO／监督新训练各4包，共8个新包，复用8个原共同裁剪包；率控制不训练新网络，只在相同全fit Hann状态按Q gain匹配对应PPO确定性接受率，实际评价率可以不同。

| 产物／计算 | 固定正式预算 |
|---|---:|
| 新完整fit段 | 96，两折各6 fit／6 held-out、2seed、4遍 |
| 新实时环境步 | 108200 |
| 新训练SGLA搜索 | 432800 |
| 新PPO／myopic每臂更新 | 1888，按原视频长度及mini-batch精确重建 |
| 完整同卡评价 | 204条，其中132旧政策重测、72新政策 |
| 评价SGLA搜索 | 919700 |
| 额外来源bridge | 8搜索 |
| 本轮总计 | 1352508搜索／9467556块 |

训练smoke复用第一job第一条完整视频和更新（2392搜索），评价smoke首视频前32帧×17政策／seed实例（2176搜索），正式流程复用，不重复收费。只累计正式提交证据；异常未提交尝试的实际耗时不声称为0。

JSON记录参数、Adam状态、原模型指纹和前链；NPZ＋receipt绑定更新前模型。中断后已提交轨迹可补未提交CPU更新，已提交更新不重做；未提交视频从首帧和固定行为seed重做，不使用后续GT续接。初始化只改变CPU控制器的局部随机流，不重设SGLA CUDA流。

## 5. 判断标准：优化效果与方法采用分开

主要AUC比较是**新ppo−joint_ppo**。同时报告对新myopic、Q、future、新率控制、旧监督／率、mixed／q0／Hann的全部差值；增加myopic−joint_myopic与裁剪×目标描述性交互，保留逐视频／seed和0304／0307／0176／0204／0226等完整关注轨迹。

种子先在视频内平均，再对12视频bootstrap，固定2171／20000次。四关键AUC比较对joint_ppo／myopic／Q／future使用Bonferroni分位0.05/8与1−0.05/8；其余为名义区间，不能混写成校正结论。

`optimization_contrast_supported`需：主比较家族下界>0、均值≥+0.5、两个seed各自主差>0、能力保护通过。它只说明本轮训练处理值得保留，不代表已超过强控制。

`pilot_pass`／PROMISING需：四关键AUC家族下界均>0，对myopic和Q均至少+0.5，两个seed对joint_ppo／myopic／Q均正，至少4视频有接受且4视频有否决，能力保护通过。否则`REVIEW_SEPARATE_CLIP_PPO_PILOT`，仍保存所有结果，不挑seed、步数、奖励或阈值。

能力保护沿原规则：原q0稳定视频任一种子0新增失败，低IoU宏差对q0≤0，原生可靠帧AUC差名义下界≥−1。训练概率、梯度或value改善都不能代替完整跟踪保护。12个已消费源组只支持开发判断，没有新外部确认或部署加速主张。

## 6. 同步回传的训练动态

新runtime记录每mini-batch裁剪前actor／critic／joint范数、分支裁剪后范数和各自系数，沿段聚合；保留loss／entropy／ratio clip／KL。汇总同时读取原共同裁剪与新分别裁剪的更新前后参数，比较参数步长、eligible概率／近半比例／接受率、固定原behavior GAE target MSE和完整原行为MC explained variance。

共同裁剪旧日志没有分支裁剪后范数与系数，对应表格留空，不从均值范数伪造。两种条件的value检查分别针对自己的旧行为轨迹，不能称更新后政策的真实return或独立value泛化。

生成17张CSV。脚本用cat合并decision＋12张摘要表：policy11／paired44／seed_effects2／factor2／calibration4／training_scope4／fit_support24／optimization16／training_summary64／focus204／scope204／video_influence12。training_episode384／training_log96及seed_video／video／events五张长表留服务器，参数、Adam、NPZ和receipt全部保留。缺事件为空，不补造时间点。

## 7. 服务器入口

仓库同步新增文件后，在`~/workspace/paper`、`sglatrack-test`环境运行：

```bash
bash tools/run_stage73_ppo.sh
```

默认输入`output/stage72/cached_ppo_diagnostic_v1`，新根`output/stage73/separate_gradient_clip_ppo_v1`，原V1 checkpoint和GPU0。按prepare→source_bridge→train_smoke→train→smoke→evaluate→summary执行。服务器需要Torch／OpenCV；控制器更新在CPU float64，SGLA搜索在L40。

断线后台执行：

```bash
mkdir -p output/stage73
nohup bash tools/run_stage73_ppo.sh > output/stage73/run_stage73.log 2>&1 < /dev/null &
echo $!
```

查看日志用`tail -f output/stage73/run_stage73.log`。结束后回传脚本生成的`docs/experiments/06_sgla_uav_tracking/results/stage73_results_<UTC>_<PID>.txt`一份即可。脚本不代commit／push；旧来源、权重和缓存不清理。

## 8. 本机验证记录

`python -B -m unittest discover -s tools -p test_stage73_ppo.py -v`三项、432.698秒、无跳过，通过：

- 生产规模12视频的204评价／1352508搜索预算、fit／held-out隔离、只允许裁剪处理变化和独立新root。
- 真实CPU float64网络的独立clip机制、相同原loss／Adam、NumPy与Torch前向一致、CPU RNG隔离、带Adam状态的JSON续跑逐值一致；合成actor梯度3、critic40时分别裁剪各约1，区别于共同裁剪actor约0.0748。
- 完整真实CENTER来源链到Stage-71／72后，prepare／bridge／新实时PPO收集／共享监督更新／34条微型完整评价／重复summary通过，22旧评价逐字段精确复现；所有旧文件和SGLA网络不变。提交轨迹后更新前中断可零重复搜索补更新，完整提交后零新搜索／更新续跑。改变后续GT不改变动作／历史；对应held-out GT不可读不影响fit更新。源Stage-72表／Stage-71 checkpoint、新checkpoint／NPZ／结果表篡改及嵌套输出拒绝。汇总可在禁止Torch／OpenCV导入下完成，joint未记录的post-clip单元格为空；主留一统计独立按视频表重算。

微型使用2视频×4帧、2遍fit、hidden4／epochs2／mini2，32新环境步／128训练搜索、34评价／544搜索、bridge8，共680。它验证执行和审计，不是正式训练预算或服务器精度。首次完整测试在新实时阶段继承了上游汇总fixture禁用OpenCV的状态，已修正新测试入口环境；未放宽生产代码。新汇总审查同时将primary_leave_one_out明确改为对joint_ppo并验证算术。

额外独立数值检查：仅在测试中将clip上限设得足够大，使其不触发，双臂新旧更新输出及序列化Adam完全相同，支持除了裁剪操作没有引入额外更新差异。正式clip上限仍固定1，未据测试或成绩选择。

5新Python AST、全新进程禁止Torch／OpenCV的两CLI帮助、Bash `-n`通过。9份相关文档225处本地链接、协议20行表格列数、shell LF和Stage-72原件字节保护通过。维护检查505保留／200移动／106活动模块、0缓存／0换行差异，`git diff --check`通过；旧tracking／lib／tools已跟踪源码未改。未代commit／push或运行服务器GPU实验。
