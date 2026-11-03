# Stage-68：fit组成对后继增益学习与完整轨迹检验

日期：2026-11-03，Asia/Hong_Kong。已读交接0.27及[Stage-67复核](STAGE67_RESULTS_REVIEW.md)。本轮已实现，服务器尚未启动；以下在取得Stage-68结果前固定，不按结果补搜时间窗、岭系数、阈值或样本。

## 问题与新增检验

Stage-67在0176:310显示：共同起点放行一次，未来仍执行protect，32帧均值可提高0.4961；0307:370同样放行反而更差。当前q0资格和当前IoU差不足以代替接管后继价值。Stage-66完整保护政策失败与Stage-60绝对未来均值评分器负结果都保留。

本轮判断对象为：**相同状态下，本次采用冻结Q提议，相对于采用本帧q0，未来都执行同一个冻结Q时，32帧平均IoU相差多少？** 不重新排列四候选、不更新SGLA，只学习Q提议与q0之间的一次选择。该差值是固定Q后继下的一步偏离收益，不是最优价值，也不等于部署新门控后的自洽未来价值。

## fit隔离与统一取样

沿用12个已消费开发源组、两折每折6 fit／6 held-out，只使用原生V1，不含E。Stage-67结果和修复receipt完整绑定为来源；其7个事件和分支标签不进入本轮训练。

fit起点只用Stage-66的原生Hann历史。Hann的实际输出与四候选由冻结网络、Hann规则产生，不依赖OOF质量预测器；缓存中的OOF Q／q0critic辅助决定不作为训练输入。对当前折fit视频，**重新用当前折的fit-local冻结Q**产生提议，后继也使用该折Q。不能将这些fit视频原先由相反折OOF Q推进的历史当作本折fit状态，否则会引入held-out监督信息。

每个fit视频按时间排序全部满足以下条件的时刻：Q提议不是q0、提议框与q0框实际不同、从该帧起能完整推进32帧。在这个可用列表上均匀取最多32点，无GT／事件／IoU触发，不强行包含310、370或其他已知成功点。每视频空支持明确记录，不找held-out事件代替；整折无可用点则退出。prepare封存具体锚点及支持统计，后续不变。

同一锚点只做一次共享四搜索，随后两条分支分别接受q0(veto)或Q提议(allow)，再各自执行31帧Q并使用自身预测历史。共享候选观测须精确复现原Hann缓存，评分值则用当前折Q重算；后继无GT父框重置。保存raw／size／offset并逐帧验证真实裁剪、特征、dense解码、原生块／head调用。

GT只在实际分支采集完成后计算监督：

\[
y_{\rm immediate}=\operatorname{IoU}(b^{allow}_t,g_t)-\operatorname{IoU}(b^{veto}_t,g_t),\qquad
y_{\rm future}=\frac1{32}\sum_{j=0}^{31}\left[\operatorname{IoU}(b^{allow}_{t+j},g_{t+j})-\operatorname{IoU}(b^{veto}_{t+j},g_{t+j})\right].
\]

两种标签使用完全相同的真实分支样本，不用影子当帧IoU虚构后继收益。future的训练目标是均值IoU差；低IoU／持续失跟用于后续审计，没有拟合风险概率。

## 匹配预测器与七种完整政策

复用64维因果候选特征，输入为`[f(Q)−f(q0), f(q0)]`共128维。每折两个带截距的岭回归，ridge=0.01；视频等权，视频内部锚点等权，中心／尺度只从该折fit对求得。immediate与future共享输入、采样、权重、岭系数和参数量，只改变目标。预测为有符号增益，不解释成概率。

| 政策 | 完整自身轨迹上的选择 |
|---|---|
| Hann | 原冻结Hann规则 |
| q0 | 本帧原搜索框的网络输出 |
| quality（Q） | 冻结连续质量规则 |
| protect | 冻结Stage-66绝对q0资格保护 |
| immediate | 本折新预测的当帧增益>1e-6时放行Q，否则q0 |
| future | 本折新预测的32帧相对增益>1e-6时放行Q，否则q0 |
| rate_margin | 冻结Q预测增益超过本折fit匹配门槛时放行，否则q0 |

rate_margin在**全部fit原生Hann缓存帧**上匹配future预测的接管率，以视频等权，并完整接纳同分组；不用held-out轨迹率拟合门槛，不新增标签或GPU调用。匹配表明确报告目标率、实现率、最大同分权重、状态数和误差。实际评价各有自身历史，率不保证相同。所有在线决策只访问当前四候选和已接受历史，不访问未来帧或GT。

Hann／q0／Q／protect须在新L40运行中复现Stage-66完整来源控制；仍是每帧真实4次搜索，不通过减少推理调用获得成绩。新增SGLA训练为0；网络、原始首帧模板、候选块、原选择器／CENTER头均不变，最多新增4个CPU岭预测器。

## 评价、保护与统计

主要比较future−immediate，关键控制future−rate_margin，同时比较Q／protect／Hann／q0。以12视频为统计单位，bootstrap固定种子2168、20000次；6项AUC比较的家族区间端点为0.05/12与1−0.05/12。其余指标保留名义区间，不能用帧或重叠短分支当独立样本。

pilot需六项AUC家族区间下界都>0、对immediate和Q均值至少+0.5 AUC，并满足：相对原生q0稳定集无新增失败、低IoU均值不恶化、可靠帧AUC名义区间下界≥−1.0；fit与评价有足够实际使用支持。否则保留REVIEW，不能由某个局部案例或单一对照变好宣布成功。

报告完整宏成绩、配对、逐视频影响、首失效／恢复／末尾未恢复、fit标签正负及是否支持放行的分歧、率匹配与两类预算。标签分歧按是否>1e-6统计，含零附近情形，不等同于严格正负号互换。额外区分非q0索引接管率和实际输出框改变率。当前12组反复参与设计，OOF和区间是开发证据；固定Hann训练状态到自身部署历史的分布差异仍未消除。

## 预算与运行

每锚点4共享搜索＋2分支×31×4＝252搜索；最多384点、768短分支、96768搜索／677376块。7政策×12视频＝84条新完整轨迹，共378700搜索／2650900块。最大总搜索475468／3328276块，首pair的252次smoke及首视频七政策896次smoke均包含并复用，额外smoke为0。实际支持少于384时按manifest计费，不补齐已知好案例。

依赖原始V1 checkpoint、已完成Stage-67及全部来源、`sglatrack-test`和L40。同步本轮新文件后，服务器仓库中执行：

```bash
conda activate sglatrack-test
cd /home/u25600003160604/workspace/paper
bash tools/run_stage68_advantage.sh
```

流程为prepare→pair_smoke→collect→fit(CPU)→smoke→evaluate→summary。每阶段缓存都有receipt与哈希，已提交的缓存不覆写，可复用续跑；输入／源码变化则拒绝继续。空根默认`output/stage68/fit_paired_takeover_advantage_v1`，输入默认`output/stage67/matched_veto_continuation_v1`。可用`STAGE67_ROOT`、`STAGE68_ROOT`、`STAGE68_GPU`、`CHECKPOINT`覆盖默认值。

后台运行并保留日志：

```bash
mkdir -p output/stage68
nohup bash tools/run_stage68_advantage.sh > output/stage68/run_stage68.log 2>&1 < /dev/null &
echo $!
tail -f output/stage68/run_stage68.log
```

日志放在实验根的父目录，不污染prepare空目录。调度环境仍受实际分配时限约束。脚本最后以`cat`汇总decision／policy_summary／paired／calibration／fit_summary／pair_support／scope／focus／video_influence共9份为`docs/experiments/06_sgla_uav_tracking/results/stage68_results_<UTC>_<PID>.txt`，回传此TXT即可。fit_pairs／pair_scope及完整视频／事件长表保留服务器。

## 本机验证与交接

新增4项测试全部通过，167.173秒、无跳过，覆盖真实双分支标签与当帧／后继方向相反、无GT的fit抽样、折内Q重算、GT标签隔离、同状态观测复现、smoke预算复用、四控制完整复现、动作不读未来GT、冻结状态／来源不变、receipt／结果表篡改拒绝和混合可选CSV字段。微型来源每视频仅4跟踪帧，测试中只在fit路径设置确定性因果Q系数曝光以运行两折分支，真实CENTER前向仍执行，原评价Q控制不变；正式参数与Q模型不做此测试替换。

准确通过记录见WORK_HANDOFF 0.28。CPU小样本不是服务器L40实验结果；当前不更新模型框架的成功主张，也未代commit／push。Stage-1～67实验、修复脚本及旧模型／结果字节继续冻结。
