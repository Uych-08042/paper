# Stage-70：固定配对预算的训练历史来源对照

状态：2026-11-04已实现并通过4项CPU测试，服务器尚未运行。接续[Stage-69诊断](STAGE69_RESULTS_REVIEW.md)和[Stage-68完整结果](STAGE68_RESULTS_REVIEW.md)，沿用原生SGLA V1／VisDrone-SOT。当前已收到的全部旧结果保留，不重跑旧阶段。

本轮只检验一个主要问题：**用本折模型实际走出的Q／protect历史补充训练，能否改善后继收益门控及完整轨迹保护？** 这是进入小规模强化学习前的一次有边界的训练状态覆盖对照，不执行PPO，不要求现有监督方法先达到完美，也不把本轮通过设为尝试RL的硬性前提。

## 固定处理与对照

12个重复消费的开发源组，两折各6 fit／6 held-out。SGLA网络、CENTER头、候选块／选择器、原始首帧模板、四搜索候选生成、128维特征、ridge=0.01、32帧冻结Q后继标签、零增益门槛1e-6保持不变。每视频32个配对起点、各视频等权，两种目标在同一批配对上分别拟合。

| 训练来源 | Hann配对／视频 | Q自身历史配对／视频 | protect自身历史配对／视频 | 总数／视频 |
|---|---:|---:|---:|---:|
| Hann-only控制 | 32 | 0 | 0 | 32 |
| mixed处理 | 16 | 8 | 8 | 32 |

Hann-only预测器和接管率控制逐项复用Stage-68模型，重新核对原标签、系数及率门槛可重构；不重新选checkpoint。mixed的16个Hann起点从原32点按索引均匀取子集，复用原共享输入和两个真实后继。新增Q／protect起点分别从各自完整fit历史中均匀选8个可用时刻，不看GT、失效／恢复事件、原Stage-67七个关注点或收益符号。

可用时刻必须有足够32帧剩余长度，当前折Q提议非q0且实际框与q0不同。某一来源不足固定数量即报错，不重复抽样、不把配额改给另一来源，不拿held-out视频补齐。固定样本量不等于相同时间戳、相同数据分布或完全相同采集成本；数据分布变化就是本轮处理。

## fit历史与标签隔离

新增24条fit完整历史：每个视频在其作为fit组的折，分别由该折冻结Q／protect从首帧初始化运行。不能直接用早期该视频的OOF Q／protect评价轨迹替代，因为那条轨迹的控制器来自另一折。Hann行为固定，与存储的OOF辅助Q决定无关；Hann提议仍按当前fold-local Q重算。

从所选历史恢复同一起点，重新观测四候选，校验父框、取整裁剪、图像裁剪SHA、特征和输出与该历史观测一致。分别强制本帧接受q0和Q提议，再各自沿冻结本折Q继续31帧，使用真实自身预测历史；结束后才加载fit GT计算当帧差及32帧均值差。历史采集／门控／后继动作不接收未来GT。

mixed immediate与mixed future共享384对、权重、特征及回归设置；只改变标签定义。新增4个CPU预测器，SGLA不训练。继续Q的配对目标仍与新门控自己未来的动作有差别，不能宣称已经解决完整策略价值学习。

两个来源各自的rate_margin均在**同一全fit Hann状态总体**上匹配相应future预测接管率，采用旧等视频权重和同分处理；不把mixed率控制改到混合历史上拟合，避免多改一个因素。真实评价历史与接管率可以不同。

## 运行身份与统一完整评价

输入为完整`output/stage69/cached_paired_gate_audit_v1`及其Stage-68／67／66上游链，不能只用合并TXT；旧manifest、models、run、NPZ／receipt、全部审计CSV继续保留。新空根为`output/stage70/fit_history_source_v1`。

仅允许NVIDIA L40，软件／参数／checkpoint／网络状态／精度保持来源身份；允许物理L40 UUID改变并显式记录。先在当前卡逐折复现一个Stage-68 fit Hann共享起点，共8次搜索，比较原共享raw／size／offset、特征、选择和输出，独立核对解码容差。然后再采集新fit历史。

完整评价有十组：Hann、q0、Q、protect、原rate_margin、原immediate、原future，以及mixed_rate_margin、mixed_immediate、mixed_future。全部在本轮绑定L40上重新运行，七个旧政策必须逐帧复现Stage-68源轨迹（decode独立按原容差审计），不混合新旧设备完整成绩。每组都4搜索／帧，无自适应少算和速度收益主张。

## 预算

默认有192个新增Q／protect共同起点和192个复用Hann共同起点。每新增点共享4搜索＋2×31×4＝252搜索；mixed训练总量仍384对。复用Hann的48384次旧搜索不算本轮新增；原Hann-only 384对也都是旧训练资料。

| 新增部分 | 完整历史／轨迹或配对点 | 搜索次数 | 块调用 |
|---|---:|---:|---:|
| 来源同输入复现 | 2共享点 | 8 | 56 |
| fit Q／protect自身历史 | 24完整历史 | 108200 | 757400 |
| Q／protect配对 | 192点／384短分支 | 48384 | 338688 |
| held-out十政策 | 120完整轨迹 | 541000 | 3787000 |
| 合计 | 144完整历史／轨迹＋384短分支 | **697592** | **4883144** |

history smoke256、pair smoke252、eval smoke1280均包含并复用正式缓存前缀；来源复现8次是单列的额外验证成本，合计已含，不重复计费。搜索计数不包含图像读取／hash和模板初始化的部署成本，不据此推算实际FPS。

## 预声明比较与判读

主要比较是`mixed_future_minus_future`，回答训练来源是否有帮助。另保留mixed future相对mixed immediate、mixed rate、Q三项关键AUC比较；四项AUC家族区间用同一固定bootstrap（2170，20000次，视频为单位），端点0.05/8及1−0.05/8。还报告mixed immediate−immediate、mixed future−q0／Hann的次要比较。

报告来源主效应、future目标主效应及二者交互，防止把所有改善归于“后继目标”，或把混合历史只对当帧目标有效的结果省略。保留各视频效果、首次失效／恢复、原稳定新增失败、可靠帧表现和低IoU。

保护条件继续沿用：原稳定集0新增失败，相对q0低IoU宏均值不更差、可靠帧AUC配对区间下界≥−1.0。`pilot_pass`还要求四项关键AUC家族下界>0，且相对原future及Q均值至少+0.5。此门槛描述强证据，不是下一轮是否允许尝试RL的门槛；无论结果怎样，先封存这一次固定实验，不按结果继续搜历史配比、时间点、ridge、窗口或门槛。

fit_summary仅为样本内预测误差。对Stage-67七个真实起点分别重新打分，输出baseline／mixed共14行诊断：病例来自已消费开发数据，后继仍Q，不能作为独立测试或用来筛选模型。history_support用fit GT离线描述新历史的失效／恢复与候选可用性，绝不反馈采样。

本轮结束后，结合动作后继敏感性决定下一次有预算上限的PPO对照。监督模型未通过全部门槛时也可合理进行RL小试验；若现有动作明显没有可改善的后继，再调整状态／动作设计。Stage-18／23旧PPO涉及不同处理，其结果不直接裁决本轮搜索决策。

## 服务器运行及回传

同步新增文件后，在`sglatrack-test`环境和仓库根运行：

```bash
cd ~/workspace/paper
bash tools/run_stage70_history.sh
```

顺序：prepare → source_bridge → history_smoke → histories → resolve → pair_smoke → collect → fit(CPU) → smoke → evaluate → summary。CPU的resolve／fit不创建GPU运行时；每个GPU步骤核对真实设备和原始checkpoint。可复用已提交缓存续跑，不覆盖不同设计／收据的目录。`CHECKPOINT`、`STAGE69_ROOT`、`STAGE70_ROOT`、`STAGE70_GPU`可显式设置，默认GPU0／原V1权重。

断开SSH后继续运行：

```bash
cd ~/workspace/paper
mkdir -p output/stage70
nohup bash tools/run_stage70_history.sh > output/stage70/run_stage70.log 2>&1 < /dev/null &
```

日志位于实验根父目录，不污染prepare的新空根。脚本以`cat`合并decision＋policy_summary／paired／factor_summary／calibration／fit_summary／pair_support／history_support／paired_probe／focus／scope／video_influence，共12份为`docs/experiments/06_sgla_uav_tracking/results/stage70_results_<UTC>_<PID>.txt`。默认CSV行数10／28／3／4／8／36／24／14／120／120／12。fit_pairs、pair_scope、video_summary、events及所有原始缓存留服务器；报告仅列来源集合／运行指纹和表SHA，避免数千路径重复进入TXT。

实现入口为`tools/stage70_history_common.py`／`stage70_history_experiment.py`／`stage70_history_summary.py`，测试为`tools/test_stage70_history.py`，脚本为`tools/run_stage70_history.sh`。本机验证及待运行状态见WORK_HANDOFF第0.32节。旧实验Python／shell、模型和回传原件不修改，不代commit／push。

本机4项测试全部通过，149.798秒、无跳过；包含真实CPU CENTER生成的fit本折历史、真实成对分支、20条微型完整评价、七旧控制逐帧复现、模型重构、held-out标签隔离、动作不读未来GT、原网络／来源不变、续跑和篡改拒绝。微型源2视频×4帧、每视频2对，配额H0／Q1／P1；复用Hann的真实读取和审计另行覆盖，生产16／8／8及预算用独立语义测试核对。继承旧测试仅用于fit路径的因果Q曝光以保证短序列支持，正式参数及held-out控制不替换。微型新增440搜索已含8来源复现，三个可复用smoke包含正式预算。CPU测试不是服务器效果。

4份Python AST、全新进程CPU-only导入、Bash语法、202处所涉本地链接、协议表格和维护完整性检查通过；Stage-68／69原件SHA保持不变。
