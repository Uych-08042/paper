# Stage-200 同视频状态差分与匹配点监督

日期：2026-11-25。运行前协议；真实训练尚待服务器Stage-194特征缓存，本机合成测试不作为方法结果。

## 根据与问题

[Stage-199结果](STAGE199_RESULTS_REVIEW.md)残差fit排序77.75%、held54.91%，未超过固定先验59.30%。探索性分解中74.8%的训练MSE改善来自视频平均层偏好，held视频均值与视频内变化误差都变差。因此检验训练时剥离视频平均误差是否有利于学习可迁移的状态差异。

不预设同视频差分必然有效。该监督不等于新时序骨干，不增加在线历史、模板更新或PPO；推理接口仍只接收当前L6特征及固定fit统计。

## 理论与唯一监督干预

保持原Stage-199有符号残差架构：`score_t=μ+P rθ(F_t,g_t)`，`P=I−11ᵀ/6`，rθ使用fit中心化的ordered输入。μ、F̄、ḡ只由fit源组等权确定并冻结。两组都是6720参数、同初始权重／buffer、同初始固定μ选择。

令`e_t=P(score_t−y_t)`。同视频独立均匀抽取两个缓存状态t、s，允许t=s：

`E_{t,s|video} mean((e_t−e_s)²)/2 = E_{t|video} mean((e_t−E[e|video])²)`

系数1/2由方差恒等式确定。差分消去视频平均误差项，仍可能失去有用的视频级差异，需对照检验。原固定先验保持可用，但不能据此保证残差能泛化。

| 新训练组 | 优化目标 | 每步输入 |
| --- | --- | --- |
| point | mean(e²) | 相同两对、共4个状态引用 |
| within_video_difference | mean((e_t−e_s)²)/2 | 与point逐项完全相同 |

point按本轮状态对重训，不能复用旧Stage-199作为唯一控制。配对改变了批内样本相关性，必须把该变化在两组之间固定。

不使用特定帧间隔、失败窗口或真值选样。每对先从19个fit视频均匀选一个，再在该视频的已缓存状态中独立有放回选两个索引；每步2对。重复状态及自配对保留，自配对差分损失为零，是独立采样总体的一部分，不额外删掉或补抽。point仍看到相同重复索引，引用次数明确记录。

源组均匀后再状态均匀，512对／1024状态引用／模型、256步。两折×两种子×两组共8模型、2048更新、8192训练状态引用；独立抽样日程为4套×512对=2048行，由两组共同使用，模型训练的对引用合计4096次。不能将重复引用称为独立训练样本。

## 固定结构与预算

38个历史开发源组，沿用19 fit／19 held两折和2132／2133种子。每视频48个原生父框缓存状态，每模型评价1824状态，8模型共14592条预测。抽样使用原种子映射，但随机数消耗方式变为按对生成，因此新日程不应等于199旧点抽样哈希；只要求本轮两组日程及样本完全一致。

同原AdamW学习率1e-5、weight decay1e-4、梯度上限1。两组都使用原Stage-199 interaction_residual架构及完整fit统计，固定模型和优化机会，只改变上表目标。报每步point与half_difference两项误差及梯度，不比较不同目标的loss数值就宣称性能改善。

无新图像、无原SGLA checkpoint加载、无骨干／CENTER前向、无完整轨迹和确认源组；训练仅在CPU运行。不增加层数、在线支路、动作数量或门槛。

## 来源、审计和评价

来源为完整Stage-194特征目录及已归档Stage-199结果包。绑定199决策 `6e98e01801e9f63f34d1a6d97ffb769a34d45566e698e608817805dfa41f10a5`、计划 `0b73e5c12d53685cae7228f94baea461231cf915503f9f3416fda1d5afbd1e06`、原包SHA `12fe41d3e3643e937bbabe922f0253f53788185fb89a82095e9182ecf0ade8df`，核验199全部10表、manifest、代码和原协议，保持旧文件冻结。读取194原NPZ／receipt并逐状态对齐六层真值；fit先验须与199一致。

模型、优化器、统计NPZ、预测与receipt独立保存。已提交缓存和模型复用，来源／代码／运行版本改变拒绝覆盖原manifest；单个未提交模型中断后重训。总体进度条覆盖输入、训练和预测，最终显示100%。

12表为model_scope、fit_summary、source_bridge、pair_schedule、pair_support、error_summary、error_video、policy_summary、policy_video、paired、frame_predictions、training_history。额外记录每一步两对的源组和索引、实际帧号、自配对、每视频引用和唯一状态数，以便重建训练计数。误差表保留视频平均项及视频内项、各自能量与视频内skill；`video_decomposition_error`独立于原绝对／共同误差恒等式字段，避免混同。

选择控制保持trained、prior_only、fit_prediction_mean、L9、oracle。统一原诊断并列规则：L9距最大分数≤1e-8保留L9，否则第一个精确最大。状态无对应期望为Σ_lπ_lȳ_l，held标签只作评价，不用于拟合或决策。frame_predictions记本轮对抽样的状态引用次数，不套用199旧计数。

共同主要判断仅看held差分组：

1. 对同次重训point的实际IoU差。
2. 对固定prior_only的实际IoU差。
3. 对自身同频率无状态对应期望的IoU差。

附视频内MSE、排序和源组／种子一致性。先源组内平均种子，再按原38源组bootstrap；区间为探索性的名义区间，不作独立外部确认。若主要增量有正向区间支持、两种子一致、最小留一源组仍正，才考虑正常入口完整自身轨迹。仅差分训练误差下降、动作增加或超过受损控制不算通过。

本轮不依据held调配对间隔、差分权重或学习率，不因旧固定L7在缓存较高就部署它；Stage-194已有固定L7完整轨迹受损证据。缓存即时IoU不能称新AUC。

## 服务器运行

同步新代码、协议和仓库内Stage-199结果包后，在原环境运行，无需GPU：

```bash
conda activate sglatrack-test
cd ~/workspace/paper
bash tools/run_stage200_within_video.sh
```

后台运行：

```bash
conda activate sglatrack-test
cd ~/workspace/paper
export STAGE200_ROOT="$PWD/output/stage200/matched_within_video_v1"
mkdir -p "$STAGE200_ROOT"
nohup bash tools/run_stage200_within_video.sh > "$STAGE200_ROOT/run.log" 2>&1 < /dev/null &
echo $! > "$STAGE200_ROOT/run.pid"
tail -f "$STAGE200_ROOT/run.log"
```

来源可改STAGE194_ROOT和STAGE199_BUNDLE，默认分别为`output/stage194/module_ablation_v1`与`docs/experiments/06_sgla_uav_tracking/results/stage199_results_20261125T113420Z_1204543.txt`。不需要重跑198／199。

完成自动生成`stage200_results_<UTC>_<PID>.txt`，包含decision、manifest与12表，回传该文件即可。单独核验：

```bash
python -B tools/stage200_within_video.py --output-dir "$STAGE200_ROOT" --check-results
```

## 本地验证边界

7项测试通过：配对日程的确定性、fit范围、有放回及1024引用预算；真实199原包身份和缓存篡改拒绝；视频内／视频间分解恒等式及字段保留；遍历全部有序状态对验证1/2方差恒等式、视频级层偏移抵消和梯度不变、自配对零损失；两组同初始化／同日程／同参数训练、buffer冻结及模型复用／篡改拒绝；真实尺寸6720参数及固定先验初始化；38组合成缓存完整8模型／2048步／12表／三项主比较、按日程重建每状态计数、禁止CUDA初始化和完成复用。测试文件的CSV读取使用显式上下文关闭。

合成端到端为小维数特征以检验流程，不能代替真实192维L6表征的训练。本机没有Stage-194真实特征NPZ，真实Stage-200尚未运行。旧199研究结果及其协议、源码保持冻结。
