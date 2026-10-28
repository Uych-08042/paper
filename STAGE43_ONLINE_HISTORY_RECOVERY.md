# Stage-43：单模型历史状态恢复的小规模验证

Stage-42 在增强模型失败后第 10 帧观察到一次父框替换的条件性收益，但当帧替换没有稳定收益，固定时间对照也没有总体收益。另一模型的父框和 GT 确定的失败时刻不能用于部署。本轮检验一个更窄的问题：**只用自身历史预测与当前响应，恢复搜索能否改善完整闭环轨迹，同时保留正常跟踪表现？**

这是已消耗开发数据上的无训练可行性实验。局部搜索、历史运动与失跟恢复均是已有研究思路；本轮不能确立新方法或压缩专属机制。

## 冻结对象与四组对照

- 复用 Stage-40 的全部 16 个冻结六层模型，clean/augmented、4 OOF 折、种子 2110/2111。
- 使用同一批 8 个开发视频的完整序列，两个种子先在视频内平均。没有加载原 checking/tune/validation 划分进行选择。
- 每个模型、视频、种子运行下列四组，共 128 条完整轨迹。从原首帧及首帧初始化框开始；此后父框均来自本组自身输出。

| 组 | 警报且预算允许时的操作 | 额外网络调用 |
|---|---|---|
| native | 始终输出原生预测；影子控制器只记录警报 | 0 |
| motion_hold | 输出历史中心运动外推框，本次不写入视觉历史 | 0 |
| local_search | 以当前上一帧预测为中心和尺度，将搜索父框放大 2 倍，重跑同一个模型 | 1 |
| history_search | 以历史运动外推中心和最后接受尺度构造放大 2 倍的父框，重跑同一个模型 | 1 |

始终使用原首帧模板和原 CENTER 解码器。控制器不调用 SGLA、另一学生、检测器或 GT。原 SGLA 网络仅用于恢复冻结模型和核对来源身份，不执行教师预测。

## 事先固定的在线规则

1. 缓存最近 8 个被控制器接受的自身预测及其响应，前 8 个预测直接接受。这里的“接受”表示模型自身一致，不代表 GT 确认正确。
2. 用最后两个接受框的中心位移除以实际帧间距估计速度，速度上限为每帧最后框等效边长的 0.5 倍；最多外推 10 帧，尺度取最后接受框。外推框按原解码器边界规则裁剪。
3. 当前原生框与预测中心的距离超过一个历史框等效边长，**或**当前响应低于过去接受响应中位数的 0.5 倍，即为警报。响应是原生 Hann 加权最大格点处的原始 score，不是跨裁剪校准过的概率。
4. 时刻 t 已执行干预数不得超过 `floor(0.1*t)`；相邻干预至少间隔 10 帧。motion_hold 同样消耗干预额度。未满足预算时保持原生输出，不会使用未来额度。
5. 两个搜索组均只额外推理一次。仅当候选响应通过同一历史响应下限，且候选中心比原生框更接近历史外推中心时选择候选，否则保留原生预测。两者只在恢复父框构造上不同。
6. 非运动回退的最终预测，满足响应下限及中心一致性才写入历史。没有未来帧、GT 事件时间、GT 候选选择或另一条轨迹的父框进入上述步骤。

这是一个固定基线设计，不保证运动预测正确，不进行无人机相机运动补偿。搜索组每次干预最多增加一次完整六层前向，不能声称每帧只有一次推理或已获得加速。

## 完整轨迹验证门槛

主检验模型为 augmented，不能在结果出来后改选 clean：

- history_search 相对 native 的视频宏平均 AUC 至少 +0.5 点，配对视频 bootstrap 95% 下界大于 0，至少 5/8 视频为正，去掉任一视频后平均仍为正，两个种子各自的宏平均差值均为正。
- 相对 motion_hold 和 local_search 的平均 AUC 差值均大于 0。报告实际前向数；由于各组轨迹不同，触发数可能不同，不能称实际计算量严格相等。
- 原生 IoU>=0.5 的离线定义片段：至少 4 个视频两种子均有 10 帧支持；平均 AUC 差值不低于 -0.5，CI 下界不低于 -1。该片段仅用于评估，不能作在线触发器。
- 宏平均低 IoU(<0.2) 比例不增加；原 native 无持续低 IoU 事件的视频/种子不得引入新事件。
- clean 模型上的 history_search 宏平均 AUC 差值不低于 -0.5，CI 下界不低于 -1；低 IoU 比例不增加，不引入原稳定视频/种子的新失败。
- 主模型实际改框比例至少 0.1%，且至少影响两个视频。零操作不能通过。
- 所有原生轨迹必须复现 Stage-40，框误差<=0.001px，官方 IoU 误差<=1e-12；逐帧核对因果控制器状态及前向/前缀预算。

bootstrap 5000 次、种子 2122，先平均视频内两个种子，再重采样视频。重叠 OOF 训练集与被反复使用的开发视频意味着这些区间不是独立确认性结论。事件后 40 帧均值只是自身轨迹的描述，末尾不足 40 帧保持缺失，不当作匹配起点的干预收益。

## 服务器运行

通过仓库同步以下新文件，无需下载脚本包。先在仓库根目录设置路径，避免沿用指向 `~/workspace/output` 的旧变量。

```bash
cd ~/workspace/paper
export STAGE42_ROOT="$PWD/output/stage42/frozen_model_parent_swap_v1"
export STAGE43_ROOT="$PWD/output/stage43/online_history_recovery_v1"

python tracking/test_stage43_recovery.py

python tracking/stage43_recovery_experiment.py \
  --input "$STAGE42_ROOT" --mode prepare \
  --output_dir "$STAGE43_ROOT"

python tracking/stage43_recovery_experiment.py \
  --input "$STAGE43_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

cat "$STAGE43_ROOT/smoke/stage43_smoke.json"
```

smoke 固定为首视频、种子 2110、两模型、四组各前 64 帧。`PASS_CAUSAL_RECOVERY_SMOKE` 只表示执行、原生复现、状态与调用核算通过；真实警报可能为零，不表示恢复有效。合成单元测试另外覆盖真实执行的恢复路径。

通过后执行：

```bash
python tracking/stage43_recovery_experiment.py \
  --input "$STAGE43_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage43_recovery_summary.py --input "$STAGE43_ROOT"

cat "$STAGE43_ROOT/analysis/stage43_decision.json"
cat "$STAGE43_ROOT/analysis/stage43_policy_summary.csv"
cat "$STAGE43_ROOT/analysis/stage43_paired.csv"
cat "$STAGE43_ROOT/analysis/stage43_video_summary.csv"
```

权重解析顺序为 `--checkpoint`、已有 `CHECKPOINT` 环境变量、仓库原 tracker 配置，并始终核对原权重哈希。若环境变量过期，请先纠正或 `unset CHECKPOINT` 后使用正确的仓库配置。

## 中断、状态与输出

```bash
python tracking/stage43_recovery_experiment.py --input "$STAGE43_ROOT" --mode status
```

中断后重跑原 `evaluate` 指令，已经提交的完整视频会核验复用，当前未写完的视频从首帧重跑。可用 `--sequence 视频名` 分视频执行完整评估，smoke 不接受该参数。汇总必须等全部 128 条轨迹完成，不允许混入其他目录结果。模型、来源、配置、结果及表格都绑定哈希；不能修改旧产物或重写指纹绕过来源审计。

`full/traces/` 保存每帧原生候选、额外搜索候选、父框、响应、外推框、警报、实际选择、历史更新与调用数。`analysis/stage43_interventions.csv` 为离线事件描述，`stage43_seed_video_summary.csv` 为种子级结果，`stage43_evidence.json` 保存完整哈希账本，主 decision 保持紧凑。

本轮如果失败，保存原结果，不在这些结果上搜索触发阈值、窗口、预算或 checkpoint。通过也只进入单独设计的独立验证，不构成论文方法成立或部署速度改善。

## 相关方法边界

[LTMU，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dai_High-Performance_Long-Term_Tracking_With_Meta-Updater_CVPR_2020_paper.pdf) 已使用历史框、响应与外观信息辅助跟踪状态管理；[Global Tracking via Ensemble of Local Trackers，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Global_Tracking_via_Ensemble_of_Local_Trackers_CVPR_2022_paper.pdf) 讨论了局部跟踪、恢复与相似干扰物的困难。Stage-43 不复现这些完整方法，也不把普通恢复规则当作新的压缩创新。
