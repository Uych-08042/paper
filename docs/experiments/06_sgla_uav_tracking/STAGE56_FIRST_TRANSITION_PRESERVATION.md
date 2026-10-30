# Stage-56：第一步状态更新保护与末帧保持的匹配训练

日期：2026-10-30。状态：用户已报告`REVIEW_FIRST_TRANSITION_PRESERVATION_PILOT`，五份结果已收齐，详见[Stage-56结果复核](STAGE56_RESULTS_REVIEW.md)。以下保留预定设计及历史复现命令，不重复启动已完成训练。主任务仍是原生SGLATrack上的VisDrone-SOT单目标跟踪，不更换检测器、骨干、候选结构或选择器。Stage-1～55源码及产物保持冻结。

## 问题与本轮差异

[Stage-55复核](STAGE55_RESULTS_REVIEW.md)显示，0204／0304的巨大当帧差多随进入窗口前的状态变化；0307恢复附近，在共同起点换头后又能产生后继收益。这使“保持一次正确的状态更新”成为值得检验的训练目标。

Stage-54第一帧已有普通GT监督，但原生保持项只直接约束第二帧。父框detach后，第二帧保持项没有经过状态链直接约束第一帧几何。本轮将固定总名义保持系数的一半移至第一帧，检验能否保留0307的后继改善，同时减少0204／0304等损伤；不预言这些视频一定受益。

仅比较旧末帧保持和新双时刻保持，无法排除“减弱旧约束就够了”的解释。因此固定以下三个臂，训练前一次写入manifest，不根据结果选择系数或种子：

| 臂 | 优化目标 | 用途 |
|---|---|---|
| `endpoint_control` | Lsup + Pend | 完整复现Stage-54 rollout_preserve |
| `endpoint_half` | Lsup + 0.5 Pend | 只减弱末帧约束的控制 |
| `transition_preserve` | Lsup + 0.5 Pend + 0.5 Pfirst | 检验新增第一步原生保持的价值 |

三个臂均计算两种保持损失和两种影子梯度，只有表中系数进入优化。不增加推理调用，不增加在线GT门控。该轮是明确的时刻分配及第一步保护原型，不声称全新蒸馏原理或已经证明长期稳定性。

## 数据、状态与梯度

来源为完整Stage-54目录，通过Stage-55已有加载器核验Stage-54模型、轨迹、108行逐种子结果、60行逐视频结果、5行策略结果以及上游来源；无需重新采集Stage-55诊断。Stage-55只提供已消费的设计依据，其五个结果选择锚点不用于专门加权、训练或在线规则。

沿用12个已消费源组、同样的2折及2132／2133两个种子；每折仅6个fit视频参加训练。每步从相同的均匀抽样fit锚点，分别取原生和fit-only donor预测父框，原生参考与当前训练头都从这两个相同父框各走两帧。当前头第二帧使用自己第一帧的实际输出；GT标签在所有分支推进之后才附加。

“第一步”是每个训练短分支的第一步，分布于fit视频的均匀时刻，不是仅保护整视频的GT初始化帧；每视频仍沿用48个预定采样时刻，没有围绕held-out失败位置重新抽样。

原生两个时刻的IoU均≥0.5时，才可能启用保持。第一步和末帧还分别要求参考框中心落在对应学生实际可用裁剪内；不因缺少资格而换样本。Pfirst及Pend各自对两条起点分支取固定分母均值，未合格项为0，不除以合格项数量。报告资格数，不能把名义系数相同当作实际梯度强度相同。

每个保持项用实际原生Hann格点解码到图像坐标，惩罚按参考宽高归一化的中心差与log宽高比，沿用1e-6数值死区。普通监督仍是原有GT监督。只训练原头size／offset参数；原生骨干、候选、选择器、响应参数和BN缓冲冻结。

Pfirst直接约束第一帧几何，Pend直接约束第二帧几何。父框、图像裁剪、取整和argmax仍截断；**不是完整BPTT，不是直接优化32帧恢复率**。训练参考在第一步与学生同输入，之后可形成不同历史；不把末帧参考假设为同裁剪。原生未合格而适配后正确的状态仍靠普通监督，本设计不保证保护全部适配收益。

## 预算、复现和中断恢复

继承Stage-54的初始化、学习率1e-5、AdamW、weight decay 1e-4、梯度裁剪1、256步、固定抽样序列及每步1个双起点样本。正式3臂×2折×2种子＝12个新头，72条完整OOF轨迹；另复用原生／旧几何／旧静态60条来源轨迹作上下文。

每个正式模型1,024个普通监督视图、1,024次原生参考前向、1,024次当前头完整前向、1,024次仅头重算。合计24,576次完整训练前向、172,032个block调用、12,288次仅头重算；另报72条自身完整轨迹的实际推理调用。每臂还执行相同的两种保持梯度审计，不能与Stage-54原有单种影子梯度宣称总计算完全相同。首次smoke另训练3个8步头，并运行原生及3个学习头短轨迹，不计入正式上述预算。

`endpoint_control`正式模型必须与相同折／种子的Stage-54 rollout_preserve最终权重摘要、监督计数、抽样指纹一致，完整OOF的boxes／previous／peak／layer须逐元素一致。若失败，停止解释本轮效果，先排查确定性或环境差异，不能默默放宽检查后声称控制成立。

训练在第128／256步保存包含AdamW状态的带收据checkpoint，重跑同一条train命令从已提交步继续；已提交完整视频推理缓存复用，未提交视频重跑。不得修改已封存设计、旧阶段Python或产物SHA来绕过校验。

## 评价与固定判读

主要对比为`transition_preserve_minus_endpoint_control`，第二个必要控制为`transition_preserve_minus_endpoint_half`；同时报告half−control。所有完整轨迹首帧GT初始化后使用各自预测推进，不用离线可靠性标签在线挑头或重置。先视频内平均两个种子，再对视频做5,000次条件bootstrap，种子2156。

两条匹配AUC对比各报名义95%区间以及97.5%家族区间（分位点0.0125／0.9875，对两个预定比较作Bonferroni）。积极开发信号要求：主要均值≥+0.5 AUC、两条家族区间下界均>0、主要两个种子均值均>0；同时相对原生宏AUC不低、低IoU不恶化、可靠帧AUC至少−0.5、原稳定视频无新增失效，各模型后继实际裁剪确有变化。

全部满足标记`PROMISING_FIRST_TRANSITION_PRESERVATION_PILOT`，否则`REVIEW_FIRST_TRANSITION_PRESERVATION_PILOT`。两者都只是已消费开发组的结果，不能建立独立泛化、新方法采用或部署速度收益。主要区间未过时仍记录效应、逐视频和逐种子的进展；若只胜完整末帧控制却不胜half控制，不能将收益归结为第一步保护。

保留固定原先10条获益视频及0176／0226／0304／0307／0204关注表，分别检查收益幅度、早期失败、恢复、复发、可靠帧损伤。不得只看平均AUC或挑一个种子的成功轨迹。原生可靠标签和旧控制失败队列都是离线描述，不能变成在线控制器。

## 本机验证

真实CPU Torch／OpenCV环境4项测试全部通过，74.919秒、无跳过：缓存身份及篡改拒绝；真实CENTER头、自己预测推进与后续GT隔离；首末节点直接梯度及half控制的代数；完整Stage-51→54来源、fit隔离、3臂12模型／72轨迹合成流程、精确控制复现、AdamW中断续训、缓存复用、checkpoint／来源表／结果表篡改拒绝。CPU合成测试不是服务器跟踪精度结果。

另增强真实目标函数测试并定向复验2项（2.267秒通过），确认实际`transition_preserve−endpoint_half`损失差恰为0.5 Pfirst，实际`endpoint_control−endpoint_half`恰为0.5 Pend；不只测试手写公式或记录一个未参与优化的影子损失。

## 服务器执行

以下为已经由用户完成的历史复现入口，不是下一轮指令。输入必须是已经完成的Stage-54，**不是Stage-55根目录**；输出使用新的独立目录。

```bash
cd ~/workspace/paper
conda activate sglatrack-test
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE54_ROOT="$PWD/output/stage54/short_rollout_endpoint_v1"
export STAGE56_ROOT="$PWD/output/stage56/first_transition_preservation_v1"

python -B tracking/stage56_transition_experiment.py --mode prepare \
  --input "$STAGE54_ROOT" --output_dir "$STAGE56_ROOT"
python -B tracking/stage56_transition_experiment.py --mode smoke --input "$STAGE56_ROOT"
python -B tracking/stage56_transition_experiment.py --mode train --input "$STAGE56_ROOT" --seed 2132
python -B tracking/stage56_transition_experiment.py --mode train --input "$STAGE56_ROOT" --seed 2133
python -B tracking/stage56_transition_experiment.py --mode evaluate --input "$STAGE56_ROOT"
python -B tracking/stage56_transition_summary.py --input "$STAGE56_ROOT"
```

逐条执行，上一条成功再继续。查看结果使用`cat`：

```bash
cat "$STAGE56_ROOT/analysis/stage56_decision.json"
cat "$STAGE56_ROOT/analysis/stage56_policy_summary.csv"
cat "$STAGE56_ROOT/analysis/stage56_scope_summary.csv"
cat "$STAGE56_ROOT/analysis/stage56_paired.csv"
cat "$STAGE56_ROOT/analysis/stage56_focus.csv"
```

另有retention_summary、video_summary、seed_video_summary、events；大表可附件传回。上述5份材料均已收到，不重复索取，也不重复提交Stage-54／55已有表。服务器完整模型、收据、缓存和上游来源需保留；Git不备份这些被忽略的实验产物。本助手未登录服务器、未执行本轮GPU实验，未代为commit／push。
