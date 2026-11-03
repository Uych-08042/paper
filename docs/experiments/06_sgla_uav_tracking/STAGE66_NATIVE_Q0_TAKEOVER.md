# Stage-66：连续质量选择与绝对q0可靠性的接管保护

2026-11-03结果更新：服务器已完成60条完整轨迹，状态`REVIEW_NATIVE_Q0_TAKEOVER_PILOT`。七份回传材料已齐，主比较及保护未过，详见[Stage-66结果复核](STAGE66_RESULTS_REVIEW.md)。下文保留结果前固定协议与测试记录，不按本轮结果补搜参数。

日期：2026-11-03，Asia/Hong_Kong。已先读WORK_HANDOFF第0.22节与[Stage-65复核](STAGE65_RESULTS_REVIEW.md)。以下设计在取得Stage-66服务器结果前固定；本机只实现和测试，服务器尚未运行。

## 问题与本轮处理

Stage-65确认了良好q0被较差替代候选接管的状态，也发现0226／0307已经退化的历史常没有良好公共框。现有quality模型回归相对q0的IoU差，q0分数固定0，不能用该0判断q0的绝对可靠性。本轮新增一个绝对q0质量预测器，在原连续质量选择上限制证据不足的接管，保留弱q0状态的非q0恢复通路。

Stage-64 quality在0003本来稳定，本轮不能将其稳定保留宣传为新增修复reliability的失败。主要检验对象是连续quality尚存的0307等损伤及全12视频取舍；0003作为稳定保护病例保留。

本轮先检验原生SGLATrack V1的五组完整自身轨迹，E迁移待原生结果之后单独检验。原骨干、独立候选块、选择器、CENTER头和首帧模板均冻结；网络仍执行6个前缀块＋1个独立候选块。四个搜索父框、公共解码框和历史长度沿用原实现，不增加候选或重排dense网格。

## 固定五组

| 政策 | 规则 | 用途 |
|---|---|---|
| hann | 原history的Hann峰值选择 | 原生历史基线 |
| q0 | 接受当前帧q0公共解码框 | 原生current输出控制 |
| quality | 原Stage-64连续IoU差评分器 | 主要匹配基线 |
| margin | quality的最好候选只有增益超过本折统一阈值才接管 | 匹配fit接管率，控制单纯减少干预 |
| protect | 预测q0良好时采用较高增益门槛；否则沿用quality | 新增状态条件保护 |

所有政策每帧真正执行4次搜索，含固定q0。返回q0意味着接受本帧q0解码输出，历史继续更新；不是冻结上一帧框，也不提供所有候选都差时的独立几何修复。

记原quality提议为q，预测增益为g；原规则保留q0优先并列及1e-6资格门槛。protect使用：

```text
如果预测的当前q0 IoU >= 0.5：仅在 g > 0.02 时接受quality替代，否则接受q0。
如果预测的当前q0 IoU < 0.5：沿用quality，包括非q0选择。
```

0.5来自既有良好框定义，0.02为预声明的实用预测增益容差；二者本轮不按结果搜索。g是旧模型的预测IoU差，不是实际保证。q0预测裁剪到[0,1]，它是IoU回归值，不是校准的可靠概率或置信下界。低q0分支允许恢复机会，但本轮不保证识别Stage-63的475帧q2。

## 绝对预测器与数据隔离

只用该折6个fit组的原生history／recrop完整缓存；E轨迹、评价组GT及新政策评价结果不进入拟合或门槛匹配。每视频、视频内两来源政策、政策内帧依次等权，沿用Stage-64状态池与权重。合计27050个状态，训练两个64特征＋截距的岭回归预测器；不训练SGLA。

输入为Stage-64同一64项GT-free特征的q0行，保留绝对响应、几何和当前四候选一致性信息，**不减去q0自身**。按fit加权均值中心化、加权标准差归一化（下限1e-6）；截距为fit的q0 IoU加权均值，权重岭系数固定0.01。标签为本帧q0公共输出与GT的IoU，预测截断[0,1]。

原quality系数、scale及目标原值复用，拟合时重建核对，不再训练或选择quality模型。在线只有当前四候选与自身接受历史，函数不接收GT、未来图像、视频ID或帧号作决策输入。帧号仅用于缓存与记账。

## 接管率匹配控制

在相同fit状态、同一封存quality提议上，先计算protect的加权非q0接管率。margin按quality预测增益从高到低，以统一门槛选择最接近该加权率的接管集合；整组同分增益不可拆分，距离完全相同时保留较少接管的前缀。

门槛取最高未接受增益，实际使用严格大于；全接受取0，全拒绝取最大增益。这样相邻浮点数也不会因中点取整而选错前缀。报告protect／margin／quality三种fit接管率、差额和最大同分组权重，不伪称任何情况下精确相等。

该统一门槛只由fit侧决定，评价不追踪或强制匹配接管次数。两政策有各自历史，评价接管率可以不同；本控制区分条件状态信息与fit侧整体减少干预，不能声称评价逐帧或实现算力严格配平。网络预算本来对五组完全相同。

## 来源、硬件与完整轨迹预算

输入完整`output/stage65/cached_same_state_selection_v1/`，新输出`output/stage66/native_q0_takeover_v1/`。加载并验证Stage-65 manifest／reference／diagnostic、11表及其Stage-64完整模型、报告、cache receipt和上游来源，不运行旧summary。新manifest绑定源码、设计、来源及观察到的L40 UUID。

允许显式记录物理L40 UUID变化；其他配置、软件和原权重身份保持封存值。五组都在该绑定运行上新执行，Hann完整字段对原history精确复现、q0输出对原current精确复现、quality完整字段对Stage-64原生quality精确复现。控制不符直接停止，不放松容差。

12视频×5政策＝60条新完整轨迹，共270500搜索、1893500块调用。首视频五组各32帧smoke共640搜索，直接作为完整轨迹前缀续跑，额外smoke为0。中断时完整已提交视频复用，未提交视频可重跑；失败尝试额外成本不在完成缓存预算内。新增模型只有两个q0预测器，无新SGLA权重，没有部署速度、内存或能耗收益检验。

## 预定评价

主要比较protect−quality，必要比较protect−margin、protect−Hann和protect−q0。报告AUC、低IoU、最长失败、原生可靠帧、恢复／复发／尺寸误差及逐视频影响；另报告实际非q0率、原quality提议率、否决率、预测q0良好比例和离线预测误差／条件精度。绝对q0精度只作离线描述，不作为在线GT触发。

视频bootstrap固定种子2166、20000次，四项AUC家族端点0.05/8与1−0.05/8。积极开发pilot要求四家族下界均>0、对quality与Hann均值均≥+0.5 AUC；相对q0无新增稳定视频失效、低IoU均值不增加、可靠帧名义CI下界≥−1，且至少4视频发生保护否决、至少4视频仍接受非q0。否则为REVIEW，保留完整结果。

所有12组和先前结果均已用于设计，是开发验证；PROMISING也不代表独立外部泛化或论文方法成立。fit来源重叠，bootstrap条件于固定拟合，不覆盖独立训练不确定性。无原始checking标签读取，不在本轮结果上搜索岭系数、0.5／0.02、fold、checkpoint或门槛配方。

## 服务器操作与回传

同步5个新增tools文件、`.gitattributes`及文档后，在原`sglatrack-test`环境、分配到L40时执行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE65_ROOT="$PWD/output/stage65/cached_same_state_selection_v1"
export STAGE66_ROOT="$PWD/output/stage66/native_q0_takeover_v1"
export STAGE66_GPU=0
bash -n tools/run_stage66_takeover.sh
bash tools/run_stage66_takeover.sh
```

脚本顺序prepare→fit→smoke→evaluate→summary。需要断开SSH时，将最后一行改为：

```bash
mkdir -p output/stage66
nohup bash tools/run_stage66_takeover.sh > output/stage66/stage66_run.log 2>&1 < /dev/null &
```

日志放在新实验root的父目录，避免prepare把日志误当成非空实验产物。运行期间GPU分配仍需有效；nohup只解决终端挂断，不延长计算任务分配。

```bash
tail -f output/stage66/stage66_run.log
```

脚本用cat将decision及policy_summary／paired／calibration／scope／focus／video_influence六表合并为`docs/experiments/06_sgla_uav_tracking/results/stage66_results_<UTC>_<pid>.txt`，回传一个TXT即可；逐视频、事件和fit支持等留服务器。不自动commit／push。

## 本机验证入口

`tools/test_stage66_takeover.py`包含四项测试：绝对q0状态门控与低q0的替代通路；加权接管率及同分／相邻浮点边界；schema／模型绑定及篡改；真实小型CENTER Stage-59→64缓存、65诊断后接66的拟合／smoke／完整评价／summary流程。

集成测试核对只读本折原生fit组、改变held-out GT不改拟合输入、改变未来GT不改演员输出，三条控制复现、首视频smoke前缀复用、重复运行无新推理、权重与旧来源字节不变，以及来源表／模型／缓存／结果篡改拒绝。小流程2视频、10条Stage-66轨迹、160正式搜索，80次smoke已含；另有单独GT隔离重放，不混入正式预算。上游骨干／图像／设备为fixture，真实CENTER解码，不代表服务器60轨迹已完成。

最终4项测试通过，102.906秒、无跳过；包括统一门槛在相邻浮点值下保留正确加权接管前缀的检查。CLI帮助、4个Python AST、Bash LF和只读`bash -n`通过；Git Bash语法检查经自动审批在沙箱外执行。服务器尚未运行，不能预报保护效果。
