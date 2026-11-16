# Stage-118：提供者隔离的提交收益学习

固定日期：2026-11-16（Asia/Hong_Kong）。**实现与本机验证完成，正式服务器结果待运行。** 状态 `COMPLETE_PROVIDER_ISOLATED_COMMIT_LEARNING` 仅表示固定计算完成，不是方法采用或能力保护通过。

Stage-117的均匀样本没有建立预测兼容性零阈值规则的优势，而且几乎没有严格救援支持。Stage-115则已测到保q0有时能避免错误反馈、有时会破坏必要救援。本轮检验：**相同公开证据和模型容量下，学习后继收益，能否比学习当前收益更好地决定一次q0提交，同时避免在普通状态和旧救援案例上产生大损伤。**

它是SGLA生成搜索／共享评分之后的提交机制实验。固定原policy与q0_once两种已测动作，不更换任务、不扩大候选数、不加训旧PPO。没有新图像、GPU/SGLA推理或完整自身轨迹；会拟合新的小型CPU模型，不能笼统称为“无训练”。

[Stage-117结果](STAGE117_RESULTS_REVIEW.md) · [Stage-116公开证据](STAGE116_CACHED_ONLINE_EVIDENCE.md) · [Stage-115后继干预](STAGE115_MATCHED_Q0_CONTINUATION.md)

## 1. 冻结来源与样本

检查Stage-112/113/115/116/117原manifest、decision、模型、缓存、CSV及代码SHA，绑定以下decision指纹：

| 来源 | 固定指纹 |
| --- | --- |
| Stage-112 | `4622ef840a90ccfc658ff91b86652ac460a1371a3da9579e37b08425d8f3803f` |
| Stage-113 | `11cada1ae2be6ad882acb3f6a55e4a0073ea44dc9957b70dfe1c69a799d93d0c` |
| Stage-115 | `141baa4bae77280cf89cf063665b8b4fa03545ba3ae3c98e52ce0137f0b90edc` |
| Stage-116 | `e15637be797024fd188b10dd2f02783ba6b51f381b5e9e81c35b687ea71d2075` |
| Stage-117 | `59f645ac852adc35080cf94dbce9dd8708b4118ee27d8a6bad879ee3e476d3cc` |

Stage-117提供744个去重物理替换起点，其中uniform448、stratified309、交集13。Stage-116的anchor_evidence提供Stage-115原34个已测压力起点。本轮共778个去重预测点，791个研究membership；不是791个独立样本。

三个研究分开报告：

- **ordinary_uniform：** 448点，主要评价研究，也是唯一允许用于拟合的样本来源。
- **ordinary_stratified：** 309点，沿用Stage-117总体层质量权重，辅助评价；不将新增分层点并入训练。与uniform重叠的13点属于同一测量，其uniform身份可按原训练规则使用，不能把它们另称独立测试集。
- **pressure：** 原34个条件起点，全部只用于评价，既不拟合也不校准。好q0被换低／中等q0被换低／原坏q0获救分开给结果；不与普通样本混成一个平均数。

28组均为已消耗开发源组，28确认组继续封存。pressure已按GT选出并反复查看，不是独立验证。Stage-117的抽样也不使这些视频重新成为新数据。

## 2. 为什么每次只用同折另外6组拟合

Stage-112有4个上游提供者，每个生成器／共享评分器在21个源组训练、7个源组留出。Stage-113/117每个视频使用其对应留出提供者。

对待评价组g，门控只用**同一提供者留出7组中的其余6组**训练：

1. g不在该生成器及两个种子共享评分器的训练组内。
2. 门控的6个训练组也未被该提供者用于训练；它们与g使用同一上游模型。
3. g的全部起点、种子、当前／后继标签不进入门控训练和归一化。
4. 34个pressure起点在所有拟合中都排除。

禁止简单混入另外三个上游折的轨迹再称严格无泄漏：那些提供者可能在g上训练过。固定划分下每次6视频×2种子×8均匀点=96训练例；28个待评价组各有一个模型包。少量训练源组会限制学习能力，应报告这种限制，不以失败结果排除更充分训练的可能性。

这只保证当前生成／评分提供者和提交头的指定组隔离。上游方法设计、公共预训练权重与此前开发过程并非独立确认；不声称整个项目从未接触这些视频。

## 3. 同容量的两个监督目标

沿用Stage-116的13维在线证据，白名单为：

`hann_advantage, raw_advantage, psr_advantage, entropy_advantage, continuity_advantage, size_stability_advantage, forecast_advantage, agreement_advantage, score_advantage, center_conflict, log_size_conflict, radius_max, scorer_clip_fraction`。

这些量由当前候选观测、历史预测框及冻结模型得出。GT、当前IoU、后继IoU、视频身份与条件角色均不是模型输入。训练时使用GT构造监督目标是允许的；推断只接收公开特征矩阵。

- **myopic：** 回归当前 `IoU(q0) − IoU(policy)`。
- **future：** 回归一次q0提交相对原选择的**实际后31帧平均IoU差**。后续两分支都恢复原冻结生成／评分策略，沿自身历史演化。

同一训练行、相同权重、归一化、容量及求解方法。视频等权→视频内种子等权→种子内点等权，权重和为1；中心和标准差只在96个训练例上拟合，标准差小于1e−6的维度置1，标准化后截断至±10。

每头是13个系数＋截距的岭回归，固定惩罚 `ridge=1`，截距不惩罚，通过加权正规方程求解，不搜索惩罚、阈值、特征、种子或checkpoint。每个目标另有同训练标签的加权均值常数控制。得分>1e−12时提交q0，否则保留原policy选择；回归输出是收益估计，不是校准后的正确概率。

共56个岭头＋56个常数控制=112个小型CPU模型，存为28个JSON包。优化器步数为0不等于没有训练；不更新SGLA、上游生成器或共享评分器。两头只预测一次动作的条件收益，不是PPO，也不拟合多步持续门控策略。

## 4. 固定对照与主要比较

13项：Stage-117的6项固定规则 `policy/q0_all/hann_pair/continuity_pair/corroborated/forecast_pair`；`myopic/future/myopic_constant/future_constant`；两个学习头各自的同率期望控制；两动作 `future_oracle`。

**主要比较固定为 ordinary_uniform／all队列／all角色下，future−myopic的后31帧平均IoU差。** 同时报告future对policy、统一q0、连续性、future_constant和future同率期望控制，不能只选最有利对照。

同率期望控制在每研究、视频、评分种子内，按原分析权重计算学习头保护率，再用该概率混合两份已测后继。先计算全研究上下文的概率，再报告队列／角色切片，不在救援子组内重新匹配保护率。它是同加权期望频率的无选择控制，不是新增随机推理；全保／全不保／单点没有可辨筛选空间，另报有效上下文和点数。

oracle用后继GT选两动作，仅为条件上界，不是可部署算法。policy是Stage-113的generated_shared，不是原生SGLA。

## 5. 效用与损伤必须同时报告

每研究按原点权重在视频／种子内平均，再视频内种子平均、视频等权。all/historical/expanded分别报告。pressure额外分三类条件角色，普通样本支持表保留真实角色及空支持。

令两动作后继差为d，门控选择q0概率为p（确定规则为0/1）。报告：后继IoU、低IoU比例、保护率、实际收益pd、有害干预成本p·max(−d,0)、漏益(1−p)·max(d,0)，以及对两动作上界遗憾；遗憾=有害成本＋漏益。另报单点实际收益<−0.1的数量及最差起点收益。0.1仅为固定描述阈值，不参与训练或选择。

差值另报中位数、最差视频、最小留一视频均差及胜平负。视频层10,000次条件bootstrap、seed23180；不将相邻窗口当独立视频。区间未经多重校正，头的训练集合互有重叠，也没有在每次bootstrap内重训，不能当无条件或确认性置信保证。

若普通平均效用提高但pressure原救援受损，仍不能宣布保护通过；若只pressure改善也不能代表普通母体。即使两者都改善，还需冻结规则后跑完整自身轨迹，检查稳定视频、新失败和参考恢复，再用封存确认组验证。这里没有新完整视频AUC，也没有“通过即可采用”的gate。

## 6. 产物、恢复与本机验证

新增文件：`tools/stage118_nested_commit.py`、`tools/test_stage118_nested_commit.py`、`tools/run_stage118_nested_commit.sh`；Stage-1～117 Python冻结。

manifest固定设计、源组提供者、来源／代码哈希及Python/NumPy版本。每组JSON保存训练ID、训练签名、归一化、系数／常数及正规方程残差；预测NPZ只含公开x、四个分数及元数据，不写GT标签。28包＋28预测NPZ＋28receipt=84个文件。全部预测保存后才计算评价表；完成结果和来源均校验哈希，已存在产物必须一致，续跑复用模型／预测，不重新拟合。

九表：fold_scope、model_scope、fit_metrics、sample_predictions、policy_decisions、policy_summary、paired、support、video_effects。固定模型作用域112行、训练指标56行、预测778行、逐决策791×13=10283行。自动TXT含decision及fold_scope/model_scope/fit_metrics/policy_summary/paired/support/sample_predictions七表；逐决策与逐视频表留服务器。不自动commit/push。

六项本机测试已通过：提供者训练／留出分区与交叉折拒绝；更改待评组／其他提供者／所有pressure标签不影响拟合，改变训练标签会改变签名；已知信号岭解与同容量；两研究独立权重、同率／压力分离和损伤分解；来源归属／区间与权重拒绝；六组两种子合成全流程、先保存预测、零图像／SGLA调用、源文件不变、零重拟合续跑和篡改拒绝。合成流程不等于正式服务器结果。

## 7. 服务器运行

同步本轮文件后，在已有sglatrack-test环境执行。使用原服务器绝对路径来源，不移动旧output；本轮不用CHECKPOINT或GPU参数，分到L20/L40均可，来源核验不重新初始化CUDA。

```bash
cd ~/workspace/paper
mkdir -p output/stage118
nohup bash tools/run_stage118_nested_commit.sh > output/stage118/run.log 2>&1 < /dev/null &
```

```bash
tail -f output/stage118/run.log
```

默认结果根 `output/stage118/provider_isolated_commit_learning_v1`。支持 `STAGE112_ROOT/STAGE113_ROOT/STAGE115_ROOT/STAGE116_ROOT/STAGE117_ROOT/STAGE118_ROOT` 覆盖；CPU线程缺省1。断开SSH后nohup继续。已完成可用 `bash tools/run_stage118_nested_commit.sh --pack-only` 校验并重新生成TXT，正式文件名形如 `stage118_results_UTC时间_PID.txt`。

来源完整核验仍可能读取旧NPZ以检查完整性，但不会读图像或执行网络。若本轮源文件／代码／Python或NumPy环境在部分运行后改变，不覆写manifest，先保留原结果并使用新Stage-118根。
