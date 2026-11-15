# Stage-108：状态关系 × 来源模式交互的固定门控对照

准备日期：2026-11-15（Asia/Hong_Kong）。服务器已完成，复核见[Stage-108结果](STAGE108_RESULTS_REVIEW.md)。以下保留固定设计，COMPLETE不是方法采用。

## 本轮解决什么问题

Stage-107显示，future在原49维表示上的训练内区分有限，留组后进一步退化。0304有害提交与0176有益提交互为最近训练近邻；0348两来源的48维相同、只有模式标志不同，真实后继效果却一正一负。需要检验当前证据之间的关系与模式作用随状态变化是否有新增选择价值。

本轮固定检验一种16维状态关系及其模式交互，不搜索特征、窗口、模型、ridge或阈值。SGLA、PPO、候选生成、外观核验／写入资格保持原样；只拟合判断一次both_pair提交相对keep是否有益的小门控。它属于搜索历史及外观提交机制的开发验证，不是仅凭增加特征就宣布论文创新成立。

## 来源与四组设计

固定Stage-107决策`7782ea245f6f17fe2b0bd3575edcadcee3da12a23a9bafbb2987d24a465cd0ba`及其Stage-106来源`644e44f73492caf63b5a8dcecb475b3728f3415b13d01eb9651a482e7d8b06d6`。

28个已消耗开发源组、259唯一起点、224个共同有效且有实际提交变化的训练支持；两个既有来源模式共同拟合，全部结果必须回传。来源角色仍只用于描述，不复制样本或作为输入。

| 表示 | 原49维 | 16维状态关系 | 模式×非模式特征 | 维数 |
|---|---|---|---|---:|
| baseline | 保留 | 无 | 无 | 49 |
| relation | 保留 | 有 | 无 | 65 |
| mode | 保留 | 无 | 有，48项 | 97 |
| relation_mode | 保留 | 有 | 有，64项 | 129 |

baseline复用Stage-106的56个模型，逐样本／全曝光分数与动作须复现；不重新拟合或替换旧权重。三个新表示各按28个整源组留出×future／myopic拟合，共168个新CPU模型。新旧模型分别列示，不将168写成SGLA或PPO训练次数。

四组共享相同样本、目标、共同有限性支持、源组划分、加权规则、岭回归损失、ridge=1.0、截距定义和收益>0阈值。维数与容量不同，按列标准化后的有效正则及CPU成本也可能不同，不能称纯粹相同容量／完全相同计算量的比较。

## 固定状态关系

所有新量仅由Stage-106当前49维白名单导出，不读取GT、角色、序列ID、种子、未来或新图像。下列长度量均已按最近自身预测框的面积平方根归一化，宽高量为log比。

四个关系量：

1. `C = log(1 + ||center(b2)-center(b4)||)`：两个提交框中心冲突。
2. `S = log(1 + ||log(size(b2)/size(b4))||)`：尺寸冲突。
3. `M = tanh(error_motion(b4)) - tanh(error_motion(b2))`：pair相对自身过去匀速预测是否更一致。
4. `G = tanh(||log(size(b4)/history_median_size)||) - tanh(||log(size(b2)/history_median_size)||)`：pair尺寸相对过去中位尺寸是否更一致。

再增加12个乘积：

| 关系量 | 与之相乘的当前证据 |
|---|---|
| C | anchor分数差、历史参照分数差、Hann峰值差、PPO概率差，共4项 |
| S | anchor分数差、历史参照分数差，共2项 |
| M | anchor分数差、历史参照分数差，共2项 |
| G | anchor分数差、历史参照分数差，共2项 |
| pair写入资格−four写入资格 | anchor分数差、历史参照分数差，共2项 |

所有分数差沿用pair−four方向。log1p和tanh是固定变换，不依据结果选尺度或截断阈值。新增关系量对横向镜像的符号翻转不变，但四组仍保留原有带方向输入，**不宣称整个模型已经镜像不变**。历史一致也不代表GT正确，只是可检验的线上证据。

## 模式交互与归一化隔离

mode_pairwise为0／1，是当前已有核验模式。对每个留出组，先仅用其他训练组共同有效样本得到原始stem（49或65维）的加权均值及标准差。然后对除模式标志外的各列生成：

```text
mode_interaction_j = (2*mode_pairwise - 1) * ((stem_j - fit_stem_mean_j)/fit_stem_std_j)
```

常量列标准差设1，因此中心化后恒为0，不会重复成为额外模式偏置。stem中只有模式的列不再自乘。全部stem及交互列拼接后，再用训练组统计量做固定岭回归的最终标准化。

两层统计量均不接触被评价组，包括其其他片段、种子和模式；两种监督头使用相同支持，映射及归一化相同。映射和系数一起保存在模型JSON，预测时从当前49维和该固定映射生成输入。留出GT只在决定完成后用于评价。

新增模型的输入指纹绑定训练身份、当前特征及对应训练目标；续跑复用同一输入对应的封存模型，验证映射／归一化与正规方程，浮点容差沿Stage-107的atol1e-10／rtol1e-9。不对已有模型再次求解beta；中断后已发布的表仍须一致，跨环境微差不能覆盖旧表混写。

## 目标、控制与主比较

future目标为既有both_pair与keep后最多31帧平均IoU差。myopic仍为当前pair提交框与原四候选框IoU差；四分支当帧发布相同框，它是提交质量代理。无变化或必要支持无效时keep。

15个评价政策：keep、always_pair、离线future_oracle，四种表示各自的future／myopic，以及各表示future相同提交率的离线均匀选择解析期望。每个率对照按同序列／种子／来源模式的合法锚点提交数量匹配，不是可上线策略。oracle也只作离线上界。

预声明主比较为`relation_mode__future - baseline__future`。必要对照同时看combined对relation-only、mode-only、相同表示myopic、always_pair、自己的同率对照及keep。不能主比较均值为正就忽略必要控制或严重反例。

另外输出两因素交互`combined - relation - mode + baseline`与两个顺序平均，均为future政策效用的整体条件差，不把它们称为唯一因果贡献分摊。

每项效用仍是一次提交后恢复原控制器的短窗差；主要表包含全部259锚点，按锚点→模式→种子→源组宏平均，再分历史／追加与两来源。胜平负使用1e-12并列容差，不用重叠窗口制造独立置信区间。

另给224模型×训练内／留出共448条条件评价，16行宏表。训练内及留组误差只对共同合法、有变化支持计算，与全锚点表的分母不同。两头目标不同，不直接比较自身目标MSE绝对值。宏表给每个指标的有效模型数，R²零方差留空，避免Stage-107极端均值被误读。

## 正例、反例和旧轨迹曝光

Stage-107固定13重点状态继续全部报告，四表示双头共104条，不只挑0304／556或0176／251。关注拒绝有害提交时是否漏掉救援，以及是否保留0091、0304／552等已有正例。

在80份原来源、85,144个旧状态上应用8个固定门控，仅记录预测／提交率／训练范围及标准化值；不回写原历史、不重新推理或产生新策略完整轨迹。baseline须逐条复现原预测。每来源保存一份带回执的8门控预测NPZ，共80份，元数据绑定8模型指纹。

比较不同维数的范围外比例需要谨慎：增加列更容易至少一维超范围，它不是同尺度风险概率，也不新增范围回退门槛。记录中的旧轨迹提交仍不是新策略实际干预，不能报告新完整AUC或恢复保护通过。

## 产物与执行

新增入口`tools/stage108_relational_commit_gate.py`、`tools/run_stage108_relational_gate.sh`和`tools/test_stage108_relational_gate.py`。默认新目录`output/stage108/relational_mode_commit_gate_v1`，不覆盖来源。

正式预期：224行model_scope、448行training_summary、16行training_macro、135行policy_summary、90行paired、2072行state_predictions、3885行sample_decisions、104行focus、640行exposure_units、72行exposure_summary。全部10CSV与decision合并回传；168新模型与80曝光预测NPZ留服务器并SHA绑定。旧56模型仅引用原件。

状态固定`COMPLETE_RELATIONAL_COMMIT_GATE_DEVELOPMENT`，没有新方法采用门槛。特征设计已参考消耗的Stage-107结果，本轮是开发实验；旧SGLA／PPO拟合可能与被留组重叠，不是完整系统外部泛化。没有新图像／搜索／SGLA更新／PPO更新／完整轨迹／确认推理。即使有积极信号，仍需预声明完整自身轨迹与封存源组确认。

本机4项CPU测试覆盖：49／65／97／129维和固定关系公式、训练内中心化／常量交互、整源组排除与留出标签隔离、合成模式依赖关系可学习、正规方程／掩码、率控制与两因素算术、端到端缓存／精确旧控制、曝光GT列隔离、续跑不重复拟合及篡改拒绝。合成结果不代替正式服务器结果。

同步代码后运行：

```bash
cd ~/workspace/paper
bash tools/run_stage108_relational_gate.sh
```

无需GPU、checkpoint或L20／L40参数。用原Python环境及完整Stage-106／107来源；可通过`STAGE106_ROOT`、`STAGE107_ROOT`、`STAGE108_ROOT`覆盖路径。断开远程后继续：

```bash
cd ~/workspace/paper
mkdir -p output/stage108
nohup bash tools/run_stage108_relational_gate.sh > output/stage108/run.log 2>&1 < /dev/null &
echo $!
```

```bash
tail -n 40 output/stage108/run.log
bash tools/run_stage108_relational_gate.sh --pack-only
```

脚本顺序运行拟合／评价、完整性核对、cat合并，打印`stage108_results_*.txt`路径；不自动Git提交或推送。正式回传前，最近完整成绩仍为Stage-103，当前没有新增完整方法通过。
