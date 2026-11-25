# Stage-199 固定均值先验与状态层间收益残差训练

日期：2026-11-25。此为运行前协议。真实Stage-194特征NPZ仅在服务器，本机只进行合成验证；不预报新训练结果。

## 根据与待证假设

[Stage-198](STAGE198_RESULTS_REVIEW.md)用fit层均值校准旧预测后，三个模型的全部held状态都选择L7，IoU等同固定均值控制。原始选择相对同频率无状态对应期望也没有可靠增量。问题尚未解决在于学习状态×层交互，而不是缺少一个输出偏移。

本轮检验：将固定层平均偏好、每帧各层共有的难度和状态相关层间差异分开训练，能否改善held的实际选择及状态对应价值？这是指定参数化下的可学习性试验，不是“残差学习必然有效”的结论，也不把常见残差参数化本身当作新颖性。

## 输入和固定统计

沿用Stage-197的ordered输入：L6模板／搜索分别逐通道均值、总体标准差g∈R^768，加上每个token共享192→1投影的320个有序值。所有三组使用相同输入，不在本轮重选空间接口、历史长度或投影宽度。

每折只用19个fit源组，先视频内平均，再对源组等权平均，计算：

- μ：六层真实IoU均值，作为共同固定先验。
- F̄：每个token位置的特征均值。
- ḡ：全局统计g的均值；不能用g(F̄)替代，尤其std不是线性运算。

用可训练投影w和读出W构造：

`rθ(F,g) = W [g−ḡ ; (F−F̄)w]`

对任意当前w、W，在精确运算下，fit源组等权期望E_fit[rθ]=0。因此固定层偏好不能通过另一个可训练常数被重新引入，且训练时无需每步重跑全部fit特征。F̄、ḡ、μ在训练和评价期间冻结，不包含held统计。

中心化会消去投影偏置和六个输出偏置，故三组统一不设这7个冗余参数：192＋1088×6＝6720个训练参数。Stage-197的6727参数不作为本轮同容量对照；本轮三组均为6720，不用无效参数凑数量。fit统计属于额外的训练集统计信息，三组和固定μ控制获得完全相同的信息。

w以首通道单位向量初始化，W全零。三组参数与固定buffer初始哈希相同，初始决策都为固定μ最优层。Sigmoid重构μ会有float32舍入误差，单独报告，不要求分数逐bit相同。

## 三组匹配对照

令P=I−11ᵀ/6；y_t为同一原生父框、六个独立L7–L12块经原CENTER＋Hann解码的真实IoU。

| 组别 | 决策分数 | 训练目标 |
| --- | --- | --- |
| absolute_sigmoid | sigmoid(logit(μ)+rθ) | 分数对y的绝对MSE |
| absolute_additive | μ+rθ | 分数对y的绝对MSE |
| interaction_residual | μ+P rθ | mean((P rθ−P(y−μ))²) |

absolute_sigmoid是使用相同先验和中心化接口的现有Sigmoid回归控制，不宣称严格复现Stage-197的未中心化模型。absolute_additive用于分开输出参数化和训练目标：前两组只改变输出形式；后两组移除与选层无关的共同难度拟合。同一组参数下，μ+rθ与μ+P rθ的argmax相同，直接差异来自训练过程而不是多一个在线门槛。

残差目标P(y−μ)带正负号，不再要求用[0,1]输出直接表示它。所有组使用同一目标IoU缓存；加性分数仅用于排序，不截断、不宣称校准后的绝对概率。loss差异同时改变梯度和裁剪，故报告梯度历史，不将所有效果归因某一个优化因素。

与Stage-196的区别：本轮同时固定列均值先验、对fit输入中心化，并有同输出形式的absolute_additive控制；196在原Sigmoid结构上只去掉行共同误差。与Stage-198的区别：本轮训练新的残差函数，198只校准已有分数。所有新增设定都需由本轮匹配对照检验，不能借用先前排序信号宣称成立。

## 数据、预算、来源与冻结边界

使用Stage-194的38历史开发源组、两折19 fit／19 held、2132／2133两个种子，每源组48个已采样状态。三组共12次CPU训练，每模型256步×4例，AdamW学习率1e-5、weight decay1e-4、梯度上限1，沿用原draw，合计3072更新、21888次缓存预测。

来源只需要完整Stage-194目录与已归档Stage-198结果包。Stage-194代码、特征、六层标签、receipt和scope按原指纹／SHA验证；全部标签另与Stage-198逐状态真值完全对齐，fit μ与其先验复核。Stage-198原包SHA为 `0f3bcebd12370081e9e8a2319bb12458a537a1cdffe095491c422f2a2024ae77`，决策为 `447f9394ced978f2c3586b28b52ca89327d1c2cb05533b7990147f4732d93e34`。不依赖本机Stage-198运行目录，归档包已含所需证据。

不读图像、不加载原SGLA checkpoint、不跑骨干／CENTER或完整轨迹，不打开确认源组。新模型、统计NPZ、预测、优化器状态、receipt和结果均独立存储，已提交产物复用；未提交的单模型中断后从头训练。代码／版本变化会使manifest不同，须使用新输出目录，不能修改旧manifest绕过绑定。

## 评价和接续条件

报告fit／held的绝对与相对误差、目标交互能量、交互MSE及相对零残差skill、差值加权排序、IoU／regret／层分布，以及视频内状态对应增量。无状态对应期望仍为Σ_lπ_lȳ_l，held标签只参与评价，不改变决策。

这里交互MSE与相对误差MSE在代数上相等；另列交互能量及skill，是为了明确零残差基准为固定fit μ，不是声称新增一种独立误差指标。两个种子改变数据抽样，模型初始化本身是确定性的。

保留trained、固定prior_only、fit预测均值常数、L9、oracle五种选择。统一沿用数值并列规则：L9距最大值≤1e-8时保留L9，否则取第一个精确最大。训练／评价全程检查fit buffer不变、同初始化／同参数／同draw、投影梯度非零、重算g和预聚合路径一致、fit raw读出均值误差≤1e-5。这个1e-5仅为float32数值一致性容差，不是采用门槛；Sigmoid组的fit分数均值偏移另行报告，因为非线性不保证E[sigmoid]=sigmoid(E)。

共同主要判断均限held、interaction_residual：

1. 对prior_only的实际选框平均IoU增量。
2. 对自身同频率无对应期望的增量。
3. 对absolute_additive的实际平均IoU增量。

先源组内平均种子，再按原38源组bootstrap，报告双种子、最小留一源组和支持数。所有区间是历史开发集的探索性名义区间，不能充当独立外部确认。仅训练误差下降或动作更多变化不算通过。

若同时有正方向及区间支持、两种子一致、最小留一仍正，才考虑下一阶段完整自身轨迹；还需验证稳定损伤、持续失跟、恢复和开销。即使本轮固定L7即时较好，也不能忽略Stage-194其全视频退化的证据。若无held交互增量，记录该参数化未解决问题，不追加残差缩放、训练步或门槛搜索。

## 服务器运行

同步新代码、协议及已归档Stage-198结果包后，使用原环境运行，无需GPU：

```bash
conda activate sglatrack-test
cd ~/workspace/paper
bash tools/run_stage199_residual_routing.sh
```

后台运行：

```bash
conda activate sglatrack-test
cd ~/workspace/paper
export STAGE199_ROOT="$PWD/output/stage199/fit_centered_interaction_v1"
mkdir -p "$STAGE199_ROOT"
nohup bash tools/run_stage199_residual_routing.sh > "$STAGE199_ROOT/run.log" 2>&1 < /dev/null &
echo $! > "$STAGE199_ROOT/run.pid"
tail -f "$STAGE199_ROOT/run.log"
```

默认Stage-194来源为`output/stage194/module_ablation_v1`，可设置STAGE194_ROOT；归档包默认`docs/experiments/06_sgla_uav_tracking/results/stage198_results_447f9394ced9.txt`，可设置STAGE198_BUNDLE。无需服务器补跑Stage-198。

只显示一个总体进度条，后台定时刷新及最终100%。完成自动生成`stage199_results_<UTC>_<PID>.txt`，包含manifest、decision和10表，回传此文件即可。单独核验：

```bash
python -B tools/stage199_residual_routing.py --output-dir "$STAGE199_ROOT" --check-results
```

## 本地验证边界

6项CPU测试通过：真实Stage-198归档包身份读取、缓存及来源篡改拒绝；三组同初始状态／决策／6720参数；任意权重下fit读出中心化和held统计隔离；目标投影恒等式、同参数加性与投影分数的决策等价及共同梯度为零；三组256步训练／buffer冻结／断点复用及模型篡改拒绝；38组合成缓存完整12模型／3072步、10表与三项主比较、源文件不变、禁止CUDA初始化及结果复用。添加加性分数均值约束后，端到端用例再次通过。

这些测试不证明真实路由有效。本机缺少Stage-194原始特征NPZ，尚未运行真实Stage-199训练；Stage-198回传数据只能验证来源和监督，不能生成L6特征。
