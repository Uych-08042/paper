# Stage-206：原生路由固定的空间目标条件适配

固定日期2026-11-26。目标状态`REVIEW_NATIVE_ROUTED_SPATIAL_ADAPTATION`。38历史开发源组、两fold各19 fit／19 held、seed2132／2133；确认集封存。

## 假设与改动范围

205收缩正确关系后IoU接近固定先验，但3648动作只有33次非L7，未建立动态收益。本轮停止调分数幅度，直接检验可学习的逐token目标—搜索交互是否比一般适配更有价值。它不是205已证明的病因，也不保证改动有效。区别于194模板全局均值生成scale／shift、164固定局部匹配：本轮在L6边界学习空间参考与各搜索token的对应。

SGLA原骨干本来就联合处理模板和搜索；本轮不称首次引入目标条件，不用PPO、多搜索或新动态路由。首帧模板、单搜索裁剪、原候选块、CENTER几何／响应、Hann及状态反馈都沿原生；原生权重及BN buffer始终冻结，仅新增适配权重可训练。

## 空间模块与三种输入

F6=[Z,X]，Z为64×192，X为256×192。令LN为无可学习仿射的逐token LayerNorm，四个无bias、全宽线性变换Q、K、V、O均192×192，单头注意力：

`A = softmax(Q(LN(X)) K(LN(R))ᵀ / sqrt(192))`

`X' = X + O(A V(LN(R)))`，`F6'=[Z,X']`。

维度完全来自原骨干，不引入任意瓶颈宽度／头数／残差倍率。每组147456可训练参数。相同seed的Q/K/V初始化相同，O严格零初始化，初始F6'=F6；首步主要更新O，随后须观察四个路径均有非零梯度。没有新增位置编码，使用原L6已有的空间token信息。

| 组 | 参考R | 用途 |
| --- | --- | --- |
| search_control | 16×16搜索网格2×2平均池化为8×8＝64 token | 同参数当前帧适配控制 |
| matched_spatial | 本帧联合前缀的64个模板token | 正确空间目标条件 |
| mismatched_spatial | 来自另一fit源组的64个参考token | 错配关系控制 |

三组参考数、注意力矩阵256×64、参数和训练样本匹配。搜索池化固定来自已存在的2倍空间分辨率差，不调采样格数。错配参考对每个fit源组，把该组48缓存状态的模板token按对应空间位置求均值；fit按排序循环取下一组，held按SHA256(源组名)模19选fit参考，保证不同于自身。仅新增适配器参考错配，原SGLA实际模板不换，不能理解为原跟踪器失去目标模板。held当前／未来图像、标签不参与参考统计。每fold保存参考NPZ／receipt及76行映射审计。

## 原生路由隔离

在L6输出hook中保存未适配F6的channel0，返回F6'。原MLP的forward-pre-hook用保存的未适配输入替代其参数；MLP正常执行一次，不修改分数／权重、不增加第二次MLP调用。候选块使用F6'。每帧核对前缀与路由hook次数、原MLP输出所选层和实际候选层；实际块／头／裁剪计数沿既有审计。

这是**同一帧、同一自身搜索状态内**的原生选层决策保持，不要求整段适配轨迹与原生轨迹选层序列一致；输出反馈使后续裁剪及原生决策输入自然可能分叉。

训练也先在未适配F6上计算原生离散选层，无梯度经过选层，只对所选候选块的定位损失反传到适配器。不用六层真值择优，也不重新训练质量路由。batch中按实际所选层分组执行，每样本一个候选块及一个CENTER；实际batch调用数与样本数分别报告。

## 数据、预算与执行

输入为194原生父框缓存F6、target等几何和已核验205原样TXT；205 SHA256 `0f342f0b40ad5f877045b55c0e19f39255848978418031a63f09ac7961e8972d`，decision `074e680f182d8752a1796ffb7ece24458d17d0994f82c3235723a2e36655da7c`。核验194原manifest、代码、NPZ及receipt，并重用194既有数据源入口绑定38组与原图像标注；不新采训练状态，不打开确认集。

三组×两fold×两seed＝12个新适配器，256步／batch4／每模型1024训练状态引用；总3072更新、12288引用。源组等概率、组内有放回取状态，完全继承194原抽样，三组精确同日程。AdamW lr1e-5、wd1e-4、全适配器clip1。原定位目标2GIoU+5L1+focal，GT heatmap仅在训练forward_head时提供；适配器和原生选择器输入没有GT。12个模型参数、更新、所选候选样本匹配，不能称与旧194六层监督同计算预算，更不能以参数相等代替实测耗时。

全部训练后才固定模型做fit／held原生父框读出；每模型38×48状态，共21888读出。读出只执行当前原生选层，native_cached_mean_iou为194历史缓存参考，cached_native_layer_differences显式报告硬件数值等差异；它不替代新同卡完整原生对照。

正式完整视频全部新跑：38个原生＋3组×2seed×38＝266条OOF轨迹。每视频首帧初始化后连续自身预测反馈，不GT重置。原生与三适配策略在同卡／相同软件／checkpoint／sealed config上执行。首次允许L20或L40，记录GPU UUID及与194的差异；同一输出根续跑必须保持manifest中的运行环境，不把不同GPU结果混入同一次运行。

零初始化公共路径smoke先做一次原生和三适配，要求框误差≤.001、层相同；后续冻结原生哈希及各适配器哈希不变。非有限loss／梯度／参数／Adam状态写失败JSON并停止；普通中断只复用带receipt的完整小模型／读出／视频，未提交的小模型从头拟合，不声称逐优化步续训。保留训练history和所选层调用账本。

单一总体进度条包含4次公共smoke、3072训练步、21888固定状态读出和7倍全开发帧数；启动阶段另行核对源缓存。后台约10秒输出一次总体快照，不逐视频刷屏。

## 评价与解释边界

三项主要比较：matched_spatial分别对native、search_control、mismatched_spatial的完整视频AUC。先平均两seed再按38源组等权，沿用20000次源组bootstrap、seed24194、名义95%区间；另报稳定视频新增失败、原生恢复丢失、原生好帧损失、最长失败。报告两个seed，不能挑较好seed。scope／训练history／参考／smoke／fit-held读出／policy／paired／seed-video／routing九表，及模型、参考、readout和完整轨迹NPZ/receipt、sealed来源核验记录。

最终固定REVIEW，不自动宣称采用／SOTA／时序记忆或独立确认。若正确和错配／自交互一样好，归为一般适配；若fit改善而held或完整轨迹退化，保留过拟合／状态分布错配解释，不后验挑层或超参。稳定／恢复保护未过不能只凭AUC均值宣称成功。延迟含instrumentation，不是部署测速；无加深Transformer块的主张。

## 后台运行

在服务器`sglatrack-test`环境、同步本轮代码后：

```bash
cd ~/workspace/paper
mkdir -p output/stage206
nohup bash tools/run_stage206_spatial_adaptation.sh > output/stage206/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage206/run.log
```

默认输出`output/stage206/native_routed_spatial_adaptation_v1`，GPU逻辑号0，可设置STAGE206_GPU／STAGE206_ROOT／STAGE194_ROOT／STAGE205_BUNDLE／CHECKPOINT及原76／89／93／121／170／188／191根目录。支持`--check-results`只校验并重新打包。

完成自动生成`stage206_results_<UTC>_<PID>.txt`（decision、manifest、九表），回传该文件。此阶段需要GPU和原38组图像，不是205那样CPU-only；本机无真实194特征和服务器GPU，合成测试不代表正式精度复现。所有1～205代码与原协议保持冻结。
