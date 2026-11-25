# Stage-197 空间顺序与同参数无序读取对照

日期：2026-11-25。本文保留原运行协议；服务器实验已完成，结果见[Stage-197复核](STAGE197_RESULTS_REVIEW.md)。

## 依据与待证问题

[Stage-196](STAGE196_RESULTS_REVIEW.md)中相对损失改善训练误差，但held选框IoU、相对误差与排序均未改善。下一步检查输入接口是否支持任务相关的层间区别，不继续按结果修改损失权重。

原生SGLA的选择器读取每个token的首通道 `x[:,:,0]`，保留320个token的顺序。Stage-194新增R则把模板与搜索分别压缩为逐通道均值、标准差，形成768维输入。此读取对各自token的排列不敏感，无法直接利用显式网格地址。token已经含有位置编码与上下文交互，因此不能称其完全没有空间信息，更不能据此认定骨干缺少空间表示是失跟主因。

待证假设：在相同训练机会下，保留投影token的原排列，能否比只保留同样投影值集合的读取，更好地预测六分支质量并选择分支？

## 模型与同参数对照

令L6缓存为F∈R^(N×D)，模板与搜索各占N_z、N_x个token，实际N=320、D=192。全局输入仍为g=[μ_z,σ_z,μ_x,σ_x]∈R^(4D)。另设所有token共用的可训练线性投影s_j=wᵀF_j+b，每token得到一个数。w初始化为首通道单位向量、b为0，直接衔接原生选择器的输入形式，不增加任意隐藏宽度。

| 组别 | 路由输入 | 训练参数 |
| --- | --- | ---: |
| ordered | g与按原token顺序的s拼接 | 6727 |
| unordered | g与每样本内分别排序的模板s、搜索s拼接 | 6727 |
| global_stats | 只有g；重新训练的Stage-196 absolute结构 | 4614 |

主比较两组均使用1088→6线性读出加Sigmoid，读出权重、偏置零初始化；共享投影193参数，读出6534参数，总6727。两组初始参数哈希必须相同，之后独立学习。global_stats为768→6，不增加投影，是次要原接口控制；它与主比较的容量不同，不能用单独超过它来证明空间顺序有效。

unordered对每个样本的64个模板值、256个搜索值分别升序排序，保留两个值集合及其归属，删除投影值对应的原网格地址。同一固定全局置换是可逆重编号，不能作为无序控制。排序对投影值分段可微，两组都需要实际梯度到达投影。排序同时改变特征坐标的学习条件，因此结论限于该读取方式及优化预算，不是完整跟踪因果归因。

两组都保留完全相同的g。比较的是显式布局相对数值集合的增量，而不是全局统计之外增加特征本身。输入审计验证逐样本流内反序后unordered不变，投影数值集合未变，ordered存在变化。单元测试另使用逐样本不同置换、非初始投影和非零读出检验此性质。

## 固定训练与评价

复用Stage-194的38个历史开发源组、两折19 fit／19 held、种子2132／2133、每组48个均匀状态。全部状态已有原生父框L6缓存，以及L7–L12六个独立块经原CENTER＋Hann解码的真实IoU标签。

三组共12次CPU训练，每模型256步×4例，共3072更新。保持Stage-194的AdamW、学习率1e-5、weight decay=1e-4、梯度上限1、absolute MSE。各折各种子三组的draw哈希必须一致并与Stage-194相同；ordered／unordered额外核验初始状态、参数量、输入维数一致。held标签只评价，不选训练步、超参数或投影。

每模型评价全部1824缓存状态，总21888输出。训练使用预先计算的g，评价逐状态检验与直接从token重算g的路径一致。原生SGLA权重、骨干、CENTER均不加载，不读图像、不采集新特征、不跑新完整轨迹，28个确认源组继续封存。

三个组均报告：

- fit／held／实际fit_draws／fit_seen_unique／fit_unseen的共同、相对与绝对误差。
- 差值加权成对准确率、regret、所选IoU、选择层分布。
- 固定L9、fit标签均值、fit抽样标签均值、fit预测均值、真值上界、均匀期望控制。
- 每步误差与梯度，包括投影梯度；完整逐帧六层标签、分数、选择及fit抽样次数。
- 新global_stats与Stage-196 absolute的概率及动作差异，单独报告环境／数值复现，不用旧CPU结果替代本轮重训控制。

主比较是held `ordered_minus_unordered` 的平均IoU。先源组内平均两种子，再对38源组bootstrap；同时报告held排序及相对误差。固定控制只用fit确定。所有区间为历史开发集上的探索性名义区间，未作多重比较校正，不能称独立外部验证或新AUC。

## 解释与接续标准

仅训练内误差改善不算支持。优先要求held主比较均值与区间为正、两种子方向一致、最小留一源组差仍正，并且ordered优于自身fit预测均值与fit标签均值固定控制，排序指标也支持改善。即便满足这些条件，也只能进入下一轮完整自身轨迹验证，还需评价稳定视频损伤、持续失跟、恢复及开销。

如果ordered与unordered同样优于global_stats，只支持新增读出整体有效，不能独立归因空间顺序。若只超过受损路由、没有超过固定控制，仍不能进入采用验证。若两组均未获留出增量，记录当前接口／监督／预算下假设未获支持，不据此预设更大骨干或历史记忆必然有效。

Stage-164已检验4×4局部余弦匹配16维特征，未获稳健收益。本轮没有增加相似度图、局部匹配网格、历史窗口或换层门槛，核心变量是投影值的显式排列，不能把它包装为首次验证空间信息。

## 来源与产物保护

完整保留Stage-194／195／196目录。沿用三阶段已复核指纹，Stage-196决策为 `b2769871bf752da66efd6d5a6bb1f2f211affd3e18b0105113e27429eff9bc91`。核验原代码、表格、特征、标签与receipt哈希；本轮输出目录须独立。旧代码及模型不修改。

本轮manifest记录新代码、Python、NumPy、Torch版本。模型和预测有独立receipt及哈希，已提交产物复用，身份不同拒绝覆盖；中断时未提交的单模型从头训练。只有相同元数据的未提交孤立文件可被重新发布。最终decision包含10张表及全部新模型／NPZ的哈希。`--check-results`可独立核验已完成产物。

## 服务器运行

同步新代码到服务器后，在有真实缓存的仓库运行，无需GPU和原SGLA checkpoint。

```bash
conda activate sglatrack-test
cd ~/workspace/paper
bash tools/run_stage197_spatial_router.sh
```

后台运行：

```bash
conda activate sglatrack-test
cd ~/workspace/paper
export STAGE197_ROOT="$PWD/output/stage197/spatial_routing_interface_v1"
mkdir -p "$STAGE197_ROOT"
nohup bash tools/run_stage197_spatial_router.sh > "$STAGE197_ROOT/run.log" 2>&1 < /dev/null &
echo $! > "$STAGE197_ROOT/run.pid"
tail -f "$STAGE197_ROOT/run.log"
```

默认来源为`output/stage194/module_ablation_v1`、`output/stage195/router_objective_audit_v1`、`output/stage196/relative_loss_comparison_v1`；非默认路径分别设置STAGE194_ROOT、STAGE195_ROOT、STAGE196_ROOT。

只显示一个总体进度条，后台日志约每10秒刷新，完成时显示100%。退出`tail -f`不停止nohup任务。

完成自动生成`$STAGE197_ROOT/stage197_results_<UTC>_<PID>.txt`，包含decision和10表，回传此文件即可。

```bash
python -B tools/stage197_spatial_router.py --output-dir "$STAGE197_ROOT" --check-results
```

## 本地验证边界

本地7项CPU测试覆盖：投影值集合及流内排列不变性、同初始化及6727参数、投影梯度、global控制与原函数及256步训练一致、模型中断复用与篡改拒绝、缓存及目录保护，以及38组合成缓存的12模型端到端训练、全部表、哈希、完成复用和禁止CUDA初始化。合成测试不代表真实选层收益。本机没有真实来源NPZ及模型，服务器21888条真实缓存评价现已回传复核，见结果文档。
