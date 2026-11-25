# Stage-203 损失目标×梯度裁剪：匹配四组协议

日期：2026-11-26。真实训练前固定。Stage-202的定位遗憾监督未建立held收益，但1022／1024次更新范数超过继承裁剪阈值，MSE仅4次。该现象不能证明裁剪导致退化，本轮只检验这一优化交互，不追加损失、学习率或裁剪阈值搜索。

## 固定四组与科学问题

| arm | 训练目标 | 梯度处理 |
| --- | --- | --- |
| point_clip | 原层间point MSE | 原全局L2裁剪，阈值1 |
| point_no_clip | 同一个point MSE | 只计算范数，不写入／缩放梯度 |
| regret_clip | 原定位遗憾上界 | 原全局L2裁剪，阈值1 |
| regret_no_clip | 同一个定位遗憾上界 | 只计算范数，不写入／缩放梯度 |

两损失完整复用Stage-202实现：point为`mean(P(s−y)^2)`；regret为`mean(max_l[s_l−s_l*+y_l*−y_l])`，l*是当前精度标签的精确最大值。L9近最大值容差1e-8、标签／模型训练float32、决策上界核验float64均保持不变。未新增margin、温度或组合权重。

无裁剪分支通过各参数梯度范数的全局L2合成只读测量，绝不调用有限阈值裁剪，也不更改梯度。开启分支仍调用原`torch.nn.utils.clip_grad_norm_`，不替换其范数算法／epsilon或缩放顺序；额外测量只用于记录。AdamW和全部训练机会一致。

## 来源和不变条件

1. 完整Stage-194原始L6 token／六层标签NPZ与receipt目录，原决策`aecfc2112422a0cf5f1af3daa1ddc61bcce9f54fc4c3f2ae94f762429a9dd547`、原计划`69ce5be3cf224c940d203698d07049760a4018e9928dee31270babdd9f8a27e0`。
2. 已审202原字节包`stage202_results_20261125T154651Z_1417349.txt`，SHA256 `8f0b46df691d9c9a1027dc96af56bf699431f470b58027451742f536371f3562`，决策`5048f7fe1c427f22b2f3e8c61533abe8b5cfeb70d362536a6e96acaa976602a4`，计划`715da640896f6884dc930515174d62815d839d0520d7c9c9b48ee188a7e47616`。
3. 202指纹／manifest／13表／冻结代码和协议全部核验；194每个NPZ及receipt按原清单核验，逐状态标签与fit均值先验对齐202。不加载原202 PT作为新模型，不用旧结果替代同次四组训练。
4. 同38历史开发源组、19／19两折、2132／2133；每视频48个原生父框缓存状态。原6720参数、1088输入的μ+P r中心化残差架构，同初始化／固定buffer／fit统计。输入、骨干、CENTER和选框规则不变。
5. 同Stage-200/202状态对日程：同视频两个状态独立均匀有放回，每步2对／4个状态引用，自配对保留。四组共享同一抽样；优化仍是逐状态目标。
6. AdamW lr1e-5、weight_decay1e-4、256步／512对／1024引用，每模型同预算。唯一因素是目标种类与裁剪开关，所有组同次重训。

总计16模型、4096优化更新、16384训练状态引用、8192模型配对引用；独立日程仍4套、2048行，不把四组共用的抽样算成更多独立样本。评价16×1824=29184模型状态。只做CPU小模型训练／预测，无新图像、SGLA骨干、CENTER、完整轨迹或确认源组。

## 数值异常与恢复

逐步记录两个损失、优化损失、投影梯度、全局梯度范数（处理前／后）、实际缩放系数、是否实际裁剪、是否超过继承阈值、参数更新L2及相对更新L2。未裁剪组的实际缩放系数必须始终为1，实际裁剪次数为0；可同时报告它有多少步本来会超过阈值。

loss／梯度／参数／Adam状态／参数范数或更新量出现非有限值，封存`diagnostics/nonfinite_failure.json`（组、fold、seed、step、阶段、计划身份），立即停止，不生成完整成功决策。该输出路径不允许自动重试或继续其他组，不临时改阈值／学习率。需要复核异常后才能另行决定新实验。

普通中断的完整模型／预测缓存可按原元数据和哈希复用；未知缓存、篡改或非有限失败标记均拒绝复用。原1～202代码／协议／证据不修改。

## 预设比较与解释

所有评价按同源组先平均两种子、38源组等权；按原bootstrap seed24194、20000次计算名义95%区间。fit与held分开，不挑种子或失败视频。已有开发暴露意味着这些区间不是独立外部确认；多个比较未声称已控制全族误差。

主要诊断为held即时IoU上的：

- **裁剪效应**：regret_no_clip − regret_clip。
- **无裁剪下的目标效应**：regret_no_clip − point_no_clip。
- **交互**：`(regret_no_clip−regret_clip)−(point_no_clip−point_clip)`。

同时固定两项实用比较：regret_no_clip − prior_only，以及regret_no_clip − 自身状态无对应期望。五项均列在decision，不因其中一项为正就采用策略，也不要求诊断交互必为正才承认其他实用收益。状态无对应期望保留各视频自己的动作频率、打散动作与状态对应，只用于离线评价。

支持分析包括point裁剪效应、开启裁剪时的目标效应、相应状态对应增量、误差／排序／定位遗憾上界、实际更新尺度、源组／双seed／留一分析。需区分“取消裁剪改变了优化”与“取消裁剪改善held决策”；频繁裁剪和较小fit损失都不是方法采用条件。

同次开启裁剪的两个控制另与202各自原模型比对最终状态哈希、分数和动作差异。该历史复现表是审计；若运行库变化造成差异，应如实记录，仍以本轮同次四组作因子比较，不能隐去漂移或假称逐bit复现。

如果取消裁剪仅改善fit而held不升，降低裁剪假设优先级，转查表征／数据支持，不继续扫描阈值。若出现跨源组和seed一致的held增量，才另行预注册自身轨迹验证。当前缓存即时IoU不是全视频AUC，平均选层编号不是实际Transformer执行块数或速度。

## 产物与运行

14表：model_scope16、fit_summary2、source_bridge38、pair_schedule2048、pair_support76、error_summary24、error_video1824、policy_summary40、policy_video3040、paired92、frame_predictions29184、training_history4096、seed_summary80、control_replay8。

decision／manifest封存202来源、全部代码／协议、表与训练／预测产物SHA。预计1252条统计／模型／预测及receipt物理文件引用。单一总体进度条，真实完整任务35144工作单元，重定向日志定时刷新。完成打包decision／manifest／14表为单一文本，不用逐张粘贴。

本机没有真实194特征。服务器在`sglatrack-test`、`~/workspace/paper`运行，无需GPU和SGLA checkpoint：

```bash
bash tools/run_stage203_loss_clipping.sh
```

后台运行：

```bash
mkdir -p output/stage203
nohup bash tools/run_stage203_loss_clipping.sh > output/stage203/run.log 2>&1 < /dev/null &
echo $! > output/stage203/run.pid
tail -f output/stage203/run.log
```

默认来源194=`output/stage194/module_ablation_v1`，202包为仓库内已审原包，输出=`output/stage203/loss_clipping_factorial_v1`。可用STAGE194_ROOT／STAGE202_BUNDLE指定相同来源的新路径，用STAGE203_ROOT指定独立输出；不能复用旧实验目录。

```bash
python -B tools/test_stage203_loss_clipping.py
python -B tools/stage203_loss_clipping.py --output-dir "${STAGE203_ROOT:-output/stage203/loss_clipping_factorial_v1}" --check-results
```

完成后回传`stage203_results_<UTC>_<PID>.txt`；若停止于数值异常，回传失败JSON和日志，不将缺失组按成功组平均。
