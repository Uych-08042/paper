# Stage-202 定位遗憾上界监督：匹配训练协议

日期：2026-11-25。真实训练前固定；状态为待服务器真实缓存训练。Stage-201两种分数均值校正未建立收益，回到选层模块的学习目标。本轮只改变监督目标，不增加输入、参数、训练次数或校正规则。

## 科学问题与损失推导

目前优化平均层分数误差的路由，在held上仍未超过固定fit先验；Stage-200出现小幅时机信号，但Stage-201移除视频均值没有解决整体选择问题。需检验直接对齐定位选择的训练是否优于拟合六层分数。不是先认定MSE为唯一病因，也不将标准代价敏感最大间隔损失本身当作新理论创新。

六层真实即时IoU为y，预测分数为s，`l*=argmax_l y_l`，真实定位遗憾为`Δ_l=y_l*−y_l`。两训练组固定为：

- `point`：`mean(P(s−y)^2)`，P去掉六层共同分量；与Stage-200点监督相同，必须本轮匹配重训。
- `regret_margin`：`mean_t max_l[s_l−s_l*+y_l*−y_l]`，与`max_l(s_l+Δ_l)−s_l*`等价。先做差再max，使真值最优项逐元素严格为0，损失非负。

若a是精确分数argmax，`s_a≥s_l*`，所以`L≥s_a−s_l*+Δ_a≥Δ_a`。既有选择规则允许L9在距最大值1e-8内保留，则`Δ_a≤L+1e-8`。1e-8是继承的数值容差，不是新性能阈值。`s=y+c`时损失为0，对逐帧六层共同平移不变；训练最优层采用当前float32标签的精确argmax，同值取首个。所有最优标签的定位遗憾均为0。

间隔直接由IoU差决定，没有温度、手调margin、负样本阈值或组合权重。结构化max只处理已有六个标量，不是增加骨干计算。两损失的梯度尺度不同，沿用相同优化器／裁剪；如结果变化，属于该固定优化规则下的训练目标对照，不自动证明唯一因果病因。

## 固定来源与预算

- 原始Stage-194特征／六层标签／receipt完整目录，决策`aecfc2112422a0cf5f1af3daa1ddc61bcce9f54fc4c3f2ae94f762429a9dd547`，计划`69ce5be3cf224c940d203698d07049760a4018e9928dee31270babdd9f8a27e0`。
- 已审Stage-200文本包`stage200_results_20261125T140231Z_1329509.txt`，SHA256 `00a487b9552785b6a4eb9f2a64a7477591e4ddbed32b5d348df362026a8d8838`。决策`b9b4aafb6a648a38d54347e76dd1e2138f8ad9b188cfbea51c962bb59b30f2dc`，计划`8792d5489f5d1711191ea202e4c2d550bbc310fa17f72615a308fd68f437522f`。
- 200包、指纹、12表、冻结代码和原协议验证；194每个NPZ／receipt按原清单验证。全部逐状态标签和fit先验须与200相符，失败即停止。
- 38个历史开发源组、原19／19两折、种子2132／2133、每视频48个原生父框状态。fit与held按源组分离。没有重新按失败视频选样。
- 原Stage-199 `interaction_residual`架构：μ+P r，ordered token投影及pooled输入，fit中心化统计与固定先验，两组同初始化／buffer。真实320×192 token输入时参数6720、读出输入1088。参数和结构均不变。
- 沿用Stage-200日程：同视频两个独立均匀有放回状态，每更新两对，四个状态引用，自配对保留。原seed+100×fold抽样生成器；两组共用同样日程。两目标均按状态监督，不再优化同视频差分，自配对只是重复状态引用。
- 原AdamW lr1e-5、weight_decay1e-4、gradient_clip1；CPU单线程，256更新／模型、512状态对／1024引用。共8模型2048更新、8192状态引用、4096模型配对引用、4套抽样2048行。
- 8×1824=14592评价输出；只在CPU训练和评价路由小模型。不读图像、不运行SGLA骨干／CENTER、不新增完整轨迹、不打开确认源组。旧1～201代码／协议／证据冻结。

## 评价与解释边界

两组均报trained、prior_only、fit_prediction_mean、native_L9、oracle。主比较为regret_margin／held的：

1. trained − 本轮point，mean IoU；
2. trained − prior_only，mean IoU；
3. trained − 自身状态无对应期望，mean IoU。

无对应期望保留该模型逐视频各层选择比例，等于动作频率乘真实各层视频均值，且只用于评价。按源组等权、先在同源组平均两个种子，再使用原bootstrap seed24194／20000次给名义95%区间。历史开发源组并非独立外部确认；三共同主比较不声称已做全族校正，不允许挑一个通过就宣告采用。

支持指标含层间MSE／视频内误差、排序、遗憾上界、实际遗憾、上界违反量、双seed／源组／留一分析、层选择分布。逐状态遗憾和上界用float64原标签／导出分数重算；训练使用float32，不能要求两精度每一数值逐bit相同。

`mean_selected_layer`只是层编号的平均值，**不是执行Transformer块数**。本轮骨干调用为0，不能以层号估计实际延迟或动态加深收益；真实块数、耗时及自身历史跟踪需后续正常运行测量。cached native-parent即时IoU也不是完整视频AUC。固定L7曾在194闭环受损，不能据此采用。

## 审计产物与复用

13表：model_scope 8、fit_summary 2、source_bridge 38、pair_schedule 2048、pair_support 76、error_summary 12、error_video 912、policy_summary 20、policy_video 1520、paired 40、frame_predictions 14592、training_history 2048、seed_summary 40。

scope记录参数、初始／最终状态哈希、固定buffer、抽样哈希、梯度／投影梯度与接口一致性；history同时报两种目标与实际优化目标。误差／逐状态表附上界、实际遗憾和违反量；policy表附层分布／平均层号及骨干成本未测量标志。decision／manifest封存来源、所有表和模型／预测缓存SHA。

已有封存模型／预测只能按元数据和文件哈希复用，不覆盖；统计／模型更新中断可恢复，完成决策存在时只做核验。单一总体进度条，重定向日志定时刷新。完成后打包decision／manifest／13表为单一文本。

若主要条件和源组／双seed一致性不成立，不追加步数、缩放或后验阈值；若成立，才另行预注册完整自身轨迹验证。真实结果返回后分析，并提出下一步建议。

## 服务器运行

本机没有真实194 NPZ；本地合成测试通过不代表真实训练结果。服务器`sglatrack-test`、`~/workspace/paper`执行，无需GPU和原SGLA checkpoint：

```bash
bash tools/run_stage202_regret_routing.sh
```

后台运行，断开连接后可继续：

```bash
mkdir -p output/stage202
nohup bash tools/run_stage202_regret_routing.sh > output/stage202/run.log 2>&1 < /dev/null &
echo $! > output/stage202/run.pid
tail -f output/stage202/run.log
```

默认194来源`output/stage194/module_ablation_v1`，默认输出`output/stage202/matched_regret_margin_v1`。若来源目录不同，启动前设置`STAGE194_ROOT`；同字节200包可用`STAGE200_BUNDLE`指定路径；独立输出可用`STAGE202_ROOT`。不要复用旧实验输出路径。

```bash
python -B tools/test_stage202_regret_routing.py
python -B tools/stage202_regret_routing.py --output-dir "${STAGE202_ROOT:-output/stage202/matched_regret_margin_v1}" --check-results
```

完成后回传脚本输出的`stage202_results_<UTC>_<PID>.txt`即可，不必逐张粘表。
