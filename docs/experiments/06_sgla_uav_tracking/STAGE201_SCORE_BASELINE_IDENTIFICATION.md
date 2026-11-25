# Stage-201 分数基准可识别性诊断：运行前协议

日期：2026-11-25。Stage-200真实结果复核后、Stage-201评价前固定。只消费已审Stage-200文本包，不增加训练、模型推理、图像读取、完整轨迹或确认集。

## 问题与理论依据

同视频差分损失对逐视频时间恒定的六层偏置向量不变。固定fit总体均值并不能识别每个新视频的分数基准。Stage-200差分held状态对应增量为正，但总选择低于固定先验；尚不能认定偏移是唯一根因。

在不拟合新参数的情况下，检验移除视频平均预测偏移是否改善选择，以及过去信息能否近似这一离线参考。真实的视频特有层偏好也可能有用，故不预设去均值必然改善。

## 来源、精度与处理规则

- 原始包：`stage200_results_20261125T140231Z_1329509.txt`，SHA256 `00a487b9552785b6a4eb9f2a64a7477591e4ddbed32b5d348df362026a8d8838`。
- 决策`b9b4aafb6a648a38d54347e76dd1e2138f8ad9b188cfbea51c962bb59b30f2dc`；计划`8792d5489f5d1711191ea202e4c2d550bbc310fa17f72615a308fd68f437522f`。核验包、指纹、12表、冻结代码／原协议。
- 38历史开发源组、原两折19／19、种子2132／2133、point与within_video_difference两组，8个原模型，每视频48缓存状态，共14592模型状态。原数据是原生父框下抽样状态，非策略自身连续轨迹。
- 每视频按已归档帧号严格递增，不额外取失败点、不重排、不丢第一点。每层原分数先按float32恢复，再升为float64计算校正。标签只用于fit先验来源复核和离线评价，不参与新视频校正。
- `μ`是原fold的fit标签均值先验，按源组等权。本轮无新统计量用held标签拟合。校正分数不做Sigmoid、裁剪或缩放。
- 所有新选择沿用既有规则：L9距最大值不超过1e-8时保留L9，否则第一个最大值。该数值容差不是新性能门槛。原raw选择须逐状态精确复现。

## 固定六策略

设视频预测序列为`p_0,…,p_(n−1)`，真实帧号可不连续。

1. `raw`：保持原`p_t`及其选择。
2. `offline_video_mean`：`s_t = μ + p_t − mean_(j=0..n−1) p_j`。包括当前和未来预测，离线诊断参考，不能部署，也不是性能上界。
3. `causal_past_mean`：t=0时`s_0=μ`；t≥1时`s_t = μ + p_t − mean_(j<t) p_j`。先决定当前校正，再将当前原分数纳入历史。历史每视频／模型独立重置；无滑动窗口、衰减、预热期或参数选择。第一状态参与全部评价。
4. `prior_only`：常数μ。
5. `native_L9`：常数L9。
6. `oracle`：离线真值最优层，只作能力参考。

因果校正的当前分数来自当前状态，但基准估计只能使用更早状态。即使满足这个约束，离线缓存状态本身仍依赖原生轨迹而不是校正后的自身轨迹，不能宣称完整在线可行性。

## 统计和主要比较

同时报告fit与held，不跨pool合并。先种子内逐视频平均，再两种子等权、38源组等权。对照间先在同视频内求种子均差，再按源组做原协议固定seed的20000次bootstrap。种子单独列出，不能挑种子。历史开发集的名义95%区间不作独立外部确认，多个比较不作为已控制全族误差的发现。

`I=mean_t y[t,a_t]`，`E=Σ_l freq(a=l)·mean_t y[t,l]`，状态对应增量为`I−E`。每个策略使用自身的动作频率；另列`I−prior=(E−prior)+(I−E)`。

主要因果比较固定为within_video_difference／held的：

- causal_past_mean − prior_only，平均IoU；
- causal_past_mean − raw，平均IoU；
- causal_past_mean − independent_expectation，平均IoU。

离线诊断重点是offline_video_mean对raw、prior及其自身无对应期望；因果与离线参考差距、point同规则结果、两组间差和预测误差分解为支持分析。不得因离线参考效果好而称因果成功。

## 可核验产物与预算

9表：source_audit、calibration_video、policy_video、policy_summary、seed_summary、paired、error_video、error_summary、frame_predictions。逐状态表记录三种分数、标签、各策略动作／IoU、过去样本数、过去基准最大帧号、离线基准最大帧号；校正不访问真实标签。逐视频表列分数均值偏移、过去基准对离线均值的误差及因果范围检查。误差表采用视频平均偏好／视频内变化分解。

预计8 source_audit、304 calibration_video、1824 policy_video、24 policy_summary、48 seed_summary、38 paired、1216 error_video、16 error_summary、14592 frame_predictions。0新训练／参数拟合／模型前向／编码器／图像／完整轨迹／确认源组。单一总体进度条。决策与manifest封存代码、协议、源包及输出表哈希；完整结果可导出为单一文本包。

## 解释和后续

离线参考也不改善，则降低“单独校正视频均值足以解决问题”的优先级；仅未来参考改善则仍缺可用因果估计；因果规则稳定改善且保留状态对应收益后，才能另行制定自身历史闭环验证。离线参考不是保证改进的上界；失败也不能排除其他可识别建模或估计方式。

Stage-194固定L7曾损伤完整轨迹，本轮不能因缓存先验更好就采用固定L7。确认集继续封存，旧Stage-1～200代码／协议／产物不修改。

## 复现

```bash
python -B tools/test_stage201_score_baseline.py
bash tools/run_stage201_score_baseline.sh
```

也可直接CPU执行：

```bash
python -u -B tools/stage201_score_baseline.py --output-dir output/stage201/score_baseline_v1
python -B tools/stage201_score_baseline.py --output-dir output/stage201/score_baseline_v1 --check-results
```

默认来源是仓库内已审200包；`--source-bundle`仅允许相同原字节包，`STAGE201_ROOT`可指定独立输出。已有封存决策时仅核验复用，不覆盖。
