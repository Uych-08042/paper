# Stage-198 固定层偏好校准与状态对应诊断

日期：2026-11-25。以下规则在本轮真实分析前固定。来源为已复核Stage-197全部回传分数与标签，实验不需要服务器原PT／特征NPZ。

## 依据与问题

[Stage-197](STAGE197_RESULTS_REVIEW.md) ordered相对global_stats的即时IoU增量为+0.00345615，其中自身fit预测均值固定策略差为+0.00373397，逐帧选择增量差反而为−0.000277819。ordered对自身固定预测均值仅+0.0000348556，且低于fit标签均值固定策略。主要差异集中一个折／种子单元。空间排序小幅改善仍不能证明逐帧选对时机。

本轮检查：把六层整体均值偏好校准到fit真实均值后，当前模型的逐状态残差是否具有增量？同时区分源视频整体适合哪层与同一视频内在正确状态选对层。

## 固定规则

对每个冻结模型，只取其fit源组，先各视频逐层平均，再对fit视频等权平均，得到真实均值μ_y和预测均值μ_p。保留Stage-197的预测float32及其均值计算，以复现原固定预测均值控制。然后用float64构造：

`e_t = p_t − μ_p`

`s_t = μ_y + e_t`

每模型计算六个列偏移μ_y−μ_p，共12模型／72个确定性校准量。没有优化器更新、阈值、残差缩放或追加模型训练，但仍使用fit标签估计这六个偏移；不能说完全不使用监督。分数只用于层间排序，不截断、不再过Sigmoid、不称为已校准IoU概率。因原均值使用float32，零均值残差仅在数值精度内成立。

这是按层的列均值校准，区别于Stage-196去掉每状态各层共有难度的行中心化。两者处理不同误差分量。

## 对照与状态无对应期望

每个模型均评价以下七种选择：

| 策略 | 定义 |
| --- | --- |
| raw | 原Stage-197存档动作，保留原Torch topk处理 |
| raw_redecoded | 原分数采用新诊断统一并列规则重新解码 |
| calibrated | 根据s_t选层 |
| fit_label_mean | 根据μ_y固定选层 |
| fit_prediction_mean | 根据μ_p固定选层 |
| native_L9 | 固定L9 |
| oracle | 逐状态真实IoU最优，仅作诊断上界 |

新解码沿用固定控制的规则：L9距最大分数不超过1e-8时保留L9，否则选精确最大值中第一个。raw_redecoded用于暴露近并列规则差异，不能将它的变化算作校准收益。

对每个视频、每种策略计算选层频率π_l，以及该视频真实各层平均IoUȳ_l，状态无对应的精确期望为：

`U_independent = Σ_l π_l ȳ_l`

真实对应的IoU为`U_actual = mean_t y_(t,a_t)`；差`U_actual − U_independent`精确等于各层选择指示与该层IoU的总体协方差之和。这相当于将同一视频内选择向量与所有状态独立配对后的期望，保留选层次数，无需选随机种子或打乱次数。固定选层的该差为零。

这个期望使用held标签进行评价，不是可部署规则，不向校准或选择函数提供held标签；也不是新的因果干预／完整轨迹或随机化检验p值。

## 来源、预算与统计

绑定原包SHA256 `b824c702af4a80f15ecebd418f4b55f3cdf3cfa8309c780368b1645220cc18b6`、Stage-197决策 `4b0f88097a9f748eb12d58a53e458b6e53d77d0016da7a001f3b8c7ea026b7b0` 及计划引用 `43a49f31db5532f7d7a934cdc6aad18f3b9e3a1869dd0c244956db10d137fb53`。检查决策指纹、10表原字节哈希、12模型网格、38源组、两折、两种子、真值与状态跨模型一致、原draw计数及存档动作有效。报告明示只复核回传表，没有原PT／NPZ的本机字节核验。

数据包含12模型×1824状态=21888条现有预测。0新网络前向、0新训练更新、0图像读取、0完整轨迹、0确认源组。不导入Torch；只依赖NumPy和已有轻量工具。沿用总体进度条，输出独立目录、不可变manifest、9表哈希及决策。完成时生成包含manifest的结果包。

先源组内平均两个种子，再对38历史开发源组bootstrap，保留既有20000次、原bootstrap种子。报告fit／held分数误差、加权排序、IoU／regret、状态对应增量、各层频率与并列解码差异。区间是历史开发集上的探索性名义区间，未做多重比较校正，不称外部确认或新AUC。

共同主要比较只看ordered的held：

1. calibrated相对fit_label_mean的平均IoU差。
2. calibrated相对自身无状态对应期望的平均IoU差。

同时报告calibrated相对raw、raw_redecoded、fit_prediction_mean、L9的差，以及ordered对unordered的校准后差，不能据此再挑一个分支作为确认方法。

若两主比较均有正方向及名义区间支持、两个种子方向一致、最小留一源组仍正，才值得讨论下一轮正常入口验证；当前阶段不会自动采用。若只改善层均值却没有状态对应增量，记录逐帧交互尚未建立，不反复调校准强度。已知Stage-194固定L7的缓存即时质量与完整轨迹相反，故本轮即使通过，也必须另测闭环精度、失跟、恢复和稳定损伤。

## 复现运行

源码及归档结果同步后，本地或服务器都能执行；无需GPU和SGLA权重。

```bash
cd ~/workspace/paper
conda activate sglatrack-test
bash tools/run_stage198_routing_calibration.sh
```

脚本默认使用仓库内已归档Stage-197原包。可通过STAGE197_BUNDLE指定同字节副本，STAGE198_ROOT修改独立输出目录；默认`output/stage198/fit_mean_state_correspondence_v1`。

```bash
mkdir -p output/stage198
nohup bash tools/run_stage198_routing_calibration.sh > output/stage198/run.log 2>&1 < /dev/null &
tail -f output/stage198/run.log
```

完成自动生成`stage198_results_<decision前12位>.txt`，重复运行核验并复用。若需单独检查：

```bash
python -B tools/stage198_routing_calibration.py \
  --output-dir output/stage198/fit_mean_state_correspondence_v1 --check-results
```

本轮运行结果另存复核文档，禁止将合成测试作为真实研究结果。
