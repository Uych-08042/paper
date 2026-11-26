# Stage-204 目标—搜索关系输入匹配实验

日期：2026-11-26。真实训练前固定。Stage-203取消裁剪未建立held收益，转查路由输入的关系表示；不预认定缺少关系就是根因。

三组concat、matched_relation、mismatched_relation。原全局描述为a=meanZ、b=stdZ、c=meanX、d=stdX的拼接；正确关系为[a*c,abs(a-c),b*d,abs(b-d)]，乘积逐通道。错配关系保持相同运算，用另一fit源组的模板统计替换a/b。原模板、原token与真实定位标签不变，因此不是有模板／无模板实验；原联合骨干本来已包含目标条件。

参考规则固定：fit组按排序循环取下一个fit源组；held组以源组名称SHA256整数模fit数量选择fit参考。每参考的a/b取该fit视频全部缓存状态的模板统计均值，转换float32；不读取参考标签、不用held未来或held标签决定参考。此错配组只作诊断。每视频映射及reference哈希导出。正确关系仅用当前token。

三组均在fit内按源组等权计算每个描述坐标的均值、总体方差，再中心化／除以标准差；仅零方差坐标分母为1。没有held校准或可调尺度参数。原输入控制也采用同规则，因此不要求它与203原始未标准化控制逐bit相同。ordered token投影与其fit中心化保持不变。描述计算float32，与直接Torch路径对齐；描述均值／尺度以float64统计并保存buffer，中心化／缩放后再转float32接读出，避免小方差坐标被均值舍入误差放大。

三组均为4×192=768描述＋320有序投影的1088输入，6720可训练参数，零读出／相同投影初始化、同先验。固定buffer因输入不同而不同，故初始匹配哈希只计算可训练参数，最终状态哈希含buffer。训练／直接路径／预聚合评价均使用同一关系定义与固定参考。

38历史开发源组、原19／19两fold、2132／2133、每视频48原生父框缓存状态。同Stage-200配对日程，自配对保留；固定point MSE、clip=1、AdamW lr1e-5 wd1e-4、256步512对1024引用。12模型3072更新12288训练引用6144模型对引用，独立日程仍2048行；21888评价输出。来源为完整194 NPZ／receipt目录与已审203包，SHA256 1fc37a4c5aa9e21cd385771b22b393158da0330aeefcaa7376b9c58197887fee、决策e5f2e4bd5c58d3728cc80b1726b8590bcc22ec80a024fa74ec4e4d694055568e、计划7d8f3a19278b77d522fece421f957f294a5457cb3a42499d4feef21f94a76079。核验来源指纹、14表、代码、协议及每个194文件和逐状态标签。

主要held比较四项：matched−concat、matched−mismatched、matched−prior、matched自身状态对应增量，均为即时平均IoU。先平均同源两seed、再源组等权，固定bootstrap seed24194／20000次名义区间。历史开发暴露不是独立确认，多比较未做全族校正。只有两种关系一起变好不能归因正确目标匹配；fit好而held不升不能称泛化改进。此类乘积／差异运算本身不是原创理论。

14表：model_scope12、fit_summary6、source_bridge38、pair_schedule2048、pair_support76、error_summary18、error_video1368、policy_summary30、policy_video2280、paired66、frame_predictions21888、training_history3072、seed_summary60、descriptor_audit228。预计948条模型／统计／预测及receipt引用。记录梯度、实际裁剪和参数更新，非有限值封存失败JSON并停止、禁止自动重试。正常中断可按metadata／哈希复用；未知或篡改缓存拒绝。

0新图像／骨干／CENTER／完整轨迹／确认组。没有新全视频AUC、块数或速度结论；不据缓存固定L7优就采用。满足缓存条件后才另立自身轨迹验证。旧1～203代码／协议／证据冻结。本机无真实194特征，仅做合成CPU测试，真实结论待服务器。

服务器sglatrack-test、~/workspace/paper：

```bash
mkdir -p output/stage204
nohup bash tools/run_stage204_target_relation.sh > output/stage204/run.log 2>&1 < /dev/null &
echo $! > output/stage204/run.pid
tail -f output/stage204/run.log
```

默认194目录output/stage194/module_ablation_v1，203原包在仓库，输出output/stage204/target_search_relation_v1，可由STAGE204_ROOT指定；可设置STAGE194_ROOT／STAGE203_BUNDLE。单一总体进度条，完成回传stage204_results_<UTC>_<PID>.txt（decision／manifest／14表）。无需GPU／SGLA checkpoint。

```bash
python -B tools/test_stage204_target_relation.py
python -B tools/stage204_target_relation.py --output-dir "${STAGE204_ROOT:-output/stage204/target_search_relation_v1}" --check-results
```
