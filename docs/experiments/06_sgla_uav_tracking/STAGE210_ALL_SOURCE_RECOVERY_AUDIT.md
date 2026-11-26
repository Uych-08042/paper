# Stage-210：全开发源恢复保护机制审计

固定协议，2026-11-26。诊断实验，不新增训练、完整轨迹或在线恢复方法，确认集封存。

209固定权重正确输入的两个主比较均未建立可靠增量，原生恢复保护仍受损。本轮不再增加交互结构或扫系数，使用全部38开发源及其既有原生／matched轨迹，区分输入父框状态、模型预测供给和标准读出的作用。

## 冻结输入与选点

209原包SHA256 `24c4323b9266efe2bd6589f164d752f48503c5258292674a29cb3720639225e5`，decision `6f0dd285f797b36b3a6a52244c02b458d89c29aa36ae157a1b4910ed52aca25f`。核验完整209目录、12段/10表、旧代码/协议、38原生及76 matched轨迹和NPZ/receipt。冻结208 own_pair/matched四权重，加载路径、scope及state沿用209校验，不使用其他fold模型。需要208物理模型/参考和209物理轨迹，只有TXT不够。

沿用failure_events：连续3帧IoU<.2形成失败，连续3帧IoU≥.5形成恢复，source frame=trace index+1。列出所有原生失败（含未恢复），纳入所有实际恢复的3帧，不用人为失跟前32/16/4等偏移。每seed以matched在这同3帧是否全部≥.5分为retained/lost，与209导出的missed_native_recoveries逐源逐seed必须一致。

正常对照只从原生连续3好帧中选，排除原生事件跨度及恢复3帧；每源取min(合格窗口数,max(1,原生恢复数))个起点，在合格起点列表中均匀取整，不看matched表现决定对照。窗口允许重叠；统计先窗口、再seed、再源，不能将重叠帧作为独立样本。无支持源仍在38源support表报告，条件均值明确supported_sources，不能用0冒充缺失。没有恢复的源仍提供正常对照；未恢复事件只做清单，不伪造恢复点。

这些是离线机制诊断，选点使用评价GT不构成可部署触发器，不能以事件条件样本上的结果声称总体AUC。

## 父框×模型交叉

每帧每seed运行四格：原生父框/原生模型、原生父框/matched模型、自身父框/原生模型、自身父框/matched模型。父框严格取209该帧实际输入父框，不把预测输出框错当同帧父框。每格从原首帧模板初始化，设置原frame_id，执行原生公开track一次；不续写反事实框。跨格只换父框或冻结模型，其余模板、路由、CENTER、Hann、裁剪公式不变。记录实际调用次数。

捕获同一次公开调用的CENTER score/size/offset图及sample_target的真实resize_factor。每个网格位置使用原cal_bbox强制该格，再用原map_box_back和clip_box，保持原torch缩放顺序，选中格框必须与公开输出逐元素严格相同。没有二次网络/头forward，不借重新执行生成不同预测图。

真实整数裁剪覆盖沿用207 round、padding及既有右下边界；完整包含GT不是无遮挡或身份可靠的证明。记录全网格best IoU、标准输出IoU以及互斥类别：good_output（输出≥.5）；good_grid_missed（输出<.5且网格best≥.5）；covered_no_good_grid（输出与best都<.5但完整覆盖）；not_fully_covered_no_good_grid（前者且不完整覆盖）。最后一类只表示几何与供给同时受限，不能直接断言裁剪是唯一原因。候选阈值沿用已有恢复IoU .5。

同runtime下两个对角格须精确复现来源框及层；换卡使用新根，当前所有格同卡执行，冻结事件/父框保持，记录来源误差，不混减新旧AUC。软件/原权重/config不改，续跑不能换GPU/UUID/workers/code。

## 执行与汇总

默认两个spawn进程跨序列，每序列帧串行、独立runtime/hook/缓存；一个总体进度条。名单首个有支持源额外串行全审计，随后工作进程的全部同源缓存数组必须逐字段相同（不含额外计时字段）。这是执行一致性检查，不是普遍机制证明。原始maps、grid、box、factor、parent、call数组和元数据/receipt保留，汇总只在父进程。

汇总all_recovery/retained/lost/normal，先每三帧形成事件指标，再每seed平均、源等权。报告完整38源清单及各层supported_sources；lost/retained可能单seed受支持，须明示，不冒充两个独立seed。比较固定模型下原生减自身父框，以及固定父框下matched减原生模型，输出即时IoU、网格供给和三帧均好率的名义源bootstrap区间。只有1源时不计算区间，0源报告空值。所有均为探索性诊断，不设方法采用门槛，不把同帧交叉差当闭环因果结论。

若覆盖/供给不足优先考虑状态或表征；若好格点漏选才考虑读出监督；若瞬时改善无法保持则另验写回后效。不能在运行前认定主因，不能用离线最佳格点写回来声称恢复成功。

## 运行

```bash
cd ~/workspace/paper
mkdir -p output/stage210
nohup bash tools/run_stage210_recovery_audit.sh > output/stage210/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage210/run.log
```

默认output/stage210/all_source_recovery_audit_v1、GPU0、workers2，可覆盖STAGE210_ROOT/GPU/WORKERS、STAGE208_ROOT/BUNDLE、STAGE209_ROOT/BUNDLE、CHECKPOINT及原七数据根。完成自动汇集10表及manifest/decision为stage210_results_<UTC>_<PID>.txt。--check-results可复核重打包；失败回传主错误与worker_logs，不降低精确一致条件。
