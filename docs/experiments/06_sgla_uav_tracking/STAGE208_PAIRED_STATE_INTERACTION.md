# Stage-208：父框状态来源 × 空间交互方式

固定协议，2026-11-26。实施准备不等于已取得真实 GPU 结果。

Stage-207 的 own_pair 对 native_pair 完整视频 AUC +2.6796，名义源组 CI 为正，但对原生 +.6261 的区间跨零，匹配恢复仍受损。该阶段三个学生都使用正确模板，无法分辨自身状态训练收益与正确目标交互的贡献。Stage-206 的原生状态正确模板也未胜过同参数搜索控制。因此本轮不增加选层、预测头或搜索区域模块。

## 1. 冻结来源与干预

- 使用已复核 Stage-207 bundle（SHA256 `3cc83a070037f8be34f76fb6432ac8415ccb1e24f9d5442ff67fad97616281eb`）、完整物理运行和 Stage-194 原生特征。
- 两种父框来源：native_pair、own_pair。逐 fold/seed 重新核对既有 3648 个抽样位置的两侧实际整数裁剪覆盖、3145 个共同支持状态以及全部支持表。不得增加状态、重新选帧、删掉难源或按评价真值选择训练样本。
- 三种交互沿用 Stage-206：search_control 从当前搜索 token 空间池化得到键值；matched_spatial 使用当前模板 token；mismatched_spatial 使用其他 fit 源组的模板特征均值。
- 错配参考均值只由 native_pair 的 fit 样本计算，同一 fold/seed 在两种父框来源间完全共享。fit 源按既有循环错配，held 源按哈希映射到 fit；不从 held 计算参考。实际跟踪模板不替换。
- 三种交互都是全宽 192 的 Q/K/V/O 单头残差，4×192²=147456 个参数。O 零初始化，初始适配严格为恒等。原生 L6 前缀、MLP 路由权重、CENTER、Hann、裁剪、写回均不训练；MLP 接收适配前 F6，不把适配引起的路由输入变化混为选层改进。

## 2. 匹配训练

2 状态来源 × 3 交互 × 2 外层 fold × 2 seed（2132、2133），共 24 个新训练模型、6144 次更新。每模型 256 更新，batch 4，AdamW lr 1e-5、wd 1e-4、梯度裁剪 1；使用原 tracking loss：2 GIoU + 5 L1 + focal。保持 Stage-207 抽样日程、共同支持、初始化。六条件每个 fold/seed 的日程／初始化／参数量／更新数／样本数必须完全一致。

每例只执行原生路由选中的一个候选层，不用六层事后最优标签。正确模板两组是 Stage-207 的控制复现，同一完整 runtime 必须精确复现各自 8 个模型的状态哈希；若 GPU 型号或 UUID 改变，另建输出根，全部 24 个模型和 494 条评价在该卡重做，记录控制差异，不能混入旧卡模型结果。两侧训练特征均对称复用原冻结缓存，不只重算一侧；它们是固定离线输入，非声称新卡采集。软件、checkpoint、配置及旧代码哈希必须相同。

数值非有限时记录诊断并停止。不能在同根更换设置继续。24 组均做原生父框 fit/held 读出，用于解释，不作为新 AUC。

## 3. 完整视频与并行

38 个历史开发源组，原生 38 条 + 六条件 × 两 seed × 38 条，共 494 条完整 OOF 轨迹；训练模型只评价自身 outer held 源。未来真值仅用于离线训练标签、共同支持核验及评价，不进入在线动作。确认集继续封存。

默认训练和固定父框读出串行；完整视频评价使用两个 spawn 进程，按视频分片，每个视频内部逐帧顺序执行、先原生再十二学生。各进程独立模型、原生运行时、hook 和缓存，写不同视频文件；父进程持总 writer lock 和汇总权限。线程数默认 1，子进程日志单独保存，父进程只显示一个总体进度条。

放行完整评价前，对固定名单前两个视频、各前最多 8 个跟踪帧、全部 13 个策略做串行及双进程 smoke。逐帧框、父框、层、前缀／block／head／selector／crop 调用计数必须严格相同，计时排除。26 对轨迹检查全部通过才运行完整评价。失败不能放宽误差、替换为成功个例或继续评价。`--workers 1` 仅使随后完整评价串行，仍保留双进程 smoke，使用独立输出根。

并行是执行方式，不是算法贡献。GPU 上的并发可能改变浮点结果或使吞吐更慢，不能预设加速。记录 smoke 的串行墙钟、含启动的并行墙钟及完整评价墙钟；缓存复用时不是有效速度基准。并行时 pipeline_ms 受竞争影响，仅作描述，不与旧串行 FPS 作效率结论。

## 4. 分析与下一步判据

两个预先声明的主要比较：own_pair 内 matched_spatial 分别减 search_control 和 mismatched_spatial 的完整视频 AUC。先在每个源内平均双 seed，再以 38 源组 bootstrap；报告名义 95% CI 和两主要比较 Bonferroni 家族区间，不将帧当独立样本。报告胜负、中位数、最差 leave-one-source-out、两个 seed、原生比较，以及状态来源 × 正确交互的差中差。

同时报告低 IoU、最长失跟、原生恢复丢失、原生好帧损失、新增稳定源失跟。正确模板胜过两控制才增加其特异性证据；自身状态来源有益而交互无差，下一轮优先状态匹配，不能把收益归因目标条件交互。局部读出提高不等于完整跟踪通过；任何结果都保持 REVIEW，不自动采用、不开放确认组。

## 5. 运行与回传

须保留 Stage-207 完整目录（provider 缓存、paired 模型、receipt、manifest、decision、表）及 Stage-194 特征，不能只有 TXT。原 Stage-206 模型无需作为新运行输入，因为 provider 的冻结状态与帧已保存在 207。

```bash
cd ~/workspace/paper
mkdir -p output/stage208
nohup bash tools/run_stage208_state_interaction.sh > output/stage208/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage208/run.log
```

默认 GPU 0；`STAGE208_WORKERS=2`，`STAGE208_GPU`／`STAGE208_ROOT`／`STAGE207_ROOT` 等支持覆盖。中断后保持 GPU/UUID、源码、配置和参数不变原样重跑，校验后复用已提交缓存。换卡或换 workers 必须更换新输出根。完成后自动生成 `stage208_results_*.txt`；失败先回传报错和 `worker_logs` 对应日志。核验入口：`python -B tools/stage208_state_interaction.py --output-dir "$STAGE208_ROOT" --check-results`。
