# Stage-212：在线输出框与搜索状态解耦

固定协议，2026-11-26。0新训练；38历史开发源完整OOF，确认集封存。真实GPU结果待运行。

211原生轨迹父框分解提示中心/尺度共同影响恢复，替换同时会伤害保留恢复。原生轨迹几何是离线参照。本轮改为同一当前裁剪内可计算的原生读出，检验把适配输出直接反馈成搜索父框是否造成能力损失，不预设中心方案已经最优。

## 来源和预算

读取已复核211原包SHA256 `381e3f7797bb96397bc8f431e742a8d055a6eb1255437ed71980e3e60255798b`，decision `42626623990807863c851939f51c191823accc1085fa218801f0daf7a58d1be0`，核12段10表和冻结代码/211协议。只消费其运行身份、38源名单、fold及四模型scope；不读取其中事件选择、旧父框或未来GT作为动作输入。无需211/210原始审计目录。

实际模型仍是208 own_pair/matched四权重，由208原包、PT/receipt/condition子plan/scope/state绑定，scope与211表一致；不训练新选择器、骨干或头。需要208物理模型/参考、原checkpoint及七数据入口。帧/GT hash及原软件/config/权重保持。L20/L40或UUID变化时全部控制和新策略同卡重跑，使用新输出根；续跑不得换GPU/UUID/workers/代码/参数。

正式策略：native单路（seed0）；matched_single单路（双seed）；双路mm/nm/mn/nn（各双seed）。共38+76+304=418完整轨迹、445687跟踪帧，0训练更新。全部新跑，不把旧卡或旧运行结果混入主比较。

## 单裁剪双读出

每帧先在当前自身搜索父框上执行原生公开track一次，获得适配前F6、原生MLP选层及原生读出框r_t。保留同一F6，在冻结残差适配后只再执行同一被选候选块和CENTER，按原Hann/cal_bbox/map_box_back/clip计算适配框a_t。F6捕获和复制不触发第二次前缀；第二路不再调用MLP或sample_target。模板保持原首帧模板。对外输出始终为a_t。

下一帧搜索状态取当前两读出的中心/尺寸组合，m来自a_t、n来自r_t：mm为全适配反馈，nm只采用原生读出中心，mn只采用原生读出宽高，nn全原生读出。按真实cx/cy还原xywh，端点直接复制。n不是另一条旧原生轨迹，也不是GT；在nm/mn时参考分支会看到该组合策略自己的裁剪，不能假设等于单路原生轨迹。

四个双路策略都执行同样计算：1 crop、6共享前缀块各1次、1 MLP、被选候选块2次、CENTER2次，总block执行8次。native和matched_single为7 block/1 head。完整记录真实调用及计时，额外候选/头不能说成免费；并发pipeline_ms不是隔离FPS。

## 执行身份与一致性

双路使用独立审计：parent必须等于上一帧next_parent，next_parent必须等于当前r_t/a_t的预定混合；不能继续用旧parent=上一帧对外boxes的断言来强迫解耦状态等于输出。保存parent、next_parent、reference_boxes、boxes、层和全部调用数组。

在每个held fold首个视频前最多8帧，全部11策略串行及双进程各一次，共22对严格比较所有状态/输出/调用数组，计时排除。先验收，再完整评价。每个视频及每seed完整要求双路mm的frame/parent/boxes/layer精确复现新单路matched；nn的reference_boxes/parent/layer精确复现新原生。这里只保证nn原生分支和搜索状态，nn对外适配框未必等原生，不能替换身份检查对象。

默认两个spawn进程跨视频，各独立runtime/模型/hooks/cache，序列内部顺序；主进程writer lock、汇总和单总体进度条，worker日志单独保存。模型与原生参数/缓冲不变。错误不放宽精确条件，不以另一个成功片段替代。

## 主要比较和能力保护

预先声明两个主要完整AUC比较：nm−mm、mn−mm；没有因211中心点估计较大就删掉尺寸对照。先源内平均双seed，再38源bootstrap20000次seed24194，报告名义95%及两个主要比较的Bonferroni家族区间；nn−mm和各原生/单路比较为支持性分析，报告2×2平均效应和交互，不事后只挑赢家。

同时评价原生和matched_single的能力保护：AUC/低IoU/最长失败；分别以两者为参照的恢复丢失、好帧损失、新增稳定源失跟。native行对matched的指标平均两seed参照，其余行对对应seed参照。回到原生水平并丢掉0176等适配既有改善，不算方法成功。不得删除坏源或用评价GT在线选择状态；所有未来GT仅用于完整轨迹完成后的评价。

最终保持REVIEW，无自动采用或确认集开放。若改善只表现为回退原生，需要重新考察适配收益的可保留性；若某单分量改善完整精度和两种能力保护，再做独立验证或针对状态更新学习；若均无收益，降低直接替换状态的优先级，不继续扫混合系数掩盖失败。

## 运行

```bash
cd ~/workspace/paper
mkdir -p output/stage212
nohup bash tools/run_stage212_state_experiment.sh > output/stage212/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage212/run.log
```

默认output/stage212/online_output_search_state_v1、GPU0、workers2。可覆盖STAGE212_ROOT/GPU/WORKERS、STAGE211_BUNDLE、STAGE208_ROOT/BUNDLE、CHECKPOINT及七数据根。完成自动打包10表+manifest/decision为stage212_results_<UTC>_<PID>.txt，--check-results核验重打包。需要208模型，只有TXT不够；不需要旧原生轨迹或211原始NPZ。失败回传主日志及worker_logs。
