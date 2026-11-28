# Stage-217：冻结预测器的原目标拟合与泛化审计

状态：已固定实现；真实服务器缓存待运行。此轮不是新跟踪方法，也不报告新 AUC。

## 问题和依据

Stage-215 关闭 coverage 更新器的尺度分量，显著减轻完整轨迹损伤，但只中心仍不优于不修正。Stage-216 固定共同当前状态发现，一步尺度修正平均使下一帧 IoU 下降 .0169559；67.76% 状态的诊断几何误差也变差。几何改善子集平均输出贡献为正，所以不能直接认定“几何已预测好，只是与任务质量错配”。

先检验预测器是否学会了原本的目标，再决定是否值得更换目标、扩大网络或重新训练。只消费 Stage-214 已训练的八个 MLP、原始 fit 提供轨迹、Stage-215 当前尺寸缓存和 Stage-216 共同状态；固定两个折与两个种子，覆盖全部 38 个开发源组。确认集继续封存。

## 固定对照和防泄漏

每个折／种子独立重建 Stage-214 的完整 fit 样本（19 源，每源 nn/mm 各至多 48 点），以原始抽样表和全 fit 特征核查归一化缓冲区与 draws 哈希。只读已有模型，绝不优化模型参数。

在同一特征、原生 reference box、下一帧 GT 上比较：

1. `mlp`：原 forecast 或 coverage 冻结预测器在 CPU 上重新前向。
2. `zero`：三维修正全零；仍按原 objective 的边界约束计算。
3. `fit_constant`：只用该折该种子的 fit 标签。先逐源求标准化中心变化与 log 几何平均侧长变化的均值，再等源平均为一个三维常量。fit 的 nn/mm 合并，两个 family 共享此常数。**它是简单的标签均值基线，不是 Smooth-L1 或 coverage 的最优常数解。** 不使用 held 标签、held 特征归一化或事后最佳尺度。

fit 使用 Stage-214 原始全部提供样本；held 使用 Stage-216 的每源 48 个均匀点、四种自身历史来源 `coverage__none / mm / coverage__center / coverage__joint`。前两者是提供风格对应的 held 状态，并非宣称不同 GPU 下与 Stage-214 轨迹逐字节一致。

模型文件、receipt、源 NPZ、metadata、折、种子、归一化、训练 draws 和权重 digest 全部绑定来源已审阅 decision。GT 与帧表仍通过既有七来源入口核验，只读取标注和路径，不调用图像 reader。不得通过读取图片补齐缺失缓存。

## 原目标、时间对齐与统计

直接调用未经修改的 `stage214_search_state_common.loss`，使用 float32 CPU：

\[
L_{\rm forecast}=L_{\rm SmoothL1},\qquad
L_{\rm coverage}=L_{\rm SmoothL1}+L_{\rm overflow}.
\]

分别导出 total、regression、coverage 三项。中心按 reference 几何平均侧长标准化，尺寸回归为 log 侧长比；预测中心及尺度沿用原图像边界裁剪。此损失与 Stage-216 的几何诊断量定义不同，不混用。

输入来自当前帧 t，目标为 t+1。边界约束严格使用 **当前 t 的 image_hw**；held 从绑定的 Stage-215 trace 读取，并校验 reference/output 与 Stage-216 当前状态精确一致、下一帧尺寸与各分支一致。不得以 t+1 图像尺寸替代 t。三种方法统一 CPU 计算，以免把设备差异计为学习收益；CPU 模型输出相对原 GPU 缓存的最大绝对漂移另表记录，不宣称 CUDA bitwise replay。

**唯一主要比较**：held 的前两种提供风格状态等权后，forecast MLP − zero 的原始 regression loss。负值有利。先源内平均两 seed／两风格，再 38 源等权，沿用既有 20,000 次源组 bootstrap；报告名义 95% CI、正负源数及删除最有利源后的最坏均值。fit 与 held、两 family、其余常数比较均为机制支持分析。反复开发下的名义区间不作为独立确认结论。

不把 fit/held 差异直接解释为纯分布漂移：两折模型、GPU 来源、状态历史也不同。若 fit 本身不能优于零或常数，应优先检查原拟合／优化；若 fit 好但 held 差，应检查泛化及状态来源；若原目标明显改善而任务质量仍损失，再检查目标与真实搜索质量的对应。此轮不训练新方案，不选择门槛，不重新开完整闭环。

## 预算、产物和实现约束

0 新训练、0 图像读取、0 CUDA 推理、0 新完整轨迹、0 确认源。CPU 单进程；无需 checkpoint 或 GPU 型号／UUID匹配，不能把 CPU 数值审计当成换卡完整复现。

真实固定支持下：7,296 fit 样本，14,592 held 状态，两个 family × 三种方法共 131,328 个逐样本损失计算。一个总体进度条 469 项；不增加多进程 CUDA 工作者。

九表：`support` 456 行、`model_scope` 8、`constants` 4、`normalization` 4、`group_metrics` 2,736、`policy_summary` 36、`paired` 73、`prediction_replay` 608、`execution` 1。另封存来源数组哈希及 decision，自动汇总为一个 TXT。失效或缺失的来源缓存必须停止，不能重跑 GPU 自动补齐。

Stage-216 服务器使用换行修复前入口（SHA `7fa3dd495dbeb17ad7a4eb1719bb9360d9c3e15545124ab7b7e494a0f2b17bbc`）。仅允许已封存源码的 `bundle` 读取补丁差异；全部计算函数 AST 必须一致，其余来源代码及协议仍核验。Stage-215/216 导出仅允许固定规范化哈希下的 CRLF/LF 差异，数值和额外字节变化拒绝。不修改已冻结的 Stage-214～216 协议／计算代码。

## 服务器运行

同步本轮代码、协议和已审阅 216 TXT／源码快照后，在原 conda 环境和仓库根目录执行。若本地未提交／推送，服务器 `git pull` 不会自动得到这些修改。

```bash
mkdir -p output/stage217
LOG="output/stage217/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage217_predictor_audit.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage217/run.pid
echo "$LOG"
tail -f "$LOG"
```

`tail` 可随时 Ctrl-C；退出 SSH 后 nohup 任务继续运行。默认结果目录为 `output/stage217/cached_predictor_objective_v1`。Stage-215 默认使用 `same_model_gpu_replay_v1`，其他根目录沿用上一轮环境变量；可用 `STAGE217_ROOT` 指定独立输出目录。旧结果不覆盖，来源绑定改变时使用新目录。

```bash
python -B tools/stage217_predictor_audit.py \
  --output-dir "${STAGE217_ROOT:-output/stage217/cached_predictor_objective_v1}" \
  --check-results
```

完成后提供脚本打印的 `stage217_results_*.txt`，再依据 fit/held、零／常数对照判断下一步。此时尚无真实 Stage-217 结果。
