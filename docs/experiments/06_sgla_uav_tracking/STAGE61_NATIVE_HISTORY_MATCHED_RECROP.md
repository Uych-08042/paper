# Stage-61：原始 V1 历史候选与同预算当帧重裁剪

日期：2026-11-02，Asia/Hong_Kong。主任务仍为 SGLA 无人机单目标跟踪，使用既有 VisDrone-SOT 的 12 个已消耗开发源组。接续 [Stage-60 复核](STAGE60_RESULTS_REVIEW.md)和 WORK_HANDOFF 第 0.11 节，保持原 V1 基线，不接入 V2。本文是本轮固定协议，尚无服务器结果。

**当前运行入口更新：用户已确认仅能使用 L40，应使用[独立 L40 六组重测协议](STAGE61_L40_DEVICE_MATCHED_RUN.md)和 `tools/run_stage61_l40.sh`。本文保留原 L20 来源复用方案及启动诊断，原始封存代码不变。**

## 问题与本轮新增信息

Stage-60 的 response 使用几何适配头 E，对 current 宏 AUC +3.7427、9 胜 3 负，但区间跨零，而且从每帧一次增加至四次搜索。尚不能确定收益是否依赖 E 的适配，或普通额外裁剪也能提供相同收益。

本轮回答：

1. **原始 V1 权重上**，自身历史候选＋Hann 选择是否也优于原生单裁剪？
2. 在完全相同的四次搜索预算下，历史候选是否优于**只用当帧初次预测构造的重裁剪**？
3. 在已有 E 头上重复第二项比较，观察候选来源的作用是否随基线改变。

这为“因果历史中心—尺寸组合”的新增价值提供更明确的对照。若当帧重裁剪已达到相同效果，应据结果调整历史先验的贡献定位。该控制不穷尽普通多尺度、空间搜索方案，不把对一个控制的结果推广到所有额外搜索。

## 六组完整跟踪策略

| 组别 | 网络 | 搜索父框及选择 | 每帧完整搜索 | 来源 |
|---|---|---|---:|---|
| native_current | 原始 V1 | 原生上一预测父框 | 1 | 复用封存原生轨迹 |
| native_history | 原始 V1 | 原 Stage-60 四个历史候选，Hann 峰值选择 | 4 | 新推理 |
| native_recrop | 原始 V1 | 下述当帧中心×尺寸四候选，同一 Hann 规则 | 4 | 新推理 |
| endpoint_current | E | 原 E 单裁剪 | 1 | 复用 Stage-60 current |
| endpoint_history | E | 原四候选＋响应选择 | 4 | 复用 Stage-60 response |
| endpoint_recrop | E | 当帧四候选＋同一响应选择 | 4 | 新推理 |

E 沿用 Stage-54 rollout_preserve／Stage-58 endpoint_control：保留原生骨干、独立候选、选择器与响应参数，只曾适配 size／offset。原始 V1 和 E 在本轮均完全固定，不训练新模型或评分器，不平滑输出，不更新模板。

原始 V1 是每视频一个确定性算法，只运行一份，统计中不复制为两个种子。表里的 native seed=0 是来源标识，不代表额外随机试验。E 按原 held-out 折加载对应的两个种子 2132／2133，先在视频内平均，再按视频比较。所有旧 fit／held-out 组仍是开发证据，不重新划组或宣称新的持出集。

## 四候选的具体定义

历史组严格保留 Stage-60 配方：最近至多 5 个已接受框；当前中心／t−2 滞后中心 × 当前尺寸／逐维 log 中位历史尺寸。候选 0 整框直接复制；历史不足时按原规则回退。

当帧重裁剪组令上一已接受框为 B；先以 B 运行一次原生公共 track，得到当前图像的解码框 D。B 和 D 的中心、尺寸组合为：

| 候选 | 中心来源 | 宽高来源 |
|---|---|---|
| q0 | B | B，整个父框直接复制 |
| q1 | B | D |
| q2 | D | B |
| q3 | D | D，整个 D 直接复制 |

q0 已执行的响应及输出直接复用，只再执行 q1～q3，**总计四次，不能变成五次**。同一真实当前帧上完成这四次搜索；没有未来图像、GT、其他轨迹或教师框参与构造，没有新增尺度、偏移幅度或触发阈值。四候选即使同像素也实际运行并计账。

两个候选组都按原 float32 `raw × Hann` 的峰值选择。非 current 候选需高于 q0 分数 1e−6；并列保持 q0 或按固定候选顺序。只接受所选候选的原始解码输出，用它生成下一帧历史。未选输出不进入历史、模板或隐状态。frame_id 每个真实图像只推进一个逻辑帧。

重裁剪的**候选及选择**只依赖 B 和当前图像的第一次输出，不依赖更早历史。为兼容 Stage-60 的审计，仍记录同一套 24 维特征；其中历史统计只供日志，不进入该组的父框或峰值选择。本轮没有训练／使用原来的 immediate／future scorer。

## 来源校验和实际成本

输入完整 `output/stage60/causal_search_state_value_v1/`。prepare 只读取和核验 Stage-60 的 manifest、完成回执、模型、原始缓存、9 表及完整上游来源链；不调用旧 summary 改写报告，不重跑旧推理。E current／history 的全部来源轨迹重新作因果父框与选择审计，原始 native 来源沿既有链验证。

新目录为 `output/stage61/v1_history_matched_recrop_v1/`。按已有 13,525 跟踪帧：

- 原始 V1 两组：12 × 2 = 24 条新增完整轨迹，108,200 次搜索。
- E 当帧重裁剪：12 × 2 种子 = 24 条新增轨迹，108,200 次搜索。
- 正式合计 **48 条、216,400 次完整搜索、1,514,800 次 block**，无额外仅头前向。
- 复用 60 条完整轨迹：native current 12 条、E current 24 条、E history 24 条；其 148,775 次来源搜索不算本轮新增计算。
- smoke 在固定首视频前 32 帧运行 native 及两个 E 种子的三种策略，共 **864 次额外搜索**。其中两种 current 及 E history 必须精确复现对应来源前缀；新组还核对解码和调用。
- 正式＋smoke 共 **217,264 次搜索**。模板初始化未计入搜索次数，pipeline 计时也不包含初始化、读图和哈希。

smoke 是额外短轨迹，**不冒充完整评价，也不复用为正式完整轨迹**。formal 按冻结的视频轮换组别顺序执行，降低固定先后次序的影响。每项提交 NPZ＋密封 receipt；已提交项重复执行只读取审计，未提交项重新运行，已有回执被修改时拒绝继续。

同预算指每帧网络调用数完全相同，不保证两种政策自身轨迹上的实际毫秒数相同。只对新产生的轨迹报告新 pipeline 中位／P95，排除前 10 帧；不混用旧 E response 的历史计时来声称同环境速度比较，不建立部署加速或能耗结论。

## 固定判读规则

主要对比是 `native_history − native_recrop` 的视频宏 AUC；必要对比是 `native_history − native_current`。两项 AUC 给出名义 95% 区间和预定 Bonferroni 两比较家族区间，家族每尾为 0.0125。源视频是抽样单位，原生只有一个确定性运行，E 两种子先在视频内平均。区间条件于既有开发组与固定 E 拟合，不覆盖全部训练不确定性。

若两项均值都 ≥+0.5 AUC、家族下界都 >0，并满足以下要求，标记 `PROMISING_NATIVE_HISTORY_RECROP_PILOT`；否则为 `REVIEW_NATIVE_HISTORY_RECROP_PILOT`：

- native 原稳定视频无新增失效；低 IoU 均值不升。
- 原生可靠帧（原生 IoU≥0.5）AUC 差的名义下界 ≥−1.0，这是预定实践容差。
- 至少 4 视频发生实际输出改变，视频宏改变比例 >1e−4。

E 内对比及 `[(E history−E recrop)−(native history−native recrop)]` 为次要开发描述，不反向改变主要比较。这个交互是完整策略间的条件差，不能当成适配训练的唯一因果机制。

分别报告低 IoU、最长失效、恢复／末尾未恢复／再失效、尺寸误差、原生可靠帧、原生失败帧队列、逐视频影响及候选曝光。原生失败队列为固定原生 IoU<0.2 的帧，与可靠帧口径不同，不把子队列均值相加分摊总体收益。逐视频／种子表同时列出已有 Stage-60 E 轨迹，便于定位此前 response 的负例。

本轮按指标方向计算 `worst_video_delta`：AUC 取最小差，低 IoU／最长失效取最大差；另给出方向正确的最差留一值。旧 Stage-60 字段及代码保持封存，不追改。

## 本机验证与边界

3 项 CPU 测试已通过，16.958 秒、无跳过：中心／尺寸交叉及 current 逐位复制；真实小通道 CENTER 头与原生公共解码、四次而非五次调用、修改未来 GT 不改变预测／候选／特征／动作、候选顺序和已接受状态隔离；Stage-59→60→61 合成完整来源流程、旧文件逐字节不变、来源前缀精确重放、读取恢复与缓存／报告篡改拒绝。5 文件 AST、两个 NumPy-only CLI help、Bash `-n`、文档链接和维护检查通过。

合成 fixture 的上游 Stage-58 来源边界、骨干与采样图像由测试构造；使用真实 CENTER、原公共 decode 和实际调用 hook。它不是完整 GPU 精度复现，也没有重新训练上游阶段。服务器完整 Stage-61 尚未运行。

## 启动身份校验修复（2026-11-02）

服务器首次启动在旧 `make_runtime` 的原生运行环境身份比较处退出，尚未进入图像推理。新增诊断后用户报告明确差异：**封存源为 `NVIDIA L20`，当前选中的设备是 `NVIDIA L40`**。当前错误来自硬件型号不一致；此前的配置漂移仅是假设，不能写成已证实原因，也不能由这条错误断言权重错误。

新增独立入口 [stage61_runtime_replay.py](../../../tools/stage61_runtime_replay.py)：从现有封存 manifest 的 `runtime.config` 构造独立 EasyDict／TrackerParams，只在内存中替换参数加载；仍调用原 Stage-46 工厂并严格核对原 checkpoint SHA、完整配置、PyTorch／CUDA、GPU 型号／UUID、网格维度、OpenCV／NumPy 和原网络状态摘要。当前 YAML 与封存配置的差异另行记录；不同权重或环境直接报出具体字段并停止，不能用此入口放行。

Stage-1～60及 Stage-61 五个实验 Python 文件、现有 manifest／run／轨迹均不修改，因此保留已 prepare 的 Stage-61 目录即可。该入口只修复启动参数复现与诊断，不改变候选、选择、种子、预算或分析。4 项本机启动回归通过，覆盖真实旧工厂调用契约、配置快照恢复，以及错误权重／软件／模型摘要／GPU UUID／网格维度／封存绑定拒绝；CUDA 构造与设备在测试中由 fixture 替代，不等于真实 GPU 验证。

修复后原 Stage-61 的 3 项 CPU 测试也全部通过，20.689 秒、无跳过；新入口 CLI help、Bash 只读 `-n`、维护检查均通过。没有服务器 GPU 结果。

启动记录位于 `diagnostics/stage61_runtime_smoke.json` 和 `stage61_runtime_evaluate.json`，含封存计划指纹与新入口 SHA。成功状态为 `EXACT_SOURCE_RUNTIME_VERIFIED`；身份差异为 `RUNTIME_MISMATCH`；模型构造异常为 `RUNTIME_CONSTRUCTION_FAILED`，保留原异常。单独读取：

```bash
cat "$STAGE61_ROOT/diagnostics/stage61_runtime_smoke.json"
```

补充只读设备清单与 `--gpu auto`：清单使用 PyTorch 的 CUDA 逻辑编号，直接对应原脚本的 `--gpu`，不把 `nvidia-smi` 物理编号误当成 `CUDA_VISIBLE_DEVICES` 重映射后的编号。自动选择同时匹配源型号及 UUID；同型号另一张卡、L40 或多个模糊匹配均不会替代原卡。7 项启动回归通过。以下命令仅用于仍可访问原卡时查询或续跑；用户随后已确认仅有 L40，当前使用本文顶部链接的新入口：

```bash
python -B tools/stage61_runtime_replay.py --list-gpus --input "$STAGE61_ROOT"
STAGE61_GPU=auto bash tools/run_stage61_search.sh
```

原卡不可见时，此命令在推理前停止并打印清单。**不能把旧 manifest 的 L20 改成 L40，也不能只跳过设备／UUID检查后混用旧L20对照与新L40轨迹。** 用户确认仅有 L40 后，已按独立协议实现新目录中全部六组同设备重测，保持原始 V1 和固定 E，无重新训练。原 L20 证据只作来源及描述性桥接。这个独立流程尚无服务器结果，自动选卡也不是跨型号迁移入口。

## 服务器运行

同步这次代码后，在原 conda 环境执行：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE60_ROOT="$PWD/output/stage60/causal_search_state_value_v1"
export STAGE61_ROOT="$PWD/output/stage61/v1_history_matched_recrop_v1"

python -B tracking/stage61_search_experiment.py --mode prepare --input "$STAGE60_ROOT" --output_dir "$STAGE61_ROOT"
python -B tools/stage61_runtime_replay.py --mode smoke --input "$STAGE61_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage61_runtime_replay.py --mode evaluate --input "$STAGE61_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage61_search_summary.py --input "$STAGE61_ROOT"
```

也可使用[一键脚本](../../../tools/run_stage61_search.sh)，顺序执行以上流程并以 `cat` 合并本轮全部 9 份分析材料及 2 份启动记录到仓库 `results` 下的 TXT；脚本不执行 commit／push：

```bash
bash tools/run_stage61_search.sh
```

主要查看文件：

```bash
cat "$STAGE61_ROOT/analysis/stage61_decision.json"
cat "$STAGE61_ROOT/analysis/stage61_policy_summary.csv"
cat "$STAGE61_ROOT/analysis/stage61_paired.csv"
cat "$STAGE61_ROOT/analysis/stage61_video_influence.csv"
cat "$STAGE61_ROOT/analysis/stage61_scope_summary.csv"
cat "$STAGE61_ROOT/analysis/stage61_proposal_summary.csv"
```

保留 manifest、run、smoke、evaluation NPZ／receipt 和所有 analysis 表；Git 同步报告不能替代服务器完整产物。原 Stage-1～60 的 Python／封存证据冻结，无 V2 安装、训练、推理或基线切换。
