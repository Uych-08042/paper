# Stage-61：L40 上的六组完整重测

日期：2026-11-02。用户确认目前只能使用 NVIDIA L40；原来源运行记录为 NVIDIA L20。接续[原 Stage-61 协议](STAGE61_NATIVE_HISTORY_MATCHED_RECROP.md)，研究问题、模型、候选、选择与统计规则均沿用原方案。此文件在收到任何 L40 推理结果前声明设备迁移及额外计算。

## 设备迁移与证据边界

原 Stage-61 复用 L20 上的完整对照，而新运行选中了 L40，因身份不符在图像推理前退出。既有 L20 manifest、代码、权重、报告和轨迹均保留。新的输出为 `output/stage61/v1_history_matched_recrop_l40_v1/`，独立 schema 为 `stage61_l40_device_matched_recrop_v1`。

新 manifest 同时记录原始来源运行身份与实际 L40 运行身份，明确只变更 GPU 型号／UUID。配置快照、checkpoint SHA、模型状态摘要、Torch／CUDA、OpenCV／NumPy、网格维度及全部 E 权重必须精确符合原来源；resume 也必须匹配新封存设备。记录实际 L40 身份不等于把原 L20 来源字段改写成 L40。

**全部主要对比使用新 L40 完整轨迹。** 原 L20 数据仅用于来源验证与单独的描述性 `device_bridge` 表；旧新运行的 AUC 差不能作为方法增益或纯粹的硬件因果效应。新 L40 原生 current 轨迹定义本次可靠帧、失败帧及原稳定视频队列。

已有 E 模型的训练来源不变：沿用原折、原 held-out 配置、种子 2132／2133、Stage-54 rollout_preserve／Stage-58 endpoint_control 的既有权重；不重新训练、选 checkpoint、改学习率或新增 scorer。原生 V1 每视频一个确定性运行，E 两种子先在视频内平均。仍为反复使用的 12 个开发源组，不成为新的独立外部验证。

## 六组政策与预算

| 组别 | 网络 | 每帧搜索 | 本次轨迹来源 |
|---|---|---:|---|
| native_current | 原始 V1 | 1 | 新 L40 完整推理 |
| native_history | 原始 V1 | 4 | 新 L40 完整推理 |
| native_recrop | 原始 V1 | 4 | 新 L40 完整推理 |
| endpoint_current | 固定 E | 1 | 新 L40 完整推理，两个原种子 |
| endpoint_history | 固定 E | 4 | 新 L40 完整推理，两个原种子 |
| endpoint_recrop | 固定 E | 4 | 新 L40 完整推理，两个原种子 |

历史与重裁剪严格使用原 Stage-61 actor。历史为当前／滞后中心×当前／近 5 框 log 中位尺寸；重裁剪为上一已接受框 B 与当帧第一次解码 D 的中心×尺寸四组合。q0 结果复用，只额外执行三次，总计四次。沿用原 float32 Hann 峰值与 1e−6 并列规则，只接收所选原始输出。模板、原始骨干、原生独立候选及选择器均不更新。

按既有 13,525 跟踪帧：

- 原生三组：36 条完整轨迹、121,725 次搜索。
- E 三组×两种子：72 条完整轨迹、243,450 次搜索。
- 正式共 **108 条完整轨迹、365,175 次搜索、2,556,225 次 block**；复用完整评价轨迹数为 0，新增模型数为 0。
- 首视频前 32 帧，原生及两个 E 种子×三政策 smoke 为 864 次搜索；另独立执行三条原生公共 current 路径各 32 帧，共 96 次。smoke 合计 **960 次搜索**。
- 正式＋smoke 共 **366,135 次搜索**。模板初始化不计入搜索数，无额外仅头前向。

同一视频内的九个运行单元按视频编号轮换执行顺序。每项 NPZ／密封回执绑定新计划指纹、实际运行身份指纹、网络来源及模型摘要；重复启动只审计已提交项，不重复调用网络。旧 L20 receipt 不能移入新目录替代本次控制。

## smoke 与统计判读

独立公共 current 路径直接调用原 `NativeObserver.step`，逐帧提交自身输出，不调用候选构造／选择辅助函数。原生及两个 E 种子的 current actor 前缀须与这一路径的框和所选层精确一致；同时验证解码、每次六前缀＋一个独立候选、选择器及头调用数。四候选 actor 原有的状态、裁剪、分数和调用审计继续执行。

smoke 不要求跨 L20／L40 的框逐位相同。与旧来源前缀的差异记录为描述性诊断，不用于调参或按结果挑选卡。正式的全部六组由同一个已核验的 L40 运行环境生成，所有新计时均来自本次运行，排除读图／哈希、初始化及前十帧；不据此声称部署加速。

主比较仍为 `native_history − native_recrop`；必要比较仍为 `native_history − native_current`。沿用原视频 bootstrap 种子、20,000 次重采样、两比较 Bonferroni 家族区间、+0.5 AUC 实践目标、可靠帧下界≥−1.0、低 IoU 不升、无新原稳定失效及非平凡使用要求。均通过才标 `PROMISING_L40_NATIVE_HISTORY_RECROP_PILOT`，其余标 `REVIEW_L40_NATIVE_HISTORY_RECROP_PILOT`。

E 内比较和来源×候选交互仍为次要描述。设备迁移不改变研究假设，不引入 V2、训练或新的阈值搜索，也不追认 Stage-60 为 PASS。

## 服务器入口

同步本次新增的 `tools/stage61_l40_*.py`、`tools/test_stage61_l40.py`、`tools/run_stage61_l40.sh` 及更新后的 `tools/stage61_runtime_replay.py`。保留原 Stage-61 目录中的 manifest 与完整 Stage-60 来源链；原 Stage-61 的 smoke 未完成不影响读取它的已封存计划。进入原 conda 环境：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE61_ROOT="$PWD/output/stage61/v1_history_matched_recrop_v1"
export STAGE61_L40_ROOT="$PWD/output/stage61/v1_history_matched_recrop_l40_v1"
bash tools/run_stage61_l40.sh
```

默认使用 CUDA 逻辑 GPU0；如 L40 位于其他逻辑编号，设置 `STAGE61_L40_GPU`。prepare 只读设备信息与来源文件，封存新 L40 身份；smoke／evaluate 仍用严格身份入口，summary 只读完整缓存，不需要 GPU。

单独执行顺序为：

```bash
python -B tools/stage61_l40_experiment.py --mode prepare --input "$STAGE61_ROOT" --output_dir "$STAGE61_L40_ROOT" --gpu 0
python -B tools/stage61_l40_experiment.py --mode smoke --input "$STAGE61_L40_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage61_l40_experiment.py --mode evaluate --input "$STAGE61_L40_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tools/stage61_l40_summary.py --input "$STAGE61_L40_ROOT"
```

一键脚本使用 `cat` 合并 decision、原八张分析表、新 device_bridge 表及两份启动身份记录，写入 `docs/experiments/06_sgla_uav_tracking/results/stage61_l40_results_<UTC>_<pid>.txt`，不执行 Git commit／push。排查启动时读取：

```bash
cat "$STAGE61_L40_ROOT/diagnostics/stage61_runtime_smoke.json"
cat "$STAGE61_L40_ROOT/analysis/stage61_l40_decision.json"
cat "$STAGE61_L40_ROOT/analysis/stage61_l40_policy_summary.csv"
cat "$STAGE61_L40_ROOT/analysis/stage61_l40_device_bridge.csv"
```

本机 3 项 L40 测试全部通过，23.837 秒、无跳过：硬件迁移只接受 L40 且保持源软件；旧 L20 schema／receipt 不能填入新缓存；真实小通道 CENTER 和原公共解码的 Stage-59→60→61 来源流程、六组新推理、三条独立 public smoke、总预算、源文件逐字节不变、恢复不重复前向、缓存／源／报告篡改拒绝。主动改变旧 L20 数组后，设备桥接表改变，主要／必要比较、交互与保护结论保持不变。fixture 的上游 Stage-58 边界、骨干与图像为合成构造，设备绑定使用 fixture，不能当作真实 GPU 复现。

另有 7 项启动回归通过；新 CLI help、AST、Bash 只读 `-n`、LF 换行和维护检查均通过。本机 CPU 合成验证与服务器真实精度须分开记录。当前没有 L40 的真实 smoke 或 AUC 结果。
