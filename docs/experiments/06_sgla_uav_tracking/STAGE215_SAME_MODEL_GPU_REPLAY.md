# Stage-215：同型号不同物理 GPU 的独立重放

日期：2026-11-28。原 Stage-214 绑定 NVIDIA L20 / UUID `48862e47-a2db-e143-d434-ac5ae4f149d4`；服务器现仅可见同型号 L20 / UUID `75aa7472-dbcf-3dfa-c22b-a1b755c44605`。原卡不可见，不删除或改写来源绑定，不放宽原 Stage-215 的续跑限制。

## 运行身份与来源

新增独立入口 `tools/stage215_same_model_replay.py`，保留原 Stage-215 的四个 mask、八个更新器、四个适配器、38开发源、两折和两个种子。0训练，不修改目标或阈值，不开放确认组。引用的原协议与既有来源代码/模型/数据哈希仍须精确匹配。

新 manifest schema 为 `stage215_same_model_gpu_replay_v1`，同时记录来源 runtime、新绑定 runtime 及逐字段差异。仅允许同一受支持型号 L20→L20 或 L40→L40 的 GPU UUID 差异；型号、软件、权重、配置等其他 runtime 字段变更均拒绝。后续每个 worker 仍检查实际运行环境，续跑必须使用该独立目录已经绑定的同一 GPU/runtime。

默认新目录为 `output/stage215/same_model_gpu_replay_v1`。拒绝原默认目录 `frozen_center_scale_v1`、其他 schema 的既有 manifest 及没有本方案 manifest 的非空目录。旧 Stage-215 与 Stage-214 缓存不移动、不修改、不作为本次轨迹复用。进程锁继续生效，不自动清除锁。

## 参照、预算与一致性门槛

- 608条因素轨迹仍全部重新计算；另外新卡重跑 native 38条、mm 76条，合计 **722条新完整轨迹、0条复用评价轨迹**。冻结模型仍绑定旧来源，读取历史轨迹只用于一致性核验。
- 串行/双spawn短轨迹由原32对增加为 **38对**（两个固定源，每源19种运行）；两种执行须精确一致。进入完整评价前，短轨迹 native/mm 与 none/joint 端点须复现旧来源相同前缀，排除 pipeline_ms，因素端点另排除新增的 updater_calls。
- 完整轨迹中，114条新参照及304条因素端点也必须逐数组精确复现旧来源的非计时内容。任何不一致即停止，不扩大容差，不以相近AUC替代。
- 38源总跟踪帧40517，完整重放769823帧；短轨迹608帧，加最终封存一步，总进度 **770432**。单总体进度条；默认两个worker。不声称双worker的独立FPS或硬件加速收益。
- 10张表的格式与原Stage-215一致。smoke38行、source_replay418行、seed_video/calls各722行、policy10行、paired132行、factor36行、search_geometry684行；source_replay的114条参照必须标为fresh。同卡新参照用于本次主统计，旧卡轨迹仅作桥接证据。

唯一主要比较继续是 coverage只中心−联合 AUC；其他比较仍为支持分析，不因新卡或更多参照更换主比较。保持REVIEW，不自动采用。

## 运行

```bash
cd ~/workspace/paper
mkdir -p output/stage215
LOG="output/stage215/replay_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage215_same_model_replay.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage215/replay.pid
echo "$LOG"
tail -f "$LOG"
```

与原入口相同，继承 STAGE214_ROOT、STAGE208_ROOT、CHECKPOINT 等来源变量，GPU逻辑编号由 STAGE215_GPU（默认0）控制。独立输出变量为 **STAGE215_REPLAY_ROOT**，不继承原 STAGE215_ROOT。如本次新卡之后再次改变，必须使用另一新目录重新绑定，并重新执行同样的参照和端点检查。

完成后自动汇总10表、manifest及decision到新目录的时间戳TXT。只核验结果：

```bash
python -B tools/stage215_same_model_replay.py --check-results \
  --output-dir output/stage215/same_model_gpu_replay_v1
```

本机CPU合成测试仅验证实现与拒绝逻辑；未在新L20进行真实推理，不能声称桥接已经通过或已有新AUC。
