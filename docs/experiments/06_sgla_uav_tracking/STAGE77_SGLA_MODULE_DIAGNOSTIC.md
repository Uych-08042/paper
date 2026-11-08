# Stage-77 SGLA 搜索与解码定位

Stage-76 的匹配 8 帧监督比当帧监督低 6.26 AUC 点，PPO 稳定保护仍未通过。Stage-77 用已保存的预测历史检查失跟前的搜索覆盖、SGLA CENTER 网格及 Hann 峰值和四候选选择。目的是为论文原型选定一个可独立消融的 SGLA 改动。 [Stage-76 结果复核](STAGE76_RESULTS_REVIEW.md)

## 固定回放

| 开发窗口 | 帧号 | 输入历史 | 状态数 |
|---|---|---|---:|
| 0304 早期 | 551–560 | 当帧监督和 8 帧监督各两个种子，另加 q0 | 50 |
| 0304 晚期 | 1076–1092 | PPO 两个种子和 q0 | 51 |
| 0003 错选对照 | 138、140 | PPO 2171、2172 各自首败帧 | 2 |
| 恢复对照 | 0176:321、0204:664 | PPO 2172 | 2 |

每个状态使用原 Stage-76 历史、同一帧图像和同一原生 SGLA 检查点重放四个搜索。逐帧核对输入历史、四个父框、裁剪坐标和裁剪像素哈希。共 105 个状态、420 次搜索、2,940 次 Transformer block 调用。每帧独立开始，原动作由 Stage-76 缓存读取。训练更新和新完整轨迹均为零。

## 如何读表

逐状态按以下顺序分类。好框定义为 IoU ≥ 0.50，严重偏差另计 IoU < 0.20。

1. `selected_good` 表示原动作在重放中仍得到好框。
2. `outer_choice_missed` 表示有其他候选输出好框。检查四候选选择的失误。
3. `decode_grid_has_good` 表示四个公开输出均无好框，至少一张 CENTER 网格里可解出好框。结合去 Hann 峰、Hann 前五和好框排名，决定是否优先改峰值解码。
4. `crop_support_limited` 表示网格无好框且四个裁剪均不满足目标中心在可用区、目标面积覆盖率 ≥ 0.90。检查搜索状态和裁剪覆盖。
5. `covered_grid_lacks_good` 表示至少一个裁剪达到覆盖条件，整组网格仍无好框。结合目标在输入中的像素尺寸及宽高误差，定位尺度表征或几何回归。覆盖率是几何量，无法确认可见性。

好框上界、原始峰与 Hann 峰对照、真值中心／尺寸替换均为离线诊断。不同分组及种子分别报告。若换卡导致公开框变化，先看 `public_support_changed_states` 和逐候选原始／重放 IoU，再决定该状态能否用来解释旧失败。此开发样本按已知事件固定，结果用于选择原型和制定消融，后续仍需完整视频闭环评价。

## 运行和回传

```bash
cd ~/workspace/paper
mkdir -p output/stage77
nohup bash tools/run_stage77_module.sh > output/stage77/run_stage77.log 2>&1 < /dev/null &
echo $!
```

在已分配 GPU 的会话中运行。默认读取 `output/stage76/matched_four_candidate_horizon_v1` 及其绑定的 Stage-74 来源，使用原生 `sglatrack_ep0297.pth.tar`。可用 `STAGE76_ROOT`、`CHECKPOINT`、`STAGE77_GPU` 覆盖路径与逻辑卡号。L20 或 L40 均可新建运行。续跑需原 GPU UUID，换物理卡时为 `STAGE77_ROOT` 指定一个新的目录。

分析表默认保存于 `output/stage77/sgla_search_decode_diagnostic_v1/analysis`。启动脚本用 `cat` 合并 `stage77_diagnostic.json`、`stage77_summary.csv`、`stage77_states.csv` 与 `stage77_candidates.csv`，结果包路径写入运行日志。

首轮服务器已打印 `COMPLETE_SGLA_MODULE_DIAGNOSTIC (105 states, 420 searches)`，随后在退出时因清理了实际不存在的 `pre_handle` 而中断打包。同步修复后，从原目录直接校验并导出已经写好的四份文件。

```bash
bash tools/run_stage77_module.sh --pack-only
```

如果原运行指定过 `STAGE77_ROOT`，打包时沿用该目录。此入口检查原 manifest、报告指纹和三张表的 SHA-256，无需 GPU。启动脚本检测到现成的诊断报告也会直接进入同一校验流程。旧运行的 manifest 绑定了原脚本哈希，直接调用 Python 回放会尝试重新生成 manifest 并被拒绝。若确实要启动新的回放，为 `STAGE77_ROOT` 指定新目录。

## 本地验证

两项 CPU 测试通过，覆盖分类、小型真实 CENTER 头、缓存续跑和来源回执篡改拒绝。根据 Stage-76 回传表核对了 105 个固定状态及五个首败锚点。Bash 语法、LF 行尾和仓库维护检查均通过。正式结果依赖服务器的完整 Stage-76 缓存及 GPU。
