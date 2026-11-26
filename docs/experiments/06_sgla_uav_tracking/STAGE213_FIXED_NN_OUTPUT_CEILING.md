# Stage-213：固定 nn 搜索状态的双输出互补与真值上界审计

本轮基于已复核的 Stage-212，检查一个必要条件：保留原生搜索状态时，两路既有输出是否足以同时保住原生能力和适配后的既有收益。全量使用 38 个开发源组、两个既有模型种子及完整视频缓存，不选取个别困难视频，不新增训练、图像读取、GPU 推理或完整轨迹，不开启确认集。

## 依据与固定来源

Stage-212 中 nn 对原生 AUC +0.6920，属于支持比较，仍丢失 4 次原生恢复及 14 次 matched 恢复。两个主要状态因素比较未建立收益，不能据此宣布解耦方法已通过。

- 来源结果：`stage212_results_20261126T153146Z_2174614.txt`。
- 原始 TXT SHA256：`45bf89388460228c66785214615e10d3a1f14b3c805c8b5199d8061f090c21f0`。
- 来源 decision 指纹：`6913f1ed1755f3b6eb9a7718e8cd217d4cb03d707d9a527981cf0e6d74205f02`。
- 来源 manifest 指纹：`dc93467f24459fff6c405ec8dcbd8106e71eb23c35ed4f89c14af0f630415561`。
- 保留原 38 源划分、2132/2133 两种子、所有跟踪帧及原始 IoU/AUC/失跟恢复定义。

先验证来源完整 Stage-212 manifest、decision、代码、协议及其内部产物哈希，再以 266 条绑定 NPZ/receipt 读取 native、matched_single、mm 和 nn。七个旧来源目录仍用于核验帧清单与标注；不调用其中的图像读取函数，不加载任何模型。需要服务器完整缓存，只有回传 TXT 无法完成本轮。

## 审计对象与数学定义

令同一个 nn 搜索状态下原生读出为 $b_t^r$，适配读出为 $b_t^a$，真值为 $y_t$。nn 的下帧搜索父框始终取 $b_t^r$。缓存中 raw 输出、父框、层号必须逐帧精确复现 fresh native；mm 必须精确复现 matched_single。

离线真值上界定义为

$$b_t^*=\arg\max_{b\in\{b_t^r,b_t^a\}}\operatorname{IoU}(b,y_t).$$

IoU 使用既有官方坐标转换与量化口径；相等时固定选 raw，不扫描新阈值。上界只改变用于计算质量的输出序列，**绝不写回下帧父框**。其 IoU 逐帧等于两路 IoU 的最大值，因此不应损失任何 raw 好帧或 raw 参考恢复。这是程序应满足的数学不变量，不是学习结果。

四个审计参照：

| 名称 | 来源与含义 |
| --- | --- |
| raw | nn 中原生输出；逐帧等于既有原生轨迹 |
| adapted | nn 中适配输出，搜索状态仍由 raw 更新 |
| mm | Stage-212 既有 matched 状态闭环轨迹，只作能力参照 |
| oracle | 在同一 nn 状态的 raw/adapted 之间按 GT 选取的离线质量上界 |

mm 经过不同搜索历史，不能混入 nn 候选集合。本轮不生成“raw/adapted/mm 三选一”的伪上界，不把其他历史上的框当作当前可获得输出。raw 在两种子行中是相同轨迹的重复参照，统计先在源内平均种子，再以 38 个源组为单位；不当作 76 个独立样本。

## 输出与解释

输出 7 张表：

1. `policy_summary`：四种参照的既有质量/稳定性指标。这里的 AUC 是缓存推导质量，oracle 不是新方法性能；不报告虚构运行时或 FPS。
2. `paired`：oracle−raw、oracle−adapted、oracle−mm、adapted−raw、mm−raw；每项含 AUC、低 IoU、原生恢复/好帧损失、matched 恢复/好帧损失，合计 30 行。
3. `seed_video_summary`：38×2×4=304 行推导质量。
4. `complementarity`：76 行。记录两路谁好、互补好帧、两路均低于 .5/.2、mm 好但两路均无好框的帧数，以及 mm 超出 oracle 的正截断质量差。
5. `recovery_events`：纳入 raw 与 mm 的全部失跟事件，含未恢复右截尾；对每个已恢复事件核对原记录恢复时刻起三帧均 ≥.5，标记 oracle 仍无法保留的恢复以及其中缺少好框的帧数。不挑固定失跟前后窗口。
6. `source_replay`：76 行，核对 raw=原生、mm=single、nn 状态未变、oracle=逐帧最大值；质量重算须复现来源逐视频表（误差 ≤1e−12）。原生 matched 相对指标在原表为跨种子均值，不误作单种子参照。
7. `source_influence`：五个 AUC 比较的 190 行源组差与逐源剔除结果。

使用既有 20,000 次源组 bootstrap、原固定随机种子，区间只作探索性描述。本轮无自动采用门槛，也不把 oracle 名义正区间解释成可部署选择器已有效。模型/代码/协议/NPZ 元数据和 receipt 均绑定；已有产物不同则拒绝覆盖，改用新版本目录。缓存重用后仍重算上界数组并验证等式，完整结果的 `--check-results` 校验封存哈希。

若理想两路选择也无法保留 mm 的重要恢复或既有收益，说明固定原生搜索状态下的两路供给不足，单独训练输出选择器不能补齐这些框；下一阶段应检验可学习状态更新及其保护机制。若上界在多个源组具备互补且能力可保留，再考虑仅使用在线信息的输出质量学习，并做独立完整轨迹验证。两种情况都不自动转向 PPO 或新阈值扫描。

## 运行

服务器仓库内，使用原 `sglatrack-test` 环境即可；本轮不占 GPU，无需声明 L20/L40，也不新增双进程推理。记录本次 CPU Python/NumPy 版本，原 GPU 身份作为历史来源保留。

```bash
bash tools/run_stage213_output_ceiling.sh
```

后台运行与总体进度：

```bash
mkdir -p output/stage213
nohup bash tools/run_stage213_output_ceiling.sh > output/stage213/run.log 2>&1 < /dev/null &
echo $! > output/stage213/run.pid
tail -f output/stage213/run.log
```

默认输出 `output/stage213/fixed_nn_output_ceiling_v1`，自动将 decision、manifest 和七张 CSV 打包成时间戳 TXT。若旧来源目录不同，沿用 `STAGE76_ROOT` 等变量；212 非默认路径时设置 `STAGE212_ROOT`。全量验证来源缓存后开始一个总体进度条，不逐视频刷屏。

```bash
python -B tools/stage213_output_ceiling.py --check-results \
  --output-dir output/stage213/fixed_nn_output_ceiling_v1
```

本地合成缓存测试仅验证实现与指标，不构成真实开发视频结果。真实审计须在拥有完整缓存的服务器运行。
