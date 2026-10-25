# Stage-35：保留可靠定位能力的条件纠错蒸馏

## 这次检验什么

Stage-34 的主方法相对普通表征训练平均增加 1.1163 AUC，但区间跨零；其中一个视频增加 9.1716，另两个原本稳定的视频相对 Fast 分别下降 4.3557 和 1.4401。这支持检查“训练纠错时是否损伤了可靠定位能力”，尚不能确定遗忘就是原因。原 Stage-34 结论保持不变。

Stage-35 检验一条训练机制假设：**可靠状态保持教师的定位函数；教师失误状态使用 GT 纠正空间竞争，同时保留单路径推理。** 暂不回到 PPO、选层、候选排序、动态阈值或新记忆模块。

条件蒸馏和困难负样本监督本身不构成已证明的新方法。只有它们的作用可分辨、完整视频收益和保留能力同时成立，才有依据进一步发展压缩方法。此轮是开发筛选，不能作为最终创新或外部泛化结论。

## 学生、监督和匹配对照

学生从**原始 SGLA 权重**初始化，物理保留 L1–L6，删除 L7–L12 和选层 MLP；不加载 Stage-34 训练后的权重，也不保留身份读出。只训练 L5–L6。原 patch embedding、位置编码、L1–L4、最终 norm 和 CENTER 头冻结，全部保持 eval 模式。

推理只调用一次原定位头、使用原窗口和解码器。没有教师调用、GT 门控、额外身份图、候选排序或跨帧描述子银行。初始输出必须与 Fast 等价。

| 组别 | 所有训练状态 | 教师可靠状态 | 教师失误状态 |
|---|---|---|---|
| `supervised` | 原 GT 定位损失 | — | — |
| `preserve` | 原 GT 定位损失 | 保留蒸馏 | — |
| `correct` | 原 GT 定位损失 | — | 困难空间负样本纠错 |
| `preserve_correct`（主方法） | 原 GT 定位损失 | 保留蒸馏 | 困难空间负样本纠错 |
| `distill_all_correct`（普通蒸馏对照） | 原 GT 定位损失＋所有状态均蒸馏 | 同样蒸馏 | 同样蒸馏＋困难空间负样本纠错 |

前四组构成 2×2；额外的普通蒸馏组与主方法只差“是否在教师失误状态也模仿教师”，避免把普通 KD 的效果归因于条件保留机制。

可靠/失误标签使用 Stage-34 已冻结训练样本的 `easy`/`hard`：原 SGLA 在其实际预测裁剪中的输出 IoU≥0.5 / <0.5。它们是**仅用于离线训练的 GT 标签**，不是可部署的置信度判断。五组都计算同一个冻结教师前向和所有损失，只对指定项回传梯度；同折同 seed 初始化、训练帧、采样顺序、优化器、预算完全一致。

- 原定位损失：2×GIoU＋5×L1＋focal，与原 CENTER 量化中心监督一致。
- 保留项：将教师和学生响应分别乘原推理窗口，取对数、温度 2，计算空间 KL；再在**教师窗口选中的同一格**蒸馏归一化框（5×L1）。总保留权重 1。教师输出 detach；不能直接以可能定位错误的 raw argmax 当可靠输出。
- 纠错项：直接作用于原响应 logit，不添加推理模块。训练 GT 框内和原 CENTER 中心格为正位置；目标 1.5 倍区域及正位置一格邻域之外，按 detach 后的响应矿选最多 8 个负位置。softplus 排序 margin=0.2，权重 0.25。几何负样本不能自动解释成语义干扰物。

## 数据、OOF 和预算

复用原 Stage-31 的 **8 个 fit 源组**和 Stage-34 的冻结训练样本/裁剪父框。暂不选择新的 checking 视频，不打开 Stage-31/34 checking 图像或标注，也不使用 OTB、UAV123 或此前 tune。

按源组名和固定 seed2110 分成 4 折，每折 6 组训练、2 组完整视频 OOF。划分不根据任何视频成绩挑选。样本均衡权重只在该折 6 个训练组内计算：easy/hard 各 50%，组内按视频、帧均衡。每折每类至少 16 帧、来自至少 2 视频，否则 `INSUFFICIENT_FOLD_SUPPORT`，不放宽门槛或重新挑折。

固定 seeds=2110、2111；4 折×2 seeds×5 组＝**40 个模型**。每模型 AdamW 256 次更新、batch=2、lr=1e-4、weight decay=1e-4、梯度裁剪 1。没有早停、最佳 checkpoint、种子筛选或检查集调参。smoke 只做首折、seed2110、五组各 8 次更新。

所有 40 个模型和哈希先冻结，才允许 OOF 预测；每个视频只能使用未训练过该视频的模型，从首帧独立运行到结尾，以自己的预测生成下一帧裁剪。GT 仅用于首框初始化，未来标注在整段推理结束后读取并计分。

8 视频×（Fast、SGLA、五组×两 seeds）＝**96 个完整轨迹文件**。原 backbone 可能接触过这些数据，且这 8 组已参与方法开发，因此这种 OOF 仍是已消费开发数据上的筛选，绝非 untouched 外部验证。

## 冻结通过门槛

先在每视频内平均两 seeds，再以 8 个源组为单位 bootstrap 5000 次，seed2112。不能把帧或重复种子当独立视频。以下为同时满足的开发筛选条件，名义 95% 区间不作确认性显著性宣称；折间训练集重叠，区间也不包含重新训练的不确定性。

1. 主方法完整视频 AUC 相对 Fast 均值至少 +0.3 点，配对区间下界>0，两 seeds 各自均值均>0；相对 `supervised` 的均值也>0；相对 `distill_all_correct` 的配对区间下界>0。
2. 在固定 **Fast 原本 IoU≥0.5 的时间点**，主方法相对 `correct` 的条件 AUC 配对下界>0，用于识别保留项的作用。
3. 在固定 **SGLA 原本 IoU<0.5 的时间点**，主方法相对 `preserve` 的条件 AUC 配对下界>0，用于识别纠错项的作用。
4. 正常定位保留：Fast 可靠时间点上，主方法相对 Fast 的宏平均差异区间下界≥−0.5 AUC 点，且最差单视频的 seed 平均差异≥−1.0 点。它是预定的实际容忍界，不是“完全无损”的证明。
5. 主方法宏平均 IoU<0.2 帧占比不增加；Fast 原本无持续失跟的视频，任一 seed 都不能引入新持续失跟。至少有 1 个 Fast 稳定视频，否则此项不可评估。

条件队列中每视频至少需 10 帧，且每种条件至少支持 4 视频。支持不足报告 `INSUFFICIENT_OOF_SUPPORT`，不能删掉困难视频后宣称通过。条件掩码完全由两条基线轨迹固定，各方法仍运行/计入所有帧；条件比较是结果分层诊断，不是线上触发规则。2×2 交互和相对 SGLA 的结果另外报告，不把“较弱监督受损后恢复”当作超过原模型。

只有全部满足才是 `PASS_PRESERVE_CORRECT_OOF_SCREEN`；否则 `NO_PRESERVE_CORRECT_GAIN_ESTABLISHED`。失败后保留结果，不在这些 OOF 结果上搜索权重、步数、折、阈值或 checkpoint。

## 服务器执行

通过 Git 同步新增文件。无需下载新数据。仍用现有 `sglatrack-test` 环境；以下命令在仓库根目录执行。

```bash
export STAGE34_ROOT="$PWD/output/stage34/spatial_identity_student_v1"
export STAGE35_ROOT="$PWD/output/stage35/preserve_correct_oof_v1"
# CHECKPOINT 必须为原始 SGLA 权重，不能指向 Stage-34 学生。
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

python tracking/test_stage35_preservation.py

python tracking/stage35_preserve_experiment.py \
  --mode prepare --input "${STAGE34_ROOT:?}" \
  --output_dir "${STAGE35_ROOT:?}"

python tracking/stage35_preserve_experiment.py \
  --mode smoke --input "${STAGE35_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0

cat "$STAGE35_ROOT/smoke/stage35_training_decision.json"
```

prepare 不用 GPU，只读取旧计划、已完成 Stage-34 decision 和 fit 来源，不读旧 checking 预测缓存或原图。代码/来源、折、样本、门槛写入新 `manifest.json`。只有 `PASS_CORRECTIVE_DISTILLATION_SMOKE` 才允许完整训练。smoke 通过只建立前向/梯度/冻结正确性，尚无精度结论。

smoke 通过后，两个 seeds 在同一 GPU 上依次完成四折训练：

```bash
for SEED in 2110 2111; do
  python tracking/stage35_preserve_experiment.py \
    --mode train --input "${STAGE35_ROOT:?}" \
    --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0 \
    --seed "$SEED" || break
done

python tracking/stage35_preserve_experiment.py \
  --mode status --input "${STAGE35_ROOT:?}"
```

确认两 seeds 均为 `COMPLETE_MATCHED_OOF_TRAINING` 后：

```bash
python tracking/stage35_preserve_experiment.py \
  --mode evaluate --input "${STAGE35_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" --parameter deit_distilled --gpu 0

python tracking/stage35_preserve_summary.py --input "${STAGE35_ROOT:?}"

cat "$STAGE35_ROOT/analysis/stage35_decision.json"
cat "$STAGE35_ROOT/analysis/stage35_policy_summary.csv"
cat "$STAGE35_ROOT/analysis/stage35_paired.csv"
```

## 中断、结果与边界

每组每 128 更新以及最后一步原子保存参数、Adam 状态和训练诊断，并用独立 JSON receipt 绑定 checkpoint SHA256。中断后重新执行**相同命令/目录**，会从最近完整 receipt 续跑；未提交的更新重算。整组已完成则审计后复用，不覆盖已有 checkpoint。CPU 测试包含真实中断恢复后逐张量一致性验证。Ctrl+C 正常退出会释放锁；kill -9 等异常终止可能留下 `collect.lock`，需先核对其中 PID 并确认原任务已终止再处理残留锁。不要删锁绕过正在运行的进程，训练也不要对同目录并发。

`--mode status` 无需 GPU，仅显示已提交训练步数/轨迹数。训练 loss 和 smoke 状态都不是精度收益；精度门槛只在 96 个完整 OOF 文件齐全后计算。evaluate 以整视频为单位恢复；中途单视频需重算，完整且元数据一致的缓存复用。

输出报告的 `instrumented_ms` 是带同步的 crop/model/decode 墙钟时间，排除图像读取、标注记录。物理参数量减少不能直接当实际加速，也不支持边缘端能耗结论。若此轮通过，再冻结可行机制，并预定真正未用过的数据与硬件计时验证；不把本轮重新包装成最终验证。
