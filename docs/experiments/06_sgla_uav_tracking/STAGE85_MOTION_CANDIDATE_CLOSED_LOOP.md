# Stage-85 运动生成四候选完整闭环

Stage-84 的同历史回放通过候选供给门槛，最佳候选 IoU 增加 0.00179。逐状态新增 107 帧好框、丢失 110 帧好框。本轮检验冻结 PPO 能否用上运动候选，并把候选变化转化为完整轨迹收益。生成器逐字节绑定 Stage-84 已通过运行中的源码与设计。 [Stage-84 结果](STAGE84_RESULTS_REVIEW.md)

| 臂 | 布局 | 选框 | 搜索／帧 |
|---|---|---|---:|
| 原生 SGLA | 最新预测框 | 原生输出 | 1 |
| 固定四候选 PPO | 原始中心 × 尺寸 | Stage-74 冻结 PPO | 4 |
| 运动生成四候选 PPO | Stage-84 生成器 | 同折、同种子、同一冻结 PPO | 4 |

三臂同卡、同一 SGLA 检查点、同一首帧模板和原生解码。每臂从首帧初始化，每帧用自身选出的框推进。运动生成规则保留 q0 与各槽位宽高，其余中心按 Stage-84 的运动预测与互补覆盖规则生成。前 3 个跟踪帧两种 PPO 布局相同，必须逐框、逐动作复现。两种 PPO 都执行四次完整搜索，即使同帧父框重复。确认集清单仅检查分组互斥与绑定指纹。

12 个开发视频，PPO 两个既有种子，原生单轨迹，共 60 条完整轨迹、229,925 次搜索和 1,609,475 次 block 调用。PPO、SGLA 权重和标准化参数冻结。与 Stage-76 的基线框和动作误差单独记录，正式比较使用 Stage-85 的同卡结果。

## 判读

主比较为运动生成 PPO 减固定 PPO。种子先在视频内平均，视频等权。通过要求 AUC 至少 +0.5 点、95% 视频配对 bootstrap 区间下界大于零、低 IoU 和最长失跟比例均不增加，以及相对原生稳定视频零新增失跟。报告可靠帧 AUC、首次失跟、恢复和未恢复事件。

同时报告最佳候选 IoU、选中框 IoU、两者差、好框支持率和漏选率。好框固定为 IoU ≥ 0.5，漏选率为所有状态中有好框但未选到的比例。这里的候选与选框指标来自各臂自己的轨迹。两臂历史分叉后，这些指标可以描述闭环后果，同历史的候选供给差由 Stage-84 提供。确认集推理为零。

## 运行

同步 Stage-85 新文件，在服务器的 `sglatrack-test` 环境运行。支持已分配的 L20 或 L40。

```bash
cd ~/workspace/paper
mkdir -p output/stage85
bash -n tools/run_stage85_motion_closed_loop.sh && {
  nohup bash tools/run_stage85_motion_closed_loop.sh > output/stage85/run_stage85.log 2>&1 < /dev/null &
}
```

可用 `STAGE76_ROOT`、`STAGE84_ROOT`、`STAGE85_ROOT`、`STAGE85_GPU` 和 `CHECKPOINT` 覆盖默认路径或设备。保留 Stage-76、其 Stage-74 来源和 Stage-84 全套目录，回传 TXT 无法替代这些来源。相同指纹与物理 GPU 支持从逐轨迹收据续跑，换卡使用新输出目录。完成后用 `cat output/stage85/run_stage85.log` 查看回传包路径，回传 `stage85_results_*.txt`。已完成目录可用 `--pack-only` 校验并重新打包。
