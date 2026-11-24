# Stage-178 选择器 Sigmoid 前分数审计

Stage-177 的 243 个状态均选择 L9，Sigmoid 输出的前两名分差均为 1.0。Stage-178 检查饱和前分数还有多少变化，以及这些分数与既有换层后效的关系。

## 单帧测量

复用 Stage-177 的 243 个扫描状态和 Stage-176 的 26 个六层回放状态。合并四个重复状态后，对 265 个起点各执行一次原生前向。只读 `backbone.MLP.fc2` 的输出和 MLP 的 Sigmoid 输出。逐点核对原前向的 Sigmoid 结果、无额外裁剪或预测头调用、以及已有归档的原生层、父框、预测框与 Hann 峰值。原路由沿用 Sigmoid 输出，全部后效标签取自原归档。

## 诊断指标

记录六层的两种分数、原生层与其他层最高分的差、排序推荐层相对原生层的分差。先读 `stage178_dispersion.csv`，确认分数有无变化。接着用完整 128 帧的归档后效比较两类关系。跨起点时在同源组内比较，使用原生分差或排序层相对分差。同起点时只比较不同非原生候选的分数和后效。较高的候选相对分或较小的原生领先幅度对应更有可能有益，反向用于判断有害。支持不足时保持关联指标为空，列出可比较对数和覆盖源组。Stage-177 的四个已知点单列。此轮不拟合阈值，也不新增续跑。

只有与原归档运行身份相同才计算主关联。设备不同可检查分数分散程度及原生框偏差，归档后效仅供追溯。即使分差有明显变化，也需先有多源组的保护标签支持，才值得做预算匹配的在线信号验证。

## 运行

沿用 Stage-170、176 与 177 的输出目录和原始 checkpoint。更换 GPU 运行身份时使用独立 `STAGE178_ROOT`。

```bash
cd ~/workspace/paper
mkdir -p output/stage178
nohup bash tools/run_stage178_selector_logits_audit.sh > output/stage178/run_stage178.log 2>&1 < /dev/null &
tail -f output/stage178/run_stage178.log
```

脚本显示总体进度并生成 `stage178_results_*.txt`。先看 `stage178_dispersion.csv`、`stage178_association.csv` 与 `stage178_sentinels.csv`。
