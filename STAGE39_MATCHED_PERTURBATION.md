# Stage-39：固定预测父框扰动下的压缩敏感性验证

Stage-38 没有建立 size-only 或 offset-only 替换的稳定收益；全部几何的正均值也依赖少数视频。本轮回到方法前提：**去掉 SGLA 的候选计算和选择器之后，六层模型是否更容易把同样的父框误差延续为未来跟踪损失？**

这是机制小实验，不训练新方法。它与普通序列训练、学生自身轨迹训练或蒸馏不是同一个实验问题，但也不单独构成创新方法。

## 匹配设计

使用原8个已消耗开发视频，固定三个时间锚点；不增加 checking 数据，不根据失跟或模型输赢选帧。若共有 N 张图、分支长20帧，锚点图索引为 `floor((N−20)×q/4)`，q=1、2、3，第一张初始化图索引为0。

每个锚点都读取**冻结原生 SGLA 在该帧之前的预测父框**作为共同起点。

| 模型 | 本轮作用 | 与原 SGLA 的关系 |
|---|---|---|
| sgla | 原生参照 | 前六层＋原生选择的一层 L7–L12，保留选择器 |
| fast_initial | 主要压缩对照 | 物理删除 L7–L12 和选择器，六层与 head 的权重逐项相同，无新训练 |
| supervised_seed2110 | 补充微调对照 | 复用原 Stage-35 对应 OOF 折的六层学生 |
| supervised_seed2111 | 补充微调对照 | 复用另一 seed 的同折学生 |

SGLA 每帧实际执行七个 Transformer block，不把它称为十二层顺序教师。主要比较是当前部署结构的精确截断，包含候选计算和选择器的共同移除，不能据此分离二者或宣称一般压缩理论。两份微调学生不属于“新训练匹配”的主要比较。

在共同预测父框 `(x,y,w,h)` 上固定七个条件：

- 无扰动。
- 水平正／负平移、垂直正／负平移：位移均为 `0.15×sqrt(w×h)`。
- 保持中心不变，将宽高同时乘0.8或1.25。

四个模型接受完全相同的初始父框干预。父框本身不额外裁到图像边界；原生裁剪函数照常补边，输出仍使用原生边界裁剪。整个过程不使用 GT 大小或位置设计扰动。

**只在分支第一帧施加一次干预。之后19帧都使用该分支自己的上一帧输出。** 不每帧重置，不返回源轨迹，不使用 GT 修正。每个分支重新使用原视频第一帧模板和初始框初始化，避免上一个分支的状态串入。

全部为 `8×3×4×7=672` 条20帧短分支，合计13,440个跟踪帧。重复锚点、方向、seed、帧都不是额外独立视频。

## 主要指标与预先冻结的门槛

对同一模型，定义后续损失：

`D(model) = mean(IoU_clean − IoU_perturbed)，offset=2..20`

第一帧单独报告，避免把第一帧裁剪的直接影响全部计为持续损失。第2至20帧仍包括持续输入影响、模型误差和裁剪反馈，不能解释为“纯反馈效应”。

主要比较为：

`E = D(fast_initial) − D(sgla)`

正值表示六层截断对同样干预出现更多后续 IoU 损失。它不同于 Fast 与 SGLA 的绝对精度差。损失允许为负，不截零。

先在每视频内等权平均六个扰动和三个锚点，再做5000次视频配对 bootstrap，seed2118。以下四条须同时满足，才记为当前开发数据支持额外敏感性：

1. E 的视频均值≥0.02 IoU。
2. 名义95%配对区间下界>0。
3. 至少6／8个视频 E>0。
4. 去掉任一视频后的 E 均值仍>0。

这些是预算有限的探索门槛，不是外部确认或普适显著性结论。bootstrap 条件于固定模型／视频与此前重叠 OOF 训练。不得依据结果更改幅度、方向、锚点或排除视频。

必须同时查看无扰动／有扰动绝对 IoU：原模型已经失跟时，IoU 地板可能使它表现出较小的“额外损失”。该门槛本身不能消除地板或共同起点分布的影响。两份微调学生的对比另列，不能把微调差异直接归于压缩。

## 服务器命令

通过 Git 同步新增文件后，在仓库根目录执行：

```bash
export STAGE38_ROOT="${STAGE38_ROOT:-$PWD/output/stage38/fixed_response_geometry_v1}"
export STAGE39_ROOT="$PWD/output/stage39/matched_parent_perturbation_v1"

python tracking/test_stage39_perturbation.py

python tracking/stage39_perturb_experiment.py \
  --input "$STAGE38_ROOT" --mode prepare \
  --output_dir "$STAGE39_ROOT"

python tracking/stage39_perturb_experiment.py \
  --input "$STAGE39_ROOT" --mode smoke \
  --parameter deit_distilled --gpu 0

python tracking/stage39_perturb_experiment.py \
  --input "$STAGE39_ROOT" --mode evaluate \
  --parameter deit_distilled --gpu 0

python tracking/stage39_perturb_summary.py --input "$STAGE39_ROOT"
```

权重查找顺序：`--checkpoint`、环境变量 `CHECKPOINT`、原 tracker 配置；会核对原 SGLA 权重哈希。两份 OOF 学生自动从旧结果恢复。prepare/status/summary 不需要加载 PyTorch 或 GPU，但原数据／权重文件／结果须保留以验证来源。

smoke 只运行第一视频的第一固定锚点：4模型×7条件×4帧=112帧。`PASS_MATCHED_PARENT_SMOKE` 表示执行完整性通过，不代表科学前提通过。

原生 SGLA 无扰动分支必须复现冻结旧轨迹（框≤0.001像素、整数框 IoU≤1e-12、选择层一致）；每个 Fast 条件的首帧还会核验物理六层模型与原始 Fast 计算的三个输出图一致。全部分支检查父框连续性、head 调用数、原生／密集解码、输入与模型不可变。当前分支包的评价标注在28条分支完成后读取；旧轨迹来源审计可能先读取旧标注，但未来标注不进入分支控制器。

## 中断与结果

重复相同命令可以继续。提交单位是**一个锚点的28条完整分支**；已提交锚点经审计后复用，中断时当前未完成锚点重新计算。不在同一输出目录并发运行 collector。

```bash
python tracking/stage39_perturb_experiment.py \
  --input "$STAGE39_ROOT" --mode status

cat "$STAGE39_ROOT/analysis/stage39_model_summary.csv"
cat "$STAGE39_ROOT/analysis/stage39_paired.csv"
cat "$STAGE39_ROOT/analysis/stage39_primary_video_influence.csv"
cat "$STAGE39_ROOT/analysis/stage39_decision.json"
```

full 应为24／24个锚点包。可加 `evaluate --sequence <原8条之一>` 分次运行，汇总要求全部齐全。保存目录 `full/branches/` 中每包含全部原始响应、宽高、偏移图和分支父框，附密封哈希收据。无 GPU 的汇总支持重复校验复用，拒绝不完整或被改写的来源／结果。

如果主要门槛不通过，状态为 `NO_EXCESS_SENSITIVITY_ESTABLISHED`；应保留结果，不能据此说所有压缩模型都没有闭环问题。如果通过，状态为 `EXCESS_SENSITIVITY_SUPPORTED_ON_CONSUMED_DEVELOPMENT`，才有理由围绕这一现象设计区别于普通监督、KD和序列训练的机制，并用匹配训练对照及新开发组再验证。

本轮不提供完整视频 AUC、不估计自然失败发生率、不证明 UAV 场景特异性、不训练新策略，也不测试部署加速。四个模型从 SGLA 共同父框开始，是相同状态下的压力实验，不能替代各自自然轨迹评价。以上边界在决策 JSON 中一并记录。
