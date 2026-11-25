# Stage-196 同预算的路由损失对照

日期为2026-11-25。本文保留运行前协议，真实缓存训练与复核已完成，结果见[Stage-196复核](STAGE196_RESULTS_REVIEW.md)。

## 根据与问题

Stage-195发现，R-only四模型的实际抽样MSE优于初始化和抽样常数，但下降的99.15%～99.97%来自共同质量项；留出逐帧选择对预测均值常数只增加0.0001376平均IoU，区间跨零。这支持检验共同误差项是否影响层间差异学习，尚未证明因果关系。

本轮只改变训练目标。两个目标在同一CPU环境重新训练，保持原网络、输入、Sigmoid、初始化、优化器、抽样和更新数。旧GPU模型作为来源诊断，不能代替新CPU绝对MSE控制。

## 唯一训练干预

令六层预测为p、真实缓存IoU为y，P=I−11ᵀ/6。

| 组名 | 优化目标 |
| --- | --- |
| absolute | mean((p−y)²) |
| relative | mean(((p−mean_l p)−(y−mean_l y))²) |

绝对MSE等于共同均值误差与relative误差之和。relative保留其中原有的层间相对项，不增加经验权重、不转换标签范围、不更换输出层或决策规则。减去同一均值不改变argmax。

投影不保证能学到有用的逐帧差异，也不会自动去掉各层的固定平均偏好。因此相对固定均值的增量仍是必要对照。relative组的输出仅用于相对排序，不再将其解释为已校准的绝对IoU。

## 固定的结构和预算

沿用Stage-194 R-only结构，模板与搜索token各自逐通道的均值、标准差组成4D输入，再以4D→6线性映射和Sigmoid输出。D由原骨干确定，实际为192，训练参数4614。A、Q均关闭且保持零参数，不加载或训练原SGLA。

38历史开发源组，19 fit／19 held两折，种子2132／2133。两目标共8个模型，每个256步、每步4例，即1024个抽样机会。使用原AdamW学习率1e-5、weight decay=1e-4、梯度上限1，全部更新CPU执行。各折各种子的两个目标初始参数哈希、draw哈希和训练参数量必须一致，draw也必须与Stage-194记录一致。

按实际原抽样有放回取数据，不额外添加失败帧、真值挑选窗口或新视频。新两组均使用同一份已经冻结的六分支缓存标签。标签来自原生父框的batch-one采集，不能要求新CPU训练精确复现旧GPU batch-four训练参数；source_bridge单独报告差异。

四项统计先按原CPU规则逐状态计算并缓存于内存，训练只运行小线性层。评价时全部状态同时调用原model.route和预聚合线性路径作数值一致性检查，防止缓存输入悄然改变模型。

不使用图像、SGLA原始权重、骨干或CENTER新前向。CUDA_VISIBLE_DEVICES在独立运行进程中设为空，避免CUDA版PyTorch优化器探测可见GPU。不会改变父shell的GPU配置。

## 来源绑定

必须保留完整Stage-194与Stage-195目录。绑定已复核的决策指纹：

- Stage-194为 `aecfc2112422a0cf5f1af3daa1ddc61bcce9f54fc4c3f2ae94f762429a9dd547`。
- Stage-195为 `a282dbfb2ca856e5a2fe4009356a10e35a4784a011837fa8cf2381bd1622b764`。

核对Stage-194特征、六层标签、scope，以及Stage-195表格与来源哈希清单。原阶段实现文件也需要保持原哈希。本轮不编辑这些冻结Python文件。确认集继续封存。

## 评价

八模型各评价38组×48状态，总计14592个状态输出。为两个目标都报告：

- fit、held、实际fit_draws、fit_seen_unique、fit_unseen上的绝对／共同／相对误差。
- 层间差值加权排序、相对L9的预测误差与regret。
- L9、fit全池均值、实际抽样标签均值、fit预测均值、真值上界与均匀选择期望控制。
- 所有状态的六层分数、真值、选择层和实际训练次数。
- 每步三项MSE与梯度，不能把不同目标的optimized_loss直接当作性能比较。

主比较是held缓存平均IoU的relative−absolute配对差，先在源组内平均两种子，再对38源组bootstrap。还比较相对误差、差值加权排序和各自对固定均值控制的增量。没有真实层间差异的源组不用于必须排序的指标，支持数单独由paired的videos报告；主IoU比较保留全部38组。

区间均为探索性的名义区间，不产生自动采用或SOTA结论。本轮没有完整自身轨迹，缓存平均IoU不能称作新AUC。若差异学习无增量，先记录该假设在当前输入与预算下未获支持；若有增量，后续再按同卡正常跟踪入口验证稳定视频、失跟和恢复。

## 运行

在具有真实缓存的服务器运行，无需分配GPU。

```bash
conda activate sglatrack-test
cd ~/workspace/paper
bash tools/run_stage196_relative_loss.sh
```

后台运行。

```bash
export STAGE194_ROOT="$PWD/output/stage194/module_ablation_v1"
export STAGE195_ROOT="$PWD/output/stage195/router_objective_audit_v1"
export STAGE196_ROOT="$PWD/output/stage196/relative_loss_comparison_v1"
mkdir -p "$STAGE196_ROOT"
nohup bash tools/run_stage196_relative_loss.sh > "$STAGE196_ROOT/run.log" 2>&1 < /dev/null &
echo $! > "$STAGE196_ROOT/run.pid"
tail -f "$STAGE196_ROOT/run.log"
```

保持一个总体进度条，后台定时刷新。已提交模型和预测缓存可复用，中断时未完成的单个256步模型从头训练。新结果独立写入Stage-196，旧产物只读。

完成自动生成stage196_results_时间戳_PID.txt，包含decision和9表。模型、优化器状态与NPZ保留服务器。

```bash
python -B tools/stage196_relative_loss.py --output-dir "$STAGE196_ROOT" --check-results
```

## 本地验证边界

5项本地测试通过，包括投影恒等式和梯度、均值平移不改变argmax、预聚合输入与原函数一致、A/Q冻结、同初始化同draw同预算、模型复用、缓存身份及篡改拒绝。另有38组合成缓存的完整8模型训练、所有表格和最终哈希核验，检查GPU隐藏并禁止CUDA初始化。

本机没有真实Stage-194／195模型与NPZ。真实训练与14592次缓存评价已在服务器完成并回传复核，本地合成测试只验证代码。 [上轮复核](STAGE195_RESULTS_REVIEW.md)
