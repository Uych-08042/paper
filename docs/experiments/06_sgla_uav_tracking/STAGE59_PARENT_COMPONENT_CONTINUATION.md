# Stage-59：后继搜索父框的中心／尺寸与冻结头交叉诊断

日期：2026-10-31。主任务仍为原生SGLA无人机单目标跟踪，使用现有VisDrone-SOT。用户已报告`COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC`，五份优先材料已齐，见[结果复核](STAGE59_RESULTS_REVIEW.md)。本协议保留预定设计与历史复现入口，Stage-1～59实验Python及固定结果冻结，不重复采集。本轮原先4项真实CPU测试通过（301.661秒、无跳过）；当前新入口是[Stage-60](STAGE60_CAUSAL_SEARCH_STATE_VALUE.md)。

## 本轮问题与新增信息

[Stage-58复核](STAGE58_RESULTS_REVIEW.md)显示，全部参考保留且梯度投影充分发生，仍未超过完整保持；0226的2133种子恢复崩溃，0176／0204有局部收益，0304／0307仍受损。它不足以指定损伤由中心、尺寸还是响应选点引起。

Stage-55已经比较过“整个父框×头”。本轮进一步把**预测父框拆成中心和尺寸两个可干预分量**，再交叉当前冻结头，检验差异更依赖哪一个输入分量以及两者的交互。新增训练为0；用较少原生搜索前向，为后续具体的状态控制或训练处理提供证据。本轮没有预设尺寸平滑或中心修正一定有效。

E为Stage-58完整末帧保持，P为Stage-58单侧梯度投影。每个种子分别取来源轨迹在锚点之前的预测框，不混合两个种子的分量，不用GT构造父框。

| 起点 | 中心来源 | 宽高来源 |
|---|---|---|
| EE | E预测 | E预测 |
| PE | P预测 | E预测 |
| EP | E预测 | P预测 |
| PP | P预测 | P预测 |

同来源组合直接复制原框，避免通过中心重建引入额外舍入；混合框由所选中心和宽高重建，不额外clip父框以免改变被干预分量。图像边界由原生裁剪的padding及输出解码处理。四个组合都运行，不依据GT挑选较好组合。

## 固定案例与32帧分支

| 案例 | 视频 | 锚点 | 来自Stage-58的固定事件 |
|---|---|---:|---|
| C01 | uav0000226_05370_s | 338 | E2132恢复，P2133丢失恢复 |
| C02 | uav0000176_00000_s | 238 | P2133恢复 |
| C03 | uav0000176_00000_s | 315 | P2132推迟后的失效 |
| C04 | uav0000304_00253_s | 555 | P2133早期持续失效 |
| C05 | uav0000304_00253_s | 1086 | P2132较晚持续失效 |
| C06 | uav0000307_04531_s | 511 | E2133恢复，P2133没有该恢复 |
| C07 | uav0000204_00000_s | 664 | E2132失效，P2132保持无失效路径 |

prepare从来源完整轨迹核对指定种子／策略的事件锚点；每个锚点同时评价2132和2133，不仅评价选锚点的种子。全部为已消费结果引导的事后案例，不构成新外部验证，也不能把同视频两个锚点当成两个独立视频。

每个组合先使用原生SGLA做一次完整前向，并将同一个GPU特征张量交给同种子的E、P两个冻结头，记录原生／E／P的响应与几何图。随后E、P分别从该组合父框推进32帧。

**只交换起点父框，之后每一步使用本分支自己的实际输出。** 不持续注入E／P来源框，不用GT重置或教师接管。首帧GT只用于原始首帧模板；后续GT只进入离线评价、几何和来源事件核对。原始骨干、候选、选择器、响应及全部头权重均冻结。

这形成每案例、每种子的中心2×尺寸2×头2完整析因。EE起点＋E头、PP起点＋P头的32帧对角线必须复现各自来源轨迹；所有分支第一帧还必须复现同特征探针的图与输出。来源其他状态不进入actor，原生SGLA没有在线模板更新，当前检查范围不扩展到带隐藏历史状态的任意跟踪器。

## 来源、预算与断点恢复

输入是完整Stage-58根目录。prepare严格核验manifest、smoke、12个来源模型、72条来源轨迹、8张表及其SHA，重新从轨迹核对132行逐种子／72行逐视频／6行宏表和预算，并继续验证上游来源。E／P共8个OOF头用于诊断，N模型仍作为完整Stage-58来源一并校验，不参与新的分支。

正式7案例×2种子×4起点＝56个共享输入；每输入额外运行2个头，共112次仅头前向。7×2×4×2＝112条32帧分支，3,584次搜索前向。合计：

- 原生完整搜索前向3,640次，block25,480次、selector3,640次。
- 仅头前向112次，总head3,752次；首帧模板初始化开销沿用既有预算口径，不计作搜索前向。
- 新训练0、新模型0。没有完整视频新策略评估或方法采用门槛。

smoke完成C01全部组合与分支，即520次完整搜索前向＋16次仅头；这些缓存直接被collect复用，不另加到正式总预算。缓存按输入／分支逐项提交收据并绑定来源、设计和代码；中断后重跑同一命令复用已提交项，未提交分支完整重跑。校验失败先排查，不放宽原定0.001像素框误差、1e-5峰值、1e-6图误差容差。

## 报告与解释边界

输出8张表：

- `crossed_matrix`：56行，父框中心误差、宽高、实际取整搜索边长、GT在搜索图中的像素尺寸、覆盖率及原生／E／P当帧IoU。
- `point_details`：168行，进一步包含输出中心／宽高误差、raw／Hann格点、dense候选及GT Top-5情况。
- `rollout_summary`：112行，32帧均值／末帧IoU、低IoU及局部失效／恢复、几何指标。
- `focus`：14行，两个精确记账路径及当帧／32帧均值／末帧的自身差。
- `factor_summary`：42行，各锚点种子的中心、尺寸、头顺序平均效应及两两交互。
- `branch_contrasts`：56行，同一混合起点下P−E及裁剪／响应首次分岔。
- `exposure`：14行，四个起点实际形成多少不同父框、取整ROI和裁剪像素，允许相同或近似相同，不据此换锚点。
- `timelines`：3,584行，保存后继逐帧输出与几何。

令y(C,S,H)表示指定起点和评价头的当帧／窗口均值／末帧IoU。自身差D=y(P,P,P)−y(E,E,E)。报告两条路径：

```text
D = 头在EE起点的差 + P头下先换中心的差 + 再换尺寸的差
D = 头在EE起点的差 + P头下先换尺寸的差 + 再换中心的差
```

两条均为精确条件记账，不是唯一因果百分比分摊。顺序平均效应在其他因子的全部水平上平均；存在交互时不应要求三个平均主效应直接相加等于D。32帧换头效应包含之后的自身状态演化，不是纯当帧参数作用。

搜索尺寸交换会改变搜索范围、目标分辨率、padding和响应竞争；不能直接称为“尺寸回归错误”。原生搜索裁剪主要由预测面积决定，本轮不是独立的宽高比回归实验。即使coverage=1，也可能因目标分辨率或响应选择不同而失效。相同取整ROI仍可能有浮点映射差，需结合exposure和几何读数解释。

窗口IoU是0～1，不能称为全视频AUC百分点。局部事件在窗口边界可能截断；完整视频最终恢复／未恢复仍以Stage-58为准。无总体bootstrap或方法PASS，完成只标记`COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC`。不得按本轮GT结果在线挑起点／头，也不据此搜索权重、分支长度或高分种子。

## 服务器执行（历史复现入口，已完成）

本机4项CPU Torch／OpenCV测试此前全部通过，301.661秒、无跳过。覆盖分量重建与原组合逐位复制、事件锚点和精确预算、含交互的两条记账恒等式、真实CENTER小通道头／原生解码／共享响应、自己输出形成后继父框、后续GT不改变actor、完整Stage-51→58合成来源及本轮prepare／smoke／collect／summary、来源对角线复现、smoke缓存复用与断点重复运行、结果／缓存／来源表／checkpoint篡改拒绝。4文件AST、纯NumPy CLI帮助、文档链接和维护检查通过。CPU合成流程不代表服务器精度结果。用户现已运行完成并提供结果，助手没有登录服务器或执行GPU采集，不代用户commit／push。以下命令仅供历史复现，不作为本次或下次默认操作；继续保留Stage-58和上游模型／轨迹：

```bash
cd ~/workspace/paper
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE58_ROOT="$PWD/output/stage58/projected_endpoint_gradient_v1"
export STAGE59_ROOT="$PWD/output/stage59/parent_component_continuation_v1"

python -B tracking/stage59_parent_collect.py --mode prepare --input "$STAGE58_ROOT" --output_dir "$STAGE59_ROOT"
python -B tracking/stage59_parent_collect.py --mode smoke --input "$STAGE59_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage59_parent_collect.py --mode collect --input "$STAGE59_ROOT" --checkpoint "$CHECKPOINT" --gpu 0
python -B tracking/stage59_parent_summary.py --input "$STAGE59_ROOT"
```

以下5份优先材料已全部收到并复核，不重复索取；cat命令保留作定位参考，逐帧大表与完整缓存继续保留在服务器：

```bash
cat "$STAGE59_ROOT/analysis/stage59_diagnostic.json"
cat "$STAGE59_ROOT/analysis/stage59_focus.csv"
cat "$STAGE59_ROOT/analysis/stage59_crossed_matrix.csv"
cat "$STAGE59_ROOT/analysis/stage59_factor_summary.csv"
cat "$STAGE59_ROOT/analysis/stage59_exposure.csv"
```

完整中心／尺寸／响应解释若依赖更细信息，使用本轮已有point_details、branch_contrasts、rollout_summary和timelines，不默认重跑采集。
