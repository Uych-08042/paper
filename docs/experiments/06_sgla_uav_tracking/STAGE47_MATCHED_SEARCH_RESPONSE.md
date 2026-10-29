# Stage-47：固定历史状态的搜索干预与响应重放

日期：2026-10-29。状态：实现与4项本地 CPU 测试完成；实际原始权重下的服务器 GPU 重放待运行。

## 1. 要回答的问题

Stage-46B 已由用户完成，三个主要案例的失效起点仍有较高裁剪覆盖率：`0204:664` 的原生父框中心误差约0.071个GT边长，输出误差却约1.362；`0304:555` 的仅相机补偿父框误差约0.086，输出误差约0.992。`0176:158` 的相对速度项远大于相机项。`0304:1087` 的组合策略输出中心误差仅约0.065，IoU却约0.172，需进一步核查形状与横纵误差。

本轮固定预测历史，直接观察本帧搜索干预后的原生响应、候选与几何。保留 SGLA 首帧模板、原始权重、选择器、候选结构、预测头和 Hann 解码。每次直接调用原生 `tracker.track()`，用模块hooks记录响应/尺寸/偏移图与实际调用次数。

本轮是四个已消耗开发案例的事后机制实验，不训练新模型，不建立新策略采用门槛。一次重放的结果不继续生成下一帧历史，也不汇报为新完整视频AUC。

## 2. 固定事件与规模

| 编号 | 视频 | 起点 | 重点历史 |
| --- | --- | ---: | --- |
| E01 | `uav0000176_00000_s` | 158 | `camera_velocity` |
| E02 | `uav0000204_00000_s` | 664 | `sgla` |
| E03 | `uav0000304_00253_s` | 555 | `camera_only` |
| E04 | `uav0000304_00253_s` | 1087 | `camera_velocity` |

每个起点固定观察偏移−2、−1、0、+1、+2帧。在每帧分别取Stage-46四组保存的自身上一预测框、当前背景变换、历史普通速度和相对速度，再应用五种搜索方式。因此每个比较内部共享一个历史状态；不同历史之间仍有各自的预测、遮罩与运动估计。

| 干预 | 本帧搜索中心 |
| --- | --- |
| `no_shift` | 保存的自身上一预测中心 |
| `raw_velocity` | 上述中心加保存的普通速度 |
| `camera_only` | 用保存的背景矩阵映射上述中心 |
| `relative_only` | 上述中心加映射后的历史相对速度 |
| `camera_relative` | 保存的背景变换与相对速度共同作用 |

沿用Stage-46每帧位移上限0.5个预测目标等效边长、图像边界限制和原宽高。背景估计无效时，camera_only与relative_only保持原位置，camera_relative退回普通速度。各组独立施加限制，因此受限后的组合不必等于两个单项之和。

4起点×5帧×4历史×5干预＝**400次独立前向**。每次6前缀块＋1候选块，合计400次头调用、2,800次块调用。smoke完整运行E01的100次，后续采集复用该缓存，仅新增另外300次。

不固定候选为L9。保留原选择器决定候选，并记录实际选层；如果新裁剪导致选层变化，解释的是原生管线整体响应。

## 3. 来源与重放校验

prepare读取原Stage-46目录，核验48条完整轨迹和对应receipt、原Stage-27来源，以及完整Stage-46B manifest/report/表SHA。四个固定起点还必须是重点历史中已记录的首次失败。它只固定计划，不打开GPU。

GPU阶段使用同一原始SGLA checkpoint与Stage-46相同运行身份。图像按原manifest哈希读取，模板仍来自同一首帧初始化框。每次设置保存的本帧搜索父框与帧号，执行一次原始前向，不使用GT父框、GT触发或新光流。

五种干预中必有一种复现该历史原先执行的方式：sgla→no_shift，velocity→raw_velocity，camera_only→camera_only，camera_velocity→camera_relative。逐帧核对这些对照的框、响应幅度与选层，并要求官方截断后的整数框完全相同；密集解码另按0.001像素容差核对原生输出。

保存实际裁剪像素与attention-mask的SHA、完整raw/size/offset图、浮点父框、裁剪矩形和实际调用计数。缓存按case提交，NPZ与receipt字节哈希绑定，断点重复命令复用已通过核验的case；被篡改的已提交结果报错，不覆盖。旧实验代码和产物保持原样。

## 4. 分析及其解释

### 响应与目标候选

输出原生IoU、裁剪覆盖率、raw/weighted峰位置、固定Top-5正确目标保留率与排名、GT分数margin和dense最佳GT匹配IoU。使用Stage-28原local-peak与NMS规则，候选结果依赖该固定规则。

目标匹配只在前向结束后离线计算，不进入搜索控制。grid cell属于各自裁剪坐标，跨裁剪相同cell编号不代表相同原图位置；低IoU框也不自动证明语义干扰物，GT框在裁剪中不等于目标可见。

### 裁剪输入变化与浮点映射

原生裁剪原点取整，map-back使用浮点父框。保存像素SHA以辨别输入是否实际改变；在同一历史内，再离线计算2×2解码：

| 响应/几何图 | 解码父框 | 用途 |
| --- | --- | --- |
| no_shift图 | no_shift父框 | 不偏移基准 |
| no_shift图 | 干预父框 | 仅改变浮点映射的离线探针 |
| 干预图 | no_shift父框 | 将改变后的图映射到共同原点 |
| 干预图 | 干预父框 | 实际干预的原生结果 |

均使用Hann峰与原密集解码/裁剪口径。交互是该2×2的离线差值，不能当作唯一的响应头归因，也不是另外评估过的跟踪策略。连续解码的小浮点差异不允许改变原轨迹复现对照的官方整数框。

### E04形状诊断

输出预测/GT宽高比、未裁剪宽高比、边界裁剪差、横向中心误差/GT宽、纵向中心误差/GT高，以及两个离线框替换：保留预测中心而用GT尺寸、保留预测尺寸而用GT中心。它们只辅助区分形状与中心，不输入模型；对应IoU不可相加构成因果分解。

窄长目标也可能在等效边长归一化的中心误差较小时出现较低IoU，因此不预先认定E04必然只是尺度问题。

## 5. 服务器命令

通过现有Git流程同步新增文件后，在项目根目录运行；不需要新数据集或新环境。

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
export STAGE46_ROOT="$PWD/output/stage46/sgla_camera_search_v1"
export STAGE47_ROOT="$PWD/output/stage47/matched_search_response_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python -B tracking/test_stage47_search.py

python -B tracking/stage47_search_collect.py --mode prepare \
  --input "${STAGE46_ROOT:?}" --output_dir "$STAGE47_ROOT"

python -B tracking/stage47_search_collect.py --mode smoke \
  --input "$STAGE47_ROOT" --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

cat "$STAGE47_ROOT/smoke/stage47_smoke_decision.json"
```

smoke通过状态为 `PASS_MATCHED_SEARCH_REPLAY_SMOKE`，只说明重放/采集校验通过。随后：

```bash
python -B tracking/stage47_search_collect.py --mode collect \
  --input "$STAGE47_ROOT" --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python -B tracking/stage47_search_summary.py --input "$STAGE47_ROOT"

cat "$STAGE47_ROOT/analysis/stage47_onset_summary.csv"
cat "$STAGE47_ROOT/analysis/stage47_onset_hybrids.csv"
cat "$STAGE47_ROOT/analysis/stage47_shape_focus.csv"
```

`--case_id E02`等参数可在collect阶段单独续跑一个预定case；最后汇总仍要求全部四个case。重复完整命令也会自动复用。缺失时汇总打印 `PILOT_INCOMPLETE`，不伪造完整诊断。

完整状态为 `COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC`，不表示方法PASS。输出另有全400条的`stage47_responses.csv`、`stage47_hybrids.csv`和绑定来源/预算/解码误差的`stage47_diagnostic.json`。优先阅读重点历史在起点的20条响应及其20条混合解码结果，再核对E04前后25条形状记录。

本地测试使用合成冻结缓存与小型CPU网络验证控制状态、解码、输出hooks、真实调用计数、全流程续跑、来源不变和篡改拒绝；未加载完整原SGLA进行服务器GPU重放。
