# Stage-65：共同缓存状态下的选择错误与拟合支持诊断

日期：2026-11-03，Asia/Hong_Kong。已先读WORK_HANDOFF第0.20节及[Stage-64结果复核](STAGE64_RESULTS_REVIEW.md)。本协议在取得Stage-65结果前固定。本机已完成实现与小流程验证，服务器尚未运行。

## 本轮解决的问题

Stage-64的reliability对匹配quality为−4.3668 AUC，0226恢复丢失、0003新增末尾失效；quality虽然原生宏AUC最高，也丢失0307恢复。E0304／2132有局部完整轨迹修复，E总体却下降。现在需要区分三个可处理的问题：

- 已有良好q0，却被较差候选替换。
- q0不足但其他候选良好，评分没有选中机会。
- 当前四个公共解码框都不足，单纯改变选择没有当帧良好框可选。

本轮直接比较同一缓存预测历史上的四种选择，避免把完整自身轨迹不同误当成同起点换规则的效果。同时检查原生fit状态与实际运行状态的特征支持，给后续“连续质量＋有效状态保护”提供具体依据。仍以原始SGLA V1无人机SOT为基础，不切换V2或检测任务。

## 输入、范围和成本

入口为完整`output/stage64/causal_candidate_reliability_v1/`；新输出为`output/stage65/cached_same_state_selection_v1/`，不得嵌在输入目录内或覆盖旧结果。

只读取封存模型、manifest、完整候选缓存和上游来源。不读取图像，不构造网络，不需要CUDA、GPU或checkpoint参数；沿用原`sglatrack-test`环境的NumPy以重建原系数。原4个评分器只做数值重建核对，不保存新评分器、不改变旧系数，也不搜索标签、特征或超参数。

| 状态来源 | 缓存轨迹／数据集 | 状态出现次数 | 每状态公共框 |
|---|---:|---:|---:|
| Stage-64原生四政策＋E两政策双种子 | 96 | 108200 | 4 |
| 两折各自原生fit history／recrop | 24 | 27050 | 4 |
| 合计 | 120 | 135250 | 4 |

每个状态分别应用q0、Hann、quality、reliability。共541000次被报告的规则应用、541000个缓存框出现次数；后者不是新增网络搜索，也不表示不同历史中的框内容互不重复。Stage-64已付出432800次搜索，原生fit参考另复用108200个旧L40公共框；本轮新增搜索、图像读取、训练模型、完整轨迹均为0。来源审计和模型核对可能耗时，进度按fold和轨迹输出。

E原来只实际执行Hann／reliability。这里新增的E quality／q0只是对已保存状态的影子选择，不能称作新增E完整轨迹或跨头方法收益。

## 固定比较和统计口径

同一状态使用完全相同的四个公共框、原24项及Stage-64的64项特征、同一折封存评分器。选择函数不接受GT；GT仅在之后为候选打离线标签。影子选择不更新父框、历史或下一帧输入。

全量输出四规则的已选／最好／q0当前IoU、当前候选regret、良好与低IoU比例，以及以下条件率：

| 条件率 | 分母 | 分子 |
|---|---|---|
| `miss_given_good_decoded` | 四框至少一个IoU≥0.5 | 所选IoU<0.5 |
| `loss_given_good_q0` | q0 IoU≥0.5 | 所选IoU<0.5 |
| `low_given_good_q0` | q0 IoU≥0.5 | 所选IoU<0.2 |
| `rescue_given_opportunity` | q0 IoU<0.5且至少一个良好替代 | 所选IoU≥0.5 |

分母为0保持空白，并报告每项支持的种子／视频数；不能当成0错误。先在同视频内平均有支持的种子，再对有支持的视频等权。状态数与各计数保留在seed表，不能以全视频帧池替代视频宏平均。fit参考为拟合内描述，held_out为OOF评分器在实际保存历史上的描述，分开列出。

同状态quality−reliability、quality−Hann、reliability−Hann另外报告当前平均IoU、良好／低IoU比例、良好q0降为低IoU的比例差和选择分歧率。不把这些结果命名为全视频AUC，不用它们构造未执行的反事实跟踪轨迹或未来收益。

## 特征支持与标签信息

只使用本折原生fit组、按Stage-64视频／政策／帧／三个相对q0候选对的原权重，计算64维标准化特征差的1%／99%加权经验分位数。两个目标的scale须完全相同；加权分位数固定为逆经验CDF。

输出坐标超出区间比例、任一维超出区间的候选对比例、绝对标准化特征均值和候选对最大绝对特征的99%分位数。64维“任一维超界”本来就可能常见；这是支持范围描述，不是经校准的OOD检测器、在线阈值或分布偏移因果证明。

另报告“三级效用相同但连续IoU有差别”的相对q0候选对比例，以及各学习规则对应标签单位的平均候选对绝对预测误差。quality与reliability标签单位不同，不直接用两者误差大小宣称哪种模型校准更好；同目标跨状态来源的比较更有意义。

## 固定关注窗口

每例按指定来源实际事件解析锚点，不用结果挑最好帧；前后各30帧，视频边界截断。每个锚点同时列出同视频、origin、种子的所有Stage-64状态提供政策，在每个保存状态上比较四规则。不同提供政策即便同帧号，进入历史仍然不同；只有单条状态内的规则对比是共同输入。

| 例 | 视频／origin／种子 | 锚点来源 |
|---|---|---|
| F01 | 0003／native／0 | reliability首次失效 |
| F02 | 0226／native／0 | reliability首次失效 |
| F03 | 0226／native／0 | quality首次恢复 |
| F04 | 0307／native／0 | quality首次失效 |
| F05 | 0307／native／0 | q0首次恢复 |
| F06／F07 | 0226／E／2132、2133 | reliability各自首次失效 |
| F08 | 0304／E／2132 | Hann首次失效 |
| F09／F10 | 0307／E／2132、2133 | 旧L40自身E current首次恢复 |

预期10例、30个保存历史窗口、120条规则窗口摘要。旧结果提示帧号为138／124／339／386／474／127／127／1093／471／511，但代码按封存事件解析，不用这些帧号作在线规则。缺少预定事件直接报错，不替换病例。窗口逐帧表保存四候选IoU、各规则选择／已选IoU和两评分器的四项分数。

关注病例均由已有结果引导，属于消耗过的开发描述；完整12视频不移除。该轮没有方法采用门槛，`COMPLETE_CACHED_SAME_STATE_SELECTOR_DIAGNOSTIC`仅表示诊断完成。

## 来源与不可改写边界

验证完整Stage-64 plan／models／run／smoke／decision及9张CSV、来源代码和所有缓存receipt；继承全部上游来源绑定。核对两个fold的旧拟合系数、fit参考输入、96条实际选择、Hann与q0来源复现，以及封存头／运行身份。

新manifest封存本轮源码、设计、关注锚点和来源指纹。发布前再次校验来源和代码；相同输出可重复执行，不同既有输出拒绝覆盖。Stage-1～64源码、权重、模型、NPZ和报告不改写。Stage-65只新增NumPy分析，不运行旧summary来生成新来源。

## 服务器命令与回传

同步新增tools文件、`.gitattributes`及文档后，在原环境运行：

```bash
cd ~/workspace/paper
export STAGE64_ROOT="$PWD/output/stage64/causal_candidate_reliability_v1"
export STAGE65_ROOT="$PWD/output/stage65/cached_same_state_selection_v1"
bash -n tools/run_stage65_selection.sh
bash tools/run_stage65_selection.sh
```

也可单独执行：

```bash
python -B tools/stage65_selection_audit.py --input "$STAGE64_ROOT" --output_dir "$STAGE65_ROOT"
```

脚本用cat将diagnostic及policy_summary／matched_summary／feature_shift／scope／focus五表合并为`docs/experiments/06_sgla_uav_tracking/results/stage65_results_<UTC>_<pid>.txt`；回传一个TXT即可。逐视频、逐种子、特征明细和focus_frames六表留服务器，不默认要求全部上传。脚本不commit／push。

## 本机验证

3项测试通过，50.983秒、无跳过。语义测试覆盖同状态有害替换与恢复机会、空分母、GT改变不影响评分、加权分位数、种子→视频宏平均及缺失锚点拒绝。真实小型CENTER的Stage-59→60→61→L40→62→63→64流程生成缓存，再读取执行Stage-65；在审计期间阻断Torch／OpenCV导入、图像原字节读取／哈希和新演员推理，重复输出一致、来源字节不变，篡改源表／旧模型／新结果及不匹配头身份均拒绝。

小流程为2视频、16条Stage-64轨迹、64个held-out状态＋16个fit参考状态，320次规则应用／缓存框出现次数、16条窗口摘要。上游骨干／图像／设备为fixture，真实CENTER解码，不代表服务器120个缓存数据集已完成。CLI、3个Python AST及Bash LF通过；沙箱阻止Git Bash初始化后，经自动审批在沙箱外执行的只读`bash -n`通过。尚无Stage-65服务器结论。
