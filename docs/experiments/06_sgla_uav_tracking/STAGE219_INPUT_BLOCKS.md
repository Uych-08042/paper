# Stage-219：固定MLP的视觉/几何输入分块对照

状态：实现待真实服务器CPU运行。218的linear对MLP held回归损失+.0224785，低容量没有解决泛化，且梯度裁剪差异较大；本轮固定网络，检验当前输入的分块贡献，不预设视觉无用。

固定392→32→3 Tanh MLP（12675个参数）及初始化、fit全样本mean/std、原抽样draws、forecast原SmoothL1、AdamW与梯度裁剪1、256步×batch4：

| 输入组 | 保留的标准化输入 | 活跃维度 |
| --- | --- | ---: |
| full | 全部 | 392 |
| geometry | 最后8维：raw/adapted相对parent的中心与宽高几何 | 8 |
| visual | 前384维：模板/搜索F6 token的各192维均值 | 384 |

移除块在用**同一完整fit mean/std**标准化之后置零，等价于原单位用fit均值插补。不得用held均值，不能分别重新计算掩码后归一化；保留全部网络参数，三个初始state digest必须相同且精确等218 MLP。有效输入维度/有效模型自由度不同，不声称有效容量完全匹配。

几何8维是当前两读出的几何差异，不是新的长期运动历史。visual组仅限制预测网络的输入：原始reference box仍用于同一坐标映射和损失计算，不能声称整个算法不使用几何。

冻结214～218模型及缓存，不改SGLA/适配器/原层选择；全38开发源、两折双seed，fit7296样本、held14592状态、四种历史均保留。t输入/t+1 GT/t尺寸沿218相同绑定路径读取。12新CPU模型、3072步、12288抽样监督机会，四方法含zero；0新图像/CUDA推理/完整轨迹/确认源，无新AUC。

完整输入四模型的最终权重digest必须精确复现218 MLP，full与zero的912条逐组损失也须精确。任何复现失败立即停止并保留缓存，不放宽容差；如CPU软件变化须先查明原因并独立重测控制，不能跳过。数据和metadata从217 source_arrays与214～216 decision绑定，218包固定规范化哈希，只允许LF/CRLF；来源代码/协议/模型/数组/receipt仍核验。缺缓存不自动GPU补采。

唯一主要比较：held提供风格coverage__none/mm等权后的geometry−full回归损失，负有利。源内先平均seed/两风格，再38源bootstrap20000次，报告名义CI、源中位数、正负数与负方向最坏留一。visual−full及各组对zero是支持比较；geometry也必须优于zero，才能考虑后继验证。禁止held调阈值、超参或早停；不以相对坏控制改善当方法采用。开发名义CI不是独立确认，不把输入干预直接解释成信息论无信息或单一因果解释。

八表：support456/scope12/training3072/group_metrics1824/policy_summary24/paired10/source_replay912/execution1。scope导出活跃维度、相同初始/归一化/draws；training保留每步梯度范数以核算裁剪。一个总体进度3077步；CPU单线程、完成模型可复用，未完成模型从固定初始化重训，自动汇总TXT。

同步代码和已归档218结果后，在服务器原conda环境与仓库根目录执行：

```bash
mkdir -p output/stage219
LOG="output/stage219/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage219_input_blocks.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage219/run.pid
tail -f "$LOG"
```

默认output/stage219/matched_mlp_input_blocks_v1，可用STAGE219_ROOT指定独立目录。继承STAGE218_ROOT、STAGE217_ROOT、STAGE215_REPLAY_ROOT和既有七来源变量。无需GPU或checkpoint；退出SSH任务继续，Ctrl-C仅停止tail。

```bash
python -B tools/stage219_input_blocks.py --check-results \
  --output-dir "${STAGE219_ROOT:-output/stage219/matched_mlp_input_blocks_v1}"
```

完成后返回脚本打印的stage219_results_*.txt。先核四个full权重及912损失复现，再解读主要和zero比较；若各块仍不及zero，转因果时序/状态支持与目标建模。即使几何块改善，也需真实后继/完整轨迹验证才能说明跟踪收益。本轮真实结果尚未获得。
