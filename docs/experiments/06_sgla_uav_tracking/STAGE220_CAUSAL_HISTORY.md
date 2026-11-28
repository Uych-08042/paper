# Stage-220：二阶因果历史的增量价值

状态：实现待真实服务器CPU运行。219中geometry比full改善，但未优于zero；当前几何在nn历史已隐含一步位移。本轮检验至少三个时刻才可构造的速度变化是否有额外价值，不预设运动外推有效。

同一冻结来源轨迹的原生reference框R_t定义u_t=(cx_t,cy_t,log sqrt(w_t h_t))，v_t=u_t−u_(t−1)，a_t=u_t−2u_(t−1)+u_(t−2)。中心差分按当前R_t侧长标准化，尺度用log侧长；对齐原修正器的三维delta。a是预测轨迹二阶差分，包含真实运动与预测噪声，不能称真实物理加速度。t=1时a固定为0，v可使用共同已知的首帧框；t≥2只使用R_t、R_(t−1)、R_(t−2)，不读未来框或GT历史。

fit从214对应nn/mm提供轨迹，held从215对应自身轨迹取历史，当前raw必须精确匹配216/219共同状态。保留原38源、两折双seed、48均匀时刻、7296fit/14592held状态及四种held历史。GT仅首帧共同初始化及动作后t+1监督标签，不用GT定义运动。历史计算不受目标未来位置或未来预测尾部变化影响。

| 方法 | 学习输入/动作 |
| --- | --- |
| geometry | 原8维当前几何，精确复现219当前几何控制 |
| history | 当前几何8维 + 三维a |
| shuffled | 与history相同结构；仅训练时在每个fit源内打乱a的对应行 |
| zero | 不修正 |
| velocity | 直接用v_t作delta，相当于无学习恒速度外推 |

不再使用视觉均值条件。保持392→32→3和12675名义参数；8维当前几何仍占最后8槽，a占381:384三槽，其余置零。三个学习组使用**同一份真实fit历史的mean/std**，同初始化、原draws、forecast SmoothL1、AdamW、clip1、256步batch4。shuffled的a来自同一fit源合并nn/mm样本的一次固定置换（seed+100fold+220），仅替换历史三列；当前输入、raw、目标、尺寸、抽样不变。导出置换哈希及实际改变历史值的样本数。held包括shuffled模型始终输入实际可用的因果历史，不进行held打乱或估计held归一化。负控制只检验训练对应性的作用，不证明唯一机制。

加入历史三槽后，相应fit归一化buffer改变，所以不要求整份state digest等于219；geometry学习网络的参数digest必须精确等219 geometry，原8维mean/std也须精确。geometry与zero的912条逐组loss必须精确复现219，否则停止保留缓存，不放宽容差。所有新组初始网络参数相同；全新三组mean/std、draws相同，有效活跃维度为8/11/11，不宣称有效容量相同。

12新CPU模型、3072次更新、12288监督机会；无新图像、CUDA、完整轨迹、确认源或新AUC。原SGLA及适配器、选层和读出不改。来源219包固定规范化哈希，仅允许LF/CRLF差异；219物理产物、218/217 lineage、217 source_arrays与214～216所有所用数组/receipt仍绑定，原GT和帧表沿七来源核验。CPU无需GPU型号/UUID或checkpoint。缺缓存不补采。

唯一主要比较：held提供风格coverage__none/mm等权的history−geometry回归损失，负有利。源内平均时刻/两seed/两风格后38源bootstrap20000，报告名义CI、中位数、源正负数、负方向最坏留一。history−shuffled以及各学习组/恒速度对zero为支持比较；必须同时有对zero的改善，才考虑真实后继实验。重复开发区间不是独立确认，禁止held超参/阈值/早停。

九表：support456/scope12/training3072/group_metrics2280/policy_summary30/paired12/source_replay912/actions2280/execution1。support记录二阶历史可用与warmup计数及a哈希；actions记录无量纲中心修正均值、log侧长绝对值均值和非零比例，防止将接近零动作误作预测成功。保留梯度范数、模型/receipt和来源哈希。CPU单线程、3077总体进度，完成模型复用，未完成模型按固定初始化重训，自动汇总TXT。

代码和已审阅219结果同步服务器后，在原conda环境仓库根目录执行：

```bash
mkdir -p output/stage220
LOG="output/stage220/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage220_causal_history.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage220/run.pid
tail -f "$LOG"
```

默认output/stage220/causal_second_difference_v1，STAGE220_ROOT可指定独立输出；继承STAGE219_ROOT/218_ROOT/217_ROOT/215_REPLAY_ROOT与旧七来源变量。退出SSH后nohup继续；Ctrl-C只停止tail。

```bash
python -B tools/stage220_causal_history.py --check-results \
  --output-dir "${STAGE220_ROOT:-output/stage220/causal_second_difference_v1}"
```

完成后返回stage220_results_*.txt，先核当前几何网络与912损失复现，再核主要/zero/错配控制及动作幅度。若仍无对zero收益，暂停增强该修正映射并重新评估状态可观测性/监督目标；若有增量，再真实后继与完整轨迹验证，不从几何loss伪造IoU。真实结果尚未获得。
