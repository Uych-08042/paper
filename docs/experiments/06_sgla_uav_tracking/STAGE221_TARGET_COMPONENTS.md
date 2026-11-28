# Stage-221：冻结预测监督标签的误差/运动分解

状态：实现待真实服务器CPU运行。220的三帧历史未超过当前几何或zero，本轮暂停增加预测器复杂度，仅解释原标签要求修正什么。0训练、0新图像/CUDA搜索/完整轨迹/确认源，无新IoU/AUC。

## 固定定义与解释边界

R_t为原生reference预测框，G_t和G_(t+1)为当前和下一GT，s(R)=sqrt(w h)。中心标签用**同一s(R_t)**归一化：

\[
y_c=(c(G_{t+1})-c(R_t))/s(R_t),\quad
E_c=(c(G_t)-c(R_t))/s(R_t),\quad
M_c=(c(G_{t+1})-c(G_t))/s(R_t).
\]

尺度为y_s=log[s(G_(t+1))/s(R_t)]、E_s=log[s(G_t)/s(R_t)]、M_s=log[s(G_(t+1))/s(G_t)]。必须精确核y=E+M（浮点容差1e−12）。GT只作为离线分解和原回归loss标签，不进预测器；不把GT分量当在线动作，不声称oracle跟踪收益。

分别报告中心E/M/y的L2、平方能量与2E·M交互、E·M<0的反向比例、E模长大于M的比例；log尺度的绝对值、平方与交互同理。平方能量必须含交互才能还原y能量；**Smooth-L1不是可加的，不能用L(E)+L(M)当L(y)**。量级更大不等于因果根因、损失贡献占比或可预测性更低。

唯一主要描述比较：held提供风格coverage__none/mm等权的mean(||E_c||−||M_c||)。正值仅说明标签中的当前定位误差量级更大，不是方法收益。源内先平均有效状态/两seed/两风格，再源组等权、沿原20000次bootstrap。fit同比较是支持；不混合中心与log尺度选择方向，不据held调阈值。名义区间为开发固定模型条件的描述，非独立确认。

## 范围、来源和复现

固定38开发源、两折双seed、214的nn/mm fit样本7296及216/215共同held状态14592，沿220完全相同的四种held历史与48均匀时刻。GT/帧表沿七来源指纹；220已审阅包固定规范化哈希，只允许CRLF/LF；220物理manifest/decision及冻结12模型、219/218/217来源图、217 source_arrays与214～216所用数组/receipt仍核验。源文件不改、缺缓存不自动补采。

重新加载220 geometry/history/shuffled三类冻结模型，velocity直接用原因果v，zero为0；不创建优化器，不更新参数。原来全部2280条逐组forecast损失及请求动作统计须精确复现220；否则停止保留结果，不放宽容差。评价前后模型digest保持相同。

当前GT无效时保留该cell与valid标记，E/M为空，原下一标签y仍可存在。group报告有效对/无效当前/无效下一计数，不静默删点；汇总只用两seed都有有效对的源，主要比较要求两风格也都有支持，缺失源名单显式导出。支持不足2源时区间/留一留空。下一GT若无效则无法复现已审阅220有限loss，立即停止，不能变更样本子集假报原loss精确。

actions表使用原loss相同float32边界处理后的有效delta，与E/M/y报告中心平方距离、log尺度平方距离和中心方向cosine；零向量cosine未定义，留空并报告有效支持数。方向/距离是动作关系诊断，不是训练优化目标，也不能当IoU。请求幅度仍按原220动作表复现；应用边界后数值与请求不混淆。

## 产物和预算

九表：support456/scope12/cells21888/actions2280/group_metrics456/policy_summary6/paired2/source_replay2280/execution1。cells逐点保留三分量及加法残差；group_metrics只汇总有效分解对。scope为12冻结模型、0新更新，source_replay同时核loss与请求动作。一个总体进度469项，CPU单线程，自动汇总TXT，原始来源PT/NPZ/receipt哈希封存。诊断导出真值派生量，不作为新模型训练数据或在线输入。

同步代码和已审阅220结果后，在原conda环境、仓库根目录执行：

```bash
mkdir -p output/stage221
LOG="output/stage221/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage221_target_components.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage221/run.pid
tail -f "$LOG"
```

默认output/stage221/frozen_target_components_v1，STAGE221_ROOT可指定独立目录；继承STAGE220_ROOT及此前各缓存变量，无GPU型号/UUID或checkpoint要求。退出SSH后nohup继续，Ctrl-C只停止tail。

```bash
python -B tools/stage221_target_components.py --check-results \
  --output-dir "${STAGE221_ROOT:-output/stage221/frozen_target_components_v1}"
```

返回stage221_results_*.txt后先核2280复现及分解恒等式/有效支持，再判断定位误差或目标真实运动的量级与抵消关系。若主要是定位误差，应优先寻找目标对应/重定位证据；若主要是真实运动，再考虑动态观测与预测噪声。两种解释都需结果支持，不自动回到模型/历史扫描。真实结果尚未获得。
