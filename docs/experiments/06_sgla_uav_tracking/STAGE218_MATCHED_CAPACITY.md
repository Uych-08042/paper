# Stage-218：同训练机会的搜索修正容量对照

状态：实现待真实缓存运行。动机来自217训练内回归改善、留出多数源恶化，检验限制映射自由度能否减轻反转；容量只是待检假设，不预设小模型一定有效。

固定214/215/216/217缓存和全部38开发源，两折、seed2132/2133。每折只用fit源的nn/mm样本及归一化，256步×batch4，同原draws、AdamW(lr=.001,weight_decay=.0001)、梯度裁剪1、原forecast SmoothL1。所有模型float32 CPU训练：

| 模型 | 映射 | 参数 |
| --- | --- | ---: |
| bias | 可训练三维偏置 | 3 |
| linear | 392→3 | 1179 |
| mlp | 392→32→3，Tanh | 12675 |

全部初始输出为零；线性权重为零，MLP沿用原初始化。归一化只来自完整fit1824样本，每个fold/seed同buffer/draws哈希。匹配的是训练机会，不是参数、FLOPs或相同优化难度。12次训练、3072次更新、12288次抽样监督；保存权重与receipt，可断点复用已完成模型，不重写来源。

评价同217全部7296fit样本、14592held状态，五方法包括三新模型、zero、旧冻结forecast。held四历史均保留，主要统计只用提供风格coverage__none/mm等权。直接调用原loss，t输入/t+1 GT/t图像尺寸，来源数组与receipt哈希精确绑定217的source_arrays及214～216decision；七来源帧表/GT仍核验。旧forecast与zero共912条逐组损失须精确复现217，否则停止，不调容差。CPU训练MLP不要求等旧GPU训练权重。

唯一主要比较：held提供风格linear−匹配CPU mlp的回归损失，负有利。先每源内seed/风格均值，再38源bootstrap20000；源中位数、正负数及负方向最坏留一。线性/bias/MLP对zero属于必要支持比较：只比坏模型好不构成采用，应同时见对zero有明确改善。fit及四历史均值是机制支持；重复开发的区间是固定模型条件的名义区间，不能视为独立确认。所有模型held后才评价，禁止held选超参数/早停/阈值。

八表support456/scope12/training3072/group_metrics2280/policy_summary30/paired8/source_replay912/execution1；一个总体进度3077步（3072训练+4载入评价阶段+1封存）。零新图像、CUDA搜索、完整轨迹、确认源；无新AUC，无采用门禁。当前原生SGLA及空间适配器不变，研究的是新增下一搜索父框修正映射。若新映射held仍不及zero，转输入信息/状态支持与目标；若优于zero，再验证真实后继及完整轨迹，不能由几何loss伪造IoU。

CPU单进程、单线程，后台脚本自动打包八表TXT。同步代码及217已审阅结果包后，在原服务器环境仓库根目录执行：

```bash
mkdir -p output/stage218
LOG="output/stage218/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash tools/run_stage218_capacity.sh > "$LOG" 2>&1 < /dev/null &
echo $! > output/stage218/run.pid
tail -f "$LOG"
```

默认输出 `output/stage218/matched_forecast_capacity_v1`，可用STAGE218_ROOT覆盖。STAGE217_ROOT默认cached_predictor_objective_v1，STAGE215_REPLAY_ROOT默认same_model_gpu_replay_v1，其他缓存沿用217设置。CPU不要求当前GPU型号/UUID一致，不需checkpoint。缺失来源即停止，不自动GPU补采。退出SSH后nohup继续；Ctrl-C只停止tail。已完成模型可复用，未完成模型从固定初始化重新训练。

```bash
python -B tools/stage218_capacity.py --check-results \
  --output-dir "${STAGE218_ROOT:-output/stage218/matched_forecast_capacity_v1}"
```

返回脚本打印的stage218_results_*.txt后，先核验来源、912复现及训练预算，再解读主比较和zero对照。此时尚无真实结果。
