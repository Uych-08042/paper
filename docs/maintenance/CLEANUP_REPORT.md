# 仓库整理报告

日期：2026-10-29（Asia/Hong_Kong）  
范围：`E:\Program Files\paper\paper` 本地工作树  
整理前Git基点：`91052cd71d49d1397197ea9d5f532f12d37d2c91`

## 已执行的整理

| 操作 | 数量 | 说明 |
|---|---:|---|
| 保留原位置及字节 | 505个文件 | 核心模型、配置、当前源码依赖、原始权重、原始跟踪结果等 |
| 归档历史Python文件 | 136个 | 92个脚本＋44个测试，移至`archive/legacy_tracking/`，内容不变 |
| 集中阶段说明 | 61份 | 原根目录58份Stage说明＋SINGLE_GPU说明＋tracking内Stage-19/20说明，移至4个分类目录 |
| 归档旧写作材料 | 3份 | 旧组会、Introduction及创新性梳理移至`docs/archive/manuscript/`，内容不变 |
| 删除Python字节码 | 194个 | 包含旧Python版本及已无对应源码的缓存 |
| 删除绘图汇总缓存 | 1个 | `output/test/result_plots/uav123/eval_data.pkl`；原始预测和时间文件保留 |
| 编辑导航与忽略规则 | 2个 | README增加交接入口；.gitignore增加Python/测试/Notebook缓存规则 |

共移动200个文件，删除195个可重建缓存。工作树文件字节减少1,070,674字节，约1.02 MiB；没有重写Git历史，Git对象库不会因此立即缩小。约96 MiB的原始checkpoint仍是必要初始化与复现依赖，未删除。

所有移动/删除的原路径、目标路径、SHA256和原因均记录于[CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json)。待后续明确需要历史路线时，可以恢复归档源码；无需下载脚本包。

## 当前保留策略

以原训练/测试/评估工具、Stage-44入口与其测试、tracking loss测试为根，解析直接导入、局部导入以及字符串中的冻结代码文件名，保守保留106个`tracking`模块，其中92个为阶段源码/测试。冻结实验的Python源码未被改写。

该集合用于保持当前依赖和复核能力，不是所有未来设计的最小理论集合。已停止路线中不在该集合的文件移入归档，避免直接删掉可能需要审计的历史证据。

原始`lib/`、YAML、环境文件、数据集适配器和权重保留。`output/stage4/`、`output/stage5/`、`output/test/tracking_results/`内容和路径均保持不变。本地没有完整Stage-16～44实验结果，因此没有声称清理或重新核验这些服务器产物。

## 新增入口与维护工具

- [工作交接文档](../../WORK_HANDOFF.md)：当前目标、Stage-16～44结论、Stage-44精确状态、路径、运行口径、停止条件与协作约定。
- [实验说明索引](../README.md)：61份原始阶段说明的分类导航。
- [历史代码恢复说明](../../archive/README.md)：恢复归档脚本至原路径。
- `tools/check_workspace.py`：核对整理快照、原/归档字节与活动依赖。
- `tools/restore_legacy_tracking.py`：默认仅预览；加`--apply`恢复全部归档脚本，拒绝覆盖不同内容。
- `tools/test_workspace_maintenance.py`：恢复预检、幂等、越界和覆盖保护测试。
- `pytest.ini`：活动测试路径限定为tracking/tools，排除archive/output/docs，避免把历史归档误作当前测试集。

## 整理后验证

| 检查 | 结果 |
|---|---|
| 505个原位文件字节＋200个归档/文档移动哈希 | PASS |
| 106个保留tracking模块的导入/冻结文件名依赖 | PASS，无缺失归档依赖 |
| 缓存删除状态 | 195个旧缓存均不存在；未恢复任何历史脚本 |
| 历史恢复工具 | 2项测试PASS；只在仓库内临时夹具上实际复制，没有恢复真实归档 |
| Stage-44真实CPU测试 | 3项全部PASS，28.100秒，无跳过 |
| Stage-44轻量帮助命令 | PASS |

Stage-44测试包含原生选层/输出复现、冻结候选中的前缀梯度传递、真实匹配训练、精确续训、OOF隔离、未来GT隔离、16条新短测试轨迹、结果复用及文件变更拒绝。旧Stage-43完成产物边界在测试中使用合成适配器，不等于远程来源全链实际复核；没有执行服务器GPU训练或精度评估。

维护测试首次使用系统临时目录时受Windows沙箱权限限制，现已显式使用仓库tools下的临时目录并验证其位置；修复后2项测试通过，没有修改实验源码或放宽实验审计。

## 后续使用

```bash
python -B tools/check_workspace.py
python -B tools/test_workspace_maintenance.py

# 只有明确要恢复旧路线时执行：
python -B tools/restore_legacy_tracking.py
python -B tools/restore_legacy_tracking.py --apply
```

整理快照用于证明本次操作保留了哪些文件；未来主动修改被记录文件时应维护/解释该快照。恢复工具在完整批次预检后复制，已存在且相同的文件跳过，内容不同则拒绝覆盖。

本次没有新建模型实验，没有修改服务器文件，没有自动stage、commit或push。工作树中的移动与删除需要通过正常Git流程纳入后再同步；本报告不包含强制清空或覆盖服务器工作树的命令。
