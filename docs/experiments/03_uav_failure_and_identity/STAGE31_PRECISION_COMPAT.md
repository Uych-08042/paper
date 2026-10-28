# Stage-31：原始 float32 响应分数的回放审计修复

## 原因与修复范围

原 `NativeObserver` 输出的加权响应是 float32。`run_video` 在线计算特征时沿用 float32 做响应比值，然后把分数与特征写进 float64 缓存。旧审计从缓存重算时使用了 float64 除法，导致 `relative_response` 出现约 1e-9–1e-8 差异。原单元测试的分数为 float64，未覆盖这个边界。

新入口 `tracking/stage31_precision_compat.py` 仅为**审计输入**恢复分数的 float32 dtype。恢复前要求 float64→float32→float64 数值完全相等，不能损失信息。随后仍调用原始完整审计，保留 `atol=1e-9, rtol=0`，以及原有框、标签、历史、特征和选框一致性检查。

在线前向、在线特征计算、关联器选择、训练特征、训练参数、数据划分和进入门槛均保持原样。没有改写原来的六个冻结脚本、manifest 或 NPZ 文件，也没有通过重算原契约指纹绕过校验。

## 保留原实验，使用新入口

Git 同步新增文件后，沿用当前 `STAGE31_ROOT` 和权重路径。**不重新建表，不删除已有缓存。**

```bash
python tracking/test_stage31_precision_compat.py

python tracking/stage31_precision_compat.py collect \
  --input "${STAGE31_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage31_precision_compat.py train \
  --input "${STAGE31_ROOT:?}"
```

原采集在审计成功后才保存一条完整视频。这次 `uav0000016_00000_s` 在保存前失败，因此该条会重新采集一次；其他已经成功保存的完整 fit 视频会被核验并复用。

之后 checking 与 summary 也使用同一兼容入口，保证读取旧缓存时的审计精度一致：

```bash
python tracking/stage31_precision_compat.py evaluate \
  --input "${STAGE31_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0

python tracking/stage31_precision_compat.py summary \
  --input "${STAGE31_ROOT:?}"
```

仅在需要重做工程 smoke 时使用 `smoke` 子命令；此前已通过的 smoke 无需重跑。子命令代替旧脚本的 `--mode`，不要同时传入 `--mode`。其他参数继续转交原脚本，例如 `collect --sequence ...`、`summary --allow_incomplete`。

## 补充溯源记录

原契约保持原指纹，新执行版本通过以下记录显式绑定：

- `stage31_precision_compat.json`：原契约指纹、原六个代码哈希、新入口代码哈希、NumPy 版本、源/存储/回放 dtype 和未放宽的容差；
- `precision_compat/collect_audit.json`、`train_audit.json`、`evaluate_audit.json`、`summary_audit.json`：最近一次相应命令的状态、已审计序列、float64 与 float32 回放差异、原有模型/运行/决策文件哈希。

报告结果时保留这些补充记录。它们记录的是审计实现修复，不是新的实验结果或修改后的判定门槛。

可核对：

```bash
cat "$STAGE31_ROOT/precision_compat/collect_audit.json"
cat "$STAGE31_ROOT/precision_compat/train_audit.json"
```

源 float32 回放差异应恢复为 0 或保持在原容差内。若出现分数不能无损表示为 float32、其他特征不一致、历史/选框不一致等错误，兼容入口仍会拒绝；不能继续放宽检查来强行通过。

## 回归测试

测试覆盖实际 float32 分数进入 float64 缓存后的误差复现、只读 dtype 视图、其他特征损坏仍被拒绝、非无损分数拒绝、近似并列时的原选框保持、审计补丁退出后恢复，以及完整训练/汇总中原 manifest 和缓存字节不变。
