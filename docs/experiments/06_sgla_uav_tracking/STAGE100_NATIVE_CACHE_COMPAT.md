# Stage-100 原生q0缓存兼容修复

## 报错与原因

服务器运行在`stage100_online_scale.py`的原生参照检查处退出：`Call accounting shape changed: block_calls`。Stage-76历史q0轨迹始终选择q0，但每帧仍计算并保存四个候选，`block_calls`实际为`[N,4,12]`。Stage-100原检查预期单搜索的`[N,1,12]`，所以在本次记录格式上失败。

原本的本地测试用单搜索合成参照，没有覆盖这个历史格式。原生q0输出框没有因此被判定错误，当前故障是缓存表达与单搜索评价接口不一致。

## 修复范围

新增`tools/stage100_native_compat.py`与`tools/run_stage100_native_compat.sh`。原Stage-100 Python、原Bash入口及所有上游冻结实验源码保持字节不变。兼容入口委托原参数解析、硬件身份检查、来源哈希、采集、在线规则和评价，只在原生参照读入时建立q0视图。

兼容步骤为：

1. 接受且仅接受`[N,1,12]`或`[N,4,12]`的原生调用布局，先对**全部记录的候选调用**执行原有检查。
2. 核对每帧选择恒为0，发布框精确等于q0候选框，候选银行的形状、有限性及尺寸合法。
3. 复制q0的父框、候选框、裁剪和调用字段，得到单候选视图，再按单搜索原生轨迹重建逐帧历史与裁剪检查。
4. 保留`native_source_slots`及`native_source_head_calls`，区别历史四候选实际调用和本轮q0评价视图。原生好候选比例也只评q0，避免把未使用的q1～q3计作原生候选供给。

输入数组、来源文件、旧manifest及已提交缓存均不被改写。历史四次调用并未被删除；复用它们时，本轮新增推理仍为0。换卡时仍按原协议重测所有对照，预算和启停规则保持原值。兼容转换自身没有图像读取或新增网络推理。

原manifest已经绑定旧代码哈希，因此不直接修改旧脚本。新入口在原计划发布后、进入轨迹循环前写入`stage100_native_compat.json`，绑定原计划指纹、原代码哈希及适配代码／脚本哈希。最终decision新增`native_compatibility`和逐视频`native_source_views`，关联兼容回执并列出原始槽位数、历史调用数和q0视图调用数。完成核验同时检查原报告、兼容回执和源码哈希。

## 服务器继续运行

同步本轮新增文件、激活`sglatrack-test`环境后，在仓库根目录执行：

```bash
bash tools/run_stage100_native_compat.sh
```

默认继续使用原目录`output/stage100/online_scale_episode_pulse_v1`。不用删除manifest或改名旧输出。原有可校验缓存继续复用；若中断后GPU型号或UUID变化，沿用原硬件规则，用`STAGE100_ROOT`指定新的独立目录。

后台执行可用：

```bash
mkdir -p output/stage100
nohup bash tools/run_stage100_native_compat.sh > output/stage100/run_stage100_native_compat.log 2>&1 < /dev/null &
tail -f output/stage100/run_stage100_native_compat.log
```

检查没有旧Stage-100作业仍在运行后再启动，避免两个进程竞争同一个输出目录。已退出的异常进程会释放原写入锁，不需要手工删除文件。

完成后回传新脚本打印的`stage100_results_*.txt`。包内增加兼容回执，其余仍为decision和六张既定表。仅核验及重新打包可执行：

```bash
bash tools/run_stage100_native_compat.sh --pack-only
```

继续使用旧`run_stage100_online_scale.sh`不会安装兼容视图，请使用新的兼容入口。

## 验证与状态

七项CPU测试通过。新增回归先复现四槽q0触发的原报错，再验证全部槽位调用仍被检查、q0输出精确保留、输入数组不变、混合一槽／四槽来源、相同卡复用和换卡重测、旧manifest及代码字节不变、所有缓存无推理续跑，以及缓存篡改拒绝。其余在线因果、退出／重新允许和恢复集合保护测试也保持通过。网络输出由合成观测器提供，几何、解码、特征、记忆、缓存与发布流程使用实际实现。

Bash语法和CLI入口通过检查。服务器完整实验仍待用兼容入口继续运行，本次修复没有产生正式方法成绩。
