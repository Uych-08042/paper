# Stage-211：固定模型的父框中心×尺寸分解

固定协议，2026-11-26。离线几何诊断，0训练、0新完整轨迹、确认集封存。真实GPU结果待运行。

210固定matched模型换原生父框，在全部原生恢复的即时IoU差+.2003、lost层+.4633，但同时存在网格无好框和好框漏选。本轮固定同一模型、同一批窗口，把父框变化分解成裁剪平移与尺度两个几何自由度，不增加网络或人为触发门槛。

## 冻结来源

210原包SHA256 `abba43753092b39109ba02ef762af177fc008404c6e9eb4c223faeb8c063d1d0`，decision `7cb046228b6674d1d59b0b3f848e6cc4377649de5954d32b71b2730f6d94aaf8`。核验12段/10表、完整210物理目录/旧代码/协议；严格复用原events22、38源support及324条selection，不重新选帧或改变lost/retained定义。自然恢复13次来自10源，lost9个seed窗来自6源；41个正常窗覆盖38源。这不是38源都有恢复证据。

只读210的native父框/matched模型、matched父框/matched模型原始NPZ/receipt，绑定frames/model state/parent来源，取真实输入父框而非输出框。四个208 own-pair matched权重沿用209加载器，scope/file/state须对应210。需要208模型与210原始审计目录，无需另传209轨迹目录。

## 几何干预

模式首字母为中心来源，次字母为宽高来源，n原生、m自身：mm、nm、mn、nn。设B=(x,y,w,h)，中心c=(x+w/2,y+h/2)；混合框为(c_x−w/2,c_y−h/2,w,h)。mm/nn直接复制来源数值，避免无必要重构造成端点浮点变化。nm/mn重新计算左上角，不能直接交换x/y和另一组w/h。

所有模式使用同一matched权重、原生模板、MLP、CENTER、Hann、裁剪/resize/回映/clip公式。每点从首帧模板初始化，设置来源frame_id，独立公开track一次，不写回续跑，不把GT输入模型或用它选择父框。网格由同次CENTER图的原cal_bbox/map/clip解码，标准选中格必须严格等公开输出。

原sample_target正方形裁剪边长为ceil(sqrt(w×h)×search_factor)，所以本轮尺寸项主要改变几何均值尺度、覆盖和resize后的目标分辨率，并非单独估计长宽比贡献；不能把宽高替换的结果全部解释成长宽比建模收益。

完整保存score/size/offset/grid、parent、factor、calls；报告真实整数裁剪left/top/side、中心重构数值误差、完整覆盖、标准IoU、网格best IoU与210原互斥类别。gt_width/height_network_px为GT整框宽高×实际resize_factor的名义投影尺寸，不代表遮挡后的可见目标尺寸。混合框存在浮点重构及取整边界效应，记录实际执行窗口，不能假设数学中心一致就一定有同一整数crop。保持宽高不变的中心对照名义投影尺寸应不变；实际取整和factor以执行为准。

## 复现与评价

同完整runtime时mm、nn全部原始数组逐字段精确复现210，不只框/层。新L20/L40或UUID另建根，全部模式同卡新跑，报告跨卡端点误差，不要求跨卡逐位一致；软件/config/checkpoint不变，续跑禁止改GPU/UUID/workers/源码。

全部固定点四模式各一次：正式1296次搜索；首个支持源额外24次串行，与独立worker同源全部数组严格一致。默认两spawn进程跨源、每源顺序，单总体进度条，父进程持lock并汇总。原始证据/receipt均绑定；并发墙钟不是隔离FPS/完整视频加速证据。

先每三帧窗口、再seed、再源等权，分all_recovery/retained/lost/normal，明确supported_sources、无支持空值、单源不算CI。报告四模式，四条件简单效应（nm−mm、nn−mn、mn−mm、nn−nm）、中心/尺寸顺序平均效应、差中差交互nn−nm−mn+mm、总父框效应nn−mm。两个平均效应之和等于总效应，是分配交互后的描述性分解，不是因果归因比例。bootstrap沿用源组20000次、seed24194，区间均名义探索性，不设采用门槛、不按结果选择主比较。

端点和选点固定，原生几何仍是离线参照，不是部署时已有的安全中心/尺寸预测。三帧独立输出达标不等于闭环恢复，不产生新AUC。若中心项更支持供给/输出，再设计可学习或受约束的中心更新；若尺寸项更重要，再检验尺度保护；供给恢复但漏选保留为读出问题，不一次同时改三模块。

## 运行

```bash
cd ~/workspace/paper
mkdir -p output/stage211
nohup bash tools/run_stage211_geometry_audit.sh > output/stage211/run.log 2>&1 < /dev/null &
echo $!
tail -f output/stage211/run.log
```

默认output/stage211/recovery_center_size_v1、GPU0、workers2。支持STAGE211_ROOT/GPU/WORKERS、STAGE210_ROOT/BUNDLE、STAGE208_ROOT/BUNDLE、CHECKPOINT及七数据入口覆盖。完成自动打包10表+manifest/decision为stage211_results_<UTC>_<PID>.txt。--check-results核验重打包。失败回传主报错及worker_logs，不放宽精确条件。
