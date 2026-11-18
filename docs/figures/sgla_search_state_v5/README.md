# 论文框架图 v5：真实图像、简洁 a/b 两分区

按用户2026-11-16要求修改v4：采用真实无人机街景及车辆裁剪，移除(c)离线提交检验和(d)独立时间轴，压缩(a)(b)中的解释性文字。保留必要的历史反馈箭头。原v4及其结果边界说明保留，不覆盖。

| 格式 | 英文论文版 | 中文版 |
| --- | --- | --- |
| 高清PNG，3840×1300 | [英文PNG](sgla_history_search_paper_en_v5.png) | [中文PNG](sgla_history_search_paper_zh_v5.png) |
| 可编辑SVG（内嵌实景图片） | [英文SVG](sgla_history_search_paper_en_v5.svg) | [中文SVG](sgla_history_search_paper_zh_v5.svg) |

## 图注草稿

**Overview of history-conditioned search and shared candidate selection.** (a) Historical predictions and response cues form a 32-D state descriptor. A state predictor estimates the mean and calibrated error scale, from which 32 antithetic Gaussian scenarios are constructed. The standard and mean proposals complete a pool of 34 search options. Greedy marginal-coverage allocation retains the standard proposal and selects three additional search regions from the current image. (b) Each search crop and the fixed first-frame template are processed by the same SGLA backbone, comprising six prefix blocks and one selected block among L7–L12, followed by the CENTER head. A shared 64–32–1 quality regressor scores the four candidate descriptors; the highest-scoring box is selected and updates the next-frame history. Photographs are illustrative crops of one real image, with manually drawn boxes; they are not consecutive video frames or tracking results. Image credit: *visdrone Dataset*, lab / Roboflow Universe, 2022 (source page lists CC BY 4.0), cropped and annotated for this diagram.

**历史条件搜索与共享候选选择框架。** (a) 将历史预测框及响应线索编码为32维状态，预测均值和经校准的误差尺度，构造32个正反配对高斯场景；加入标准与均值输入形成34个备选项，按边际覆盖增益保留标准区域并选择另外三个区域。(b) 四个搜索裁剪与固定首帧模板经过同一个SGLA，其六个前缀块之后仅执行L7～L12中的一个独立块，再经CENTER头解码。共享64–32–1质量回归器对候选描述打分，选中框更新下一帧历史。照片为同一张真实图片的示例裁剪，框为手工示意标注，不是连续视频帧、真实标注或实验预测。图片来源：lab在Roboflow Universe发布的visdrone Dataset（2022，来源页标注CC BY 4.0），经裁剪及加框。

## 图片出处与处理

- [公开样例及引用信息](https://universe.roboflow.com/lab-lpaya/visdrone-ocpsh)
- [原始图片URL](https://source.roboflow.com/WD8jyT4rYRekug52HV2T/01QmL7sdP4j5CPoNxUiB/original.jpg)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [本地原图](assets/visdrone_street_source.jpg)：1916×1078，420325字节，SHA256 `cb3e28faeb6dd05b4e65af096874bd4233ffc85920c0e780707bd4d1ffe2dcbb`。
- [逐裁剪来源、坐标及SHA记录](assets/provenance.json)。仅裁剪和加框，没有生成或修改场景内容。

公开样例来自图像检测数据展示页，仅作为架构插图素材；不作为本项目VisDrone-SOT实验中的某一指定视频、GT或预测结果，也未接入任何训练／评价／封存确认流程。图中的历史堆叠重复使用该图的不同裁剪，不能解读为真实连续帧。正式论文保留以上图注的示意及来源说明。

```bibtex
@misc{visdrone_ocpsh_figure_sample,
  title = {visdrone Dataset},
  author = {lab},
  year = {2022},
  publisher = {Roboflow Universe},
  url = {https://universe.roboflow.com/lab-lpaya/visdrone-ocpsh},
  note = {Photographic example used for framework illustration; accessed 2026-11-16}
}
```

## 方法与结果边界

主流程对应Stage-113的generated/shared结构：固定SGLA、已冻结的历史预测器和共享当前IoU评分器。四是本轮搜索预算，不代表已证明最优；图中的34→4包含标准父框与均值父框，重复裁剪可能减少实际唯一调用。图不展示旧PPO、扩散模型、在线模板更新，也不将Stage-120离线提交模块接入主干。当前完整收益与稳定／恢复保护仍未通过，论文式排版不表示方法已验证成功。

## 再生成

SVG文本、框、路径可编辑，真实照片以JPEG数据内嵌，传递SVG时不依赖外部图片路径。PNG为2倍像素并写入300dpi元数据。无需Pillow、网络、GPU或实验数据；需Python与Windows的System.Drawing／Arial／Microsoft YaHei字体。

```powershell
python -B docs/figures/sgla_search_state_v5/render_method_framework.py --rasterize
```

仅在需要从归档原图重建照片裁剪时运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File docs/figures/sgla_search_state_v5/prepare_assets.ps1
```

[绘图源文件](render_method_framework.py) · [照片裁剪脚本](prepare_assets.ps1) · [PNG导出器](rasterize_svg.ps1) · [完整说明版v4](../sgla_search_state_v4/README.md) · [Stage-113协议](../../experiments/06_sgla_uav_tracking/STAGE113_FROZEN_SHARED_CLOSED_LOOP.md)
