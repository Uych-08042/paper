"""Draw an editable, evidence-aligned SGLA method diagram.

The diagram is an illustration, not an experiment or an execution trace. Raster
export uses Pillow to render the SVG primitives generated here, without network
requests. SVG generation itself uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
W, H = 2160, 1270
C = dict(
    ink="#172D42", muted="#596C7F", line="#D7E1E9", panel="#FBFCFE",
    teal="#087F8C", teal_bg="#EAF7F7", blue="#3568AB", blue_bg="#F0F5FC",
    amber="#A86B18", amber_bg="#FFF8EB", gray="#69798A", gray_bg="#F1F4F7",
    white="#FFFFFF", red="#C36557",
)


class SVG:
    def __init__(self, lang: str):
        self.lang = lang
        self.parts: list[str] = []
        self.family = "Microsoft YaHei, SimHei, Arial, Cambria, sans-serif" if lang == "zh" else "Arial, Helvetica, Cambria, sans-serif"

    def add(self, value: str):
        self.parts.append(value)

    def rect(self, x, y, w, h, fill="white", stroke=None, r=12, dash=False, sw=1.6):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"'
                 + (' stroke-dasharray="8 6"' if dash else '') + '/>')

    def text(self, x, y, value, size=21, color=None, weight=400, anchor="start", italic=False):
        self.add(f'<text x="{x}" y="{y}" font-family="{self.family}" font-size="{size}" '
                 f'fill="{color or C["ink"]}" font-weight="{weight}" text-anchor="{anchor}"'
                 + (' font-style="italic"' if italic else '') + f'>{html.escape(value)}</text>')

    def lines(self, x, y, values, size=21, gap=29, **kwargs):
        for i, line in enumerate(values):
            self.text(x, y + i * gap, line, size=size, **kwargs)

    def path(self, points, color=None, arrow=False, dash=False, width=2.4):
        col = color or C["gray"]
        d = "M " + " L ".join(f"{x},{y}" for x, y in points)
        marker = f' marker-end="url(#{"amber" if col == C["amber"] else "teal" if col == C["teal"] else "gray"})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}" stroke-linecap="round" '
                 f'stroke-linejoin="round"' + (' stroke-dasharray="8 6"' if dash else '') + marker + '/>')

    def circle(self, x, y, r, fill):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

    def pill(self, x, y, w, label, fill=None, color=None, size=18, h=34):
        self.rect(x, y, w, h, fill or C["teal_bg"], r=h / 2)
        self.text(x + w / 2, y + h * .69, label, size, color or C["teal"], 600, "middle")

    def scene(self, x, y, w, h, result=False):
        # A synthetic overhead scene: never presented as a VisDrone frame.
        self.rect(x, y, w, h, "#E8EEF0", r=7)
        self.add(f'<path d="M {x},{y+h*.84} L {x+w},{y+h*.25}" stroke="#C5D0D5" stroke-width="{h*.36}"/>')
        self.add(f'<path d="M {x},{y+h*.84} L {x+w},{y+h*.25}" stroke="#F7FAFB" stroke-width="2" stroke-dasharray="12 9"/>')
        for xx, yy, ww, hh in ((.07,.09,.21,.23),(.48,.04,.2,.19),(.70,.66,.23,.22)):
            self.rect(x+w*xx, y+h*yy, w*ww, h*hh, "#CBD8CF", r=3)
        cx, cy = x+w*.54, y+h*.52
        self.rect(cx-w*.05, cy-h*.075, w*.1, h*.15, C["ink"], r=3)
        col = C["teal"] if result else C["blue"]
        self.rect(cx-w*.085, cy-h*.13, w*.17, h*.26, "none", col, r=1, sw=2.6)

    def candidate(self, x, y, k, label, center_lag, history_size):
        self.rect(x, y, 184, 126, C["white"], C["line"], r=8)
        self.text(x+14, y+28, f"{k}  {label}", 18, C["blue"], 600)
        self.rect(x+16, y+44, 69, 58, "#F1F4F7", r=4)
        cx = x+49 + (-6 if center_lag else 0)
        cy = y+74 + (3 if center_lag else 0)
        cw, ch = (39, 35) if history_size else (51, 25)
        self.rect(cx-cw/2, cy-ch/2, cw, ch, "none", C["blue"], r=1, sw=2)
        self.rect(x+45, y+68, 10, 12, C["ink"], r=1)
        self.circle(cx, cy, 2.5, C["blue"])
        self.text(x+98, y+67, "c(t−2)" if center_lag else "c(t−1)", 17)
        self.text(x+98, y+94, "s(hist)" if history_size else "s(t−1)", 17)

    def finish(self):
        defs = ''.join(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{C[name]}"/></marker>' for name in ("teal", "gray", "amber"))
        title = "SGLA自身历史搜索状态选择：初始论文方法框架" if self.lang == "zh" else "History-guided search-state selection for SGLA UAV tracking"
        desc = ("Solid teal arrows show the response-selection route. Amber dashed boxes show the tested learned-scoring alternative. "
                "Each crop is processed separately by the same fixed SGLA-based tracker E: six prefix blocks and one independent candidate. "
                "Only accepted past predictions enter history. Future images and ground truth are used only for offline scorer labels.")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">'
                f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><defs>{defs}</defs>'
                + ''.join(self.parts) + '</svg>\n')


def draw(lang):
    z = lang == "zh"
    s = SVG(lang)
    def tr(zh, en):
        return zh if z else en

    s.rect(0, 0, W, H, C["white"], r=0)
    s.text(54, 63, tr("基于自身历史的 SGLA 搜索状态选择", "History-guided Search-State Selection for SGLA UAV Tracking"), 36, weight=700)
    s.pill(1940, 32, 166, tr("论文框架初稿 v1", "METHOD DRAFT v1"), C["gray_bg"], C["gray"], 16)
    s.text(56, 106, tr("研究问题：预测框决定下一帧裁剪；中心与尺寸误差可能改变响应竞争并累积为持续失跟。", "Problem: predicted boxes determine future crops; center and scale errors can alter response competition and cause persistent drift."), 23, C["muted"])

    s.path([(57, 147), (104, 147)], C["teal"], True, width=3)
    s.text(117, 154, tr("当前保留主线", "Retained main route"), 20)
    s.path([(365, 147), (412, 147)], C["amber"], True, dash=True)
    s.text(425, 154, tr("已测试的学习支线；待完善", "Tested learned alternative; exploratory"), 20)
    s.rect(929, 135, 22, 22, C["gray_bg"], C["gray"], r=4)
    s.text(963, 154, tr("同一个固定跟踪器，逐候选独立推理", "One fixed tracker, evaluated separately per crop"), 20, C["muted"])

    # Panel a: the online pipeline.
    s.rect(40, 184, 2080, 673, C["panel"], C["line"], r=17)
    s.text(65, 224, tr("(a) 在线：自身历史 → 中心×尺寸候选 → SGLA → 选择 → 自身历史", "(a) Online: own history → center × size proposals → SGLA → selection → own history"), 25, weight=700)
    s.pill(1750, 202, 340, tr("在线 GT 仅限首帧初始化", "GT only for first-frame initialization"), C["gray_bg"], C["muted"], 17)

    # Current image + accepted history.
    s.rect(66, 281, 276, 436, C["white"], C["line"])
    s.text(86, 317, tr("输入与历史", "Image & own history"), 22, weight=700)
    s.scene(87, 338, 234, 129)
    s.text(204, 491, tr("当前图像 I(t)（示意）", "Current image I(t), schematic"), 18, C["muted"], anchor="middle")
    s.path([(87, 508), (321, 508)], C["line"], width=1)
    s.text(86, 540, tr("已接受预测 H(t−1)", "Accepted history H(t−1)"), 20, weight=600)
    for i in range(5):
        s.rect(88+i*47, 561-i*2, 37, 33, C["teal_bg"] if i == 4 else C["gray_bg"], C["teal"] if i == 4 else C["line"], r=4)
        s.rect(98+i*47, 571-i*2, 16, 12, "none", C["teal"] if i == 4 else C["gray"], r=1)
    s.lines(87, 627, tr(["最近至多 5 个框", "当前中心 / 滞后中心", "当前尺寸 / 历史尺寸"], ["Up to 5 past accepted boxes", "Latest / lagged center", "Latest / historical size"]), 19, 28, color=C["muted"])

    # Four parent boxes, grouped as a 2 x 2 factorial state set.
    s.rect(382, 281, 426, 436, C["blue_bg"], "#B9CDE5")
    s.text(403, 317, tr("① 构造搜索父框", "① Construct search parents"), 23, C["blue"], 700)
    s.text(403, 348, tr("中心 × 尺寸的四种组合", "Four center × size combinations"), 20, C["muted"])
    s.candidate(402, 368, 0, tr("当前", "Current"), False, False)
    s.candidate(604, 368, 2, tr("滞后中心", "Lag center"), True, False)
    s.candidate(402, 512, 1, tr("历史尺寸", "History size"), False, True)
    s.candidate(604, 512, 3, tr("二者组合", "Combined"), True, True)
    s.text(402, 674, tr("s(hist) = exp[median(log s)]", "s(hist) = exp[median(log s)]"), 21, C["blue"])
    s.text(402, 702, tr("仅用已接受历史；探测不写入历史", "Accepted history only; probes stay separate"), 17.5, C["muted"])
    s.path([(342, 579), (381, 579)], C["teal"], True, width=2.8)

    # The same frozen E network is run once for each candidate crop.
    s.rect(848, 281, 476, 436, C["white"], C["line"])
    s.text(869, 317, tr("② 固定的 SGLA 跟踪器 E", "② Fixed SGLA-based tracker E"), 23, weight=700)
    s.pill(870, 337, 432, tr("原首帧模板 z₀，始终固定", "Original first-frame template z₀ stays fixed"), C["gray_bg"], C["gray"], 18, h=39)
    s.rect(870, 396, 432, 52, C["blue_bg"], r=7)
    s.text(1086, 430, tr("按父框裁剪 + 模板组成输入对", "Parent-defined search crop + template pair"), 20, anchor="middle")
    s.path([(1086, 376), (1086, 395)], C["gray"], True)
    s.path([(321, 390), (356, 390), (356, 258), (830, 258), (830, 411), (869, 411)], C["gray"], True, width=2)
    s.text(553, 250, tr("当前图像供四次裁剪使用", "Current image for all four crops"), 17, C["muted"], anchor="middle")
    s.path([(808, 435), (869, 435)], C["teal"], True, width=2.8)
    s.path([(1086, 448), (1086, 465)], C["gray"], True)
    s.rect(870, 466, 432, 50, C["gray_bg"], r=7)
    s.text(893, 499, tr("6 个前缀块", "6 prefix blocks"), 20, weight=600)
    for i in range(6):
        s.rect(1067+i*37, 479, 29, 24, "#DEE7EE", r=3)
        s.text(1081+i*37, 496, str(i+1), 14, C["muted"], anchor="middle")
    s.path([(1086, 516), (1086, 526)], C["gray"], width=1.8)
    s.text(1086, 540, tr("原选择器：从独立候选中选 1 个", "Original selector: choose ONE independent candidate"), 17, C["muted"], anchor="middle")
    for i in range(6):
        active = i == 2  # L9 is illustrative, not a claim of a fixed exit.
        s.rect(870+i*73, 552, 67, 36, C["teal_bg"] if active else C["gray_bg"], C["teal"] if active else C["line"], r=5)
        s.text(903+i*73, 577, f"L{i+7}", 19, C["teal"] if active else C["gray"], 600 if active else 400, "middle")
    s.text(871, 614, tr("仅 1 个；高亮为示意", "ONE block; illustrative"), 15, C["muted"])
    s.path([(1049, 588), (1049, 625), (1086, 625), (1086, 627), (972, 627), (972, 635)], C["gray"], True, width=1.8)
    s.path([(1086, 627), (1197, 627), (1197, 635)], C["gray"], True, width=1.8)
    s.rect(870, 636, 204, 56, C["gray_bg"], r=7)
    s.rect(1092, 636, 210, 56, C["gray_bg"], r=7)
    s.text(972, 660, tr("原响应分支", "Response head"), 18, anchor="middle")
    s.text(972, 683, "Mᵏ → Hann", 17, C["muted"], anchor="middle")
    s.text(1197, 660, tr("尺寸 / 偏移解码", "Size / offset decode"), 18, anchor="middle")
    s.text(1197, 683, tr("候选输出框 bᵏ(t)", "Candidate box bᵏ(t)"), 17, C["muted"], anchor="middle")
    s.path([(972, 692), (972, 705), (1313, 705), (1313, 665)], C["teal"], width=2.2)
    s.path([(1302, 665), (1343, 665), (1343, 400), (1385, 400)], C["teal"], True, width=2.8)

    # Alternative scoring policies: no fusion/safety gate is invented.
    s.rect(1364, 281, 400, 436, C["white"], C["line"])
    s.text(1385, 317, tr("③ 候选评分与选择", "③ Score the candidates"), 23, weight=700)
    s.rect(1386, 348, 356, 98, C["teal_bg"], "#8DC9CB", r=9)
    s.text(1564, 381, tr("主线：Hann 响应峰值", "Main route: Hann response peak"), 20, C["teal"], 700, "middle")
    s.text(1564, 416, "rᵏ = max(Hann ⊙ Mᵏ)", 22, C["teal"], anchor="middle")
    s.text(1564, 475, tr("或：学习评分（独立对比策略）", "OR: learned scoring, a separate policy"), 17.5, C["amber"], anchor="middle")
    s.rect(1386, 490, 356, 70, C["amber_bg"], C["amber"], dash=True, r=9)
    s.text(1564, 518, tr("24 维响应 / 几何 / 历史特征 φᵏ", "24-D response / geometry / history φᵏ"), 18, C["amber"], 600, "middle")
    s.text(1564, 546, tr("相对 current 特征 + fit-only 缩放", "Current-relative features; fit-only scaling"), 16.5, C["muted"], anchor="middle")
    s.path([(1343, 526), (1385, 526)], C["amber"], True, dash=True)
    s.path([(1564, 560), (1564, 582)], C["amber"], True, dash=True)
    s.rect(1386, 584, 356, 78, C["amber_bg"], C["amber"], dash=True, r=9)
    s.text(1564, 614, tr("即时收益 θ₁ / 8 帧收益 θ₈", "Immediate θ₁ / 8-frame utility θ₈"), 20, C["amber"], 600, "middle")
    s.text(1564, 643, tr("参数由离线匹配分支的回报监督得到", "Parameters fitted from offline branch returns"), 17, C["amber"], anchor="middle")
    s.text(1564, 696, tr("两条评分路径二选一，不相加", "Use one scoring rule; do not combine them"), 17, C["muted"], anchor="middle")
    s.path([(1742, 400), (1826, 400)], C["teal"], True, width=2.8)
    s.path([(1742, 623), (1784, 623), (1784, 441), (1826, 441)], C["amber"], True, dash=True)

    # Accepted output and the causal feedback loop.
    s.rect(1804, 281, 286, 436, C["white"], C["line"])
    s.text(1826, 317, tr("④ 接受选中预测", "④ Accept output"), 22, weight=700)
    s.scene(1828, 349, 238, 135, result=True)
    s.text(1947, 518, "k* = argmaxₖ rᵏ", 23, C["teal"], 600, "middle")
    s.text(1947, 555, "b(t) = bᵏ*(t)", 24, C["teal"], 600, "middle")
    s.lines(1828, 600, tr(["接受原始解码输出", "仅选中框写入历史", "用于下一帧搜索"], ["Accept the decoded box", "Update accepted history only", "Use it for the next frame"]), 18.5, 31, color=C["muted"])
    # Cover incoming lines inside the scene; their arrowheads terminate on its edge.
    s.path([(1947, 717), (1947, 784), (204, 784), (204, 718)], C["teal"], True, width=3)
    loop_label = tr("闭环反馈：选中预测改变下一帧的候选、裁剪与输入证据", "Closed-loop feedback: the accepted box changes future proposals, crops and visual evidence")
    s.rect(632, 764, 991, 39, C["panel"], r=0)
    s.text(1128, 791, loop_label, 21, C["teal"], 600, "middle")
    s.text(67, 836, tr("预算：四候选策略 = 每帧 4 次完整搜索；current 对照 = 1 次。各候选不共享搜索特征。", "Budget: 4 full searches per frame for each selector; 1 for current-only control. Search features are not shared across crops."), 19, C["muted"])

    # Panel b: supervision is explicitly offline and belongs only to the learned branch.
    s.rect(40, 885, 2080, 327, C["white"], C["line"], r=17)
    s.text(65, 925, tr("(b) 离线：为学习支线建立匹配起点的收益监督", "(b) Offline: matched-start utility supervision for the learned alternative"), 25, weight=700)
    s.pill(1656, 902, 433, tr("未来图像与 GT 仅在 fit 离线标注使用", "Future images and GT: fit-split labels only"), C["amber_bg"], C["amber"], 17)
    boxes = [(66, 350), (456, 470), (966, 550), (1556, 534)]
    for x, width in boxes:
        s.rect(x, 958, width, 194, C["amber_bg"], "#E5CFA8", r=10, dash=True)

    s.text(87, 990, tr("Fit 组自身轨迹", "Own trajectories on fit groups"), 21, weight=600)
    s.lines(87, 1027, tr(["固定 E 头生成自身历史", "均匀抽取锚点，不按失效筛选", "每个锚点构造相同四候选"], ["Fixed E generates its own history", "Uniform anchors; no failure mining", "Same four proposals at each anchor"]), 18, 30, color=C["muted"])
    s.pill(87, 1118, 307, tr("按源视频分组隔离 fit / held-out", "Source-group fit / held-out isolation"), C["white"], C["muted"], 15.5, 24)
    s.path([(416, 1053), (455, 1053)], C["amber"], True, dash=True)

    s.text(476, 990, tr("4 条匹配起点的 8 帧分支", "Four matched-start 8-frame branches"), 21, weight=600)
    for row in range(4):
        yy = 1008 + row*24
        s.text(479, yy+16, str(row), 15, C["amber"])
        for col in range(8):
            s.rect(506+col*48, yy, 39, 15, "#E4BC79" if col == 0 else "#DCE4EA", r=3)
            if col < 7:
                s.path([(545+col*48, yy+7.5), (553+col*48, yy+7.5)], C["gray"], width=1)
    s.text(476, 1126, tr("仅首帧改父框；随后按普通 E 自身推进", "Change the first parent only; then ordinary E rollout"), 17, C["muted"])
    s.path([(926, 1053), (965, 1053)], C["amber"], True, dash=True)

    s.text(987, 990, tr("相对 current 的质量标签", "Quality labels relative to current"), 21, weight=600)
    s.text(988, 1032, "y₁ᵏ = IoUₜᵏ − IoUₜ⁰", 24, C["amber"])
    s.text(988, 1070, "y₈ᵏ = (1/8) Σⱼ₌₀⁷ [IoUₜ₊ⱼᵏ − IoUₜ₊ⱼ⁰]", 22, C["amber"])
    s.text(987, 1102, tr("预测产生后才附加 fit GT，IoU 为 0–1", "Attach fit GT after predictions; IoU is on 0–1 scale"), 17, C["muted"])
    s.text(987, 1130, tr("这是一次干预收益，不是连续选择的最优价值", "One-action return, not the value of repeated selection"), 17, C["muted"])
    s.path([(1516, 1053), (1555, 1053)], C["amber"], True, dash=True)

    s.text(1577, 990, tr("两种目标，各拟合一个线性评分器", "Fit a linear scorer for each target"), 21, weight=600)
    s.text(1578, 1031, "rᵏ = θᵀ [(φᵏ − φ⁰) / RMSfit]", 23, C["amber"])
    s.lines(1578, 1067, tr(["无截距岭回归，λ = 0.01；current 得分为 0", "相同特征、样本、容量与正则，仅标签不同", "θ₁ / θ₈ 用于上方虚线评分；SGLA 参数不更新"], ["Ridge, no intercept; λ = 0.01; current score = 0", "Matched features, samples, capacity and regularizer", "θ₁ / θ₈ serve the dashed route; SGLA stays fixed"]), 17, 29, color=C["muted"])
    s.text(67, 1191, tr("两种学习策略与响应控制具有相同四候选预算；离线 GT 和未来帧不进入在线控制器。", "Learned policies and response control use the same four-probe budget; offline GT and future frames never enter the online actor."), 18, C["muted"])

    s.text(54, 1244, tr("E：已有两帧末帧保持的几何适配头（Stage-54 / 58），本轮固定；候选构造与选择对应 Stage-60。虚线是已测试、待完善的学习评分替代策略。", "E: existing geometry-adapted endpoint-preservation tracker (Stages 54 / 58), fixed here. Proposals and scoring follow Stage 60; dashed scoring is a tested exploratory alternative."), 17, C["muted"])
    return s.finish()


def rasterize(svg_path, scale):
    from PIL import Image, ImageDraw, ImageFont

    root = ET.parse(svg_path).getroot()
    image = Image.new("RGB", (W*scale, H*scale), "white")
    draw = ImageDraw.Draw(image)
    font_root = Path("C:/Windows/Fonts")
    fonts = {}
    glyph_support = {}

    def font_for(name, size):
        key = name, round(size*scale)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(font_root/name), key[1])
        return fonts[key]

    def supports(name, size, char):
        key = name, round(size*scale), char
        if key not in glyph_support:
            font = font_for(name, size)
            mask = font.getmask(char)
            missing = font.getmask("\U0010ffff")
            glyph_support[key] = (mask.size, bytes(mask)) != (missing.size, bytes(missing))
        return glyph_support[key]

    def line(points, color, width, dashed=False):
        if not dashed:
            draw.line([(x*scale, y*scale) for x, y in points], fill=color, width=max(1, round(width*scale)), joint="curve")
            return
        offset = 0.
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = math.hypot(x2-x1, y2-y1)
            if length == 0:
                continue
            pos = 0.
            while pos < length:
                phase = offset % 14
                step = min(length-pos, (8-phase) if phase < 8 else (14-phase))
                if step < 1e-8:
                    step = min(length-pos, 1e-7)
                if phase < 8:
                    segment = [(x1+(x2-x1)*pos/length, y1+(y2-y1)*pos/length),
                               (x1+(x2-x1)*(pos+step)/length, y1+(y2-y1)*(pos+step)/length)]
                    line(segment, color, width)
                pos += step
                offset += step

    for node in root:
        tag = node.tag.rsplit("}", 1)[-1]
        a = node.attrib
        if tag in ("defs", "title", "desc"):
            continue
        fill = a.get("fill", "none")
        stroke = a.get("stroke", "none")
        sw = float(a.get("stroke-width", 1))
        dashed = "stroke-dasharray" in a
        if tag == "rect":
            x, y, width, height = [float(a[k]) for k in ("x", "y", "width", "height")]
            radius = float(a.get("rx", 0))
            box = (round(x*scale), round(y*scale), round((x+width)*scale), round((y+height)*scale))
            draw.rounded_rectangle(box, radius=round(radius*scale), fill=None if fill == "none" else fill,
                                   outline=None if stroke == "none" or dashed else stroke, width=max(1, round(sw*scale)))
            if dashed:
                points = []
                for cx, cy, start in ((x+width-radius,y+radius,-90), (x+width-radius,y+height-radius,0),
                                      (x+radius,y+height-radius,90), (x+radius,y+radius,180)):
                    for i in range(13):
                        angle = math.radians(start+i*90/12)
                        points.append((cx+radius*math.cos(angle), cy+radius*math.sin(angle)))
                line(points+[points[0]], stroke, sw, True)
        elif tag == "circle":
            x, y, r = [float(a[k]) for k in ("cx", "cy", "r")]
            draw.ellipse(((x-r)*scale, (y-r)*scale, (x+r)*scale, (y+r)*scale), fill=fill)
        elif tag == "path":
            values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", a["d"])]
            points = list(zip(values[::2], values[1::2]))
            line(points, stroke, sw, dashed)
            if "marker-end" in a:
                (x0, y0), (x, y) = points[-2:]
                angle = math.atan2(y-y0, x-x0)
                length, half = 6.6*sw, 3.3*sw
                base_x, base_y = x-length*math.cos(angle), y-length*math.sin(angle)
                triangle = [(x,y),(base_x+half*math.sin(angle),base_y-half*math.cos(angle)),
                            (base_x-half*math.sin(angle),base_y+half*math.cos(angle))]
                draw.polygon([(px*scale,py*scale) for px,py in triangle], fill=stroke)
        elif tag == "text":
            value = node.text or ""
            size = float(a["font-size"])
            bold = int(a["font-weight"]) >= 600
            chinese = "Microsoft YaHei" in a["font-family"]
            name = ("msyhbd.ttc" if bold else "msyh.ttc") if chinese else ("arialbd.ttf" if bold else "arial.ttf")
            runs = []
            for char in value:
                actual = name if supports(name, size, char) else "cambria.ttc"
                if not supports(actual, size, char):
                    raise ValueError(f"No font glyph for {ascii(char)}")
                if runs and runs[-1][0] == actual:
                    runs[-1][1] += char
                else:
                    runs.append([actual, char])
            widths = [font_for(fn, size).getlength(run) for fn, run in runs]
            width = sum(widths)
            x = float(a["x"])*scale
            if a["text-anchor"] == "middle":
                x -= width/2
            elif a["text-anchor"] == "end":
                x -= width
            for (fn, run), run_width in zip(runs, widths):
                draw.text((x, float(a["y"])*scale), run, font=font_for(fn, size), fill=fill, anchor="ls")
                x += run_width
        else:
            raise ValueError(f"Unhandled generated SVG primitive: {tag}")
    png = svg_path.with_suffix(".png")
    image.save(png, dpi=(300,300), optimize=True)
    print(f"PNG {png.name} ({image.width} x {image.height}; {png.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasterize", action="store_true", help="Also export 2x PNG using Pillow and Windows fonts")
    parser.add_argument("--scale", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    for lang in ("zh", "en"):
        path = ROOT / f"sgla_search_state_framework_{lang}_v1.svg"
        content = draw(lang)
        ET.fromstring(content)
        path.write_text(content, encoding="utf-8")
        print(f"SVG {path.name}")
        if args.rasterize:
            rasterize(path, args.scale)


if __name__ == "__main__":
    main()
