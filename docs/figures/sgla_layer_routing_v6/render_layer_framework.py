"""Draw the current SGLA layer-selection framework with the archived v5 photographs."""

import argparse
import base64
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "sgla_search_state_v5"
ASSETS = PREVIOUS / "assets"
spec = importlib.util.spec_from_file_location("primitives", HERE.parent / "sgla_search_state_v1/render_method_framework.py")
primitives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primitives)
primitives.W, primitives.H = 1840, 747
COLORS = primitives.C
COLORS.update(ink="#1C2D40", blue="#2C6393", blue_bg="#EEF5FA", teal="#167868", teal_bg="#EDF7F3",
              amber="#9D5E19", amber_bg="#FFF6EA", gray="#586C80", gray_bg="#F0F4F7",
              line="#C7D3DF", panel="#FBFCFD", violet="#6C5691", violet_bg="#F3EFF9")
PHOTO = json.loads((ASSETS / "provenance.json").read_text(encoding="utf-8"))


def render(language):
    s = primitives.SVG(language)

    def tr(zh, en):
        return zh if language == "zh" else en

    def text(x, y, zh, en=None, *, size=23, color="ink", bold=False, anchor="middle"):
        s.text(x, y, tr(zh, en or zh), size, COLORS[color], 700 if bold else 400, anchor)

    def box(x, y, w, h, color, *, dashed=False):
        s.rect(x, y, w, h, COLORS[color + "_bg"], COLORS[color], r=7, sw=1.35, dash=dashed)

    def line(points, color="gray", *, arrow=True, dashed=False, width=2.1):
        d = "M " + " L ".join(f"{x},{y}" for x, y in points)
        end = f' marker-end="url(#{color})"' if arrow else ""
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        s.add(f'<path d="{d}" fill="none" stroke="{COLORS[color]}" stroke-width="{width}" '
              f'stroke-linecap="round" stroke-linejoin="round"{dash}{end}/>')

    def photo(name, x, y, w, h, marker=None, color="blue"):
        rec = PHOTO["assets"][name]
        data = (ASSETS / rec["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != rec["sha256"]:
            raise ValueError(f"Unexpected source crop for {name}")
        image = base64.b64encode(data).decode("ascii")
        s.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" '
              f'href="data:image/jpeg;base64,{image}"/>')
        s.rect(x, y, w, h, "none", COLORS["line"], r=0, sw=1.2)
        if marker is not None:
            ax, ay, aw, ah = rec["crop"]
            bx, by, bw, bh = marker
            px, py = x + (bx - ax) * w / aw, y + (by - ay) * h / ah
            pw, ph = bw * w / aw, bh * h / ah
            s.rect(px, py, pw, ph, "none", "#FFFFFF", r=0, sw=4.4)
            s.rect(px, py, pw, ph, "none", COLORS[color], r=0, sw=2.3)

    s.rect(0, 0, 1840, 747, r=0)
    s.rect(24, 24, 520, 639, COLORS["panel"], COLORS["line"], r=10, sw=1.2)
    s.rect(564, 24, 1252, 639, COLORS["panel"], COLORS["line"], r=10, sw=1.2)
    text(46, 63, "(a) 单搜索裁剪与时序状态", "(a) Single-crop state encoding", size=29, color="blue", bold=True, anchor="start")
    text(586, 63, "(b) 风险门控与成对选层", "(b) Risk gating and pairwise layer selection",
         size=29, color="blue", bold=True, anchor="start")

    # The first-frame template and current crop share one SGLA prefix.
    photo("current", 57, 117, 144, 108, PHOTO["assets"]["search1"]["crop"])
    photo("search1", 246, 117, 108, 108)
    photo("template", 402, 117, 108, 108, PHOTO["target_box_original"])
    line([(202, 171), (244, 171)], "blue")
    text(129, 253, "当前帧 Iₜ", "Frame Iₜ", size=20, bold=True)
    text(300, 253, "按 b̂ₜ₋₁ 裁剪", "Crop from b̂ₜ₋₁", size=20, bold=True)
    text(456, 253, "首帧模板 z₀", "Template z₀", size=20, bold=True)
    line([(300, 261), (300, 290)], "blue")
    line([(456, 261), (456, 275), (399, 275), (399, 290)], "blue")
    box(122, 290, 344, 77, "gray")
    text(294, 322, "冻结 SGLA 共享前缀", "Frozen SGLA prefix", size=23, bold=True)
    text(294, 349, "L1–L6", size=23, color="gray")

    # Fₜ is consumed by the native selector, the state projection, and the chosen block.
    line([(466, 327), (552, 327), (552, 89), (1590, 89), (1590, 122)], "blue")
    line([(736, 89), (736, 114)], "blue")
    s.circle(736, 89, 3.5, COLORS["blue"])
    text(490, 315, "Fₜ", size=23, color="blue", bold=True)
    line([(395, 367), (395, 416)], "blue")
    box(280, 416, 231, 83, "blue")
    text(395, 451, "视觉特征投影", "Visual projection", size=22, color="blue", bold=True)
    text(395, 479, "vₜ ∈ ℝ¹⁶", size=24)
    box(56, 416, 200, 83, "teal")
    text(156, 450, "自身历史", "Tracking history", size=22, color="teal", bold=True)
    text(156, 479, "Hₜ₋₁", size=24)
    line([(156, 499), (156, 546)], "teal")
    line([(395, 499), (395, 546)], "blue")
    box(56, 546, 455, 82, "blue")
    text(284, 578, "时序状态向量", "Temporal state", size=22, color="blue", bold=True)
    text(284, 608, "x₄(t) ∈ ℝ²⁴      x₁₆(t) ∈ ℝ²⁷", size=23)

    # The two heads of the shared gate and the pairwise ranking use the named states.
    line([(511, 586), (586, 586), (586, 251), (616, 251)], "blue")
    for y in (351, 493):
        line([(586, y), (616, y)], "blue")
        s.circle(586, y, 3.5, COLORS["blue"])
    box(616, 114, 240, 66, "gray")
    text(736, 143, "原生选择器", "Native selector", size=21, color="gray", bold=True)
    text(736, 168, "原生层 ℓ₀", "Native layer ℓ₀", size=18)
    line([(856, 146), (1078, 146), (1078, 217)], "gray")

    for y, zh, en, math, color in (
        (215, "收益预测", "Gain prediction", "x₄(t)  →  g, pₜ", "blue"),
        (315, "退化预测", "Downside prediction", "x₁₆(t)  →  r", "violet"),
        (457, "成对排序", "Pairwise ranking", "x₄(t)  →  ℓᵣ", "teal"),
    ):
        box(616, y, 240, 72, color)
        text(736, y + 30, zh, en, size=22, color=color, bold=True)
        text(736, y + 57, math, size=22)
    line([(856, 251), (934, 251)], "blue")
    line([(856, 351), (934, 351)], "violet")
    line([(856, 493), (890, 493), (890, 554), (934, 554)], "teal")

    box(934, 217, 288, 159, "blue")
    text(1078, 250, "共享风险门控", "Shared risk gate", size=24, color="blue", bold=True)
    text(1078, 289, "sₜ = g(pₜ) − r(pₜ)", size=24)
    text(1078, 324, "pₜ ≠ ℓ₀     sₜ > τ", size=22)
    text(1078, 353, "至少 4 帧历史", "At least 4 past frames", size=18, color="gray")
    line([(1078, 376), (1078, 403)], "amber", dashed=True)
    box(934, 403, 288, 74, "amber", dashed=True)
    text(1078, 433, "16 帧冷却 · 待验证", "16-frame cooldown trial",
         size=20, color="amber", bold=True)
    text(1078, 459, "t − tₗₐₛₜ > 16", size=21, color="amber")
    line([(1078, 477), (1078, 512)], "amber", dashed=True)
    box(934, 512, 288, 84, "teal")
    text(1078, 544, "本帧执行层 ℓₜ", "Executed layer ℓₜ", size=22, color="teal", bold=True)
    text(1078, 573, "通过 → ℓᵣ     回退 → ℓ₀", "Pass → ℓᵣ    Fallback → ℓ₀", size=21)
    line([(1222, 554), (1332, 554), (1332, 224), (1407, 224)], "teal")
    text(1359, 212, "ℓₜ", size=24, color="teal", bold=True)

    # One of the six independent late blocks runs before the original box head.
    box(1407, 122, 366, 206, "gray")
    text(1590, 159, "SGLA 独立候选块", "Independent SGLA blocks", size=24, bold=True)
    for i, layer in enumerate(range(7, 13)):
        x, y = 1434 + i % 3 * 106, 175 + i // 3 * 49
        color = "teal" if layer == 8 else "gray"
        box(x, y, 94, 39, color)
        text(x + 47, y + 27, f"L{layer}", size=22, color=color, bold=layer == 8)
    text(1590, 307, "只执行 ℓₜ", "Execute ℓₜ only", size=22, color="teal", bold=True)
    line([(1590, 328), (1590, 356)])
    box(1451, 356, 278, 70, "gray")
    text(1590, 400, "CENTER ＋ Hann", "CENTER + Hann", size=25, bold=True)
    line([(1590, 426), (1590, 450)], "teal")
    photo("output", 1506, 450, 168, 126, PHOTO["target_box_original"], "teal")
    text(1590, 613, "预测框 b̂ₜ 与峰值 hₜ", "Box b̂ₜ and peak hₜ", size=22, color="teal", bold=True)

    # The feedback arrives at the left-side history without crossing the feature bus.
    line([(1590, 625), (1590, 701), (39, 701), (39, 459), (56, 459)], "teal", width=2.7)
    s.rect(625, 683, 545, 35, "#FFFFFF", r=0)
    text(897, 710, "写回预测框、响应峰值与实际换层状态",
         "Feed back box, response peak and switch state", size=22, color="teal", bold=True)

    svg = s.finish()
    title = tr("无人机视角单目标跟踪的时序感知选层", "Temporal-aware SGLA layer selection for UAV-view tracking")
    description = ("One image crop and the fixed template share SGLA L1 through L6. A risk gate determines whether to "
                   "replace the original layer with the pairwise-ranked alternative. A dashed 16-frame cooldown box "
                   "marks the pending Stage-170 test. Each frame executes one independent late block. Photos are "
                   "illustrative crops from the attributed v5 asset.")
    markers = "".join(f'<marker id="{color}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
                      f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{COLORS[color]}"/></marker>'
                      for color in ("blue", "violet"))
    prefix = svg[:svg.index("<title")]
    tags = f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>'
    return prefix + tags + svg[svg.index("<defs>"):].replace("</defs>", markers + "</defs>", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasterize", action="store_true")
    args = parser.parse_args()
    for language in ("zh", "en"):
        path = HERE / f"sgla_temporal_layer_framework_{language}_v6.svg"
        path.write_text(render(language), encoding="utf-8")
        if args.rasterize:
            subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(PREVIOUS / "rasterize_svg.ps1"), "-SvgPath", str(path)], check=True, timeout=55)
        print(path)


if __name__ == "__main__":
    main()
