"""Draw the Stage-74/76 four-action PPO tracker and its fit-time workflow."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "figure_primitives", ROOT.parent / "sgla_search_state_v1/render_method_framework.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.W, base.H = 1640, 726
C = base.C
C.update(ink="#202C39", muted="#48596B", line="#CED8E3", gray="#627388", gray_bg="#F0F3F6",
         blue="#345C95", blue_bg="#EBF2FA", teal="#087568", teal_bg="#E8F5F1",
         amber="#8E551C", amber_bg="#F9EFE2")


def render(lang):
    s = base.SVG(lang)

    def tr(zh, en):
        return zh if lang == "zh" else en

    def label(x, y, zh, en, size=24, color="ink", bold=False, middle=False):
        s.text(x, y, tr(zh, en), size, C[color], 700 if bold else 400, "middle" if middle else "start")

    def line(points, color="gray", *, arrow=True, dashed=False, width=2.5):
        s.path(points, C[color], arrow=arrow, dash=dashed, width=width)

    def title(x, y, zh, en, color="ink"):
        label(x, y, zh, en, 27, color, True)

    def center(x, y, zh, en, size=24, color="ink", bold=False):
        label(x, y, zh, en, size, color, bold, True)

    def card(x, y, w, h, color):
        s.rect(x, y, w, h, C[color + "_bg"], C[color], r=7, sw=1.35)

    def crop_icon(x, y, history_size, lagged):
        s.rect(x, y, 62, 43, "#F8FAFC", r=3)
        s.rect(x+30, y+18, 8, 10, C["ink"], r=1)
        cx = x + (25 if lagged else 34)
        width, height = (32, 30) if history_size else (43, 23)
        s.rect(cx-width/2, y+23-height/2, width, height, "none", C["blue"], r=0, sw=1.6)
        s.circle(cx, y+23, 1.8, C["blue"])

    def response(x, y, k):
        for row in range(5):
            for col in range(6):
                v = math.exp(-((row-2.1+.15*k)**2+(col-2.6-.22*k)**2)/1.6)
                rgb = tuple(round(a+(b-a)*v) for a,b in zip((229,239,247), (43,106,147)))
                s.rect(x+col*8, y+row*7, 7.6, 6.6, '#%02x%02x%02x' % rgb, r=0)

    s.rect(0, 0, base.W, base.H, "white", r=0)
    s.rect(18, 18, 869, 454, "#FBFDFF", C["line"], r=8, sw=1.5)
    s.rect(909, 18, 713, 454, "#FCFEFD", C["line"], r=8, sw=1.5)
    title(38, 58, "(a) 候选生成", "(a) Candidate generation", "blue")
    title(929, 58, "(b) PPO 选框", "(b) PPO selection", "teal")

    s.scene(41, 135, 125, 86)
    center(103, 251, "I(t)", "I(t)", 25)
    for i in range(3):
        xx, yy = 43 + i*35, 324 - i*6
        s.rect(xx, yy, 52, 43, C["gray_bg"], C["line"], r=3)
        s.rect(xx+16, yy+10, 19, 22, "none", C["blue"], r=1, sw=1.5)
    center(103, 405, "H(t−1)", "H(t−1)", 25, "blue")
    line([(166, 178), (184, 178), (184, 205), (203, 205)])
    line([(165, 337), (184, 337), (184, 315), (203, 315)])

    card(204, 101, 305, 303, "blue")
    center(356, 139, "搜索裁剪", "Search crops", 25, "blue", True)
    center(326, 188, "c(t−1)", "c(t−1)", 22, "blue")
    center(442, 188, "c(t−2)", "c(t−2)", 22, "blue")
    label(216, 244, "s(t−1)", "s(t−1)", 20, "muted")
    label(216, 335, "s(hist)", "s(hist)", 20, "muted")
    for x, y, k in ((274, 206, 0), (390, 206, 2), (274, 297, 1), (390, 297, 3)):
        s.rect(x, y, 104, 63, "white", C["line"], r=4)
        crop_icon(x+5, y+10, k % 2 == 1, k >= 2)
        s.text(x+71, y+38, f"q{k}", 22, C["blue"], 700)

    line([(509, 254), (551, 254)])
    center(530, 232, "×4", "×4", 23, "blue", True)
    card(552, 101, 315, 303, "gray")
    center(709, 139, "冻结 SGLA", "Frozen SGLA", 25, bold=True)
    s.rect(570, 158, 278, 39, "white", C["line"], r=4)
    center(709, 185, "模板 z₀ + 裁剪", "Template z₀ + crop", 22)
    line([(709, 197), (709, 209)], width=1.55)
    for i in range(6):
        x = 570 + 47*i
        s.rect(x, 213, 39, 36, "white", C["line"], r=4)
        s.text(x+19.5, 238, f"L{i+1}", 19, C["gray"], 700, "middle")
        if i < 5:
            line([(x+39, 231), (x+46, 231)], width=1.0)
    line([(844, 231), (856, 231), (856, 263), (709, 263), (709, 279)], width=1.55)
    s.rect(570, 280, 278, 41, "white", C["line"], r=4)
    center(709, 308, "L7–L12 选一层", "One of L7–L12", 22)
    line([(709, 321), (709, 340)], width=1.55)
    s.rect(570, 341, 278, 43, "white", C["line"], r=4)
    center(709, 369, "CENTER 头", "CENTER head", 22, bold=True)
    line([(867, 254), (928, 254)])

    card(929, 101, 187, 303, "blue")
    center(1022, 139, "候选证据", "Candidates", 25, "blue", True)
    for k in range(4):
        y = 156 + 48*k
        s.text(942, y+25, f"q{k}", 22, C["blue"], 700)
        response(978, y, k)
        s.rect(1035, y, 60, 35, "white", C["line"], r=2)
        s.rect(1046+k*3, y+6, 33-k*3, 22, "none", C["blue"], r=0, sw=1.5)
    center(1022, 379, "4 × 64 维", "4 × 64-D", 22, "blue", True)
    line([(1116, 254), (1150, 254)])

    card(1151, 101, 234, 303, "teal")
    center(1268, 139, "PPO actor θ*", "PPO actor θ*", 25, "teal", True)
    center(1268, 188, "256 → 32 → 4", "256 → 32 → 4", 23)
    s.rect(1165, 215, 206, 46, "white", C["line"], r=4)
    center(1268, 245, "掩码 softmax", "Masked softmax", 22)
    line([(1268, 261), (1268, 289)], "teal", width=1.8)
    s.rect(1165, 290, 206, 45, "white", C["line"], r=4)
    center(1268, 320, "argmax", "argmax", 23, "teal", True)
    center(1268, 379, "a(t) ∈ {0, 1, 2, 3}", "a(t) ∈ {0, 1, 2, 3}", 22, "teal")
    line([(1385, 254), (1428, 254)], "teal")

    s.scene(1434, 193, 161, 114, True)
    center(1514, 352, "b̂(t) = b(t, a(t))", "b̂(t) = b(t, a(t))", 24, "teal")
    line([(1022, 404), (1022, 427), (1514, 427), (1514, 368)], width=1.8)
    center(1268, 453, "解码框", "Decoded boxes", 20, "muted")
    line([(1589, 371), (1589, 508), (103, 508), (103, 421)], "teal", dashed=True, width=2.6)
    s.rect(721, 490, 198, 34, "white", r=4)
    center(820, 515, "H(t) → t+1", "H(t) → t+1", 24, "teal", True)

    s.rect(18, 542, 1604, 166, "#FFFCF8", C["amber"], r=8, dash=True, sw=1.5)
    title(38, 576, "(c) PPO 训练", "(c) PPO training", "amber")
    cards = ((38, 271), (342, 276), (651, 227), (911, 381), (1325, 277))
    for x, w in cards:
        card(x, 590, w, 85, "amber")
    for (x,w),(next_x,_) in zip(cards,cards[1:]):
        line([(x+w, 632), (next_x-1, 632)], "amber", width=2)

    center(173.5, 622, "策略采样", "Policy rollout", 24, "amber", True)
    center(173.5, 654, "a(t) ∼ π(·|s(t))", "a(t) ∼ π(·|s(t))", 22)
    center(480, 622, "IoU 奖励", "IoU reward", 24, "amber", True)
    center(480, 654, "r(t) = IoU(b̂(t), g(t))", "r(t) = IoU(b̂(t), g(t))", 22)
    center(764.5, 622, "Critic + GAE", "Critic + GAE", 24, "amber", True)
    center(764.5, 654, "V(s), Â", "V(s), Â", 22)
    center(1101.5, 622, "PPO 裁剪更新", "Clipped PPO", 24, "amber", True)
    center(1101.5, 654, "min(ρÂ, clip(ρ, 0.8, 1.2)Â)", "min(ρÂ, clip(ρ, 0.8, 1.2)Â)", 22)
    center(1463.5, 622, "冻结 actor", "Freeze actor", 24, "amber", True)
    center(1463.5, 654, "θ* → (b)", "θ* → (b)", 22, "teal", True)
    line([(1101.5, 675), (1101.5, 691), (173.5, 691), (173.5, 676)], "amber", width=1.8)

    svg = s.finish()
    opening = svg[:svg.index('<title')]
    title_text = tr("原生 SGLA 四候选 PPO 跟踪框架", "Four-candidate PPO selection on native SGLA")
    description = tr("冻结 SGLA 生成四个解码框，冻结 PPO 选择一个框并更新下一帧裁剪历史。下方单列 PPO 训练。",
                     "Frozen SGLA decodes four boxes. A frozen PPO actor chooses one box and updates the next-frame crop history. Fit-time PPO is shown separately.")
    return f'{opening}<title id="title">{title_text}</title><desc id="desc">{description}</desc>' + svg[svg.index('<defs>'):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasterize", action="store_true")
    args = parser.parse_args()
    for lang in ("en", "zh"):
        path = ROOT / f"sgla_ppo_framework_{lang}_v3.svg"
        path.write_text(render(lang), encoding="utf-8")
        if args.rasterize:
            base.rasterize(path, 2)
        print(path.name)


if __name__ == "__main__":
    main()
