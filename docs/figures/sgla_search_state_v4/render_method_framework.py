"""Render an editable Stage-113/120 research framework; no experimental changes.

SVG uses the repository's vector primitives. Optional PNG export uses native
Windows drawing, without installing packages or using experiment data.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "figure_primitives", ROOT.parent / "sgla_search_state_v1/render_method_framework.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.W, base.H = 2160, 1200
base.C.update(ink="#213144", muted="#536477", line="#CBD4DE", gray="#617287",
              gray_bg="#F0F3F6", blue="#336B9B", blue_bg="#EDF4FB",
              teal="#16826F", teal_bg="#EAF5EF", amber="#A36723", amber_bg="#FCF5E9")
C = base.C
C.update(violet="#77639A", violet_bg="#F1EDF7")


def render(lang):
    s = base.SVG(lang)
    def tr(zh, en): return zh if lang == "zh" else en
    def text(x, y, zh, en=None, size=22, color="ink", bold=False, anchor="start"):
        s.text(x, y, tr(zh, en if en is not None else zh), size, C.get(color, color),
               700 if bold else 400, anchor)
    def center(x, y, zh, en=None, size=22, color="ink", bold=False):
        text(x, y, zh, en, size, color, bold, "middle")
    def line(points, color="gray", arrow=True, dashed=False, width=2.1):
        d = "M " + " L ".join(f"{x},{y}" for x,y in points)
        s.add(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="{width}" '
              'stroke-linejoin="round" stroke-linecap="round"'
              + (' stroke-dasharray="7 5"' if dashed else '')
              + (f' marker-end="url(#{color})"' if arrow else '') + '/>')
    def box(x, y, w, h, color="gray", dashed=False, fill=None, r=6):
        s.rect(x,y,w,h,fill or C[color+"_bg"],C[color],r=r,dash=dashed,sw=1.4)
    def panel(x,y,w,h,zh,en,color="blue",dashed=False):
        s.rect(x,y,w,h,"#FCFDFE" if not dashed else "#FFFCF7",C["line"] if not dashed else C[color],r=11,dash=dashed,sw=1.5)
        text(x+20,y+34,zh,en,25,color,True)
    def tiny_frame(x,y,w,h,index=0,target=True):
        s.rect(x,y,w,h,"#E8EDE7",C["line"],r=2,sw=1)
        s.rect(x+w*.04,y+h*.08,w*.23,h*.28,"#BCCABA",r=1)
        s.rect(x+w*.68,y+h*.64,w*.27,h*.25,"#CAD5C6",r=1)
        s.add(f'<path d="M {x},{y+h*.76} L {x+w},{y+h*.34}" fill="none" stroke="#CBD1D1" stroke-width="{h*.31}"/>')
        s.add(f'<path d="M {x},{y+h*.76} L {x+w},{y+h*.34}" fill="none" stroke="white" stroke-width="1.2" stroke-dasharray="6 5"/>')
        cx,cy=x+w*(.41+.035*index),y+h*(.54-.015*index)
        s.rect(cx-w*.035,cy-h*.08,w*.07,h*.16,"#45576A",r=1)
        if target:s.rect(cx-w*.075,cy-h*.13,w*.15,h*.26,"none",C["teal"],r=0,sw=1.6)
    def heat(x,y,k):
        for r in range(4):
            for c in range(5):
                t=math.exp(-((r-1.4-.12*k)**2+(c-2+.15*k)**2)/1.35)
                rgb=tuple(round(a+(b-a)*t) for a,b in zip((232,239,248),(87,104,162)))
                s.rect(x+c*8,y+r*7,7.5,6.5,"#%02x%02x%02x"%rgb,r=0)
    def chip(x,y,w,label,color="blue",h=28):
        box(x,y,w,h,color,r=4);center(x+w/2,y+h*.72,label,size=18,color=color)

    s.rect(0,0,2160,1200,"white",r=0)
    panel(24,24,1432,414,"(a) 历史条件搜索区域生成","(a) History-conditioned search proposal generation")
    panel(24,462,1432,420,"(b) SGLA 定位与共享质量评分","(b) SGLA localization and shared quality scoring")
    panel(1480,24,656,858,"(c) 离线后继收益提交检验","(c) Offline future-gain commit study","amber",True)
    panel(24,926,2112,250,"(d) 自身预测历史的时序反馈","(d) Temporal feedback through own predictions","teal")

    # a: history --> learned mean/scale --> fixed Gaussian scenarios --> allocation.
    box(47,91,191,305,"gray")
    center(142,123,"历史 H(t−1)","History H(t−1)",23,bold=True)
    for i in range(5):
        tiny_frame(67+i*17,153-i*5,83,61,i)
    center(142,252,"最近 ≤5 个预测框","≤5 past predicted boxes",17)
    center(142,281,"运动与尺寸变化","Motion / scale changes",18,color="muted")
    center(142,307,"上一帧响应与资格","Past responses / flag",18,color="muted")
    chip(83,342,119,"x(t) ∈ R³²","gray",32)
    line([(238,242),(266,242)])

    box(267,91,216,305,"blue")
    center(375,123,"历史预测模型","History predictor",23,"blue",True)
    box(286,155,178,66,"blue",fill="white")
    center(375,183,"均值回归","Mean regression",20)
    center(375,208,"μ(t)",size=22,color="blue",bold=True)
    box(286,241,178,66,"blue",fill="white")
    center(375,269,"条件误差尺度","Conditional error scale",17)
    center(375,294,"σ(t)",size=22,color="blue",bold=True)
    center(375,340,"位移 ＋ 对数宽高变化","Displacement + log-scale",17,color="muted")
    center(375,373,"离线拟合与经验校准 α","Offline fit / calibration α",17,color="muted")
    line([(483,242),(511,242)])

    box(512,91,286,305,"violet")
    center(655,123,"32 个假设场景","32 hypothetical states",23,"violet",True)
    # A fixed antithetic schematic cloud: neither empirical samples nor results.
    cx,cy=654,230
    for sign in (-1,1):
        for k in range(16):
            angle=k*2.399963;radius=math.sqrt((k+.5)/16)
            dx=91*radius*math.cos(angle)*sign;dy=60*radius*math.sin(angle)*sign
            s.circle(cx+dx,cy+dy,3,C["violet"])
    s.rect(cx-15,cy-19,30,38,"none",C["blue"],r=0,sw=2)
    line([(cx-6,cy),(cx+6,cy)],"blue",False,width=1.5)
    line([(cx,cy-6),(cx,cy+6)],"blue",False,width=1.5)
    center(655,329,"uⱼ = μ(t) + α σ(t) ⊙ zⱼ",size=21,color="violet")
    center(655,370,"固定扰动：16 对正反样本","Fixed noise: 16 ± pairs",18,color="muted")
    line([(798,242),(826,242)])

    box(827,91,276,305,"blue")
    center(965,123,"备选搜索区域","Proposal pool",23,"blue",True)
    tiny_frame(875,158,177,113,2,False)
    for i,(dx,dy,ww) in enumerate(((0,0,90),(21,-4,65),(-18,10,76),(3,-14,111))):
        s.rect(963-ww/2+dx,214-ww/2+dy,ww,ww,"none",C["teal"] if i==0 else C["blue"],r=0,sw=1.5)
    center(965,309,"32 场景 ＋ 均值 ＋ 标准输入","32 states + mean + standard",17)
    center(965,339,"34 个备选项（可重复）","34 options (duplicates possible)",17,color="muted")
    center(965,375,"Lⱼ = ceil(f √(wⱼhⱼ))",size=21,color="blue")
    line([(1103,242),(1131,242)])

    box(1132,91,301,305,"teal")
    center(1282,123,"覆盖增益贪心分配","Greedy coverage allocation",21,"teal",True)
    center(1282,174,"保留标准输入 ＋ 选 3 个","Keep standard + select three",18)
    center(1282,219,"增益 = 新增场景覆盖","Gain = added scenario support",18,color="muted")
    for j in range(4):
        bx=1151+j*68
        box(bx,249,57,57,"teal",fill="white")
        ww=(30,40,26,34)[j]
        s.rect(bx+(57-ww)/2+(j-1)*2,262,ww,ww,"none",C["teal"],r=0,sw=1.5)
        center(bx+28.5,335,f"P{j}",size=19,color="teal",bold=True)
    center(1282,373,"4 个父框 → 4 个搜索裁剪","4 parents → 4 search crops",18)
    # Link proposal output to the next row through the panel gutter.
    line([(1282,396),(1282,450),(1444,450),(1444,518),(351,518),(351,558)],"blue")

    # b: current image and immutable template feed one SGLA, reused per crop.
    tiny_frame(48,528,190,121,3)
    center(143,678,"当前图像 I(t)","Current frame I(t)",20,bold=True)
    line([(238,591),(263,591)])
    for j in range(4):
        xx=267+(j%2)*91;yy=562+(j//2)*79
        tiny_frame(xx,yy,78,58,j)
        s.rect(xx+9+j*3,yy+8,44-j*3,40,"none",C["blue"],r=0,sw=1.2)
    center(351,744,"裁剪与缩放 ×4","Crop & resize ×4",20,"blue",True)
    tiny_frame(293,768,111,67,0)
    center(349,858,"固定首帧模板 z₀","Fixed template z₀",17,color="muted")
    line([(436,630),(489,630)])
    line([(404,801),(461,801),(461,573),(490,573)],"gray")

    box(490,529,400,282,"gray")
    center(690,564,"冻结 SGLA：逐裁剪执行","Frozen SGLA · per crop",23,bold=True)
    box(509,583,362,34,"gray",fill="white")
    center(690,607,"模板／搜索图像嵌入","Template / search embedding",19)
    for j in range(6):
        xx=510+j*61
        box(xx,632,49,34,"gray",fill="white")
        center(xx+24.5,656,f"L{j+1}",size=18)
        if j<5:line([(xx+49,649),(xx+60,649)],width=1)
    line([(690,617),(690,630)],width=1.1)
    line([(861,649),(880,649),(880,682),(690,682),(690,694)],width=1.1)
    for j in range(6):
        xx=510+j*61
        box(xx,695,49,33,"violet" if j==2 else "gray",fill=C["violet_bg"] if j==2 else "white")
        center(xx+24.5,718,f"L{j+7}",size=17,color="violet" if j==2 else "muted")
    center(690,752,"原生选择器：仅执行一层","Native selector: execute one",18,color="muted")
    box(550,766,280,31,"gray",fill="white")
    center(690,789,"CENTER 头与原生解码","CENTER head / native decode",18,bold=True)
    line([(830,781),(904,781),(904,673),(917,673)])

    box(918,529,177,282,"blue")
    center(1006,564,"候选与证据","Boxes & evidence",20,"blue",True)
    for j in range(4):
        yy=585+j*41
        text(930,yy+21,f"q{j}",size=19,color="blue",bold=True)
        heat(965,yy,j)
        s.rect(1017,yy,61,28,"white",C["line"],r=1)
        s.rect(1030+j*2,yy+5,25+(j%2)*5,18,"none",C["blue"],r=0,sw=1.3)
    center(1006,788,"4 × 64 维描述","4 × 64-D descriptors",17,color="blue")
    line([(1095,673),(1123,673)])

    box(1124,529,169,282,"teal")
    center(1208,562,"共享质量","Shared quality",19,"teal",True)
    center(1208,588,"评分器 fθ","scorer fθ",19,"teal",True)
    center(1208,635,"64 → 32 → 1",size=21,bold=True)
    center(1208,675,"Tanh / Sigmoid",size=17,color="muted")
    center(1208,720,"训练：sᵢ ≈ IoU","Fit: sᵢ ≈ IoU",17,color="teal")
    center(1208,755,"监督质量回归","Supervised quality fit",16,color="muted")
    line([(1293,673),(1317,673)],"teal")
    box(1318,572,115,210,"teal",fill="white")
    center(1375,608,"argmax",size=20,color="teal",bold=True)
    tiny_frame(1330,630,90,65,3)
    center(1375,729,"初选 q*","Choice q*",19,"teal",True)
    center(1375,761,"输出框","Output box",17,color="muted")
    # True completed baseline: selected box and response state drive the next frame.
    line([(1375,782),(1375,898),(10,898),(10,243),(47,243)],"teal",width=2.4)
    s.rect(655,883,461,29,"white",r=2)
    center(885,905,"更新框历史与上一帧响应状态","Update box history and previous response state",19,"teal",True)

    # c: deliberately separate matched-action study, NOT an active rollout loop.
    center(1808,93,"同一状态的两个当前动作","Two actions at the same state",20,"amber")
    for xx,label,j in ((1524,"q0",1),(1848,"q*",3)):
        tiny_frame(xx,116,224,103,j)
        center(xx+112,246,label,size=24,color="amber",bold=True)
    center(1808,283,"响应质量、几何冲突与历史兼容性","Response / geometry / history compatibility",19)
    box(1510,302,596,55,"gray",dashed=True,fill="white")
    center(1808,338,"原有公开证据：13 维","Original public evidence: 13-D",21)

    for xx,labelzh,labelen,j in ((1534,"首帧参考","First-frame reference",0),(1843,"近期预测参考","Recent predicted reference",3)):
        tiny_frame(xx,381,159,72,j)
        center(xx+80,478,labelzh,labelen,17,color="muted")
    center(1808,514,"RGB 直方图 / NCC ＋ 覆盖、有效位、年龄","RGB histogram / NCC + coverage, validity, age",17)
    center(1808,546,"24 维外观与参考信息","24-D appearance / reference cues",21,"amber",True)
    line([(1614,453),(1614,493),(1500,493),(1500,579),(1638,579)],"amber",dashed=True,width=1.5)
    line([(1922,453),(1922,493),(2118,493),(2118,579),(1978,579)],"amber",dashed=True,width=1.5)
    line([(2106,330),(2110,330),(2110,598),(1978,598)],"amber",dashed=True,width=1.5)
    # Feature streams enter the future-gain head; reference imagery is illustrative.
    box(1639,563,338,69,"amber",dashed=True,fill="white")
    center(1808,591,"37 维 → 岭回归 → ΔJ 预测","37-D → ridge → predicted ΔJ",20,"amber",True)
    center(1808,618,"拟合目标：后 31 帧平均 IoU 差","Fit target: next-31-frame mean IoU gain",16,color="muted")
    line([(1808,632),(1808,659)],"amber",dashed=True)
    box(1643,660,330,48,"amber",dashed=True)
    center(1808,691,"预测收益 > 0？","Predicted gain > 0?",21,"amber",True)
    line([(1643,684),(1590,684),(1590,737)],"amber",dashed=True,width=1.6)
    line([(1973,684),(2026,684),(2026,737)],"amber",dashed=True,width=1.6)
    box(1512,738,222,47,"amber",dashed=True,fill="white")
    box(1869,738,235,47,"amber",dashed=True,fill="white")
    center(1623,768,"否：保留 q*","No: retain q*",20,"amber")
    center(1986,768,"是：改选 q0","Yes: use q0",20,"amber")
    center(1808,821,"一次起点动作 → 恢复冻结策略 → 后继评价","One start action → frozen policy → future evaluation",17,color="muted")
    center(1808,854,"未作为持续门控完成完整轨迹验证","Continuous gated rollout not yet validated",17,color="amber")

    # d: schematic time strip inspired by the supplied temporal diagram.
    centers=[175,520,865,1210,1555,1900]
    captions=[("历史预测","Past prediction","t−2"),("最新历史","Latest history","t−1"),
              ("区域生成","Search proposals","t"),("SGLA 候选","SGLA candidates","t"),
              ("所选框写回","Commit chosen box","t"),("下一帧搜索","Next-frame search","t+1")]
    for k,(xx,(zh,en,stamp)) in enumerate(zip(centers,captions)):
        if k in (2,3):
            for j in range(3):tiny_frame(xx-70+j*9,1000-j*6,116,67,j)
        else:tiny_frame(xx-70,992,140,81,min(k,4))
        center(xx,1098,zh,en,19,color="teal" if k>=4 else "ink",bold=k>=4)
        center(xx,1153,stamp,size=20,color="teal")
        if k<5:line([(xx+85,1030),(centers[k+1]-90,1030)],"teal",width=2)
    line([(91,1123),(2046,1123)],"teal",width=2.8)

    svg=s.finish()
    title=tr("历史条件搜索与后继收益提交研究框架","History-conditioned search and future-gain commit research framework")
    desc=("Panels a-b and d show the completed Stage-113 generated/shared baseline with frozen SGLA and a shared IoU regressor. "
          "Panel c is the separate Stage-120 offline matched-action study, not a deployed continuous gate. "
          "Gaussian states, images, response maps and the highlighted SGLA layer are schematic. "
          "GT in the scorer denotes an offline fitting target only. No PPO, diffusion, or online SGLA template update is represented.")
    svg=svg[:svg.index('<title')]+f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc>'+svg[svg.index('<defs>'):]
    markers=''.join(f'<marker id="{color}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{C[color]}"/></marker>' for color in ('blue','violet'))
    return svg.replace('</defs>',markers+'</defs>',1)


def rasterize(path):
    result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        str(ROOT/'rasterize_svg.ps1'),'-SvgPath',str(path)],capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=55)
    if result.returncode:raise RuntimeError(result.stderr[-2500:])
    print(path.with_suffix('.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rasterize',action='store_true')
    args=parser.parse_args()
    for lang in ('en','zh'):
        path=ROOT/f'sgla_history_search_framework_{lang}_v4.svg'
        path.write_text(render(lang),encoding='utf-8')
        print(path)
        if args.rasterize:rasterize(path)


if __name__=='__main__':main()
