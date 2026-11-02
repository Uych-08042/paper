"""Publication-style overview + module details, referencing OSTrack Fig. 3 layout."""
from pathlib import Path
import importlib.util
import argparse
import math

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("figure_primitives", ROOT.parent / "sgla_search_state_v1/render_method_framework.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.W, base.H = 2000, 1060
C = base.C
C.update(ink="#22313C", muted="#68747D", teal="#258B80", teal_bg="#E9F5F1", blue="#547EB6", blue_bg="#EDF3FA", amber="#B2874F", amber_bg="#FCF5EB", gray="#7A858E", gray_bg="#F0F2F4", line="#D2D8DC")


def render(lang):
    s = base.SVG(lang)
    def t(zh,en): return zh if lang == "zh" else en
    def txt(x,y,a,b,size=23,**kw): s.text(x,y,t(a,b),size,**kw)
    def arrow(points, col=None, dash=False): s.path(points,col or C['gray'],True,dash,width=2)
    def node(x,y,w,h,label,fill,col,size=23,dash=False):
        s.rect(x,y,w,h,fill,col,r=6,dash=dash,sw=1.4)
        s.text(x+w/2,y+h/2+size*.33,label,size,C['muted'] if col==C['line'] else col,600,"middle")
    def heatmap(x,y,w=86,h=75,k=0):
        for row in range(7):
            for col in range(8):
                value=math.exp(-((row-(3+.18*k))**2+(col-(3.3+.22*k))**2)/1.7)
                rgb=tuple(round(a+(b-a)*value) for a,b in zip((232,240,246),(54,116,160)))
                s.rect(x+col*w/8,y+row*h/7,w/8-.5,h/7-.5,'#%02x%02x%02x'%rgb,r=0)
    def box_icon(x,y,cx=0,scale=1,w=104,h=72):
        s.rect(x,y,w,h,"#F1F4F6",r=2)
        s.rect(x+47,y+29,12,14,C['ink'],r=2)
        s.rect(x+52+cx-30*scale,y+36-23*scale,60*scale,46*scale,'none',C['blue'],r=0,sw=1.8)
    s.rect(0,0,2000,1060,'white',r=0)
    txt(40,43,"(a) 在线跟踪总览","(a) Overall tracking pipeline",27,weight=700)
    # Compact legends; experiment results are deliberately kept in the caption.
    for x,col,label in ((1300,C['blue'],t('历史候选','History proposals')),(1515,C['teal'],t('响应选择','Response selection')),(1740,C['amber'],t('学习替代策略','Learned alternative'))):
        s.rect(x,25,18,18,C['amber_bg'] if col==C['amber'] else C['blue_bg'] if col==C['blue'] else C['teal_bg'],col,r=2,dash=col==C['amber'])
        s.text(x+28,42,label,18,C['muted'])
    # Inputs.
    s.scene(48,112,184,125)
    txt(140,269,"当前帧 I(t)","Current frame I(t)",20,anchor='middle')
    for i in range(3):
        s.rect(50+i*54,317-i*7,63,55,C['gray_bg'],C['line'],r=2)
        s.rect(74+i*54,333-i*7,19,23,'none',C['blue'],r=0,sw=1.5)
    txt(144,407,"已接受历史 H(t−1)","Accepted history H(t−1)",20,anchor='middle')
    # Proposal block, detailed below.
    s.rect(295,100,285,320,C['blue_bg'],C['blue'],r=8,sw=1.5)
    txt(437,138,"历史搜索候选","History-guided proposals",21,color=C['blue'],weight=600,anchor='middle')
    for k,(xx,yy) in enumerate(((313,164),(446,164),(313,263),(446,263))):
        box_icon(xx,yy,cx=-8 if k%2 else 0,scale=.82 if k>=2 else 1,w=112,h=76)
    txt(437,386,"中心 × 尺寸","Center × size",22,color=C['blue'],anchor='middle')
    arrow([(232,175),(264,175),(264,191),(294,191)])
    arrow([(232,338),(264,338),(264,333),(294,333)],C['blue'])
    # Base SGLA module: exact native execution topology, no sequential L7-L12.
    s.rect(641,100,525,320,'white',C['gray'],r=8,sw=1.5)
    txt(903,138,"固定 SGLA 跟踪器 E","Fixed SGLA-based tracker E",25,weight=600,anchor='middle')
    node(666,161,475,41,t('固定首帧模板 z₀','Fixed initial template z₀'),C['gray_bg'],C['gray'],20)
    txt(663,238,"前缀块","Prefix",19,color=C['muted'])
    for i in range(6):
        node(748+i*62,214,50,39,str(i+1),C['gray_bg'],C['line'],19)
        if i<5: s.path([(798+i*62,233),(809+i*62,233)],C['gray'],True,width=1)
    txt(663,289,"独立候选","Branch",19,color=C['muted'])
    for i in range(6):
        active=i==2
        node(748+i*62,266,50,36,f'L{i+7}',C['teal_bg'] if active else 'white',C['teal'] if active else C['line'],16)
    txt(663,329,"仅选 1 个；高亮为示意","Select ONE; highlight illustrative",14,color=C['muted'])
    node(668,353,220,44,t('响应分支','Response head'),C['gray_bg'],C['line'],21)
    node(912,353,229,44,t('尺寸 / 偏移分支','Size / offset head'),C['gray_bg'],C['line'],20)
    s.path([(903,202),(903,207),(735,207),(735,233),(747,233)],C['gray'],True,width=1.5)
    s.path([(1108,233),(1125,233),(1125,258),(897,258),(897,265)],C['gray'],True,width=1.5)
    s.path([(897,302),(897,342),(778,342),(778,352)],C['gray'],True,width=1.8)
    s.path([(897,342),(1027,342),(1027,352)],C['gray'],True,width=1.8)
    arrow([(580,260),(640,260)],C['blue'])
    txt(610,245,"×4","×4",19,color=C['blue'],anchor='middle')
    # Candidate evidence bank.
    for k in range(4):
        yy=121+k*66
        heatmap(1217,yy,64,51,k)
        s.rect(1292,yy,67,51,C['gray_bg'],r=2)
        s.rect(1305+k*2,yy+11,35-k*2,29,'none',C['blue'],r=0,sw=1.6)
    txt(1288,416,"{Mᵏ, bᵏ(t)}","{Mᵏ, bᵏ(t)}",22,anchor='middle')
    arrow([(1166,260),(1210,260)])
    # Two scoring alternatives, not an added feature fusion module.
    node(1410,135,267,84,t('Hann 响应评分','Hann response score'),C['teal_bg'],C['teal'],22)
    s.text(1543,201,'max(Hann ⊙ Mᵏ)',20,C['teal'],anchor='middle')
    node(1410,260,267,77,t('收益评分器 gθ','Utility scorer gθ'),C['amber_bg'],C['amber'],22,dash=True)
    txt(1543,369,"两种评分二选一","Alternative scoring rules",18,color=C['muted'],anchor='middle')
    arrow([(1359,176),(1409,176)],C['teal'])
    arrow([(1359,298),(1409,298)],C['amber'],True)
    # Common selection and decoded output.
    node(1724,168,72,155,'argmax',C['gray_bg'],C['gray'],19)
    arrow([(1677,178),(1701,178),(1701,205),(1723,205)],C['teal'])
    arrow([(1677,299),(1701,299),(1701,285),(1723,285)],C['amber'],True)
    s.scene(1830,177,126,106,True)
    txt(1893,321,"输出 b(t)","Output b(t)",21,anchor='middle')
    arrow([(1796,245),(1829,245)])
    # Main causal loop is a single uncluttered rail.
    arrow([(1893,340),(1893,471),(144,471),(144,420)],C['teal'])
    s.rect(680,453,664,36,'white',r=0)
    txt(1012,478,"选中输出 → 已接受历史 → 下一帧","Selected output → accepted history → next frame",22,color=C['teal'],anchor='middle')
    txt(40,521,"每候选独立搜索；模板固定。四候选 = 每帧四次完整推理。","Separate search per proposal; fixed template. Four proposals = four complete inference passes.",19,color=C['muted'])
    # Panel b: visual zoom of the causal candidate construction.
    s.rect(40,562,908,423,'white',C['line'],r=8,sw=1.4)
    txt(60,602,"(b) 因果搜索状态构造","(b) Causal search-state construction",26,weight=700)
    txt(480,650,"当前中心 c(t−1)","Latest center c(t−1)",20,color=C['blue'],anchor='middle')
    txt(760,650,"滞后中心 c(t−2)","Lagged center c(t−2)",20,color=C['blue'],anchor='middle')
    txt(65,724,"当前尺寸 s(t−1)","Latest size s(t−1)",20,color=C['muted'])
    txt(65,834,"历史尺寸 s(hist)","History size s(hist)",20,color=C['muted'])
    for x,y,k,lag,hist in ((358,674,0,False,False),(638,674,2,True,False),(358,784,1,False,True),(638,784,3,True,True)):
        s.rect(x,y,247,91,C['blue_bg'],C['line'],r=5)
        box_icon(x+12,y+10,-8 if lag else 0,.82 if hist else 1,94,69)
        label={0:'current',1:'history size',2:'lag center',3:'combined'}[k]
        s.text(x+176,y+37,f'q{k}',24,C['blue'],600,'middle')
        s.text(x+176,y+66,label,17,C['muted'],anchor='middle')
    txt(64,920,"s(hist) = exp[median(log s)]；最近至多 5 个已接受框","s(hist) = exp[median(log s)] over up to 5 accepted boxes",21,color=C['blue'])
    txt(64,956,"当前探测不写入历史；历史不足时复用最新中心。","Probes do not update history; use latest center if history is short.",19,color=C['muted'])
    # Panel c: scorer target, preserving offline/online separation without long notes.
    s.rect(976,562,984,423,'white',C['line'],r=8,sw=1.4)
    txt(996,602,"(c) 收益学习支线（离线）","(c) Utility learning alternative (offline)",26,weight=700)
    txt(998,649,"同一 fit 历史","Same fit history",20,color=C['muted'])
    for row in range(4):
        yy=671+row*42
        s.text(1013,yy+22,f'q{row}',19,C['amber'])
        for j in range(8):
            s.rect(1060+j*35,yy,27,26,'#E1C394' if j==0 else '#E1E6EA',r=2)
            if j<7: s.path([(1087+j*35,yy+13),(1094+j*35,yy+13)],C['gray'],width=1)
    txt(1196,864,"首帧干预 → 8 帧普通 E 推进","First-step action → 8-frame E rollout",18,color=C['muted'],anchor='middle')
    arrow([(1342,746),(1391,746)],C['amber'],True)
    s.rect(1392,652,539,190,C['amber_bg'],C['amber'],r=6,dash=True)
    txt(1662,686,"相对 current 的分支回报","Branch return relative to current",22,color=C['amber'],weight=600,anchor='middle')
    s.text(1414,731,'y₁ᵏ = IoUₜᵏ − IoUₜ⁰',24,C['amber'])
    s.text(1414,775,'y₈ᵏ = (1/8) Σⱼ₌₀⁷ (IoUₜ₊ⱼᵏ − IoUₜ₊ⱼ⁰)',21,C['amber'])
    txt(1414,817,"fit GT 仅用于离线标签","Fit GT for offline labels only",18,color=C['muted'])
    arrow([(1661,842),(1661,878)],C['amber'],True)
    node(1392,881,539,55,t('24 维特征 φ → 线性评分器 θ₁ / θ₈','24-D features φ → linear scorer θ₁ / θ₈'),C['amber_bg'],C['amber'],20,dash=True)
    txt(998,969,"只拟合评分器，SGLA 参数固定；在线不读取未来帧或 GT。","Fit scorer only; SGLA stays fixed. No future frames or GT online.",19,color=C['muted'])
    # Small visual connectors identify b/c as detail views, not extra processing.
    s.path([(437,421),(437,440)],C['blue'],width=1.5,dash=True)
    s.text(445,437,'(b)',16,C['blue'])
    s.path([(1543,337),(1543,401)],C['amber'],width=1.5,dash=True)
    s.text(1553,402,'(c)',16,C['amber'])
    txt(40,1032,"E：已有几何适配基线。彩色表示研究模块，灰色表示保留结构；示意图不代表具体帧或分数。","E: existing geometry-adapted baseline. Colored modules highlight the study; gray modules retain the tracker. Images and maps are schematic.",18,color=C['muted'])
    return s.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rasterize',action='store_true')
    args=p.parse_args()
    for lang in ('en','zh'):
        path=ROOT/f'sgla_search_state_framework_{lang}_v2.svg'
        path.write_text(render(lang),encoding='utf-8')
        print(path.name)
        if args.rasterize: base.rasterize(path,2)


if __name__=='__main__': main()
