"""Concise publication-style a/b framework with attributed real-image crops."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
spec=importlib.util.spec_from_file_location('vector_primitives',ROOT.parent/'sgla_search_state_v1/render_method_framework.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.W,base.H=1920,650
base.C.update(ink='#203145',muted='#536477',line='#CAD3DD',gray='#607187',gray_bg='#F0F3F6',
              blue='#356E9E',blue_bg='#EEF4FA',teal='#177E6C',teal_bg='#EBF5EF',amber='#A36723')
C=base.C;C.update(violet='#79659A',violet_bg='#F3EFF8')


def render(lang):
    s=base.SVG(lang)
    provenance=json.loads((ASSETS/'provenance.json').read_text(encoding='utf-8'))
    def tr(zh,en):return zh if lang=='zh' else en
    def text(x,y,zh,en=None,size=25,color='ink',bold=False,anchor='start'):
        s.text(x,y,tr(zh,en if en is not None else zh),size,C.get(color,color),700 if bold else 400,anchor)
    def center(x,y,zh,en=None,size=25,color='ink',bold=False):text(x,y,zh,en,size,color,bold,'middle')
    def box(x,y,w,h,color='blue',fill=None,r=5):s.rect(x,y,w,h,fill or C[color+'_bg'],C[color],r=r,sw=1.4)
    def arrow(points,color='gray',width=2.2):
        d='M '+' L '.join(f'{x},{y}' for x,y in points)
        s.add(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" marker-end="url(#{color})"/>')
    def photo(name,x,y,w,h,mark=False,color='blue'):
        record=provenance['assets'][name];body=(ASSETS/record['file']).read_bytes()
        assert hashlib.sha256(body).hexdigest()==record['sha256']
        encoded=base64.b64encode(body).decode('ascii')
        s.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" href="data:image/jpeg;base64,{encoded}"/>')
        s.rect(x,y,w,h,'none',C['line'],r=0,sw=1.0)
        if mark:
            cx,cy,cw,ch=record['crop'];bx,by,bw,bh=provenance['target_box_original']
            s.rect(x+(bx-cx)*w/cw,y+(by-cy)*h/ch,bw*w/cw,bh*h/ch,'none',C[color],r=0,sw=2)

    s.rect(0,0,1920,650,'white',r=0)
    s.rect(24,24,894,554,'#FBFCFE',C['line'],r=9,sw=1)
    s.rect(942,24,954,554,'#FBFCFE',C['line'],r=9,sw=1)
    text(47,67,'(a) 历史条件搜索','(a) History-conditioned search',30,'blue',True)
    text(965,67,'(b) SGLA 候选选择','(b) SGLA-based candidate selection',30,'blue',True)

    # a: own history, learned state distribution, scenario coverage, image crops.
    for k in range(3):photo(f'history{k}',58+k*20,145-k*13,128,96,True)
    center(143,286,'H(t−1)',size=27,bold=True)
    center(143,320,'x(t) ∈ R³²',size=23,color='muted')
    arrow([(226,182),(253,182)])
    box(254,126,198,140,'blue')
    center(353,164,'状态预测','State predictor',26,'blue',True)
    center(353,224,'μ(t), α σ(t)',size=29,color='blue')
    arrow([(452,192),(485,192)])

    cx,cy=571,190
    for sign in (-1,1):
        for k in range(16):
            r=math.sqrt((k+.5)/16);a=k*2.399963
            s.circle(cx+sign*67*r*math.cos(a),cy+sign*52*r*math.sin(a),2.8,C['violet'])
    s.rect(cx-9,cy-12,18,24,'none',C['blue'],r=0,sw=1.6)
    center(571,286,'32 个场景','32 scenarios',25,'violet',True)
    arrow([(650,192),(694,192)])

    box(695,126,198,140,'teal')
    center(794,163,'覆盖分配','Greedy coverage',23,'teal',True)
    center(794,209,'34 → 4',size=30,color='teal',bold=True)
    center(794,246,'P₀ ＋ 3',size=23,color='teal')
    arrow([(794,266),(794,306),(441,306),(441,333)],'blue')
    photo('current',65,373,190,142.5,True)
    center(160,554,'I(t)',size=28,bold=True)
    arrow([(255,444),(307,444),(307,382),(333,382)])
    for k in range(4):
        x=335+(k%2)*116;y=335+(k//2)*116
        photo(f'search{k}',x,y,96,96)
    center(441,570,'搜索裁剪','Search crops',24,'blue',True)

    # b: the template is immutable; SGLA is reused, not four independent models.
    photo('template',1061,115,112,112,True)
    center(1117,252,'z₀',size=27,bold=True)
    arrow([(1117,260),(1117,280)])
    box(987,281,264,226,'gray')
    center(1119,317,'SGLA',size=29,bold=True)
    for yy,zh,en in ((334,'L1–L6','L1–L6'),(392,'L7–L12 选1','L7–L12: select 1'),(450,'CENTER','CENTER')):
        box(1007,yy,224,39,'gray',fill='white')
        center(1119,yy+28,zh,en,size=22 if yy==392 else 24)
    arrow([(1119,373),(1119,390)],width=1.5)
    arrow([(1119,431),(1119,448)],width=1.5)
    arrow([(547,440),(965,440),(965,353),(986,353)],'blue')
    arrow([(1231,470),(1274,470),(1274,350),(1297,350)])

    for k in range(4):
        yy=222+k*72
        text(1300,yy+39,f'q{k}',size=25,color='blue',bold=True)
        photo(f'candidate{k}',1340,yy,60,60,True)
    center(1353,553,'4 × 64',size=24,color='blue')
    arrow([(1414,350),(1450,350)])

    box(1451,278,217,166,'teal')
    center(1559,317,'共享评分','Shared scorer',27,'teal',True)
    center(1559,366,'64 → 32 → 1',size=27,bold=True)
    center(1559,414,'s₀, s₁, s₂, s₃',size=25,color='teal')
    arrow([(1668,350),(1714,350)],'teal')
    center(1792,294,'argmax',size=28,color='teal',bold=True)
    photo('output',1715,315,156,117,True,'teal')
    center(1793,478,'b̂(t) = q*',size=27,color='teal',bold=True)

    # Completed baseline feedback. No experimental commit gate is depicted.
    arrow([(1793,492),(1793,611),(39,611),(39,190),(57,190)],'teal',2.5)
    s.rect(766,592,313,36,'white',r=0)
    center(922,620,'历史更新','History update',25,'teal',True)

    svg=s.finish()
    title=tr('历史条件搜索与SGLA共享质量选择','History-conditioned search and shared quality selection with SGLA')
    desc=('Concise panels a and b for the Stage-113 generated/shared architecture. Real photographic crops from the attributed '
          'VisDrone example illustrate history, search crops, template and candidates; all boxes and repeated crops are schematic '
          'and are not ground truth, consecutive frames, or experimental predictions. The standard proposal is retained. '
          'SGLA uses six prefix blocks and one selected block among L7-L12. The offline Stage-120 commit study is not drawn.')
    svg=svg[:svg.index('<title')]+f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc>'+svg[svg.index('<defs>'):]
    markers=''.join(f'<marker id="{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{C[c]}"/></marker>' for c in ('blue','violet'))
    return svg.replace('</defs>',markers+'</defs>',1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rasterize',action='store_true');args=parser.parse_args()
    for lang in ('en','zh'):
        path=ROOT/f'sgla_history_search_paper_{lang}_v5.svg';path.write_text(render(lang),encoding='utf-8');print(path)
        if args.rasterize:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'rasterize_svg.ps1'),
                '-SvgPath',str(path)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=55)
            if result.returncode:raise RuntimeError(result.stderr[-2500:])
            print(path.with_suffix('.png'))


if __name__=='__main__':main()
