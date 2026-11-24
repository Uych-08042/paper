import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const size = {width: 1560, height: 668};
export const colors = {
  ink: '#222222', white: '#FFFFFF', gray: '#A4A4A4', paleGray: '#F4F4F4',
  cream: '#FFF6D9', yellow: '#FFC01D', blue: '#2079C5', blueLight: '#BDD7EE',
  sky: '#C1E5F5', lavender: '#CED5EF', green: '#196B24', greenLight: '#C5E0B5',
  mint: '#D9F2D0', pink: '#FEE0E0', red: '#C00000', orange: '#EE822F',
  orangePale: '#FFF2CD', teal: '#008E77',
};
const assets = new URL('../sgla_search_state_v5/assets/', import.meta.url);
const provenance = JSON.parse(readFileSync(new URL('provenance.json', assets), 'utf8'));

export function framework(language = 'zh'){
  const C = colors;
  const ops = [];
  const add = (kind, args) => ops.push({kind, ...args});
  const tr = (zh, en) => language === 'zh' ? zh : en;
  const box = (x,y,w,h,bg,stroke='none',sw=1.6,dash=false,rx=7) =>
    add('box',{x,y,w,h,bg,stroke,sw,dash,rx});
  const text = (x,y,w,h,zh,en,fs=23,bold=false,c=C.ink,align='center') =>
    add('text',{x,y,w,h,value:tr(zh,en),fs,bold,c,align});
  const path = (pts,c=C.ink,sw=2.05,arrow=true,dash=false) =>
    add('path',{pts,c,sw,arrow,dash});
  const photo = (name,x,y,w,h,border,marker) => {
    const record = provenance.assets[name];
    add('photo',{name,file:record.file,x,y,w,h});
    box(x,y,w,h,'none',border,1.65,false,0);
    if (marker){
      const [sx,sy,sw,sh] = record.crop;
      const [a,b,c,d] = marker;
      const bounds=[x+(a-sx)*w/sw,y+(b-sy)*h/sh,c*w/sw,d*h/sh];
      box(...bounds,'none',C.white,4.2,false,0);
      box(...bounds,'none',border,2.1,false,0);
    }
  };
  const T = provenance.target_box_original;

  // Upper path has one source crop and one independent late block per frame.
  box(26,28,1510,237,C.white,C.gray,1.2,false,10);
  box(494,82,754,146,C.cream,'none',0,false,12);
  text(42,39,600,35,'(a) SGLA 单裁剪跟踪','(a) Single-crop SGLA tracking',26,true,C.ink,'left');
  photo('current',43,103,144,108,C.blue,provenance.assets.search1.crop);
  photo('search1',224,103,108,108,C.blue);
  photo('template',367,103,108,108,C.red,T);
  text(38,216,154,35,'当前帧 Iₜ','Frame Iₜ',20,true);
  text(210,216,136,35,'搜索裁剪','Search crop',20,true);
  text(351,216,140,35,'首帧模板 z₀','Template z₀',20,true);
  path([[187,156],[224,156]]);
  path([[332,144],[348,144],[348,89],[498,89],[498,124],[514,124]],C.blue);
  path([[475,175],[514,175]],C.ink);
  box(514,105,212,101,C.lavender,'#5F6877',1.6,false,5);
  text(514,112,212,39,'共享前缀 L1–L6','Shared prefix L1–L6',20,true);
  [C.yellow,C.yellow,C.greenLight,C.greenLight,C.greenLight,C.greenLight].forEach((col,i) =>
    box(558+22*i,162,15,24,col,i<2?'#AA7900':C.green,0.8,false,0));
  path([[726,130],[1053,130]]);
  text(735,97,55,27,'Fₜ','Fₜ',22,true,C.blue);
  path([[841,130],[841,158]],C.ink);
  box(765,158,152,51,C.paleGray,C.gray,1.35);
  text(765,158,152,51,'原生选层 ℓ₀','Native layer ℓ₀',20,true);
  box(1053,103,180,108,C.white,'#5F6877',1.45);
  text(1053,104,180,31,'独立候选块','Independent blocks',18,true);
  for(let i=0;i<6;i++){
    const x=1065+(i%3)*52,y=138+Math.floor(i/3)*31;
    box(x,y,47,26,i===1?C.blueLight:C.white,i===1?C.blue:C.gray,1.2,false,2);
    text(x,y,47,26,'L'+(i+7),'L'+(i+7),17,i===1);
  }
  path([[1233,156],[1267,156]]);
  box(1267,118,121,77,C.mint,C.green,1.55);
  text(1267,126,121,31,'CENTER','CENTER',20,true);
  text(1267,157,121,25,'＋ Hann','+ Hann',18);
  path([[1388,156],[1408,156]]);
  photo('output',1408,108,112,100,C.green,[T[0]+3,T[1]+2,T[2],T[3]]);
  text(1402,215,123,35,'预测框 b̂ₜ','Box b̂ₜ',20,true);

  // Color follows function. The orange dashed inset marks the pending trial.
  box(26,289,1510,310,C.white,C.gray,1.2,false,10);
  text(42,300,690,35,'(b) 时序风险门控与成对选层','(b) Temporal risk gating and pairwise selection',26,true,C.ink,'left');
  box(48,349,413,207,'#F4FAFE',C.blue,2,true,9);
  box(65,371,170,57,C.white,C.blue,1.35);
  text(65,371,170,57,'自身历史 Hₜ₋₁','History Hₜ₋₁',20,true);
  box(256,371,185,57,C.blueLight,C.blue,1.35);
  text(256,371,185,57,'视觉投影 16 维','16D visual projection',17,true);
  path([[598,206],[598,280],[690,280],[690,342],[349,342],[349,371]],C.blue);
  path([[150,428],[150,473],[170,473],[170,482]],C.blue);
  path([[349,428],[349,473],[336,473],[336,482]],C.blue);
  box(66,482,375,52,C.white,C.blue,1.35);
  text(66,482,375,52,'时序状态   x₄（24 维）   x₁₆（27 维）','Temporal states   x₄ (24D)   x₁₆ (27D)',19,true);
  path([[441,508],[480,508],[480,391],[517,391]],C.blue);
  path([[480,455],[517,455]],C.blue);
  path([[480,508],[480,520],[517,520]],C.blue);
  box(517,365,204,52,C.sky,C.blue,1.35);
  text(517,365,204,52,'收益预测 g、pₜ','Gain and proposal pₜ',17,true);
  box(517,429,204,52,C.pink,C.red,1.35);
  text(517,429,204,52,'退化预测 r','Downside risk r',18,true);
  box(517,494,204,52,C.mint,C.green,1.35);
  text(517,494,204,52,'成对排序 ℓᵣ','Pairwise ranking ℓᵣ',19,true);
  box(752,349,327,207,'#FFFBF3',C.orange,2,true,9);
  path([[841,209],[841,274],[1000,274],[1000,370]],C.ink);
  path([[721,391],[774,391]],C.blue);
  path([[721,455],[774,455]],C.red);
  box(774,370,281,103,C.orangePale,C.orange,1.45);
  text(774,377,281,30,'共享风险门控','Shared risk gate',22,true);
  text(774,410,281,27,'sₜ = g(pₜ) − r(pₜ) > τ','sₜ = g(pₜ) − r(pₜ) > τ',21,true);
  text(774,439,281,25,'pₜ ≠ ℓ₀，至少 4 帧历史','pₜ ≠ ℓ₀, at least 4 past frames',16,false);
  path([[914,473],[914,490]],C.orange,1.75,true,true);
  box(774,490,281,49,C.white,C.orange,1.65,true);
  text(774,490,281,49,'16 帧冷却试验','16-frame cooldown trial',19,true);
  path([[1055,400],[1137,400]],C.ink);
  path([[1055,515],[1137,515]],C.orange,1.7,true,true);
  path([[721,520],[735,520],[735,574],[1262,574],[1262,539]],C.green,2.45);

  box(1137,370,261,169,C.mint,C.green,1.8);
  text(1137,379,261,40,'实际执行层 ℓₜ','Executed layer ℓₜ',24,true);
  path([[1157,425],[1378,425]],C.green,1,false);
  text(1137,431,261,36,'允许换层   ℓᵣ','Switch allowed   ℓᵣ',20,true);
  text(1137,473,261,35,'否则沿用   ℓ₀','Otherwise use   ℓ₀',20);
  path([[1267,370],[1267,252],[1143,252],[1143,211]],C.green,2.5);
  text(1154,267,109,23,'仅执行一层','One block',17,true,C.green);

  path([[1470,251],[1470,275],[1546,275],[1546,628],[37,628],[37,399],[65,399]],C.teal,2.3);
  box(530,612,500,31,C.white,'none',0,false,0);
  text(530,612,500,31,'预测框、响应峰值与实际换层写入下一帧','Feed back box, peak and switch state',19,true,C.teal);
  return ops;
}

const esc = (value) => value.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function svg(language){
  const ops = framework(language);
  const markers = [...new Set(ops.filter(o => o.arrow).map(o => o.c))];
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="${size.width}" height="${size.height}" viewBox="0 0 ${size.width} ${size.height}">`,
    '<title id="title">Temporal layer selection in SGLA</title>',
    '<desc id="desc">A single crop runs through a shared prefix. Causal history drives the risk gate and pairwise ranking. The dashed inset denotes a cooldown trial.</desc>',
    '<defs>'+markers.map(c => `<marker id="a${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`).join('')+'</defs>',
    `<rect width="${size.width}" height="${size.height}" fill="#FFFFFF" x="0" y="0"/>`,
  ];
  for(const o of ops){
    if(o.kind === 'box'){
      parts.push(`<rect x="${o.x}" y="${o.y}" width="${o.w}" height="${o.h}" rx="${o.rx}" fill="${o.bg}" stroke="${o.stroke}" stroke-width="${o.sw}"${o.dash?' stroke-dasharray="7 5"':''}/>`);
    }else if(o.kind === 'text'){
      if(!o.value) continue;
      const anchor=o.align==='left'?'start':'middle';
      parts.push(`<text x="${o.x+(anchor==='middle'?o.w/2:0)}" y="${o.y+o.h/2+o.fs*.36}" fill="${o.c}" font-size="${o.fs}" font-family="${language==='zh'?'Microsoft YaHei, sans-serif':'Arial, sans-serif'}" font-weight="${o.bold?700:400}" text-anchor="${anchor}">${esc(o.value)}</text>`);
    }else if(o.kind === 'path'){
      const d=o.pts.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ');
      parts.push(`<path d="${d}" fill="none" stroke="${o.c}" stroke-width="${o.sw}" stroke-linejoin="round"${o.arrow?' marker-end="url(#a'+o.c.slice(1)+')"':''}${o.dash?' stroke-dasharray="7 5"':''}/>`);
    }else if(o.kind === 'photo'){
      const photo=readFileSync(new URL(o.file,assets));
      if(createHash('sha256').update(photo).digest('hex')!==provenance.assets[o.name].sha256) throw new Error(o.file);
      parts.push(`<image x="${o.x}" y="${o.y}" width="${o.w}" height="${o.h}" preserveAspectRatio="none" href="data:image/jpeg;base64,${photo.toString('base64')}"/>`);
    }
  }
  return parts.concat('</svg>').join('\n');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]){
  const en = new URL('sgla_temporal_layer_framework_en_v7.svg', import.meta.url);
  const zh = new URL('sgla_temporal_layer_framework_zh_v7.svg', import.meta.url);
  writeFileSync(en, svg('en'));
  writeFileSync(zh, svg('zh'));
}
