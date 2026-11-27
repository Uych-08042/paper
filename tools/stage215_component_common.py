"""Frozen learned-parent center/scale factorial; no training or new thresholds."""
import csv,hashlib,io,json,re
from pathlib import Path
import numpy as np
import stage214_search_state as source
old=source.old;protocol=source.c.protocol;SEEDS=source.c.SEEDS
SCHEMA='stage215_frozen_parent_components_v1'
SOURCE214='ed45014f53c42067f3ff9381263c36a9cd9736c02051bc4719ce05bd5dab9340'
BUNDLE214='e9fde8f36b8da411d31dd91fe7bafb95853f9089953567b57ff728bdd59907b6'
MASKS=dict(none=(0,0,0),center=(1,1,0),scale=(0,0,1),joint=(1,1,1))
FAMILIES=source.c.TRAINED
FRESH=tuple(f'{f}__{m}' for f in FAMILIES for m in MASKS)
ARMS=('native','mm',*FRESH)
TABLES=('model_scope','smoke','source_replay','policy_summary','paired','factor_summary','seed_video_summary','search_geometry','calls','execution')


def source214(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE214:raise ValueError('Need exact reviewed Stage-214 bundle')
    mm=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw));pp={m[1].decode():raw[m.end():mm[i+1].start() if i+1<len(mm) else len(raw)] for i,m in enumerate(mm)}
    d,p=json.loads(pp['stage214_decision.json']),json.loads(pp['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if len(pp)!=len(mm) or set(pp)!={'manifest.json','stage214_decision.json',*d['table_sha256']} or d['fingerprint']!=SOURCE214 or d['plan_fingerprint']!=p['fingerprint'] or (d['source_groups'],d['training_runs'],d['training_updates'],d['full_evaluation_trajectories'],d['confirmation_groups_opened'])!=(38,8,2048,342,0):raise ValueError('Changed source scope')
    tables={}
    for n,h in d['table_sha256'].items():
        b=pp[n];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==h),None)
        if data is None:raise ValueError('Changed source table '+n)
        tables[n[9:-4]]=list(csv.DictReader(io.StringIO(data.decode())))
    if set(tables)!=set(source.c.TABLES):raise ValueError('Incomplete source tables')
    return d,p,tables


class Masked:
    """Always evaluate the original network once, then disable specified outputs."""
    def __init__(self,model,mode):
        if mode not in MASKS:raise ValueError('Unknown component rule')
        self.model,self.mode=model,mode
    def parameters(self):return self.model.parameters()
    def state_dict(self):return self.model.state_dict()
    def __call__(self,x):
        value=self.model(x)
        if self.mode=='joint':return value
        # where makes disabled values exactly zero, including a negative output.
        import torch
        return torch.where(torch.as_tensor(MASKS[self.mode],device=value.device,dtype=torch.bool),value,torch.zeros_like(value))


def audit(v,initial,n,arm):
    family,mode=arm.split('__');source.c.audit(v,initial,n,family)
    if family not in FAMILIES or mode not in MASKS:raise ValueError('Unknown factorial cell')
    for j,enabled in enumerate(MASKS[mode]):
        if not enabled:np.testing.assert_array_equal(v['state_delta'][:,j],np.zeros(n))
    np.testing.assert_array_equal(v['updater_calls'],np.ones(n,int))
    if mode=='none':np.testing.assert_array_equal(v['next_parent'],v['reference_boxes'])


def summarize(rows):
    groups=sorted({r['group'] for r in rows});idx={};summary=[];paired=[];factors=[]
    for a in ARMS:
        for g in groups:
            rr=[r for r in rows if r['group']==g and r['arm']==a]
            if sorted(r['seed'] for r in rr)!=([0] if a=='native' else SEEDS):raise ValueError('Incomplete OOF grid')
            idx[a,g]={m:float(np.mean([r[m] for r in rr])) for m in source.c.METRICS}
            idx[a,g]['introduced']=any(r['introduced_failure'] for r in rr);idx[a,g]['introduced_mm']=any(r['introduced_matched_failure'] for r in rr)
        summary.append(dict(arm=a,videos=len(groups),provenance='new_frozen_component_rollout' if a in FRESH else 'reused_stage214_reference',introduced_failure_videos=sum(idx[a,g]['introduced'] for g in groups),introduced_matched_failure_videos=sum(idx[a,g]['introduced_mm'] for g in groups),**{m:float(np.mean([idx[a,g][m] for g in groups])) for m in source.c.METRICS}))
    metrics=('auc','low_iou_fraction','missed_native_recoveries','native_good_frame_loss','missed_matched_recoveries','matched_good_frame_loss')
    for family in FAMILIES:
        name=lambda m:family+'__'+m
        comparisons=[(name(a),name(b)) for a,b in (('center','joint'),('scale','joint'),('center','none'),('scale','none'),('joint','none'))]+[(name(a),b) for a in ('center','scale','joint') for b in ('native','mm')]
        for a,b in comparisons:
            for metric in metrics:
                paired.append(dict(contrast=a+'_minus_'+b,metric=metric,primary=(a,b,metric)==('coverage__center','coverage__joint','auc'),favorable_direction='positive' if metric=='auc' else 'negative',**protocol.contrast([idx[a,g][metric]-idx[b,g][metric] for g in groups])))
        weights=dict(center_order_average=dict(center=.5,joint=.5,none=-.5,scale=-.5),scale_order_average=dict(scale=.5,joint=.5,none=-.5,center=-.5),center_by_scale_interaction=dict(joint=1,none=1,center=-1,scale=-1))
        for label,w in weights.items():
            for metric in metrics:factors.append(dict(family=family,contrast=label,metric=metric,**protocol.contrast([sum(v*idx[name(a),g][metric] for a,v in w.items()) for g in groups])))
    return summary,paired,factors
