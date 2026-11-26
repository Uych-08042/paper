"""Frozen recovery sample and center x size parent interventions."""
import csv,hashlib,io,json,re
from pathlib import Path
import numpy as np
import stage210_recovery_audit as previous
old=previous.old
SCHEMA='stage211_recovery_center_size_factorial_v1'
SOURCE210='7cb046228b6674d1d59b0b3f848e6cc4377649de5954d32b71b2730f6d94aaf8'
BUNDLE210='abba43753092b39109ba02ef762af177fc008404c6e9eb4c223faeb8c063d1d0'
SEEDS=previous.c.SEEDS
MODES=('mm','nm','mn','nn')
TABLES=('events','source_support','selection','cells','event_cells','summary','contrasts','endpoint_replay','model_scope','execution')


def source210(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE210:raise ValueError('Need exact reviewed Stage-210 bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage210_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if (len(parts)!=len(ms) or set(parts)!={'manifest.json','stage210_decision.json',*d['table_sha256']}
            or d['fingerprint']!=SOURCE210 or d['plan_fingerprint']!=p['fingerprint'] or d['source_groups']!=38
            or d['new_training_updates']!=0 or d['new_full_trajectories']!=0 or d['confirmation_groups_opened']!=0):raise ValueError('Unexpected source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table')
        tables[name.removeprefix('stage210_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    if set(tables)!=set(previous.c.TABLES):raise ValueError('Changed source table inventory')
    return d,p,tables


def parent(native,matched,mode):
    a,b=np.asarray(native,float),np.asarray(matched,float)
    if mode not in MODES or a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(np.r_[a,b]).all() or min(*a[2:],*b[2:])<=0:
        raise ValueError('Invalid hybrid parent')
    # Endpoints preserve original bits; do not reconstruct x/y needlessly.
    if mode=='nn':return a.copy()
    if mode=='mm':return b.copy()
    center=(a if mode[0]=='n' else b);size=(a if mode[1]=='n' else b)[2:]
    return np.r_[center[:2]+center[2:]/2-size/2,size]


def summarize(cells,roster):
    event_rows=[]
    keys=sorted({(r['sequence'],r['seed'],r['kind'],r['event'],r['mode']) for r in cells})
    for seq,seed,kind,event,mode in keys:
        rr=[r for r in cells if (r['sequence'],r['seed'],r['kind'],r['event'],r['mode'])==(seq,seed,kind,event,mode)]
        if sorted(r['offset'] for r in rr)!=[0,1,2]:raise ValueError('Incomplete window')
        event_rows.append(dict(sequence=seq,group=rr[0]['group'],seed=seed,kind=kind,event=event,stratum=rr[0]['stratum'],mode=mode,
            recovery_three_good=all(r['output_iou']>=.5 for r in rr),
            **{m:float(np.mean([r[k] for r in rr])) for m,k in (
                ('mean_output_iou','output_iou'),('mean_dense_best_iou','dense_best_iou'),('full_coverage_fraction','fully_covered'),
                ('mean_gt_width_network_px','gt_width_network_px'),('mean_gt_height_network_px','gt_height_network_px'))},
            readout_miss_fraction=float(np.mean([r['category']=='good_grid_missed' for r in rr])),
            covered_no_good_fraction=float(np.mean([r['category']=='covered_no_good_grid' for r in rr])),
            not_full_no_good_fraction=float(np.mean([r['category']=='not_fully_covered_no_good_grid' for r in rr]))))
    metrics=('recovery_three_good','mean_output_iou','mean_dense_best_iou','full_coverage_fraction','mean_gt_width_network_px','mean_gt_height_network_px',
             'readout_miss_fraction','covered_no_good_fraction','not_full_no_good_fraction')
    summaries=[];contrasts=[];protocol=previous.previous.prior.common.source.common.protocol
    for stratum in ('all_recovery','retained','lost','normal'):
        pool=[r for r in event_rows if (r['kind']=='recovery' if stratum=='all_recovery' else r['stratum']==stratum)]
        groups=sorted({r['group'] for r in pool});index={}
        for mode in MODES:
            rr=[r for r in pool if r['mode']==mode]
            for g in groups:
                gg=[r for r in rr if r['group']==g]
                index[mode,g]={m:float(np.mean([np.mean([r[m] for r in gg if r['seed']==s]) for s in sorted({r['seed'] for r in gg})])) for m in metrics}
            summaries.append(dict(stratum=stratum,mode=mode,roster_sources=len(roster),supported_sources=len(groups),event_seed_windows=len(rr),
                **{m:float(np.mean([index[mode,g][m] for g in groups])) if groups else '' for m in metrics}))
        effects=dict(center_at_matched_size={'nm':1,'mm':-1},center_at_native_size={'nn':1,'mn':-1},
            size_at_matched_center={'mn':1,'mm':-1},size_at_native_center={'nn':1,'nm':-1},
            center_order_average={'nm':.5,'mm':-.5,'nn':.5,'mn':-.5},size_order_average={'mn':.5,'mm':-.5,'nn':.5,'nm':-.5},
            center_by_size_interaction={'nn':1,'nm':-1,'mn':-1,'mm':1},total_native_parent={'nn':1,'mm':-1})
        for name,weights in effects.items():
            for metric in ('mean_output_iou','mean_dense_best_iou','recovery_three_good','full_coverage_fraction'):
                ds=[sum(w*index[mode,g][metric] for mode,w in weights.items()) for g in groups]
                result=protocol.contrast(ds) if len(ds)>1 else dict(videos=len(ds),mean_delta=ds[0] if ds else '',median_delta=ds[0] if ds else '',ci_low='',ci_high='',positive=sum(v>0 for v in ds),negative=sum(v<0 for v in ds),worst_leave_one_video_out='')
                contrasts.append(dict(stratum=stratum,contrast=name,metric=metric,**result))
    return event_rows,summaries,contrasts
