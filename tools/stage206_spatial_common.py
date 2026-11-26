"""Fixed Stage-206 comparisons, frozen sources and fit-only spatial donors."""
import hashlib
import numpy as np
import stage194_module_common as old
import stage194_module_ablation as io
import stage206_spatial_model as model

SCHEMA='stage206_native_routed_spatial_adaptation_v1'
ARMS=('native',*model.ARMS)
DESIGN=dict(old.DESIGN)
METRICS=old.METRICS


def references(bank,fold,lens_z):
    fit=list(fold['fit'])
    if len(fit)<2 or set(fit)&set(fold['held']):raise ValueError('Invalid fit reference split')
    mapping={g:fit[(fit.index(g)+1)%len(fit)] if g in fit else fit[int(hashlib.sha256(g.encode()).hexdigest(),16)%len(fit)] for g in bank}
    means={g:np.asarray(bank[g]['tokens'],np.float64)[:,:lens_z].mean(0).astype(np.float32) for g in fit}
    return {g:means[mapping[g]].copy() for g in bank},mapping


def summarize(rows):
    groups=sorted({r['group'] for r in rows});index={};summary=[]
    for arm in ARMS:
        for g in groups:
            cell=[r for r in rows if r['arm']==arm and r['group']==g]
            seeds=[0] if arm=='native' else DESIGN['seeds']
            if sorted(r['seed'] for r in cell)!=seeds:raise ValueError('Incomplete full OOF grid')
            index[arm,g]={k:float(np.mean([r[k] for r in cell])) for k in METRICS}
            index[arm,g]['introduced']=any(r['introduced_failure'] for r in cell)
        summary.append(dict(arm=arm,videos=len(groups),introduced_failure_videos=sum(index[arm,g]['introduced'] for g in groups),
            **{k:float(np.mean([index[arm,g][k] for g in groups])) for k in METRICS}))
    pairs=[]
    for control in ('native','search_control','mismatched_spatial'):
        for metric in ('auc','low_iou_fraction','longest_failure_fraction','missed_native_recoveries','native_good_frame_loss'):
            pairs.append(dict(contrast='matched_spatial_minus_'+control,metric=metric,primary=metric=='auc',
                favorable_direction='positive' if metric=='auc' else 'negative',
                **old.contrast([index['matched_spatial',g][metric]-index[control,g][metric] for g in groups])))
    for arm in ('search_control','mismatched_spatial'):
        pairs.append(dict(contrast=arm+'_minus_native',metric='auc',primary=False,favorable_direction='positive',
            **old.contrast([index[arm,g]['auc']-index['native',g]['auc'] for g in groups])))
    return summary,pairs
