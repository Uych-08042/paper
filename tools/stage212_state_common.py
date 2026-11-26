"""Online raw/adapted state factorization and complete-video statistics."""
import csv,hashlib,io,json,re
from pathlib import Path
import numpy as np
import stage211_geometry_audit as prior
old=prior.old
SCHEMA='stage212_online_output_search_state_factorial_v1'
SOURCE211='42626623990807863c851939f51c191823accc1085fa218801f0daf7a58d1be0'
BUNDLE211='381e3f7797bb96397bc8f431e742a8d055a6eb1255437ed71980e3e60255798b'
MODES=('mm','nm','mn','nn')
ARMS=('native','matched_single',*MODES)
SEEDS=prior.c.SEEDS
PROTOCOL=prior.runtime.common.source.common.protocol
METRICS=(*PROTOCOL.METRICS,'missed_matched_recoveries','matched_good_frame_loss')
TABLES=('model_scope','smoke','control_replay','policy_summary','paired','factor_summary','seed_video_summary','routing','calls','execution')


def source211(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE211:raise ValueError('Need exact reviewed Stage-211 bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage211_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if (len(parts)!=len(ms) or set(parts)!={'manifest.json','stage211_decision.json',*d['table_sha256']}
            or d['fingerprint']!=SOURCE211 or d['plan_fingerprint']!=p['fingerprint'] or d['source_groups']!=38
            or d['new_training_updates']!=0 or d['new_full_trajectories']!=0 or d['confirmation_groups_opened']!=0):raise ValueError('Changed source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table')
        tables[name.removeprefix('stage211_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    if set(tables)!=set(prior.c.TABLES):raise ValueError('Incomplete source tables')
    return d,p,tables


def next_state(raw,adapted,mode):return prior.c.parent(raw,adapted,mode)


def audit(trace,initial,frames,arm):
    if arm not in MODES:
        old.audit_trace(trace,initial,frames);return
    np.testing.assert_array_equal(trace['frame_id'],np.arange(1,frames+1))
    for k in ('boxes','reference_boxes','next_parent','parent'):
        if trace[k].shape!=(frames,4) or not np.isfinite(trace[k]).all() or (trace[k][:,2:]<=0).any():raise ValueError('Invalid state or output '+k)
    expected=np.stack([next_state(r,a,arm) for r,a in zip(trace['reference_boxes'],trace['boxes'])])
    np.testing.assert_array_equal(trace['next_parent'],expected)
    np.testing.assert_array_equal(trace['parent'],np.vstack((initial,trace['next_parent'][:-1])))
    if not np.isin(trace['layer'],range(7,13)).all():raise ValueError('Unexpected layer')
    calls=np.zeros((frames,12),int);calls[:,:6]=1;calls[np.arange(frames),trace['layer']-1]=2
    np.testing.assert_array_equal(trace['block_calls'],calls)
    for k,n in (('head_calls',2),('selector_calls',1),('crop_calls',1)):np.testing.assert_array_equal(trace[k],np.full(frames,n,int))


def summarize(rows):
    groups=sorted({r['group'] for r in rows});index={};summary=[];pairs=[];factors=[]
    for arm in ARMS:
        for g in groups:
            rr=[r for r in rows if r['arm']==arm and r['group']==g]
            if sorted(r['seed'] for r in rr)!=([0] if arm=='native' else SEEDS):raise ValueError('Incomplete OOF result')
            index[arm,g]={m:float(np.mean([r[m] for r in rr])) for m in METRICS}
            index[arm,g]['introduced_native']=any(r['introduced_failure'] for r in rr)
            index[arm,g]['introduced_matched']=any(r['introduced_matched_failure'] for r in rr)
        summary.append(dict(arm=arm,videos=len(groups),introduced_failure_videos=sum(index[arm,g]['introduced_native'] for g in groups),
            introduced_matched_failure_videos=sum(index[arm,g]['introduced_matched'] for g in groups),
            **{m:float(np.mean([index[arm,g][m] for g in groups])) for m in METRICS}))
    metrics=('auc','low_iou_fraction','longest_failure_fraction','missed_native_recoveries','native_good_frame_loss','missed_matched_recoveries','matched_good_frame_loss')
    comparisons=[('nm','mm'),('mn','mm'),('nn','mm'),('mm','matched_single')]+[(a,b) for a in MODES for b in ('native','matched_single') if (a,b)!=('mm','matched_single')]
    for a,b in comparisons:
        for metric in metrics:
            ds=np.asarray([index[a,g][metric]-index[b,g][metric] for g in groups]);primary=(a,b) in (('nm','mm'),('mn','mm')) and metric=='auc'
            family=dict(family_ci_low=None,family_ci_high=None)
            if primary:
                rng=np.random.default_rng(PROTOCOL.DESIGN['bootstrap_seed']);boot=ds[rng.integers(len(ds),size=(PROTOCOL.DESIGN['bootstrap_samples'],len(ds)))].mean(1)
                family=dict(family_ci_low=float(np.quantile(boot,.0125)),family_ci_high=float(np.quantile(boot,.9875)))
            pairs.append(dict(contrast=a+'_minus_'+b,metric=metric,primary=primary,favorable_direction='positive' if metric=='auc' else 'negative',**PROTOCOL.contrast(ds),**family))
    weights=dict(center_order_average={'nm':.5,'nn':.5,'mm':-.5,'mn':-.5},size_order_average={'mn':.5,'nn':.5,'mm':-.5,'nm':-.5},
                 center_by_size_interaction={'nn':1,'nm':-1,'mn':-1,'mm':1})
    for name,w in weights.items():
        for metric in metrics:factors.append(dict(contrast=name,metric=metric,**PROTOCOL.contrast([sum(v*index[a,g][metric] for a,v in w.items()) for g in groups])))
    return summary,pairs,factors
