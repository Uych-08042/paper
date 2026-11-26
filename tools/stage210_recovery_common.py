"""Fixed natural-recovery events and source-equal diagnostic summaries."""
import csv,hashlib,io,json,re
from pathlib import Path
import numpy as np
import stage209_reference_intervention as previous

old=previous.old
SCHEMA='stage210_all_source_recovery_mechanism_v1'
SOURCE209='6f0dd285f797b36b3a6a52244c02b458d89c29aa36ae157a1b4910ed52aca25f'
BUNDLE209='24c4323b9266efe2bd6589f164d752f48503c5258292674a29cb3720639225e5'
SEEDS=previous.c.DESIGN['seeds']
TABLES=('events','source_support','selection','cells','event_cells','summary','contrasts','replay','model_scope','execution')


def source209(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE209:raise ValueError('Need exact reviewed Stage-209 bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage209_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if (len(parts)!=len(ms) or set(parts)!={'manifest.json','stage209_decision.json',*d['table_sha256']}
            or d['fingerprint']!=SOURCE209 or d['plan_fingerprint']!=p['fingerprint'] or d['new_training_updates']!=0
            or d['full_evaluation_trajectories']!=342 or d['confirmation_groups_opened']!=0):raise ValueError('Changed source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table')
        tables[name.removeprefix('stage209_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    return d,p,tables


def select(spec,native,matched,truth):
    """All native recoveries; uniformly located normal three-frame controls."""
    ni=old.tracking_iou(native['boxes'],truth[1:]);events=old.failure_events(ni)
    inventory=[];selection=[];recoveries=[];excluded=np.zeros(len(ni),bool)
    for j,e in enumerate(events):
        r=e['recovery'];excluded[e['start']:min(len(ni),e['end']+1)]=True
        if r is not None:excluded[r:r+3]=True;recoveries.append((j,r))
        inventory.append(dict(sequence=spec['sequence'],group=spec['group'],event=j,onset=e['start']+1,
            confirmation=e['confirmation']+1,recovery='' if r is None else r+1,censor=e['censor']))
    eligible=[i for i in range(len(ni)-2) if not excluded[i:i+3].any() and (ni[i:i+3]>=.5).all()]
    count=min(len(eligible),max(1,len(recoveries)))
    controls=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,count).round().astype(int))] if count else []
    windows=[('recovery',j,r) for j,r in recoveries]+[('control',j,r) for j,r in enumerate(controls)]
    for seed in SEEDS:
        mi=old.tracking_iou(matched[seed]['boxes'],truth[1:])
        for kind,j,r in windows:
            preserved=bool((mi[r:r+3]>=.5).all())
            for offset in range(3):selection.append(dict(sequence=spec['sequence'],group=spec['group'],seed=seed,
                kind=kind,event=j,window_start=r+1,frame=r+offset+1,offset=offset,
                stratum=('retained' if preserved else 'lost') if kind=='recovery' else 'normal',
                native_iou=float(ni[r+offset]),matched_iou=float(mi[r+offset])))
    support=dict(sequence=spec['sequence'],group=spec['group'],native_events=len(events),native_recoveries=len(recoveries),
        native_unrecovered=len(events)-len(recoveries),eligible_control_starts=len(eligible),controls=len(controls),has_diagnostic_support=bool(windows))
    return inventory,selection,support


def category(covered,best_iou,output_iou):
    if output_iou>=.5:return 'good_output'
    if best_iou>=.5:return 'good_grid_missed'
    return 'covered_no_good_grid' if covered else 'not_fully_covered_no_good_grid'


def summarize(cells,roster):
    # Overlapping control windows are allowed, but each named window is one event.
    event_rows=[]
    keys=sorted({(r['sequence'],r['seed'],r['kind'],r['event'],r['parent_origin'],r['model']) for r in cells})
    for seq,seed,kind,event,parent,model in keys:
        rr=[r for r in cells if (r['sequence'],r['seed'],r['kind'],r['event'],r['parent_origin'],r['model'])==(seq,seed,kind,event,parent,model)]
        if sorted(r['offset'] for r in rr)!=[0,1,2]:raise ValueError('Incomplete three-frame diagnostic')
        event_rows.append(dict(sequence=seq,group=rr[0]['group'],seed=seed,kind=kind,event=event,stratum=rr[0]['stratum'],
            parent_origin=parent,model=model,recovery_three_good=bool(all(r['output_iou']>=.5 for r in rr)),
            mean_output_iou=float(np.mean([r['output_iou'] for r in rr])),
            mean_dense_best_iou=float(np.mean([r['dense_best_iou'] for r in rr])),
            full_coverage_fraction=float(np.mean([r['fully_covered'] for r in rr])),
            readout_miss_fraction=float(np.mean([r['category']=='good_grid_missed' for r in rr])),
            covered_no_good_fraction=float(np.mean([r['category']=='covered_no_good_grid' for r in rr])),
            not_full_no_good_fraction=float(np.mean([r['category']=='not_fully_covered_no_good_grid' for r in rr]))))
    summaries=[];contrasts=[];metrics=('recovery_three_good','mean_output_iou','mean_dense_best_iou','full_coverage_fraction','readout_miss_fraction','covered_no_good_fraction','not_full_no_good_fraction')
    for stratum in ('all_recovery','retained','lost','normal'):
        pool=[r for r in event_rows if (r['kind']=='recovery' if stratum=='all_recovery' else r['stratum']==stratum)]
        index={}
        for parent in ('native','matched'):
            for model in ('native','matched'):
                rr=[r for r in pool if r['parent_origin']==parent and r['model']==model];groups=sorted({r['group'] for r in rr})
                for g in groups:
                    gg=[r for r in rr if r['group']==g]
                    # seed first, then source. Lost/retained strata may have one supported seed.
                    index[parent,model,g]={m:float(np.mean([np.mean([r[m] for r in gg if r['seed']==s]) for s in sorted({r['seed'] for r in gg})])) for m in metrics}
                summaries.append(dict(stratum=stratum,parent_origin=parent,model=model,roster_sources=len(roster),supported_sources=len(groups),
                    event_seed_windows=len(rr),**{m:float(np.mean([index[parent,model,g][m] for g in groups])) if groups else '' for m in metrics}))
        groups=sorted({r['group'] for r in pool})
        for name,a,b in (('parent_effect_native_model',('native','native'),('matched','native')),
                         ('parent_effect_matched_model',('native','matched'),('matched','matched')),
                         ('model_effect_native_parent',('native','matched'),('native','native')),
                         ('model_effect_matched_parent',('matched','matched'),('matched','native'))):
            for m in ('mean_output_iou','mean_dense_best_iou','recovery_three_good'):
                ds=[index[a[0],a[1],g][m]-index[b[0],b[1],g][m] for g in groups]
                result=previous.prior.common.source.common.protocol.contrast(ds) if len(ds)>1 else dict(videos=len(ds),mean_delta=ds[0] if ds else '',median_delta=ds[0] if ds else '',ci_low='',ci_high='',positive=sum(v>0 for v in ds),negative=sum(v<0 for v in ds),worst_leave_one_video_out='')
                contrasts.append(dict(stratum=stratum,contrast=name,metric=m,**result))
    return event_rows,summaries,contrasts
