"""Frozen Stage-207 paired states and the Stage-208 state x interaction design."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import stage207_state_training as source

old, previous, spatial = source.old, source.previous, source.spatial
SCHEMA = 'stage208_paired_state_interaction_factorial_v1'
SOURCE207 = '83a876782fdab0b95b24ea06197ce63cbd2dbb43f8a158cc272d640bdd9e149f'
PLAN207 = '36d2f75d22ba459e2ed698fe9afb66d11d51f1449b33511256a937a17c5e056f'
BUNDLE207 = '3cc83a070037f8be34f76fb6432ac8415ccb1e24f9d5442ff67fad97616281eb'
ORIGINS = ('native_pair', 'own_pair')
INTERACTIONS = spatial.ARMS
TRAINED = tuple(a+'__'+b for a in ORIGINS for b in INTERACTIONS)
ARMS = ('native', *TRAINED)
DESIGN = dict(source.common.DESIGN)
METRICS = source.common.protocol.METRICS
TABLES = ('support', 'sampling', 'reference_audit', 'scope_summary', 'training_history',
          'control_replay', 'readout_video', 'parallel_smoke', 'evaluation_execution',
          'policy_summary', 'paired', 'factor_summary', 'seed_video_summary', 'routing')


def source207(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE207:
        raise ValueError('Need the exact reviewed Stage-207 bundle')
    matches = list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n', raw))
    parts = {m[1].decode(): raw[m.end():matches[i+1].start() if i+1<len(matches) else len(raw)]
             for i,m in enumerate(matches)}
    d,p = json.loads(parts['stage207_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v['fingerprint'] != source.fingerprint({k:w for k,w in v.items() if k!='fingerprint'}):
            raise ValueError('Changed source fingerprint')
    if (d['fingerprint']!=SOURCE207 or p['fingerprint']!=PLAN207 or d['plan_fingerprint']!=PLAN207
            or d['training_runs']!=12 or d['training_updates']!=3072 or d['full_evaluation_trajectories']!=342
            or d['confirmation_groups_opened']!=0 or len(matches)!=len(parts)
            or set(parts)!={'manifest.json','stage207_decision.json','stage207_support_decision.json',*d['table_sha256']}
            or set(d['table_sha256'])!={f'stage207_{k}.csv' for k in source.TABLES}):
        raise ValueError('Unexpected Stage-207 scope')
    tables = {}
    for name,sha in d['table_sha256'].items():
        chunk=parts[name]
        data=next((b for b in (chunk,chunk[:-1],chunk[:-2]) if hashlib.sha256(b).hexdigest()==sha),None)
        if data is None: raise ValueError('Changed source table '+name)
        tables[name.removeprefix('stage207_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    support=json.loads(parts['stage207_support_decision.json'])
    if (old.sealed({k:v for k,v in support.items() if k!='fingerprint'})!=support
            or support['plan_fingerprint']!=PLAN207 or support['status']!='SUPPORT_READY'):
        raise ValueError('Missing frozen common-support preflight')
    return d,p,tables


def bound_arrays(root,decision,relative,metadata,verified):
    path=source.source_file(root,decision,relative,verified)
    receipt=old.read_sealed(source.source_file(root,decision,path.with_suffix('.receipt.json').relative_to(root).as_posix(),verified))
    if receipt['metadata']!=metadata or receipt['sha256']!=verified[relative]:
        raise ValueError('Changed source cache receipt')
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f['metadata_json'].item()))!=metadata: raise ValueError('Changed source NPZ identity')
        return {k:f[k] for k in f.files if k!='metadata_json'}


def paired_inputs(root,decision,plan,tables,native_bank,entries,search_size,verified):
    """Consume old raw prefixes; never recollect either side on the new GPU."""
    if not plan['same_source_runtime']:
        native_bank={g:bound_arrays(root,decision,'native_replay/'+e['sequence']+'.npz',
            dict(plan_fingerprint=plan['fingerprint'],group=g,role='current_runtime_native_parents'),verified)
            for e in plan['entries'] for g in [e['group']]}
    result={};support=[];sampling=[]
    for fold in plan['folds']:
        for seed in DESIGN['seeds']:
            provider=next(r for r in tables['provider_scope'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed)
            own={}
            for g in fold['fit']:
                spec=next(e for e in plan['entries'] if e['group']==g);seq=spec['sequence']
                value=bound_arrays(root,decision,f"providers/f{fold['fold']}__s{seed}__{seq}.npz",
                    dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,group=g,
                         provider_sha256=provider['model_sha256'],role='outer_fit_only'),verified)
                old.audit_trace({k.removeprefix('trace_'):v for k,v in value.items() if k.startswith('trace_')},
                                entries[seq][1][0],len(entries[seq][0])-1)
                own[g]=source.attach_truth(value,entries[seq][1],search_size)
            a,b,s,m=source.common.paired_bank({g:native_bank[g] for g in fold['fit']},own,search_size)
            if any(v['paired_states']==0 for v in s): raise ValueError('Frozen paired support became empty')
            result[fold['fold'],seed]={'native_pair':a,'own_pair':b}
            support.extend(dict(fold=fold['fold'],seed=seed,**r) for r in s)
            sampling.extend(dict(fold=fold['fold'],seed=seed,**r) for r in m)
    # These contain ints/bools only, so CSV string equality is exact.
    for name,rows in (('support',support),('sampling',sampling)):
        if [{k:str(v) for k,v in r.items()} for r in rows]!=tables[name]:
            raise ValueError('Stage-207 support/sampling changed: '+name)
    return native_bank,result,support,sampling


def control_states(torch,rt,root,decision,plan,tables,verified):
    result={}
    for fold in plan['folds']:
        for seed in DESIGN['seeds']:
            for origin in ORIGINS:
                rel=f"conditions/{origin}/training/f{fold['fold']}__s{seed}__matched_spatial.pt"
                path=source.source_file(root,decision,rel,verified)
                receipt=old.read_sealed(source.source_file(root,decision,path.with_suffix('.receipt.json').relative_to(root).as_posix(),verified))
                payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
                if (meta!=receipt['metadata'] or receipt['sha256']!=source.file_hash(path)
                        or meta['plan_fingerprint']!=plan['fingerprint'] or meta['fold']!=fold['fold']
                        or meta['seed']!=seed or meta['arm']!='matched_spatial' or meta['updates']!=256
                        or meta['fit']!=fold['fit'] or meta['held']!=fold['held']):
                    raise ValueError('Wrong Stage-207 paired student')
                row=next(r for r in tables['scope_summary'] if r['training_condition']==origin
                         and int(r['fold'])==fold['fold'] and int(r['seed'])==seed)
                previous.inputs.compare_scope(payload['scope'],{k:v for k,v in row.items() if k!='training_condition'})
                model=spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,seed,'matched_spatial')
                model.load_state_dict(payload['state'],strict=True)
                digest=old.model_digest(model)
                if digest!=payload['scope']['state_sha256']: raise ValueError('Changed control model')
                result[fold['fold'],seed,origin]=dict(state_sha256=digest,draws_sha256=meta['draws_sha256'],initial_sha256=meta['initial_sha256'])
    return result


def references(native_pair,fold,all_groups,lens_z):
    # The same wrong identity/content is used for both parent origins.
    refs,mapping=previous.common.references(native_pair,fold,lens_z)
    means={g:np.asarray(native_pair[g]['tokens'],np.float64)[:,:lens_z].mean(0).astype(np.float32) for g in fold['fit']}
    for g in all_groups:
        if g not in refs:
            donor=fold['fit'][int(hashlib.sha256(g.encode()).hexdigest(),16)%len(fold['fit'])]
            refs[g]=means[donor].copy();mapping[g]=donor
    if any(v not in fold['fit'] or v in fold['held'] or g==v for g,v in mapping.items()):
        raise ValueError('Held or self donor')
    return refs,mapping


def summarize(rows):
    groups=sorted({r['group'] for r in rows});index={};summary=[]
    for arm in ARMS:
        for g in groups:
            cell=[r for r in rows if r['arm']==arm and r['group']==g]
            seeds=[0] if arm=='native' else DESIGN['seeds']
            if sorted(r['seed'] for r in cell)!=seeds: raise ValueError('Incomplete factorial OOF')
            index[arm,g]={m:float(np.mean([r[m] for r in cell])) for m in METRICS}
            index[arm,g]['introduced']=any(r['introduced_failure'] for r in cell)
        summary.append(dict(arm=arm,videos=len(groups),introduced_failure_videos=sum(index[arm,g]['introduced'] for g in groups),
                            **{m:float(np.mean([index[arm,g][m] for g in groups])) for m in METRICS}))
    paired=[];factor=[]
    comparisons=[]
    for origin in ORIGINS:
        comparisons.extend((origin+'__matched_spatial',origin+'__'+c,origin=='own_pair') for c in ('search_control','mismatched_spatial'))
    comparisons.extend((a,'native',False) for a in TRAINED)
    comparisons.extend(('own_pair__'+a,'native_pair__'+a,False) for a in INTERACTIONS)
    metrics=('auc','low_iou_fraction','longest_failure_fraction','missed_native_recoveries','native_good_frame_loss')
    for a,b,primary in comparisons:
        for m in metrics:
            delta=[index[a,g][m]-index[b,g][m] for g in groups]
            family=dict(family_ci_low=None,family_ci_high=None)
            if primary and m=='auc':
                rng=np.random.default_rng(DESIGN['bootstrap_seed'])
                values=np.asarray(delta)
                boot=values[rng.integers(len(values),size=(DESIGN['bootstrap_samples'],len(values)))].mean(1)
                family=dict(family_ci_low=float(np.quantile(boot,.0125)),family_ci_high=float(np.quantile(boot,.9875)))
            paired.append(dict(contrast=a+'_minus_'+b,metric=m,primary=primary and m=='auc',
                favorable_direction='positive' if m=='auc' else 'negative',
                **source.common.protocol.contrast(delta),**family))
    for m in metrics:
        values=[float(np.mean([index['own_pair__'+a,g][m]-index['native_pair__'+a,g][m] for a in INTERACTIONS])) for g in groups]
        factor.append(dict(contrast='state_source_average_over_interactions',metric=m,
            favorable_direction='positive' if m=='auc' else 'negative',**source.common.protocol.contrast(values)))
    for control in ('search_control','mismatched_spatial'):
        for m in metrics:
            values=[float(np.mean([index[o+'__matched_spatial',g][m]-index[o+'__'+control,g][m] for o in ORIGINS])) for g in groups]
            factor.append(dict(contrast='matched_minus_'+control+'_average_over_origins',metric=m,
                favorable_direction='positive' if m=='auc' else 'negative',**source.common.protocol.contrast(values)))
            values=[(index['own_pair__matched_spatial',g][m]-index['own_pair__'+control,g][m])
                    -(index['native_pair__matched_spatial',g][m]-index['native_pair__'+control,g][m]) for g in groups]
            factor.append(dict(contrast='state_source_by_matched_minus_'+control,metric=m,
                favorable_direction='positive' if m=='auc' else 'negative',**source.common.protocol.contrast(values)))
    return summary,paired,factor
