"""Frozen-weight reference interventions, without replacing SGLA's template."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np
import stage208_state_interaction as prior

SCHEMA='stage209_frozen_weight_reference_intervention_v1'
SOURCE208='0058f9409c0437c4c8401e15f204fcdb41c2d1aa694dde56962486647bb8eadc'
BUNDLE208='1b5e2589d0e554792a07ab29d5f160087aa95d3c1c0c7aaa9a8c1430196c10bd'
MODES=('matched','search','wrong','zero')
ARMS=('native',*MODES)
DESIGN=dict(prior.common.DESIGN)
TABLES=('model_scope','reference_audit','parent_smoke','parallel_smoke','evaluation_execution',
        'source_replay','policy_summary','paired','seed_video_summary','routing')


def source208(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE208:raise ValueError('Need exact reviewed Stage-208 result bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage208_decision.json']),json.loads(parts['manifest.json'])
    if len(parts)!=len(ms) or set(parts)!={'manifest.json','stage208_decision.json',*d['table_sha256']}:raise ValueError('Changed source inventory')
    for v in (d,p):
        if v!=prior.old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if d['fingerprint']!=SOURCE208 or d['plan_fingerprint']!=p['fingerprint'] or d['training_runs']!=24 or d['full_evaluation_trajectories']!=494 or d['confirmation_groups_opened']!=0:
        raise ValueError('Unexpected source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table '+name)
        tables[name.removeprefix('stage208_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    if set(tables)!=set(prior.common.TABLES):raise ValueError('Incomplete source tables')
    return d,p,tables


def view(torch,rt,state,seed,mode):
    if mode not in MODES:raise ValueError('Unknown reference intervention')
    arm=dict(matched='matched_spatial',search='search_control',wrong='mismatched_spatial',zero='matched_spatial')[mode]
    base=prior.spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,seed,arm).to(next(rt.net.parameters()).device).eval()
    base.load_state_dict(state,strict=True);base.requires_grad_(False)
    class FrozenView:
        def state_dict(self):return base.state_dict()
        def adapt(self,tokens,reference=None):return tokens if mode=='zero' else base.adapt(tokens,reference)
    return FrozenView()


def load_models(root,decision,plan,tables,native,verified):
    rt=native.rt;models={};refs={};scopes=[];reference_rows=[]
    bound=prior.common.source.source_file
    for fold in plan['folds']:
        for seed in DESIGN['seeds']:
            rel=f"conditions/own_pair/training/f{fold['fold']}__s{seed}__matched_spatial.pt"
            path=bound(root,decision,rel,verified)
            receipt=prior.old.read_sealed(bound(root,decision,path.with_suffix('.receipt.json').relative_to(root).as_posix(),verified))
            meta=receipt['metadata'];payload=prior.old.load_model(rt.torch,path,meta)
            if (meta['plan_fingerprint']!=prior.training_plan(plan,'own_pair')['fingerprint'] or meta['fold']!=fold['fold']
                    or meta['seed']!=seed or meta['arm']!='matched_spatial' or meta['fit']!=fold['fit'] or meta['held']!=fold['held'] or meta['updates']!=256):
                raise ValueError('Wrong frozen own-pair matched model')
            row=next(r for r in tables['scope_summary'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed and r['training_condition']=='own_pair' and r['arm']=='matched_spatial')
            prior.previous.inputs.compare_scope(payload['scope'],{k:v for k,v in row.items() if k!='training_condition'})
            digests=[]
            for mode in MODES:
                m=view(rt.torch,rt,payload['state'],seed,mode);sha=prior.old.model_digest(m);digests.append(sha)
                models[fold['fold'],seed,mode]=m
            if set(digests)!={payload['scope']['state_sha256']}:raise ValueError('Reference mode changed frozen weights')
            scopes.append(dict(fold=fold['fold'],seed=seed,source_arm='own_pair__matched_spatial',state_sha256=digests[0],
                               source_file_sha256=verified[rel],new_training_updates=0,all_modes_same_weights=True))
            r=f"references/f{fold['fold']}__s{seed}.npz"
            value=prior.common.bound_arrays(root,decision,r,dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,fit=fold['fit']),verified)
            refs[fold['fold'],seed]={str(g):a for g,a in zip(value['groups'],value['reference'])}
            if set(refs[fold['fold'],seed])!={e['group'] for e in plan['entries']}:raise ValueError('Incomplete wrong references')
            rr=[r for r in tables['reference_audit'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed]
            for r in rr:
                if r['donor'] not in fold['fit'] or r['donor']==r['group']:raise ValueError('Held or self reference')
            reference_rows.extend(rr)
    return models,refs,scopes,reference_rows


def summarize(rows):
    protocol=prior.common.source.common.protocol;groups=sorted({r['group'] for r in rows});index={};summary=[];pairs=[]
    for mode in ARMS:
        for g in groups:
            rr=[r for r in rows if r['arm']==mode and r['group']==g]
            if sorted(r['seed'] for r in rr)!=([0] if mode=='native' else DESIGN['seeds']):raise ValueError('Incomplete OOF')
            index[mode,g]={k:float(np.mean([r[k] for r in rr])) for k in protocol.METRICS}
            index[mode,g]['introduced']=any(r['introduced_failure'] for r in rr)
        summary.append(dict(arm=mode,videos=len(groups),introduced_failure_videos=sum(index[mode,g]['introduced'] for g in groups),
                            **{k:float(np.mean([index[mode,g][k] for g in groups])) for k in protocol.METRICS}))
    for a,b in (('matched','search'),('matched','wrong'),('matched','zero'),('search','native'),('wrong','native'),('zero','native')):
        for metric in ('auc','low_iou_fraction','longest_failure_fraction','missed_native_recoveries','native_good_frame_loss'):
            delta=np.asarray([index[a,g][metric]-index[b,g][metric] for g in groups]);primary=a=='matched' and b in ('search','wrong') and metric=='auc'
            family=dict(family_ci_low=None,family_ci_high=None)
            if primary:
                rng=np.random.default_rng(DESIGN['bootstrap_seed'])
                boot=delta[rng.integers(len(delta),size=(DESIGN['bootstrap_samples'],len(delta)))].mean(1)
                family=dict(family_ci_low=float(np.quantile(boot,.0125)),family_ci_high=float(np.quantile(boot,.9875)))
            pairs.append(dict(contrast=a+'_minus_'+b,metric=metric,primary=primary,favorable_direction='positive' if metric=='auc' else 'negative',
                              **protocol.contrast(delta),**family))
    return summary,pairs
