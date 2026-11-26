"""Fit-source leave-one-out calibration of frozen Stage-204 routing residuals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO/'tools', REPO/'tracking'):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))

import numpy as np
import stage204_target_relation as old
import stage205_shrinkage_common as common
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_csv

source, metrics, protocol = old.source, old.metrics, old.protocol
SCHEMA = 'stage205_fit_source_crossfit_residual_shrinkage_v1'
SOURCE204 = '3cc164919abc8af737a21520e9e53b9775c5bf055ddfb6d0eda1d43a0ee36935'
PLAN204 = '603de9cf615c4d318c1bc54de29de745c76827a7319d5593b4d5f1c79828ff5c'
BUNDLE204 = '3d1a89cad966001838cc8145917bd12d08b11030737081cf5e159a4b54a48811'
DEFAULT_BUNDLE = REPO/'docs/experiments/06_sgla_uav_tracking/results/stage204_results_20261126T011205Z_1583527.txt'
PROTOCOL_FILE = REPO/'docs/experiments/06_sgla_uav_tracking/STAGE205_CROSSFIT_RESIDUAL_SHRINKAGE.md'
TABLES = ('source_bridge','inner_scope','inner_statistics','inner_schedule','inner_support','descriptor_audit',
          'training_history','inner_predictions','calibration','frame_predictions','policy_video','policy_summary',
          'error_video','error_summary','paired','seed_summary')


def load204(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE204: raise ValueError('Need exact reviewed Stage-204 bundle')
    matches = list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n', raw))
    parts = {m[1].decode():raw[m.end():matches[i+1].start() if i+1 < len(matches) else len(raw)] for i,m in enumerate(matches)}
    if len(parts) != len(matches): raise ValueError('Duplicate bundle section')
    d, p = json.loads(parts['stage204_decision.json']), json.loads(parts['manifest.json'])
    for value in (d,p):
        if value['fingerprint'] != fingerprint({k:v for k,v in value.items() if k != 'fingerprint'}):
            raise ValueError('Changed Stage-204 fingerprint')
    if (d['fingerprint'] != SOURCE204 or p['fingerprint'] != PLAN204 or d['plan_fingerprint'] != PLAN204
            or d['training_runs'] != 12 or d['predicted_states'] != 21888 or d['source_groups'] != 38
            or d['status'] != 'REVIEW_TARGET_SEARCH_RELATION'
            or any(d[k] != 0 for k in ('new_image_reads','new_encoder_calls','new_full_trajectories','confirmation_groups_opened'))
            or set(d['table_sha256']) != {f'stage204_{n}.csv' for n in old.TABLES}
            or set(parts) != {'stage204_decision.json','manifest.json',*d['table_sha256']}):
        raise ValueError('Unexpected Stage-204 identity, scope or inventory')
    tables = {}
    for name, sha in d['table_sha256'].items():
        b = parts[name]
        data = next((v for v in (b,b[:-1] if b.endswith(b'\n') else b,b[:-2] if b.endswith(b'\r\n') else b)
                     if hashlib.sha256(v).hexdigest() == sha), None)
        if data is None: raise ValueError('Changed source table '+name)
        tables[name.removeprefix('stage204_').removesuffix('.csv')] = list(csv.DictReader(io.StringIO(data.decode())))
    for name, sha in p['code_sha256'].items():
        if file_hash(REPO/'tools'/name) != sha: raise ValueError('Frozen code changed: '+name)
    if file_hash(old.PROTOCOL_FILE) != p['protocol_sha256']: raise ValueError('Frozen Stage-204 protocol changed')
    return d,p,tables


def load_sources(root194, bundle):
    d,p,tables = load204(bundle)
    p194,d194,reader,_ = old.inputs.load_source(root194)
    if d['source_stage194'] != d194['fingerprint'] or p['folds'] != p194['folds']:
        raise ValueError('Feature source or split differs from Stage-204')
    return p194,d194,reader,d,p,tables


def register(root, path, artifacts):
    path = Path(path)
    relative = path.relative_to(root).as_posix()
    artifacts[relative] = file_hash(path)


def failure_guard(root):
    if list((root/'inner').glob('*/drop_*/diagnostics/nonfinite_failure.json')):
        raise ValueError('Recorded nonfinite inner training failure; review before using a new path')


def outer_records(tables, folds):
    """Restore the exact float32 Stage-204 model outputs, not rounded new scores."""
    result = {}
    for row in tables['frame_predictions']:
        f = next(v for v in folds if v['fold'] == int(row['fold']))
        if row['group'] not in f['held']: continue
        key = int(row['fold']),int(row['seed']),row['arm'],row['group']
        result.setdefault(key,[]).append(row)
    expected = {(f['fold'],s,a,g) for f in folds for s in protocol.DESIGN['seeds'] for a in common.ARMS for g in f['held']}
    if set(result) != expected: raise ValueError('Outer held prediction inventory changed')
    for key, rows in result.items():
        frames = [int(r['frame']) for r in rows]
        if len(frames) != len(set(frames)) or any(b <= a for a,b in zip(frames,frames[1:])):
            raise ValueError('Repeated or unordered source states')
        p = np.asarray([[float(r[f'prediction_L{k}']) for k in range(7,13)] for r in rows],np.float32)
        y = np.asarray([[float(r[f'label_L{k}']) for k in range(7,13)] for r in rows],float)
        metrics.validate_scores(y,p)
        np.testing.assert_array_equal(metrics.choose(p)+7,[int(r['selected_layer']) for r in rows])
        result[key] = dict(prediction=p,labels=y,frame_id=np.asarray(frames),sequence=rows[0]['sequence'])
    return result


def load_bank(torch, p194, reader, tables, progress):
    labels = {}
    for row in tables['frame_predictions']:
        key = row['group'],int(row['frame'])
        y = np.asarray([float(row[f'label_L{k}']) for k in range(7,13)])
        if key in labels: np.testing.assert_array_equal(labels[key],y)
        labels[key] = y
    bank, bridge = {},[]
    lens = int(p194['runtime']['tracker']['lens_z'])
    shape = None
    for spec in p194['entries']:
        g,seq = spec['group'],spec['sequence']
        features = reader.arrays(f'features/{seq}.npz',dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix'))
        y = reader.arrays(f'reference/{g}.npz',dict(plan_fingerprint=p194['fingerprint'],kind='fixed_layer_fit_labels',group=g))['iou']
        np.testing.assert_array_equal(features['frame_id'],spec['sample_frames'])
        np.testing.assert_array_equal(y,np.stack([labels[g,int(f)] for f in features['frame_id']]))
        tokens = np.asarray(features['tokens'],np.float32)
        metrics.validate_scores(y,y)
        if tokens.ndim != 3 or len(tokens) != len(y) or not np.isfinite(tokens).all() or (shape is not None and tokens.shape[1:] != shape):
            raise ValueError('Changed token layout')
        shape = tokens.shape[1:]; progress.advance()
        pooled = []
        with torch.no_grad():
            for token in tokens:
                pooled.append(old.architecture.pooling.pool(torch,torch.as_tensor(token[None]),lens).numpy()[0].copy())
                progress.advance()
        bank[g] = dict(tokens=tokens,pooled=np.asarray(pooled),labels=np.asarray(y,float),frame_id=features['frame_id'],sequence=seq)
        bridge.append(dict(group=g,sequence=seq,states=len(y),maximum_label_error=0.))
    if len(bank) != 38 or sum(len(v['labels']) for v in bank.values()) != len(labels): raise ValueError('Incomplete feature inventory')
    return bank,bridge,lens


def run_inner(torch, root, plan, bank, lens, progress, artifacts, rows):
    oof = {(f['fold'],s,a):{} for f in plan['folds'] for s in plan['seeds'] for a in common.ARMS}
    for outer in plan['folds']:
        # Physically restrict the provider: outer held tokens and labels cannot reach inner preparation.
        inner_bank = {g:bank[g] for g in outer['fit']}
        for dropped in outer['fit']:
            fold = common.inner_fold(outer,dropped)
            inner_root = root/'inner'/f"fold{outer['fold']}"/('drop_'+dropped)
            sizes = {g:len(v['frame_id']) for g,v in inner_bank.items()}
            for arm in common.ARMS:
                prepared,stats,mapping = old.relation.prepare(inner_bank,fold,arm)
                meta = dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],inner_held=dropped,arm=arm,fit=fold['fit'])
                statpath = inner_root/'statistics'/(arm+'.npz')
                saved,_ = old.cache(statpath,meta,lambda:stats)
                for name,value in stats.items(): np.testing.assert_array_equal(value,saved[name])
                source.record(root,statpath,artifacts); progress.advance()
                rows['inner_statistics'].append(dict(fold=fold['fold'],inner_held=dropped,arm=arm,fit_groups='|'.join(fold['fit']),
                    fit_states=sum(sizes[g] for g in fold['fit']),statistics_sha256=file_hash(statpath),
                    **{f'prior_L{k}':float(stats['prior'][k-7]) for k in range(7,13)}))
                for g in inner_bank:
                    if mapping[g] not in fold['fit'] or mapping[g] == g: raise ValueError('Invalid inner donor')
                    rows['descriptor_audit'].append(dict(fold=fold['fold'],inner_held=dropped,arm=arm,group=g,donor=mapping[g],
                        donor_in_inner_fit=True,donor_differs_source=True,reference_sha256=hashlib.sha256(prepared[g]['reference'].tobytes()).hexdigest(),
                        scale_min=float(stats['scale_pooled'].min()),scale_max=float(stats['scale_pooled'].max())))
                for seed in plan['seeds']:
                    draws = old.common.pair_draws(fold,seed,sizes)
                    if arm == common.ARMS[0]:
                        counts = {g:np.zeros(sizes[g],int) for g in fold['fit']}
                        pairs = dict.fromkeys(fold['fit'],0); self_pairs = pairs.copy()
                        for step,batch in enumerate(draws,1):
                            for slot,(g,a,b) in enumerate(batch):
                                counts[g][a]+=1;counts[g][b]+=1;pairs[g]+=1;self_pairs[g]+=int(a==b)
                                rows['inner_schedule'].append(dict(fold=fold['fold'],inner_held=dropped,seed=seed,step=step,pair_slot=slot,
                                    group=g,index_a=a,index_b=b,frame_a=int(bank[g]['frame_id'][a]),frame_b=int(bank[g]['frame_id'][b]),self_pair=a==b))
                        for g in fold['fit']:
                            rows['inner_support'].append(dict(fold=fold['fold'],inner_held=dropped,seed=seed,group=g,pairs=pairs[g],
                                self_pairs=self_pairs[g],state_references=int(counts[g].sum()),unique_states=int(np.count_nonzero(counts[g]))))
                    # Frozen Stage-204 training/checkpoint engine. Its receipt schema is retained;
                    # the metadata binds the new Stage-205 plan and the strict 18/1 inner split.
                    model,scope,history = old.fit(torch,inner_root,plan,prepared,stats,fold,seed,arm,draws,lens,progress,{})
                    path = inner_root/'training'/f"fold{fold['fold']}"/f'seed{seed}'/(arm+'.pt')
                    source.record(root,path,artifacts)
                    digest = source.model_digest(model)
                    predpath = inner_root/'predictions'/f'{arm}__{seed}.npz'
                    predmeta = dict(meta,seed=seed,model_sha256=digest,statistics_sha256=file_hash(statpath))
                    data,reused = old.cache(predpath,predmeta,lambda:old.predict(torch,model,prepared[dropped],progress))
                    if reused: source.reused_work(progress,len(bank[dropped]['labels']))
                    if source.model_digest(model) != digest: raise ValueError('Inner evaluation mutated model')
                    source.record(root,predpath,artifacts)
                    metrics.validate_scores(bank[dropped]['labels'],data['prediction'])
                    np.testing.assert_array_equal(data['choice'],metrics.choose(data['prediction']))
                    if float(data['maximum_error']) > 1e-7: raise ValueError('Inner pooling interface changed')
                    rows['inner_scope'].append(dict(inner_held=dropped,**scope,prepooled_interface_maximum_error=float(data['maximum_error'])))
                    rows['training_history'].extend(dict(inner_held=dropped,**r) for r in history)
                    # Use the actual float32 model buffer prior in the sample-out risk objective.
                    prior = stats['prior'].astype(np.float32).astype(float)
                    oof[fold['fold'],seed,arm][dropped] = dict(prediction=data['prediction'],labels=bank[dropped]['labels'],prior=prior)
                    for i,frame in enumerate(bank[dropped]['frame_id']):
                        rows['inner_predictions'].append(dict(fold=fold['fold'],seed=seed,arm=arm,group=dropped,frame=int(frame),
                            **{f'prediction_L{k}':float(data['prediction'][i,k-7]) for k in range(7,13)},
                            **{f'prior_L{k}':float(prior[k-7]) for k in range(7,13)},
                            **{f'label_L{k}':float(bank[dropped]['labels'][i,k-7]) for k in range(7,13)}))
    # Match initialization, draws and effective budgets across inputs within every inner context.
    for f in plan['folds']:
        for g in f['fit']:
            for s in plan['seeds']:
                cell = [r for r in rows['inner_scope'] if r['fold']==f['fold'] and r['inner_held']==g and r['seed']==s]
                for field in ('initial_sha256','pair_draws_sha256','examples','pairs','updates','trainable_parameters','input_features'):
                    if len(cell)!=3 or len({r[field] for r in cell})!=1: raise ValueError('Unmatched inner '+field)
    return oof


def lock_calibration(root, plan, oof, rows, artifacts, progress):
    locked = {}
    for fold in plan['folds']:
        for seed in plan['seeds']:
            for arm in common.ARMS:
                key = fold['fold'],seed,arm
                value = common.estimate_alpha(oof[key],fold['fit'])
                row = dict(fold=fold['fold'],seed=seed,arm=arm,**value)
                path = root/'calibration'/f'f{key[0]}__s{seed}__{arm}.json'
                publish_identical(path,source.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],
                    fit=fold['fit'],held_excluded=fold['held'],oof_predictions_sha256=fingerprint([
                        r for r in rows['inner_predictions'] if r['fold']==key[0] and r['seed']==seed and r['arm']==arm]),**row)))
                register(root,path,artifacts);rows['calibration'].append(row);locked[key]=value['alpha'];progress.advance()
    return locked


def evaluate_outer(root, plan, tables, bank, locked, rows, progress):
    # Called only after ALL 12 fit-only coefficient files have been sealed on disk.
    if len(locked)!=12: raise ValueError('All coefficients must be locked before held evaluation')
    restored = outer_records(tables,plan['folds'])
    for fold in plan['folds']:
        for arm in common.ARMS:
            fitrow = next(r for r in tables['fit_summary'] if int(r['fold'])==fold['fold'] and r['arm']==arm)
            prior = np.asarray([float(fitrow[f'prior_L{k}']) for k in range(7,13)])
            np.testing.assert_array_equal(prior,old.architecture.fit_statistics(bank,fold['fit'])['prior'])
            for seed in plan['seeds']:
                alpha = locked[fold['fold'],seed,arm]
                for g in fold['held']:
                    video = restored[fold['fold'],seed,arm,g]
                    y,p = video['labels'],video['prediction']
                    np.testing.assert_array_equal(video['frame_id'],bank[g]['frame_id'])
                    np.testing.assert_array_equal(y,bank[g]['labels'])
                    scores = dict(shrunk=common.shrink(p,prior,alpha),raw=np.asarray(p,float),prior_only=common.shrink(p,prior,0.))
                    actions = {name:metrics.choose(value) for name,value in scores.items()}
                    actions.update(native_L9=np.full(len(y),2),oracle=metrics.choose(y))
                    base = dict(fold=fold['fold'],seed=seed,arm=arm,group=g,sequence=video['sequence'],pool='held')
                    for policy in common.POLICIES:
                        a = actions[policy]
                        rows['policy_video'].append(dict(base,policy=policy,states=len(y),
                            **old.predictor.diagnosis.correspondence(y,a),mean_selected_layer=float(np.mean(a+7)),
                            nonprior_fraction=float(np.mean(a!=actions['prior_only']))))
                    for name,value in scores.items():
                        h,z = common.centered(value-prior),common.centered(y-prior)
                        rows['error_video'].append(dict(base,predictor=name,interaction_mse=float(np.mean((h-z)**2)),**metrics.ranking(y,value)))
                    for i,frame in enumerate(video['frame_id']):
                        rows['frame_predictions'].append(dict(base,frame=int(frame),alpha=float(alpha),
                            raw_layer=int(actions['raw'][i])+7,shrunk_layer=int(actions['shrunk'][i])+7,prior_layer=int(actions['prior_only'][i])+7,
                            **{f'raw_score_L{k}':float(p[i,k-7]) for k in range(7,13)},
                            **{f'shrunk_score_L{k}':float(scores['shrunk'][i,k-7]) for k in range(7,13)},
                            **{f'prior_L{k}':float(prior[k-7]) for k in range(7,13)},
                            **{f'label_L{k}':float(y[i,k-7]) for k in range(7,13)}))
                    progress.advance()
    rows['policy_summary'] = common.summarize(rows['policy_video'],'policy',common.POLICIES,common.FIELDS)
    rows['error_summary'] = common.summarize(rows['error_video'],'predictor',common.PREDICTORS,('interaction_mse','gap_weighted_pair_accuracy'))
    rows['paired'] = common.paired(rows['policy_video'])
    for seed in plan['seeds']:
        rows['seed_summary'].extend(common.summarize(rows['policy_video'],'policy',common.POLICIES,common.FIELDS,seed))
    # Raw controls must reproduce Stage-204, source by source, without refitting outer models.
    for r in rows['policy_video']:
        if r['policy'] not in ('raw','prior_only'): continue
        name = 'trained' if r['policy']=='raw' else 'prior_only'
        original = next(v for v in tables['policy_video'] if int(v['fold'])==r['fold'] and int(v['seed'])==r['seed']
            and v['arm']==r['arm'] and v['group']==r['group'] and v['policy']==name and v['pool']=='held')
        for field in common.FIELDS[:-1]:
            if abs(float(original[field])-r[field])>1e-12: raise ValueError('Frozen outer control replay failed: '+field)


def verify_results(root):
    root = Path(root).resolve(); failure_guard(root)
    d,p = source.read_sealed(root/'stage205_decision.json'),source.read_sealed(root/'manifest.json')
    n = p['states_per_outer_model']
    if (d['schema']!=SCHEMA or p['schema']!=SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage204']!=SOURCE204
            or d['status']!='REVIEW_CROSSFIT_RESIDUAL_SHRINKAGE' or d['training_runs']!=228 or d['training_updates']!=58368
            or d['training_state_references']!=233472 or d['training_pair_uses']!=116736 or d['inner_schedule_rows']!=38912
            or d['inner_predicted_states']!=6*n or d['outer_held_states']!=6*n or d['coefficients_locked']!=12
            or any(d[k]!=0 for k in ('outer_model_refits','new_image_reads','new_encoder_calls','new_full_trajectories','confirmation_groups_opened'))
            or d['actual_transformer_calls_measured'] or set(d['table_sha256'])!={f'stage205_{name}.csv' for name in TABLES}):
        raise ValueError('Incomplete Stage-205 comparison')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha: raise ValueError('Changed Stage-205 artifact: '+name)
    for name,sha in p['code_sha256'].items():
        if file_hash(REPO/'tools'/name)!=sha: raise ValueError('Stage-205 dependency changed: '+name)
    if file_hash(PROTOCOL_FILE)!=p['protocol_sha256']: raise ValueError('Stage-205 protocol changed')
    print('Stage-205 verified; cache-only REVIEW, not full-video adoption',flush=True)


def execute(root, root194, bundle):
    p194,d194,reader,d204,p204,tables = load_sources(root194,bundle)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    n = sum(len(e['sample_frames']) for e in p194['entries'])
    plan = source.sealed(dict(schema=SCHEMA,source_stage194=d194['fingerprint'],source_stage204=d204['fingerprint'],source_plan204=p204['fingerprint'],
        source204_bundle_sha256=file_hash(bundle),folds=p194['folds'],seeds=protocol.DESIGN['seeds'],arms=list(common.ARMS),
        states_per_outer_model=n,training=protocol.DESIGN,device='cpu',torch=str(torch.__version__),numpy=str(np.__version__),python=sys.version,
        inner_design='outer-fit 19 source leave-one-out; 18 fit, 1 calibration; all statistics and donor pools recomputed inside each inner fold',
        alpha='source-equal E<h,z>/E||h||^2 clipped [0,1], zero denominator=0; h=P(score-inner prior), z=P(label-inner prior)',
        evaluation='reuse frozen Stage-204 outer held scores; 12 coefficients sealed before held policy evaluation; float64 shrinkage, alpha=1 exact original scores',
        cost='228 additional CPU fits, 256 updates each; no outer refit; calibration cost not budget matched to raw controls',
        primary='matched shrunk vs raw, prior, own independent expectation, concat shrunk and mismatched shrunk',
        limitations='historically exposed development caches; immediate IoU not AUC; layer ID not executed blocks; prior collapse is not dynamic gain',
        inherited_engine='Stage-204 frozen fit/cache engine and its receipt schema; Stage-205 plan fingerprint binds every inner receipt',
        protocol_sha256=file_hash(PROTOCOL_FILE),code_sha256={**p204['code_sha256'],Path(__file__).name:file_hash(__file__),
            Path(common.__file__).name:file_hash(common.__file__)}))
    publish_identical(root/'manifest.json',plan)
    progress = OverallProgress('Stage-205',1+38+n+114+228*256+6*n+12+228+1,unit='work')
    progress.report('Verifying frozen sources and pooling cached tokens'); progress.advance()
    bank,bridge,lens = load_bank(torch,p194,reader,tables,progress)
    rows = {name:[] for name in TABLES};rows['source_bridge']=bridge;artifacts={}
    progress.report('Inner leave-one-source-out training and predictions')
    oof = run_inner(torch,root,plan,bank,lens,progress,artifacts,rows)
    progress.report('Sealing fit-only residual trust coefficients')
    locked = lock_calibration(root,plan,oof,rows,artifacts,progress)
    progress.report('Evaluating frozen outer held scores')
    evaluate_outer(root,plan,tables,bank,locked,rows,progress)
    hashes = {}
    for name in TABLES:
        path = root/f'stage205_{name}.csv';write_csv(path,rows[name]);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1: raise ValueError('Incomplete overall progress accounting')
    decision = source.sealed(dict(schema=SCHEMA,status='REVIEW_CROSSFIT_RESIDUAL_SHRINKAGE',plan_fingerprint=plan['fingerprint'],
        source_stage194=d194['fingerprint'],source_stage204=d204['fingerprint'],source_groups=38,
        training_runs=len(rows['inner_scope']),training_updates=len(rows['training_history']),
        training_state_references=sum(r['examples'] for r in rows['inner_scope']),training_pair_uses=sum(r['pairs'] for r in rows['inner_scope']),
        inner_schedule_rows=len(rows['inner_schedule']),inner_predicted_states=len(rows['inner_predictions']),outer_held_states=len(rows['frame_predictions']),
        coefficients_locked=len(locked),coefficients_at_zero=sum(a==0 for a in locked.values()),coefficients_at_one=sum(a==1 for a in locked.values()),
        primary_contrasts=[r for r in rows['paired'] if r['primary']],outer_model_refits=0,new_image_reads=0,new_encoder_calls=0,
        new_full_trajectories=0,confirmation_groups_opened=0,actual_transformer_calls_measured=False,
        source_verified_artifacts=reader.verified,table_sha256=hashes,artifact_sha256=artifacts,
        next_step='Require nonconstant held gains versus prior/raw and matched controls; inspect alpha and action collapse; no automatic full-video adoption'))
    publish_identical(root/'stage205_decision.json',decision)
    progress.phase='Complete';progress.advance();progress.report();verify_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage194-root',type=Path)
    parser.add_argument('--stage204-bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-results',action='store_true')
    args = parser.parse_args();root = args.output_dir.expanduser().resolve()
    if args.check_results: verify_results(root);return
    if args.stage194_root is None: parser.error('Original Stage-194 feature directory required')
    r194,bundle = args.stage194_root.expanduser().resolve(),args.stage204_bundle.expanduser().resolve()
    protocol.safe_output(root,[r194,bundle.parent])
    with writer_lock(root):
        failure_guard(root)
        if (root/'stage205_decision.json').exists(): verify_results(root);return
        execute(root,r194,bundle)


if __name__=='__main__': main()
