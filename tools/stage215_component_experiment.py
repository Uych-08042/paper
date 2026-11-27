"""Frozen center/scale interventions with identical updater call opportunity."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import sys,traceback
from time import perf_counter
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage215_component_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
old=c.old;previous=c.source;runtime=previous.runtime
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage214_results_20261126T194237Z_2307464.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE215_FROZEN_CENTER_SCALE.md'


def bound_models(config,native):
    root=Path(config['state_root']);d,p,t=c.source214(Path(config['state_bundle']));verified={}
    if old.read_sealed(root/'manifest.json')!=p or old.read_sealed(root/'stage214_decision.json')!=d:raise ValueError('Physical Stage-214 changed')
    for f in p['folds']:
        for s in c.SEEDS:
            for a in c.FAMILIES:
                rel=f"training/f{f['fold']}__s{s}__{a}.pt"
                for n in (rel,rel.replace('.pt','.receipt.json')):runtime.common.source.source_file(root,d,n,verified)
    models=previous.load_updaters(root,p,native.rt.torch,next(native.rt.net.parameters()).device)
    for (fold,seed,family),model in models.items():
        r=next(r for r in t['scope_summary'] if int(r['fold'])==fold and int(r['seed'])==seed and r['arm']==family)
        if old.model_digest(model)!=r['state_sha256']:raise ValueError('Updater differs from reviewed scope')
    return models,p,d,t,verified


def jobs(plan,spec,adapters,updaters):
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    return fold,[(arm,seed,adapters[fold,seed,'matched'],c.Masked(updaters[fold,seed,arm.split('__')[0]],arm.split('__')[1])) for seed in c.SEEDS for arm in c.FRESH]


def meta(plan,seq,fold,seed,arm,adapter,updater,phase):return dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,adapter_sha256=old.model_digest(adapter),updater_sha256=old.model_digest(updater),phase=phase)


def evaluate_video(root,plan,native,adapters,updaters,spec,entry,dataset,progress,phase):
    files,gt,_,reader,_=entry
    if phase!='full':files=files[:1+c.protocol.DESIGN['smoke_updates']]
    seq=spec['sequence'];fold,work=jobs(plan,spec,adapters,updaters)
    for arm,seed,a,u in work:
        path=previous.path_for(root,phase,seq,seed,arm);metadata=meta(plan,seq,fold,seed,arm,a,u,phase)
        v,reused=old.cache(path,metadata,lambda:previous.rollout(native,a,arm.split('__')[0],u,files,gt[0],reader,dataset,progress,truth=gt))
        if reused:old.reused_work(progress,len(files)-1)
        c.audit(v,gt[0],len(files)-1,arm)
        if old.model_digest(a)!=metadata['adapter_sha256'] or old.model_digest(u)!=metadata['updater_sha256']:raise ValueError('Frozen model changed')
    native.assert_frozen()


def check_code(p):
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Frozen dependency changed '+n)
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Changed protocol')


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);runtime.guard(root);native=runtime.open_runtime(config,plan,f'stage215_{phase}_{worker}')
            entries,dataset=runtime.load_entries(config,plan);adapters,_,_,_,_=previous.previous.previous.load_bound(config,native);updaters,_,_,_,_=bound_models(config,native)
            progress=runtime.WorkerProgress(channel,worker)
            for seq in sequences:evaluate_video(root,plan,native,adapters,updaters,next(e for e in plan['entries'] if e['sequence']==seq),entries[seq],dataset,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def collect(root,plan,adapters,updaters,spec,entry,phase,artifacts):
    fold,work=jobs(plan,spec,adapters,updaters);result={};n=len(entry[0])-1 if phase=='full' else min(len(entry[0])-1,c.protocol.DESIGN['smoke_updates'])
    for arm,seed,a,u in work:
        path=previous.path_for(root,phase,spec['sequence'],seed,arm);v=previous.previous.required(path,meta(plan,spec['sequence'],fold,seed,arm,a,u,phase))
        c.audit(v,entry[1][0],n,arm);old.record(root,path,artifacts);result[arm,seed]=v
    return result


def source_trace(config,p,d,t,spec,arm,seed,verified):
    if arm not in ('native','nn','mm',*c.FAMILIES) or seed not in ([0] if arm=='native' else c.SEEDS):raise ValueError('Unknown source trace identity')
    fold=next(f['fold'] for f in p['folds'] if spec['group'] in f['held']);updater=None;adapter=None
    if arm!='native':adapter=next(r['state_sha256'] for r in t['frozen_scope'] if int(r['fold'])==fold and int(r['seed'])==seed)
    if arm in c.FAMILIES:updater=next(r['state_sha256'] for r in t['scope_summary'] if int(r['fold'])==fold and int(r['seed'])==seed and r['arm']==arm)
    metadata=dict(plan_fingerprint=p['fingerprint'],sequence=spec['sequence'],fold=fold,seed=seed,arm=arm,phase='full',adapter_sha256=adapter,updater_sha256=updater)
    return runtime.common.bound_arrays(Path(config['state_root']),d,f"evaluation/{spec['sequence']}__s{seed}__{arm}.npz",metadata,verified)


def experiment(root,plan,native,adapters,updaters,scope,entries,dataset,config,source):
    p,d,t=source;specs=previous.previous.previous.specs_for_smoke(plan);sf=sum(min(c.protocol.DESIGN['smoke_updates'],len(entries[e['sequence']][0])-1) for e in specs)
    frames=sum(len(entries[e['sequence']][0])-1 for e in plan['entries']);progress=OverallProgress('Stage-215',32*sf+16*frames+1,unit='frames')
    tables={k:[] for k in c.TABLES};tables['model_scope']=scope;artifacts={};verified={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Same weights and per-frame updater call; serial/two-worker smoke');start=perf_counter()
    for spec in specs:evaluate_video(root,plan,native,adapters,updaters,spec,entries[spec['sequence']],dataset,progress,'serial')
    serial=perf_counter()-start;parallel=runtime.run_workers(config,[e['sequence'] for e in specs],'parallel',2,progress,target=worker_main)
    for spec in specs:
        aa=collect(root,plan,adapters,updaters,spec,entries[spec['sequence']],'serial',artifacts);bb=collect(root,plan,adapters,updaters,spec,entries[spec['sequence']],'parallel',artifacts)
        for arm,seed in aa:
            previous.previous.previous.equal_quality(aa[arm,seed],bb[arm,seed]);tables['smoke'].append(dict(sequence=spec['sequence'],arm=arm,seed=seed,exact_state_output_calls=True,timing_excluded=True))
    progress.report('All 38 sources: closed-loop center x scale interventions')
    elapsed=runtime.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    for spec in plan['entries']:
        seq=spec['sequence'];entry=entries[seq];vv=collect(root,plan,adapters,updaters,spec,entry,'full',artifacts)
        baseline=source_trace(config,p,d,t,spec,'native',0,verified);previous.c.audit(baseline,entry[1][0],len(entry[0])-1,'native')
        mm={seed:source_trace(config,p,d,t,spec,'mm',seed,verified) for seed in c.SEEDS}
        for seed,v in mm.items():previous.c.audit(v,entry[1][0],len(entry[0])-1,'mm')
        vv.update({('native',0):baseline,**{('mm',s):v for s,v in mm.items()}})
        for family in c.FAMILIES:
            for seed in c.SEEDS:
                for mode,source_arm in (('none','nn'),('joint',family)):
                    a=vv[family+'__'+mode,seed];b=source_trace(config,p,d,t,spec,source_arm,seed,verified)
                    if set(a)!=set(b):raise ValueError('Endpoint schema changed')
                    for k in a:
                        if k not in ('pipeline_ms','updater_calls'):np.testing.assert_array_equal(a[k],b[k],err_msg='Endpoint differs '+k)
                    np.testing.assert_array_equal(a['updater_calls'],np.ones(len(a['boxes']),int))
                    tables['source_replay'].append(dict(sequence=seq,arm=family+'__'+mode,seed=seed,provenance='fresh_endpoint',source_arm=source_arm,exact_non_timing_arrays=True,expected_updater_calls=1))
        for (arm,seed),v in vv.items():
            fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held']);row=old.describe_trace(v,baseline,entry[1],seq,arm,seed,fold)
            refs=c.SEEDS if arm=='native' else [seed];other=[old.describe_trace(v,mm[s],entry[1],seq,arm,seed,fold) for s in refs]
            row.update(missed_matched_recoveries=float(np.mean([r['missed_native_recoveries'] for r in other])),matched_good_frame_loss=float(np.mean([r['native_good_frame_loss'] for r in other])),introduced_matched_failure=any(r['introduced_failure'] for r in other))
            tables['seed_video_summary'].append(row)
            fresh=arm in c.FRESH
            tables['calls'].append(dict(sequence=seq,arm=arm,seed=seed,provenance='new' if fresh else 'reused_stage214',frames=len(v['boxes']),block_calls=int(v['block_calls'].sum()),head_calls=int(v['head_calls'].sum()),selector_calls=int(v['selector_calls'].sum()),crop_calls=int(v['crop_calls'].sum()),updater_calls=0 if arm=='native' else int(v['updater_calls'].sum())))
            if not fresh:tables['source_replay'].append(dict(sequence=seq,arm=arm,seed=seed,provenance='bound_cache_reuse',source_arm=arm,exact_non_timing_arrays=True,expected_updater_calls=0))
            if arm!='native':
                output=old.tracking_iou(v['boxes'],entry[1][1:]);covered=v['fully_covered'];grid=v['dense_best_iou']
                tables['search_geometry'].append(dict(sequence=seq,arm=arm,seed=seed,fully_covered_fraction=float(covered.mean()),mean_gt_side_network_px=float(v['gt_side_network_px'].mean()),
                    covered_no_good_grid_fraction=float((covered&(grid<.5)).mean()),good_grid_missed_fraction=float(((grid>=.5)&(output<.5)).mean()),mean_parent_side=float(np.sqrt(v['parent'][:,2:].prod(1)).mean()),
                    changed_from_raw_state_frames=int((v['next_parent']!=v['reference_boxes']).any(1).sum())))
    tables['policy_summary'],tables['paired'],tables['factor_summary']=c.summarize(tables['seed_video_summary'])
    tables['execution']=[dict(phase='serial_smoke',workers=1,wall_seconds=serial),dict(phase='parallel_smoke',workers=2,wall_seconds=parallel),dict(phase='full',workers=plan['workers'],wall_seconds=elapsed)]
    publish_identical(root/'evaluation_source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_arrays=verified)));artifacts['evaluation_source_receipt.json']=file_hash(root/'evaluation_source_receipt.json')
    hashes={}
    for n,rows in tables.items():
        path=root/f'stage215_{n}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete work budget')
    publish_identical(root/'stage215_decision.json',old.sealed(dict(schema=c.SCHEMA,status='REVIEW_FROZEN_CENTER_SCALE',plan_fingerprint=plan['fingerprint'],source_stage214=c.SOURCE214,source_groups=38,new_training_updates=0,
        new_full_trajectories=608,reused_reference_trajectories=114,reported_quality_sequences=len(tables['seed_video_summary']),confirmation_groups_opened=0,primary_contrasts=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='Frozen models and physical bounds; all factorial cells execute one updater; old references explicitly reused; no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)
    for r in tables['policy_summary']:print(f"{r['arm']:18s} AUC={r['auc']:.4f} low-IoU={r['low_iou_fraction']:.2%}",flush=True)


def verify(root):
    root=Path(root).resolve();runtime.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage215_decision.json');check_code(p)
    if d['schema']!=c.SCHEMA or d['status']!='REVIEW_FROZEN_CENTER_SCALE' or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage214']!=c.SOURCE214 or (d['source_groups'],d['new_training_updates'],d['new_full_trajectories'],d['reused_reference_trajectories'],d['reported_quality_sequences'],d['confirmation_groups_opened'])!=(38,0,608,114,722,0) or set(d['table_sha256'])!={f'stage215_{k}.csv' for k in c.TABLES}:raise ValueError('Incomplete Stage-215')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=h:raise ValueError('Changed artifact '+n)
    print('Stage-215 verified; fixed weights, REVIEW, no adoption',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (208,214,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage214-bundle',type=Path,default=DEFAULT_BUNDLE);parser.add_argument('--stage208-bundle',type=Path,default=previous.previous.previous.DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');args=parser.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,214,*sources.SOURCES)):parser.error('Need frozen Stage-214 tree, Stage-208 adapters, checkpoint and source roots')
    paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES};c.protocol.safe_output(root,[args.checkpoint,args.stage208_root,args.stage214_root,args.stage214_bundle.parent,*paths.values()])
    with writer_lock(root):
        runtime.guard(root)
        if (root/'stage215_decision.json').exists():verify(root);return
        d,p,t=c.source214(args.stage214_bundle);previous.verify(args.stage214_root)
        if old.read_sealed(args.stage214_root/'manifest.json')!=p or old.read_sealed(args.stage214_root/'stage214_decision.json')!=d:raise ValueError('Physical source differs')
        import torch
        torch.set_num_threads(1);identity=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p['runtime']),torch,args.gpu);sources.old.base.s109.check_runtime(p['runtime'],identity,p['runtime'])
        code=dict(p['code_sha256'])
        for f in (__file__,c.__file__,REPO/'tools/run_stage215_components.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        plan=old.sealed(dict(schema=c.SCHEMA,source_stage214=d['fingerprint'],entries=p['entries'],folds=p['folds'],runtime=identity,search_factor=p['search_factor'],workers=args.workers,masks={k:list(v) for k,v in c.MASKS.items()},code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_root=str(args.stage208_root.resolve()),bundle=str(args.stage208_bundle.resolve()),state_root=str(args.stage214_root.resolve()),state_bundle=str(args.stage214_bundle.resolve()),source_paths={str(k):str(v) for k,v in paths.items()})
        entries,dataset=runtime.load_entries(config,plan);c.protocol.safe_output(root,[dataset]);native=runtime.open_runtime(config,plan,'stage215')
        try:
            adapters,_,adapter_scope,_,verified=previous.previous.previous.load_bound(config,native);updaters,sp,sd,st,bound=bound_models(config,native)
            if [{k:str(v) for k,v in r.items()} for r in adapter_scope]!=t['frozen_scope']:raise ValueError('Frozen adapter mismatch')
            scope=[dict(fold=int(r['fold']),seed=int(r['seed']),family=r['arm'],state_sha256=r['state_sha256'],all_masks_same_weights=True,new_training_updates=0) for r in t['scope_summary']]
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],adapters=verified,updaters=bound,source_bundle_sha256=file_hash(args.stage214_bundle))))
            experiment(root,plan,native,adapters,updaters,scope,entries,dataset,config,(sp,sd,st))
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
