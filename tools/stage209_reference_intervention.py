"""No-training, frozen-weight adapter reference intervention with full OOF tracks."""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage209_reference_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
prior,old=c.prior,c.prior.old
SCHEMA=c.SCHEMA
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage208_results_20261126T095430Z_1911332.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE209_FROZEN_REFERENCE_INTERVENTION.md'


def check_code(plan):
    for name,sha in plan['code_sha256'].items():
        if file_hash(REPO/name)!=sha:raise ValueError('Changed dependency '+name)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def jobs(plan,spec,models):
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    return fold,[('native',0,None)]+[(a,s,models[fold,s,a]) for s in c.DESIGN['seeds'] for a in c.MODES]


def cache_path(root,phase,seq,seed,mode):
    return (root/'evaluation' if phase=='full' else root/('smoke_'+phase)/'evaluation')/f'{seq}__s{seed}__{mode}.npz'


def metadata(plan,seq,fold,seed,mode,model,phase):
    return dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=mode,
                model_sha256=old.model_digest(model) if model is not None else None,phase=phase)


def required(path,meta):
    return old.cache(path,meta,lambda:(_ for _ in ()).throw(ValueError('Missing committed trajectory')))[0]


def evaluate_video(root,plan,native,models,refs,spec,entry,dataset,progress,phase):
    seq=spec['sequence'];files,truth,_,reader,_=entry
    if phase!='full':files=files[:1+c.DESIGN['smoke_updates']]
    fold,work=jobs(plan,spec,models)
    for mode,seed,model in work:
        meta=metadata(plan,seq,fold,seed,mode,model,phase);path=cache_path(root,phase,seq,seed,mode)
        ref=None if model is None else native.rt.torch.as_tensor(refs[fold,seed][spec['group']][None],device=next(native.rt.net.parameters()).device)
        value,reused=old.cache(path,meta,lambda:prior.previous.trace(native,model,ref,files,truth[0],reader,dataset,progress))
        if reused:old.reused_work(progress,len(files)-1)
        old.audit_trace(value,truth[0],len(files)-1)
        if model is not None and old.model_digest(model)!=meta['model_sha256']:raise ValueError('Frozen model changed')
    native.assert_frozen()


def load_bound(config,native):
    root=Path(config['source_root']);d,p,t=c.source208(Path(config['bundle']));verified={}
    if old.read_sealed(root/'manifest.json')!=p or old.read_sealed(root/'stage208_decision.json')!=d:raise ValueError('Changed source run')
    models,refs,scope,rr=c.load_models(root,d,p,t,native,verified)
    return models,refs,scope,rr,verified


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);prior.guard(root)
            native=prior.open_runtime(config,plan,f'stage209_{phase}_{worker}');entries,dataset=prior.load_entries(config,plan)
            models,refs,_,_,_=load_bound(config,native);progress=prior.WorkerProgress(channel,worker)
            for seq in sequences:
                spec=next(e for e in plan['entries'] if e['sequence']==seq)
                evaluate_video(root,plan,native,models,refs,spec,entries[seq],dataset,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def specs_for_smoke(plan):
    return [next(e for e in plan['entries'] if e['group'] in f['held']) for f in plan['folds']]


def equal_quality(a,b):
    if set(a)!=set(b):raise ValueError('Trace schema changed')
    for k in a:
        if k!='pipeline_ms':np.testing.assert_array_equal(a[k],b[k],err_msg='Trace mismatch '+k)


def compare_smoke(root,plan,models,specs,entries,artifacts):
    rows=[]
    for spec in specs:
        seq=spec['sequence'];fold,work=jobs(plan,spec,models)
        for mode,seed,model in work:
            traces=[]
            for phase in ('serial','parallel'):
                path=cache_path(root,phase,seq,seed,mode)
                value=required(path,metadata(plan,seq,fold,seed,mode,model,phase));old.record(root,path,artifacts);traces.append(value)
            equal_quality(*traces)
            rows.append(dict(sequence=seq,seed=seed,arm=mode,tracked_frames=len(traces[0]['boxes']),exact_quality_and_calls=True,timing_excluded=True))
    return rows


def parent_smoke(root,plan,native,models,refs,specs,entries,dataset,progress):
    """Same parent, pixels and raw route; only adapter KV source may differ."""
    rt=native.rt;rows=[]
    for spec in specs:
        seq=spec['sequence'];files,gt,_,reader,_=entries[seq];fold,work=jobs(plan,spec,models)
        base=required(cache_path(root,'serial',seq,0,'native'),metadata(plan,seq,fold,0,'native',None,'serial'))
        for i,frame in enumerate(base['frame_id']):
            first=None
            for mode,seed,model in work:
                native.initialize(reader(dataset,files[0]),gt[0]);rt.tracker.frame_id=int(frame)-1;raw=[];scores=[]
                ref=None if model is None else rt.torch.as_tensor(refs[fold,seed][spec['group']][None],device=next(rt.net.parameters()).device)
                with ExitStack() as stack:
                    stack.callback(rt.backbone.blocks[5].register_forward_hook(lambda m,a,o:raw.append(o.detach().cpu().numpy().copy())).remove)
                    stack.callback(rt.backbone.MLP.register_forward_hook(lambda m,a,o:scores.append(o.detach().cpu().numpy().copy())).remove)
                    if model is not None:stack.enter_context(prior.spatial.Installed(native,model,ref))
                    out=native.step(reader(dataset,files[int(frame)]),base['parent'][i])
                if len(raw)!=1 or len(scores)!=1:raise ValueError('Unexpected public prefix/router calls')
                if mode=='native':first=(raw[0],scores[0],out)
                else:
                    np.testing.assert_array_equal(raw[0],first[0]);np.testing.assert_array_equal(scores[0],first[1])
                    if out['layer']!=first[2]['layer']:raise ValueError('Reference changed native routing')
                    if mode=='zero':np.testing.assert_array_equal(out['box'],first[2]['box'])
                rows.append(dict(sequence=seq,frame=int(frame),arm=mode,seed=seed,raw_prefix_equal=True,router_scores_equal=True,
                    layer=int(out['layer']),zero_exact_native=mode=='zero',iou=float(old.tracking_iou(out['box'],gt[int(frame)]))))
                progress.advance()
    native.assert_frozen();return rows


def aggregate(root,plan,models,entries,config,source_plan,source_decision,artifacts,tables):
    verified={};source_root=Path(config['source_root'])
    for spec in plan['entries']:
        seq=spec['sequence'];files,gt,_,_,_=entries[seq];fold,work=jobs(plan,spec,models);baseline=None
        for mode,seed,model in work:
            path=cache_path(root,'full',seq,seed,mode);value=required(path,metadata(plan,seq,fold,seed,mode,model,'full'))
            old.audit_trace(value,gt[0],len(files)-1);old.record(root,path,artifacts)
            if mode=='native':baseline=value
            if mode=='zero':equal_quality(value,baseline)
            if mode in ('native','matched'):
                arm='native' if mode=='native' else 'own_pair__matched_spatial'
                rel=f'evaluation/{seq}__s{seed}__{arm}.npz'
                meta=dict(plan_fingerprint=source_plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,
                          model_sha256=None if mode=='native' else old.model_digest(model),phase='full')
                previous=prior.common.bound_arrays(source_root,source_decision,rel,meta,verified)
                exact=all(np.array_equal(value[k],previous[k]) for k in value if k!='pipeline_ms')
                if plan['same_source_runtime'] and not exact:raise ValueError('Same-runtime native/matched must reproduce Stage-208 trace')
                tables['source_replay'].append(dict(sequence=seq,arm=mode,seed=seed,same_source_runtime=plan['same_source_runtime'],
                    exact_quality_and_calls=exact,box_max_error=float(np.abs(value['boxes']-previous['boxes']).max())))
            tables['seed_video_summary'].append(old.describe_trace(value,baseline,gt,seq,mode,seed,fold))
            for layer in range(7,13):tables['routing'].append(dict(sequence=seq,arm=mode,seed=seed,fold=fold,layer=layer,frames=int((value['layer']==layer).sum())))
    return verified


def experiment(root,plan,native,models,refs,scope,rr,entries,dataset,config,source_plan,source_decision):
    specs=specs_for_smoke(plan);sf=sum(min(c.DESIGN['smoke_updates'],len(entries[e['sequence']][0])-1) for e in specs)
    frames=sum(len(v[0])-1 for v in entries.values());progress=OverallProgress('Stage-209',27*sf+9*frames+1,unit='work')
    tables={k:[] for k in c.TABLES};tables['model_scope']=scope;tables['reference_audit']=rr;artifacts={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Frozen weights: serial smoke (both held folds)');start=perf_counter()
    for spec in specs:evaluate_video(root,plan,native,models,refs,spec,entries[spec['sequence']],dataset,progress,'serial')
    serial=perf_counter()-start
    progress.report('Same-parent unadapted-prefix and native-route identity')
    tables['parent_smoke']=parent_smoke(root,plan,native,models,refs,specs,entries,dataset,progress)
    progress.report('Independent two-worker smoke')
    parallel=prior.run_workers(config,[e['sequence'] for e in specs],'parallel',2,progress,target=worker_main)
    tables['parallel_smoke']=compare_smoke(root,plan,models,specs,entries,artifacts)
    progress.report('Full OOF frozen-weight reference interventions')
    elapsed=prior.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    tables['evaluation_execution']=[dict(phase='serial_smoke',workers=1,wall_seconds=serial,includes_worker_startup=False),
        dict(phase='parallel_smoke',workers=2,wall_seconds=parallel,includes_worker_startup=True),
        dict(phase='full_evaluation',workers=plan['workers'],wall_seconds=elapsed,includes_worker_startup=True)]
    verified=aggregate(root,plan,models,entries,config,source_plan,source_decision,artifacts,tables)
    publish_identical(root/'replay_source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],artifacts=verified)))
    artifacts['replay_source_receipt.json']=file_hash(root/'replay_source_receipt.json')
    tables['policy_summary'],tables['paired']=c.summarize(tables['seed_video_summary']);hashes={}
    for name,rows in tables.items():
        path=root/f'stage209_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete progress budget')
    publish_identical(root/'stage209_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_FROZEN_WEIGHT_REFERENCE_INTERVENTION',
        plan_fingerprint=plan['fingerprint'],source_stage208=c.SOURCE208,new_training_updates=0,source_models=4,
        full_evaluation_trajectories=len(tables['seed_video_summary']),confirmation_groups_opened=0,
        primary_contrasts=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='Dependence of an added frozen adapter on reference inputs; not removal of original SGLA template information; no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();prior.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage209_decision.json');check_code(p)
    if (d['schema']!=SCHEMA or d['status']!='REVIEW_FROZEN_WEIGHT_REFERENCE_INTERVENTION'
            or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage208']!=c.SOURCE208 or d['source_models']!=4
            or d['new_training_updates']!=0 or d['full_evaluation_trajectories']!=342 or d['confirmation_groups_opened']!=0
            or set(d['table_sha256'])!={f'stage209_{k}.csv' for k in c.TABLES}):raise ValueError('Incomplete Stage-209')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+name)
    print('Stage-209 verified; REVIEW, no new training or method adoption',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in (208,76,89,93,121,170,188,191):p.add_argument(f'--stage{k}-root',type=Path)
    p.add_argument('--stage208-bundle',type=Path,default=DEFAULT_BUNDLE);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-results',action='store_true');args=p.parse_args()
    root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,*sources.SOURCES)):p.error('Need complete Stage-208, checkpoint and dataset source roots')
    paths={k:getattr(args,f'stage{k}_root').expanduser().resolve() for k in sources.SOURCES};src=args.stage208_root.expanduser().resolve()
    c.prior.common.source.common.protocol.safe_output(root,[src,*paths.values(),args.checkpoint,args.stage208_bundle.parent])
    with writer_lock(root):
        prior.guard(root)
        if (root/'stage209_decision.json').exists():verify(root);return
        d208,p208,t208=c.source208(args.stage208_bundle);prior.verify(src)
        if old.read_sealed(src/'manifest.json')!=p208 or old.read_sealed(src/'stage208_decision.json')!=d208:raise ValueError('Changed physical source')
        import torch
        torch.set_num_threads(1)
        runtime=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p208['runtime']),torch,args.gpu)
        differences=sources.old.base.s109.check_runtime(p208['runtime'],runtime);code=dict(p208['code_sha256'])
        for file in (__file__,c.__file__,REPO/'tools/run_stage209_reference_intervention.sh'):
            code[Path(file).resolve().relative_to(REPO).as_posix()]=file_hash(file)
        plan=old.sealed(dict(schema=SCHEMA,source_stage208=c.SOURCE208,entries=p208['entries'],folds=p208['folds'],runtime=runtime,
            runtime_differences=differences,same_source_runtime=not differences,workers=args.workers,design=c.DESIGN,arms=list(c.ARMS),
            intervention='Same own-pair matched Q/K/V/O weights; adapter KV only matched/search/fit-wrong, zero bypass; native template unchanged',
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.expanduser().resolve()),source_paths={str(k):str(v) for k,v in paths.items()},
                    source_root=str(src),bundle=str(args.stage208_bundle.resolve()))
        entries,dataset=prior.load_entries(config,plan);c.prior.common.source.common.protocol.safe_output(root,[dataset])
        native=prior.open_runtime(config,plan,'stage209')
        try:
            models,refs,scope,rr,verified=load_bound(config,native)
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_artifacts=verified,new_training_updates=0)))
            experiment(root,plan,native,models,refs,scope,rr,entries,dataset,config,p208,d208)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
