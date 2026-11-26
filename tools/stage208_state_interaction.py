"""Matched parent-state x spatial-interaction training and full OOF evaluation."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
from time import monotonic, perf_counter
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import numpy as np
import stage208_factor_common as common
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_csv

source,old,previous,spatial=common.source,common.old,common.previous,common.spatial
SCHEMA=common.SCHEMA
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage207_results_20261126T070603Z_1751488.txt'
PROTOCOL_FILE=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE208_PAIRED_STATE_INTERACTION.md'


def guard(root):
    source.guard(root)
    if list(root.rglob('nonfinite_failure.json')): raise ValueError('Recorded nonfinite failure; review before continuing')


def check_code(plan):
    for name,sha in plan['code_sha256'].items():
        if file_hash(REPO/name)!=sha: raise ValueError('Changed frozen dependency '+name)
    if file_hash(PROTOCOL_FILE)!=plan['protocol_sha256']: raise ValueError('Changed Stage-208 protocol')


def open_runtime(config,plan,mode):
    import torch
    import stage192_development_grid_rescue as prior
    torch.set_num_threads(1)
    runtime=prior.old.base.s103.s76.device_runtime(dict(runtime_identity=plan['runtime']),torch,config['gpu'])
    prior.old.base.s109.check_runtime(plan['runtime'],runtime,plan['runtime'])
    args=argparse.Namespace(checkpoint=Path(config['checkpoint']),gpu=config['gpu'],parameter=runtime['tracker']['parameter'],mode=mode)
    diagnostic=Path(config['root'])/'worker_diagnostics'/mode
    native,_=prior.old.base.s103.s76.startup.checked_runtime(args,
        dict(fingerprint=plan['fingerprint'],runtime_identity=runtime,runtime=runtime['tracker']),diagnostic)
    return native


def load_entries(config,plan):
    import stage192_development_grid_rescue as prior
    entries,dataset,_=prior.load_sources({int(k):Path(v) for k,v in config['source_paths'].items()})
    for e in plan['entries']:
        files,truth,_,_,_=entries[e['sequence']]
        if fingerprint(files)!=e['frames_sha256'] or fingerprint(np.asarray(truth).tolist())!=e['truth_sha256']:
            raise ValueError('Changed video input '+e['sequence'])
    return entries,dataset


def model_path(root,fold,seed,origin,interaction):
    return root/'conditions'/origin/'training'/f'f{fold}__s{seed}__{interaction}.pt'


def training_plan(plan,origin):
    if origin not in common.ORIGINS:raise ValueError('Unknown training-state origin')
    # Bind the state source in the checkpoint metadata, as well as its path.
    return old.sealed(dict(parent_plan_fingerprint=plan['fingerprint'],training_condition=origin))


def load_students(root,plan,native):
    rt=native.rt;device=next(rt.net.parameters()).device;models={};refs={}
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            rp=root/'references'/f"f{fold['fold']}__s{seed}.npz"
            meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,fit=fold['fit'])
            value,_=old.cache(rp,meta,lambda:(_ for _ in ()).throw(ValueError('Missing references')))
            refs[fold['fold'],seed]={str(g):a for g,a in zip(value['groups'],value['reference'])}
            for origin in common.ORIGINS:
                for interaction in common.INTERACTIONS:
                    path=model_path(root,fold['fold'],seed,origin,interaction)
                    receipt=old.read_sealed(path.with_suffix('.receipt.json'));meta=receipt['metadata']
                    if (meta['plan_fingerprint']!=training_plan(plan,origin)['fingerprint'] or meta['fold']!=fold['fold'] or meta['seed']!=seed
                            or meta['arm']!=interaction or meta['fit']!=fold['fit'] or meta['held']!=fold['held'] or meta['updates']!=256):
                        raise ValueError('Wrong worker student identity')
                    payload=old.load_model(rt.torch,path,meta)
                    if payload is None: raise ValueError('Missing student')
                    model=spatial.build(rt.torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,seed,interaction).to(device).eval()
                    model.load_state_dict(payload['state'],strict=True);model.requires_grad_(False)
                    if old.model_digest(model)!=payload['scope']['state_sha256']: raise ValueError('Changed student state')
                    models[fold['fold'],seed,origin+'__'+interaction]=model
    return models,refs


def job_list(plan,spec,models):
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    return fold,[('native',0,None)]+[(a,s,models[fold,s,a]) for s in common.DESIGN['seeds'] for a in common.TRAINED]


def evaluate_video(root,plan,native,models,refs,spec,entry,dataset,progress,phase):
    seq=spec['sequence'];files,truth,_,reader,_=entry
    if phase!='full': files=files[:1+common.DESIGN['smoke_updates']]
    fold,jobs=job_list(plan,spec,models)
    directory=root/'evaluation' if phase=='full' else root/('smoke_'+phase)/'evaluation'
    for arm,seed,model in jobs:
        path=directory/f'{seq}__s{seed}__{arm}.npz'
        digest=old.model_digest(model) if model is not None else None
        meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,model_sha256=digest,phase=phase)
        ref=None if model is None else native.rt.torch.as_tensor(refs[fold,seed][spec['group']][None],device=next(native.rt.net.parameters()).device)
        value,reused=old.cache(path,meta,lambda:previous.trace(native,model,ref,files,truth[0],reader,dataset,progress))
        if reused:old.reused_work(progress,len(files)-1)
        old.audit_trace(value,truth[0],len(files)-1)
        if model is not None and old.model_digest(model)!=digest: raise ValueError('Worker mutated student')
    native.assert_frozen()


class WorkerProgress:
    """Only the parent prints a bar. Queue deltas are bounded/coalesced."""
    def __init__(self,channel,worker):
        self.channel,self.worker=channel,worker;self.done=0;self.reused=0;self.sent=0;self.sent_reused=0;self.last=monotonic()
    def advance(self,message='',count=1):
        self.done+=count
        if self.done-self.sent>=16 or monotonic()-self.last>=.5:self.flush()
    def flush(self):
        self.channel.put(('progress',self.worker,self.done-self.sent,self.reused-self.sent_reused))
        self.sent,self.sent_reused,self.last=self.done,self.reused,monotonic()


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as handle,redirect_stdout(handle),redirect_stderr(handle):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);guard(root)
            native=open_runtime(config,plan,f'stage208_{phase}_{worker}');entries,dataset=load_entries(config,plan)
            models,refs=load_students(root,plan,native);progress=WorkerProgress(channel,worker)
            for seq in sequences:
                spec=next(e for e in plan['entries'] if e['sequence']==seq)
                evaluate_video(root,plan,native,models,refs,spec,entries[seq],dataset,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def run_workers(config,sequences,phase,workers,progress,target=worker_main):
    """Independent spawn runtimes, disjoint videos, no shared CUDA/model objects."""
    if workers not in (1,2) or len(sequences)!=len(set(sequences)): raise ValueError('Invalid worker partition')
    ctx=mp.get_context('spawn');channel=ctx.Queue();processes=[];complete=set();start=perf_counter()
    try:
        for i in range(min(workers,len(sequences))):
            proc=ctx.Process(target=target,args=(config,sequences[i::workers],phase,i,channel))
            proc.start();processes.append(proc)
        while len(complete)<len(processes):
            try:kind,i,count,reused=channel.get(timeout=.5)
            except queue.Empty:
                if any(p.exitcode is not None and p.exitcode!=0 for p in processes):raise RuntimeError('Stage-208 worker failed; inspect worker_logs')
                # A terminated worker must never hang the master indefinitely.
                if all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker exited without a completion message')
                continue
            if kind=='error':raise RuntimeError(f'Stage-208 worker {i} failed:\n{count}')
            if kind=='complete':complete.add(i)
            elif kind=='progress':progress.reused+=reused;progress.advance(count=count)
        for p in processes:
            p.join(timeout=30)
            if p.is_alive() or p.exitcode!=0:raise RuntimeError('Worker did not exit cleanly')
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
            p.join(timeout=10)
        channel.close()
    return perf_counter()-start


def aggregate(root,plan,models,entries,artifacts,tables):
    for spec in plan['entries']:
        seq=spec['sequence'];files,truth,_,_,_=entries[seq];fold,jobs=job_list(plan,spec,models);baseline=None
        for arm,seed,model in jobs:
            path=root/'evaluation'/f'{seq}__s{seed}__{arm}.npz';meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,
                fold=fold,seed=seed,arm=arm,model_sha256=old.model_digest(model) if model is not None else None,phase='full')
            value,_=old.cache(path,meta,lambda:(_ for _ in ()).throw(ValueError('Incomplete evaluation')))
            old.audit_trace(value,truth[0],len(files)-1)
            if arm=='native':baseline=value
            tables['seed_video_summary'].append(old.describe_trace(value,baseline,truth,seq,arm,seed,fold));old.record(root,path,artifacts)
            for layer in range(7,13):tables['routing'].append(dict(sequence=seq,fold=fold,seed=seed,arm=arm,layer=layer,frames=int((value['layer']==layer).sum())))


def compare_smoke(root,plan,models,specs,entries,artifacts):
    rows=[]
    for spec in specs:
        seq=spec['sequence'];_,jobs=job_list(plan,spec,models)
        for arm,seed,_ in jobs:
            paths=[root/f'smoke_{mode}'/'evaluation'/f'{seq}__s{seed}__{arm}.npz' for mode in ('serial','parallel')]
            values=[]
            for path in paths:
                r=old.read_sealed(path.with_suffix('.receipt.json'))
                value,_=old.cache(path,r['metadata'],lambda:(_ for _ in ()).throw(ValueError('Missing smoke')))
                old.audit_trace(value,entries[seq][1][0],min(common.DESIGN['smoke_updates'],len(entries[seq][0])-1))
                values.append(value);old.record(root,path,artifacts)
            if set(values[0])!=set(values[1]):raise ValueError('Parallel output schema changed')
            for k in values[0]:
                if k!='pipeline_ms':np.testing.assert_array_equal(values[0][k],values[1][k],err_msg='Serial/parallel changed '+k)
            rows.append(dict(sequence=seq,arm=arm,seed=seed,tracked_frames=len(values[0]['boxes']),exact_quality_and_call_match=True,
                box_max_error=float(np.abs(values[0]['boxes']-values[1]['boxes']).max()),timing_excluded=True))
    return rows


def experiment(root,plan,native,entries,dataset,bank,conditions,support,sampling,controls,config):
    rt=native.rt;states=sum(len(v['frame_id']) for v in bank.values());frames=sum(len(v[0])-1 for v in entries.values())
    smoke_specs=plan['entries'][:2];smoke_frames=sum(min(common.DESIGN['smoke_updates'],len(entries[e['sequence']][0])-1) for e in smoke_specs)
    progress=OverallProgress('Stage-208',24*256+24*states+26*smoke_frames+13*frames+1,unit='work')
    artifacts={};tables={k:[] for k in common.TABLES};tables['support']=support;tables['sampling']=sampling
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    models={};refbank={};progress.report('Matched state x interaction training (24 models)')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            paired=conditions[fold['fold'],seed]
            refs,mapping=common.references(paired['native_pair'],fold,list(bank),rt.lens_z);refbank[fold['fold'],seed]=refs
            groups=list(bank);rp=root/'references'/f"f{fold['fold']}__s{seed}.npz"
            value,_=old.cache(rp,dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,fit=fold['fit']),
                lambda:dict(groups=np.asarray(groups),reference=np.stack([refs[g] for g in groups])))
            np.testing.assert_array_equal(value['reference'],np.stack([refs[g] for g in groups]));old.record(root,rp,artifacts)
            tables['reference_audit'].extend(dict(fold=fold['fold'],seed=seed,group=g,donor=mapping[g],donor_fit=True,
                shared_across_parent_origins=True,template_mean_source='native_pair') for g in groups)
            scopes=[]
            for origin in common.ORIGINS:
                subroot=root/'conditions'/origin;context=training_plan(plan,origin)
                publish_identical(subroot/'condition.json',context)
                artifacts[(subroot/'condition.json').relative_to(root).as_posix()]=file_hash(subroot/'condition.json')
                for interaction in common.INTERACTIONS:
                    local={}
                    model,scope,history=previous.fit(subroot,context,native,paired[origin],refs,fold,seed,interaction,progress,local)
                    source.register_nested(root,subroot,artifacts,local);arm=origin+'__'+interaction
                    models[fold['fold'],seed,arm]=model;scopes.append(scope)
                    tables['scope_summary'].append(dict(training_condition=origin,**scope))
                    tables['training_history'].extend(dict(training_condition=origin,**r) for r in history)
                    if interaction=='matched_spatial':
                        c=controls[fold['fold'],seed,origin];exact=scope['state_sha256']==c['state_sha256']
                        if scope['draws_sha256']!=c['draws_sha256'] or scope['initial_sha256']!=c['initial_sha256']:
                            raise ValueError('Stage-207 control schedule/initialization changed')
                        if plan['same_source_runtime'] and not exact:raise ValueError('Same-runtime paired control must reproduce Stage-207 exactly')
                        tables['control_replay'].append(dict(fold=fold['fold'],seed=seed,origin=origin,same_source_runtime=plan['same_source_runtime'],
                            exact_state_match=exact,state_sha256=scope['state_sha256'],source_state_sha256=c['state_sha256']))
            for key in ('draws_sha256','initial_sha256','trainable_parameters','updates','examples'):
                if len({s[key] for s in scopes})!=1:raise ValueError('Unmatched factorial '+key)
    progress.report('Frozen native-parent fit/held readout')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            for origin in common.ORIGINS:
                for interaction in common.INTERACTIONS:
                    arm=origin+'__'+interaction;local={};subroot=root/'conditions'/origin
                    rows=previous.readout(subroot,plan,native,bank,refbank[fold['fold'],seed],fold,seed,interaction,
                                          models[fold['fold'],seed,arm],progress,local)
                    tables['readout_video'].extend(dict(training_condition=origin,**v) for v in rows);source.register_nested(root,subroot,artifacts,local)
    progress.report('Serial smoke for all six conditions, both seeds, two videos')
    start=perf_counter()
    for spec in smoke_specs:evaluate_video(root,plan,native,models,refbank,spec,entries[spec['sequence']],dataset,progress,'serial')
    serial_seconds=perf_counter()-start
    progress.report('Two-worker smoke; exact boxes/layers/call counts required')
    parallel_seconds=run_workers(config,[e['sequence'] for e in smoke_specs],'parallel',2,progress)
    tables['parallel_smoke']=compare_smoke(root,plan,models,smoke_specs,entries,artifacts)
    publish_identical(root/'parallel_smoke_passed.json',old.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],
        trajectories=len(tables['parallel_smoke']),all_exact=True,timing_excluded=True)))
    artifacts['parallel_smoke_passed.json']=file_hash(root/'parallel_smoke_passed.json')
    progress.report(f"Full OOF evaluation across videos ({plan['workers']} independent workers)")
    elapsed=run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress)
    tables['evaluation_execution']=[dict(phase='serial_smoke',workers=1,wall_seconds=serial_seconds,tracked_work=13*smoke_frames,includes_worker_startup=False),
        dict(phase='parallel_smoke',workers=2,wall_seconds=parallel_seconds,tracked_work=13*smoke_frames,includes_worker_startup=True),
        dict(phase='full_evaluation',workers=plan['workers'],wall_seconds=elapsed,tracked_work=13*frames,includes_worker_startup=True)]
    aggregate(root,plan,models,entries,artifacts,tables)
    tables['policy_summary'],tables['paired'],tables['factor_summary']=common.summarize(tables['seed_video_summary'])
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage208_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete overall progress budget')
    publish_identical(root/'stage208_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_PAIRED_STATE_INTERACTION_FACTORIAL',
        plan_fingerprint=plan['fingerprint'],source_stage207=common.SOURCE207,source_groups=len(bank),training_runs=len(models),
        training_updates=len(tables['training_history']),full_evaluation_trajectories=len(tables['seed_video_summary']),confirmation_groups_opened=0,
        primary_contrasts=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts,
        next_step='Assess matched-template increments over both controls, state interaction and native recovery protection; no automatic adoption',
        timing_scope='Parallel pipeline_ms are contended descriptive measurements, not isolated FPS or speedup evidence')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage208_decision.json');check_code(p)
    if (d['schema']!=SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage207']!=common.SOURCE207
            or d['status']!='REVIEW_PAIRED_STATE_INTERACTION_FACTORIAL' or d['source_groups']!=38 or d['training_runs']!=24
            or d['training_updates']!=6144 or d['full_evaluation_trajectories']!=494 or d['confirmation_groups_opened']!=0
            or set(d['table_sha256'])!={f'stage208_{k}.csv' for k in common.TABLES}): raise ValueError('Incomplete Stage-208')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+name)
    print('Stage-208 verified; REVIEW, no method adoption or confirmation opening',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (194,207,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage207-bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    keys=(194,207,76,89,93,121,170,188,191)
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in keys):
        parser.error('Need Stage-194 features, complete Stage-207 run, original checkpoint and dataset source roots')
    import stage192_development_grid_rescue as prior
    paths={k:getattr(args,f'stage{k}_root').expanduser().resolve() for k in prior.SOURCES}
    r194,r207=args.stage194_root.expanduser().resolve(),args.stage207_root.expanduser().resolve()
    common.source.common.protocol.safe_output(root,[*paths.values(),r194,r207,args.checkpoint,args.stage207_bundle.parent])
    with writer_lock(root):
        guard(root)
        if (root/'stage208_decision.json').exists():verify(root);return
        d207,p207,t207=common.source207(args.stage207_bundle);source.verify(r207)
        if old.read_sealed(r207/'manifest.json')!=p207 or old.read_sealed(r207/'stage207_decision.json')!=d207:
            raise ValueError('Physical Stage-207 differs from reviewed bundle')
        p194,d194,reader,_=previous.inputs.load_source(r194)
        if p194['folds']!=p207['folds'] or p194['entries']!=p207['entries'] or d194['fingerprint']!=p207['source_stage194']:
            raise ValueError('Frozen feature roster changed')
        import torch
        torch.set_num_threads(1)
        runtime=prior.old.base.s103.s76.device_runtime(dict(runtime_identity=p207['runtime']),torch,args.gpu)
        differences=prior.old.base.s109.check_runtime(p207['runtime'],runtime)
        code=dict(p207['code_sha256'])
        for file in (__file__,common.__file__,REPO/'tools/run_stage208_state_interaction.sh'):
            code[Path(file).resolve().relative_to(REPO).as_posix()]=file_hash(file)
        plan=old.sealed(dict(schema=SCHEMA,source_stage194=d194['fingerprint'],source_stage207=common.SOURCE207,
            source207_bundle_sha256=file_hash(args.stage207_bundle),entries=p194['entries'],folds=p194['folds'],runtime=runtime,
            runtime_differences=differences,same_source_runtime=not differences,design=common.DESIGN,arms=list(common.ARMS),workers=args.workers,
            feature_policy='Both frozen Stage-207 raw-prefix banks reused symmetrically on any declared GPU; never recollect only one side',
            training='Fresh 2 origins x 3 interactions x 2 folds x 2 seeds; 256 updates; unchanged paired masks and draws',
            reference_policy='Native-pair fit template means, deterministic off-source donors, shared across both origins; real template unchanged',
            evidence='Historical development full OOF; fresh same-card native and all six trained conditions; no confirmation opening',
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL_FILE)))
        publish_identical(root/'manifest.json',plan)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.expanduser().resolve()),source_paths={str(k):str(v) for k,v in paths.items()})
        entries,dataset=load_entries(config,plan);common.source.common.protocol.safe_output(root,[dataset])
        native=open_runtime(config,plan,'stage208_training')
        try:
            rt=native.rt
            if (rt.lens_z,rt.lens_x,rt.identity['embed_dim'])!=(64,256,192):raise ValueError('Need original native token layout')
            bank={}
            for spec in p194['entries']:
                seq=spec['sequence'];value=reader.arrays(f'features/{seq}.npz',dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix'))
                np.testing.assert_array_equal(value['frame_id'],spec['sample_frames'])
                if value['tokens'].shape!=(len(value['frame_id']),320,192) or not all(np.isfinite(v).all() for v in value.values()):
                    raise ValueError('Bad source features')
                bank[spec['group']]=value
            verified={}
            bank,conditions,support,sampling=common.paired_inputs(r207,d207,p207,t207,bank,entries,rt.tracker.params.search_size,verified)
            controls=common.control_states(torch,rt,r207,d207,p207,t207,verified)
            publish_identical(root/'source_receipt.json',old.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],
                features=reader.verified,stage207_artifacts=verified,frozen_banks_reused=True)))
            experiment(root,plan,native,entries,dataset,bank,conditions,support,sampling,controls,config)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
