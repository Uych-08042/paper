"""One crop, shared prefix, two readouts: output/search-state full-video OOF."""
import argparse
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from pathlib import Path
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage212_state_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
old=c.old;previous=c.prior.base.previous;runtime=previous.prior
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage211_results_20261126T131139Z_2129215.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE212_ONLINE_OUTPUT_SEARCH_STATE.md'


def decode(rt,out,factor,parent,image):
    response=rt.tracker.output_window*out['score_map']
    boxes=rt.net.box_head.cal_bbox(response,out['size_map'],out['offset_map']).view(-1,4)
    box=(boxes.mean(0)*rt.tracker.params.search_size/factor).tolist()
    namespace=rt.tracker.track.__func__.__globals__;saved=rt.tracker.state;rt.tracker.state=np.asarray(parent,float).tolist()
    try:return np.asarray(namespace['clip_box'](rt.tracker.map_box_back(box,factor),image.shape[0],image.shape[1],margin=10),float)
    finally:rt.tracker.state=saved


def dual_trace(native,model,mode,files,initial,reader,dataset,progress):
    if mode not in c.MODES:raise ValueError('Unknown online state rule')
    rt=native.rt;native.initialize(reader(dataset,files[0]),initial);parent=np.asarray(initial,float);rows=[];captured=[]
    counts=dict(block=np.zeros(12,int),head=0,selector=0)
    def block_hook(i):
        def called(m,a,o):counts['block'][i]+=1
        return called
    def head(m,a,o):counts['head']+=1
    def selector(m,a,o):counts['selector']+=1
    with ExitStack() as stack:
        for i,b in enumerate(rt.backbone.blocks):stack.callback(b.register_forward_hook(block_hook(i)).remove)
        stack.callback(rt.net.box_head.register_forward_hook(head).remove)
        stack.callback(rt.backbone.MLP.register_forward_hook(selector).remove)
        stack.callback(rt.backbone.blocks[5].register_forward_hook(lambda m,a,o:captured.append(o.detach().clone())).remove)
        for frame,file in enumerate(files[1:],1):
            captured.clear();counts['block'][:]=0;counts['head']=counts['selector']=0
            image=reader(dataset,file);namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target'];factors=[]
            def counted(*a,**kw):
                v=sampler(*a,**kw);factors.append(float(v[1]));return v
            runtime.previous.sync(rt);start=perf_counter()
            with patch.dict(namespace,sample_target=counted):
                raw=native.step(image,parent)
                if len(captured)!=1 or len(factors)!=1 or counts['head']!=1 or counts['selector']!=1:raise ValueError('Not one public raw readout')
                np.testing.assert_array_equal(counts['block'],raw['block_calls'])
                with rt.torch.no_grad():
                    output=rt.candidate(model.adapt(captured[0],None),raw['layer'])
                    adapted=decode(rt,output,factors[0],parent,image)
            state=c.next_state(raw['box'],adapted,mode);rt.tracker.state=state.tolist()
            runtime.previous.sync(rt);elapsed=1000*(perf_counter()-start)
            if len(factors)!=1 or len(captured)!=1 or rt.tracker.frame_id!=frame:raise ValueError('Duplicated crop, prefix or frame increment')
            rows.append(dict(frame_id=frame,parent=parent.copy(),boxes=adapted,reference_boxes=raw['box'],next_parent=state,
                layer=raw['layer'],block_calls=counts['block'].copy(),head_calls=counts['head'],selector_calls=counts['selector'],crop_calls=len(factors),pipeline_ms=elapsed))
            parent=state;progress.advance()
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}


def jobs(plan,spec,models):
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    return fold,[('native',0,None)]+[('matched_single',s,models[fold,s,'matched']) for s in c.SEEDS]+[(a,s,models[fold,s,'matched']) for s in c.SEEDS for a in c.MODES]


def path_for(root,phase,seq,seed,arm):return (root/'evaluation' if phase=='full' else root/('smoke_'+phase)/'evaluation')/f'{seq}__s{seed}__{arm}.npz'


def meta_for(plan,seq,fold,seed,arm,model,phase):
    return dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,phase=phase,model_sha256=old.model_digest(model) if model is not None else None)


def required(path,meta):return old.cache(path,meta,lambda:(_ for _ in ()).throw(ValueError('Missing worker cache')))[0]


def evaluate_video(root,plan,native,models,spec,entry,dataset,progress,phase):
    files,gt,_,reader,_=entry
    if phase!='full':files=files[:1+c.PROTOCOL.DESIGN['smoke_updates']]
    seq=spec['sequence'];fold,work=jobs(plan,spec,models)
    for arm,seed,model in work:
        path=path_for(root,phase,seq,seed,arm);meta=meta_for(plan,seq,fold,seed,arm,model,phase)
        def compute():
            if arm in c.MODES:return dual_trace(native,model,arm,files,gt[0],reader,dataset,progress)
            return runtime.previous.trace(native,model,None,files,gt[0],reader,dataset,progress)
        value,reused=old.cache(path,meta,compute)
        if reused:old.reused_work(progress,len(files)-1)
        c.audit(value,gt[0],len(files)-1,arm)
        if model is not None and old.model_digest(model)!=meta['model_sha256']:raise ValueError('Frozen weights changed')
    native.assert_frozen()


def check_code(plan):
    for n,sha in plan['code_sha256'].items():
        if file_hash(REPO/n)!=sha:raise ValueError('Changed dependency '+n)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);runtime.guard(root)
            native=runtime.open_runtime(config,plan,f'stage212_{phase}_{worker}');entries,dataset=runtime.load_entries(config,plan)
            models,_,_,_,_=previous.load_bound(config,native);progress=runtime.WorkerProgress(channel,worker)
            for seq in sequences:
                spec=next(e for e in plan['entries'] if e['sequence']==seq)
                evaluate_video(root,plan,native,models,spec,entries[seq],dataset,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def check_controls(values,seq,phase):
    rows=[];native=values['native',0]
    for seed in c.SEEDS:
        mm=values['mm',seed];single=values['matched_single',seed];nn=values['nn',seed]
        for k in ('frame_id','parent','boxes','layer'):np.testing.assert_array_equal(mm[k],single[k],err_msg='Dual mm must reproduce single matched '+k)
        for a,b in (('frame_id','frame_id'),('parent','parent'),('reference_boxes','boxes'),('layer','layer')):
            np.testing.assert_array_equal(nn[a],native[b],err_msg='nn raw branch must reproduce native state '+a)
        rows.append(dict(sequence=seq,phase=phase,seed=seed,mm_matches_single_quality=True,nn_reference_matches_native=True,
                         mm_heads=2,single_heads=1,selector_calls_per_frame=1))
    return rows


def collect(root,plan,models,spec,entry,phase,artifacts):
    seq=spec['sequence'];fold,work=jobs(plan,spec,models);values={}
    frames=len(entry[0])-1 if phase=='full' else min(c.PROTOCOL.DESIGN['smoke_updates'],len(entry[0])-1)
    for arm,seed,model in work:
        path=path_for(root,phase,seq,seed,arm);v=required(path,meta_for(plan,seq,fold,seed,arm,model,phase));c.audit(v,entry[1][0],frames,arm)
        old.record(root,path,artifacts);values[arm,seed]=v
    return values


def describe(values,gt,seq,fold,arm,seed):
    trace=values[arm,seed];row=old.describe_trace(trace,values['native',0],gt,seq,arm,seed,fold)
    refs=c.SEEDS if arm=='native' else [seed]
    matched=[old.describe_trace(trace,values['matched_single',s],gt,seq,arm,seed,fold) for s in refs]
    row.update(missed_matched_recoveries=float(np.mean([r['missed_native_recoveries'] for r in matched])),
        matched_good_frame_loss=float(np.mean([r['native_good_frame_loss'] for r in matched])),
        introduced_matched_failure=any(r['introduced_failure'] for r in matched))
    return row


def experiment(root,plan,native,models,scope,entries,dataset,config):
    specs=previous.specs_for_smoke(plan);sf=sum(min(c.PROTOCOL.DESIGN['smoke_updates'],len(entries[e['sequence']][0])-1) for e in specs)
    frames=sum(len(v[0])-1 for v in entries.values());progress=OverallProgress('Stage-212',22*sf+11*frames+1,unit='frames')
    artifacts={};tables={k:[] for k in c.TABLES};tables['model_scope']=scope
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Single and dual readout serial identity smoke');start=perf_counter()
    for spec in specs:evaluate_video(root,plan,native,models,spec,entries[spec['sequence']],dataset,progress,'serial')
    serial_seconds=perf_counter()-start
    for spec in specs:
        vv=collect(root,plan,models,spec,entries[spec['sequence']],'serial',artifacts)
        tables['control_replay'].extend(check_controls(vv,spec['sequence'],'serial'))
    progress.report('Two-worker smoke with identical state/output histories')
    parallel_seconds=runtime.run_workers(config,[e['sequence'] for e in specs],'parallel',2,progress,target=worker_main)
    for spec in specs:
        a=collect(root,plan,models,spec,entries[spec['sequence']],'serial',artifacts);b=collect(root,plan,models,spec,entries[spec['sequence']],'parallel',artifacts)
        tables['control_replay'].extend(check_controls(b,spec['sequence'],'parallel'))
        for arm,seed in a:
            previous.equal_quality(a[arm,seed],b[arm,seed])
            tables['smoke'].append(dict(sequence=spec['sequence'],arm=arm,seed=seed,exact_state_output_calls=True,timing_excluded=True))
    progress.report('Complete-video output/search-state factorial (38 source groups)')
    elapsed=runtime.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    for spec in plan['entries']:
        seq=spec['sequence'];values=collect(root,plan,models,spec,entries[seq],'full',artifacts);fold,_=jobs(plan,spec,models)
        tables['control_replay'].extend(check_controls(values,seq,'full'))
        for arm,seed in values:
            v=values[arm,seed];tables['seed_video_summary'].append(describe(values,entries[seq][1],seq,fold,arm,seed))
            tables['calls'].append(dict(sequence=seq,arm=arm,seed=seed,frames=len(v['boxes']),block_calls=int(v['block_calls'].sum()),
                head_calls=int(v['head_calls'].sum()),selector_calls=int(v['selector_calls'].sum()),crop_calls=int(v['crop_calls'].sum()),
                sum_pipeline_ms=float(v['pipeline_ms'].sum())))
            for layer in range(7,13):tables['routing'].append(dict(sequence=seq,arm=arm,seed=seed,fold=fold,layer=layer,frames=int((v['layer']==layer).sum())))
    tables['policy_summary'],tables['paired'],tables['factor_summary']=c.summarize(tables['seed_video_summary'])
    tables['execution']=[dict(phase='serial_smoke',workers=1,wall_seconds=serial_seconds,includes_worker_startup=False),
        dict(phase='parallel_smoke',workers=2,wall_seconds=parallel_seconds,includes_worker_startup=True),
        dict(phase='full',workers=plan['workers'],wall_seconds=elapsed,includes_worker_startup=True)]
    hashes={}
    for n,rows in tables.items():
        path=root/f'stage212_{n}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete overall progress budget')
    publish_identical(root/'stage212_decision.json',old.sealed(dict(schema=c.SCHEMA,status='REVIEW_ONLINE_OUTPUT_SEARCH_STATE_FACTORIAL',
        plan_fingerprint=plan['fingerprint'],source_stage211=c.SOURCE211,source_groups=len(plan['entries']),new_training_updates=0,
        full_evaluation_trajectories=len(tables['seed_video_summary']),confirmation_groups_opened=0,
        primary_contrasts=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='Online same-crop raw reference, dual-readout compute counted; no oracle geometry or automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();runtime.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage212_decision.json');check_code(p)
    if (d['schema']!=c.SCHEMA or d['status']!='REVIEW_ONLINE_OUTPUT_SEARCH_STATE_FACTORIAL' or d['plan_fingerprint']!=p['fingerprint']
            or d['source_stage211']!=c.SOURCE211 or d['source_groups']!=38 or d['new_training_updates']!=0 or d['full_evaluation_trajectories']!=418
            or d['confirmation_groups_opened']!=0 or set(d['table_sha256'])!={f'stage212_{k}.csv' for k in c.TABLES}):raise ValueError('Incomplete Stage-212')
    for n,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+n)
    print('Stage-212 verified; REVIEW, no training or adoption',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (208,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage211-bundle',type=Path,default=DEFAULT_BUNDLE);parser.add_argument('--stage208-bundle',type=Path,default=previous.DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');args=parser.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,*sources.SOURCES)):parser.error('Need Stage-208 models, original checkpoint and seven dataset sources')
    paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES};src=args.stage208_root.resolve()
    c.PROTOCOL.safe_output(root,[src,args.checkpoint,*paths.values(),args.stage211_bundle.parent])
    with writer_lock(root):
        runtime.guard(root)
        if (root/'stage212_decision.json').exists():verify(root);return
        d,p,t=c.source211(args.stage211_bundle)
        for n,sha in p['code_sha256'].items():
            if file_hash(REPO/n)!=sha:raise ValueError('Frozen source dependency changed '+n)
        if file_hash(c.prior.PROTOCOL)!=p['protocol_sha256']:raise ValueError('Frozen Stage-211 protocol changed')
        import torch
        torch.set_num_threads(1)
        identity=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p['runtime']),torch,args.gpu)
        differences=sources.old.base.s109.check_runtime(p['runtime'],identity);code=dict(p['code_sha256'])
        for f in (__file__,c.__file__,REPO/'tools/run_stage212_state_experiment.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        plan=old.sealed(dict(schema=c.SCHEMA,source_stage211=c.SOURCE211,entries=p['entries'],folds=p['folds'],runtime=identity,
            runtime_differences=differences,workers=args.workers,arms=list(c.ARMS),code_sha256=code,protocol_sha256=file_hash(PROTOCOL),
            design='Fresh native/single-matched controls; four online dual-readout states, adapted external output; no offline parents or future GT; no training'))
        publish_identical(root/'manifest.json',plan)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_paths={str(k):str(v) for k,v in paths.items()},
                    source_root=str(src),bundle=str(args.stage208_bundle.resolve()))
        entries,dataset=runtime.load_entries(config,plan);c.PROTOCOL.safe_output(root,[dataset]);native=runtime.open_runtime(config,plan,'stage212')
        try:
            models,_,scope,_,verified=previous.load_bound(config,native)
            if [{k:str(v) for k,v in r.items()} for r in scope]!=t['model_scope']:raise ValueError('Frozen model identity differs from Stage-211')
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],models=verified,stage211_bundle_sha256=file_hash(args.stage211_bundle))))
            experiment(root,plan,native,models,scope,entries,dataset,config)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
