"""All-source natural recovery: frozen parent x model CENTER mechanism audit."""
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
import stage210_recovery_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
previous,old=c.previous,c.old
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage209_results_20261126T113342Z_1990700.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE210_ALL_SOURCE_RECOVERY_AUDIT.md'


def traces(config,spec,source_plan,decision,tables,verified):
    root=Path(config['trace_root']);fold=next(f['fold'] for f in source_plan['folds'] if spec['group'] in f['held']);seq=spec['sequence'];result={}
    for mode,seed in [('native',0)]+[('matched',s) for s in c.SEEDS]:
        sha=None if mode=='native' else next(r['state_sha256'] for r in tables['model_scope'] if int(r['fold'])==fold and int(r['seed'])==seed)
        meta=dict(plan_fingerprint=source_plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=mode,model_sha256=sha,phase='full')
        result[mode,seed]=previous.prior.common.bound_arrays(root,decision,f'evaluation/{seq}__s{seed}__{mode}.npz',meta,verified)
    return result['native',0],{s:result['matched',s] for s in c.SEEDS}


def dense_boxes(rt,size,offset,factor,parent,hw):
    """Original cal_bbox at every cell and original public map/clip arithmetic."""
    torch=rt.torch;side=rt.search_side;n=side*side
    with torch.no_grad():
        forced=torch.eye(n,device=size.device,dtype=size.dtype).reshape(n,1,side,side)
        predicted=rt.net.box_head.cal_bbox(forced,size.expand(n,-1,-1,-1),offset.expand(n,-1,-1,-1))
        physical=(predicted*rt.tracker.params.search_size/factor).tolist()
    namespace=rt.tracker.track.__func__.__globals__;clip=namespace['clip_box'];saved=rt.tracker.state;rt.tracker.state=np.asarray(parent,float).tolist()
    try:return np.asarray([clip(rt.tracker.map_box_back(v,factor),int(hw[0]),int(hw[1]),margin=10) for v in physical])
    finally:rt.tracker.state=saved


def capture(native,model,image,initial_image,initial,parent,frame):
    rt=native.rt;native.initialize(initial_image,initial);rt.tracker.frame_id=int(frame)-1;heads=[];factors=[]
    namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target']
    def sampled(*a,**kw):
        value=sampler(*a,**kw);factors.append(float(value[1]));return value
    with ExitStack() as stack:
        stack.enter_context(patch.dict(namespace,sample_target=sampled))
        stack.callback(rt.net.box_head.register_forward_hook(lambda m,a,o:heads.append(tuple(v.detach().clone() for v in (o[0],o[2],o[3])))).remove)
        if model is not None:stack.enter_context(previous.prior.spatial.Installed(native,model,None))
        out=native.step(image,parent)
    if len(heads)!=1 or len(factors)!=1:raise ValueError('Need one public crop and CENTER forward')
    score,size,offset=heads[0];grid=dense_boxes(rt,size,offset,factors[0],parent,image.shape[:2])
    cell=int((score*rt.tracker.output_window).flatten(1).argmax(1).item())
    np.testing.assert_array_equal(grid[cell],out['box'],err_msg='Dense grid must exactly match public selected box')
    return dict(box=out['box'],grid=grid,score=score.cpu().numpy()[0],size=size.cpu().numpy()[0],offset=offset.cpu().numpy()[0],
        factor=factors[0],image_hw=np.asarray(image.shape[:2]),cell=cell,layer=out['layer'],
        block_calls=out['block_calls'],head_calls=out['head_calls'],selector_calls=out['selector_calls'],crop_calls=1)


def audit_sequence(root,plan,native,models,spec,entry,dataset,nt,mt,selection,progress,phase='full'):
    files,gt,_,reader,_=entry;seq=spec['sequence'];fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held']);paths=[]
    for seed in c.SEEDS:
        frames=sorted({r['frame'] for r in selection if r['seed']==seed})
        if not frames:continue
        for origin in ('native','matched'):
            trace=nt if origin=='native' else mt[seed]
            for mode in ('native','matched'):
                model=None if mode=='native' else models[fold,seed,'matched']
                path=root/phase/f'{seq}__s{seed}__{origin}__{mode}.npz';sha=old.model_digest(model) if model is not None else None
                meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,seed=seed,parent_origin=origin,model=mode,frames=frames,model_sha256=sha)
                def compute():
                    rows=[]
                    for frame in frames:
                        parent=trace['parent'][frame-1]
                        out=capture(native,model,reader(dataset,files[frame]),reader(dataset,files[0]),gt[0],parent,frame)
                        rows.append(dict(frame_id=frame,parent=parent,**out));progress.advance()
                    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                value,reused=old.cache(path,meta,compute)
                if reused:old.reused_work(progress,len(frames))
                if model is not None and old.model_digest(model)!=sha:raise ValueError('Model changed')
                np.testing.assert_array_equal(value['frame_id'],frames)
                paths.append((path,value,origin,mode,seed))
    native.assert_frozen();return paths


def check_code(plan):
    for n,sha in plan['code_sha256'].items():
        if file_hash(REPO/n)!=sha:raise ValueError('Changed dependency '+n)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan)
            native=previous.prior.open_runtime(config,plan,f'stage210_{worker}');entries,dataset=previous.prior.load_entries(config,plan)
            models,_,_,_,_=previous.load_bound(config,native)
            d,p,t=c.source209(config['trace_bundle']);progress=previous.prior.WorkerProgress(channel,worker)
            for seq in sequences:
                spec=next(e for e in plan['entries'] if e['sequence']==seq);nt,mt=traces(config,spec,p,d,t,{})
                selected=[r for r in plan['selection'] if r['sequence']==seq]
                audit_sequence(root,plan,native,models,spec,entries[seq],dataset,nt,mt,selected,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def interpret(spec,paths,selection,gt,nt,mt,same_runtime):
    cells=[];replay=[]
    for path,v,origin,mode,seed in paths:
        lookup={int(f):i for i,f in enumerate(v['frame_id'])}
        for frame,i in lookup.items():
            expected=nt if mode=='native' else mt[seed]
            diagonal=origin==mode
            exact=bool(np.array_equal(v['box'][i],expected['boxes'][frame-1])) and int(v['layer'][i])==int(expected['layer'][frame-1])
            if diagonal and same_runtime and not exact:raise ValueError('Same-runtime diagonal must reproduce source box/layer')
            if diagonal:replay.append(dict(sequence=spec['sequence'],frame=frame,seed=seed,parent_origin=origin,model=mode,
                same_runtime=same_runtime,exact_source_box_layer=exact,box_max_error=float(np.abs(v['box'][i]-expected['boxes'][frame-1]).max())))
            truth=gt[frame];covered=previous.prior.common.source.common.covered(truth,v['parent'][i],float(v['factor'][i]),v['image_hw'][i],256)
            best=float(old.tracking_iou(v['grid'][i],truth).max());output=float(old.tracking_iou(v['box'][i],truth))
            # Inherit the audited original six-prefix/one-candidate call accounting.
            blocks=np.zeros(12,int);blocks[:6]=1;blocks[int(v['layer'][i])-1]=1
            np.testing.assert_array_equal(v['block_calls'][i],blocks)
            if any(int(v[k][i])!=1 for k in ('head_calls','selector_calls','crop_calls')):raise ValueError('Call budget changed')
            for r in selection:
                if r['seed']==seed and r['frame']==frame:
                    cells.append(dict(**r,parent_origin=origin,model=mode,fully_covered=covered,dense_best_iou=best,output_iou=output,
                        category=c.category(covered,best,output),layer=int(v['layer'][i])))
    return cells,replay


def experiment(root,plan,native,models,entries,dataset,trace_bank,config,scope):
    work=sum(4*len({r['frame'] for r in plan['selection'] if r['sequence']==e['sequence'] and r['seed']==seed}) for e in plan['entries'] for seed in c.SEEDS)
    # One serial full-sequence subset is repeated in the worker run for parity.
    smoke=next(e for e in plan['entries'] if any(r['sequence']==e['sequence'] for r in plan['selection']))
    smoke_selection=[r for r in plan['selection'] if r['sequence']==smoke['sequence']]
    smoke_work=sum(4*len({r['frame'] for r in smoke_selection if r['seed']==s}) for s in c.SEEDS)
    progress=OverallProgress('Stage-210',smoke_work+work+1,unit='searches');artifacts={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Serial parity reference at first supported source')
    nt,mt=trace_bank[smoke['sequence']]
    serial=audit_sequence(root,plan,native,models,smoke,entries[smoke['sequence']],dataset,nt,mt,smoke_selection,progress,'serial')
    progress.report('All-source recovery/control parent x model audit')
    elapsed=previous.prior.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    tables={k:[] for k in c.TABLES};tables['events']=plan['events'];tables['source_support']=plan['source_support'];tables['selection']=plan['selection'];tables['model_scope']=scope
    for spec in plan['entries']:
        selection=[r for r in plan['selection'] if r['sequence']==spec['sequence']];nt,mt=trace_bank[spec['sequence']]
        # Load only; a missing worker result cannot be computed by the master.
        with patch.object(native,'step',side_effect=AssertionError('Missing worker result')):
            paths=audit_sequence(root,plan,native,models,spec,entries[spec['sequence']],dataset,nt,mt,selection,NoProgress())
        for path,v,origin,mode,seed in paths:
            old.record(root,path,artifacts)
            if spec['sequence']==smoke['sequence']:
                other=next(x[1] for x in serial if x[2:]==(origin,mode,seed))
                previous.equal_quality(v,other)
        cells,replay=interpret(spec,paths,selection,entries[spec['sequence']][1],nt,mt,plan['same_source_runtime'])
        tables['cells'].extend(cells);tables['replay'].extend(replay)
    for path,*_ in serial:old.record(root,path,artifacts)
    tables['event_cells'],tables['summary'],tables['contrasts']=c.summarize(tables['cells'],plan['entries'])
    tables['execution']=[dict(workers=plan['workers'],wall_seconds=elapsed,unique_searches=work,serial_smoke_searches=smoke_work,
        serial_worker_arrays_exact=True,new_training_updates=0,new_full_trajectories=0)]
    hashes={}
    for name,rows in tables.items():
        if not rows:
            # Explicit sentinel for a source roster with no natural failures; never fabricate events.
            rows=[dict(status='NO_SUPPORTED_ROWS')]
        path=root/f'stage210_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete progress budget')
    publish_identical(root/'stage210_decision.json',old.sealed(dict(schema=c.SCHEMA,status='REVIEW_ALL_SOURCE_RECOVERY_MECHANISM',
        plan_fingerprint=plan['fingerprint'],source_stage209=c.SOURCE209,source_groups=len(plan['entries']),
        diagnostic_searches=work,new_training_updates=0,new_full_trajectories=0,confirmation_groups_opened=0,
        table_sha256=hashes,artifact_sha256=artifacts,interpretation='Same-frame conditional diagnostics, not causal blame or an online recovery method')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


class NoProgress:
    reused=0
    def advance(self,*a,**kw):pass


def verify(root):
    root=Path(root).resolve();previous.prior.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage210_decision.json');check_code(p)
    if (d['schema']!=c.SCHEMA or d['status']!='REVIEW_ALL_SOURCE_RECOVERY_MECHANISM' or d['plan_fingerprint']!=p['fingerprint']
            or d['source_stage209']!=c.SOURCE209 or d['source_groups']!=38 or d['new_training_updates']!=0 or d['new_full_trajectories']!=0
            or d['confirmation_groups_opened']!=0 or set(d['table_sha256'])!={f'stage210_{n}.csv' for n in c.TABLES}):raise ValueError('Incomplete Stage-210')
    for n,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+n)
    print('Stage-210 verified; REVIEW diagnostic, no training or adoption',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (208,209,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage209-bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--stage208-bundle',type=Path,default=previous.DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,209,*sources.SOURCES)):parser.error('Need Stage-208 models, Stage-209 traces, checkpoint and seven data roots')
    r208=args.stage208_root.resolve();r209=args.stage209_root.resolve();paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES}
    previous.prior.common.source.common.protocol.safe_output(root,[r208,r209,args.checkpoint,*paths.values(),args.stage209_bundle.parent])
    with writer_lock(root):
        previous.prior.guard(root)
        if (root/'stage210_decision.json').exists():verify(root);return
        d,p,t=c.source209(args.stage209_bundle);previous.verify(r209)
        if old.read_sealed(r209/'manifest.json')!=p or old.read_sealed(r209/'stage209_decision.json')!=d:raise ValueError('Physical source changed')
        import torch
        torch.set_num_threads(1)
        runtime=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p['runtime']),torch,args.gpu)
        differences=sources.old.base.s109.check_runtime(p['runtime'],runtime)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_paths={str(k):str(v) for k,v in paths.items()},
            source_root=str(r208),bundle=str(args.stage208_bundle.resolve()),trace_root=str(r209),trace_bundle=str(args.stage209_bundle.resolve()))
        entries,dataset=previous.prior.load_entries(config,p);previous.prior.common.source.common.protocol.safe_output(root,[dataset])
        bank={};events=[];selection=[];support=[];verified={}
        for spec in p['entries']:
            nt,mt=traces(config,spec,p,d,t,verified);gt=entries[spec['sequence']][1]
            for trace in (nt,*mt.values()):old.audit_trace(trace,gt[0],spec['tracked_frames'])
            inventory,chosen,s=c.select(spec,nt,mt,gt);events.extend(inventory);selection.extend(chosen);support.append(s);bank[spec['sequence']]=(nt,mt)
            for seed in c.SEEDS:
                lost=len({r['event'] for r in chosen if r['seed']==seed and r['kind']=='recovery' and r['stratum']=='lost'})
                expected=next(r for r in t['seed_video_summary'] if r['sequence']==spec['sequence'] and r['seed']==str(seed) and r['arm']=='matched')
                if lost!=int(expected['missed_native_recoveries']):raise ValueError('Recovery selection differs from Stage-209 metric')
        if not selection:raise ValueError('No diagnostic support; keep source inventory, do not fabricate points')
        code=dict(p['code_sha256'])
        for f in (__file__,c.__file__,REPO/'tools/run_stage210_recovery_audit.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        plan=old.sealed(dict(schema=c.SCHEMA,source_stage209=c.SOURCE209,entries=p['entries'],folds=p['folds'],runtime=runtime,
            same_source_runtime=not differences,runtime_differences=differences,workers=args.workers,events=events,selection=selection,source_support=support,
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL),design='All native recoveries, three-frame criterion, uniform normal controls; crossed parent/model; no online GT actions'))
        publish_identical(root/'manifest.json',plan)
        native=previous.prior.open_runtime(config,plan,'stage210')
        try:
            models,_,scope,_,model_verified=previous.load_bound(config,native)
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],traces=verified,models=model_verified)))
            experiment(root,plan,native,models,entries,dataset,bank,config,scope)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
