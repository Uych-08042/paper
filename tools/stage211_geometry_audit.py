"""Frozen matched model: cross native/matched parent center and dimensions."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage211_geometry_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
base=c.previous;runtime=base.previous.prior;old=c.old
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage210_results_20261126T121031Z_2077681.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE211_RECOVERY_CENTER_SIZE.md'


def endpoints(config,spec,p,d,t,verified):
    root=Path(config['diagnostic_root']);result={};seq=spec['sequence']
    fold=next(f['fold'] for f in p['folds'] if spec['group'] in f['held'])
    for seed in c.SEEDS:
        frames=sorted({r['frame'] for r in p['selection'] if r['sequence']==seq and r['seed']==seed})
        if not frames:continue
        sha=next(r['state_sha256'] for r in t['model_scope'] if int(r['fold'])==fold and int(r['seed'])==seed)
        for origin in ('native','matched'):
            meta=dict(plan_fingerprint=p['fingerprint'],sequence=seq,seed=seed,parent_origin=origin,model='matched',frames=frames,model_sha256=sha)
            result[seed,origin]=runtime.common.bound_arrays(root,d,f'full/{seq}__s{seed}__{origin}__matched.npz',meta,verified)
            np.testing.assert_array_equal(result[seed,origin]['frame_id'],frames)
    return result


def audit_sequence(root,plan,native,models,spec,entry,dataset,ends,selection,progress,phase='full'):
    files,gt,_,reader,_=entry;seq=spec['sequence'];fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held']);paths=[]
    for seed in c.SEEDS:
        frames=sorted({r['frame'] for r in selection if r['seed']==seed})
        if not frames:continue
        a,b=ends[seed,'native'],ends[seed,'matched'];np.testing.assert_array_equal(a['frame_id'],b['frame_id'])
        lookup={int(f):i for i,f in enumerate(a['frame_id'])};model=models[fold,seed,'matched'];sha=old.model_digest(model)
        for mode in c.MODES:
            path=root/phase/f'{seq}__s{seed}__{mode}.npz'
            meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,seed=seed,mode=mode,frames=frames,model_sha256=sha)
            def compute():
                rows=[]
                for frame in frames:
                    i=lookup[frame];parent=c.parent(a['parent'][i],b['parent'][i],mode)
                    out=base.capture(native,model,reader(dataset,files[frame]),reader(dataset,files[0]),gt[0],parent,frame)
                    rows.append(dict(frame_id=frame,parent=parent,**out));progress.advance()
                return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            value,reused=old.cache(path,meta,compute)
            if reused:old.reused_work(progress,len(frames))
            if old.model_digest(model)!=sha:raise ValueError('Frozen model changed')
            np.testing.assert_array_equal(value['frame_id'],frames);paths.append((path,value,mode,seed))
    native.assert_frozen();return paths


def interpret(spec,paths,selection,truth,ends,same_runtime):
    cells=[];replay=[]
    for path,v,mode,seed in paths:
        a,b=ends[seed,'native'],ends[seed,'matched'];indices={int(f):i for i,f in enumerate(a['frame_id'])}
        if mode in ('mm','nn'):
            source=b if mode=='mm' else a
            exact=set(v)==set(source) and all(np.array_equal(v[k],source[k]) for k in v)
            if same_runtime and not exact:raise ValueError('Same-runtime endpoint arrays must exactly reproduce Stage-210')
            replay.append(dict(sequence=spec['sequence'],seed=seed,mode=mode,same_source_runtime=same_runtime,
                exact_all_arrays=exact,box_max_error=float(np.abs(v['box']-source['box']).max())))
        for i,frame in enumerate(v['frame_id']):
            j=indices[int(frame)];expected=c.parent(a['parent'][j],b['parent'][j],mode)
            np.testing.assert_array_equal(v['parent'][i],expected)
            center_source=a['parent'][j] if mode[0]=='n' else b['parent'][j]
            wh_source=a['parent'][j] if mode[1]=='n' else b['parent'][j]
            error=float(np.abs(v['parent'][i,:2]+v['parent'][i,2:]/2-(center_source[:2]+center_source[2:]/2)).max())
            np.testing.assert_array_equal(v['parent'][i,2:],wh_source[2:])
            factor=float(v['factor'][i]);gt=truth[int(frame)]
            covered=runtime.common.source.common.covered(gt,v['parent'][i],factor,v['image_hw'][i],256)
            best=float(old.tracking_iou(v['grid'][i],gt).max());output=float(old.tracking_iou(v['box'][i],gt))
            blocks=np.zeros(12,int);blocks[:6]=1;blocks[int(v['layer'][i])-1]=1
            np.testing.assert_array_equal(v['block_calls'][i],blocks)
            if any(int(v[k][i])!=1 for k in ('head_calls','selector_calls','crop_calls')):raise ValueError('Public call accounting changed')
            side=int(round(256/factor));origin=np.asarray([round(expected[k]+expected[k+2]/2-side/2) for k in range(2)])
            for r in selection:
                if r['seed']==seed and r['frame']==int(frame):
                    cells.append(dict(**r,mode=mode,center_origin='native' if mode[0]=='n' else 'matched',size_origin='native' if mode[1]=='n' else 'matched',
                        center_reconstruction_error_px=error,actual_crop_left=int(origin[0]),actual_crop_top=int(origin[1]),actual_crop_side=side,
                        gt_width_network_px=float(gt[2]*factor),gt_height_network_px=float(gt[3]*factor),fully_covered=covered,
                        dense_best_iou=best,output_iou=output,category=base.c.category(covered,best,output),layer=int(v['layer'][i])))
    return cells,replay


def check_code(plan):
    for name,sha in plan['code_sha256'].items():
        if file_hash(REPO/name)!=sha:raise ValueError('Changed dependency '+name)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);runtime.guard(root)
            native=runtime.open_runtime(config,plan,f'stage211_{worker}');entries,dataset=runtime.load_entries(config,plan)
            models,_,_,_,_=base.previous.load_bound(config,native)
            d,p,t=c.source210(config['diagnostic_bundle']);progress=runtime.WorkerProgress(channel,worker)
            for seq in sequences:
                spec=next(e for e in plan['entries'] if e['sequence']==seq);ends=endpoints(config,spec,p,d,t,{})
                selection=[r for r in plan['selection'] if r['sequence']==seq]
                audit_sequence(root,plan,native,models,spec,entries[seq],dataset,ends,selection,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def experiment(root,plan,native,models,entries,dataset,end_bank,config,scope):
    work=sum(4*len({r['frame'] for r in plan['selection'] if r['sequence']==e['sequence'] and r['seed']==seed}) for e in plan['entries'] for seed in c.SEEDS)
    smoke=next(e for e in plan['entries'] if any(r['sequence']==e['sequence'] for r in plan['selection']))
    chosen=[r for r in plan['selection'] if r['sequence']==smoke['sequence']]
    smoke_work=sum(4*len({r['frame'] for r in chosen if r['seed']==s}) for s in c.SEEDS)
    progress=OverallProgress('Stage-211',smoke_work+work+1,unit='searches');artifacts={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Serial center x size parity reference')
    serial=audit_sequence(root,plan,native,models,smoke,entries[smoke['sequence']],dataset,end_bank[smoke['sequence']],chosen,progress,'serial')
    progress.report('Frozen-model all-source center x size audit')
    elapsed=runtime.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    tables={k:[] for k in c.TABLES};tables['events']=plan['events'];tables['source_support']=plan['source_support'];tables['selection']=plan['selection'];tables['model_scope']=scope
    for spec in plan['entries']:
        selection=[r for r in plan['selection'] if r['sequence']==spec['sequence']]
        with patch.object(native,'step',side_effect=AssertionError('Worker cache missing')):
            paths=audit_sequence(root,plan,native,models,spec,entries[spec['sequence']],dataset,end_bank[spec['sequence']],selection,base.NoProgress())
        for path,v,mode,seed in paths:
            old.record(root,path,artifacts)
            if spec['sequence']==smoke['sequence']:
                other=next(x[1] for x in serial if x[2:]==(mode,seed));base.previous.equal_quality(v,other)
        cells,replay=interpret(spec,paths,selection,entries[spec['sequence']][1],end_bank[spec['sequence']],plan['same_source_runtime'])
        tables['cells'].extend(cells);tables['endpoint_replay'].extend(replay)
    for path,*_ in serial:old.record(root,path,artifacts)
    tables['event_cells'],tables['summary'],tables['contrasts']=c.summarize(tables['cells'],plan['entries'])
    tables['execution']=[dict(workers=plan['workers'],wall_seconds=elapsed,diagnostic_searches=work,serial_smoke_searches=smoke_work,
        serial_worker_arrays_exact=True,new_training_updates=0,new_full_trajectories=0)]
    hashes={}
    for name,rows in tables.items():
        if not rows:rows=[dict(status='NO_SUPPORTED_ROWS')]
        path=root/f'stage211_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete progress budget')
    publish_identical(root/'stage211_decision.json',old.sealed(dict(schema=c.SCHEMA,status='REVIEW_RECOVERY_CENTER_SIZE_FACTORIAL',
        plan_fingerprint=plan['fingerprint'],source_stage210=c.SOURCE210,source_groups=len(plan['entries']),diagnostic_searches=work,
        new_training_updates=0,new_full_trajectories=0,confirmation_groups_opened=0,table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='Offline parent geometry decomposition, not an executable oracle-center or scale policy')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();runtime.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage211_decision.json');check_code(p)
    if (d['schema']!=c.SCHEMA or d['status']!='REVIEW_RECOVERY_CENTER_SIZE_FACTORIAL' or d['plan_fingerprint']!=p['fingerprint']
            or d['source_stage210']!=c.SOURCE210 or d['source_groups']!=38 or d['new_training_updates']!=0 or d['new_full_trajectories']!=0
            or d['confirmation_groups_opened']!=0 or set(d['table_sha256'])!={f'stage211_{k}.csv' for k in c.TABLES}):raise ValueError('Incomplete Stage-211')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+name)
    print('Stage-211 verified; REVIEW, no training or adoption',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (208,210,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage210-bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--stage208-bundle',type=Path,default=base.previous.DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,210,*sources.SOURCES)):parser.error('Need Stage-208 models, complete Stage-210, checkpoint and dataset sources')
    r208=args.stage208_root.resolve();r210=args.stage210_root.resolve();paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES}
    runtime.common.source.common.protocol.safe_output(root,[r208,r210,args.checkpoint,*paths.values(),args.stage210_bundle.parent])
    with writer_lock(root):
        runtime.guard(root)
        if (root/'stage211_decision.json').exists():verify(root);return
        d,p,t=c.source210(args.stage210_bundle);base.verify(r210)
        if old.read_sealed(r210/'manifest.json')!=p or old.read_sealed(r210/'stage210_decision.json')!=d:raise ValueError('Changed physical Stage-210')
        import torch
        torch.set_num_threads(1)
        rt=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p['runtime']),torch,args.gpu)
        differences=sources.old.base.s109.check_runtime(p['runtime'],rt)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_paths={str(k):str(v) for k,v in paths.items()},
            source_root=str(r208),bundle=str(args.stage208_bundle.resolve()),diagnostic_root=str(r210),diagnostic_bundle=str(args.stage210_bundle.resolve()))
        entries,dataset=runtime.load_entries(config,p);runtime.common.source.common.protocol.safe_output(root,[dataset])
        bank={};verified={}
        for spec in p['entries']:bank[spec['sequence']]=endpoints(config,spec,p,d,t,verified)
        code=dict(p['code_sha256'])
        for f in (__file__,c.__file__,REPO/'tools/run_stage211_geometry_audit.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        plan=old.sealed(dict(schema=c.SCHEMA,source_stage210=c.SOURCE210,entries=p['entries'],folds=p['folds'],events=p['events'],selection=p['selection'],
            source_support=p['source_support'],runtime=rt,runtime_differences=differences,same_source_runtime=not differences,workers=args.workers,
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL),design='Same frozen matched model; center x width/height from fixed native/matched parent endpoints; no new support or GT actions'))
        publish_identical(root/'manifest.json',plan);native=runtime.open_runtime(config,plan,'stage211')
        try:
            models,_,scope,_,model_verified=base.previous.load_bound(config,native)
            if [{k:str(v) for k,v in r.items()} for r in scope]!=t['model_scope']:raise ValueError('Source-210 model identity changed')
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],endpoints=verified,models=model_verified)))
            experiment(root,plan,native,models,entries,dataset,bank,config,scope)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
