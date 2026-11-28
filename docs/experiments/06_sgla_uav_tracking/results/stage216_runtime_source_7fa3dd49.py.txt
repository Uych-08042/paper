"""Frozen, common-state one-step geometry versus successor tracking quality."""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import csv, io, json, re, sys, traceback
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage215_same_model_replay as source
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
from experiment_progress import OverallProgress
old=source.old;runtime=source.runtime
SCHEMA='stage216_common_state_successor_quality_v1'
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage215_results_20261128T102259Z_26003.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE216_COMMON_STATE_SUCCESSOR.md'
ORIGINS=('coverage__none','mm','coverage__center','coverage__joint')
ACTIONS=tuple((f,m) for f in source.c.FAMILIES for m in source.c.MASKS)
TABLES=('selection','model_scope','source_replay','cells','policy_summary','paired','alignment','calls','smoke','execution')


def bundle(path):
    raw=Path(path).read_bytes()
    if file_hash(path)!='275f66a26c14458045ca5af41a1831ac2537242fb77223df19a0cf6566067f87':raise ValueError('Need reviewed Stage-215 bundle')
    marks=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():marks[i+1].start() if i+1<len(marks) else len(raw)] for i,m in enumerate(marks)}
    d,p=json.loads(parts['stage215_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed fingerprint')
    if len(marks)!=len(parts) or set(parts)!={'manifest.json','stage215_decision.json',*d['table_sha256']} or d['schema']!=source.SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['new_full_trajectories']!=722 or d['confirmation_groups_opened']!=0:raise ValueError('Wrong source scope')
    for name,sha in d['table_sha256'].items():
        import hashlib
        if not any(hashlib.sha256(b).hexdigest()==sha for b in (parts[name],parts[name][:-1],parts[name][:-2])):raise ValueError('Changed source table')
    return d,p


def selection(plan):
    return [dict(sequence=e['sequence'],group=e['group'],frame=int(t)) for e in plan['entries'] for t in source.c.protocol.uniform_frames(e['tracked_frames']-1)]


def capture(native,adapter,image,initial_image,initial,parent,frame):
    """One crop, raw selected candidate and same-layer adapted readout; no GT."""
    rt=native.rt;native.initialize(initial_image,initial);rt.tracker.frame_id=int(frame)-1
    tokens=[];factors=[];counts=np.zeros(12,int);heads=[];selectors=[]
    namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target']
    def sample(*a,**kw):
        value=sampler(*a,**kw);factors.append(float(value[1]));return value
    def block(i):
        def hook(*args):counts[i]+=1
        return hook
    with ExitStack() as stack:
        stack.enter_context(patch.dict(namespace,sample_target=sample))
        for i,b in enumerate(rt.backbone.blocks):stack.callback(b.register_forward_hook(block(i)).remove)
        stack.callback(rt.backbone.blocks[5].register_forward_hook(lambda m,a,o:tokens.append(o.detach().clone())).remove)
        stack.callback(rt.net.box_head.register_forward_hook(lambda *a:heads.append(1)).remove)
        stack.callback(rt.backbone.MLP.register_forward_hook(lambda *a:selectors.append(1)).remove)
        raw=native.step(image,parent)
        if len(tokens)!=1 or len(factors)!=1:raise ValueError('Not a single current prefix/crop')
        with rt.torch.no_grad():
            out=rt.candidate(adapter.adapt(tokens[0],None),raw['layer'])
            box=source.previous.previous.decode(rt,out,factors[0],parent,image)
            x=source.previous.c.features(tokens[0].cpu().numpy(),rt.lens_z,parent,raw['box'],box)
    expected=np.zeros(12,int);expected[:6]=1;expected[int(raw['layer'])-1]=2
    np.testing.assert_array_equal(counts,expected)
    if (len(heads),len(selectors))!=(2,1):raise ValueError('Call accounting changed')
    grid=old.protocol.dense_image_boxes(out['size_map'][0].cpu().numpy(),out['offset_map'][0].cpu().numpy(),parent,factors[0],rt.tracker.params.search_size,image.shape[:2])
    return dict(box=box,reference=raw['box'],x=x,grid=grid,layer=raw['layer'],factor=factors[0],image_hw=np.asarray(image.shape[:2]),blocks=counts)


def branches(native,adapter,models,current_image,next_image,initial_image,initial,parent,frame,progress):
    now=capture(native,adapter,current_image,initial_image,initial,parent,frame);progress.advance()
    torch=native.rt.torch
    with torch.no_grad():
        deltas=np.stack([models[f](torch.as_tensor(now['x'][None],device=next(models[f].parameters()).device)).cpu().numpy()[0].astype(float) for f in source.c.FAMILIES])
    candidates=[];values=[]
    for f,m in ACTIONS:
        z=deltas[source.c.FAMILIES.index(f)]*np.asarray(source.c.MASKS[m])
        candidate=source.previous.c.parent(now['reference'],z,now['image_hw'])
        nxt=capture(native,adapter,next_image,initial_image,initial,candidate,frame+1)
        candidates.append(candidate);values.append(nxt);progress.advance()
    return dict(current_box=now['box'],current_reference=now['reference'],current_layer=now['layer'],current_blocks=now['blocks'],features=now['x'],deltas=deltas,
        candidate_parent=np.stack(candidates),next_boxes=np.stack([v['box'] for v in values]),next_reference=np.stack([v['reference'] for v in values]),
        next_grid=np.stack([v['grid'] for v in values]),next_layer=np.asarray([v['layer'] for v in values]),next_factor=np.asarray([v['factor'] for v in values]),
        next_image_hw=np.stack([v['image_hw'] for v in values]),next_blocks=np.stack([v['blocks'] for v in values]))


def state_trace(config,plan,spec,seed,origin,verified):
    d,p=bundle(config['trace_bundle']);fold=next(f['fold'] for f in p['folds'] if spec['group'] in f['held'])
    # Require the exact original metadata, including frozen model hashes.
    path=source.previous.path_for(Path(config['trace_root']),'full',spec['sequence'],seed,origin)
    receipt=old.read_sealed(path.with_suffix('.receipt.json'));meta=receipt['metadata']
    if any(meta[k]!=v for k,v in dict(plan_fingerprint=p['fingerprint'],sequence=spec['sequence'],fold=fold,seed=seed,arm=origin,phase='full').items()):raise ValueError('Wrong source trace identity')
    value=runtime.common.bound_arrays(Path(config['trace_root']),d,str(path.relative_to(config['trace_root'])).replace('\\','/'),meta,verified)
    return value


def audit_sequence(root,plan,native,adapters,updaters,spec,entry,dataset,config,progress,phase):
    files,gt,_,reader,_=entry;seq=spec['sequence'];fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    frames=[r['frame'] for r in plan['selection'] if r['sequence']==seq]
    if phase=='smoke':frames=frames[:1]
    results=[];verified={}
    initial_image=reader(dataset,files[0])
    for seed in source.c.SEEDS:
        adapter=adapters[fold,seed,'matched'];models={f:updaters[fold,seed,f] for f in source.c.FAMILIES}
        for origin in ORIGINS:
            trace=state_trace(config,plan,spec,seed,origin,verified)
            metadata=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,origin=origin,seed=seed,frames=frames,adapter_sha256=old.model_digest(adapter),updaters={f:old.model_digest(m) for f,m in models.items()})
            path=root/phase/f'{seq}__s{seed}__{origin}.npz'
            def compute():
                pieces=[branches(native,adapter,models,reader(dataset,files[t]),reader(dataset,files[t+1]),initial_image,gt[0],trace['parent'][t-1],t,progress) for t in frames]
                return {k:np.asarray([v[k] for v in pieces]) for k in pieces[0]}
            data,reused=old.cache(path,metadata,compute)
            if reused:old.reused_work(progress,9*len(frames))
            for i,t in enumerate(frames):
                for name,key in (('current_box','boxes'),('current_reference','reference_boxes'),('current_layer','layer'),('current_blocks','block_calls')):np.testing.assert_array_equal(data[name][i],trace[key][t-1],err_msg='Current source replay differs '+name)
                f,m=('coverage','none') if origin=='coverage__none' else ('coverage',origin.split('__')[1]) if origin!='mm' else (None,None)
                if f is not None:
                    j=ACTIONS.index((f,m));np.testing.assert_array_equal(data['candidate_parent'][i,j],trace['next_parent'][t-1])
                    np.testing.assert_array_equal(data['next_boxes'][i,j],trace['boxes'][t],err_msg='Successor endpoint differs')
                    np.testing.assert_array_equal(data['next_layer'][i,j],trace['layer'][t])
                    np.testing.assert_equal(float(old.tracking_iou(data['next_grid'][i,j],gt[t+1]).max()),trace['dense_best_iou'][t],err_msg='Successor dense diagnostic differs')
            if old.model_digest(adapter)!=metadata['adapter_sha256'] or any(old.model_digest(m)!=metadata['updaters'][f] for f,m in models.items()):raise ValueError('Frozen weights changed')
            results.append((path,seed,origin,frames,data))
    native.assert_frozen();return results,verified


def check_code(plan):
    for n,h in plan['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Changed frozen dependency '+n)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def worker(config,sequences,phase,index,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{index}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);native=runtime.open_runtime(config,plan,'stage216_'+str(index))
            entries,dataset=runtime.load_entries(config,plan);adapters,_,_,_,_=source.previous.previous.previous.load_bound(config,native);updaters,*_=source.bound_models(config,native)
            progress=runtime.WorkerProgress(channel,index)
            for seq in sequences:audit_sequence(root,plan,native,adapters,updaters,next(e for e in plan['entries'] if e['sequence']==seq),entries[seq],dataset,config,progress,'full')
            progress.flush();channel.put(('complete',index,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',index,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for handle in native.handles:handle.remove()


def measures(v,i,j,truth):
    parent=v['candidate_parent'][i,j];target=np.asarray(truth,float);side=np.sqrt(np.prod(target[2:]));center=lambda b:b[:2]+b[2:]/2
    geom=float(np.mean(np.abs((center(parent)-center(target))/side)))+abs(float(np.log(np.sqrt(np.prod(parent[2:]))/side)))
    factor=float(v['next_factor'][i,j]);hw=v['next_image_hw'][i,j]
    return dict(geometry_error=geom,center_error=float(np.linalg.norm(center(parent)-center(target))/side),log_side_error=abs(float(np.log(np.sqrt(np.prod(parent[2:]))/side))),
        fully_covered=runtime.common.source.common.covered(target,parent,factor,hw,256),gt_side_network_px=float(side*factor),
        dense_best_iou=float(old.tracking_iou(v['next_grid'][i,j],target).max()),output_iou=float(old.tracking_iou(v['next_boxes'][i,j],target)))


def summarize(cells):
    groups=sorted({r['group'] for r in cells});means={};policy=[];pairs=[];align=[];index={}
    for r in cells:index.setdefault((r['origin'],r['family'],r['mask'],r['group']),[]).append(r)
    for o in ORIGINS:
        for f,m in ACTIONS:
            for g in groups:
                rr=index[o,f,m,g]
                means[o,f,m,g]={k:float(np.mean([r[k] for r in rr])) for k in ('output_iou','geometry_error','fully_covered','gt_side_network_px','dense_best_iou','center_error','log_side_error')}
            policy.append(dict(origin=o,family=f,mask=m,groups=len(groups),**{k:float(np.mean([means[o,f,m,g][k] for g in groups])) for k in means[o,f,m,groups[0]]}))
            if m=='none':continue
            stats=[]
            for g in groups:
                rr=index[o,f,m,g]
                good=[r for r in rr if r['geometry_gain']>1e-9]
                stats.append(dict(group=g,geometry_improved_fraction=len(good)/len(rr),geometry_improved_output_worse_fraction=sum(r['output_gain']< -1e-9 for r in good)/len(rr),conditional_output_gain=float(np.mean([r['output_gain'] for r in good])) if good else None))
            supported=[s['conditional_output_gain'] for s in stats if s['conditional_output_gain'] is not None]
            align.append(dict(origin=o,family=f,mask=m,groups=len(groups),geometry_improved_fraction=float(np.mean([s['geometry_improved_fraction'] for s in stats])),geometry_improved_output_worse_fraction=float(np.mean([s['geometry_improved_output_worse_fraction'] for s in stats])),conditional_supported_groups=len(supported),conditional_output_gain=float(np.mean(supported)) if supported else ''))
            for metric in ('output_iou','geometry_error'):
                pairs.append(dict(origin=o,family=f,mask=m,metric=metric,primary=False,**source.c.protocol.contrast([means[o,f,m,g][metric]-means[o,f,'none',g][metric] for g in groups])))
    pairs.append(dict(origin='provider_style_mean',family='coverage',mask='scale',metric='output_iou',primary=True,**source.c.protocol.contrast([np.mean([means[o,'coverage','scale',g]['output_iou']-means[o,'coverage','none',g]['output_iou'] for o in ORIGINS[:2]]) for g in groups])))
    return policy,pairs,align


def experiment(root,plan,native,adapters,updaters,entries,dataset,config):
    specs=source.previous.previous.previous.specs_for_smoke(plan);total=72*(len(plan['selection'])+len(specs))+1
    progress=OverallProgress('Stage-216',total,unit='searches');artifacts={};tables={k:[] for k in TABLES};tables['selection']=plan['selection']
    tables['model_scope']=[dict(fold=f,seed=s,family=a,state_sha256=old.model_digest(m),new_training_updates=0) for (f,s,a),m in updaters.items()]
    tables['model_scope'] += [dict(fold=f,seed=s,family='adapter',state_sha256=old.model_digest(m),new_training_updates=0) for (f,s,a),m in adapters.items()]
    serial={};progress.report('Current-state replay and serial reference');start=perf_counter()
    for spec in specs:serial[spec['sequence']]=audit_sequence(root,plan,native,adapters,updaters,spec,entries[spec['sequence']],dataset,config,progress,'smoke')[0]
    serial_time=perf_counter()-start
    progress.report('All sources: matched current state and independent successor inputs')
    elapsed=runtime.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker)
    progress.report('Checking source endpoints and summarizing successor quality')
    verified={}
    with patch.object(native,'step',side_effect=AssertionError('Master must not fill missing worker caches')):
        for spec in plan['entries']:
            results,refs=load_results(root,plan,spec,adapters,updaters,config)
            verified.update(refs);gt=entries[spec['sequence']][1]
            for path,seed,origin,frames,v in results:
                old.record(root,path,artifacts)
                if spec['sequence'] in serial:
                    _,_,_,ff,aa=next(r for r in serial[spec['sequence']] if r[1:3]==(seed,origin))
                    for k in aa:np.testing.assert_array_equal(aa[k],v[k][:len(ff)])
                    tables['smoke'].append(dict(sequence=spec['sequence'],seed=seed,origin=origin,all_arrays_exact=True))
                for i,t in enumerate(frames):
                    tables['source_replay'].append(dict(sequence=spec['sequence'],seed=seed,origin=origin,frame=t,current_exact=True,successor_checked=origin!='mm',successor_exact=True if origin!='mm' else ''))
                    tables['calls'].append(dict(sequence=spec['sequence'],seed=seed,origin=origin,frame=t,search_calls=9,block_calls=int(v['current_blocks'][i].sum()+v['next_blocks'][i].sum()),head_calls=18,selector_calls=9,crop_calls=9,updater_calls=2))
                    for j,(f,m) in enumerate(ACTIONS):
                        value=measures(v,i,j,gt[t+1]);base=measures(v,i,ACTIONS.index((f,'none')),gt[t+1])
                        tables['cells'].append(dict(sequence=spec['sequence'],group=spec['group'],seed=seed,origin=origin,frame=t,family=f,mask=m,**value,geometry_gain=base['geometry_error']-value['geometry_error'],output_gain=value['output_iou']-base['output_iou']))
    for values in serial.values():
        for path,*_ in values:old.record(root,path,artifacts)
    tables['policy_summary'],tables['paired'],tables['alignment']=summarize(tables['cells'])
    tables['execution']=[dict(phase='serial_smoke',wall_seconds=serial_time,workers=1),dict(phase='full',wall_seconds=elapsed,workers=plan['workers'])]
    publish_identical(root/'source_arrays.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_arrays=verified)));artifacts['source_arrays.json']=file_hash(root/'source_arrays.json')
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage216_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete search budget')
    publish_identical(root/'stage216_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_COMMON_STATE_SUCCESSOR',plan_fingerprint=plan['fingerprint'],source_stage215=plan['source_stage215'],new_training_updates=0,new_full_trajectories=0,confirmation_groups_opened=0,states=len(tables['calls']),search_calls=9*len(tables['calls']),primary=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts)))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def load_results(root,plan,spec,adapters,updaters,config):
    """Load only with exact expected metadata; never recompute or add progress."""
    frames=[r['frame'] for r in plan['selection'] if r['sequence']==spec['sequence']];results=[];verified={}
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    for seed in source.c.SEEDS:
        for origin in ORIGINS:
            state_trace(config,plan,spec,seed,origin,verified)
            meta=dict(plan_fingerprint=plan['fingerprint'],sequence=spec['sequence'],origin=origin,seed=seed,frames=frames,adapter_sha256=old.model_digest(adapters[fold,seed,'matched']),updaters={f:old.model_digest(updaters[fold,seed,f]) for f in source.c.FAMILIES})
            path=root/'full'/f"{spec['sequence']}__s{seed}__{origin}.npz"
            value=source.previous.previous.required(path,meta)
            results.append((path,seed,origin,frames,value))
    return results,verified


def verify(root):
    root=Path(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage216_decision.json');check_code(p)
    if p['schema']!=SCHEMA or d['schema']!=SCHEMA or d['status']!='REVIEW_COMMON_STATE_SUCCESSOR' or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage215']!=p['source_stage215'] or d['states']!=8*len(p['selection']) or d['search_calls']!=9*d['states'] or len(d['primary'])!=1 or set(d['table_sha256'])!={f'stage216_{k}.csv' for k in TABLES} or any(d[k]!=0 for k in ('new_training_updates','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Incomplete result')
    b=p['runtime_binding']
    if source.bind_runtime(b['source_runtime'],p['runtime'],b['bound_runtime'])!=b:raise ValueError('Changed runtime binding')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=h:raise ValueError('Changed artifact '+n)
    print('Stage-216 verified: fixed-state next-frame diagnostic, not full-video AUC',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (208,214,215,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage215-bundle',type=Path,default=DEFAULT_BUNDLE);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');args=parser.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,214,215,*sources.SOURCES)):parser.error('Need checkpoint and all source roots')
    source.c.protocol.safe_output(root,[args.checkpoint,args.stage215_bundle.parent,*[getattr(args,f'stage{k}_root') for k in (208,214,215,*sources.SOURCES)]])
    with writer_lock(root):
        runtime.guard(root)
        if not (root/'manifest.json').exists() and any(f.name!='collect.lock' for f in root.iterdir()):raise ValueError('Use an empty independent STAGE216_ROOT')
        if (root/'stage216_decision.json').exists():verify(root);return
        d,p=bundle(args.stage215_bundle);source.verify(args.stage215_root)
        if old.read_sealed(args.stage215_root/'manifest.json')!=p or old.read_sealed(args.stage215_root/'stage215_decision.json')!=d:raise ValueError('Wrong physical source')
        import torch
        torch.set_num_threads(1);actual=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p['runtime']),torch,args.gpu)
        existing=old.read_sealed(root/'manifest.json') if (root/'manifest.json').exists() else None
        binding=source.bind_runtime(p['runtime'],actual,None if existing is None else existing['runtime'])
        code=dict(p['code_sha256']);code.update({Path(n).relative_to(REPO).as_posix():file_hash(n) for n in (Path(__file__).resolve(),REPO/'tools/run_stage216_successor_quality.sh')})
        plan=old.sealed(dict(schema=SCHEMA,source_stage215=d['fingerprint'],entries=p['entries'],folds=p['folds'],runtime=actual,runtime_binding=binding,search_factor=p['search_factor'],workers=args.workers,selection=selection(p),code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_root=str(args.stage208_root.resolve()),bundle=str(source.previous.previous.previous.DEFAULT_BUNDLE),state_root=str(args.stage214_root.resolve()),state_bundle=str(source.DEFAULT_BUNDLE),trace_root=str(args.stage215_root.resolve()),trace_bundle=str(args.stage215_bundle.resolve()),source_paths={str(k):str(getattr(args,f'stage{k}_root').resolve()) for k in sources.SOURCES})
        entries,dataset=runtime.load_entries(config,plan);source.c.protocol.safe_output(root,[dataset]);native=runtime.open_runtime(config,plan,'stage216')
        try:
            adapters,_,_,_,_=source.previous.previous.previous.load_bound(config,native);updaters,*_=source.bound_models(config,native)
            experiment(root,plan,native,adapters,updaters,entries,dataset,config)
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
