"""Fit-only next-search-state supervision; frozen SGLA/adapter, full video OOF."""
import argparse
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from pathlib import Path
import sys,traceback
from time import perf_counter
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage214_search_state_common as c
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv
previous=c.source.previous;runtime=previous.runtime;old=c.old
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage213_results_20261126T161130Z_2281744.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE214_LEARNED_SEARCH_STATE.md'


def rollout(native,adapter,arm,updater,files,initial,reader,dataset,progress,samples=None,truth=None):
    if arm not in ('nn','mm',*c.TRAINED):raise ValueError('Unknown state update')
    rt=native.rt;native.initialize(reader(dataset,files[0]),initial);parent=np.asarray(initial,float);rows=[];captured=[];selected=[];xx=[]
    counts=dict(block=np.zeros(12,int),head=0,selector=0)
    def block(i):
        def call(m,a,o):counts['block'][i]+=1
        return call
    def head(m,a,o):counts['head']+=1
    def selector(m,a,o):counts['selector']+=1
    with ExitStack() as stack:
        for i,b in enumerate(rt.backbone.blocks):stack.callback(b.register_forward_hook(block(i)).remove)
        stack.callback(rt.net.box_head.register_forward_hook(head).remove)
        stack.callback(rt.backbone.MLP.register_forward_hook(selector).remove)
        stack.callback(rt.backbone.blocks[5].register_forward_hook(lambda m,a,o:captured.append(o.detach().clone())).remove)
        for frame,file in enumerate(files[1:],1):
            captured.clear();counts['block'][:]=0;counts['head']=counts['selector']=0
            image=reader(dataset,file);namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target'];factors=[]
            def counted(*a,**kw):
                value=sampler(*a,**kw);factors.append(float(value[1]));return value
            runtime.previous.sync(rt);start=perf_counter()
            with patch.dict(namespace,sample_target=counted):
                raw=native.step(image,parent)
                if len(captured)!=1 or len(factors)!=1 or counts['head']!=1 or counts['selector']!=1:raise ValueError('Not one raw crop/prefix/head/selector')
                np.testing.assert_array_equal(raw['block_calls'],counts['block'])
                with rt.torch.no_grad():
                    out=rt.candidate(adapter.adapt(captured[0],None),raw['layer'])
                    adapted=previous.decode(rt,out,factors[0],parent,image)
                    x=c.features(captured[0].cpu().numpy(),rt.lens_z,parent,raw['box'],adapted)
                    z=np.zeros(3,float)
                    if arm in c.TRAINED:
                        if updater is None:raise ValueError('Missing learned updater')
                        z=updater(rt.torch.as_tensor(x[None],device=next(updater.parameters()).device)).cpu().numpy()[0].astype(float)
                state=(np.asarray(raw['box'],float).copy() if arm=='nn' else adapted.copy()) if arm in ('nn','mm') else c.parent(raw['box'],z,image.shape[:2])
                rt.tracker.state=state.tolist()
            runtime.previous.sync(rt);elapsed=1000*(perf_counter()-start)
            if len(captured)!=1 or len(factors)!=1 or rt.tracker.frame_id!=frame:raise ValueError('Duplicate crop/prefix/frame')
            row=dict(frame_id=frame,parent=parent.copy(),boxes=adapted,reference_boxes=raw['box'],next_parent=state,state_delta=z,
                image_hw=np.asarray(image.shape[:2]),factor=factors[0],layer=raw['layer'],block_calls=counts['block'].copy(),head_calls=counts['head'],
                selector_calls=counts['selector'],crop_calls=len(factors),updater_calls=int(arm in c.TRAINED),pipeline_ms=elapsed)
            # Evaluation labels are attached after the online action, never passed to updater.
            if truth is not None:
                gt=truth[frame]
                grid=old.protocol.dense_image_boxes(out['size_map'][0].cpu().numpy(),out['offset_map'][0].cpu().numpy(),parent,factors[0],rt.tracker.params.search_size,image.shape[:2])
                row.update(fully_covered=runtime.common.source.common.covered(gt,parent,factors[0],image.shape[:2],rt.tracker.params.search_size),
                    dense_best_iou=float(old.tracking_iou(grid,gt).max()),gt_side_network_px=float(np.sqrt(np.prod(gt[2:]))*factors[0]))
            rows.append(row)
            if samples is not None and frame in samples:selected.append(frame);xx.append(x)
            parent=state;progress.advance()
    result={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    if samples is not None:result.update(sample_frame=np.asarray(selected,int),sample_features=np.stack(xx))
    return result


def fit_bank(root,plan,native,adapters,fold,seed,entries,dataset,progress,artifacts):
    bank={};support=[]
    for g in fold['fit']:
        if g in fold['held']:raise ValueError('Held training source')
        spec=next(e for e in plan['entries'] if e['group']==g);seq=spec['sequence'];files,gt,_,reader,_=entries[seq]
        frames=c.protocol.uniform_frames(len(files)-2)
        valid=np.asarray([i for i in frames if np.isfinite(gt[i+1]).all() and (gt[i+1,2:]>0).all()],int)
        if len(valid)==0:raise ValueError('No valid next-frame training labels '+seq)
        pieces=[];model=adapters[fold['fold'],seed,'matched'];sha=old.model_digest(model)
        for mode in ('nn','mm'):
            path=root/'fit_rollouts'/f"f{fold['fold']}__s{seed}__{seq}__{mode}.npz"
            meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,group=g,mode=mode,fit=fold['fit'],held=fold['held'],samples=valid.tolist(),adapter_sha256=sha,role='outer_fit_only')
            v,reused=old.cache(path,meta,lambda:rollout(native,model,mode,None,files,gt[0],reader,dataset,progress,set(valid)))
            if reused:old.reused_work(progress,len(files)-1)
            c.audit(v,gt[0],len(files)-1,mode);np.testing.assert_array_equal(v['sample_frame'],valid);old.record(root,path,artifacts)
            idx=valid-1
            pieces.append(dict(x=v['sample_features'],raw=v['reference_boxes'][idx],target=gt[valid+1],image_hw=v['image_hw'][idx]))
            support.append(dict(fold=fold['fold'],seed=seed,sequence=seq,group=g,mode=mode,in_fit=True,in_held=False,rollout_frames=len(files)-1,
                candidate_samples=len(frames),valid_next_labels=len(valid),invalid_next_labels=len(frames)-len(valid),
                sampled_current_low_iou=int((old.tracking_iou(v['boxes'][idx],gt[valid])<.2).sum()),adapter_sha256=sha))
        bank[g]={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        if old.model_digest(model)!=sha:raise ValueError('Frozen fit adapter changed')
    if set(bank)!=set(fold['fit']) or set(bank)&set(fold['held']):raise ValueError('Wrong training roster')
    return bank,support


def train(root,plan,native,bank,fold,seed,arm,progress,artifacts):
    if set(bank)!=set(fold['fit']) or set(bank)&set(fold['held']):raise ValueError('Held source entered training')
    rt=native.rt;torch=rt.torch;device=next(rt.net.parameters()).device
    allx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=allx.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(allx.std(0,dtype=np.float64),1e-6).astype(np.float32)
    model=c.build(torch,allx.shape[1],mean,std,seed).to(device).eval();initial=old.model_digest(model)
    draws=c.protocol.draws_for(fold,seed,{g:len(bank[g]['x']) for g in fold['fit']})
    meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],initial_sha256=initial,
        draws_sha256=fingerprint(draws),updates=c.DESIGN['updates'],normalization_sha256=fingerprint(dict(mean=mean.tolist(),std=std.tolist())))
    path=root/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt";saved=old.load_model(torch,path,meta)
    if saved is not None:
        model.load_state_dict(saved['state'],strict=True)
        if old.model_digest(model)!=saved['scope']['state_sha256']:raise ValueError('Changed updater')
        old.record(root,path,artifacts);old.reused_work(progress,len(draws));return model.requires_grad_(False),saved['scope'],saved['history']
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.DESIGN['learning_rate'],weight_decay=c.DESIGN['weight_decay']);history=[];maximum=0.;start=perf_counter()
    factor=float(plan['search_factor'])
    for step,batch in enumerate(draws,1):
        values={k:torch.as_tensor(np.stack([bank[g][k][i] for g,i in batch]),device=device,dtype=torch.float32) for k in ('x','raw','target','image_hw')}
        optimizer.zero_grad(set_to_none=True);delta=model(values['x']);loss,reg,cover=c.loss(torch,delta,values['raw'],values['target'],values['image_hw'],factor,arm)
        if not bool(torch.isfinite(loss)):runtime.previous.nonfinite(root,meta,step)
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c.DESIGN['gradient_clip'],error_if_nonfinite=True));maximum=max(maximum,norm)
        if any(p.grad is not None for p in rt.net.parameters()):raise ValueError('Gradient reached SGLA')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):runtime.previous.nonfinite(root,meta,step)
        history.append(dict(fold=fold['fold'],seed=seed,arm=arm,step=step,loss=float(loss.detach()),regression=float(reg.detach()),coverage_penalty=float(cover.detach()),gradient_norm=norm))
        progress.advance()
    native.assert_frozen()
    scope=dict(fold=fold['fold'],seed=seed,arm=arm,fit_groups='|'.join(fold['fit']),held_groups='|'.join(fold['held']),updates=len(draws),examples=sum(len(b) for b in draws),
        trainable_parameters=sum(p.numel() for p in model.parameters()),feature_dimensions=allx.shape[1],initial_sha256=initial,state_sha256=old.model_digest(model),
        draws_sha256=meta['draws_sha256'],normalization_sha256=meta['normalization_sha256'],maximum_gradient_norm=maximum,
        coverage_active_batches=sum(r['coverage_penalty']>0 for r in history),frozen_native_unchanged=True,training_seconds=perf_counter()-start)
    if scope['state_sha256']==initial or maximum<=0:raise ValueError('Inactive learned state module')
    old.save_model(torch,path,dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},scope=scope,history=history),meta);old.record(root,path,artifacts)
    return model.requires_grad_(False),scope,history


def load_updaters(root,plan,torch,device):
    result={}
    for fold in plan['folds']:
        for seed in c.SEEDS:
            for arm in c.TRAINED:
                path=root/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt";receipt=old.read_sealed(path.with_suffix('.receipt.json'));meta=receipt['metadata']
                if any(meta[k]!=v for k,v in dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],updates=c.DESIGN['updates']).items()):raise ValueError('Wrong learned model binding')
                payload=old.load_model(torch,path,meta);state=payload['state'];dim=len(state['mean']);model=c.build(torch,dim,state['mean'],state['std'],seed).to(device).eval()
                model.load_state_dict(state,strict=True);model.requires_grad_(False)
                if old.model_digest(model)!=payload['scope']['state_sha256']:raise ValueError('Wrong learned state')
                result[fold['fold'],seed,arm]=model
    return result


def jobs(plan,spec,adapters,updaters):
    fold=next(f['fold'] for f in plan['folds'] if spec['group'] in f['held'])
    return fold,[('native',0,None,None)]+[(a,s,adapters[fold,s,'matched'],updaters.get((fold,s,a))) for s in c.SEEDS for a in ('nn','mm',*c.TRAINED)]


def path_for(root,phase,seq,seed,arm):return (root/'evaluation' if phase=='full' else root/('smoke_'+phase)/'evaluation')/f'{seq}__s{seed}__{arm}.npz'


def metadata(plan,seq,fold,seed,arm,adapter,updater,phase):
    return dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,phase=phase,
        adapter_sha256=None if adapter is None else old.model_digest(adapter),updater_sha256=None if updater is None else old.model_digest(updater))


def evaluate_video(root,plan,native,adapters,updaters,spec,entry,dataset,progress,phase):
    files,gt,_,reader,_=entry
    if phase!='full':files=files[:1+c.protocol.DESIGN['smoke_updates']]
    seq=spec['sequence'];fold,work=jobs(plan,spec,adapters,updaters)
    for arm,seed,adapter,updater in work:
        meta=metadata(plan,seq,fold,seed,arm,adapter,updater,phase);path=path_for(root,phase,seq,seed,arm)
        def compute():
            if arm=='native':return runtime.previous.trace(native,None,None,files,gt[0],reader,dataset,progress)
            return rollout(native,adapter,arm,updater,files,gt[0],reader,dataset,progress,truth=gt)
        v,reused=old.cache(path,meta,compute)
        if reused:old.reused_work(progress,len(files)-1)
        c.audit(v,gt[0],len(files)-1,arm)
        if adapter is not None and old.model_digest(adapter)!=meta['adapter_sha256']:raise ValueError('Frozen adapter changed')
        if updater is not None and old.model_digest(updater)!=meta['updater_sha256']:raise ValueError('Updater changed during inference')
    native.assert_frozen()


def check_code(p):
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Frozen dependency changed '+n)
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Protocol changed')


def worker_main(config,sequences,phase,worker,channel):
    root=Path(config['root']);log=root/'worker_logs'/f'{phase}_{worker}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as h,redirect_stdout(h),redirect_stderr(h):
        native=None
        try:
            plan=old.read_sealed(root/'manifest.json');check_code(plan);runtime.guard(root);native=runtime.open_runtime(config,plan,f'stage214_{phase}_{worker}')
            entries,dataset=runtime.load_entries(config,plan);adapters,_,_,_,_=previous.previous.load_bound(config,native)
            updaters=load_updaters(root,plan,native.rt.torch,next(native.rt.net.parameters()).device);progress=runtime.WorkerProgress(channel,worker)
            for seq in sequences:evaluate_video(root,plan,native,adapters,updaters,next(e for e in plan['entries'] if e['sequence']==seq),entries[seq],dataset,progress,phase)
            progress.flush();channel.put(('complete',worker,0,0))
        except BaseException:
            traceback.print_exc();channel.put(('error',worker,traceback.format_exc(),0));raise
        finally:
            if native is not None:
                for h in native.handles:h.remove()


def collect(root,plan,adapters,updaters,spec,entry,phase,artifacts):
    seq=spec['sequence'];fold,work=jobs(plan,spec,adapters,updaters);vv={};n=len(entry[0])-1 if phase=='full' else min(len(entry[0])-1,c.protocol.DESIGN['smoke_updates'])
    for arm,seed,a,u in work:
        path=path_for(root,phase,seq,seed,arm);v=previous.required(path,metadata(plan,seq,fold,seed,arm,a,u,phase));c.audit(v,entry[1][0],n,arm);old.record(root,path,artifacts);vv[arm,seed]=v
    return vv


def experiment(root,plan,native,adapters,frozen_scope,entries,dataset,config,source212):
    frames=sum(len(entries[e['sequence']][0])-1 for e in plan['entries']);specs=previous.previous.specs_for_smoke(plan)
    sf=sum(min(c.protocol.DESIGN['smoke_updates'],len(entries[e['sequence']][0])-1) for e in specs)
    progress=OverallProgress('Stage-214',4*frames+8*c.DESIGN['updates']+18*sf+9*frames+1,unit='work')
    tables={k:[] for k in c.TABLES};tables['frozen_scope']=frozen_scope;artifacts={};updaters={};frozen={k:old.model_digest(v) for k,v in adapters.items()}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    progress.report('Fit-only nn/mm provider rollouts and matched updater training')
    for fold in plan['folds']:
        for seed in c.SEEDS:
            bank,support=fit_bank(root,plan,native,adapters,fold,seed,entries,dataset,progress,artifacts);tables['fit_support'].extend(support);scopes=[]
            for arm in c.TRAINED:
                model,scope,history=train(root,plan,native,bank,fold,seed,arm,progress,artifacts);updaters[fold['fold'],seed,arm]=model
                tables['scope_summary'].append(scope);tables['training_history'].extend(history);scopes.append(scope)
            for k in ('initial_sha256','draws_sha256','normalization_sha256','examples','trainable_parameters'):
                if scopes[0][k]!=scopes[1][k]:raise ValueError('Unmatched training opportunity '+k)
    if {k:old.model_digest(v) for k,v in adapters.items()}!=frozen:raise ValueError('Adapter changed during training')
    progress.report('Serial/two-worker identical-state smoke')
    start=perf_counter()
    for spec in specs:evaluate_video(root,plan,native,adapters,updaters,spec,entries[spec['sequence']],dataset,progress,'serial')
    serial=perf_counter()-start;parallel=runtime.run_workers(config,[e['sequence'] for e in specs],'parallel',2,progress,target=worker_main)
    for spec in specs:
        aa=collect(root,plan,adapters,updaters,spec,entries[spec['sequence']],'serial',artifacts);bb=collect(root,plan,adapters,updaters,spec,entries[spec['sequence']],'parallel',artifacts)
        for arm,seed in aa:
            previous.previous.equal_quality(aa[arm,seed],bb[arm,seed]);tables['smoke'].append(dict(sequence=spec['sequence'],arm=arm,seed=seed,exact_state_output_calls=True,timing_excluded=True))
    progress.report('Complete OOF tracking; native and matched recovery protection')
    elapsed=runtime.run_workers(config,[e['sequence'] for e in plan['entries']],'full',plan['workers'],progress,target=worker_main)
    source_plan,source_decision,source_tables=source212;verified={}
    for spec in plan['entries']:
        seq=spec['sequence'];entry=entries[seq];vv=collect(root,plan,adapters,updaters,spec,entry,'full',artifacts);fold,_=jobs(plan,spec,adapters,updaters)
        ref=c.source.read_traces(Path(config['stage212_root']),source_plan,source_decision,source_tables,spec,entry[1],verified)
        for a,s in [('native',0)]+[(a,s) for s in c.SEEDS for a in ('nn','mm')]:
            for k in ref[a,s]:
                if k!='pipeline_ms':np.testing.assert_array_equal(vv[a,s][k],ref[a,s][k],err_msg='Fresh control differs from Stage-212 '+k)
            tables['source_replay'].append(dict(sequence=seq,arm=a,seed=s,exact_source_state_output_calls=True))
        native_trace=vv['native',0]
        for (arm,seed),v in vv.items():
            row=old.describe_trace(v,native_trace,entry[1],seq,arm,seed,fold);refs=c.SEEDS if arm=='native' else [seed]
            rr=[old.describe_trace(v,vv['mm',s],entry[1],seq,arm,seed,fold) for s in refs]
            row.update(missed_matched_recoveries=float(np.mean([r['missed_native_recoveries'] for r in rr])),matched_good_frame_loss=float(np.mean([r['native_good_frame_loss'] for r in rr])),introduced_matched_failure=any(r['introduced_failure'] for r in rr))
            tables['seed_video_summary'].append(row)
            tables['calls'].append(dict(sequence=seq,arm=arm,seed=seed,frames=len(v['boxes']),block_calls=int(v['block_calls'].sum()),head_calls=int(v['head_calls'].sum()),selector_calls=int(v['selector_calls'].sum()),crop_calls=int(v['crop_calls'].sum()),updater_calls=0 if arm=='native' else int(v['updater_calls'].sum())))
            if arm!='native':
                output=old.tracking_iou(v['boxes'],entry[1][1:]);covered=v['fully_covered'];grid=v['dense_best_iou']
                tables['search_geometry'].append(dict(sequence=seq,arm=arm,seed=seed,fully_covered_fraction=float(covered.mean()),mean_gt_side_network_px=float(v['gt_side_network_px'].mean()),
                    covered_no_good_grid_fraction=float((covered&(grid<.5)).mean()),good_grid_missed_fraction=float(((grid>=.5)&(output<.5)).mean()),
                    changed_from_raw_state_frames=int((v['next_parent']!=v['reference_boxes']).any(1).sum()),mean_parent_side=float(np.sqrt(v['parent'][:,2:].prod(1)).mean())))
    tables['policy_summary'],tables['paired']=c.summarize(tables['seed_video_summary'])
    tables['execution']=[dict(phase='serial_smoke',workers=1,wall_seconds=serial),dict(phase='parallel_smoke',workers=2,wall_seconds=parallel),dict(phase='full',workers=plan['workers'],wall_seconds=elapsed)]
    publish_identical(root/'evaluation_source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_arrays=verified)));artifacts['evaluation_source_receipt.json']=file_hash(root/'evaluation_source_receipt.json')
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage214_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete work budget')
    publish_identical(root/'stage214_decision.json',old.sealed(dict(schema=c.SCHEMA,status='REVIEW_LEARNED_SEARCH_STATE',plan_fingerprint=plan['fingerprint'],source_stage213=c.SOURCE213,
        source_groups=38,training_runs=len(tables['scope_summary']),training_updates=sum(r['updates'] for r in tables['scope_summary']),fit_provider_trajectories=len(tables['fit_support']),
        full_evaluation_trajectories=len(tables['seed_video_summary']),confirmation_groups_opened=0,primary_contrasts=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='Frozen SGLA and adapter; learned next search parent, adapted external output; no GT in online policy, no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)
    for r in tables['policy_summary']:print(f"{r['arm']:10s} AUC={r['auc']:.4f} low-IoU={r['low_iou_fraction']:.2%}",flush=True)


def verify(root):
    root=Path(root).resolve();runtime.guard(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage214_decision.json');check_code(p)
    if d['schema']!=c.SCHEMA or d['status']!='REVIEW_LEARNED_SEARCH_STATE' or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage213']!=c.SOURCE213 or (d['source_groups'],d['training_runs'],d['training_updates'],d['fit_provider_trajectories'],d['full_evaluation_trajectories'],d['confirmation_groups_opened'])!=(38,8,2048,152,342,0) or set(d['table_sha256'])!={f'stage214_{k}.csv' for k in c.TABLES}:raise ValueError('Incomplete Stage-214')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=h:raise ValueError('Changed artifact '+n)
    print('Stage-214 verified; REVIEW, no adoption or confirmation opening',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in (208,212,76,89,93,121,170,188,191):p.add_argument(f'--stage{k}-root',type=Path)
    p.add_argument('--stage213-bundle',type=Path,default=DEFAULT_BUNDLE);p.add_argument('--stage212-bundle',type=Path,default=c.source.DEFAULT_BUNDLE)
    p.add_argument('--stage208-bundle',type=Path,default=previous.previous.DEFAULT_BUNDLE);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-results',action='store_true')
    args=p.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (208,212,*sources.SOURCES)):p.error('Need original checkpoint, Stage-208 models, Stage-212 caches and seven source roots')
    paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES};c.protocol.safe_output(root,[args.checkpoint,args.stage208_root,args.stage212_root,args.stage213_bundle.parent,*paths.values()])
    with writer_lock(root):
        runtime.guard(root)
        if (root/'stage214_decision.json').exists():verify(root);return
        d213,p213,_=c.source213(args.stage213_bundle);d212,p212,t212=c.source.source212(args.stage212_bundle);previous.verify(args.stage212_root)
        if p213['source_plan']!=p212['fingerprint'] or p213['entries']!=p212['entries'] or p213['folds']!=p212['folds'] or old.read_sealed(args.stage212_root/'manifest.json')!=p212 or old.read_sealed(args.stage212_root/'stage212_decision.json')!=d212:raise ValueError('Physical source mismatch')
        for n,h in p213['code_sha256'].items():
            if file_hash(REPO/n)!=h:raise ValueError('Frozen source changed '+n)
        if file_hash(c.source.PROTOCOL)!=p213['protocol_sha256']:raise ValueError('Frozen Stage-213 protocol changed')
        import torch
        torch.set_num_threads(1);identity=sources.old.base.s103.s76.device_runtime(dict(runtime_identity=p213['source_runtime']),torch,args.gpu)
        sources.old.base.s109.check_runtime(p213['source_runtime'],identity,p213['source_runtime'])
        code=dict(p213['code_sha256'])
        for f in (__file__,c.__file__,REPO/'tools/run_stage214_search_state.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        # Resolve the source-bound runtime first; derive the real crop factor from its configuration below.
        provisional=dict(fingerprint=d213['fingerprint'],runtime=identity)
        config=dict(root=str(root),gpu=args.gpu,checkpoint=str(args.checkpoint.resolve()),source_root=str(args.stage208_root.resolve()),bundle=str(args.stage208_bundle.resolve()),
            stage212_root=str(args.stage212_root.resolve()),source_paths={str(k):str(v) for k,v in paths.items()})
        entries,dataset=runtime.load_entries(config,p212);c.protocol.safe_output(root,[dataset]);native=runtime.open_runtime(config,provisional,'stage214')
        try:
            factor=float(native.rt.tracker.params.search_factor)
            plan=old.sealed(dict(schema=c.SCHEMA,source_stage213=d213['fingerprint'],source_stage212=d212['fingerprint'],entries=p212['entries'],folds=p212['folds'],runtime=identity,
                workers=args.workers,search_factor=factor,design=c.DESIGN,code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
            publish_identical(root/'manifest.json',plan);adapters,_,scope,_,verified=previous.previous.load_bound(config,native)
            if [{k:str(v) for k,v in r.items()} for r in scope]!=t212['model_scope']:raise ValueError('Changed frozen adapter scope')
            publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],models=verified,stage213_bundle_sha256=file_hash(args.stage213_bundle))))
            experiment(root,plan,native,adapters,scope,entries,dataset,config,(p212,d212,t212))
        finally:
            for h in native.handles:h.remove()


if __name__=='__main__':main()
