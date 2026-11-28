"""CPU-only cached original-objective audit; no images, CUDA or optimization."""
import argparse, ast, hashlib, json, re, sys
from pathlib import Path
from time import perf_counter
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage216_successor_quality as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv
from experiment_progress import OverallProgress
old=prior.old;c=prior.source.previous.c;runtime=prior.runtime
SCHEMA='stage217_cached_predictor_objective_audit_v1'
RESULTS=REPO/'docs/experiments/06_sgla_uav_tracking/results'
BUNDLE216=RESULTS/'stage216_results_20261128T112342Z_288263.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE217_PREDICTOR_OBJECTIVE_AUDIT.md'
TABLES=('support','model_scope','constants','normalization','group_metrics','policy_summary','paired','prediction_replay','execution')


def source216(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!='14bce8f87a15b4a1990ff487f90467d38134e3ff4d50a1fd860d2061a8410264':raise ValueError('Need reviewed Stage-216 export; content differs')
    marks=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw));parts={m[1].decode():raw[m.end():marks[i+1].start() if i+1<len(marks) else len(raw)] for i,m in enumerate(marks)}
    d,p=json.loads(parts['stage216_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed fingerprint')
    if len(marks)!=len(parts) or set(parts)!={'manifest.json','stage216_decision.json',*d['table_sha256']} or d['schema']!=prior.SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['states']!=14592 or d['new_training_updates']!=0:raise ValueError('Wrong source scope')
    for n,h in d['table_sha256'].items():
        b=parts[n].replace(b'\r\n',b'\n')
        if not any(hashlib.sha256(v).hexdigest()==h for chunk in (b,b[:-1],b[:-2]) for v in (chunk,chunk.replace(b'\n',b'\r\n'))):raise ValueError('Changed source table')
    return d,p


def check_source_code(p):
    compatibility=[]
    for n,h in p['code_sha256'].items():
        data=(REPO/n).read_bytes()
        if h in (hashlib.sha256(data).hexdigest(),hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()):continue
        if n!='tools/stage216_successor_quality.py':raise ValueError('Frozen dependency changed '+n)
        saved=RESULTS/'stage216_runtime_source_7fa3dd49.py.txt';original=saved.read_bytes()
        if hashlib.sha256(original).hexdigest()!=h or h!='7fa3dd495dbeb17ad7a4eb1719bb9360d9c3e15545124ab7b7e494a0f2b17bbc':raise ValueError('Unrecognized source reader revision')
        def computation(b):
            tree=ast.parse(b)
            tree.body=[f for f in tree.body if not (isinstance(f,ast.FunctionDef) and f.name=='bundle')]
            for node in tree.body:
                if isinstance(node,ast.Import):node.names=[v for v in node.names if v.name!='hashlib']
            tree.body=[v for v in tree.body if not (isinstance(v,ast.Import) and not v.names)]
            return ast.dump(tree,include_attributes=False)
        if computation(original)!=computation(data):raise ValueError('Source computation changed')
        compatibility.append(dict(path=n,source_sha256=h,current_sha256=file_hash(REPO/n),snapshot_sha256=file_hash(saved),permitted_difference='bundle reader only'))
    return compatibility


def bound(root,decision,rel,metadata,verified):
    return runtime.common.bound_arrays(Path(root),decision,rel,metadata,verified)


def label(raw,target):
    side=np.sqrt(raw[:,2:].prod(1));rc=raw[:,:2]+raw[:,2:]/2;tc=target[:,:2]+target[:,2:]/2
    return np.c_[(tc-rc)/side[:,None],np.log(np.sqrt(target[:,2:].prod(1))/side)]


def constant_for_fit(bank,fit):
    if set(bank)!=set(fit):raise ValueError('Constant may use exactly fit groups only')
    return np.mean([label(bank[g]['raw'],bank[g]['target']).mean(0) for g in fit],axis=0).astype(np.float32)


def evaluate(torch,delta,data,family,factor):
    args=[torch.as_tensor(data[k],dtype=torch.float32) for k in ('raw','target','image_hw')]
    with torch.no_grad():
        total,reg,cover=c.loss(torch,torch.as_tensor(delta,dtype=torch.float32),*args,factor,family)
    values=dict(total=float(total),regression=float(reg),coverage=float(cover))
    if not all(np.isfinite(list(values.values()))):raise ValueError('Nonfinite audit loss')
    return values


def loss_contrast(values):
    values=np.asarray(values,float)
    result=prior.source.c.protocol.contrast(values)
    # Loss decreases are favorable: the worst deletion removes the BEST source.
    result['worst_leave_one_video_out']=float((values.sum()-values.min())/(len(values)-1))
    return result


def summarize(rows):
    contexts=sorted({(r['split'],r['origin'],r['family']) for r in rows});summary=[];paired=[];index={}
    for split,origin,family in contexts:
        subset=[r for r in rows if (r['split'],r['origin'],r['family'])==(split,origin,family)];groups=sorted({r['group'] for r in subset})
        for method in ('mlp','zero','fit_constant'):
            for g in groups:
                rr=[r for r in subset if r['method']==method and r['group']==g]
                if sorted(r['seed'] for r in rr)!=[2132,2133]:raise ValueError('Incomplete seed grid')
                index[split,origin,family,method,g]={k:float(np.mean([r[k] for r in rr])) for k in ('total','regression','coverage')}
            summary.append(dict(split=split,origin=origin,family=family,method=method,groups=len(groups),**{k:float(np.mean([index[split,origin,family,method,g][k] for g in groups])) for k in ('total','regression','coverage')}))
        for other in ('zero','fit_constant'):
            for metric in ('total','regression','coverage'):
                paired.append(dict(split=split,origin=origin,family=family,contrast='mlp_minus_'+other,metric=metric,primary=False,favorable_direction='negative',**loss_contrast([index[split,origin,family,'mlp',g][metric]-index[split,origin,family,other,g][metric] for g in groups])))
    groups=sorted({r['group'] for r in rows if r['split']=='held'})
    paired.append(dict(split='held',origin='provider_style_mean',family='forecast',contrast='mlp_minus_zero',metric='regression',primary=True,favorable_direction='negative',**loss_contrast([np.mean([index['held',o,'forecast','mlp',g]['regression']-index['held',o,'forecast','zero',g]['regression'] for o in prior.ORIGINS[:2]]) for g in groups])))
    return summary,paired


def audit(root,plan,config,entries,p214,d214,t214,p215,d215,p216,d216,torch):
    tables={k:[] for k in TABLES};verified={'stage214':{},'stage215':{},'stage216':{}};start=perf_counter()
    progress=OverallProgress('Stage-217',sum(len(f['fit'])*2+len(f['held'])*4+3 for f in p214['folds'])*2+1,unit='checks')
    progress.report('CPU cached fit/held objective audit')
    for fold in p214['folds']:
        for seed in c.SEEDS:
            bank={};fit_items=[]
            for g in fold['fit']:
                spec=next(e for e in plan['entries'] if e['group']==g);seq=spec['sequence'];gt=entries[seq][1];frames=prior.source.c.protocol.uniform_frames(len(gt)-2)
                valid=[int(t) for t in frames if np.isfinite(gt[t+1]).all() and (gt[t+1,2:]>0).all()]
                adapter=next(r['state_sha256'] for r in t214['frozen_scope'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed)
                pieces=[]
                for origin in ('nn','mm'):
                    rel=f"fit_rollouts/f{fold['fold']}__s{seed}__{seq}__{origin}.npz"
                    meta=dict(plan_fingerprint=p214['fingerprint'],fold=fold['fold'],seed=seed,group=g,mode=origin,fit=fold['fit'],held=fold['held'],samples=valid,adapter_sha256=adapter,role='outer_fit_only')
                    v=bound(config['stage214_root'],d214,rel,meta,verified['stage214']);np.testing.assert_array_equal(v['sample_frame'],valid);idx=np.asarray(valid)-1
                    data=dict(x=v['sample_features'],raw=v['reference_boxes'][idx],target=gt[np.asarray(valid)+1],image_hw=v['image_hw'][idx]);pieces.append(data);fit_items.append((g,origin,data))
                    tables['support'].append(dict(fold=fold['fold'],seed=seed,group=g,split='fit',origin=origin,samples=len(valid),target_frame_offset=1));progress.advance()
                bank[g]={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
            const=constant_for_fit(bank,fold['fit']);allx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=allx.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(allx.std(0,dtype=np.float64),1e-6).astype(np.float32)
            tables['constants'].append(dict(fold=fold['fold'],seed=seed,fit_groups='|'.join(fold['fit']),dx=float(const[0]),dy=float(const[1]),log_side=float(const[2]),rule='equal_fit_group_mean_label_delta'))
            models={}
            for family in c.TRAINED:
                path=Path(config['stage214_root'])/'training'/f"f{fold['fold']}__s{seed}__{family}.pt"
                for pp in (path,path.with_suffix('.receipt.json')):runtime.common.source.source_file(Path(config['stage214_root']),d214,pp.relative_to(config['stage214_root']).as_posix(),verified['stage214'])
                receipt=old.read_sealed(path.with_suffix('.receipt.json'));meta=receipt['metadata']
                if any(meta[k]!=v for k,v in dict(plan_fingerprint=p214['fingerprint'],fold=fold['fold'],seed=seed,arm=family,fit=fold['fit'],held=fold['held'],updates=256).items()):raise ValueError('Wrong model split identity')
                payload=old.load_model(torch,path,meta);model=c.build(torch,allx.shape[1],mean,std,seed).eval().requires_grad_(False);model.load_state_dict(payload['state'],strict=True)
                np.testing.assert_array_equal(model.mean.numpy(),mean);np.testing.assert_array_equal(model.std.numpy(),std)
                draws=prior.source.c.protocol.draws_for(fold,seed,{g:len(bank[g]['x']) for g in fold['fit']})
                if meta['normalization_sha256']!=fingerprint(dict(mean=mean.tolist(),std=std.tolist())) or meta['draws_sha256']!=fingerprint(draws):raise ValueError('Fit normalization/draws differ')
                expected=next(r['state_sha256'] for r in t214['scope_summary'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed and r['arm']==family)
                if old.model_digest(model)!=expected:raise ValueError('Changed frozen predictor')
                models[family]=model;tables['model_scope'].append(dict(fold=fold['fold'],seed=seed,family=family,state_sha256=expected,updates=0,device='cpu'));progress.advance()
            tables['normalization'].append(dict(fold=fold['fold'],seed=seed,full_fit_samples=len(allx),dimensions=allx.shape[1],stored_buffers_exact=True,draws_exact=True));progress.advance()
            def collect(g,split,origin,data,cached=None):
                for family,model in models.items():
                    with torch.no_grad():prediction=model(torch.as_tensor(data['x'],dtype=torch.float32)).numpy()
                    if cached is not None:
                        ref=cached[:,c.TRAINED.index(family)]
                        tables['prediction_replay'].append(dict(fold=fold['fold'],seed=seed,group=g,origin=origin,family=family,cpu_gpu_max_abs_delta=float(np.max(np.abs(prediction-ref))),note='measured CPU/GPU arithmetic drift; not a bitwise gate'))
                    for method,delta in (('mlp',prediction),('zero',np.zeros_like(prediction)),('fit_constant',np.broadcast_to(const,prediction.shape).copy())):
                        tables['group_metrics'].append(dict(fold=fold['fold'],seed=seed,group=g,split=split,origin=origin,family=family,method=method,samples=len(prediction),**evaluate(torch,delta,data,family,plan['search_factor'])))
            for g,origin,data in fit_items:collect(g,'fit',origin,data)
            for g in fold['held']:
                spec=next(e for e in plan['entries'] if e['group']==g);seq=spec['sequence'];gt=entries[seq][1]
                frames=[r['frame'] for r in p216['selection'] if r['sequence']==seq]
                for origin in prior.ORIGINS:
                    rel=f'full/{seq}__s{seed}__{origin}.npz';path=Path(config['stage216_root'])/rel;meta=old.read_sealed(path.with_suffix('.receipt.json'))['metadata']
                    if any(meta[k]!=v for k,v in dict(plan_fingerprint=p216['fingerprint'],sequence=seq,origin=origin,seed=seed,frames=frames).items()):raise ValueError('Wrong held sample identity')
                    if meta['updaters']!={f:old.model_digest(m) for f,m in models.items()}:raise ValueError('Held predictor binding differs')
                    v=bound(config['stage216_root'],d216,rel,meta,verified['stage216'])
                    # The original objective clips the next parent using CURRENT
                    # image dimensions. Do not silently substitute successor HW.
                    trace_path=prior.source.previous.path_for(Path(config['stage215_root']),'full',seq,seed,origin)
                    trace_meta=old.read_sealed(trace_path.with_suffix('.receipt.json'))['metadata']
                    if any(trace_meta[k]!=w for k,w in dict(plan_fingerprint=p215['fingerprint'],sequence=seq,fold=fold['fold'],seed=seed,arm=origin,phase='full').items()):raise ValueError('Wrong current-state trace binding')
                    trace=bound(config['stage215_root'],d215,trace_path.relative_to(config['stage215_root']).as_posix(),trace_meta,verified['stage215']);idx=np.asarray(frames)-1
                    np.testing.assert_array_equal(v['current_reference'],trace['reference_boxes'][idx])
                    np.testing.assert_array_equal(v['current_box'],trace['boxes'][idx])
                    for j in range(len(prior.ACTIONS)):np.testing.assert_array_equal(v['next_image_hw'][:,j],trace['image_hw'][idx+1])
                    data=dict(x=v['features'],raw=v['current_reference'],target=gt[np.asarray(frames)+1],image_hw=trace['image_hw'][idx])
                    collect(g,'held',origin,data,v['deltas']);tables['support'].append(dict(fold=fold['fold'],seed=seed,group=g,split='held',origin=origin,samples=len(frames),target_frame_offset=1));progress.advance()
            for family,model in models.items():
                if old.model_digest(model)!=next(r['state_sha256'] for r in tables['model_scope'] if r['fold']==fold['fold'] and r['seed']==seed and r['family']==family):raise ValueError('Model changed during audit')
    tables['policy_summary'],tables['paired']=summarize(tables['group_metrics']);tables['execution']=[dict(device='cpu',wall_seconds=perf_counter()-start,image_reads=0,cuda_calls=0,optimizer_updates=0)]
    artifacts={};publish_identical(root/'source_arrays.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_files=verified)));artifacts['source_arrays.json']=file_hash(root/'source_arrays.json')
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage217_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete cache audit budget')
    publish_identical(root/'stage217_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_CACHED_PREDICTOR_OBJECTIVE',plan_fingerprint=plan['fingerprint'],new_training_updates=0,new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,primary=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts)))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage217_decision.json');check_code(p)
    if p['schema']!=SCHEMA or d['schema']!=SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['status']!='REVIEW_CACHED_PREDICTOR_OBJECTIVE' or len(d['primary'])!=1 or set(d['table_sha256'])!={f'stage217_{n}.csv' for n in TABLES} or any(d[k]!=0 for k in ('new_training_updates','new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Incomplete audit')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=h:raise ValueError('Changed artifact '+n)
    print('Stage-217 verified: CPU original-objective audit, not tracking AUC',flush=True)


def check_code(p):
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Changed dependency '+n)
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Changed protocol')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (214,215,216,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage216-bundle',type=Path,default=BUNDLE216);parser.add_argument('--stage215-bundle',type=Path,default=prior.DEFAULT_BUNDLE);parser.add_argument('--stage214-bundle',type=Path,default=prior.source.DEFAULT_BUNDLE);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');args=parser.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if any(getattr(args,f'stage{k}_root') is None for k in (214,215,216,*sources.SOURCES)):parser.error('Need cached sources; no checkpoint or GPU is required')
    prior.source.c.protocol.safe_output(root,[getattr(args,f'stage{k}_root') for k in (214,215,216,*sources.SOURCES)]+[args.stage216_bundle.parent,args.stage215_bundle.parent,args.stage214_bundle.parent,RESULTS])
    with writer_lock(root):
        if (root/'stage217_decision.json').exists():verify(root);return
        d216,p216=source216(args.stage216_bundle);compatibility=check_source_code(p216)
        if file_hash(prior.PROTOCOL)!=p216['protocol_sha256']:raise ValueError('Source protocol changed')
        d215,p215=prior.bundle(args.stage215_bundle)
        d214,p214,t214=prior.source.c.source214(args.stage214_bundle)
        for k,d,p in ((214,d214,p214),(215,d215,p215),(216,d216,p216)):
            src=getattr(args,f'stage{k}_root')
            if old.read_sealed(src/'manifest.json')!=p or old.read_sealed(src/f'stage{k}_decision.json')!=d:raise ValueError('Physical source differs')
        if any(p[k]!=p214[k] for p in (p215,p216) for k in ('entries','folds','search_factor')) or p216['source_stage215']!=d215['fingerprint'] or p215['source_stage214']!=d214['fingerprint']:raise ValueError('Source split/lineage changed')
        code={n:file_hash(REPO/n) for n in p216['code_sha256']};code.update({str(Path(n).relative_to(REPO)):file_hash(n) for n in (Path(__file__).resolve(),REPO/'tools/run_stage217_predictor_audit.sh')})
        code={k.replace('\\','/'):v for k,v in code.items()}
        plan=old.sealed(dict(schema=SCHEMA,source_stage214=d214['fingerprint'],source_stage215=d215['fingerprint'],source_stage216=d216['fingerprint'],entries=p214['entries'],folds=p214['folds'],search_factor=p214['search_factor'],source_reader_compatibility=compatibility,code_sha256=code,protocol_sha256=file_hash(PROTOCOL),prediction_device='cpu',baseline_rule='equal_fit_group_mean_label_delta',policy='No image/optimizer/CUDA; original loss on full fit and held caches'))
        publish_identical(root/'manifest.json',plan)
        config=dict(source_paths={str(k):str(getattr(args,f'stage{k}_root').resolve()) for k in sources.SOURCES},stage214_root=str(args.stage214_root.resolve()),stage215_root=str(args.stage215_root.resolve()),stage216_root=str(args.stage216_root.resolve()))
        entries,dataset=runtime.load_entries(config,plan);prior.source.c.protocol.safe_output(root,[dataset])
        import torch
        torch.set_num_threads(1)
        audit(root,plan,config,entries,p214,d214,t214,p215,d215,p216,d216,torch)


if __name__=='__main__':main()
