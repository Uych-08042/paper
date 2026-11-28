"""Offline decomposition of frozen forecast labels; no images or training."""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
from time import perf_counter
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage220_causal_history as source
old=source.old;c=source.c
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv
from experiment_progress import OverallProgress
SCHEMA='stage221_frozen_target_components_v1'
TABLES=('support','scope','cells','actions','group_metrics','policy_summary','paired','source_replay','execution')
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE221_TARGET_COMPONENTS.md'
BUNDLE=source.source.RESULTS/'stage220_results_20261128T125221Z_382507.txt'
METRICS=('error_center_l2','motion_center_l2','label_center_l2','error_center_energy','motion_center_energy','label_center_energy','center_interaction','opposed_center_fraction','error_center_larger_fraction','error_log_abs','motion_log_abs','label_log_abs','error_log_energy','motion_log_energy','label_log_energy','log_interaction','opposed_log_fraction','error_log_larger_fraction')


def bundle(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=BUNDLE_CANONICAL:raise ValueError('Need reviewed Stage-220 packet')
    marks=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw));parts={m[1].decode():raw[m.end():marks[i+1].start() if i+1<len(marks) else len(raw)] for i,m in enumerate(marks)}
    d,p=json.loads(parts['stage220_decision.json']),json.loads(parts['manifest.json'])
    if any(old.sealed({k:v for k,v in x.items() if k!='fingerprint'})!=x for x in (d,p)) or d['schema']!=source.SCHEMA or d['plan_fingerprint']!=p['fingerprint']:raise ValueError('Wrong source identity')
    if len(parts)!=len(marks) or set(parts)!={'stage220_decision.json','manifest.json',*d['table_sha256']}:raise ValueError('Wrong source inventory')
    tables={}
    for n,h in d['table_sha256'].items():
        b=parts[n].replace(b'\r\n',b'\n');v=next((v for chunk in (b,b[:-1],b[:-2]) for v in (chunk,chunk.replace(b'\n',b'\r\n')) if hashlib.sha256(v).hexdigest()==h),None)
        if v is None:raise ValueError('Source table changed')
        tables[n[9:-4]]=list(csv.DictReader(io.StringIO(v.decode())))
    return d,p,tables


def valid_boxes(b):return np.isfinite(b).all(1)&(b[:,2:]>0).all(1)


def components(raw,current,future):
    raw,current,future=[np.asarray(v,np.float64) for v in (raw,current,future)]
    if raw.ndim!=2 or raw.shape[1]!=4 or current.shape!=raw.shape or future.shape!=raw.shape or not valid_boxes(raw).all():raise ValueError('Invalid component geometry')
    vc,vn=valid_boxes(current),valid_boxes(future);mask=vc&vn
    y=np.full((len(raw),3),np.nan);e=y.copy();m=y.copy()
    y[vn]=source.source.label(raw[vn],future[vn])
    e[mask]=source.source.label(raw[mask],current[mask])
    side=np.sqrt(raw[mask,2:].prod(1));gc=current[mask,:2]+current[mask,2:]/2;gn=future[mask,:2]+future[mask,2:]/2
    m[mask]=np.c_[(gn-gc)/side[:,None],np.log(np.sqrt(future[mask,2:].prod(1)/current[mask,2:].prod(1)))]
    np.testing.assert_allclose(e[mask]+m[mask],y[mask],rtol=1e-12,atol=1e-12)
    return vc,vn,mask,e,m,y


def component_metrics(e,m,y):
    n=len(e)
    if not n:return {k:None for k in METRICS}
    ee=np.sum(e[:,:2]**2,1);mm=np.sum(m[:,:2]**2,1);yy=np.sum(y[:,:2]**2,1);dot=np.sum(e[:,:2]*m[:,:2],1)
    np.testing.assert_allclose(ee+mm+2*dot,yy,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(e[:,2]**2+m[:,2]**2+2*e[:,2]*m[:,2],y[:,2]**2,rtol=1e-11,atol=1e-10)
    vals=(np.sqrt(ee),np.sqrt(mm),np.sqrt(yy),ee,mm,yy,2*dot,dot<0,ee>mm,np.abs(e[:,2]),np.abs(m[:,2]),np.abs(y[:,2]),e[:,2]**2,m[:,2]**2,y[:,2]**2,2*e[:,2]*m[:,2],e[:,2]*m[:,2]<0,np.abs(e[:,2])>np.abs(m[:,2]))
    return {k:float(np.mean(v)) for k,v in zip(METRICS,vals)}


def applied_delta(torch,z,raw,hw):
    # Match the ORIGINAL loss's float32 parent bounds, independent of GT labels.
    r=torch.tensor(raw,dtype=torch.float32);d=torch.tensor(z,dtype=torch.float32);h=torch.tensor(hw,dtype=torch.float32)
    with torch.no_grad():
        s=r[:,2:].prod(1).sqrt();center=r[:,:2]+r[:,2:]/2
        clipped=torch.minimum(torch.maximum(center+d[:,:2]*s[:,None],torch.zeros_like(center)),h.flip(1))
        logside=torch.minimum(torch.clamp(s.log()+d[:,2],min=0),h.max(1).values.log())
        return torch.cat(((clipped-center)/s[:,None],(logside-s.log())[:,None]),1).numpy().astype(float)


def action_metrics(pred,e,m,y):
    out={}
    for name,v in (('error',e),('motion',m),('label',y)):
        out['center_mse_to_'+name]=float(np.mean(np.sum((pred[:,:2]-v[:,:2])**2,1))) if len(v) else None
        out['log_mse_to_'+name]=float(np.mean((pred[:,2]-v[:,2])**2)) if len(v) else None
        den=np.linalg.norm(pred[:,:2],axis=1)*np.linalg.norm(v[:,:2],axis=1);ok=den>0
        out['center_cosine_'+name+'_samples']=int(ok.sum())
        out['center_cosine_to_'+name]=float(np.mean(np.sum(pred[ok,:2]*v[ok,:2],1)/den[ok])) if ok.any() else None
    return out


def contrast(v):
    if len(v)>=2:return source.c.protocol.contrast(v)
    return dict(videos=len(v),mean_delta=float(v[0]) if len(v) else None,median_delta=float(v[0]) if len(v) else None,ci_low=None,ci_high=None,positive=int(len(v)>0 and v[0]>0),negative=int(len(v)>0 and v[0]<0),worst_leave_one_video_out=None)


def summarize(rows):
    summary=[];paired=[];idx={}
    groups=sorted({r['group'] for r in rows})
    for split,origin in sorted({(r['split'],r['origin']) for r in rows}):
        for g in groups:
            rr=[r for r in rows if (r['split'],r['origin'],r['group'])==(split,origin,g)]
            if sorted(r['seed'] for r in rr)!=c.SEEDS:raise ValueError('Incomplete source seed grid')
            idx[split,origin,g]={k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in METRICS}
        supported=[g for g in groups if idx[split,origin,g][METRICS[0]] is not None]
        summary.append(dict(split=split,origin=origin,supported_groups=len(supported),unsupported_groups='|'.join(g for g in groups if g not in supported),**{k:float(np.mean([idx[split,origin,g][k] for g in supported])) if supported else None for k in METRICS}))
    for split in ('fit','held'):
        origins=('nn','mm') if split=='fit' else source.source.prior.ORIGINS[:2]
        supported=[g for g in groups if all(idx[split,o,g][METRICS[0]] is not None for o in origins)]
        values=[float(np.mean([idx[split,o,g]['error_center_l2']-idx[split,o,g]['motion_center_l2'] for o in origins])) for g in supported]
        paired.append(dict(split=split,contrast='current_error_minus_true_motion_center_l2',primary=split=='held',interpretation='positive means current localization error has larger magnitude; not a performance gain',unsupported_groups='|'.join(g for g in groups if g not in supported),**contrast(values)))
    return summary,paired


def experiment(root,plan,config,plans,catalog,entries,t220,torch):
    if (root/'stage221_decision.json').exists():verify(root);return
    start=perf_counter();tables={k:[] for k in TABLES};verified={};artifacts={}
    progress=OverallProgress('Stage-221',sum(2*len(f['fit'])+4*len(f['held'])+3 for f in plan['folds'])*len(c.SEEDS)+1,unit='checks');progress.report('Cached label decomposition; frozen predictors')
    metric_ref={(r['split'],r['group'],int(r['seed']),r['origin'],r['arm']):r for r in t220['group_metrics']}
    action_ref={(r['split'],r['group'],int(r['seed']),r['origin'],r['arm']):r for r in t220['actions']}
    for fold in plan['folds']:
        for seed in c.SEEDS:
            _,items=source.data_for(config,plans,catalog,entries,fold,seed,verified);models={}
            for arm in source.ARMS:
                path=Path(config['stage220_root'])/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt"
                for pp in (path,path.with_suffix('.receipt.json')):
                    rel=pp.relative_to(config['stage220_root']).as_posix()
                    if file_hash(pp)!=config['source220_artifacts'][rel]:raise ValueError('Changed frozen model')
                    verified.setdefault('220',{})[rel]=file_hash(pp)
                meta=old.read_sealed(path.with_suffix('.receipt.json'))['metadata']
                if any(meta[k]!=v for k,v in dict(plan_fingerprint=config['source220_plan'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],updates=256).items()):raise ValueError('Wrong frozen model split')
                saved=old.load_model(torch,path,meta);state=saved['state'];model=source.build(torch,arm,392,state['mean'],state['std'],seed);model.load_state_dict(state);model.eval().requires_grad_(False)
                ref=next(r for r in t220['scope'] if (int(r['fold']),int(r['seed']),r['arm'])==(fold['fold'],seed,arm))
                if old.model_digest(model)!=ref['state_sha256']:raise ValueError('Frozen predictor differs')
                models[arm]=model;tables['scope'].append(dict(fold=fold['fold'],seed=seed,arm=arm,state_sha256=ref['state_sha256'],new_updates=0,device='cpu'));progress.advance()
            digests={a:old.model_digest(m) for a,m in models.items()}
            for split,g,origin,data in items:
                spec=next(v for v in plan['entries'] if v['group']==g);seq=spec['sequence'];gt=entries[seq][1]
                frames=[int(t) for t in c.protocol.uniform_frames(len(gt)-2) if np.isfinite(gt[t+1]).all() and (gt[t+1,2:]>0).all()] if split=='fit' else [r['frame'] for r in plans[2]['selection'] if r['sequence']==seq]
                vc,vn,mask,e,m,y=components(data['raw'],gt[frames],data['target'])
                if not vn.all():raise ValueError('Reviewed Stage-220 future labels changed; cannot silently subset loss replay')
                common=dict(fold=fold['fold'],seed=seed,split=split,group=g,sequence=seq,origin=origin)
                tables['support'].append(dict(common,samples=len(frames),valid_pairs=int(mask.sum()),invalid_current=int((~vc).sum()),invalid_next=int((~vn).sum())))
                for i,t in enumerate(frames):
                    vals={prefix+'_'+axis:float(arr[i,j]) if np.isfinite(arr[i,j]) else None for prefix,arr in (('error',e),('motion',m),('label',y)) for j,axis in enumerate(('dx','dy','log_side'))}
                    tables['cells'].append(dict(common,frame=t,valid_current=bool(vc[i]),valid_next=bool(vn[i]),valid_pair=bool(mask[i]),additivity_max_abs=float(np.max(np.abs(e[i]+m[i]-y[i]))) if mask[i] else None,**vals))
                tables['group_metrics'].append(dict(common,valid_pairs=int(mask.sum()),**component_metrics(e[mask],m[mask],y[mask])))
                for arm in source.METHODS:
                    with torch.no_grad():z=np.zeros((len(frames),3),np.float32) if arm=='zero' else data['velocity'] if arm=='velocity' else models[arm](torch.tensor(data['x'],dtype=torch.float32)).numpy()
                    losses=source.source.evaluate(torch,z,data,'forecast',plan['search_factor']);ref=metric_ref[split,g,seed,origin,arm]
                    if any(losses[k]!=float(ref[k]) for k in losses):raise ValueError('Stage-220 objective replay differs')
                    requested=dict(mean_center_delta=float(np.linalg.norm(z[:,:2],axis=1).mean()),mean_abs_log_side_delta=float(np.abs(z[:,2]).mean()),nonzero_fraction=float((np.abs(z).max(1)>1e-9).mean()))
                    if any(requested[k]!=float(action_ref[split,g,seed,origin,arm][k]) for k in requested):raise ValueError('Requested action replay differs')
                    effective=applied_delta(torch,z,data['raw'],data['image_hw'])
                    tables['actions'].append(dict(common,arm=arm,valid_pairs=int(mask.sum()),source_regression=losses['regression'],**requested,**action_metrics(effective[mask],e[mask],m[mask],y[mask])))
                    tables['source_replay'].append(dict(common,arm=arm,loss_exact=True,requested_action_exact=True))
                progress.advance()
            if digests!={a:old.model_digest(m) for a,m in models.items()}:raise ValueError('Frozen weights changed')
    tables['policy_summary'],tables['paired']=summarize(tables['group_metrics']);tables['execution']=[dict(device='cpu',seconds=perf_counter()-start,image_reads=0,cuda_calls=0,optimizer_updates=0)]
    publish_identical(root/'source_arrays.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],files=verified)));artifacts['source_arrays.json']=file_hash(root/'source_arrays.json');hashes={}
    for n,rr in tables.items():write_csv(root/f'stage221_{n}.csv',rr);hashes[f'stage221_{n}.csv']=file_hash(root/f'stage221_{n}.csv')
    if progress.done!=progress.total-1:raise ValueError('Incomplete progress budget')
    publish_identical(root/'stage221_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_TARGET_COMPONENTS',plan_fingerprint=plan['fingerprint'],new_training_updates=0,new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,primary=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts)))
    progress.advance();progress.report('Complete');verify(root)


def verify(root):
    root=Path(root).resolve();p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage221_decision.json')
    if d['schema']!=p['schema'] or d['schema']!=SCHEMA or d['status']!='REVIEW_TARGET_COMPONENTS' or d['plan_fingerprint']!=p['fingerprint'] or len(d['primary'])!=1 or set(d['table_sha256'])!={f'stage221_{n}.csv' for n in TABLES} or any(d[k]!=0 for k in ('new_training_updates','new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Incomplete diagnostic')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        pp=(root/n).resolve()
        if not pp.is_relative_to(root) or file_hash(pp)!=h:raise ValueError('Changed artifact')
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Changed code')
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Changed protocol')
    print('Stage-221 verified; offline label composition, no new tracking result',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);source_keys=(76,89,93,121,170,188,191);keys=(214,215,216,217,218,219,220,*source_keys)
    for k in keys:parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage220-bundle',type=Path,default=BUNDLE);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');a=parser.parse_args();root=a.output_dir.resolve()
    if a.check_results:verify(root);return
    if any(getattr(a,f'stage{k}_root') is None for k in keys):parser.error('Need all frozen cache roots')
    c.protocol.safe_output(root,[getattr(a,f'stage{k}_root') for k in keys]+[source.source.RESULTS,a.stage220_bundle.parent])
    with writer_lock(root):
        if (root/'stage221_decision.json').exists():verify(root);return
        d,p,t=bundle(a.stage220_bundle);source.verify(a.stage220_root)
        if old.read_sealed(a.stage220_root/'manifest.json')!=p or old.read_sealed(a.stage220_root/'stage220_decision.json')!=d:raise ValueError('Physical Stage-220 differs')
        d219,p219,_=source.bundle(source.BUNDLE);d218,p218,_=source.prior.bundle(source.prior.BUNDLE);d217,p217,_=source.base.bundle(source.base.BUNDLE)
        for k,dk,pk in ((219,d219,p219),(218,d218,p218),(217,d217,p217)):
            rr=getattr(a,f'stage{k}_root')
            if old.read_sealed(rr/'manifest.json')!=pk or old.read_sealed(rr/f'stage{k}_decision.json')!=dk:raise ValueError('Changed source lineage')
        if p['source_stage219']!=d219['fingerprint'] or p219['source_stage218']!=d218['fingerprint'] or p218['source_stage217']!=d217['fingerprint']:raise ValueError('Broken source graph')
        source.source.verify(a.stage217_root);receipt=old.read_sealed(a.stage217_root/'source_arrays.json');catalog=receipt['source_files']
        if receipt['plan_fingerprint']!=p217['fingerprint']:raise ValueError('Wrong catalog')
        plans=[]
        for k in (214,215,216):
            rr=getattr(a,f'stage{k}_root');pk=old.read_sealed(rr/'manifest.json');dk=old.read_sealed(rr/f'stage{k}_decision.json')
            if p217[f'source_stage{k}']!=dk['fingerprint'] or dk['plan_fingerprint']!=pk['fingerprint'] or any(pk[n]!=p[n] for n in ('entries','folds','search_factor')) or any(dk['artifact_sha256'].get(n)!=h for n,h in catalog[f'stage{k}'].items()):raise ValueError('Changed input catalog')
            plans.append(pk)
        code={n:file_hash(REPO/n) for n in p['code_sha256']}
        for n in ('tools/stage221_target_components.py','tools/run_stage221_target_components.sh'):code[n]=file_hash(REPO/n)
        plan=old.sealed(dict(schema=SCHEMA,source_stage220=d['fingerprint'],entries=p['entries'],folds=p['folds'],search_factor=p['search_factor'],policy='offline GT decomposition with common reference-side normalization; no new model',code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan);config={f'stage{k}_root':str(getattr(a,f'stage{k}_root').resolve()) for k in (214,215,216)};config.update(stage220_root=str(a.stage220_root.resolve()),source220_artifacts=d['artifact_sha256'],source220_plan=p['fingerprint'],source_paths={str(k):str(getattr(a,f'stage{k}_root')) for k in source_keys})
        entries,dataset=source.source.runtime.load_entries(config,plan);c.protocol.safe_output(root,[dataset])
        import torch
        torch.set_num_threads(1);experiment(root,plan,config,plans,catalog,entries,t,torch)


BUNDLE_CANONICAL='729d5a8ff47b97d55b16f260bc0ad620bc9c590329db92ee94760bdcd8c66377'
if __name__=='__main__':main()
