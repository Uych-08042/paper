"""Causal second-difference inputs with matched CPU forecast training."""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
from time import perf_counter
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage219_input_blocks as prior
base=prior.base
old=base.old;c=base.c;source=base.source
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv
from experiment_progress import OverallProgress
SCHEMA='stage220_causal_second_difference_v1'
ARMS=('geometry','history','shuffled');METHODS=(*ARMS,'zero','velocity')
TABLES=('support','scope','training','group_metrics','policy_summary','paired','source_replay','actions','execution')
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE220_CAUSAL_HISTORY.md'
BUNDLE=source.RESULTS/'stage219_results_20261128T122255Z_357610.txt'


def bundle(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=BUNDLE_CANONICAL:raise ValueError('Need reviewed Stage-219 export')
    marks=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw));parts={m[1].decode():raw[m.end():marks[i+1].start() if i+1<len(marks) else len(raw)] for i,m in enumerate(marks)}
    d,p=json.loads(parts['stage219_decision.json']),json.loads(parts['manifest.json'])
    if any(old.sealed({k:v for k,v in x.items() if k!='fingerprint'})!=x for x in (d,p)) or d['schema']!=prior.SCHEMA or d['plan_fingerprint']!=p['fingerprint']:raise ValueError('Wrong source identity')
    if len(parts)!=len(marks) or set(parts)!={'stage219_decision.json','manifest.json',*d['table_sha256']}:raise ValueError('Wrong source inventory')
    tables={}
    for n,h in d['table_sha256'].items():
        b=parts[n].replace(b'\r\n',b'\n');v=next((v for chunk in (b,b[:-1],b[:-2]) for v in (chunk,chunk.replace(b'\n',b'\r\n')) if hashlib.sha256(v).hexdigest()==h),None)
        if v is None:raise ValueError('Source table changed')
        tables[n[9:-4]]=list(csv.DictReader(io.StringIO(v.decode())))
    return d,p,tables


def build(torch,mode,dim,mean,std,seed):
    if mode not in ARMS or dim!=392:raise ValueError('Expected fixed 392 input')
    original=c.build(torch,dim,mean,std,seed)
    class HistoryMLP(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('mean',original.mean);self.register_buffer('std',original.std);self.network=original.network
        def forward(self,x):
            z=(x-self.mean)/self.std
            start=384 if mode=='geometry' else 381
            return self.network(torch.cat((torch.zeros_like(z[:,:start]),z[:,start:]),1))
    return HistoryMLP()


def motion(reference,initial,frames):
    """Read only R[t], R[t-1], R[t-2]; initialization is the known first box."""
    if not frames:raise ValueError('No history frames')
    boxes=np.vstack((np.asarray(initial,float),np.asarray(reference,float)[:max(frames)]))
    if boxes.shape[1]!=4 or not np.isfinite(boxes).all() or (boxes[:,2:]<=0).any():raise ValueError('Invalid predicted history')
    centers=boxes[:,:2]+boxes[:,2:]/2;sides=np.sqrt(boxes[:,2:].prod(1));u=np.c_[centers,np.log(sides)]
    velocity=[];acceleration=[]
    for t in frames:
        if not 1<=t<len(boxes):raise ValueError('History frame outside trace')
        v=u[t]-u[t-1];a=u[t]-2*u[t-1]+u[t-2] if t>=2 else np.zeros(3)
        v[:2]/=sides[t];a[:2]/=sides[t];velocity.append(v);acceleration.append(a)
    return np.asarray(velocity,np.float32),np.asarray(acceleration,np.float32)


def data_for(config,plans,catalog,entries,fold,seed,verified):
    _,items=base.data_for(config,plans,catalog,entries,fold,seed,verified);result=[];bank={};pieces={g:[] for g in fold['fit']}
    for split,g,origin,data in items:
        spec=next(e for e in plans[0]['entries'] if e['group']==g);seq=spec['sequence'];truth=entries[seq][1]
        k=214 if split=='fit' else 215;root=Path(config[f'stage{k}_root'])
        if split=='fit':
            rel=f"fit_rollouts/f{fold['fold']}__s{seed}__{seq}__{origin}.npz"
            frames=[int(t) for t in c.protocol.uniform_frames(len(truth)-2) if np.isfinite(truth[t+1]).all() and (truth[t+1,2:]>0).all()]
        else:
            rel=source.prior.source.previous.path_for(root,'full',seq,seed,origin).relative_to(root).as_posix()
            frames=[r['frame'] for r in plans[2]['selection'] if r['sequence']==seq]
        meta=old.read_sealed((root/rel).with_suffix('.receipt.json'))['metadata']
        trace=source.bound(root,dict(artifact_sha256=catalog[f'stage{k}']),rel,meta,verified.setdefault(str(k),{}))
        np.testing.assert_array_equal(data['raw'],trace['reference_boxes'][np.asarray(frames)-1])
        v,a=motion(trace['reference_boxes'],truth[0],frames)
        x=data['x'].copy();x[:,381:384]=a
        value=dict(data,x=x,velocity=v,acceleration=a,history_available=np.asarray(frames)>=2)
        result.append((split,g,origin,value))
        if split=='fit':pieces[g].append(value)
    for g,rr in pieces.items():bank[g]={k:np.concatenate([v[k] for v in rr]) for k in rr[0]}
    return bank,result


def permutations(bank,fold,seed):
    rng=np.random.default_rng(seed+100*fold['fold']+220)
    return {g:rng.permutation(len(bank[g]['x'])) for g in fold['fit']}


def batch_values(bank,batch,mode,perm):
    values={k:np.stack([bank[g][k][i] for g,i in batch]).copy() for k in ('x','raw','target','image_hw')}
    if mode=='shuffled':values['x'][:,381:384]=np.stack([bank[g]['x'][perm[g][i],381:384] for g,i in batch])
    return values


def train(root,plan,torch,bank,fold,seed,mode,progress):
    if set(bank)!=set(fold['fit']) or set(bank)&set(fold['held']):raise ValueError('Fit roster changed')
    xx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=xx.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(xx.std(0,dtype=np.float64),1e-6).astype(np.float32)
    draws=c.protocol.draws_for(fold,seed,{g:len(bank[g]['x']) for g in fold['fit']})
    if len(draws)!=256 or any(len(b)!=4 for b in draws):raise ValueError('Budget changed')
    model=build(torch,mode,xx.shape[1],mean,std,seed).eval();initial=old.model_digest(model);initial_network=old.model_digest(model.network);perm=permutations(bank,fold,seed)
    meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=mode,fit=fold['fit'],held=fold['held'],updates=256,draws_sha256=fingerprint(draws),normalization_sha256=fingerprint(dict(mean=mean.tolist(),std=std.tolist())),mask='geometry8_plus_acceleration3',permutation_sha256=fingerprint({g:v.tolist() for g,v in perm.items()}))
    path=root/'training'/f"f{fold['fold']}__s{seed}__{mode}.pt";saved=old.load_model(torch,path,meta)
    if saved is not None:
        model.load_state_dict(saved['state'],strict=True)
        if old.model_digest(model)!=saved['scope']['state_sha256']:raise ValueError('Changed model')
        progress.reused+=256;progress.advance(count=256);return model.requires_grad_(False),saved['scope'],saved['history']
    opt=torch.optim.AdamW(model.parameters(),lr=c.DESIGN['learning_rate'],weight_decay=c.DESIGN['weight_decay']);history=[]
    for step,batch in enumerate(draws,1):
        v={k:torch.tensor(value,dtype=torch.float32) for k,value in batch_values(bank,batch,mode,perm).items()}
        opt.zero_grad(set_to_none=True);loss,_,_=c.loss(torch,model(v['x']),v['raw'],v['target'],v['image_hw'],plan['search_factor'],'forecast')
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        history.append(dict(fold=fold['fold'],seed=seed,arm=mode,step=step,loss=float(loss.detach()),gradient_norm=float(norm)));progress.advance()
    if initial==old.model_digest(model) or any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Inactive or nonfinite model')
    changed=sum(int(np.any(bank[g]['x'][:,381:384]!=bank[g]['x'][perm[g],381:384],axis=1).sum()) for g in fold['fit']) if mode=='shuffled' else 0
    scope=dict(fold=fold['fold'],seed=seed,arm=mode,parameters=sum(v.numel() for v in model.parameters()),active_inputs=8 if mode=='geometry' else 11,updates=256,examples=1024,initial_sha256=initial,initial_network_sha256=initial_network,state_sha256=old.model_digest(model),network_sha256=old.model_digest(model.network),permutation_sha256=meta['permutation_sha256'],history_changed_pairs=changed,draws_sha256=meta['draws_sha256'],normalization_sha256=meta['normalization_sha256'],device='cpu')
    old.save_model(torch,path,dict(state=model.state_dict(),scope=scope,history=history),meta)
    return model.requires_grad_(False),scope,history


def summarize(rows):
    summary=[];pairs=[];idx={}
    for split,origin in sorted({(r['split'],r['origin']) for r in rows}):
        groups=sorted({r['group'] for r in rows if (r['split'],r['origin'])==(split,origin)})
        for arm in METHODS:
            for g in groups:
                rr=[r for r in rows if (r['split'],r['origin'],r['group'],r['arm'])==(split,origin,g,arm)]
                if sorted(r['seed'] for r in rr)!=c.SEEDS:raise ValueError('Incomplete seeds')
                idx[split,origin,arm,g]=float(np.mean([r['regression'] for r in rr]))
            summary.append(dict(split=split,origin=origin,arm=arm,groups=len(groups),regression=float(np.mean([idx[split,origin,arm,g] for g in groups]))))
    groups=sorted({r['group'] for r in rows})
    for split in ('fit','held'):
        origins=('nn','mm') if split=='fit' else source.prior.ORIGINS[:2]
        for a,b in (('history','geometry'),('history','shuffled'),*[(a,'zero') for a in ARMS],('velocity','zero')):
            pairs.append(dict(split=split,contrast=a+'_minus_'+b,primary=(split,a,b)==('held','history','geometry'),favorable_direction='negative',**source.loss_contrast([np.mean([idx[split,o,a,g]-idx[split,o,b,g] for o in origins]) for g in groups])))
    return summary,pairs


def experiment(root,plan,config,plans,catalog,entries,t219,torch):
    if (root/'stage220_decision.json').exists():verify(root);return
    start=perf_counter();tables={k:[] for k in TABLES};artifacts={};verified={}
    progress=OverallProgress('Stage-220',len(plan['folds'])*len(c.SEEDS)*769+1,unit='steps');progress.report('Cached causal-history CPU training')
    refidx={(r['split'],r['group'],int(r['seed']),r['origin'],r['arm']):r for r in t219['group_metrics']}
    for fold in plan['folds']:
        for seed in c.SEEDS:
            bank,items=data_for(config,plans,catalog,entries,fold,seed,verified);models={}
            ref=next(r for r in t219['scope'] if (int(r['fold']),int(r['seed']),r['arm'])==(fold['fold'],seed,'geometry'))
            path=Path(config['stage219_root'])/'training'/f"f{fold['fold']}__s{seed}__geometry.pt"
            for pp in (path,path.with_suffix('.receipt.json')):
                if file_hash(pp)!=config['source219_artifacts'][pp.relative_to(config['stage219_root']).as_posix()]:raise ValueError('Changed source geometry model')
                verified.setdefault('219',{})[pp.relative_to(config['stage219_root']).as_posix()]=file_hash(pp)
            meta=old.read_sealed(path.with_suffix('.receipt.json'))['metadata'];saved=old.load_model(torch,path,meta)
            reference=prior.build(torch,'geometry',392,saved['state']['mean'],saved['state']['std'],seed);reference.load_state_dict(saved['state']);reference.eval().requires_grad_(False)
            if old.model_digest(reference)!=ref['state_sha256']:raise ValueError('Wrong source geometry weights')
            initial=prior.build(torch,'geometry',392,saved['state']['mean'],saved['state']['std'],seed)
            if old.model_digest(initial)!=ref['initial_sha256']:raise ValueError('Wrong source initialization')
            for mode in ARMS:
                model,scope,history=train(root,plan,torch,bank,fold,seed,mode,progress)
                if any(str(scope[k])!=str(ref[k]) for k in ('draws_sha256','parameters','updates','examples')):raise ValueError('Matched source training identity differs')
                if scope['initial_network_sha256']!=old.model_digest(initial.network):raise ValueError('Initial network weights changed')
                np.testing.assert_array_equal(model.mean.numpy()[384:],reference.mean.numpy()[384:]);np.testing.assert_array_equal(model.std.numpy()[384:],reference.std.numpy()[384:])
                if mode=='geometry' and old.model_digest(model.network)!=old.model_digest(reference.network):raise ValueError('Current-geometry network weights do not reproduce Stage-219; keep diagnostic, do not relax tolerance')
                scope=dict(scope,source_geometry_network_sha256=old.model_digest(reference.network))
                models[mode]=model;tables['scope'].append(scope);tables['training'].extend(history);old.record(root,root/'training'/f"f{fold['fold']}__s{seed}__{mode}.pt",artifacts)
            digests={a:old.model_digest(m) for a,m in models.items()}
            for split,g,origin,data in items:
                tables['support'].append(dict(fold=fold['fold'],seed=seed,split=split,group=g,origin=origin,samples=len(data['x']),history_available=int(data['history_available'].sum()),warmup=int((~data['history_available']).sum()),acceleration_sha256=fingerprint(data['acceleration'].tolist())))
                for arm in METHODS:
                    with torch.no_grad():delta=np.zeros((len(data['x']),3),np.float32) if arm=='zero' else data['velocity'] if arm=='velocity' else models[arm](torch.tensor(data['x'],dtype=torch.float32)).numpy()
                    losses=source.evaluate(torch,delta,data,'forecast',plan['search_factor'])
                    if arm in ('zero','geometry'):
                        rr=refidx[split,g,seed,origin,arm]
                        if any(losses[k]!=float(rr[k]) for k in losses):raise ValueError('Geometry/zero source loss replay differs')
                        tables['source_replay'].append(dict(split=split,group=g,seed=seed,origin=origin,arm=arm,exact=True))
                    tables['actions'].append(dict(split=split,group=g,seed=seed,origin=origin,arm=arm,mean_center_delta=float(np.linalg.norm(delta[:,:2],axis=1).mean()),mean_abs_log_side_delta=float(np.abs(delta[:,2]).mean()),nonzero_fraction=float((np.abs(delta).max(1)>1e-9).mean())))
                    tables['group_metrics'].append(dict(fold=fold['fold'],seed=seed,split=split,group=g,origin=origin,arm=arm,**losses))
            if digests!={a:old.model_digest(m) for a,m in models.items()}:raise ValueError('Evaluation changed model')
            progress.advance()
    tables['policy_summary'],tables['paired']=summarize(tables['group_metrics']);tables['execution']=[dict(device='cpu',seconds=perf_counter()-start,image_reads=0,cuda_calls=0)]
    publish_identical(root/'source_arrays.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],files=verified)));artifacts['source_arrays.json']=file_hash(root/'source_arrays.json');hashes={}
    for n,rr in tables.items():write_csv(root/f'stage220_{n}.csv',rr);hashes[f'stage220_{n}.csv']=file_hash(root/f'stage220_{n}.csv')
    if progress.done!=progress.total-1:raise ValueError('Incomplete progress budget')
    publish_identical(root/'stage220_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_CAUSAL_HISTORY',plan_fingerprint=plan['fingerprint'],training_runs=len(tables['scope']),training_updates=len(tables['training']),new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,primary=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts)))
    progress.advance();progress.report('Complete');verify(root)


def verify(root):
    root=Path(root).resolve();p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage220_decision.json')
    if d['schema']!=p['schema'] or d['schema']!=SCHEMA or d['status']!='REVIEW_CAUSAL_HISTORY' or d['plan_fingerprint']!=p['fingerprint'] or (d['training_runs'],d['training_updates'])!=(12,3072) or len(d['primary'])!=1 or set(d['table_sha256'])!={f'stage220_{n}.csv' for n in TABLES} or any(d[k]!=0 for k in ('new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Incomplete result')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        pp=(root/n).resolve()
        if not pp.is_relative_to(root) or file_hash(pp)!=h:raise ValueError('Changed artifact')
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Changed code')
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Changed protocol')
    print('Stage-220 verified; REVIEW, no tracking AUC',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source_keys=(76,89,93,121,170,188,191)
    keys=(214,215,216,217,218,219,*source_keys)
    for k in keys:parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage219-bundle',type=Path,default=BUNDLE);parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--check-results',action='store_true');a=parser.parse_args();root=a.output_dir.resolve()
    if a.check_results:verify(root);return
    if any(getattr(a,f'stage{k}_root') is None for k in keys):parser.error('Need all source cache roots')
    c.protocol.safe_output(root,[getattr(a,f'stage{k}_root') for k in keys]+[source.RESULTS,a.stage219_bundle.parent])
    with writer_lock(root):
        if (root/'stage220_decision.json').exists():verify(root);return
        d,p,t=bundle(a.stage219_bundle);prior.verify(a.stage219_root)
        if old.read_sealed(a.stage219_root/'manifest.json')!=p or old.read_sealed(a.stage219_root/'stage219_decision.json')!=d:raise ValueError('Physical Stage-219 changed')
        d218,p218,_=prior.bundle(prior.BUNDLE)
        if old.read_sealed(a.stage218_root/'manifest.json')!=p218 or old.read_sealed(a.stage218_root/'stage218_decision.json')!=d218 or p['source_stage218']!=d218['fingerprint']:raise ValueError('Stage-218 lineage changed')
        d217,p217,_=base.bundle(base.BUNDLE);source.verify(a.stage217_root)
        if old.read_sealed(a.stage217_root/'manifest.json')!=p217 or old.read_sealed(a.stage217_root/'stage217_decision.json')!=d217 or old.read_sealed(a.stage218_root/'manifest.json')['source_stage217']!=d217['fingerprint']:raise ValueError('Stage-217 lineage changed')
        rr=old.read_sealed(a.stage217_root/'source_arrays.json');catalog=rr['source_files']
        if rr['plan_fingerprint']!=p217['fingerprint']:raise ValueError('Wrong catalog')
        plans=[]
        for k in (214,215,216):
            pk=old.read_sealed(getattr(a,f'stage{k}_root')/'manifest.json');dk=old.read_sealed(getattr(a,f'stage{k}_root')/f'stage{k}_decision.json')
            if p217[f'source_stage{k}']!=dk['fingerprint'] or dk['plan_fingerprint']!=pk['fingerprint'] or any(pk[n]!=p[n] for n in ('entries','folds','search_factor')) or any(dk['artifact_sha256'].get(n)!=h for n,h in catalog[f'stage{k}'].items()):raise ValueError('Source lineage/catalog changed')
            plans.append(pk)
        code={n:file_hash(REPO/n) for n in p['code_sha256']}
        for n in ('tools/stage220_causal_history.py','tools/run_stage220_causal_history.sh'):code[n]=file_hash(REPO/n)
        plan=old.sealed(dict(schema=SCHEMA,source_stage219=d['fingerprint'],entries=p['entries'],folds=p['folds'],search_factor=p['search_factor'],design=c.DESIGN,arms=list(ARMS),mask='geometry8_plus_acceleration3',device='cpu',code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan);config={f'stage{k}_root':str(getattr(a,f'stage{k}_root').resolve()) for k in (214,215,216)};config['source_paths']={str(k):str(getattr(a,f'stage{k}_root')) for k in source_keys}
        config['stage219_root']=str(a.stage219_root.resolve());config['source219_artifacts']=d['artifact_sha256']
        entries,dataset=source.runtime.load_entries(config,plan);c.protocol.safe_output(root,[dataset])
        import torch
        torch.set_num_threads(1);experiment(root,plan,config,plans,catalog,entries,t,torch)


BUNDLE_CANONICAL='ea19a1155888e8a78f7dc11211414a664dadb1466f115488861d2f949364a1ea'
if __name__=='__main__':main()
