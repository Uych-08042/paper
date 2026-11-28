"""Matched CPU forecast training: bias, linear and MLP; cached states only."""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
from time import perf_counter
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage217_predictor_audit as source
old=source.old;c=source.c
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv
from experiment_progress import OverallProgress
SCHEMA='stage218_matched_forecast_capacity_v1'
ARMS=('bias','linear','mlp');METHODS=(*ARMS,'zero','frozen_forecast')
TABLES=('support','scope','training','group_metrics','policy_summary','paired','source_replay','execution')
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE218_MATCHED_CAPACITY.md'
BUNDLE=source.RESULTS/'stage217_results_20261128T115012Z_330010.txt'


def bundle(path):
    raw=Path(path).read_bytes()
    # Fixed reviewed packet, permitting only Git newline normalization.
    if hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=BUNDLE_CANONICAL:raise ValueError('Need reviewed Stage-217 packet')
    marks=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():marks[i+1].start() if i+1<len(marks) else len(raw)] for i,m in enumerate(marks)}
    d,p=json.loads(parts['stage217_decision.json']),json.loads(parts['manifest.json'])
    if any(old.sealed({k:v for k,v in a.items() if k!='fingerprint'})!=a for a in (d,p)) or d['plan_fingerprint']!=p['fingerprint'] or d['schema']!=source.SCHEMA:raise ValueError('Changed source identity')
    if len(parts)!=len(marks) or set(parts)!={'stage217_decision.json','manifest.json',*d['table_sha256']}:raise ValueError('Source inventory changed')
    tables={}
    for n,h in d['table_sha256'].items():
        b=parts[n].replace(b'\r\n',b'\n');v=next((v for chunk in (b,b[:-1],b[:-2]) for v in (chunk,chunk.replace(b'\n',b'\r\n')) if hashlib.sha256(v).hexdigest()==h),None)
        if v is None:raise ValueError('Changed source table')
        tables[n[9:-4]]=list(csv.DictReader(io.StringIO(v.decode())))
    return d,p,tables


def build(torch,arm,dim,mean,std,seed):
    if arm=='mlp':return c.build(torch,dim,mean,std,seed)
    if arm not in ARMS:raise ValueError('Unknown capacity')
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('mean',torch.tensor(mean));self.register_buffer('std',torch.tensor(std))
            self.bias=torch.nn.Parameter(torch.zeros(3))
            if arm=='linear':self.weight=torch.nn.Parameter(torch.zeros(3,dim))
        def forward(self,x):
            z=self.bias.expand(len(x),3)
            return z if arm=='bias' else z+torch.nn.functional.linear((x-self.mean)/self.std,self.weight)
    return Small()


def train(root,plan,torch,bank,fold,seed,arm,progress):
    if set(bank)!=set(fold['fit']) or set(bank)&set(fold['held']):raise ValueError('Wrong fit roster')
    xx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=xx.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(xx.std(0,dtype=np.float64),1e-6).astype(np.float32)
    draws=c.protocol.draws_for(fold,seed,{g:len(bank[g]['x']) for g in fold['fit']})
    if len(draws)!=256 or any(len(b)!=4 for b in draws):raise ValueError('Training opportunity changed')
    model=build(torch,arm,xx.shape[1],mean,std,seed).eval();initial=old.model_digest(model)
    with torch.no_grad():np.testing.assert_array_equal(model(torch.tensor(xx[:2])).numpy(),np.zeros((min(2,len(xx)),3)))
    meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],draws_sha256=fingerprint(draws),normalization_sha256=fingerprint(dict(mean=mean.tolist(),std=std.tolist())),updates=256)
    path=root/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt";saved=old.load_model(torch,path,meta)
    if saved is not None:
        model.load_state_dict(saved['state']);scope=saved['scope']
        if old.model_digest(model)!=scope['state_sha256']:raise ValueError('Changed trained model')
        progress.reused+=len(draws);progress.advance(count=len(draws));return model.requires_grad_(False),scope,saved['history']
    opt=torch.optim.AdamW(model.parameters(),lr=c.DESIGN['learning_rate'],weight_decay=c.DESIGN['weight_decay']);history=[]
    for step,batch in enumerate(draws,1):
        v={k:torch.tensor(np.stack([bank[g][k][i] for g,i in batch]),dtype=torch.float32) for k in ('x','raw','target','image_hw')}
        opt.zero_grad(set_to_none=True);loss,reg,cover=c.loss(torch,model(v['x']),v['raw'],v['target'],v['image_hw'],plan['search_factor'],'forecast')
        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.DESIGN['gradient_clip'],error_if_nonfinite=True);opt.step()
        history.append(dict(fold=fold['fold'],seed=seed,arm=arm,step=step,loss=float(loss.detach()),gradient_norm=float(norm)));progress.advance()
    if old.model_digest(model)==initial or any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Inactive or nonfinite training')
    scope=dict(fold=fold['fold'],seed=seed,arm=arm,parameters=sum(v.numel() for v in model.parameters()),updates=len(draws),examples=sum(map(len,draws)),initial_sha256=initial,state_sha256=old.model_digest(model),draws_sha256=meta['draws_sha256'],normalization_sha256=meta['normalization_sha256'],device='cpu')
    old.save_model(torch,path,dict(state=model.state_dict(),scope=scope,history=history),meta)
    return model.requires_grad_(False),scope,history


def data_for(config,plans,catalog,entries,fold,seed,verified):
    p214,p215,p216=plans;items=[];bank={}
    def load(k,rel):
        root=Path(config[f'stage{k}_root']);meta=old.read_sealed(root/Path(rel).with_suffix('.receipt.json'))['metadata']
        v=source.bound(root,dict(artifact_sha256=catalog[f'stage{k}']),rel,meta,verified.setdefault(str(k),{}))
        if meta['plan_fingerprint']!={214:p214,215:p215,216:p216}[k]['fingerprint']:raise ValueError('Wrong cached plan')
        return v,meta
    for split in ('fit','held'):
        for g in fold[split]:
            spec=next(e for e in p214['entries'] if e['group']==g);seq=spec['sequence'];gt=entries[seq][1];pieces=[]
            frames=[int(t) for t in c.protocol.uniform_frames(len(gt)-2) if np.isfinite(gt[t+1]).all() and (gt[t+1,2:]>0).all()] if split=='fit' else [r['frame'] for r in p216['selection'] if r['sequence']==seq]
            idx=np.asarray(frames)-1
            for origin in (('nn','mm') if split=='fit' else source.prior.ORIGINS):
                if split=='fit':
                    v,m=load(214,f"fit_rollouts/f{fold['fold']}__s{seed}__{seq}__{origin}.npz")
                    if any(m[k]!=w for k,w in dict(fold=fold['fold'],seed=seed,group=g,mode=origin,fit=fold['fit'],held=fold['held'],samples=frames,role='outer_fit_only').items()):raise ValueError('Fit identity changed')
                    np.testing.assert_array_equal(v['sample_frame'],frames);x=v['sample_features'];raw=v['reference_boxes'][idx];hw=v['image_hw'][idx]
                else:
                    v,m=load(216,f'full/{seq}__s{seed}__{origin}.npz')
                    if any(m[k]!=w for k,w in dict(seed=seed,sequence=seq,origin=origin,frames=frames).items()):raise ValueError('Held identity changed')
                    path=source.prior.source.previous.path_for(Path(config['stage215_root']),'full',seq,seed,origin)
                    trace,tm=load(215,path.relative_to(config['stage215_root']).as_posix())
                    if any(tm[k]!=w for k,w in dict(fold=fold['fold'],seed=seed,sequence=seq,arm=origin,phase='full').items()):raise ValueError('Trace identity changed')
                    np.testing.assert_array_equal(v['current_reference'],trace['reference_boxes'][idx]);x=v['features'];raw=v['current_reference'];hw=trace['image_hw'][idx]
                data=dict(x=x,raw=raw,target=gt[np.asarray(frames)+1],image_hw=hw);items.append((split,g,origin,data));pieces.append(data)
            if split=='fit':bank[g]={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
    return bank,items


def summarize(rows):
    summaries=[];pairs=[];idx={}
    for split,origin in sorted({(r['split'],r['origin']) for r in rows}):
        groups=sorted({r['group'] for r in rows if r['split']==split and r['origin']==origin})
        for arm in METHODS:
            for g in groups:
                rr=[r for r in rows if (r['split'],r['origin'],r['arm'],r['group'])==(split,origin,arm,g)]
                if sorted(r['seed'] for r in rr)!=c.SEEDS:raise ValueError('Incomplete seeds')
                idx[split,origin,arm,g]=float(np.mean([r['regression'] for r in rr]))
            summaries.append(dict(split=split,origin=origin,arm=arm,groups=len(groups),regression=float(np.mean([idx[split,origin,arm,g] for g in groups]))))
    groups=sorted({r['group'] for r in rows})
    for split in ('fit','held'):
        origins=('nn','mm') if split=='fit' else source.prior.ORIGINS[:2]
        for a,b in (('linear','mlp'),*[(a,'zero') for a in ARMS]):
            pairs.append(dict(split=split,contrast=a+'_minus_'+b,primary=(split,a,b)==('held','linear','mlp'),favorable_direction='negative',**source.loss_contrast([np.mean([idx[split,o,a,g]-idx[split,o,b,g] for o in origins]) for g in groups])))
    return summaries,pairs


def experiment(root,plan,config,plans,catalog,entries,source_tables,torch):
    if (root/'stage218_decision.json').exists():verify(root);return
    start=perf_counter();tables={k:[] for k in TABLES};verified={};artifacts={}
    progress=OverallProgress('Stage-218',len(plan['folds'])*len(c.SEEDS)*(3*256+1)+1,unit='steps');progress.report('Cached CPU matched training')
    for fold in plan['folds']:
        for seed in c.SEEDS:
            bank,items=data_for(config,plans,catalog,entries,fold,seed,verified);models={}
            for arm in ARMS:
                model,scope,history=train(root,plan,torch,bank,fold,seed,arm,progress);models[arm]=model;tables['scope'].append(scope);tables['training'].extend(history)
                old.record(root,root/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt",artifacts)
            path=Path(config['stage214_root'])/'training'/f"f{fold['fold']}__s{seed}__forecast.pt"
            for pp in (path,path.with_suffix('.receipt.json')):
                rel=pp.relative_to(config['stage214_root']).as_posix()
                if file_hash(pp)!=catalog['stage214'][rel]:raise ValueError('Changed frozen source model')
            meta=old.read_sealed(path.with_suffix('.receipt.json'))['metadata'];saved=old.load_model(torch,path,meta);state=saved['state']
            for scope in tables['scope'][-3:]:
                if any(scope[k]!=meta[k] for k in ('draws_sha256','normalization_sha256')):raise ValueError('Original training opportunities differ')
            reference=c.build(torch,len(state['mean']),state['mean'],state['std'],seed);reference.load_state_dict(state);reference.eval().requires_grad_(False)
            np.testing.assert_array_equal(reference.mean.numpy(),models['mlp'].mean.numpy());np.testing.assert_array_equal(reference.std.numpy(),models['mlp'].std.numpy())
            models['frozen_forecast']=reference;digests={k:old.model_digest(v) for k,v in models.items()}
            for split,g,origin,data in items:
                tables['support'].append(dict(fold=fold['fold'],seed=seed,split=split,group=g,origin=origin,samples=len(data['x'])))
                for arm in METHODS:
                    with torch.no_grad():delta=np.zeros((len(data['x']),3),np.float32) if arm=='zero' else models[arm](torch.tensor(data['x'],dtype=torch.float32)).numpy()
                    losses=source.evaluate(torch,delta,data,'forecast',plan['search_factor'])
                    if arm in ('zero','frozen_forecast'):
                        r=next(r for r in source_tables['group_metrics'] if (r['split'],r['group'],int(r['seed']),r['origin'],r['family'],r['method'])==(split,g,seed,origin,'forecast','zero' if arm=='zero' else 'mlp'))
                        if any(losses[k]!=float(r[k]) for k in losses):raise ValueError('Stage-217 loss replay differs')
                        tables['source_replay'].append(dict(split=split,group=g,seed=seed,origin=origin,arm=arm,exact=True))
                    tables['group_metrics'].append(dict(fold=fold['fold'],seed=seed,split=split,group=g,origin=origin,arm=arm,**losses))
            if digests!={k:old.model_digest(v) for k,v in models.items()}:raise ValueError('Evaluation changed weights')
            progress.advance()
    tables['policy_summary'],tables['paired']=summarize(tables['group_metrics']);tables['execution']=[dict(device='cpu',seconds=perf_counter()-start,image_reads=0,cuda_calls=0)]
    publish_identical(root/'source_arrays.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],files=verified)));artifacts['source_arrays.json']=file_hash(root/'source_arrays.json')
    hashes={}
    for n,rr in tables.items():write_csv(root/f'stage218_{n}.csv',rr);hashes[f'stage218_{n}.csv']=file_hash(root/f'stage218_{n}.csv')
    publish_identical(root/'stage218_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_MATCHED_CAPACITY',plan_fingerprint=plan['fingerprint'],training_runs=len(tables['scope']),training_updates=len(tables['training']),new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,primary=[r for r in tables['paired'] if r['primary']],table_sha256=hashes,artifact_sha256=artifacts)))
    progress.advance();progress.report('Complete');verify(root)


def verify(root):
    root=Path(root);p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage218_decision.json')
    if d['schema']!=p['schema'] or d['schema']!=SCHEMA or d['status']!='REVIEW_MATCHED_CAPACITY' or d['plan_fingerprint']!=p['fingerprint'] or d['training_runs']!=12 or d['training_updates']!=3072 or len(d['primary'])!=1 or set(d['table_sha256'])!={f'stage218_{n}.csv' for n in TABLES} or any(d[k]!=0 for k in ('new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Incomplete result')
    for n,h in (d['table_sha256']|d['artifact_sha256']).items():
        pp=(root/n).resolve()
        if not pp.is_relative_to(root.resolve()) or file_hash(pp)!=h:raise ValueError('Changed artifact')
    for n,h in p['code_sha256'].items():
        if file_hash(REPO/n)!=h:raise ValueError('Changed experiment code')
    if file_hash(PROTOCOL)!=p['protocol_sha256']:raise ValueError('Changed protocol')
    print('Stage-218 verified; REVIEW, no new AUC or method adoption',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (214,215,216,217,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage217-bundle',type=Path,default=BUNDLE);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true');a=parser.parse_args();root=a.output_dir.resolve()
    if a.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    keys=(214,215,216,217,*sources.SOURCES)
    if any(getattr(a,f'stage{k}_root') is None for k in keys):parser.error('Need all frozen cache roots')
    c.protocol.safe_output(root,[getattr(a,f'stage{k}_root') for k in keys]+[source.RESULTS,a.stage217_bundle.parent])
    with writer_lock(root):
        if (root/'stage218_decision.json').exists():verify(root);return
        d,p,t=bundle(a.stage217_bundle);source.verify(a.stage217_root)
        if old.read_sealed(a.stage217_root/'manifest.json')!=p or old.read_sealed(a.stage217_root/'stage217_decision.json')!=d:raise ValueError('Physical Stage-217 differs')
        receipt=old.read_sealed(a.stage217_root/'source_arrays.json');catalog=receipt['source_files']
        if receipt['plan_fingerprint']!=p['fingerprint']:raise ValueError('Changed source catalog identity')
        plans=[]
        for k in (214,215,216):
            pk=old.read_sealed(getattr(a,f'stage{k}_root')/'manifest.json');dk=old.read_sealed(getattr(a,f'stage{k}_root')/f'stage{k}_decision.json')
            if p[f'source_stage{k}']!=dk['fingerprint'] or dk['plan_fingerprint']!=pk['fingerprint'] or any(pk[n]!=p[n] for n in ('entries','folds','search_factor')):raise ValueError('Source lineage differs')
            if any(dk['artifact_sha256'].get(n)!=h for n,h in catalog[f'stage{k}'].items()):raise ValueError('Source catalog changed')
            plans.append(pk)
        code={n:file_hash(REPO/n) for n in p['code_sha256']}
        for n in ('tools/stage218_capacity.py','tools/run_stage218_capacity.sh'):code[n]=file_hash(REPO/n)
        plan=old.sealed(dict(schema=SCHEMA,source_stage217=d['fingerprint'],entries=p['entries'],folds=p['folds'],search_factor=p['search_factor'],design=c.DESIGN,arms=list(ARMS),device='cpu',code_sha256=code,protocol_sha256=file_hash(PROTOCOL)))
        publish_identical(root/'manifest.json',plan);config={f'stage{k}_root':str(getattr(a,f'stage{k}_root').resolve()) for k in (214,215,216)};config['source_paths']={str(k):str(getattr(a,f'stage{k}_root')) for k in sources.SOURCES}
        entries,dataset=source.runtime.load_entries(config,plan);c.protocol.safe_output(root,[dataset])
        import torch
        torch.set_num_threads(1);experiment(root,plan,config,plans,catalog,entries,t,torch)


BUNDLE_CANONICAL='8b2f388b13cfccd3b122673e8f99e26633c4009b4737e4cf182482da745f8ced'
if __name__=='__main__':main()
