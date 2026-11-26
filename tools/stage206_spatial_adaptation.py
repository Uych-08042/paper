"""Train spatial SGLA adapters with pre-adapter native routing; full-video OOF pilot."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import numpy as np
import stage194_module_ablation as old
import stage195_router_objective_audit as inputs
import stage206_spatial_common as common
import stage206_spatial_model as spatial
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv

SCHEMA=common.SCHEMA
SOURCE205='074e680f182d8752a1796ffb7ece24458d17d0994f82c3235723a2e36655da7c'
BUNDLE205='0f342f0b40ad5f877045b55c0e19f39255848978418031a63f09ac7961e8972d'
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage205_results_20261126T013956Z_1605754.txt'
PROTOCOL_FILE=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE206_NATIVE_ROUTED_SPATIAL_ADAPTATION.md'
TABLES=('scope_summary','training_history','reference_audit','smoke_summary','readout_video','policy_summary','paired','seed_video_summary','routing')


def source205(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE205:raise ValueError('Need exact reviewed Stage-205 bundle')
    matches=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():matches[i+1].start() if i+1<len(matches) else len(raw)] for i,m in enumerate(matches)}
    d,p=json.loads(parts['stage205_decision.json']),json.loads(parts['manifest.json'])
    if len(parts)!=len(matches) or set(parts)!={'stage205_decision.json','manifest.json',*d['table_sha256']}:
        raise ValueError('Changed source inventory')
    for value in (d,p):
        if fingerprint({k:v for k,v in value.items() if k!='fingerprint'})!=value['fingerprint']:raise ValueError('Changed source fingerprint')
    if d['fingerprint']!=SOURCE205 or d['plan_fingerprint']!=p['fingerprint'] or d['training_runs']!=228 or d['confirmation_groups_opened']!=0:
        raise ValueError('Unexpected source run')
    for name,sha in d['table_sha256'].items():
        b=parts[name]
        if not any(hashlib.sha256(v).hexdigest()==sha for v in (b,b[:-1],b[:-2])):raise ValueError('Changed source table '+name)
    for name,sha in p['code_sha256'].items():
        if file_hash(REPO/'tools'/name)!=sha:raise ValueError('Frozen source code changed '+name)
    import stage205_crossfit_shrinkage as previous
    if file_hash(previous.PROTOCOL_FILE)!=p['protocol_sha256']:raise ValueError('Changed Stage-205 protocol')
    return d,p


def sync(rt):
    if next(rt.net.parameters()).device.type=='cuda':rt.torch.cuda.synchronize()


def guard(root):
    if (root/'nonfinite_failure.json').exists():raise ValueError('Recorded nonfinite failure; review before using a new root')


def nonfinite(root,meta,step):
    publish_identical(root/'nonfinite_failure.json',old.sealed(dict(schema=SCHEMA,metadata=meta,step=step,status='FAILED_NONFINITE')))
    raise FloatingPointError('Nonfinite Stage-206 training; diagnostic sealed')


def fit(root,plan,native,bank,refs,fold,seed,arm,progress,artifacts):
    rt,torch=native.rt,native.rt.torch;device=next(rt.net.parameters()).device
    model=spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,seed,arm).to(device).eval()
    initial=old.model_digest(model)
    draws=common.old.draws_for(fold,seed,{g:len(v['frame_id']) for g,v in bank.items()})
    meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],
        draws_sha256=fingerprint(draws),initial_sha256=initial,updates=len(draws))
    path=root/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt"
    saved=old.load_model(torch,path,meta)
    if saved is not None:
        model.load_state_dict(saved['state'],strict=True)
        if old.model_digest(model)!=saved['scope']['state_sha256']:raise ValueError('Changed committed adapter')
        old.record(root,path,artifacts);old.reused_work(progress,len(draws))
        return model.requires_grad_(False),saved['scope'],saved['history']
    optimizer=torch.optim.AdamW(model.parameters(),lr=common.DESIGN['learning_rate'],weight_decay=common.DESIGN['weight_decay'])
    history=[];maxima={k:0. for k in ('query','key','value','output')};start=perf_counter()
    for step,batch in enumerate(draws,1):
        if any(g not in fold['fit'] or g in fold['held'] for g,i in batch):raise ValueError('Held-source training')
        tokens=torch.as_tensor(np.stack([bank[g]['tokens'][i] for g,i in batch]),device=device,dtype=torch.float32)
        reference=torch.as_tensor(np.stack([refs[g] for g,i in batch]),device=device,dtype=torch.float32)
        target=torch.as_tensor(np.stack([bank[g]['target'][i] for g,i in batch]),device=device,dtype=torch.float32)
        if step==1 and not torch.equal(model.adapt(tokens,reference),tokens):raise ValueError('Nonidentity initial adapter')
        optimizer.zero_grad(set_to_none=True)
        loss,layers,calls=spatial.objective(rt,model,tokens,reference,target)
        if not bool(torch.isfinite(loss)):nonfinite(root,meta,step)
        loss.backward()
        try:norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),common.DESIGN['gradient_clip'],error_if_nonfinite=True))
        except RuntimeError as error:
            if 'non-finite' in str(error).lower() or any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in model.parameters()):nonfinite(root,meta,step)
            raise
        if any(v.grad is not None for v in rt.net.parameters()):raise ValueError('Gradient reached frozen native model')
        for name in maxima:maxima[name]=max(maxima[name],float(getattr(model,name).weight.grad.abs().max()))
        optimizer.step()
        if any(not bool(torch.isfinite(v).all()) for v in model.parameters()):nonfinite(root,meta,step)
        if any(not bool(torch.isfinite(v).all()) for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)):nonfinite(root,meta,step)
        history.append(dict(fold=fold['fold'],seed=seed,arm=arm,step=step,loss=float(loss.detach()),gradient_norm=norm,
            candidate_batch_calls=calls,candidate_examples=len(batch),selector_calls=1,
            **{f'L{k}_examples':int((layers==k).sum()) for k in range(7,13)}))
        progress.advance()
    sync(rt);native.assert_frozen()
    if any(v<=0 for v in maxima.values()) or old.model_digest(model)==initial:raise ValueError('Inactive spatial parameter path')
    scope=dict(fold=fold['fold'],seed=seed,arm=arm,fit_groups='|'.join(fold['fit']),held_groups='|'.join(fold['held']),
        updates=len(draws),examples=sum(len(v) for v in draws),trainable_parameters=sum(v.numel() for v in model.parameters()),
        initial_sha256=initial,state_sha256=old.model_digest(model),draws_sha256=meta['draws_sha256'],
        candidate_batch_calls=sum(v['candidate_batch_calls'] for v in history),candidate_examples=sum(v['candidate_examples'] for v in history),
        selector_calls=len(draws),frozen_native_unchanged=True,training_seconds=perf_counter()-start,
        **{f'maximum_{k}_gradient':v for k,v in maxima.items()})
    old.save_model(torch,path,dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scope=scope,history=history),meta)
    old.record(root,path,artifacts)
    return model.requires_grad_(False),scope,history


def trace(native,model,reference,files,initial,reader,dataset,progress):
    native.initialize(reader(dataset,files[0]),initial);parent=np.asarray(initial,float);rows=[]
    ctx=spatial.Installed(native,model,reference) if model is not None else nullcontext()
    with ctx:
        for frame,file in enumerate(files[1:],1):
            image=reader(dataset,file);namespace=native.rt.tracker.track.__func__.__globals__;sample=namespace['sample_target'];count=[0]
            def counted(*args,**kwargs):count[0]+=1;return sample(*args,**kwargs)
            sync(native.rt);start=perf_counter()
            with patch.dict(namespace,sample_target=counted):out=native.step(image,parent)
            sync(native.rt);elapsed=1000*(perf_counter()-start)
            if count[0]!=1:raise ValueError('Need one public search crop')
            if model is not None and (ctx.prefix_calls!=frame or ctx.route_calls!=frame or native.rt.selected_layer(ctx.last_scores)!=out['layer']):
                raise ValueError('Native pre-adapter routing or hook call mismatch')
            rows.append(dict(frame_id=frame,parent=parent.copy(),boxes=out['box'],layer=out['layer'],block_calls=out['block_calls'],
                head_calls=out['head_calls'],selector_calls=out['selector_calls'],crop_calls=count[0],pipeline_ms=elapsed))
            parent=out['box'];progress.advance()
    return {k:np.asarray([v[k] for v in rows]) for k in rows[0]}


def readout(root,plan,native,bank,refs,fold,seed,arm,model,progress,artifacts):
    rt,torch=native.rt,native.rt.torch;device=next(rt.net.parameters()).device;rows=[]
    for g,data in bank.items():
        path=root/'readout'/f"f{fold['fold']}__s{seed}__{arm}__{g}.npz"
        meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,group=g,model_sha256=old.model_digest(model))
        def compute():
            values=[];layers=[]
            with torch.no_grad():
                ref=torch.as_tensor(refs[g][None],device=device)
                for i,array in enumerate(data['tokens']):
                    tokens=torch.as_tensor(array[None],device=device);layer=int(spatial.chosen(rt,tokens)[0])
                    out=rt.candidate(model.adapt(tokens,ref),layer)
                    grid=common.old.dense_image_boxes(out['size_map'][0].cpu().numpy(),out['offset_map'][0].cpu().numpy(),
                        data['parent'][i],data['factor'][i],rt.tracker.params.search_size,data['image_hw'][i])
                    cell=int((out['score_map']*rt.tracker.output_window).flatten(1).argmax(1).item())
                    values.append(float(old.tracking_iou(grid[cell],data['truth'][i])));layers.append(layer);progress.advance()
            return dict(iou=np.asarray(values),layer=np.asarray(layers))
        value,reused=old.cache(path,meta,compute)
        if reused:old.reused_work(progress,len(data['frame_id']))
        if value['iou'].shape!=(len(data['frame_id']),) or not np.isfinite(value['iou']).all():raise ValueError('Bad readout cache')
        old.record(root,path,artifacts)
        rows.append(dict(fold=fold['fold'],seed=seed,arm=arm,group=g,pool='fit' if g in fold['fit'] else 'held',states=len(value['iou']),
            mean_iou=float(value['iou'].mean()),native_cached_mean_iou=float(old.tracking_iou(data['native_box'],data['truth']).mean()),
            cached_native_layer_differences=int((value['layer']!=data['native_layer']).sum())))
    return rows


def experiment(root,plan,native,entries,dataset,bank):
    rt,torch=native.rt,native.rt.torch;device=next(rt.net.parameters()).device
    n=sum(len(v['frame_id']) for v in bank.values());frames=sum(len(e[0])-1 for e in entries.values())
    progress=OverallProgress('Stage-206',4+12*common.DESIGN['updates']+12*n+7*frames+1,unit='work')
    progress.report('Identity smoke and fit-only spatial references');artifacts={};tables={k:[] for k in TABLES};refbank={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    for fold in plan['folds']:
        refs,mapping=common.references(bank,fold,rt.lens_z);refbank[fold['fold']]=refs
        path=root/'references'/f"fold{fold['fold']}.npz"
        groups=list(bank);meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],fit=fold['fit'])
        stored,_=old.cache(path,meta,lambda:dict(groups=np.asarray(groups),reference=np.stack([refs[g] for g in groups])))
        np.testing.assert_array_equal(stored['reference'],np.stack([refs[g] for g in groups]));old.record(root,path,artifacts)
        for g in bank:tables['reference_audit'].append(dict(fold=fold['fold'],group=g,donor=mapping[g],donor_in_fit=mapping[g] in fold['fit'],
            donor_differs_source=mapping[g]!=g,reference_sha256=hashlib.sha256(refs[g].tobytes()).hexdigest()))
    spec=plan['entries'][0];g,seq=spec['group'],spec['sequence'];files,gt,_,reader,_=entries[seq]
    native.initialize(reader(dataset,files[0]),gt[0]);original=native.step(reader(dataset,files[1]),gt[0]);progress.advance()
    for arm in spatial.ARMS:
        model=spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,common.DESIGN['seeds'][0],arm).to(device).eval()
        native.initialize(reader(dataset,files[0]),gt[0]);ref=torch.as_tensor(refbank[0][g][None],device=device)
        with spatial.Installed(native,model,ref):adapted=native.step(reader(dataset,files[1]),gt[0])
        error=float(np.abs(original['box']-adapted['box']).max())
        if error>.001 or original['layer']!=adapted['layer']:raise ValueError('Zero residual public path not native')
        tables['smoke_summary'].append(dict(arm=arm,zero_box_error=error,unchanged_layer=True,original_native_unchanged=True));progress.advance()
    models={};progress.report('Matched selected-branch spatial training')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            scopes=[]
            for arm in spatial.ARMS:
                model,scope,history=fit(root,plan,native,bank,refbank[fold['fold']],fold,seed,arm,progress,artifacts)
                models[fold['fold'],seed,arm]=model;tables['scope_summary'].append(scope);tables['training_history'].extend(history);scopes.append(scope)
            for field in ('initial_sha256','draws_sha256','examples','candidate_batch_calls','candidate_examples','selector_calls','trainable_parameters'):
                if len({s[field] for s in scopes})!=1:raise ValueError('Unmatched training '+field)
    progress.report('Fixed-parent fit and held readout')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            for arm in spatial.ARMS:
                model=models[fold['fold'],seed,arm];digest=old.model_digest(model)
                tables['readout_video'].extend(readout(root,plan,native,bank,refbank[fold['fold']],fold,seed,arm,model,progress,artifacts))
                if old.model_digest(model)!=digest:raise ValueError('Readout mutated model')
    progress.report('Fresh native and full OOF spatial tracking')
    for spec in plan['entries']:
        g,seq=spec['group'],spec['sequence'];fold=next(f['fold'] for f in plan['folds'] if g in f['held']);files,gt,_,reader,_=entries[seq]
        jobs=[('native',0,None)]+[(a,s,models[fold,s,a]) for s in common.DESIGN['seeds'] for a in spatial.ARMS];baseline=None
        for arm,seed,model in jobs:
            path=root/'evaluation'/f'{seq}__s{seed}__{arm}.npz';digest=old.model_digest(model) if model is not None else None
            meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,model_sha256=digest)
            ref=torch.as_tensor(refbank[fold][g][None],device=device)
            value,reused=old.cache(path,meta,lambda:trace(native,model,ref,files,gt[0],reader,dataset,progress))
            if reused:old.reused_work(progress,len(files)-1)
            old.audit_trace(value,gt[0],len(files)-1)
            if model is not None and old.model_digest(model)!=digest:raise ValueError('Full tracking mutated adapter')
            if arm=='native':baseline=value
            tables['seed_video_summary'].append(old.describe_trace(value,baseline,gt,seq,arm,seed,fold));old.record(root,path,artifacts)
            for layer in range(7,13):tables['routing'].append(dict(sequence=seq,fold=fold,seed=seed,arm=arm,layer=layer,frames=int((value['layer']==layer).sum())))
        native.assert_frozen()
    tables['policy_summary'],tables['paired']=common.summarize(tables['seed_video_summary'])
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage206_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete overall progress budget')
    publish_identical(root/'stage206_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_NATIVE_ROUTED_SPATIAL_ADAPTATION',
        plan_fingerprint=plan['fingerprint'],source_stage205=SOURCE205,source_groups=len(bank),training_runs=len(models),
        training_updates=len(tables['training_history']),full_trajectories=len(tables['seed_video_summary']),
        original_native_unchanged=True,confirmation_groups_opened=0,primary_contrasts=[r for r in tables['paired'] if r['primary']],
        training_budget='Matched parameters, draws, updates, selected candidate examples; timings instrumented, no equal latency claim',
        table_sha256=hashes,artifact_sha256=artifacts,next_step='Require matched-over-native/control evidence plus stable and recovery protection; no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    guard(root);d=old.read_sealed(root/'stage206_decision.json');p=old.read_sealed(root/'manifest.json')
    if (d['schema']!=SCHEMA or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage205']!=SOURCE205
            or d['status']!='REVIEW_NATIVE_ROUTED_SPATIAL_ADAPTATION' or d['confirmation_groups_opened']!=0
            or d['source_groups']!=38 or d['training_runs']!=12 or d['training_updates']!=3072 or d['full_trajectories']!=266
            or set(d['table_sha256'])!={f'stage206_{k}.csv' for k in TABLES}):raise ValueError('Incomplete Stage-206')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=sha:raise ValueError('Changed result '+name)
    for name,sha in p['code_sha256'].items():
        if file_hash(REPO/name)!=sha:raise ValueError('Changed dependency '+name)
    if file_hash(PROTOCOL_FILE)!=p['protocol_sha256']:raise ValueError('Changed Stage-206 protocol')
    print('Stage-206 verified; full development OOF pilot, confirmation remains sealed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (194,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage205-bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (194,76,89,93,121,170,188,191)):
        parser.error('Original checkpoint, Stage-194 features and historical dataset sources required')
    import stage192_development_grid_rescue as prior
    paths={k:getattr(args,f'stage{k}_root').resolve() for k in prior.SOURCES}
    common.old.safe_output(root,[*paths.values(),args.stage194_root,args.checkpoint,args.stage205_bundle.parent])
    with writer_lock(root):
        guard(root)
        if (root/'stage206_decision.json').exists():verify(root);return
        d205,p205=source205(args.stage205_bundle)
        p194,d194,reader,_=inputs.load_source(args.stage194_root)
        for name,sha in p194['code_sha256'].items():
            if file_hash(REPO/name)!=sha:raise ValueError('Changed frozen Stage-194 dependency '+name)
        if d205['source_stage194']!=d194['fingerprint'] or p205['folds']!=p194['folds']:raise ValueError('Source feature split mismatch')
        entries,dataset,expected=prior.load_sources(paths);common.old.safe_output(root,[dataset])
        for spec in p194['entries']:
            files,gt,_,_,_=entries[spec['sequence']]
            if fingerprint(files)!=spec['frames_sha256'] or fingerprint(np.asarray(gt).tolist())!=spec['truth_sha256']:
                raise ValueError('Dataset roster changed')
        import torch
        torch.set_num_threads(1)
        runtime=prior.old.base.s103.s76.device_runtime(dict(runtime_identity=p194['runtime']),torch,args.gpu)
        differences=prior.old.base.s109.check_runtime(p194['runtime'],runtime)
        code={**p194['code_sha256'],**{'tools/'+k:v for k,v in p205['code_sha256'].items()}}
        for file in (__file__,spatial.__file__,common.__file__,inputs.__file__):code[Path(file).resolve().relative_to(REPO).as_posix()]=file_hash(file)
        plan=old.sealed(dict(schema=SCHEMA,source_stage194=d194['fingerprint'],source_stage205=d205['fingerprint'],source205_bundle_sha256=file_hash(args.stage205_bundle),
            folds=p194['folds'],entries=p194['entries'],design=common.DESIGN,arms=list(common.ARMS),runtime=runtime,runtime_differences=differences,
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL_FILE),
            adaptation='X+Wo softmax(Q LN(X) (K LN(reference))^T/sqrt(D)) V LN(reference); template rows unchanged',
            routing='original MLP on unadapted L6 channel zero; one original selector call; only selected candidate trained',
            control='search 2x2 average pooling gives 64 reference tokens; same Q/K/V/O and initialization; wrong reference donor from fit only',
            evidence='historical development OOF; same-card fresh native; no independent confirmation or new dynamic router claim'))
        publish_identical(root/'manifest.json',plan)
        args.parameter=runtime['tracker']['parameter'];args.mode='stage206_spatial_adaptation'
        native,_=prior.old.base.s103.s76.startup.checked_runtime(args,dict(fingerprint=plan['fingerprint'],runtime_identity=runtime,runtime=runtime['tracker']),root)
        rt=native.rt
        try:
            if rt.lens_z!=64 or rt.lens_x!=256 or rt.identity['embed_dim']!=192 or rt.backbone.cat_mode!='direct':raise ValueError('Need audited 64+256 by 192 layout')
            bank={}
            for spec in p194['entries']:
                seq=spec['sequence'];data=reader.arrays(f'features/{seq}.npz',dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix'))
                np.testing.assert_array_equal(data['frame_id'],spec['sample_frames'])
                if data['tokens'].shape!=(len(data['frame_id']),320,192) or not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Bad cached feature')
                bank[spec['group']]=data
            publish_identical(root/'source_receipt.json',old.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],verified=reader.verified)))
            experiment(root,plan,native,entries,dataset,bank)
        finally:
            for handle in native.handles:handle.remove()


if __name__=='__main__':main()
