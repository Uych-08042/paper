"""Budget-matched native versus own-provider crop-state training, full OOF."""
from contextlib import ExitStack
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import numpy as np
import stage206_spatial_adaptation as previous
import stage207_state_common as common
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,fingerprint,write_csv

old,spatial=previous.old,previous.spatial
SCHEMA=common.SCHEMA
SOURCE206='0efb6e79719b9604b6438178738f038fe97f54e2037293fd89bb96dcda285385'
PLAN206='28b16c4366b878c0f6cc836492d1b651f3693d5af35e0f35d10c54ca8ba67076'
BUNDLE206='c2a3aae104ad211ad6b11358f3c8d43fd39f58fce2419890e2bf1a0c192478a5'
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage206_results_20261126T042709Z_1653346.txt'
PROTOCOL_FILE=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE207_PAIRED_PROVIDER_STATE_TRAINING.md'
TABLES=('source_replay','provider_scope','support','sampling','scope_summary','training_history','control_replay','readout_video',
        'policy_summary','paired','seed_video_summary','routing')


def guard(root):
    previous.guard(root)
    if list((root/'conditions').glob('*/nonfinite_failure.json')):raise ValueError('Recorded nonfinite student failure; review before continuing')


def source206(bundle):
    # Parsing and byte verification do not require the original GPU or NPZ files.
    raw=Path(bundle).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE206:raise ValueError('Need exact reviewed Stage-206 bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage206_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v['fingerprint']!=fingerprint({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if (d['fingerprint']!=SOURCE206 or p['fingerprint']!=PLAN206 or d['plan_fingerprint']!=PLAN206
            or d['training_runs']!=12 or d['full_trajectories']!=266 or d['confirmation_groups_opened']!=0
            or len(ms)!=len(parts) or set(parts)!={'stage206_decision.json','manifest.json',*d['table_sha256']}
            or set(d['table_sha256'])!={f'stage206_{k}.csv' for k in previous.TABLES}):raise ValueError('Unexpected source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table '+name)
        tables[name.removeprefix('stage206_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    return d,p,tables


def source_file(root,decision,relative,verified):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or relative not in decision['artifact_sha256'] or file_hash(path)!=decision['artifact_sha256'][relative]:
        raise ValueError('Unbound or changed provider artifact '+relative)
    verified[relative]=file_hash(path);return path


def load_provider(torch,rt,root,decision,source_plan,tables,fold,seed,verified):
    relative=f"training/f{fold['fold']}__s{seed}__matched_spatial.pt"
    path=source_file(root,decision,relative,verified)
    receipt=old.read_sealed(source_file(root,decision,str(Path(relative).with_suffix('.receipt.json').as_posix()),verified))
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=receipt['metadata']
    if (receipt['sha256']!=verified[relative] or meta!=payload['metadata'] or meta['plan_fingerprint']!=source_plan['fingerprint']
            or meta['fold']!=fold['fold'] or meta['seed']!=seed or meta['arm']!='matched_spatial'
            or meta['fit']!=fold['fit'] or meta['held']!=fold['held'] or meta['updates']!=256):raise ValueError('Provider split or training identity changed')
    original=next(v for v in tables['scope_summary'] if int(v['fold'])==fold['fold'] and int(v['seed'])==seed and v['arm']=='matched_spatial')
    previous.inputs.compare_scope(payload['scope'],original)
    model=spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,seed,'matched_spatial').to(next(rt.net.parameters()).device).eval()
    model.load_state_dict(payload['state'],strict=True);model.requires_grad_(False)
    if old.model_digest(model)!=payload['scope']['state_sha256']:raise ValueError('Provider weights changed')
    return model,dict(fold=fold['fold'],seed=seed,fit_groups='|'.join(fold['fit']),held_groups='|'.join(fold['held']),
        model_sha256=old.model_digest(model),file_sha256=verified[relative])


def collect_provider(native,model,files,initial,reader,dataset,frames,progress):
    """Actor only receives first-frame truth. Subsequent labels are attached later."""
    rt=native.rt;native.initialize(reader(dataset,files[0]),initial)
    frame_set=set(map(int,frames));parent=np.asarray(initial,float);trajectory=[];samples=[];current=[0];captured=[]
    def raw_prefix(module,args,output):
        if current[0] in frame_set:captured.append(output.detach().cpu().numpy()[0].copy())
    with ExitStack() as stack:
        # This hook must run BEFORE Installed.prefix transforms the output.
        stack.callback(rt.backbone.blocks[5].register_forward_hook(raw_prefix).remove)
        installed=stack.enter_context(spatial.Installed(native,model,None))
        for frame,file in enumerate(files[1:],1):
            current[0]=frame;captured.clear();image=reader(dataset,file)
            namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target'];factors=[]
            def counted(*args,**kwargs):
                value=sampler(*args,**kwargs);factors.append(float(value[1]));return value
            with patch.dict(namespace,sample_target=counted):out=native.step(image,parent)
            if len(factors)!=1 or installed.prefix_calls!=frame or installed.route_calls!=frame or rt.selected_layer(installed.last_scores)!=out['layer']:
                raise ValueError('Provider must use one original search, prefix and pre-adapter route')
            trajectory.append(dict(frame_id=frame,parent=parent.copy(),boxes=out['box'],layer=out['layer'],block_calls=out['block_calls'],
                head_calls=out['head_calls'],selector_calls=out['selector_calls'],crop_calls=1,pipeline_ms=float(out.get('network_ms',0.))))
            if frame in frame_set:
                if len(captured)!=1:raise ValueError('Missing unadapted provider prefix')
                samples.append(dict(frame_id=frame,tokens=captured[0],parent=parent.copy(),factor=factors[0],image_hw=image.shape[:2],
                    native_box=out['box'],native_layer=out['layer']))
            parent=out['box'];progress.advance()
    trace={k:np.asarray([v[k] for v in trajectory]) for k in trajectory[0]}
    old.audit_trace(trace,initial,len(files)-1)
    features={k:np.asarray([v[k] for v in samples]) for k in samples[0]}
    np.testing.assert_array_equal(features['frame_id'],frames)
    return {**{'trace_'+k:v for k,v in trace.items()},**{'sample_'+k:v for k,v in features.items()}}


def attach_truth(value,truth,search_size):
    data={k.removeprefix('sample_'):v for k,v in value.items() if k.startswith('sample_')}
    data['truth']=np.asarray(truth)[data['frame_id']].copy()
    data['target']=np.stack([old.executed_crop_roi(t,p,f,search_size) for t,p,f in zip(data['truth'],data['parent'],data['factor'])])
    return data


def register_nested(root,subroot,artifacts,local):
    for name,sha in local.items():artifacts[(subroot/name).relative_to(root).as_posix()]=sha


def replay_native(root,plan,native,entries,dataset,bank,progress,artifacts):
    """On a changed GPU, use current-runtime tokens for BOTH sides of training."""
    result={};rows=[];rt=native.rt
    for spec in plan['entries']:
        g,seq=spec['group'],spec['sequence'];data=bank[g];path=root/'native_replay'/(seq+'.npz')
        meta=dict(plan_fingerprint=plan['fingerprint'],group=g,role='current_runtime_native_parents')
        def compute():
            files,truth,_,reader,_=entries[seq];native.initialize(reader(dataset,files[0]),truth[0]);tokens=[];boxes=[];layers=[]
            captured=[]
            def hook(module,args,output):captured.append(output.detach().cpu().numpy()[0].copy())
            handle=rt.backbone.blocks[5].register_forward_hook(hook)
            try:
                for i,frame in enumerate(data['frame_id']):
                    captured.clear();rt.tracker.frame_id=int(frame)-1;image=reader(dataset,files[int(frame)])
                    namespace=rt.tracker.track.__func__.__globals__;sampler=namespace['sample_target'];factors=[]
                    def counted(*args,**kwargs):
                        value=sampler(*args,**kwargs);factors.append(float(value[1]));return value
                    with patch.dict(namespace,sample_target=counted):out=native.step(image,data['parent'][i])
                    if len(captured)!=1 or len(factors)!=1:raise ValueError('Bad native replay calls')
                    np.testing.assert_allclose(factors[0],data['factor'][i],rtol=0,atol=1e-12)
                    np.testing.assert_array_equal(image.shape[:2],data['image_hw'][i])
                    tokens.append(captured[0]);boxes.append(out['box']);layers.append(out['layer']);progress.advance()
            finally:handle.remove()
            return dict(data,tokens=np.asarray(tokens),native_box=np.asarray(boxes),native_layer=np.asarray(layers))
        value,reused=old.cache(path,meta,compute)
        if reused:old.reused_work(progress,len(data['frame_id']))
        for field in ('frame_id','truth','target','parent','factor','image_hw'):np.testing.assert_array_equal(value[field],data[field])
        old.record(root,path,artifacts);result[g]=value
        rows.append(dict(group=g,recomputed=True,states=len(data['frame_id']),token_max_error=float(np.abs(value['tokens']-data['tokens']).max()),
            box_max_error=float(np.abs(value['native_box']-data['native_box']).max()),layer_differences=int((value['native_layer']!=data['native_layer']).sum())))
    return result,rows


def experiment(root,plan,native,entries,dataset,native_bank,providers,provider_rows):
    torch,rt=native.rt.torch,native.rt;device=next(rt.net.parameters()).device
    frames=sum(len(v[0])-1 for v in entries.values());states=sum(len(v['frame_id']) for v in native_bank.values())
    # Fit-provider collection twice over the full roster + 9 evaluation policies.
    progress=OverallProgress('Stage-207',2+2*frames+76+12*256+12*states+9*frames+1+(0 if plan['same_source_runtime'] else states),unit='work')
    progress.report('Identity smoke and fit-only provider trajectories')
    artifacts={};tables={k:[] for k in TABLES};tables['provider_scope']=provider_rows
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    if not plan['same_source_runtime']:
        progress.report('Replaying native parents on current GPU for all student conditions')
        native_bank,tables['source_replay']=replay_native(root,plan,native,entries,dataset,native_bank,progress,artifacts)
    else:
        tables['source_replay']=[dict(group=g,recomputed=False,states=len(v['frame_id']),token_max_error=0.,box_max_error=0.,layer_differences=0) for g,v in native_bank.items()]
    progress.phase='Identity smoke and fit-only provider trajectories'
    first=plan['entries'][0];files,gt,_,reader,_=entries[first['sequence']]
    native.initialize(reader(dataset,files[0]),gt[0]);baseline=native.step(reader(dataset,files[1]),gt[0]);progress.advance()
    zero=spatial.build(torch,rt.identity['embed_dim'],rt.lens_z,rt.search_side,common.DESIGN['seeds'][0],'matched_spatial').to(device).eval()
    native.initialize(reader(dataset,files[0]),gt[0])
    with spatial.Installed(native,zero,None):same=native.step(reader(dataset,files[1]),gt[0])
    if not np.array_equal(baseline['box'],same['box']) or baseline['layer']!=same['layer']:raise ValueError('Zero adapter changed public path')
    publish_identical(root/'smoke.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],zero_box_error=0.,same_layer=True)))
    artifacts['smoke.json']=file_hash(root/'smoke.json');progress.advance()
    conditions={};empty=[]
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            provider=providers[fold['fold'],seed];digest=old.model_digest(provider);own={}
            for g in fold['fit']:
                spec=next(e for e in plan['entries'] if e['group']==g);seq=spec['sequence'];files,truth,_,reader,_=entries[seq]
                path=root/'providers'/f"f{fold['fold']}__s{seed}__{seq}.npz"
                meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,group=g,provider_sha256=digest,role='outer_fit_only')
                value,reused=old.cache(path,meta,lambda:collect_provider(native,provider,files,truth[0],reader,dataset,native_bank[g]['frame_id'],progress))
                if reused:old.reused_work(progress,len(files)-1)
                old.record(root,path,artifacts)
                own[g]=attach_truth(value,truth,rt.tracker.params.search_size)
                if old.model_digest(provider)!=digest:raise ValueError('Provider mutated during collection')
                progress.advance()
            original={g:native_bank[g] for g in fold['fit']}
            a,b,support,sampling=common.paired_bank(original,own,rt.tracker.params.search_size)
            conditions[fold['fold'],seed]={'native_all':original,'native_pair':a,'own_pair':b}
            for row in support:
                tables['support'].append(dict(fold=fold['fold'],seed=seed,**row))
                if row['paired_states']==0:empty.append(dict(fold=fold['fold'],seed=seed,group=row['group']))
            tables['sampling'].extend(dict(fold=fold['fold'],seed=seed,**row) for row in sampling)
    support_hashes={}
    for name in ('source_replay','provider_scope','support','sampling'):
        path=root/f'stage207_{name}.csv';write_csv(path,tables[name]);support_hashes[path.name]=file_hash(path)
    support_decision=old.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],status='INSUFFICIENT_PAIRED_SUPPORT' if empty else 'SUPPORT_READY',
        empty_groups=empty,training_started=False,table_sha256=support_hashes,artifact_sha256=dict(artifacts)))
    publish_identical(root/'stage207_support_decision.json',support_decision)
    artifacts['stage207_support_decision.json']=file_hash(root/'stage207_support_decision.json')
    if empty:
        progress.report('Stopped: a fit source has no paired support; no student training')
        print('Stage-207: INSUFFICIENT_PAIRED_SUPPORT; return support bundle, do not drop sources',flush=True);return
    models={};progress.report('Matched from-zero student training')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            scopes=[]
            for condition in common.TRAINED:
                bank=conditions[fold['fold'],seed][condition]
                # matched_spatial ignores this reference argument; no fitted donor is used by students.
                refs={g:np.zeros((rt.lens_z,rt.identity['embed_dim']),np.float32) for g in bank}
                subroot=root/'conditions'/condition;local={}
                model,scope,history=previous.fit(subroot,plan,native,bank,refs,fold,seed,'matched_spatial',progress,local)
                register_nested(root,subroot,artifacts,local)
                models[fold['fold'],seed,condition]=model
                tables['scope_summary'].append(dict(training_condition=condition,**scope));scopes.append(scope)
                tables['training_history'].extend(dict(training_condition=condition,**v) for v in history)
                if condition=='native_all':
                    provider=providers[fold['fold'],seed]
                    error=max(float((v.detach()-provider.state_dict()[k]).abs().max()) for k,v in model.state_dict().items())
                    exact=old.model_digest(model)==old.model_digest(provider)
                    if plan['same_source_runtime'] and not exact:raise ValueError('Same-runtime native_all must exactly reproduce Stage-206 weights')
                    tables['control_replay'].append(dict(fold=fold['fold'],seed=seed,same_source_runtime=plan['same_source_runtime'],
                        exact_state_match=exact,maximum_parameter_error=error,model_sha256=old.model_digest(model),provider_sha256=old.model_digest(provider)))
            if scopes[1]['draws_sha256']!=scopes[2]['draws_sha256']:raise ValueError('Paired frame schedules differ')
            if len({v['initial_sha256'] for v in scopes})!=1:raise ValueError('Unmatched initial state')
    progress.report('Native-parent fit/held readout of frozen students')
    for fold in plan['folds']:
        for seed in common.DESIGN['seeds']:
            for condition in common.TRAINED:
                model=models[fold['fold'],seed,condition];subroot=root/'conditions'/condition;local={}
                refs={g:np.zeros((rt.lens_z,rt.identity['embed_dim']),np.float32) for g in native_bank}
                rows=previous.readout(subroot,plan,native,native_bank,refs,fold,seed,'matched_spatial',model,progress,local)
                tables['readout_video'].extend(dict(training_condition=condition,**v) for v in rows);register_nested(root,subroot,artifacts,local)
    progress.report('Fresh native/provider and full student OOF trajectories')
    for spec in plan['entries']:
        g,seq=spec['group'],spec['sequence'];fold=next(f['fold'] for f in plan['folds'] if g in f['held'])
        files,truth,_,reader,_=entries[seq]
        jobs=[('native',0,None)]+[('frozen_provider',s,providers[fold,s]) for s in common.DESIGN['seeds']]
        jobs += [(a,s,models[fold,s,a]) for s in common.DESIGN['seeds'] for a in common.TRAINED]
        baseline=None
        for arm,seed,model in jobs:
            path=root/'evaluation'/f'{seq}__s{seed}__{arm}.npz';digest=old.model_digest(model) if model is not None else None
            meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,model_sha256=digest)
            value,reused=old.cache(path,meta,lambda:previous.trace(native,model,None,files,truth[0],reader,dataset,progress))
            if reused:old.reused_work(progress,len(files)-1)
            old.audit_trace(value,truth[0],len(files)-1)
            if arm=='native':baseline=value
            if model is not None and old.model_digest(model)!=digest:raise ValueError('Frozen model changed')
            tables['seed_video_summary'].append(old.describe_trace(value,baseline,truth,seq,arm,seed,fold));old.record(root,path,artifacts)
            for layer in range(7,13):tables['routing'].append(dict(sequence=seq,fold=fold,seed=seed,arm=arm,layer=layer,frames=int((value['layer']==layer).sum())))
        native.assert_frozen()
    tables['policy_summary'],tables['paired']=common.summarize(tables['seed_video_summary'])
    hashes={}
    for name,rows in tables.items():
        path=root/f'stage207_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete overall work budget')
    publish_identical(root/'stage207_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_PAIRED_PROVIDER_STATE_TRAINING',plan_fingerprint=plan['fingerprint'],
        source_stage206=SOURCE206,source_groups=len(native_bank),training_runs=len(models),training_updates=len(tables['training_history']),
        provider_fit_trajectories=len(tables['support']),full_evaluation_trajectories=len(tables['seed_video_summary']),confirmation_groups_opened=0,
        primary_contrast=next(v for v in tables['paired'] if v['primary']),table_sha256=hashes,artifact_sha256=artifacts,
        next_step='Require own-pair over native-pair with complete-video and recovery protection; common support is not recovery training; no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)


def verify(root):
    root=Path(root).resolve();guard(root)
    p=old.read_sealed(root/'manifest.json')
    support_only=not (root/'stage207_decision.json').exists()
    d=old.read_sealed(root/('stage207_support_decision.json' if support_only else 'stage207_decision.json'))
    if d['schema']!=SCHEMA or d['plan_fingerprint']!=p['fingerprint']:raise ValueError('Changed result identity')
    if support_only:
        if d['status']!='INSUFFICIENT_PAIRED_SUPPORT' or not d['empty_groups'] or d['training_started']:raise ValueError('Incomplete study')
    elif (d['status']!='REVIEW_PAIRED_PROVIDER_STATE_TRAINING' or d['source_stage206']!=SOURCE206 or d['source_groups']!=38
          or d['training_runs']!=12 or d['training_updates']!=3072 or d['provider_fit_trajectories']!=76 or d['full_evaluation_trajectories']!=342
          or d['confirmation_groups_opened']!=0 or set(d['table_sha256'])!={f'stage207_{k}.csv' for k in TABLES}):raise ValueError('Incomplete Stage-207')
    for name,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+name)
    for name,sha in p['code_sha256'].items():
        if file_hash(REPO/name)!=sha:raise ValueError('Changed frozen dependency '+name)
    if file_hash(PROTOCOL_FILE)!=p['protocol_sha256']:raise ValueError('Changed protocol')
    print('Stage-207 verified; '+d['status'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (194,206,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage206-bundle',type=Path,default=DEFAULT_BUNDLE);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:verify(root);return
    if args.checkpoint is None or any(getattr(args,f'stage{k}_root') is None for k in (194,206,76,89,93,121,170,188,191)):
        parser.error('Stage-194 features, Stage-206 providers, original checkpoint and dataset source roots required')
    import stage192_development_grid_rescue as prior
    paths={k:getattr(args,f'stage{k}_root').expanduser().resolve() for k in prior.SOURCES}
    r194,r206=args.stage194_root.expanduser().resolve(),args.stage206_root.expanduser().resolve()
    common.protocol.safe_output(root,[*paths.values(),r194,r206,args.checkpoint,args.stage206_bundle.parent])
    with writer_lock(root):
        guard(root)
        if (root/'stage207_decision.json').exists():verify(root);return
        if (root/'stage207_support_decision.json').exists() and old.read_sealed(root/'stage207_support_decision.json')['status']=='INSUFFICIENT_PAIRED_SUPPORT':verify(root);return
        d206,p206,t206=source206(args.stage206_bundle)
        # Exact runtime-source checks, including old tracking/lib files. No CRLF normalization here.
        for name,sha in p206['code_sha256'].items():
            if file_hash(REPO/name)!=sha:raise ValueError('Frozen source code changed '+name)
        if file_hash(previous.PROTOCOL_FILE)!=p206['protocol_sha256']:raise ValueError('Source protocol changed')
        if old.read_sealed(r206/'manifest.json')!=p206 or old.read_sealed(r206/'stage206_decision.json')!=d206:raise ValueError('Physical Stage-206 run differs from reviewed bundle')
        p194,d194,reader,_=previous.inputs.load_source(r194)
        if p194['folds']!=p206['folds'] or p194['entries']!=p206['entries'] or d194['fingerprint']!=p206['source_stage194']:
            raise ValueError('Feature roster differs from provider')
        entries,dataset,_=prior.load_sources(paths);common.protocol.safe_output(root,[dataset])
        for spec in p194['entries']:
            files,truth,_,_,_=entries[spec['sequence']]
            if fingerprint(files)!=spec['frames_sha256'] or fingerprint(np.asarray(truth).tolist())!=spec['truth_sha256']:raise ValueError('Dataset changed')
        import torch
        torch.set_num_threads(1)
        runtime=prior.old.base.s103.s76.device_runtime(dict(runtime_identity=p206['runtime']),torch,args.gpu)
        differences=prior.old.base.s109.check_runtime(p206['runtime'],runtime)
        code=dict(p206['code_sha256'])
        for file in (__file__,common.__file__):code[Path(file).resolve().relative_to(REPO).as_posix()]=file_hash(file)
        plan=old.sealed(dict(schema=SCHEMA,source_stage194=d194['fingerprint'],source_stage206=SOURCE206,source206_bundle_sha256=file_hash(args.stage206_bundle),
            entries=p194['entries'],folds=p194['folds'],runtime=runtime,runtime_differences=differences,same_source_runtime=not differences,
            design=common.DESIGN,arms=list(common.ARMS),provider_role='Frozen matching-fold Stage-206 model; only own outer fit groups generate training states',
            support='Both actual integer crops fully contain truth and exclude constant padding; no GT used in provider actions',
            training='Three fresh matched-spatial students; native_all, native_pair and own_pair; pair schedules identical; 256 updates each',
            evidence='Historical development OOF; new same-card native and frozen-provider controls; common-support intervention, not recovery training',
            code_sha256=code,protocol_sha256=file_hash(PROTOCOL_FILE)))
        publish_identical(root/'manifest.json',plan)
        args.parameter=runtime['tracker']['parameter'];args.mode='stage207_state_training'
        native,_=prior.old.base.s103.s76.startup.checked_runtime(args,dict(fingerprint=plan['fingerprint'],runtime_identity=runtime,runtime=runtime['tracker']),root)
        try:
            rt=native.rt
            if rt.lens_z!=64 or rt.lens_x!=256 or rt.identity['embed_dim']!=192:raise ValueError('Need native 64+256 by 192 layout')
            bank={}
            for spec in p194['entries']:
                seq=spec['sequence'];data=reader.arrays(f'features/{seq}.npz',dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix'))
                np.testing.assert_array_equal(data['frame_id'],spec['sample_frames'])
                if data['tokens'].shape!=(len(data['frame_id']),320,192) or not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Bad feature cache')
                bank[spec['group']]=data
            providers={};provider_rows=[];verified={}
            for fold in plan['folds']:
                for seed in common.DESIGN['seeds']:
                    model,scope=load_provider(torch,rt,r206,d206,p206,t206,fold,seed,verified)
                    providers[fold['fold'],seed]=model;provider_rows.append(scope)
            publish_identical(root/'source_receipt.json',old.sealed(dict(schema=SCHEMA,plan_fingerprint=plan['fingerprint'],features=reader.verified,providers=verified)))
            experiment(root,plan,native,entries,dataset,bank,providers,provider_rows)
        finally:
            for handle in native.handles:handle.remove()


if __name__=='__main__':main()
