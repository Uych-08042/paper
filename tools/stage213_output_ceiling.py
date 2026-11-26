"""CPU-only complete-cache output complementarity ceiling at fixed nn states."""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage212_state_experiment as previous
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash,write_csv,fingerprint
old=previous.old
SCHEMA='stage213_fixed_nn_output_complementarity_ceiling_v1'
SOURCE212='6913f1ed1755f3b6eb9a7718e8cd217d4cb03d707d9a527981cf0e6d74205f02'
BUNDLE212='45bf89388460228c66785214615e10d3a1f14b3c805c8b5199d8061f090c21f0'
DEFAULT_BUNDLE=REPO/'docs/experiments/06_sgla_uav_tracking/results/stage212_results_20261126T153146Z_2174614.txt'
PROTOCOL=REPO/'docs/experiments/06_sgla_uav_tracking/STAGE213_FIXED_NN_OUTPUT_CEILING.md'
ARMS=('raw','adapted','mm','oracle')
METRICS=previous.c.METRICS
TABLES=('policy_summary','paired','seed_video_summary','complementarity','recovery_events','source_replay','source_influence')


def source212(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE212:raise ValueError('Need exact reviewed Stage-212 bundle')
    ms=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    parts={m[1].decode():raw[m.end():ms[i+1].start() if i+1<len(ms) else len(raw)] for i,m in enumerate(ms)}
    d,p=json.loads(parts['stage212_decision.json']),json.loads(parts['manifest.json'])
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if (len(parts)!=len(ms) or set(parts)!={'manifest.json','stage212_decision.json',*d['table_sha256']}
            or d['fingerprint']!=SOURCE212 or d['plan_fingerprint']!=p['fingerprint'] or d['full_evaluation_trajectories']!=418
            or d['source_groups']!=38 or d['new_training_updates']!=0 or d['confirmation_groups_opened']!=0):raise ValueError('Unexpected source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=parts[name];data=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if data is None:raise ValueError('Changed source table '+name)
        tables[name.removeprefix('stage212_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(data.decode())))
    if set(tables)!=set(previous.c.TABLES):raise ValueError('Incomplete source tables')
    return d,p,tables


def read_traces(source,p,d,t,spec,gt,verified):
    seq=spec['sequence'];fold=next(f['fold'] for f in p['folds'] if spec['group'] in f['held']);result={}
    for arm,seed in [('native',0)]+[(a,s) for s in previous.c.SEEDS for a in ('matched_single','mm','nn')]:
        sha=None if arm=='native' else next(r['state_sha256'] for r in t['model_scope'] if int(r['fold'])==fold and int(r['seed'])==seed)
        meta=dict(plan_fingerprint=p['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,phase='full',model_sha256=sha)
        value=previous.runtime.common.bound_arrays(source,d,f'evaluation/{seq}__s{seed}__{arm}.npz',meta,verified)
        previous.c.audit(value,gt[0],len(gt)-1,arm);result[arm,seed]=value
    previous.check_controls(result,seq,'source_cache')
    return result


def ceiling(nn,mm,native,truth):
    raw=nn['reference_boxes'];adapted=nn['boxes']
    np.testing.assert_array_equal(raw,native['boxes']);np.testing.assert_array_equal(nn['parent'],native['parent'])
    ri=old.tracking_iou(raw,truth[1:]);ai=old.tracking_iou(adapted,truth[1:]);mi=old.tracking_iou(mm['boxes'],truth[1:])
    if not np.isfinite(np.r_[ri,ai,mi]).all():raise ValueError('Nonfinite evaluation labels')
    choose=ai>ri  # Equal-IoU ties keep raw; no tuned threshold or random choice.
    boxes=np.where(choose[:,None],adapted,raw);oi=old.tracking_iou(boxes,truth[1:])
    np.testing.assert_array_equal(oi,np.maximum(ri,ai))
    np.testing.assert_array_equal(nn['next_parent'],raw)
    return dict(frame_id=nn['frame_id'],raw_iou=ri,adapted_iou=ai,mm_iou=mi,oracle_iou=oi,
        oracle_choose_adapted=choose,oracle_boxes=boxes,parent=nn['parent'],next_parent=nn['next_parent'],reference_boxes=raw)


def describe(boxes,native,mm,gt,seq,arm,seed,fold):
    trace=dict(boxes=boxes,pipeline_ms=np.zeros(len(boxes)))
    result=old.describe_trace(trace,native,gt,seq,arm,seed,fold)
    other=old.describe_trace(trace,mm,gt,seq,arm,seed,fold)
    # This is a derived quality sequence; synthetic zero time is never exported as runtime/FPS.
    result.pop('median_pipeline_ms')
    result.update(missed_matched_recoveries=other['missed_native_recoveries'],matched_good_frame_loss=other['native_good_frame_loss'],
                  introduced_matched_failure=other['introduced_failure'],oracle_only=arm=='oracle')
    return result


def check_exported(quality,seq,seed,rows):
    """Reproduce published per-video quality, excluding cross-seed native references."""
    errors=[]
    for arm,source_arm,source_seed in (('raw','native',0),('adapted','nn',seed),('mm','mm',seed)):
        matches=[r for r in rows if r['sequence']==seq and r['arm']==source_arm and int(r['seed'])==source_seed]
        if len(matches)!=1:raise ValueError('Incomplete published source result')
        metrics=[m for m in METRICS if m!='median_pipeline_ms' and (arm!='raw' or m not in ('missed_matched_recoveries','matched_good_frame_loss'))]
        for m in metrics:
            error=abs(quality[arm][m]-float(matches[0][m]));errors.append(error)
            if error>1e-12:raise ValueError('Source quality replay differs: '+seq+' '+arm+' '+m)
        if quality[arm]['introduced_failure']!=(matches[0]['introduced_failure']=='True'):
            raise ValueError('Source failure status differs')
    return max(errors,default=0.)


def recovery_rows(data,seq,group,seed):
    result=[];qualities={a:data[a+'_iou'] for a in ('raw','adapted','mm','oracle')}
    for ref in ('raw','mm'):
        for j,e in enumerate(old.failure_events(qualities[ref])):
            r=e['recovery'];fields={a+'_three_good':'' for a in ARMS};blocked=''
            if r is not None:
                fields={a+'_three_good':bool((q[r:r+3]>=.5).all()) for a,q in qualities.items()}
                blocked=int(((qualities['raw'][r:r+3]<.5)&(qualities['adapted'][r:r+3]<.5)).sum())
                if ref=='raw' and not fields['oracle_three_good']:raise ValueError('Oracle cannot lose a raw recovery')
            result.append(dict(sequence=seq,group=group,seed=seed,reference=ref,event=j,onset=e['start']+1,
                recovery='' if r is None else r+1,censor=e['censor'],**fields,both_candidates_bad_frames=blocked,
                reference_recovery_unavailable='' if r is None else not fields['oracle_three_good']))
    return result


def complement(data,seq,group,seed):
    r,a,m,o=[data[k+'_iou'] for k in ('raw','adapted','mm','oracle')];n=len(r)
    return dict(sequence=seq,group=group,seed=seed,frames=n,adapted_better_frames=int((a>r).sum()),raw_better_frames=int((r>a).sum()),tie_frames=int((a==r).sum()),
        raw_good_adapted_bad=int(((r>=.5)&(a<.5)).sum()),adapted_good_raw_bad=int(((a>=.5)&(r<.5)).sum()),
        both_below_good=int(((r<.5)&(a<.5)).sum()),both_below_failure=int(((r<.2)&(a<.2)).sum()),
        mm_good_both_below_good=int(((m>=.5)&(o<.5)).sum()),
        mean_oracle_gain_over_raw=float((o-r).mean()),mean_oracle_gain_over_adapted=float((o-a).mean()),
        mean_mm_quality_beyond_oracle=float(np.maximum(m-o,0).mean()),
        interpretation='Oracle label audit only; no deployable action or new inference')


def summarize(rows):
    groups=sorted({r['group'] for r in rows});metrics=tuple(m for m in METRICS if m!='median_pipeline_ms');index={};summary=[];pairs=[];influence=[]
    for arm in ARMS:
        for g in groups:
            rr=[r for r in rows if r['arm']==arm and r['group']==g]
            if sorted(r['seed'] for r in rr)!=previous.c.SEEDS:raise ValueError('Incomplete derived source/seed grid')
            index[arm,g]={m:float(np.mean([r[m] for r in rr])) for m in metrics}
            index[arm,g]['introduced']=any(r['introduced_failure'] for r in rr);index[arm,g]['introduced_matched']=any(r['introduced_matched_failure'] for r in rr)
        summary.append(dict(arm=arm,videos=len(groups),oracle_only=arm=='oracle',introduced_failure_videos=sum(index[arm,g]['introduced'] for g in groups),
            introduced_matched_failure_videos=sum(index[arm,g]['introduced_matched'] for g in groups),**{m:float(np.mean([index[arm,g][m] for g in groups])) for m in metrics}))
    for a,b in (('oracle','raw'),('oracle','adapted'),('oracle','mm'),('adapted','raw'),('mm','raw')):
        for metric in ('auc','low_iou_fraction','missed_native_recoveries','native_good_frame_loss','missed_matched_recoveries','matched_good_frame_loss'):
            ds=[index[a,g][metric]-index[b,g][metric] for g in groups]
            pairs.append(dict(contrast=a+'_minus_'+b,metric=metric,oracle_comparison=a=='oracle',
                favorable_direction='positive' if metric=='auc' else 'negative',**previous.c.PROTOCOL.contrast(ds)))
        ds=np.asarray([index[a,g]['auc']-index[b,g]['auc'] for g in groups])
        for g,v in zip(groups,ds):influence.append(dict(contrast=a+'_minus_'+b,group=g,auc_delta=float(v),leave_one_source_out=float((ds.sum()-v)/(len(ds)-1))))
    return summary,pairs,influence


def check_code(plan):
    for n,sha in plan['code_sha256'].items():
        if file_hash(REPO/n)!=sha:raise ValueError('Changed frozen dependency '+n)
    if file_hash(PROTOCOL)!=plan['protocol_sha256']:raise ValueError('Changed protocol')


def experiment(root,plan,entries,source_data,source_rows,progress=None):
    total=sum(len(entries[e['sequence']][1])-1 for e in plan['entries'])*2
    progress=progress or OverallProgress('Stage-213',total+1,unit='cached frames');progress.report('Auditing paired outputs at unchanged nn search states')
    tables={k:[] for k in TABLES};artifacts={}
    if (root/'source_receipt.json').exists():artifacts['source_receipt.json']=file_hash(root/'source_receipt.json')
    for spec in plan['entries']:
        seq,g=spec['sequence'],spec['group'];gt=entries[seq][1];fold=next(f['fold'] for f in plan['folds'] if g in f['held']);vv=source_data[seq]
        native=vv['native',0]
        for seed in previous.c.SEEDS:
            nn,mm,single=vv['nn',seed],vv['mm',seed],vv['matched_single',seed]
            meta=dict(plan_fingerprint=plan['fingerprint'],sequence=seq,seed=seed,role='gt_ceiling_not_a_policy')
            path=root/'ceiling'/f'{seq}__s{seed}.npz'
            data,reused=old.cache(path,meta,lambda:ceiling(nn,mm,native,gt))
            if reused:old.reused_work(progress,len(gt)-1)
            else:progress.advance(count=len(gt)-1)
            expected=ceiling(nn,mm,native,gt)
            if set(data)!=set(expected):raise ValueError('Changed derived ceiling schema')
            for k in expected:np.testing.assert_array_equal(data[k],expected[k])
            old.record(root,path,artifacts)
            oracle=dict(nn,boxes=data['oracle_boxes']);previous.c.audit(oracle,gt[0],len(gt)-1,'nn')
            boxes=dict(raw=nn['reference_boxes'],adapted=nn['boxes'],mm=mm['boxes'],oracle=data['oracle_boxes'])
            quality={}
            for arm in ARMS:
                row=describe(boxes[arm],native,single,gt,seq,arm,seed,fold);tables['seed_video_summary'].append(row);quality[arm]=row
            replay_error=check_exported(quality,seq,seed,source_rows)
            if quality['oracle']['auc']<max(quality['raw']['auc'],quality['adapted']['auc'])-1e-12:raise ValueError('Broken AUC upper bound')
            if quality['oracle']['missed_native_recoveries']!=0 or quality['oracle']['native_good_frame_loss']!=0:raise ValueError('Broken native capability ceiling')
            tables['complementarity'].append(complement(data,seq,g,seed));tables['recovery_events'].extend(recovery_rows(data,seq,g,seed))
            tables['source_replay'].append(dict(sequence=seq,seed=seed,raw_exact_native=True,mm_exact_single=True,nn_state_unchanged=True,
                oracle_exact_framewise_max=True,maximum_published_metric_error=replay_error,
                raw_auc=quality['raw']['auc'],adapted_auc=quality['adapted']['auc'],mm_auc=quality['mm']['auc']))
    tables['policy_summary'],tables['paired'],tables['source_influence']=summarize(tables['seed_video_summary']);hashes={}
    for name,rows in tables.items():
        if not rows:rows=[dict(status='NO_REFERENCE_EVENTS')]
        path=root/f'stage213_{name}.csv';write_csv(path,rows);hashes[path.name]=file_hash(path)
    if progress.done!=progress.total-1:raise ValueError('Incomplete audit budget')
    publish_identical(root/'stage213_decision.json',old.sealed(dict(schema=SCHEMA,status='REVIEW_FIXED_NN_OUTPUT_CEILING',plan_fingerprint=plan['fingerprint'],
        source_stage212=SOURCE212,source_groups=len(plan['entries']),derived_oracle_sequences=len(plan['entries'])*2,new_training_updates=0,
        new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,table_sha256=hashes,artifact_sha256=artifacts,
        interpretation='GT ceiling and supply audit, not deployable selection; raw seed views are identical-source duplicates; no automatic adoption')))
    progress.phase='Complete';progress.advance();progress.report();verify(root)
    for r in tables['policy_summary']:
        print(f"{r['arm']:8s} derived AUC={r['auc']:.4f} low-IoU={r['low_iou_fraction']:.2%} "
              f"missed raw/mm recoveries={r['missed_native_recoveries']:.4f}/{r['missed_matched_recoveries']:.4f}",flush=True)
    print('GT oracle is an unattainable label ceiling, not a trained method or new online trajectory.',flush=True)


def verify(root):
    root=Path(root).resolve();p=old.read_sealed(root/'manifest.json');d=old.read_sealed(root/'stage213_decision.json');check_code(p)
    if (d['schema']!=SCHEMA or d['status']!='REVIEW_FIXED_NN_OUTPUT_CEILING' or d['plan_fingerprint']!=p['fingerprint'] or d['source_stage212']!=SOURCE212
            or d['source_groups']!=38 or d['derived_oracle_sequences']!=76 or any(d[k]!=0 for k in ('new_training_updates','new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened'))
            or set(d['table_sha256'])!={f'stage213_{n}.csv' for n in TABLES}):raise ValueError('Incomplete Stage-213')
    for n,sha in (d['table_sha256']|d['artifact_sha256']).items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=sha:raise ValueError('Changed artifact '+n)
    print('Stage-213 verified; oracle diagnostic only, no trained/deployed selector',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in (212,76,89,93,121,170,188,191):parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--stage212-bundle',type=Path,default=DEFAULT_BUNDLE);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-results',action='store_true');args=parser.parse_args();root=args.output_dir.resolve()
    if args.check_results:verify(root);return
    import stage192_development_grid_rescue as sources
    if any(getattr(args,f'stage{k}_root') is None for k in (212,*sources.SOURCES)):parser.error('Need complete Stage-212 caches and seven source roots for verified annotation/frame inventory; no checkpoint/GPU required')
    src=args.stage212_root.resolve();paths={k:getattr(args,f'stage{k}_root').resolve() for k in sources.SOURCES}
    previous.c.PROTOCOL.safe_output(root,[src,*paths.values(),args.stage212_bundle.parent])
    with writer_lock(root):
        if (root/'stage213_decision.json').exists():verify(root);return
        d,p,t=source212(args.stage212_bundle);previous.verify(src)
        if old.read_sealed(src/'manifest.json')!=p or old.read_sealed(src/'stage212_decision.json')!=d:raise ValueError('Physical source differs from reviewed result')
        config=dict(source_paths={str(k):str(v) for k,v in paths.items()})
        entries,dataset=previous.runtime.load_entries(config,p);previous.c.PROTOCOL.safe_output(root,[dataset])
        verified={};bank={}
        for spec in p['entries']:bank[spec['sequence']]=read_traces(src,p,d,t,spec,entries[spec['sequence']][1],verified)
        code=dict(p['code_sha256'])
        for f in (__file__,REPO/'tools/run_stage213_output_ceiling.sh'):code[Path(f).resolve().relative_to(REPO).as_posix()]=file_hash(f)
        plan=old.sealed(dict(schema=SCHEMA,source_stage212=SOURCE212,source_plan=p['fingerprint'],entries=p['entries'],folds=p['folds'],source_runtime=p['runtime'],
            audit_runtime=dict(python=sys.version.split()[0],numpy=np.__version__),code_sha256=code,protocol_sha256=file_hash(PROTOCOL),
            rule='Strict adapted IoU greater than raw chooses adapted for offline ceiling; ties raw; nn parent/next state never changed'))
        publish_identical(root/'manifest.json',plan)
        publish_identical(root/'source_receipt.json',old.sealed(dict(plan_fingerprint=plan['fingerprint'],source_arrays=verified,source_bundle_sha256=file_hash(args.stage212_bundle))))
        experiment(root,plan,entries,bank,t['seed_video_summary'])


if __name__=='__main__':main()
