"""CPU cache-only audit: source binding, fixed-state ceiling, recovery and full grid."""
import contextlib,csv,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage213_output_ceiling as run


def fixture(index=0):
    gt=np.tile([0.,0.,10.,10.],(10,1));good=gt[1:].copy();bad=good.copy();bad[:,:2]=100.
    raw=np.vstack((bad[:3],good[3:6],bad[6:]));adapt=np.vstack((good[:3],bad[3:6],good[6:] if index%3==1 else bad[6:]))
    if index%3==2:adapt=bad.copy()
    mm=np.vstack((bad[:6],good[6:]));n=len(raw)
    def trace(boxes,dual=False,reference=None):
        calls=np.zeros((n,12),int);calls[:,:6]=1;calls[:,8]=2 if dual else 1
        parent_boxes=reference if reference is not None else boxes
        value=dict(frame_id=np.arange(1,n+1),boxes=boxes.copy(),parent=np.vstack((gt[0],parent_boxes[:-1])),layer=np.full(n,9,int),
            block_calls=calls,head_calls=np.full(n,2 if dual else 1,int),selector_calls=np.ones(n,int),crop_calls=np.ones(n,int),pipeline_ms=np.ones(n))
        if dual:value.update(reference_boxes=parent_boxes.copy(),next_parent=parent_boxes.copy())
        return value
    vv={('native',0):trace(raw)}
    for seed in run.previous.c.SEEDS:
        vv['nn',seed]=trace(adapt,True,raw);vv['mm',seed]=trace(mm,True);vv['matched_single',seed]=trace(mm)
    return gt,vv


class Tests(unittest.TestCase):
    def test_reviewed_source_and_tamper(self):
        d,p,t=run.source212(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE212);self.assertEqual(len(t),10)
        self.assertEqual(len(t['seed_video_summary']),418)
        self.assertEqual(p['fingerprint'],'dc93467f24459fff6c405ec8dcbd8106e71eb23c35ed4f89c14af0f630415561')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.source212(path)

    def test_pointwise_ceiling_and_unavailable_matched_recovery(self):
        gt,v=fixture();nn=v['nn',2132];native=v['native',0];mm=v['mm',2132]
        data=run.ceiling(nn,mm,native,gt)
        np.testing.assert_array_equal(data['oracle_iou'],np.maximum(data['raw_iou'],data['adapted_iou']))
        np.testing.assert_array_equal(data['oracle_choose_adapted'],[True]*3+[False]*6)
        np.testing.assert_array_equal(data['next_parent'],native['boxes'])
        # Oracle outputs differ, yet each next parent remains original raw, not oracle.
        self.assertFalse(np.array_equal(data['oracle_boxes'],data['next_parent']))
        run.previous.c.audit(dict(nn,boxes=data['oracle_boxes']),gt[0],9,'nn')
        events=run.recovery_rows(data,'uav0000000_00000_s','uav0000000',2132)
        raw=next(e for e in events if e['reference']=='raw' and e['recovery']!='')
        matched=next(e for e in events if e['reference']=='mm')
        self.assertTrue(raw['oracle_three_good']);self.assertEqual(raw['recovery'],4)
        self.assertTrue(matched['reference_recovery_unavailable']);self.assertEqual(matched['both_candidates_bad_frames'],3)
        self.assertTrue(any(e['censor']=='video_end' and e['recovery']=='' for e in events))
        counts=run.complement(data,'s','g',2132)
        self.assertEqual(counts['mm_good_both_below_good'],3)
        self.assertEqual(counts['adapted_better_frames']+counts['raw_better_frames']+counts['tie_frames'],9)
        changed=dict(nn,parent=nn['parent'].copy());changed['parent'][1,0]+=.1
        with self.assertRaises(AssertionError):run.ceiling(changed,mm,native,gt)
        changed=dict(nn,next_parent=nn['boxes'])
        with self.assertRaises(AssertionError):run.ceiling(changed,mm,native,gt)

    def test_complete_bound_cache_pipeline_and_reuse(self):
        groups=[f'uav{i:07d}' for i in range(38)];folds=run.previous.c.PROTOCOL.split_groups(groups)
        specs=[dict(sequence=g+'_00000_s',group=g) for g in groups]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);src=base/'source';root=base/'audit';root.mkdir()
            source_plan=run.old.sealed(dict(entries=specs,folds=folds));artifacts={};bank={};entries={};source_rows=[]
            scope=[dict(fold=f['fold'],seed=s,state_sha256=f"state-{f['fold']}-{s}") for f in folds for s in run.previous.c.SEEDS]
            for i,spec in enumerate(specs):
                seq,g=spec['sequence'],spec['group'];gt,vv=fixture(i);entries[seq]=([],gt)
                fold=next(f['fold'] for f in folds if g in f['held'])
                for (arm,seed),value in vv.items():
                    sha=None if arm=='native' else next(r['state_sha256'] for r in scope if r['fold']==fold and r['seed']==seed)
                    metadata=dict(plan_fingerprint=source_plan['fingerprint'],sequence=seq,fold=fold,seed=seed,arm=arm,phase='full',model_sha256=sha)
                    path=src/'evaluation'/f'{seq}__s{seed}__{arm}.npz'
                    run.old.cache(path,metadata,lambda:value);run.old.record(src,path,artifacts)
                    source_rows.append({k:str(v) for k,v in run.previous.describe(vv,gt,seq,fold,arm,seed).items()})
            decision=dict(artifact_sha256=artifacts);tables=dict(model_scope=scope);verified={}
            for spec in specs:bank[spec['sequence']]=run.read_traces(src,source_plan,decision,tables,spec,entries[spec['sequence']][1],verified)
            self.assertEqual(len(verified),532)
            plan=run.old.sealed(dict(entries=specs,folds=folds,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan)
            run.publish_identical(root/'source_receipt.json',run.old.sealed(dict(source_arrays=verified)))
            # Any attempt at an online model, image or training path is an error.
            with patch.object(run.previous,'dual_trace',side_effect=AssertionError('No inference')),patch.object(run.previous.runtime,'load_students',side_effect=AssertionError('No weights')),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,entries,bank,source_rows)
                committed={p.name:run.file_hash(p) for p in (root/'ceiling').glob('*.npz')}
                run.experiment(root,plan,entries,bank,source_rows)
                run.verify(root)
            self.assertEqual(committed,{p.name:run.file_hash(p) for p in (root/'ceiling').glob('*.npz')})
            self.assertTrue(all(run.file_hash(src/k)==v for k,v in artifacts.items()))
            def rows(name):
                with (root/f'stage213_{name}.csv').open(encoding='utf8',newline='') as h:return list(csv.DictReader(h))
            self.assertEqual(len(rows('seed_video_summary')),304);self.assertEqual(len(rows('paired')),30)
            self.assertEqual(len(rows('source_influence')),190);self.assertEqual(len(rows('complementarity')),76)
            policy={r['arm']:r for r in rows('policy_summary')}
            self.assertEqual(float(policy['oracle']['missed_native_recoveries']),0.)
            self.assertEqual(float(policy['oracle']['native_good_frame_loss']),0.)
            self.assertGreater(float(policy['oracle']['missed_matched_recoveries']),0.)
            self.assertEqual(int(policy['raw']['videos']),38)
            self.assertNotIn('median_pipeline_ms',policy['oracle'])
            self.assertTrue(all(float(r['maximum_published_metric_error'])==0 for r in rows('source_replay')))
            for name in ('new_image_reads','new_gpu_inference_calls','new_full_trajectories','new_training_updates'):
                self.assertEqual(run.old.read_sealed(root/'stage213_decision.json')[name],0)
            # Changed model binding, published metric, and table bytes all fail closed.
            wrong_scope=[dict(r,state_sha256='wrong') for r in scope]
            with self.assertRaises(ValueError):run.read_traces(src,source_plan,decision,dict(model_scope=wrong_scope),specs[0],entries[specs[0]['sequence']][1],{})
            altered=[dict(r) for r in source_rows];altered[0]['auc']=float(altered[0]['auc'])+1
            q={a:run.describe(bank[specs[0]['sequence']]['nn',2132]['reference_boxes'] if a=='raw' else bank[specs[0]['sequence']]['nn',2132]['boxes'],
                bank[specs[0]['sequence']]['native',0],bank[specs[0]['sequence']]['matched_single',2132],entries[specs[0]['sequence']][1],specs[0]['sequence'],a,2132,0) for a in ('raw','adapted','mm')}
            with self.assertRaises(ValueError):run.check_exported(q,specs[0]['sequence'],2132,altered)
            (root/'stage213_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
