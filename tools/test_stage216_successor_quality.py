"""Actual CENTER fixture, source identity and independent-spawn cache parity."""
import contextlib,io,json,multiprocessing as mp,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage216_successor_quality as run

class Progress:
    def __init__(self):self.done=0;self.reused=0
    def advance(self,count=1):self.done+=count

def fixture():
    import test_stage210_recovery_audit as f
    import test_stage215_components as u
    native=f.make_native();folds=[dict(fold=0,fit=[],held=['g','h'])]
    return native,f.models_for(native,folds),u.updater_fixture(native,folds),folds

def bank(native,a,u,images):
    reader=lambda d,p:images[int(p)];files=list(map(str,range(len(images))));gt=np.tile([2.,2.,1.,1.],(len(images),1));result={}
    for origin in run.ORIGINS:
        f,m=('coverage',origin.split('__')[1]) if origin!='mm' else (None,None)
        for seed in run.source.c.SEEDS:
            updater=None if f is None else run.source.c.Masked(u[0,seed,f],m)
            result[seed,origin]=run.source.previous.rollout(native,a[0,seed,'matched'],'mm' if origin=='mm' else f,updater,files,gt[0],reader,None,Progress(),truth=gt)
    return result

def spawned(config,root):
    native,a,u,folds=fixture();images=np.asarray(config['images'],np.float32);b=bank(native,a,u,images)
    plan=config['plan'];spec=plan['entries'][0];entry=(['0','1','2'],np.tile([2.,2.,1.,1.],(3,1)),None,lambda d,p:images[int(p)],'fixture')
    with patch.object(run,'state_trace',side_effect=lambda cfg,plan,spec,seed,origin,verified:b[seed,origin]):
        run.audit_sequence(Path(root),plan,native,a,u,spec,entry,None,{},Progress(),'full')

def workflow_worker(config,sequences,phase,index,channel):
    try:
        native,a,u,folds=fixture();root=Path(config['root']);plan=config['plan'];progress=run.runtime.WorkerProgress(channel,index)
        for seq in sequences:
            images=np.asarray(config['images'][seq],np.float32);b=bank(native,a,u,images);spec=next(e for e in plan['entries'] if e['sequence']==seq)
            entry=(['0','1','2'],np.tile([2.,2.,1.,1.],(3,1)),None,lambda d,p:images[int(p)],'fixture')
            with patch.object(run,'state_trace',side_effect=lambda cfg,plan,spec,seed,origin,verified:b[seed,origin]):run.audit_sequence(root,plan,native,a,u,spec,entry,None,config,progress,'full')
        progress.flush();channel.put(('complete',index,0,0))
    except BaseException:
        import traceback
        channel.put(('error',index,traceback.format_exc(),0));raise

class Tests(unittest.TestCase):
    def test_complete_collection_summary_and_seal(self):
        native,a,u,folds=fixture();specs=[dict(sequence='seq'+g,group=g,tracked_frames=2) for g in ('g','h')]
        images={e['sequence']:np.random.default_rng(216+i).normal(size=(3,20,8)).astype(np.float32) for i,e in enumerate(specs)}
        banks={seq:bank(native,a,u,img) for seq,img in images.items()}
        entries={seq:(['0','1','2'],np.tile([2.,2.,1.,1.],(3,1)),None,lambda d,p,img=img:img[int(p)],'fixture') for seq,img in images.items()}
        rt=dict(tracker=dict(device='NVIDIA L20'),gpu_uuid='fixture')
        plan=run.old.sealed(dict(schema=run.SCHEMA,source_stage215='fixture',runtime=rt,runtime_binding=run.source.bind_runtime(rt,rt),entries=specs,folds=folds,selection=run.selection(dict(entries=specs)),workers=2,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run.publish_identical(root/'manifest.json',plan);config=dict(root=str(root),plan=plan,images={k:v.tolist() for k,v in images.items()})
            actual_workers=run.runtime.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual_workers(config,seqs,phase,n,progress,target=workflow_worker)
            with patch.object(run.source.previous.previous.previous,'specs_for_smoke',return_value=specs),patch.object(run,'state_trace',side_effect=lambda cfg,plan,spec,seed,origin,verified:banks[spec['sequence']][seed,origin]),patch.object(run.runtime,'run_workers',side_effect=workers),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,a,u,entries,None,config)
            decision=run.old.read_sealed(root/'stage216_decision.json');self.assertEqual((decision['states'],decision['search_calls']),(16,144));self.assertEqual(len(decision['primary']),1)
            self.assertEqual(len(decision['table_sha256']),10);run.verify(root)
            (root/'stage216_cells.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):run.verify(root)

    def test_reviewed_source_selection_and_tampering(self):
        d,p=run.bundle(run.DEFAULT_BUNDLE);s=run.selection(p)
        self.assertEqual(len(s),1824)
        for r in s:
            e=next(e for e in p['entries'] if e['sequence']==r['sequence']);self.assertTrue(1<=r['frame']<e['tracked_frames'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wrong.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.bundle(path)

    def test_actual_center_independent_branches_and_gt_labels(self):
        native,a,u,folds=fixture();images=np.random.default_rng(216).normal(size=(4,20,8)).astype(np.float32)
        b=bank(native,a,u,images);seed=2132;trace=b[seed,'coverage__center'];models={f:u[0,seed,f] for f in run.source.c.FAMILIES};progress=Progress()
        hooks=[len(v._forward_hooks) for v in native.rt.backbone.blocks];hashes={f:run.old.model_digest(m) for f,m in models.items()}
        v=run.branches(native,a[0,seed,'matched'],models,images[2],images[3],images[0],[2.,2.,1.,1.],trace['parent'][1],2,progress)
        self.assertEqual(progress.done,9);np.testing.assert_array_equal(v['current_box'],trace['boxes'][1])
        j=run.ACTIONS.index(('coverage','center'));np.testing.assert_array_equal(v['next_boxes'][j],trace['boxes'][2]);np.testing.assert_array_equal(v['candidate_parent'][j],trace['next_parent'][1])
        for key in ('next_boxes','candidate_parent','next_grid'):np.testing.assert_array_equal(v[key][0],v[key][4])
        self.assertEqual(hooks,[len(x._forward_hooks) for x in native.rt.backbone.blocks]);self.assertEqual(hashes,{f:run.old.model_digest(m) for f,m in models.items()})
        data={k:np.asarray([value]) for k,value in v.items()}
        first=run.measures(data,0,0,[2,2,1,1]);other=run.measures(data,0,0,[100,100,1,1]);self.assertNotEqual(first['geometry_error'],other['geometry_error'])
        np.testing.assert_array_equal(v['next_boxes'],data['next_boxes'][0])

    def test_cache_and_independent_spawn_parity(self):
        native,a,u,folds=fixture();images=np.random.default_rng(216).normal(size=(3,20,8)).astype(np.float32);b=bank(native,a,u,images)
        spec=dict(sequence='seq',group='g',tracked_frames=2);plan=run.old.sealed(dict(entries=[spec],folds=folds,selection=[dict(sequence='seq',group='g',frame=1)]))
        entry=(['0','1','2'],np.tile([2.,2.,1.,1.],(3,1)),None,lambda d,p:images[int(p)],'fixture')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);progress=Progress()
            with patch.object(run,'state_trace',side_effect=lambda cfg,plan,spec,seed,origin,verified:b[seed,origin]):
                aa,_=run.audit_sequence(root,plan,native,a,u,spec,entry,None,{},progress,'smoke')
                with patch.object(native,'step',side_effect=AssertionError('Cache must avoid inference')):run.audit_sequence(root,plan,native,a,u,spec,entry,None,{},progress,'smoke')
            self.assertEqual(progress.done,144);self.assertEqual(progress.reused,72)
            proc=mp.get_context('spawn').Process(target=spawned,args=(dict(images=images.tolist(),plan=plan),str(root)))
            proc.start();proc.join(90)
            if proc.is_alive():proc.terminate();proc.join();self.fail('Spawn fixture exceeded budget')
            self.assertEqual(proc.exitcode,0)
            for _,seed,origin,_,value in aa:
                path=root/'full'/f'seq__s{seed}__{origin}.npz'
                with np.load(path,allow_pickle=False) as other:
                    for k in value:np.testing.assert_array_equal(value[k],other[k])

    def test_source_equal_weight_summaries(self):
        rows=[]
        for g in ('a','b'):
            for o in run.ORIGINS:
                for f,m in run.ACTIONS:
                    delta= -.1 if m=='scale' else 0.
                    for seed in (2132,2133):
                        rows.append(dict(group=g,origin=o,family=f,mask=m,seed=seed,output_iou=.6+delta,geometry_error=1+delta,fully_covered=True,gt_side_network_px=60,dense_best_iou=.8,center_error=.1,log_side_error=.2,geometry_gain=-delta,output_gain=delta))
        policy,pairs,align=run.summarize(rows);self.assertEqual((len(policy),len(pairs),len(align)),(32,49,24))
        primary=[r for r in pairs if r['primary']];self.assertEqual(len(primary),1);self.assertAlmostEqual(primary[0]['mean_delta'],-.1)
        r=next(r for r in align if r['origin']=='mm' and r['family']=='coverage' and r['mask']=='scale');self.assertEqual(r['geometry_improved_output_worse_fraction'],1.)

if __name__=='__main__':unittest.main()
