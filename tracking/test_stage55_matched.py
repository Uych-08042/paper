"""Native shared features, predicted-parent branches and completed-source audits."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage55_matched_common as c
import stage55_matched_collect as collect
import stage55_matched_summary as summary
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_fixed_event_budget_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="stage55-plan-",dir=Path(__file__).parent) as name:
            root=Path(name); gt=[[40.,40.,20.,20.]]*50
            boxes=np.array(gt[1:]); boxes[2:,:2]+=1000
            entry=dict(sequence="fixture",frames=[{}]*50,ground_truth=gt)
            inputs=dict(root=root,plan=dict(fingerprint="source54"),report=dict(fingerprint="decision54"),bindings={},
                arrays={"fixture":(entry,{h:dict(boxes=boxes.copy()) for h in c.HEADS})})
            config=[dict(case_id="T01",sequence="fixture",frame=3,purpose="test",anchor_method="native",anchor_kind="onset_frame")]
            with patch.object(c,"load_source",return_value=inputs),patch.object(c,"CONFIG",config):
                plan,_=c.make_plan(root)
                self.assertEqual((plan["expected_points"],plan["expected_branches"]),(15,25))
                self.assertEqual((plan["expected_full_forwards"],plan["expected_head_only_forwards"]),(815,60))
                self.assertEqual(len(c.branches(plan,True)),5)
                config[0]["frame"]=4
                with self.assertRaisesRegex(ValueError,"anchor"): c.make_plan(root)
            path=root/"cache.npz"; meta=dict(identity="test")
            c.save(path,dict(values=np.arange(4)),meta)
            np.testing.assert_array_equal(c.read_cache(path,meta)["values"],np.arange(4))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,dict(identity="different"))
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,meta)

@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed native-branch test")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_center_shared_response_own_successor_and_actor_isolation(self):
        from test_stage50_crossed import TorchTests as Fixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage54_rollout_runtime import configure,frozen_digest,temporary_head
        from stage27_uav_common import checked_image
        with tempfile.TemporaryDirectory(prefix="stage55-actor-",dir=Path(__file__).parent) as name,\
                patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            old,_,native,observer,_=Fixture().fixture(Path(name)); source=old["plan"]; entry=old["focus"]["fixture"][0]
            original=digest(native.rt.net); records=[]
            for seed in c.prior.SEEDS:
                for a,arm in enumerate(c.prior.ARMS):
                    head=make_head(native.rt); _,names=configure(head,arm); frozen=frozen_digest(head,names)
                    with torch.no_grad(): head.conv5_size.bias.add_(.005*(1+a)+(seed-c.prior.SEEDS[0])*.001)
                    records.append((dict(fold=0,seed=seed,arm=arm,trainable_names=names,initial_frozen_sha256=frozen,
                        final_frozen_sha256=frozen,final_head_sha256=digest(head)),copy.deepcopy(head.state_dict())))
            bank=collect.HeadBank(native,records,0); images={f:checked_image(source["root"],r) for f,r in enumerate(entry["frames"])}
            own={}; states={"native":None,**{r["arm"]+"_seed"+str(r["seed"]):s for r,s in records}}
            for h,state in states.items():
                context=contextlib.nullcontext() if state is None else temporary_head(native,state)
                with context: trace=collect.collect_branch(observer,images,entry["initial_box"],entry["initial_box"],range(1,5))
                own[h]=dict(trace,previous=trace["parents"])
            inputs=dict(plan=source,arrays={"fixture":(entry,own)})
            plan=dict(design=dict(c.DESIGN,horizon=2)); parent=own["native"]["previous"][1]
            point=dict(case_id="T01",sequence="fixture",frame=2,offset=0,history="native",point_id="test")
            observer.initialize(images[0],entry["initial_box"])
            data=collect.collect_point(observer,bank,images[2],parent,2); c.audit_point(data,point,plan,inputs)
            self.assertEqual(float(np.max(np.abs(data["raw"]-data["raw"][:1]))),0.)
            self.assertGreater(float(np.max(np.abs(data["boxes"][0]-data["boxes"][1]))),0.)
            item=dict(case_id="T01",sequence="fixture",frame=2,history="native",evaluator=c.HEADS[2],branch_id="test")
            with temporary_head(native,states[item["evaluator"]]):
                branch=collect.collect_branch(observer,images,entry["initial_box"],parent,range(2,4))
            c.audit_branch(branch,item,data,plan,inputs)
            np.testing.assert_array_equal(branch["parents"][1],branch["boxes"][0])
            changed=copy.deepcopy(entry["ground_truth"]); entry["ground_truth"][1:]=[[9999.,9999.,20.,20.]]*4
            with temporary_head(native,states[item["evaluator"]]):
                replay=collect.collect_branch(observer,images,entry["initial_box"],parent,range(2,4))
            np.testing.assert_array_equal(replay["boxes"],branch["boxes"]); entry["ground_truth"]=changed
            broken={k:v.copy() for k,v in branch.items()}; broken["parents"][1,0]+=1
            with self.assertRaises((ValueError,AssertionError)): c.audit_branch(broken,item,data,plan,inputs)
            self.assertEqual(digest(native.rt.net),original); native.assert_frozen(); bank.assert_frozen()
            observer.handle.remove(); observer.pre_handle.remove()

    def test_completed_source54_prepare_smoke_collect_resume_accounting_and_tamper(self):
        import stage51_state_common as old
        import stage52_branch_common as b52
        import stage52_branch_experiment as e52
        import stage52_branch_summary as s52
        import stage53_geometry_common as b53
        import stage53_geometry_experiment as e53
        import stage53_geometry_summary as s53
        import stage54_rollout_common as b54
        import stage54_rollout_experiment as e54
        import stage54_rollout_summary as s54
        from test_stage52_branch import TorchTests as Fixture,actual_small_head
        from stage48_stability_runtime import digest
        config=[dict(case_id="T0"+str(i+1),sequence="v"+str(i),frame=2,purpose="test",anchor_method="native",anchor_kind="fixed_frame") for i in range(2)]
        with tempfile.TemporaryDirectory(prefix="stage55-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4),\
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b54.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(c.DESIGN,horizon=2),patch.object(c,"CONFIG",config),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=Fixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    previous_root=root51
                    for number,common,experiment,summarizer in ((52,b52,e52,s52),(53,b53,e53,s53),(54,b54,e54,s54)):
                        output=folder/("stage"+str(number)); experiment.prepare(previous_root,output); p,inputs=common.load_plan(output)
                        experiment.train(native,p,inputs,output,c.prior.SEEDS[0],True)
                        for seed in c.prior.SEEDS: experiment.train(native,p,inputs,output,seed)
                        experiment.evaluate(native,p,output); summarizer.summarize(output); previous_root=output
                    source=c.load_source(previous_root); self.assertEqual(len(source["records"]),8)
                    output=folder/"stage55"; collect.prepare(previous_root,output); plan,inputs=c.load_plan(output)
                    records,_=e54.model_records(inputs["plan"],inputs["root"],torch)
                    banks={f["fold"]:collect.HeadBank(native,records,f["fold"]) for f in inputs["plan"]["folds"]}
                    observer=collect.NativeCapture(native); original=digest(native.rt.net)
                    collect.execute(output,plan,inputs,observer,banks,inputs["plan"]["runtime_identity"],True)
                    cached={str(p):p.read_bytes() for d in ("responses","branches") for p in (output/d).glob("*")}
                    collect.execute(output,plan,inputs,observer,banks,inputs["plan"]["runtime_identity"])
                    self.assertTrue(all(Path(p).read_bytes()==v for p,v in cached.items()))
                    with patch.object(collect,"collect_point",side_effect=AssertionError("Reuse shared points")),\
                            patch.object(collect,"collect_branch",side_effect=AssertionError("Reuse branches")):
                        collect.execute(output,plan,inputs,observer,banks,inputs["plan"]["runtime_identity"])
                    report=summary.summarize(output); self.assertEqual(report,summary.summarize(output))
                    self.assertEqual(report["shared_inputs"],30); self.assertEqual(report["branches"],50)
                    self.assertEqual((report["full_network_calls"],report["head_only_calls"],report["block_calls"]),(130,120,910))
                    self.assertEqual(report["table_rows"]["focus"],4); self.assertEqual(report["table_rows"]["branch_contrasts"],20)
                    self.assertLess(report["maximum_accounting_residual"],1e-9); self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage55_focus.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    first=c.path_for(output,c.branches(plan)[0]); saved=first.read_bytes(); first.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): summary.summarize(output)
                    first.write_bytes(saved)
                    source_table=previous_root/"analysis/stage54_seed_video_summary.csv"; saved=source_table.read_bytes(); source_table.write_bytes(saved+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous_root)
                    source_table.write_bytes(saved)
                    model=Path(source["records"][0]["checkpoint"]); model.write_bytes(model.read_bytes()+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous_root)
                    observer.handle.remove(); observer.pre_handle.remove()

if __name__=="__main__": unittest.main()
