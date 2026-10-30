"""Native shared features, predicted-parent branches and completed-source audits."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage59_parent_common as c
import stage59_parent_collect as collect
import stage59_parent_summary as summary
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_components_diagonal_bytes_event_budget_and_cache(self):
        left=np.array([1.2345678901,2.3,17.9,23.7]); right=np.array([4.2,5.3,31.4,49.8])
        np.testing.assert_array_equal(c.hybrid_parent(left,left,True),left)
        mixed=c.hybrid_parent(left,right)
        np.testing.assert_allclose(mixed[:2]+mixed[2:]/2,left[:2]+left[2:]/2,atol=1e-12)
        np.testing.assert_array_equal(mixed[2:],right[2:])
        with self.assertRaises(ValueError): c.hybrid_parent(left,[0,0,-1,3])
        with tempfile.TemporaryDirectory(prefix="stage59-plan-",dir=Path(__file__).parent) as name:
            root=Path(name); gt=[[40.,40.,20.,20.]]*50; boxes=np.array(gt[1:]); boxes[2:,:2]+=1000
            methods=[c.evaluator(d,s) for s in c.prior.SEEDS for d in c.DONORS]
            entry=dict(sequence="fixture",frames=[{}]*50,ground_truth=gt)
            inputs=dict(root=root,plan=dict(fingerprint="source58"),report=dict(fingerprint="decision58"),bindings={},
                arrays={"fixture":(entry,{h:dict(boxes=boxes.copy(),previous=boxes.copy()) for h in methods})})
            config=[dict(case_id="T01",sequence="fixture",frame=3,purpose="test",anchor_method=methods[0],anchor_kind="onset_frame")]
            with patch.object(c,"load_source",return_value=inputs),patch.object(c,"CONFIG",config):
                plan,_=c.make_plan(root)
                self.assertEqual((plan["expected_points"],plan["expected_branches"]),(8,16))
                self.assertEqual((plan["expected_full_forwards"],plan["expected_head_only_forwards"]),(520,16))
                self.assertEqual(len(c.branches(plan,True)),16)
                config[0]["frame"]=4
                with self.assertRaisesRegex(ValueError,"anchor"): c.make_plan(root)
            path=root/"cache.npz"; meta=dict(identity="test")
            c.save(path,dict(values=np.arange(4)),meta)
            np.testing.assert_array_equal(c.read_cache(path,meta)["values"],np.arange(4))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,dict(identity="different"))
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,meta)

    def test_exact_paths_and_factorial_interactions(self):
        values=np.array([1+2*x+3*y+4*z+5*x*y+6*x*z+7*y*z+8*x*y*z for x in (0,1) for y in (0,1) for z in (0,1)]).reshape(2,2,2)
        effects=summary.factor_effects(values)
        self.assertEqual(effects["own_gap"],35)
        self.assertEqual(effects["head_at_EE"]+effects["center_after_head"]+effects["size_after_center"],35)
        self.assertEqual(effects["head_at_EE"]+effects["size_after_head"]+effects["center_after_size"],35)
        self.assertEqual(effects["center_order_average"],9.5)
        self.assertEqual(effects["size_order_average"],11)
        self.assertEqual(effects["head_order_average"],12.5)
        self.assertEqual(effects["center_size_interaction"],9)
        with self.assertRaises(ValueError): summary.factor_effects(np.zeros((2,2)))
        values=values.astype(float); values[0,0,0]=float("nan")
        with self.assertRaises(ValueError): summary.factor_effects(values)

@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed native-branch test")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_center_shared_response_own_successor_and_actor_isolation(self):
        from test_stage50_crossed import TorchTests as Fixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage58_gradient_runtime import configure,frozen_digest,temporary_head
        from stage27_uav_common import checked_image
        with tempfile.TemporaryDirectory(prefix="stage59-actor-",dir=Path(__file__).parent) as name,\
                patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            old,_,native,observer,_=Fixture().fixture(Path(name)); source=old["plan"]; entry=old["focus"]["fixture"][0]
            original=digest(native.rt.net); records=[]
            for seed in c.prior.SEEDS:
                for a,arm in enumerate(c.ARMS):
                    head=make_head(native.rt); _,names=configure(head,arm); frozen=frozen_digest(head,names)
                    with torch.no_grad(): head.conv5_size.bias.add_(.005*(1+a)+(seed-c.prior.SEEDS[0])*.001); head.conv5_offset.bias.add_(.002*(a+1))
                    records.append((dict(fold=0,seed=seed,arm=arm,trainable_names=names,initial_frozen_sha256=frozen,
                        final_frozen_sha256=frozen,final_head_sha256=digest(head)),copy.deepcopy(head.state_dict())))
            bank=collect.HeadBank(native,records,0); images={f:checked_image(source["root"],r) for f,r in enumerate(entry["frames"])}
            own={}; states={"native":None,**{r["arm"]+"_seed"+str(r["seed"]):s for r,s in records}}
            for h,state in states.items():
                context=contextlib.nullcontext() if state is None else temporary_head(native,state)
                with context: trace=collect.collect_branch(observer,images,entry["initial_box"],entry["initial_box"],range(1,5))
                own[h]=dict(trace,previous=trace["parents"])
            inputs=dict(plan=source,arrays={"fixture":(entry,own)})
            plan=dict(design=dict(c.DESIGN,horizon=2),cases=[dict(case_id="T01",sequence="fixture",frame=2,purpose="test")])
            all_points={}
            for point in c.points(plan):
                observer.initialize(images[0],entry["initial_box"])
                data=collect.collect_point(observer,bank,images[2],c.parent_for(inputs,point),2,point)
                c.audit_point(data,point,plan,inputs); all_points[point["point_id"]]=data
                self.assertEqual(float(np.max(np.abs(data["raw"]-data["raw"][:1]))),0.)
            item=next(b for b in c.branches(plan) if b["center_donor"]=="E" and b["size_donor"]=="P")
            data=all_points[item["point_id"]]; parent=data["parent"]
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

    def test_completed_source58_prepare_smoke_collect_resume_accounting_and_tamper(self):
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
        import stage56_transition_common as b56
        import stage56_transition_experiment as e56
        import stage56_transition_summary as s56
        import stage57_quality_common as b57
        import stage57_quality_experiment as e57
        import stage57_quality_summary as s57
        import stage58_gradient_common as b58
        import stage58_gradient_experiment as e58
        import stage58_gradient_summary as s58
        from stage48_stability_runtime import digest
        config=[dict(case_id="T0"+str(i+1),sequence="v"+str(i),frame=2,purpose="test",anchor_method=c.evaluator("E",c.prior.SEEDS[0]),anchor_kind="fixed_frame") for i in range(2)]
        with tempfile.TemporaryDirectory(prefix="stage59-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4),\
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b54.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(b56.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b57.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(b58.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(c.DESIGN,horizon=2),patch.object(c,"CONFIG",config),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=Fixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    previous_root=root51
                    for number,common,experiment,summarizer in ((52,b52,e52,s52),(53,b53,e53,s53),(54,b54,e54,s54),(56,b56,e56,s56),(57,b57,e57,s57),(58,b58,e58,s58)):
                        output=folder/("stage"+str(number)); experiment.prepare(previous_root,output); p,inputs=common.load_plan(output)
                        experiment.train(native,p,inputs,output,c.prior.SEEDS[0],True)
                        for seed in c.prior.SEEDS: experiment.train(native,p,inputs,output,seed)
                        experiment.evaluate(native,p,output); summarizer.summarize(output); previous_root=output
                    source=c.load_source(previous_root); self.assertEqual(len(source["records"]),12)
                    output=folder/"stage59"; collect.prepare(previous_root,output); plan,inputs=c.load_plan(output)
                    records,_=e58.model_records(inputs["plan"],inputs["root"],torch)
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
                    self.assertEqual(report["shared_inputs"],16); self.assertEqual(report["branches"],32)
                    self.assertEqual((report["full_network_calls"],report["head_only_calls"],report["block_calls"]),(80,32,560))
                    self.assertEqual(report["table_rows"]["focus"],4); self.assertEqual(report["table_rows"]["branch_contrasts"],16)
                    self.assertLess(report["maximum_accounting_residual"],1e-9); self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage59_focus.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    first=c.path_for(output,c.branches(plan)[0]); saved=first.read_bytes(); first.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): summary.summarize(output)
                    first.write_bytes(saved)
                    source_table=previous_root/"analysis/stage58_seed_video_summary.csv"; saved=source_table.read_bytes(); source_table.write_bytes(saved+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous_root)
                    source_table.write_bytes(saved)
                    model=Path(source["records"][0]["checkpoint"]); model.write_bytes(model.read_bytes()+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous_root)
                    observer.handle.remove(); observer.pre_handle.remove()

if __name__=="__main__": unittest.main()
