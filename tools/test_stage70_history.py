"""Fit-local sampling, fixed budgets and real CENTER histories/pairs/full controls."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage70_history_common as c
import stage70_history_experiment as e
import stage70_history_summary as s
from test_stage68_advantage import old_pack


class Semantics(unittest.TestCase):
    def test_provider_quotas_and_uniform_fit_local_Q_ignore_GT_and_stored_decisions(self):
        splits=[dict(fold=0,fit=["a"],held_out=["b"]),dict(fold=1,fit=["b"],held_out=["a"])]
        units=[dict(sequence=split["fit"][0],origin="native",seed=0,fold=split["fold"],policy=p) for split in splits for p in ("quality","protect")]
        oldcases=[dict(case_id="{}-{}".format(split["fold"],i),fold=split["fold"],sequence=split["fit"][0],frame=i+1,
            allow_choice=3,unit=dict(sequence=split["fit"][0],origin="native",seed=0,fold=split["fold"],policy="quality"),source_trace_sha256="frozen",head_sha256="native") for split in splits for i in range(32)]
        trace=dict(frame_id=np.arange(1,65),features=np.zeros((64,4,24)),choice_features=np.zeros((64,4,64)),
                   candidate_boxes=np.tile([0.,0.,4.,4.],(64,4,1)),quality_choice=np.full(64,3),ground_truth=object())
        trace["choice_features"][:,1,0]=.01;trace["choice_features"][:,2,1]=.02
        trace["candidate_boxes"][:,1,0]=1;trace["candidate_boxes"][:,2,0]=2
        plan=dict(folds=splits,history_units=units,design=dict(c.DESIGN),controller_design=e.e68.c66.DESIGN)
        source=dict(s68=dict(plan=dict(pairs=oldcases),run=dict(observed_heads={c.c66.key(dict(u,policy="hann")):"native" for u in units})))
        histories={c.c66.key(u):trace for u in units}
        with patch.object(c,"fit_pack",side_effect=lambda _,fold:old_pack(fold)),patch.object(c,"file_hash",return_value="frozen"):
            cases,support=c.resolve_pairs(plan,source,histories,"unused")
        self.assertEqual(len(cases),64);self.assertEqual(sum(x["reused"] for x in cases),32)
        self.assertEqual([x["selected_pairs"] for x in support],[16,8,8]*2)
        for row in cases:
            if not row["reused"]:self.assertEqual(row["allow_choice"],row["fold"]+1)
        picked=[x["frame"] for x in cases if x["fold"]==0 and x["provider"]=="quality"]
        self.assertEqual(picked,(np.linspace(0,32,8,dtype=int)+1).tolist())
        with self.assertRaisesRegex(ValueError,"no replacement"):c.fixed_indices(7,8)
        with self.assertRaisesRegex(ValueError,"held-out"):c.check_fit(plan,dict(fold=0,sequence="b"))

    def test_exact_production_budget_and_unchanged_target(self):
        names=["video"+str(i) for i in range(12)];counts=[1127]*11+[1128]
        splits=[dict(fold=0,fit=names[:6],held_out=names[6:]),dict(fold=1,fit=names[6:],held_out=names[:6])]
        old=dict(entries=[dict(sequence=n,frames=list(range(k+1))) for n,k in zip(names,counts)],folds=splits,
            root="unused",search_size=256,search_factor=4.,window=[1.],actor_design={},controller_design={},design=dict(c.c68.DESIGN),
            runtime_identity=dict(tracker=dict(device="NVIDIA L40"),gpu_uuid="old"),
            pairs=[dict(sequence=n,fold=f["fold"]) for f in splits for n in f["fit"] for _ in range(32)])
        source=dict(s68=dict(plan=old),root=Path("unused"),report=dict(fingerprint="old"),bindings={})
        with patch.object(c,"load_source",return_value=source):plan,_=c.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="new"))
        self.assertEqual(len(plan["history_units"]),24);self.assertEqual(len(plan["units"]),120)
        self.assertEqual([plan[k] for k in ("history_search_calls","pair_search_calls","formal_eval_calls")],[108200,48384,541000])
        self.assertEqual(sum(plan[k] for k in ("history_search_calls","pair_search_calls","formal_eval_calls")),697584)
        self.assertEqual(plan["source_bridge_search_calls"],8)
        self.assertEqual([plan[k] for k in ("history_smoke_calls_reused","pair_smoke_calls_reused","eval_smoke_calls_reused")],[256,252,1280])
        changed=copy.deepcopy(source);changed["s68"]["plan"]["design"]["ridge"]=.02
        with patch.object(c,"load_source",return_value=changed),self.assertRaisesRegex(ValueError,"target or model"):c.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="new"))

    def test_model_route_and_receipt_immutability(self):
        models=dict(baseline={"0":"old"},mixed={"0":"new"})
        self.assertEqual(c.model_pack(models,dict(fold=0,policy="future")),"old")
        self.assertEqual(c.model_pack(models,dict(fold=0,policy="mixed_future")),"new")
        self.assertEqual(c.canonical("mixed_rate_margin"),"rate_margin")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"cache.npz";c.save(path,dict(x=np.arange(3)),dict(fold=0,provider="quality"))
            np.testing.assert_array_equal(c.read_cache(path,dict(fold=0,provider="quality"))["x"],np.arange(3))
            with self.assertRaises(ValueError):c.read_cache(path,dict(fold=0,provider="protect"))
            with self.assertRaises(ValueError):c.save(path,dict(x=np.arange(3)),dict(fold=0,provider="quality"))
            path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):c.read_cache(path,dict(fold=0,provider="quality"))


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not validation")
class Workflow(unittest.TestCase):
    def test_real_fit_histories_pairs_controls_GT_isolation_resume_and_frozen_sources(self):
        from test_stage67_summary_repair import Workflow as Previous
        import stage67_summary_repair as repair
        import stage68_advantage_summary as s68
        import stage69_gate_audit as a69
        from stage48_stability_runtime import digest
        saved_summary=repair.summarize;saved_execute=e.e67.execute;saved_pack=c.c68.fit_pack
        captured={};exposed={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3]
            return saved_execute(*args,**kwargs)
        def exposed_pack(source,fold):
            # Test-only fit-local causal coefficients expose both eligible starts
            # of the four-frame real CENTER fixture. No GT is used. Production
            # coefficients and held-out source controls stay byte-for-byte intact.
            if fold in exposed:return exposed[fold]
            pack=copy.deepcopy(saved_pack(source,fold));split=pack["fit"];vectors=[]
            for seq in split:
                trace,_,_=c.c68.prior.read_unit(source["s66"],seq,"hann")
                for i in range(3):
                    f=trace["choice_features"][i]
                    vectors.extend((f[q]-f[0])/np.asarray(pack["quality"]["scale"]) for q in (1,2,3)
                                   if not np.array_equal(trace["candidate_boxes"][i,q],trace["candidate_boxes"][i,0]))
            candidates=vectors+[a+b for a in vectors for b in vectors]+[a-b for a in vectors for b in vectors]
            for v in candidates:
                pack["quality"]["weights"]=v.tolist();ok=True
                for seq in split:
                    trace,_,_=c.c68.prior.read_unit(source["s66"],seq,"hann")
                    choices=[c.c66.decide("quality",b,f,pack,source["s66"]["plan"]["design"])["selected"] for b,f in zip(trace["features"],trace["choice_features"])]
                    ok &= sum(bool(q and i+2<=len(choices) and not np.array_equal(trace["candidate_boxes"][i,q],trace["candidate_boxes"][i,0])) for i,q in enumerate(choices))>=2
                if ok:exposed[fold]=copy.deepcopy(pack);return exposed[fold]
            self.fail("Micro CENTER fixture needs two causal eligible fit starts")
        def extend(root67):
            report=saved_summary(root67)
            if captured.get("done"):return report
            captured["done"]=True;parent=Path(root67).parent;root68=parent/"stage68";root69=parent/"stage69";root70=parent/"stage70"
            observer=captured["observer"];real_capture=type(observer).capture.__get__(observer,type(observer))
            state=digest(observer.rt.net)
            e.e68.prepare(root67,root68,dict(device="NVIDIA L40",gpu_uuid="stage70-cpu-fixture"))
            p68,old=c.c68.load_plan(root68);self.assertEqual(len(p68["pairs"]),4)
            with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture):
                e.e68.collect(root68,p68,old,observer,p68["runtime_identity"],True);e.e68.collect(root68,p68,old,observer,p68["runtime_identity"])
                e.e68.fit(p68,root68,old)
                e.e68.evaluate(root68,p68,old,observer,p68["runtime_identity"],True);e.e68.evaluate(root68,p68,old,observer,p68["runtime_identity"])
            s68.summarize(root68)
            seq=p68["entries"][0]["sequence"]
            with patch.object(c.prior,"CONFIG",(dict(case_id="micro",sequence=seq,anchor_policy="future",selector="fixed",frame=2),)), \
                 patch.dict(c.prior.DESIGN,before=1,after=1):
                a69.summarize(root68,root69)
                before={str(p):p.read_bytes() for p in parent.rglob("*") if p.is_file()}
                e.prepare(root69,root70,dict(device="NVIDIA L40",gpu_uuid="stage70-cpu-fixture"))
                plan,source=c.load_plan(root70)
                self.assertEqual([plan[k] for k in ("history_search_calls","pair_search_calls","formal_eval_calls")],[64,48,320])
                with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture) as calls:
                    e.source_bridge(root70,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,8)
                    calls.reset_mock()
                    e.histories(root70,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,16)
                    calls.reset_mock();e.histories(root70,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,48)
                    pp=e.resolve(plan,root70,source);self.assertEqual({x["provider"] for x in pp["cases"]},{"quality","protect"})
                    calls.reset_mock();e.collect(root70,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,12)
                    calls.reset_mock();e.collect(root70,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,36)
                    original=source["s68"]["plan"]["pairs"][0]
                    reused=dict(case_id="reuse-probe",sequence=original["sequence"],fold=original["fold"],provider="hann",
                        frame=original["frame"],allow_choice=original["allow_choice"],unit=original["unit"],source_case=original,reused=True)
                    shared,tails=e.read_pair(root70,plan,source,pp,reused)
                    trace,entry,pack=e.context(plan,root70,source,reused)
                    e.e68.audit_pair(shared,tails,trace,entry,reused,plan,pack)
                    label=e.e68.label_pair(shared,tails,entry,reused,plan)
                    reference=next(x for x in source["s68"]["models"]["support"] if x["case_id"]==original["case_id"])
                    for target in c.c68.TARGETS:self.assertEqual(label[target],reference[target])
                    with patch.object(e.e68,"label_pair",wraps=e.e68.label_pair) as labels:models=e.fit(plan,root70,source)
                    for call in labels.call_args_list:
                        case=call.args[3];split=next(f for f in plan["folds"] if f["fold"]==case["fold"])
                        self.assertIn(case["sequence"],split["fit"]);self.assertNotIn(case["sequence"],split["held_out"])
                    self.assertEqual(models["baseline"],source["s68"]["models"]["models"])
                    changed=copy.deepcopy(plan);split=plan["folds"][0]
                    for entry in changed["entries"]:
                        if entry["sequence"] in split["held_out"]:entry["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    self.assertEqual(models["mixed"][str(split["fold"])],e.build_models(changed,root70,source)["mixed"][str(split["fold"])])
                    calls.reset_mock();e.evaluate(root70,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,80)
                    calls.reset_mock();e.evaluate(root70,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,240)
                    unit=plan["history_units"][0];entry=copy.deepcopy(e.entry_for(plan,unit));entry["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    replay=e.e66.run_actor(observer,entry["frames"],entry["initial_box"],e.history_plan(plan),unit,c.fit_pack(source,unit["fold"]),4)
                    committed=e.read_history(plan,root70,source,unit)
                    for k in e.e66.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                    unit=next(u for u in plan["units"] if u["policy"]=="mixed_future");entry=copy.deepcopy(e.entry_for(plan,unit));entry["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    replay=e.e68.run_actor(observer,entry,dict(unit,policy="future"),plan,c.model_pack(models,unit),4)
                    committed=c.read_cache(c.path_for(root70,unit,"evaluation"),c.metadata(plan,unit,"evaluation",models))
                    for k in e.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                    case=pp["cases"][0];shared,_=e.read_pair(root70,plan,source,pp,case);entry=copy.deepcopy(e.entry_for(plan,case));entry["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    replay=e.e67.collect_tail(observer,entry,shared,case,"quality","allow",plan,c.fit_pack(source,case["fold"]),e.checked_image)
                    for k in e.e67.FIELDS:np.testing.assert_array_equal(replay[k],e.read_pair(root70,plan,source,pp,case)[1]["allow"][k])
                with patch.object(observer,"capture",side_effect=AssertionError("No inference on resume/summary")),patch.dict("sys.modules",torch=None,cv2=None):
                    e.source_bridge(root70,plan,source,observer,plan["runtime_identity"])
                    e.histories(root70,plan,source,observer,plan["runtime_identity"],True);e.histories(root70,plan,source,observer,plan["runtime_identity"])
                    self.assertEqual(pp,e.resolve(plan,root70,source))
                    e.collect(root70,plan,source,observer,plan["runtime_identity"],True);e.collect(root70,plan,source,observer,plan["runtime_identity"])
                    e.evaluate(root70,plan,source,observer,plan["runtime_identity"],True);e.evaluate(root70,plan,source,observer,plan["runtime_identity"])
                    result=s.summarize(root70);self.assertEqual(result,s.summarize(root70))
                self.assertEqual(result["total_new_search_calls"],440);self.assertFalse(result["reinforcement_learning_performed"])
                self.assertFalse(result["sgla_training_performed"]);self.assertEqual(result["new_predictors"],4)
                self.assertEqual(result["full_evaluation_trajectories"],20);self.assertEqual(result["additional_smoke_calls"],8)
                self.assertTrue(all(Path(p).read_bytes()==data for p,data in before.items()))
                path=c.pair_path(root70,pp["cases"][0],"allow");payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaises(ValueError):e.pairs_completed(plan,root70,source,pp)
                path.write_bytes(payload)
                path=root70/"models/training.json";payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaises(ValueError):e.load_models(plan,root70,source)
                path.write_bytes(payload)
                path=root69/"analysis/stage69_scope.csv";payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root69)
                path.write_bytes(payload)
                path=root70/"analysis/stage70_scope.csv";payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"):s.summarize(root70)
                path.write_bytes(payload)
            self.assertEqual(digest(observer.rt.net),state);return report
        with patch.dict(c.c68.DESIGN,horizon=2,anchors_per_video=2,smoke_frames=4,bootstrap_samples=30), \
             patch.dict(c.DESIGN,horizon=2,hann_pairs=0,quality_pairs=1,protect_pairs=1,smoke_frames=2,bootstrap_samples=30), \
             patch.object(c.c68,"fit_pack",side_effect=exposed_pack),patch.object(e.e67,"execute",side_effect=capture), \
             patch.object(repair,"summarize",side_effect=extend):
            Previous().test_real_frozen_branch_chain_repaired_without_source_changes_or_inference()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
