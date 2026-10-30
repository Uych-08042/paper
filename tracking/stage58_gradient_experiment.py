"""Full endpoint anchors with gradient-direction and norm-matched controls."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path
from stage48_stability_experiment import audit_trace,load_torch
import stage58_gradient_common as c


def prepare(source,root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate new Stage-58 directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use an empty Stage-58 directory")
        c.publish_identical(root/"manifest.json",plan)
        from stage46_shift_audit import publish_table
        (root/"preparation").mkdir(exist_ok=True)
        publish_table(root/"preparation/stage58_anchor_pools.csv",plan["anchor_pools"])
    print("Stage-58: fixed start parents; 12 new heads / 72 OOF traces; live two-frame current-policy fit branches")


def smoke_gate(plan,root):
    report=c.read_sealed(root/"smoke/stage58_smoke.json")
    if report.get("status")!="PASS_SHORT_ROLLOUT_SMOKE" or report.get("plan_fingerprint")!=plan["fingerprint"]:
        raise ValueError("Need passing Stage-58 branch smoke")
    if len(report["records"])!=len(c.ARMS) or {r["arm"] for r in report["records"]}!=set(c.ARMS): raise ValueError("Incomplete branch smoke")
    for r in report["records"]:
        path=Path(r["checkpoint"])
        c.verify_files({str(path):r["checkpoint_sha256"],str(path.with_suffix(".receipt.json")):r["receipt_sha256"]})
    return report


def train(native,plan,inputs,root,seed,smoke=False):
    from stage48_stability_runtime import run_trace
    from stage51_state_runtime import installed_head
    from stage58_gradient_runtime import train_arm
    if not smoke: smoke_gate(plan,root)
    records=[]; directory=root/("smoke" if smoke else "training/seed{}".format(seed))
    with writer_lock(directory):
        for fold in ([0] if smoke else range(len(plan["folds"]))):
            pool,bindings=c.fit_pool(plan,inputs,fold)
            records.extend(train_arm(native,plan,root,pool,bindings,fold,seed,arm,smoke) for arm in c.ARMS)
        if smoke:
            entry=next(e for e in plan["entries"] if e["sequence"]==plan["folds"][0]["held_out"][0])
            frames=entry["frames"][:plan["training"]["smoke_frames"]+1]
            native_trace=run_trace(native,frames,entry["initial_box"],plan["root"])
            ref=c.native_trace(inputs,entry)
            np.testing.assert_allclose(native_trace["boxes"],ref["boxes"][:len(native_trace["boxes"])],atol=.001,rtol=0)
            np.testing.assert_array_equal(np.trunc(native_trace["boxes"]),np.trunc(ref["boxes"][:len(native_trace["boxes"])]))
            for r in records:
                state=load_torch(native.rt.torch,Path(r["checkpoint"]))["head"]
                with installed_head(native,state): run_trace(native,frames,entry["initial_box"],plan["root"])
        native.assert_frozen(); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="PASS_SHORT_ROLLOUT_SMOKE" if smoke else "COMPLETE_SHORT_ROLLOUT_TRAINING",
            plan_fingerprint=plan["fingerprint"],seed=seed,records=records,held_out_examples_used_for_fitting=False,
            native_replay_pass=smoke,learned_native_rollout_pass=smoke))
        c.publish_identical(directory/("stage58_smoke.json" if smoke else "stage58_training.json"),report)
    print("Stage-58: "+report["status"])
    return report


def model_records(plan,root,torch=None):
    result=[]; bindings={}
    for seed in c.SEEDS:
        path=root/"training"/("seed{}".format(seed))/"stage58_training.json"; report=c.read_sealed(path)
        expected={(f["fold"],a) for f in plan["folds"] for a in c.ARMS}
        if (report.get("status")!="COMPLETE_SHORT_ROLLOUT_TRAINING" or report.get("plan_fingerprint")!=plan["fingerprint"] or
            report.get("seed")!=seed or len(report["records"])!=len(expected) or {(r["fold"],r["arm"]) for r in report["records"]}!=expected):
            raise ValueError("Complete all twelve endpoint-gradient models")
        bindings[str(path)]=c.file_hash(path)
        for r in report["records"]:
            p=Path(r["checkpoint"]); receipt=c.read_sealed(p.with_suffix(".receipt.json")); m=receipt["metadata"]
            initial=c.initial_record(plan,r["fold"],seed); pool=plan["anchor_pools"][r["fold"]]
            updates=plan["training"]["updates"]; count=updates*plan["training"]["batch_pairs"]*2*plan["design"]["horizon"]
            draws=np.random.default_rng(seed+100*r["fold"]).integers(0,pool["pairs"],size=(updates,plan["training"]["batch_pairs"]))
            if (m["plan_fingerprint"]!=plan["fingerprint"] or m["fold"]!=r["fold"] or m["seed"]!=seed or r["seed"]!=seed or
                m["arm"]!=r["arm"] or m["mode"]!="full" or m["updates"]!=updates or receipt["step"]!=updates or
                m["fit"]!=plan["folds"][r["fold"]]["fit"] or m["held_out"]!=plan["folds"][r["fold"]]["held_out"] or
                m["anchor_pool"]!=pool or m["initial_head_sha256"]!=initial["initial_head_sha256"] or r["initial_head_sha256"]!=initial["initial_head_sha256"] or
                m["draws_fingerprint"]!=c.fingerprint(draws.tolist()) or r["draws_fingerprint"]!=m["draws_fingerprint"] or
                m["initial_frozen_sha256"]!=r["initial_frozen_sha256"] or r["initial_frozen_sha256"]!=r["final_frozen_sha256"] or
                m["trainable_names"]!=r["trainable_names"] or r["maximum_gradient"]["geometry"]<=0 or r["maximum_gradient"]["response"]!=0 or
                r["maximum_frozen_map_error"]>plan["design"]["frozen_map_atol"] or r["maximum_decode_error"]>plan["design"]["decode_atol"] or
                r["maximum_native_branch_error"]>plan["design"]["decode_atol"] or r["counts"]["supervised"]!=count or
                any(r["totals"][k]!=count for k in ("reference_calls","student_calls","head_only_batches","head_only_examples")) or
                r["totals"]["qualified_endpoints"]<=0 or r["totals"]["qualified_first_transitions"]<=0 or
                r["maximum_endpoint_gradient"]<=0 or
                (r["arm"]=="endpoint_control" and (r.get("reproduces_stage57_control") is not True or
                    r["final_head_sha256"]!=next(v["final_head_sha256"] for v in plan["endpoint_controls"] if (v["fold"],v["seed"])==(r["fold"],seed)))) or
                r["history"][-1]["step"]!=updates):
                raise ValueError("Short-rollout scope/budget/source changed")
            validate_gradients(plan,r,m)
            c.verify_files(m["feature_sha256"])
            bindings[str(p)]=r["checkpoint_sha256"]; bindings[str(p.with_suffix(".receipt.json"))]=r["receipt_sha256"]; c.verify_files(bindings)
            payload=load_torch(torch,p) if torch is not None else None
            if payload is not None and (payload["metadata"]!=m or payload["step"]!=receipt["step"]): raise ValueError("Short rollout payload changed")
            result.append((r,None if payload is None else payload["head"]))
    return result,bindings


def validate_gradients(plan, record, metadata):
    q = record["quality_totals"]; g = record["gradient_totals"]
    steps = plan["training"]["updates"]
    if (g["steps"] != steps or q["eligible_endpoints"] != record["totals"]["qualified_endpoints"] or
        q["retained_endpoints"] != q["eligible_endpoints"] or
        q["quality_better"]+q["quality_tied"]+q["quality_worse"] != q["eligible_endpoints"] or
        not 0 <= g["actual_modified_batches"] <= g["conflict_batches"] <= steps or
        not 0 <= g["nontrivial_projection_batches"] <= g["conflict_batches"] or
        not g["conflict_batches"] <= g["full_gradient_batches"] <= steps or
        g["conflict_all_references_better"]+g["conflict_any_reference_not_better"] != g["conflict_batches"] or
        not 0 <= g["adamw_supervised_opposition_batches"] <= steps or
        record["maximum_gradient_audit"]["norm_match_relative_error"] > plan["design"]["gradient_audit_rtol"] or
        record["maximum_gradient_audit"]["projection_relative_violation"] > plan["design"]["gradient_audit_rtol"] or
        (record["arm"] == "endpoint_control" and g["actual_modified_batches"] != 0)):
        raise ValueError("Gradient projection scope/mass/audit changed")
    values = list(g.values())+list(q.values())+list(record["maximum_gradient_audit"].values())
    if not np.isfinite(values).all(): raise ValueError("Non-finite gradient audit")
    integer_fields=("steps","full_gradient_batches","conflict_batches","nontrivial_projection_batches","actual_modified_batches",
        "conflict_all_references_better","conflict_any_reference_not_better","adamw_supervised_opposition_batches")
    if any(not isinstance(g[k],int) or g[k]<0 for k in integer_fields) or any(not isinstance(v,int) or v<0 for v in q.values()):
        raise ValueError("Invalid gradient/quality counter")
    for key in ("gradient_totals","quality_totals"):
        if any(record["history"][-1][k]!=v for k,v in record[key].items()):
            raise ValueError("Final gradient history differs from record")


def inputs_run(plan):
    return c.prior.read_sealed(Path(plan["source_input"])/"evaluation/run.json")["fingerprint"]


def evaluate(native,plan,root):
    from stage48_stability_runtime import digest,make_head,run_trace
    from stage51_state_runtime import installed_head
    from stage58_gradient_runtime import configure,frozen_digest
    smoke_gate(plan,root); records,bindings=model_records(plan,root,native.rt.torch)
    run=c.seal(dict(schema=c.SCHEMA,role="native_short_rollout_oof",plan_fingerprint=plan["fingerprint"],model_sha256=bindings))
    with writer_lock(root/"evaluation"):
        c.publish_identical(root/"evaluation/run.json",run)
        for entry in plan["entries"]:
            fold=next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
            for r,state in records:
                if r["fold"]!=fold: continue
                head=make_head(native.rt); _,names=configure(head,r["arm"])
                if names!=r["trainable_names"] or frozen_digest(head,names)!=r["initial_frozen_sha256"]:
                    raise ValueError("Frozen comparison branch differs from original checkpoint")
                head.load_state_dict(state,strict=True)
                if digest(head)!=r["final_head_sha256"] or frozen_digest(head,names)!=r["final_frozen_sha256"]:
                    raise ValueError("Loaded branch weights changed")
                method=r["arm"]+"_seed"+str(r["seed"]); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                metadata=dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method)
                data=c.read_cache(path,metadata)
                with installed_head(native,state):
                    if data is None:
                        data=run_trace(native,entry["frames"],entry["initial_box"],plan["root"])
                        audit_trace(data,entry); c.save_cache(path,data,metadata)
                audit_trace(data,entry)
                if r["arm"]=="endpoint_control":
                    old=c.prior.read_cache(Path(plan["source_input"])/"evaluation/traces"/(entry["sequence"]+"__endpoint_control_seed"+str(r["seed"])+".npz"),
                        dict(run_fingerprint=inputs_run(plan),sequence=entry["sequence"],method="endpoint_control_seed"+str(r["seed"])))
                    if old is None: raise ValueError("Missing original endpoint control")
                    for key in ("boxes","peak","layer","previous"):
                        np.testing.assert_array_equal(data[key],old[key],err_msg="Endpoint control trajectory changed")
                print("[branch OOF audited] "+path.name,flush=True)
        native.assert_frozen(); c.verify_files(bindings); c.verify_files(plan["source_file_sha256"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("prepare","smoke","train","evaluate"))
    parser.add_argument("--input",required=True,type=nonempty_path); parser.add_argument("--output_dir",type=nonempty_path)
    parser.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); parser.add_argument("--parameter",default="deit_distilled")
    parser.add_argument("--gpu",type=int,default=0); parser.add_argument("--seed",type=int,choices=c.SEEDS,default=c.SEEDS[0])
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: parser.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan,inputs=c.load_plan(root)
    if not args.checkpoint: parser.error("Set original SGLA CHECKPOINT or --checkpoint")
    from stage46_motion_experiment import make_runtime
    native,identity=make_runtime(args,dict(runtime=plan["runtime"]))
    if identity!=plan["runtime_identity"]: raise ValueError("Original SGLA runtime changed")
    if args.mode=="smoke": train(native,plan,inputs,root,c.SEEDS[0],True)
    elif args.mode=="train": train(native,plan,inputs,root,args.seed)
    else: evaluate(native,plan,root)


if __name__=="__main__": main()
