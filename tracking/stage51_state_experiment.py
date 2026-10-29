"""Prepare fit-only own states, matched ordinary head training, and native OOF evaluation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path,scene_geometry
from stage46_shift_audit import publish_table
from stage48_stability_experiment import audit_trace,load_torch
import stage51_state_common as c


def prepare(source,root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate new Stage-51 output directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use an empty Stage-51 directory")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-51: 12 fit-only donor rollouts, 576 uniformly timed extra views, 8 new models, 48 OOF trajectories")


def exposure_rows(plan,entry,base,own):
    rows=[]
    for i,frame in enumerate(own["frame_id"]):
        gt=np.asarray(entry["ground_truth"][int(frame)],float)
        hw=[entry["frames"][int(frame)]["height"],entry["frames"][int(frame)]["width"]]
        for label,parent,crop in (("native",base["parents"][i,0],base["crop"][i,0]),
                ("translated",base["parents"][i,1],base["crop"][i,1]),("own",own["parents"][i],own["crop"][i])):
            geom=scene_geometry(parent,gt,hw,plan["search_factor"],plan["search_size"])
            rows.append(dict(sequence=entry["sequence"],fold=c.fit_fold(plan,entry["sequence"]),frame=int(frame),view=label,
                coverage=float(geom["coverage"]),center_in_crop=bool(geom["center_in_usable"]),
                projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
                parent_center_error_gt_units=float(np.linalg.norm(parent[:2]+parent[2:]/2-gt[:2]-gt[2:]/2)/np.sqrt(gt[2:].prod())),
                parent_width_ratio=float(parent[2]/gt[2]),parent_height_ratio=float(parent[3]/gt[3]),
                crop_differs_native=not np.array_equal(crop,base["crop"][i,0]),
                crop_differs_translated=not np.array_equal(crop,base["crop"][i,1])))
    return rows


def collect(native,plan,source,root):
    from stage48_stability_runtime import run_trace
    from stage51_state_runtime import collect_features,installed_head,load_donor
    bindings={}; rows=[]; calls=0
    with writer_lock(root/"collection"):
        for fold in plan["folds"]:
            state=load_donor(native,plan,fold["fold"])
            for entry in plan["entries"]:
                if entry["sequence"] not in fold["fit"]: continue
                tp=c.path_for(root,entry,"donor_traces"); tm=c.meta(plan,entry,"fit_only_donor_trajectory")
                trace=c.read_cache(tp,tm)
                if trace is None:
                    with installed_head(native,state):
                        trace=run_trace(native,entry["frames"],entry["initial_box"],plan["root"])
                    audit_trace(trace,entry); c.save_cache(tp,trace,tm)
                audit_trace(trace,entry); calls+=int(trace["head_calls"].sum())
                fp=c.path_for(root,entry,"features"); fm=c.meta(plan,entry,"fit_only_original_features")
                own=c.read_cache(fp,fm)
                if own is None:
                    own=collect_features(native,plan,entry,trace)
                    c.audit_features(own,plan,entry,trace); c.save_cache(fp,own,fm)
                c.audit_features(own,plan,entry,trace)
                base=c.source_pack(Path(plan["source_stage48"]),source,entry)
                rows.extend(exposure_rows(plan,entry,base,own))
                for p in (tp,fp,c.prior.cache_path(Path(plan["source_stage48"]),entry)):
                    for q in (p,p.with_suffix(".receipt.json")): bindings[str(q)]=c.file_hash(q)
                print("[fit-only states audited] fold{} {}".format(fold["fold"],entry["sequence"]),flush=True)
        native.assert_frozen(); c.verify_files(plan["source_file_sha256"])
        support=[]
        for fold in plan["folds"]:
            selected=[r for r in rows if r["fold"]==fold["fold"] and r["view"]=="own"]
            changed=[r for r in selected if r["crop_differs_native"] and r["crop_differs_translated"]]
            support.append(dict(fold=fold["fold"],pairs=len(selected),changed_pairs=len(changed),
                changed_videos=len({r["sequence"] for r in changed}),
                changed_vs_native_pairs=sum(r["crop_differs_native"] for r in selected),
                changed_vs_translated_pairs=sum(r["crop_differs_translated"] for r in selected),
                out_of_crop_centers=sum(not r["center_in_crop"] for r in selected),
                own_mean_coverage=float(np.mean([r["coverage"] for r in selected])),
                own_median_gt_side_px=float(np.median([r["projected_gt_side_px"] for r in selected])),
                native_median_gt_side_px=float(np.median([r["projected_gt_side_px"] for r in rows if r["fold"]==fold["fold"] and r["view"]=="native"])),
                own_parent_center_error_p95=float(np.quantile([r["parent_center_error_gt_units"] for r in selected],.95))))
        passed=all(r["changed_pairs"]>=c.DESIGN["minimum_changed_pairs"] and r["changed_videos"]>=c.DESIGN["minimum_changed_videos"] for r in support)
        table=root/"collection/stage51_state_exposure.csv"; publish_table(table,rows)
        support_table=root/"collection/stage51_state_support.csv"; publish_table(support_table,support)
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FIT_ONLY_STATES",plan_fingerprint=plan["fingerprint"],
            file_sha256=bindings,exposure_sha256=c.file_hash(table),support_sha256=c.file_hash(support_table),
            support=support,nontrivial_exposure_pass=passed,
            donor_trajectories=len(plan["entries"]),donor_head_calls=calls,extra_feature_forwards=sum(r["pairs"] for r in support),
            held_out_examples_used_for_training_states=False,frame_selection="unchanged uniform Stage-48 schedule; no outcome filter"))
        c.publish_identical(root/"collection/stage51_collection.json",report)
    print("Stage-51: COMPLETE_FIT_ONLY_STATES; exposure_support="+str(passed))
    return report


def feature_gate(plan,root):
    r=c.read_sealed(root/"collection/stage51_collection.json")
    if (r.get("status")!="COMPLETE_FIT_ONLY_STATES" or r.get("plan_fingerprint")!=plan["fingerprint"] or
        r.get("donor_trajectories")!=len(plan["entries"]) or
        r.get("donor_head_calls")!=sum(len(e["frames"])-1 for e in plan["entries"]) or
        r.get("extra_feature_forwards")!=sum(len(c.prior.sample_frames(e)) for e in plan["entries"])):
        raise ValueError("Collect fit-only donor states first")
    c.verify_files(r["file_sha256"]); c.verify_files({str(root/"collection/stage51_state_exposure.csv"):r["exposure_sha256"],
        str(root/"collection/stage51_state_support.csv"):r["support_sha256"]})
    if r.get("nontrivial_exposure_pass") is not True: raise ValueError("Insufficient changed fit-only states; inspect exposure table before a new design")
    return r


def smoke_gate(plan,root):
    r=c.read_sealed(root/"smoke/stage51_smoke.json")
    if r.get("status")!="PASS_STATE_SOURCE_SMOKE" or r.get("plan_fingerprint")!=plan["fingerprint"]:
        raise ValueError("Need passing Stage-51 smoke")
    for record in r["records"]:
        p=Path(record["checkpoint"])
        c.verify_files({str(p):record["checkpoint_sha256"],str(p.with_suffix(".receipt.json")):record["receipt_sha256"]})
    return r


def train(native,plan,source,root,seed,smoke=False):
    from stage48_stability_runtime import run_trace
    from stage51_state_runtime import installed_head,train_arm
    feature_gate(plan,root)
    if not smoke: smoke_gate(plan,root)
    records=[]
    directory=root/("smoke" if smoke else "training/seed{}".format(seed))
    with writer_lock(directory):
        for fold in ([0] if smoke else range(len(plan["folds"]))):
            pool,bindings=c.fit_pool(root,plan,source,fold)
            records.extend(train_arm(native,plan,root,pool,bindings,fold,seed,arm,smoke) for arm in c.ARMS)
            current=[r for r in records if r["fold"]==fold]
            if len({r["draws_fingerprint"] for r in current})!=1 or len({r["initial_head_sha256"] for r in current})!=1:
                raise ValueError("Matched frame draws/initializations changed")
        if smoke:
            entry=next(e for e in plan["entries"] if e["sequence"]==plan["folds"][0]["held_out"][0])
            frames=entry["frames"][:c.DESIGN["smoke_frames"]+1]
            data=run_trace(native,frames,entry["initial_box"],plan["root"])
            ref=c.prior.source_native(source,entry)
            np.testing.assert_allclose(data["boxes"],ref["boxes"][:len(data["boxes"])],atol=.001,rtol=0)
            np.testing.assert_array_equal(np.trunc(data["boxes"]),np.trunc(ref["boxes"][:len(data["boxes"])]))
            for r in records:
                state=load_torch(native.rt.torch,Path(r["checkpoint"]))["head"]
                with installed_head(native,state): run_trace(native,frames,entry["initial_box"],plan["root"])
        native.assert_frozen(); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="PASS_STATE_SOURCE_SMOKE" if smoke else "COMPLETE_STATE_SOURCE_TRAINING",
            plan_fingerprint=plan["fingerprint"],seed=seed,records=records,held_out_examples_used_for_fitting=False,
            smoke_rollout_pass=smoke))
        c.publish_identical(directory/("stage51_smoke.json" if smoke else "stage51_training.json"),report)
    print("Stage-51: "+report["status"])
    return report


def model_records(plan,root,torch=None):
    records=[]; bindings={}
    for seed in c.SEEDS:
        path=root/"training"/("seed{}".format(seed))/"stage51_training.json"; report=c.read_sealed(path)
        expected={(f["fold"],a) for f in plan["folds"] for a in c.ARMS}
        if (report.get("status")!="COMPLETE_STATE_SOURCE_TRAINING" or report.get("plan_fingerprint")!=plan["fingerprint"] or
            report.get("seed")!=seed or len(report["records"])!=len(expected) or {(r["fold"],r["arm"]) for r in report["records"]}!=expected):
            raise ValueError("Freeze all eight models before evaluation")
        bindings[str(path)]=c.file_hash(path)
        for r in report["records"]:
            p=Path(r["checkpoint"]); receipt=c.read_sealed(p.with_suffix(".receipt.json")); m=receipt["metadata"]
            if (m["plan_fingerprint"]!=plan["fingerprint"] or m["fold"]!=r["fold"] or m["seed"]!=seed or m["arm"]!=r["arm"] or
                m["fit"]!=plan["folds"][r["fold"]]["fit"] or m["held_out"]!=plan["folds"][r["fold"]]["held_out"] or
                m["mode"]!="full" or m["updates"]!=c.DESIGN["updates"] or receipt["step"]!=c.DESIGN["updates"] or
                m["draws_fingerprint"]!=r["draws_fingerprint"] or m["initial_head_sha256"]!=r["initial_head_sha256"]):
                raise ValueError("OOF checkpoint source/budget mismatch")
            expected_features=set()
            for entry in plan["entries"]:
                if entry["sequence"] not in m["fit"]: continue
                for q in (c.path_for(root,entry,"donor_traces"),c.path_for(root,entry,"features"),
                        c.prior.cache_path(Path(plan["source_stage48"]),entry)):
                    expected_features.update((str(q),str(q.with_suffix(".receipt.json"))))
            if (set(m["feature_sha256"])!=expected_features or r["seed"]!=seed or r.get("buffers_unchanged") is not True or
                r["counts"]["supervised"]!=c.DESIGN["updates"]*c.DESIGN["batch_pairs"]*2 or
                r["history"][-1]["step"]!=c.DESIGN["updates"]):
                raise ValueError("Training state scope/count mismatch")
            c.verify_files(m["feature_sha256"])
            bindings[str(p)]=r["checkpoint_sha256"]; bindings[str(p.with_suffix(".receipt.json"))]=r["receipt_sha256"]
            c.verify_files(bindings)
            payload=load_torch(torch,p) if torch is not None else None
            if payload is not None and (payload["metadata"]!=m or payload["step"]!=receipt["step"]): raise ValueError("Checkpoint metadata changed")
            records.append((r,None if payload is None else payload["head"]))
    for fold in plan["folds"]:
        for seed in c.SEEDS:
            pair=[r for r,_ in records if r["fold"]==fold["fold"] and r["seed"]==seed]
            if len({r["draws_fingerprint"] for r in pair})!=1 or len({r["initial_head_sha256"] for r in pair})!=1:
                raise ValueError("Unmatched training arms")
    return records,bindings


def evaluate(native,plan,root):
    from stage48_stability_runtime import digest,run_trace
    from stage51_state_runtime import installed_head
    smoke_gate(plan,root); features=feature_gate(plan,root)
    records,bindings=model_records(plan,root,native.rt.torch)
    run=c.seal(dict(schema=c.SCHEMA,role="native_state_source_oof_evaluation",plan_fingerprint=plan["fingerprint"],
        model_sha256=bindings,collection_fingerprint=features["fingerprint"]))
    with writer_lock(root/"evaluation"):
        c.publish_identical(root/"evaluation/run.json",run)
        for entry in plan["entries"]:
            fold=next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
            for r,state in records:
                if r["fold"]!=fold: continue
                name=r["arm"]+"_seed"+str(r["seed"]); p=root/"evaluation/traces"/(entry["sequence"]+"__"+name+".npz")
                metadata=dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=name)
                data=c.read_cache(p,metadata)
                with installed_head(native,state):
                    if digest(native.rt.net.box_head)!=r["final_head_sha256"]: raise ValueError("Loaded final head changed")
                    if data is None:
                        data=run_trace(native,entry["frames"],entry["initial_box"],plan["root"])
                        audit_trace(data,entry); c.save_cache(p,data,metadata)
                audit_trace(data,entry); print("[state source OOF audited] "+p.name,flush=True)
        native.assert_frozen(); c.verify_files(bindings); c.verify_files(plan["source_file_sha256"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=("prepare","collect","smoke","train","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path); p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); p.add_argument("--parameter",default="deit_distilled")
    p.add_argument("--gpu",type=int,default=0); p.add_argument("--seed",type=int,choices=c.SEEDS,default=c.SEEDS[0])
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: p.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan,source=c.load_plan(root)
    if not args.checkpoint: p.error("Set original SGLA CHECKPOINT or --checkpoint")
    from stage46_motion_experiment import make_runtime
    native,identity=make_runtime(args,dict(runtime=plan["runtime"]))
    if identity!=plan["runtime_identity"]: raise ValueError("Original SGLA runtime changed")
    if args.mode=="collect": collect(native,plan,source,root)
    elif args.mode=="smoke": train(native,plan,source,root,c.SEEDS[0],True)
    elif args.mode=="train": train(native,plan,source,root,args.seed)
    else: evaluate(native,plan,root)


if __name__=="__main__": main()
