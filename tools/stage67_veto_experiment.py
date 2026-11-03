"""Force one allow/veto decision, then follow frozen Q or protect for 32 frames."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage67_veto_common as c
from stage27_uav_common import checked_image, crop_geometry, nonempty_path
from stage27_uav_collect import writer_lock
from stage28_response_common import dense_boxes

e63 = c.prior.c64.e63
FIELDS = c.e66.FIELDS+("raw", "size", "offset")


def prepare(source, root, hardware):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents: raise ValueError("Use a separate Stage-67 directory")
    plan, _ = c.make_plan(source, hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()): raise ValueError("Use empty Stage-67 directory")
        c.publish_identical(root/"manifest.json", plan)
    print("Stage-67 prepared: {} / {} supported anchors; {} short branches / {} searches; no fitting".format(
        len(plan["cases"]), len(plan["requests"]), plan["counterfactual_branches"], plan["formal_search_calls"]))
    for r in plan["requests"]: print("[anchor] {} {} {} {}".format(r["case_id"],r["sequence"],r["status"],r["resolved_frame"]),flush=True)


def actor_plan(plan):
    return {k:plan[k] for k in ("root","search_size","search_factor","window")} | dict(design=plan["actor_design"])


def stack(rows): return {k:np.asarray([r[k] for r in rows]) for k in FIELDS}


def observe(observer, image, history, frame, controller, plan, pack):
    """Actor receives predicted history/current image; no labels or future image."""
    row = e63.probe(observer,image,history,frame,"history",actor_plan(plan))
    f = c.prior.c64.choice_features(row["features"],row["candidate_boxes"],history)
    d = c.prior.decide(controller,row["features"],f,pack,plan["controller_design"]); q = d["selected"]
    row.update(d,choice_features=f,boxes=row["candidate_boxes"][q].copy(),parents=row["candidate_parents"][q].copy())
    observer.rt.tracker.state = row["boxes"].tolist()
    return row


def collect_shared(observer, entry, history, case, plan, pack, reader):
    observer.initialize(reader(plan["root"],entry["frames"][0]),entry["initial_box"])
    row = observe(observer,reader(plan["root"],entry["frames"][case["frame"]]),history,case["frame"],"protect",plan,pack)
    return dict(stack([row]),initial_history=np.asarray(history).copy())


def forced_choice(case, action):
    if action not in c.ACTIONS: raise ValueError("Unknown forced anchor action")
    return 0 if action == "veto" else case["allow_choice"]


def successor_history(shared, case, action, plan):
    h = [v.copy() for v in shared["initial_history"]]
    h.append(shared["candidate_boxes"][0,forced_choice(case,action)].copy())
    return h[-plan["actor_design"]["history"]:]


def collect_tail(observer, entry, shared, case, controller, action, plan, pack, reader):
    observer.initialize(reader(plan["root"],entry["frames"][0]),entry["initial_box"])
    h = successor_history(shared,case,action,plan); observer.rt.tracker.state = h[-1].tolist(); rows = []
    for frame in range(case["frame"]+1,case["frame"]+plan["design"]["horizon"]):
        row = observe(observer,reader(plan["root"],entry["frames"][frame]),h,frame,controller,plan,pack)
        rows.append(row); h.append(row["boxes"].copy()); h = h[-plan["actor_design"]["history"]:]
    return stack(rows)


def audit_segment(data, incoming, first, count, controller, plan, pack):
    np.testing.assert_array_equal(data["frame_id"],np.arange(first,first+count))
    if data["candidate_boxes"].shape != (count,4,4) or data["features"].shape != (count,4,24): raise ValueError("Incomplete four-search segment")
    e63.source_common.prior.audit_calls(data,(count,4))
    for name in ("boxes","parents","candidate_parents","candidate_boxes","features","choice_features","raw","size","offset","decode_error"):
        if not np.isfinite(data[name]).all(): raise ValueError("Nonfinite segment: "+name)
    if np.any(data["candidate_boxes"][:,:,2:] <= 0): raise ValueError("Invalid output size")
    history = [np.asarray(v,float).copy() for v in incoming]; ap = actor_plan(plan)
    for i in range(count):
        parents = e63.source_common.prior.history_parents(history,plan["actor_design"])
        np.testing.assert_array_equal(data["candidate_parents"][i],parents)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1]); errors = []
        for q,p in enumerate(parents):
            crop,usable,factor = crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])
            np.testing.assert_array_equal(data["crop"][i,q],crop)
            row = dict(parent=p,boxes=data["candidate_boxes"][i,q],usable=usable,crop=crop,image_hw=data["image_hw"][i],
                       **{k:data[k][i,q] for k in ("raw","size","offset")})
            np.testing.assert_array_equal(data["features"][i,q],e63.actor.feature_vector(row,history,ap))
            cell = int(np.argmax(row["raw"]*np.asarray(plan["window"],np.float32)))
            decoded = dense_boxes(row["size"],row["offset"],p,factor,plan["search_size"],data["image_hw"][i])[cell]
            errors.append(float(np.max(np.abs(decoded-row["boxes"]))))
        np.testing.assert_allclose(data["decode_error"][i],max(errors),rtol=0,atol=1e-12)
        if max(errors) > plan["actor_design"]["box_atol"]: raise ValueError("Public decode failed")
        f = c.prior.c64.choice_features(data["features"][i],data["candidate_boxes"][i],history)
        np.testing.assert_array_equal(data["choice_features"][i],f)
        d = c.prior.decide(controller,data["features"][i],f,pack,plan["controller_design"])
        for k,v in d.items(): np.testing.assert_array_equal(data[k][i],v)
        np.testing.assert_array_equal(data["boxes"][i],data["candidate_boxes"][i,d["selected"]])
        np.testing.assert_array_equal(data["parents"][i],parents[d["selected"]])
        history.append(data["boxes"][i].copy()); history = history[-plan["actor_design"]["history"]:]


def exact_source(data, trace, first):
    for k in c.e66.FIELDS:
        if k == "decode_error": continue  # independently decoded above, using the fixed source tolerance
        np.testing.assert_array_equal(data[k],trace[k][first-1:first-1+len(data["frame_id"])],err_msg="Source diagonal differs: "+k)


def audit_shared(shared, trace, entry, case, plan, pack):
    incoming = e63.history_at(trace,entry["initial_box"],case["frame"],plan)
    np.testing.assert_array_equal(shared["initial_history"],incoming)
    audit_segment(shared,incoming,case["frame"],1,"protect",plan,pack)
    exact_source(shared,trace,case["frame"])
    if not shared["veto"][0] or shared["quality_choice"][0] != case["allow_choice"] or case["allow_choice"] == 0:
        raise ValueError("Anchor is not a frozen changed-box veto")
    if np.array_equal(shared["candidate_boxes"][0,0],shared["candidate_boxes"][0,case["allow_choice"]]): raise ValueError("No anchor intervention")


def audit_tail(data, shared, trace, case, controller, action, plan, pack):
    audit_segment(data,successor_history(shared,case,action,plan),case["frame"]+1,plan["design"]["horizon"]-1,controller,plan,pack)
    if (controller,action) == ("protect","veto"): exact_source(data,trace,case["frame"]+1)


def run_record(plan): return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"]))


def smoke_gate(plan, root):
    r = c.sealed(Path(root)/"smoke/stage67_smoke.json")
    if r.get("status") != "PASS_MATCHED_VETO_SMOKE" or r.get("run_fingerprint") != run_record(plan)["fingerprint"] or r.get("search_calls") != plan["smoke_calls_reused"]:
        raise ValueError("Need passing exact-source smoke")
    c.VERIFY(r["cache_sha256"]); return r


def execute(root, plan, source, observer, identity, smoke=False, image_reader=checked_image):
    if identity != plan["runtime_identity"]: raise ValueError("Need bound L40 runtime")
    if not smoke: smoke_gate(plan,root)
    c.publish_identical(Path(root)/"run.json",run_record(plan)); hashes = {}; heads = {}; calls = branches = 0
    for case in plan["cases"][:1] if smoke else plan["cases"]:
        trace,entry,_ = c.read_unit(source,case["sequence"],"protect")
        pack = source["models"]["models"][str(case["unit"]["fold"])]
        with e63.source_experiment.installed(observer,[],case["unit"]) as head:
            if head != case["head_sha256"]: raise ValueError("Original native head differs")
            heads[case["case_id"]] = head
            path = c.path_for(root,case); meta = c.metadata(plan,case); shared = c.read_cache(path,meta)
            if shared is None:
                shared = collect_shared(observer,entry,e63.history_at(trace,entry["initial_box"],case["frame"],plan),case,plan,pack,image_reader)
                audit_shared(shared,trace,entry,case,plan,pack); c.save(path,shared,meta)
            audit_shared(shared,trace,entry,case,plan,pack); e63.source_experiment.bind(hashes,path); calls += int(shared["head_calls"].sum())
            combinations = [("protect","veto")] if smoke else [(p,a) for p in c.CONTROLLERS for a in c.ACTIONS]
            for controller,action in combinations:
                path = c.path_for(root,case,controller,action); meta = c.metadata(plan,case,controller,action); data = c.read_cache(path,meta)
                if data is None:
                    data = collect_tail(observer,entry,shared,case,controller,action,plan,pack,image_reader)
                    audit_tail(data,shared,trace,case,controller,action,plan,pack); c.save(path,data,meta)
                audit_tail(data,shared,trace,case,controller,action,plan,pack); e63.source_experiment.bind(hashes,path)
                calls += int(data["head_calls"].sum()); branches += 1
        print("[veto branches audited] {} {} f{}".format(case["case_id"],case["sequence"],case["frame"]),flush=True)
    observer.assert_frozen(); c.VERIFY(plan["source_file_sha256"]); c.VERIFY(plan["code_sha256"])
    if calls != (plan["smoke_calls_reused"] if smoke else plan["formal_search_calls"]) or branches != (1 if smoke else plan["counterfactual_branches"]):
        raise ValueError("Shared-anchor/tail budget differs")
    report = c.seal(dict(schema=c.SCHEMA,status="PASS_MATCHED_VETO_SMOKE" if smoke else "COMPLETE_MATCHED_VETO_BRANCHES",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run_record(plan)["fingerprint"],search_calls=calls,block_calls=7*calls,
        branches=branches,observed_heads=heads,cache_sha256=hashes,maximum_source_box_error=0.,training_performed=False))
    c.publish_identical(Path(root)/("smoke/stage67_smoke.json" if smoke else "collection/run.json"),report); return report


def completed(plan, root):
    smoke_gate(plan,root)
    if c.sealed(Path(root)/"run.json") != run_record(plan): raise ValueError("Runtime record changed")
    r = c.sealed(Path(root)/"collection/run.json")
    paths = [c.path_for(root,x) for x in plan["cases"]]+[c.path_for(root,x,p,a) for x in plan["cases"] for p in c.CONTROLLERS for a in c.ACTIONS]
    expected = {str(p.with_suffix(s).resolve()) for p in paths for s in (".npz",".receipt.json")}
    if (r.get("status") != "COMPLETE_MATCHED_VETO_BRANCHES" or r.get("plan_fingerprint") != plan["fingerprint"] or
        r.get("run_fingerprint") != run_record(plan)["fingerprint"] or r.get("search_calls") != plan["formal_search_calls"] or
        r.get("block_calls") != 7*plan["formal_search_calls"] or r.get("branches") != plan["counterfactual_branches"] or
        set(r.get("cache_sha256",{})) != expected or r.get("observed_heads") != {x["case_id"]:x["head_sha256"] for x in plan["cases"]} or
        r.get("maximum_source_box_error") != 0 or r.get("training_performed") is not False): raise ValueError("Need complete unmixed branches")
    c.VERIFY(r["cache_sha256"]); return r


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--mode",required=True,choices=("prepare","smoke","collect"))
    p.add_argument("--input",required=True,type=nonempty_path); p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); p.add_argument("--gpu",type=int,default=0); p.add_argument("--parameter",default="deit_distilled")
    args = p.parse_args()
    if args.mode == "prepare":
        if not args.output_dir: p.error("prepare requires --output_dir")
        import torch
        source = c.load_source(args.input)
        hardware = c.prior.c64.prior.prior.c.prior.device_binding(source["plan"],args.gpu,torch)
        prepare(args.input,args.output_dir,hardware); return
    if not args.checkpoint: p.error("Need original SGLA V1 checkpoint")
    root = Path(args.input).resolve(); plan,source = c.load_plan(root)
    with writer_lock(root):
        native,identity = e63.startup.checked_runtime(args,plan,root); observer = e63.Capture(native)
        try: execute(root,plan,source,observer,identity,args.mode == "smoke")
        finally: observer.handle.remove()


if __name__ == "__main__": main()
