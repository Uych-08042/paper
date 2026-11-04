"""Fixed-budget history-source comparison for paired SGLA search gates."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import stage69_gate_common as prior
from stage4_common import file_hash, fingerprint, write_cache

c68=prior.c68;e68=prior.e68;e67=e68.e67;e63=e68.e63;c66=e68.c66
SCHEMA="stage70_fit_history_source_v1"
FILES=("stage70_history_common.py","stage70_history_experiment.py","stage70_history_summary.py",
       "test_stage70_history.py","run_stage70_history.sh")
DESIGN=dict(horizon=32,hann_pairs=16,quality_pairs=8,protect_pairs=8,ridge=.01,tie_epsilon=1e-6,
            smoke_frames=32,bootstrap_seed=2170,bootstrap_samples=20000,practical_gain=.5,reliable_floor=-1.)
POLICIES=c68.POLICIES+("mixed_immediate","mixed_future","mixed_rate_margin")
PROVIDERS=("hann","quality","protect")
seal,VERIFY,publish_identical=prior.seal,prior.VERIFY,prior.publish_identical


def sealed(path):return c68.prior.prior.sealed(path,SCHEMA)


def load_source(root):
    root=Path(root).expanduser().resolve();p=sealed69(root/"manifest.json");source=prior.load_source(p["source_input"])
    report=sealed69(root/"analysis/stage69_diagnostic.json")
    tables={"stage69_"+n+".csv" for n in ("same_state_video","same_state_summary","matched_video","matched_summary",
        "candidate_summary","feature_video","feature_shift","focus","focus_frames","divergences","anchors",
        "fit_prediction","fit_sign_groups","fit_points","paired_case","paired_policy","scope_video","scope")}
    if (p.get("source_bindings")!=source["bindings"] or p.get("design")!=prior.DESIGN or
        p.get("source_report_fingerprint")!=source["report"]["fingerprint"] or
        p.get("source_training_fingerprint")!=source["models"]["fingerprint"] or
        p.get("code_sha256")!={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in prior.FILES} or
        report.get("status")!="COMPLETE_CACHED_PAIRED_GATE_DIAGNOSTIC" or report.get("plan_fingerprint")!=p["fingerprint"] or
        report.get("source_report_fingerprint")!=source["report"]["fingerprint"] or
        report.get("source_plan_fingerprint")!=source["plan"]["fingerprint"] or
        report.get("source_bindings_fingerprint")!=fingerprint(source["bindings"]) or
        report.get("source_training_fingerprint")!=source["models"]["fingerprint"] or
        report.get("source_pair_run_fingerprint")!=source["pairs"]["fingerprint"] or
        report.get("source_eval_run_fingerprint")!=source["run"]["fingerprint"] or report.get("runtime")!=source["plan"]["runtime_identity"] or
        set(report.get("table_sha256",{}))!=tables or report.get("training_performed") is not False or
        any(report.get(k)!=0 for k in ("new_models","new_image_reads","new_search_calls","new_full_trajectories","new_counterfactual_branches")) or
        any(report.get(k) is not True for k in ("exact_recorded_choices","source_controls_unchanged","source_predictors_reconstructed"))):
        raise ValueError("Need complete unmixed Stage-69 diagnostic and Stage-68 source")
    bindings=dict(source["bindings"]);bindings.update(p["code_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    bindings[str(root/"analysis/stage69_fit_reference.json")]=report["fit_reference_sha256"]
    for n in ("manifest.json","analysis/stage69_diagnostic.json"):bindings[str(root/n)]=file_hash(root/n)
    VERIFY(bindings)
    return dict(root=root,s68=source,report=report,bindings=bindings)


def sealed69(path):return c68.prior.prior.sealed(path,prior.SCHEMA)


def fit_pack(source,fold):
    # Fit-local old Q/protect, never the held-out/OOF auxiliary decisions stored
    # on this sequence in earlier evaluation caches.
    return c68.fit_pack(source["s68"]["old"],fold)


def head_for(source,unit):
    return source["s68"]["run"]["observed_heads"][c66.key(dict(unit,policy="hann"))]


def make_plan(root,hardware):
    source=load_source(root);old=source["s68"]["plan"]
    if set(hardware)!={"device","gpu_uuid"} or hardware["device"]!="NVIDIA L40" or not hardware["gpu_uuid"]:raise ValueError("Need observed L40 binding")
    identity=copy.deepcopy(old["runtime_identity"]);identity["gpu_uuid"]=hardware["gpu_uuid"]
    changes=e63.startup.differences(old["runtime_identity"],identity)
    if any(x["field"]!="runtime.gpu_uuid" for x in changes):raise ValueError("Only physical L40 UUID may change; all evaluation controls are rerun")
    total=sum(DESIGN[p+"_pairs"] for p in PROVIDERS)
    if total!=old["design"]["anchors_per_video"] or any(DESIGN[p+"_pairs"]<0 for p in PROVIDERS):raise ValueError("Mixed and Hann-only sample counts must match")
    for k in ("horizon","ridge","tie_epsilon"):
        if DESIGN[k]!=old["design"][k]:raise ValueError("Do not change the paired target or model: "+k)
    if total<=0 or DESIGN["quality_pairs"]<=0 or DESIGN["protect_pairs"]<=0:raise ValueError("Both new histories need fixed pair support")
    for split in old["folds"]:
        for seq in split["fit"]:
            if sum(x["sequence"]==seq and x["fold"]==split["fold"] for x in old["pairs"])!=total:raise ValueError("Need full matched source pair count for every fit video")
    histories=[dict(sequence=seq,origin="native",seed=0,fold=s["fold"],policy=p)
               for s in old["folds"] for seq in s["fit"] for p in ("quality","protect")]
    units=[dict(sequence=e["sequence"],origin="native",seed=0,
        fold=next(f["fold"] for f in old["folds"] if e["sequence"] in f["held_out"]),policy=p) for e in old["entries"] for p in POLICIES]
    frames={e["sequence"]:len(e["frames"])-1 for e in old["entries"]}
    history_calls=sum(4*frames[u["sequence"]] for u in histories)
    new_pairs=sum(len(f["fit"]) for f in old["folds"])*(DESIGN["quality_pairs"]+DESIGN["protect_pairs"])
    h=DESIGN["horizon"];per=4+8*(h-1)
    return seal(dict(schema=SCHEMA,role="consumed_development_fit_history_source_comparison",source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"],source_file_sha256=source["bindings"],
        design=dict(DESIGN),controller_design=copy.deepcopy(old["controller_design"]),actor_design=copy.deepcopy(old["actor_design"]),
        policies=list(POLICIES),history_units=histories,units=units,hardware=hardware,runtime=identity["tracker"],
        runtime_identity=identity,declared_runtime_differences=changes,expected_new_pairs=new_pairs,
        expected_reused_pairs=sum(len(f["fit"]) for f in old["folds"])*DESIGN["hann_pairs"],
        history_search_calls=history_calls,pair_search_calls=new_pairs*per,formal_eval_calls=sum(4*frames[u["sequence"]] for u in units),
        source_bridge_search_calls=4*len(old["folds"]),
        history_smoke_calls_reused=4*sum(min(DESIGN["smoke_frames"],frames[u["sequence"]]) for u in histories[:2]),
        pair_smoke_calls_reused=per,eval_smoke_calls_reused=4*min(DESIGN["smoke_frames"],frames[old["entries"][0]["sequence"]])*len(POLICIES),
        **{k:copy.deepcopy(old[k]) for k in ("entries","folds","root","search_size","search_factor","window")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})),source


def load_plan(root):
    p=sealed(Path(root)/"manifest.json");expected,source=make_plan(p["source_input"],p["hardware"])
    if p!=expected:raise ValueError("Frozen Stage-70 source/code/design changed")
    return p,source


def path_for(root,unit,phase,smoke=False):
    if phase not in ("history","evaluation","bridge"):raise ValueError("Unknown trace phase")
    return Path(root)/("smoke/"+phase if smoke else phase+"/traces")/(c66.key(unit)+".npz")


def metadata(plan,unit,phase,models=None,smoke=False):
    return dict(plan_fingerprint=plan["fingerprint"],unit=unit,phase=phase,smoke=smoke,
                training_fingerprint=None if models is None else models["fingerprint"])


def pair_path(root,case,action=None):return Path(root)/"collection/traces"/(case["case_id"]+"__"+(action or "shared")+".npz")
def pair_meta(plan,pp,case,action=None):return dict(plan_fingerprint=plan["fingerprint"],pair_plan_fingerprint=pp["fingerprint"],case=case,action=action)


def save(path,data,meta):
    path=Path(path)
    if path.with_suffix(".receipt.json").exists():raise ValueError("Never overwrite committed Stage-70 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item()))!=meta:raise ValueError("Unknown orphan cache")
    write_cache(path,data,meta)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))


def read_cache(path,meta):
    path=Path(path)
    if not path.with_suffix(".receipt.json").exists():return None
    receipt=sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata")!=meta or not path.is_file() or file_hash(path)!=receipt["sha256"]:raise ValueError("Stage-70 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item()))!=meta:raise ValueError("Mixed Stage-70 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}


def canonical(policy):return policy.removeprefix("mixed_")
def model_pack(models,unit):return models["mixed" if unit["policy"].startswith("mixed_") else "baseline"][str(unit["fold"])]


def eligible_times(trace,pack,plan):
    choices=[c66.decide("quality",b,f,pack,plan["controller_design"])["selected"] for b,f in zip(trace["features"],trace["choice_features"])]
    times=[i for i,q in enumerate(choices) if q and i+plan["design"]["horizon"]<=len(trace["frame_id"]) and
           not np.array_equal(trace["candidate_boxes"][i,q],trace["candidate_boxes"][i,0])]
    return times,choices


def fixed_indices(n,count):
    if count==0:return []
    if n<count:raise ValueError("Insufficient fixed provider quota: {} available, {} required; no replacement".format(n,count))
    return np.linspace(0,n-1,count,dtype=int).tolist()


def check_fit(plan,unit):
    split=next(f for f in plan["folds"] if f["fold"]==unit["fold"])
    if unit["sequence"] not in split["fit"] or unit["sequence"] in split["held_out"]:raise ValueError("Fit history crosses held-out source group")


def resolve_pairs(plan,source,histories,root):
    cases=[];support=[];old=source["s68"]
    for split in plan["folds"]:
        fold=split["fold"];pack=fit_pack(source,fold)
        for seq in split["fit"]:
            for provider in PROVIDERS:
                count=plan["design"][provider+"_pairs"]
                if provider=="hann":
                    available=[x for x in old["plan"]["pairs"] if x["fold"]==fold and x["sequence"]==seq]
                    picked=[available[i] for i in fixed_indices(len(available),count)]
                    for case in picked:cases.append(dict(case_id="mixed__"+case["case_id"],sequence=seq,fold=fold,provider=provider,
                        frame=case["frame"],allow_choice=case["allow_choice"],unit=case["unit"],source_case=case,reused=True,
                        source_trace_sha256=case["source_trace_sha256"],head_sha256=case["head_sha256"]))
                    available_count=len(available)
                else:
                    unit=next(u for u in plan["history_units"] if u["sequence"]==seq and u["fold"]==fold and u["policy"]==provider)
                    check_fit(plan,unit);trace=histories[c66.key(unit)];times,choices=eligible_times(trace,pack,plan)
                    available_count=len(times)
                    for j in fixed_indices(available_count,count):
                        i=times[j];cases.append(dict(case_id="mixed__f{}__{}__{}__t{:06d}".format(fold,seq,provider,i+1),
                            sequence=seq,fold=fold,provider=provider,frame=i+1,allow_choice=choices[i],
                            unit=dict(unit,policy="quality"),history_unit=unit,reused=False,
                            source_trace_sha256=file_hash(path_for(root,unit,"history")),head_sha256=head_for(source,unit)))
                support.append(dict(fold=fold,sequence=seq,provider=provider,available=available_count,selected_pairs=count,reused=provider=="hann"))
    if len({x["case_id"] for x in cases})!=len(cases):raise ValueError("Duplicate mixed fit pair")
    return cases,support
