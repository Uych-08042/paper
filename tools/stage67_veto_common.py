"""Frozen Stage-66 sources and predeclared matched allow/veto diagnostic anchors."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import stage66_takeover_common as prior
import stage66_takeover_experiment as e66
from stage27_uav_common import failure_events, tracking_iou
from stage4_common import file_hash

SCHEMA = "stage67_matched_veto_continuation_v1"
DESIGN = dict(horizon=32, event_lookback=32, good_iou=.5, low_iou=.2)
FILES = ("stage67_veto_common.py", "stage67_veto_experiment.py", "stage67_veto_summary.py",
         "test_stage67_veto.py", "run_stage67_veto.sh")
CONFIG = tuple(dict(case_id="F"+str(i+1).zfill(2), sequence=n, selector="first_changed_veto")
    for i,n in enumerate(("uav0000176_00000_s", "uav0000307_04531_s", "uav0000226_05370_s", "uav0000204_00000_s"))) + (
    dict(case_id="E01", sequence="uav0000176_00000_s", selector="last_veto_before_first_failure"),
    dict(case_id="E02", sequence="uav0000307_04531_s", selector="last_veto_before_first_failure"),
    dict(case_id="R01", sequence="uav0000226_05370_s", selector="last_veto_before_first_recovery"),
    dict(case_id="P01", sequence="uav0000204_00000_s", selector="last_veto_before_fixed_frame", frame=664))
CONTROLLERS = ("quality", "protect")
ACTIONS = ("veto", "allow")
seal, VERIFY, publish_identical = prior.seal, prior.VERIFY, prior.publish_identical


def sealed(path): return prior.sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve()
    plan, old = prior.load_plan(root)
    models = e66.load_models(plan, root, old)
    run = e66.completed(plan, root, old, models)
    report = prior.sealed(root/"analysis/stage66_decision.json")
    tables = {"stage66_"+n+".csv" for n in ("policy_summary", "paired", "calibration", "scope", "focus",
        "video_summary", "events", "video_influence", "fit_support")}
    if (report.get("status") not in ("REVIEW_NATIVE_Q0_TAKEOVER_PILOT", "PROMISING_NATIVE_Q0_TAKEOVER_PILOT") or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("training_fingerprint") != models["fingerprint"] or report.get("evidence_sha256") != run["cache_sha256"] or
        report.get("runtime") != plan["runtime_identity"] or set(report.get("table_sha256", {})) != tables or
        report.get("formal_search_calls") != plan["formal_search_calls"] or report.get("maximum_source_box_error") != 0 or
        report.get("sgla_training_performed") is not False or report.get("source_quality_unchanged") is not True):
        raise ValueError("Need complete unmixed Stage-66 results")
    bindings = dict(plan["source_file_sha256"])
    bindings.update(plan["code_sha256"]); bindings.update(run["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for n in ("manifest.json", "run.json", "models/training.json", "evaluation/run.json", "smoke/stage66_smoke.json", "analysis/stage66_decision.json"):
        bindings[str(root/n)] = file_hash(root/n)
    VERIFY(bindings)
    return dict(root=root, plan=plan, models=models, run=run, report=report, bindings=bindings)


def read_unit(source, sequence, policy):
    plan = source["plan"]
    unit = next(u for u in plan["units"] if u["sequence"] == sequence and u["policy"] == policy)
    entry = next(e for e in plan["entries"] if e["sequence"] == sequence)
    data = prior.read_cache(e66.path_for(source["root"], unit), prior.metadata(plan, unit, "evaluation", source["models"]))
    if data is None or len(data["frame_id"]) != len(entry["frames"])-1: raise ValueError("Missing full source trace")
    e66.audit(data, entry, unit, plan, source["models"])
    return data, entry, unit


def changed_veto(data):
    q = np.asarray(data["quality_choice"], int)
    proposed = data["candidate_boxes"][np.arange(len(q)), q]
    return data["veto"].astype(bool) & np.any(proposed != data["candidate_boxes"][:, 0], axis=1)


def pick(spec, data, events, design):
    """One fixed rule; missing support is recorded, never replaced with a better case."""
    eligible = changed_veto(data); mode = spec["selector"]
    if mode == "first_changed_veto": found = np.flatnonzero(eligible)
    else:
        if mode == "last_veto_before_first_failure": event = events[0]["start"]+1 if events else None
        elif mode == "last_veto_before_first_recovery":
            recovered = [e["recovery"]+1 for e in events if e["recovery"] is not None]
            event = recovered[0] if recovered else None
        elif mode == "last_veto_before_fixed_frame": event = spec["frame"]
        else: raise ValueError("Unknown fixed anchor rule")
        if event is None: return None, "missing_source_event"
        frames = data["frame_id"]
        found = np.flatnonzero(eligible & (frames >= max(1,event-design["event_lookback"]+1)) & (frames <= event))
    if not len(found): return None, "no_changed_veto_in_declared_window"
    frame = int(data["frame_id"][found[0] if mode == "first_changed_veto" else found[-1]])
    if frame+design["horizon"]-1 > len(data["frame_id"]): return None, "insufficient_full_horizon"
    return frame, "supported"


def resolve(source):
    requests, cases, traces = [], [], {}
    for spec in CONFIG:
        seq = spec["sequence"]
        if seq not in traces:
            data, entry, unit = read_unit(source, seq, "protect")
            events = failure_events(tracking_iou(data["boxes"], np.asarray(entry["ground_truth"])[1:]))
            traces[seq] = data, entry, unit, events
        data, entry, unit, events = traces[seq]
        frame, status = pick(spec, data, events, DESIGN)
        requests.append(dict(spec, status=status, resolved_frame=frame))
        if frame is None: continue
        cases.append(dict(spec, frame=frame, unit=unit, allow_choice=int(data["quality_choice"][frame-1]),
            source_trace_sha256=file_hash(e66.path_for(source["root"], unit)),
            head_sha256=source["run"]["observed_heads"][prior.key(unit)]))
    if not cases: raise ValueError("No predeclared anchor has a full continuation; see source support")
    if len({r["case_id"] for r in requests}) != len(requests): raise ValueError("Duplicate case ID")
    return requests, cases


def make_plan(root, hardware):
    source = load_source(root); old = source["plan"]
    if set(hardware) != {"device", "gpu_uuid"} or hardware["device"] != "NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need observed L40 identity")
    identity = copy.deepcopy(old["runtime_identity"]); identity["gpu_uuid"] = hardware["gpu_uuid"]
    differences = prior.c64.e63.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in differences): raise ValueError("Only physical L40 UUID may change")
    requests, cases = resolve(source); h = DESIGN["horizon"]
    if h < 2: raise ValueError("Need at least one successor")
    return seal(dict(schema=SCHEMA, role="consumed_development_matched_veto_continuation_diagnostic",
        source_input=str(source["root"]), source_report_fingerprint=source["report"]["fingerprint"],
        source_training_fingerprint=source["models"]["fingerprint"], source_file_sha256=source["bindings"],
        design=dict(DESIGN), controller_design=copy.deepcopy(old["design"]), actor_design=copy.deepcopy(old["actor_design"]),
        requests=requests, cases=cases, controllers=list(CONTROLLERS), actions=list(ACTIONS),
        hardware=hardware, runtime=identity["tracker"], runtime_identity=identity, declared_runtime_differences=differences,
        source_runtime_identity=old["runtime_identity"], shared_calls=4*len(cases), tail_calls=16*(h-1)*len(cases),
        formal_search_calls=(4+16*(h-1))*len(cases), smoke_calls_reused=4*h, counterfactual_branches=4*len(cases),
        **{k:copy.deepcopy(old[k]) for k in ("root", "search_size", "search_factor", "window")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})), source


def load_plan(root):
    plan = sealed(Path(root)/"manifest.json"); expected, source = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected: raise ValueError("Frozen Stage-67 source/code/anchor design changed")
    return plan, source


def path_for(root, case, controller=None, action=None):
    suffix = "shared" if controller is None else controller+"__"+action
    return Path(root)/"collection/traces"/(case["case_id"]+"__"+suffix+".npz")


def metadata(plan, case, controller=None, action=None):
    return dict(plan_fingerprint=plan["fingerprint"], case_id=case["case_id"], controller=controller, action=action,
                source_trace_sha256=case["source_trace_sha256"], training_fingerprint=plan["source_training_fingerprint"])


# Reuse receipt mechanics, with Stage-67 schema; never write any Stage-66 artifact.
def save(path, data, meta):
    import json
    from stage4_common import write_cache
    path = Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-67 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Unknown orphan cache")
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))


def read_cache(path, meta):
    import json
    path = Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    receipt = sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata") != meta or not path.is_file() or file_hash(path) != receipt["sha256"]: raise ValueError("Stage-67 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Mixed Stage-67 cache")
        return {k:f[k] for k in f.files if k != "metadata_json"}
