"""Sealed, outcome-selected development cases for one-choice continuations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import stage62_candidate_audit as prior
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical

SCHEMA = "stage63_matched_one_choice_continuation_v1"
DESIGN = dict(horizon=32, good_iou=.5, low_iou=.2, source_box_atol=0.)
FILES = ("stage63_choice_common.py", "stage63_choice_experiment.py", "stage63_choice_summary.py",
         "test_stage63_choice.py", "run_stage63_choice.sh")
TABLES62 = {"stage62_"+n+".csv" for n in ("policy_summary", "video_summary", "seed_summary", "scope",
                                         "transitions", "focus", "timeline", "candidates")}


def spec(case_id, sequence, origin, seed, policy, selector, purpose, **kwargs):
    return dict(case_id=case_id, sequence=sequence, origin=origin, seed=seed, policy=policy,
                selector=selector, purpose=purpose, **kwargs)


CONFIG = (
    spec("R01", "uav0000304_00253_s", "native", 0, "recrop", "first_q0_good_to_low", "q0_damage"),
    spec("R02", "uav0000226_05370_s", "native", 0, "history", "first_available_selected_miss", "early_missed_option"),
    spec("R03", "uav0000307_04531_s", "endpoint", 2132, "history", "first_q0_good_lost", "early_q0_loss"),
    spec("R04", "uav0000307_04531_s", "endpoint", 2133, "history", "first_q0_good_lost", "early_q0_loss"),
    spec("R05", "uav0000307_04531_s", "endpoint", 2132, "history", "unique_window_miss", "rare_recovery_option", first=441, last=501),
    spec("N01", "uav0000226_05370_s", "native", 0, "history", "first_failure_no_good", "no_good_onset"),
    spec("N02", "uav0000307_04531_s", "endpoint", 2132, "history", "first_failure_no_good", "no_good_onset"),
    spec("N03", "uav0000307_04531_s", "endpoint", 2133, "history", "first_failure_no_good", "no_good_onset"),
    spec("N04", "uav0000304_00253_s", "endpoint", 2132, "history", "first_failure_no_good", "late_no_good_onset"),
    spec("P01", "uav0000176_00000_s", "native", 0, "history", "fixed_good_q0", "positive_state", frame=162),
    spec("P02", "uav0000204_00000_s", "native", 0, "history", "fixed_good_q0", "positive_state", frame=664),
    spec("P03", "uav0000304_00253_s", "endpoint", 2133, "history", "fixed_good_q0", "positive_state", frame=554),
)


def seal(body):
    return dict(body, fingerprint=fingerprint(body))


def read_sealed(path, schema=SCHEMA):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != schema or value.get("fingerprint") != fingerprint({k:v for k,v in value.items() if k != "fingerprint"}):
        raise ValueError("Sealed Stage-63/source content changed: "+str(path))
    return value


def load_source(root):
    root = Path(root).expanduser().resolve()
    old = read_sealed(root/"manifest.json", prior.SCHEMA)
    source = prior.load_source(old["source_input"])
    code = {str(Path(prior.__file__).with_name(n)):file_hash(Path(prior.__file__).with_name(n)) for n in prior.FILES}
    if (old.get("source_file_sha256") != source["bindings"] or old.get("code_sha256") != code or
        old.get("source_plan_fingerprint") != source["plan"]["fingerprint"] or
        old.get("source_run_fingerprint") != source["run"]["fingerprint"] or
        old.get("source_report_fingerprint") != source["report"]["fingerprint"] or
        old.get("design") != prior.c.DESIGN or old.get("cases") != list(prior.c.CASES) or
        old.get("cohorts") != list(prior.c.COHORTS)):
        raise ValueError("Frozen Stage-62 design/source changed")
    report = read_sealed(root/"analysis/stage62_diagnostic.json", prior.SCHEMA)
    if (report.get("status") != "COMPLETE_CACHED_CANDIDATE_SELECTION_DIAGNOSTIC" or
        report.get("plan_fingerprint") != old["fingerprint"] or report.get("role") != old["role"] or
        report.get("source_plan_fingerprint") != source["plan"]["fingerprint"] or
        report.get("source_report_fingerprint") != source["report"]["fingerprint"] or
        report.get("source_bindings_fingerprint") != fingerprint(source["bindings"]) or
        report.get("source_runtime") != source["plan"]["runtime_identity"] or
        report.get("source_trajectories") != len(source["plan"]["units"]) or
        report.get("cached_decoded_candidates") != source["plan"]["formal_search_calls"] or
        set(report.get("table_sha256", {})) != TABLES62 or
        any(report.get(k) != 0 for k in ("new_network_calls", "new_full_trajectories", "new_image_reads", "new_models")) or
        report.get("training_performed") is not False or report.get("full_response_grids_available") is not False or
        report.get("counterfactual_continuations_performed") is not False):
        raise ValueError("Need the complete unmixed Stage-62 diagnostic")
    bindings = dict(source["bindings"])
    bindings.update(code)
    bindings.update({str(root/"analysis"/n):sha for n,sha in report["table_sha256"].items()})
    for name in ("manifest.json", "analysis/stage62_diagnostic.json"):
        bindings[str(root/name)] = file_hash(root/name)
    prior.c.prior.verify_files(bindings)
    return dict(root=root, plan62=old, report=report, source=source, bindings=bindings)


def select_frame(item, metrics, events):
    mode = item["selector"]
    if mode.startswith("first_") and mode[6:] in metrics:
        found = np.flatnonzero(metrics[mode[6:]])
        if not len(found):
            raise ValueError("Predeclared condition has no support: "+item["case_id"])
        return int(found[0])+1
    if mode == "unique_window_miss":
        frames = metrics["frame_id"]
        found = np.flatnonzero((frames >= item["first"]) & (frames <= item["last"]) & metrics["available_selected_miss"])
        if len(found) != 1:
            raise ValueError("Expected the unique cached recovery-window miss: "+item["case_id"])
        return int(found[0])+1
    if mode == "first_failure_no_good":
        if not events or not metrics["no_good_decoded"][events[0]["start"]]:
            raise ValueError("Missing predeclared no-good failure: "+item["case_id"])
        return int(events[0]["start"])+1
    if mode in ("fixed_good_q0", "fixed_frame"):
        frame = int(item["frame"])
        if not 1 <= frame <= len(metrics["frame_id"]):
            raise ValueError("Fixed anchor outside source")
        if mode == "fixed_good_q0" and not metrics["q0_good"][frame-1]:
            raise ValueError("Positive control has no good q0")
        return frame
    raise ValueError("Unknown fixed Stage-63 anchor rule")


def resolve_cases(inputs):
    old = inputs["source"]["plan"]
    cases, traces = [], {}
    for item in CONFIG:
        candidates = [u for u in old["units"] if all(u[k] == item[k] for k in ("sequence", "origin", "seed", "policy"))]
        if len(candidates) != 1 or item["policy"] not in ("history", "recrop"):
            raise ValueError("Need exactly one four-pass source unit: "+item["case_id"])
        unit = candidates[0]
        key = prior.c.prior.key(unit)
        if key not in traces:
            data, entry, path, head = prior.read_trace(inputs["source"], unit)
            metrics, _ = prior.c.frame_metrics(data, entry)
            events = prior.failure_events(metrics["selected_iou"])
            traces[key] = (data, entry, path, head, metrics, events)
        data, entry, path, head, metrics, events = traces[key]
        frame = select_frame(item, metrics, events)
        if frame+DESIGN["horizon"] > len(entry["frames"]):
            raise ValueError("Need a full predeclared continuation horizon; do not silently shorten")
        cases.append(dict(item, frame=frame, unit=dict(unit), head_sha256=head, source_trace_sha256=file_hash(path),
            recorded_choice=int(data["selected"][frame-1]), immediate_best=int(metrics["best_index"][frame-1]),
            q0_iou=float(metrics["q0_iou"][frame-1]), recorded_iou=float(metrics["selected_iou"][frame-1]),
            best_decoded_iou=float(metrics["best_decoded_iou"][frame-1])))
    if not cases or len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Need unique nonempty Stage-63 cases")
    return cases, traces


def make_plan(source_root, hardware):
    inputs = load_source(source_root)
    old = inputs["source"]["plan"]
    if set(hardware) != {"device", "gpu_uuid"} or hardware["device"] != "NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Observe an L40 and bind its UUID before preparing Stage-63")
    identity = copy.deepcopy(old["runtime_identity"])
    identity["gpu_uuid"] = hardware["gpu_uuid"]
    changed = prior.c.prior.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in changed):
        raise ValueError("Only the physical L40 UUID may differ; source diagonals must still reproduce exactly")
    cases, _ = resolve_cases(inputs)
    calls_per_case = 4+16*(DESIGN["horizon"]-1)
    plan = seal(dict(schema=SCHEMA, role="consumed_development_matched_one_choice_continuation",
        source_input=str(inputs["root"]), source_report_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"], source_runtime_identity=old["runtime_identity"],
        hardware=dict(hardware), declared_runtime_differences=changed, runtime_identity=identity, runtime=identity["tracker"],
        design=dict(DESIGN), actor_design=copy.deepcopy(old["design"]), cases=cases,
        **{k:copy.deepcopy(old[k]) for k in ("root", "search_factor", "search_size", "window")},
        shared_anchor_calls=4*len(cases), tail_calls=16*(DESIGN["horizon"]-1)*len(cases),
        formal_search_calls=calls_per_case*len(cases), counterfactual_branches=4*len(cases),
        smoke_calls_reused=4*DESIGN["horizon"], new_full_trajectories=0,
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES}))
    return plan, inputs


def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json")
    expected, inputs = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected:
        raise ValueError("Frozen Stage-63 source/code/design changed")
    return plan, inputs


def actor_plan(plan):
    return {k:plan[k] for k in ("root", "search_factor", "search_size", "window")} | {"design":plan["actor_design"]}


def metadata(plan, case, kind, candidate=None):
    return dict(plan_fingerprint=plan["fingerprint"], runtime_fingerprint=fingerprint(plan["runtime_identity"]),
                case_id=case["case_id"], kind=kind, candidate=candidate, source_trace_sha256=case["source_trace_sha256"],
                head_sha256=case["head_sha256"])


def save(path, data, meta):
    path = Path(path)
    if path.with_suffix(".receipt.json").exists():
        raise ValueError("Never overwrite a committed Stage-63 cache")
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != meta:
                raise ValueError("Unidentified orphan Stage-63 cache")
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    path = Path(path)
    if not path.with_suffix(".receipt.json").exists():
        return None
    receipt = read_sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata") != meta or not path.is_file() or file_hash(path) != receipt["sha256"]:
        raise ValueError("Committed Stage-63 cache changed")
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta:
            raise ValueError("Mixed Stage-63 cache")
        return {k:f[k] for k in f.files if k != "metadata_json"}
