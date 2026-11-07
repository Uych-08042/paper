"""Fixed Stage-74 failure windows and source-verified shared-state diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import stage74_ppo_common as source_common
import stage74_ppo_experiment as source_experiment
from stage27_uav_common import failure_events, tracking_iou
from stage68_advantage_summary import publish_table

SCHEMA = "stage75_four_choice_failure_replay_v1"
DESIGN = dict(horizon=32, lookback=32, early_offset=-8, good_iou=.5, low_iou=.2)
FILES = ("stage75_ppo_failure_common.py", "stage75_ppo_failure_experiment.py",
         "test_stage75_ppo_failure.py", "run_stage75_ppo_failure.sh")
SOURCE_TABLES = (
    "policy_summary", "paired", "seed_effects", "focus", "scope", "video_influence",
    "candidate_summary", "candidate_video", "seed_video_summary", "video_summary",
    "events", "training_episode", "training_summary", "optimization_summary",
    "training_scope", "fit_support", "training_log",
)
seal = source_common.seal
verify = source_common.VERIFY
publish_identical = source_common.publish_identical
file_hash = source_common.file_hash
fingerprint = source_common.fingerprint
e63 = source_experiment.e63


def case_spec(case_id, sequence, seed, frame, purpose, event, policy="ppo", reference_seed=None):
    return dict(case_id=case_id, sequence=sequence, seed=seed, purpose=purpose,
                reference_policy=policy, reference_seed=seed if reference_seed is None else reference_seed,
                reference_event=event, reference_frame=frame, offset=0)


FAILURES = (
    case_spec("D01", "uav0000003_00000_s", 2171, 138, "short_failure", "first_failure"),
    case_spec("D02", "uav0000003_00000_s", 2172, 140, "short_failure", "first_failure"),
    case_spec("D03", "uav0000304_00253_s", 2172, 1090, "long_failure", "first_failure"),
)
CASES = FAILURES + tuple(
    dict(spec, case_id=f"E{index:02d}", purpose="pre_failure", offset=DESIGN["early_offset"])
    for index, spec in enumerate(FAILURES, 1)
) + (
    case_spec("S01", "uav0000304_00253_s", 2171, 1090, "stable_seed", "first_failure", reference_seed=2172),
    case_spec("R01", "uav0000176_00000_s", 2171, 388, "recovery", "first_recovery"),
    case_spec("R02", "uav0000176_00000_s", 2172, 321, "recovery", "first_recovery"),
    case_spec("R03", "uav0000307_04531_s", 2171, 474, "recovery", "first_recovery"),
    case_spec("R04", "uav0000307_04531_s", 2172, 471, "recovery", "first_recovery"),
    case_spec("R05", "uav0000204_00000_s", 2171, 664, "preserved_gain", "first_failure", "q0", 0),
    case_spec("R06", "uav0000204_00000_s", 2172, 664, "preserved_gain", "first_failure", "q0", 0),
)


def sealed(path):
    return source_common.c66.sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve()
    plan, prior = source_common.load_plan(root)
    models = source_experiment.load_models(root, plan, prior)
    run = source_experiment.completed(root, plan, prior, models)
    bridge = source_experiment.bridge_gate(root, plan)
    report = source_common.sealed(root / "analysis/stage74_decision.json")
    expected = dict(
        plan_fingerprint=plan["fingerprint"], training_fingerprint=models["fingerprint"],
        run_fingerprint=run["fingerprint"], runtime=plan["runtime_identity"],
        source_report_fingerprint=prior["stage73"]["report"]["fingerprint"],
        source_training_fingerprint=prior["stage73"]["models"]["fingerprint"],
        source_bindings_fingerprint=fingerprint(plan["source_file_sha256"]),
        source_bridge_fingerprint=bridge["fingerprint"],
        training_evidence_fingerprint=fingerprint(models["training_evidence_sha256"]),
        eval_evidence_fingerprint=fingerprint(run["cache_sha256"]),
        full_evaluation_trajectories=len(plan["units"]), eval_search_calls=plan["formal_eval_calls"],
        actions=[0, 1, 2, 3], new_recovery_search_calls=0, recovery_branch_present=False,
        sgla_training_performed=False, source_models_unchanged=True, maximum_source_common_field_error=0.,
    )
    if (report.get("status") not in ("REVIEW_DIRECT_FOUR_CANDIDATE_PPO_PILOT", "PROMISING_DIRECT_FOUR_CANDIDATE_PPO_PILOT")
            or any(report.get(key) != value for key, value in expected.items())
            or set(report.get("table_sha256", {})) != {f"stage74_{name}.csv" for name in SOURCE_TABLES}):
        raise ValueError("Need the complete frozen Stage-74 four-choice evaluation")

    bindings = dict(plan["source_file_sha256"])
    for records in (plan["code_sha256"], models["training_evidence_sha256"], run["cache_sha256"], bridge["cache_sha256"]):
        bindings.update(records)
    bindings.update({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
    for name in ("manifest.json", "models/training.json", "evaluation/run.json", "analysis/stage74_decision.json",
                 "smoke/train.json", "smoke/evaluate.json", "smoke/bridge.json"):
        bindings[str(root / name)] = file_hash(root / name)
    verify(bindings)
    return dict(root=root, plan=plan, prior=prior, models=models, run=run, report=report, bindings=bindings)


def read_trace(source, unit):
    plan = source["plan"]
    entry = source_common.entry_for(plan, unit["sequence"])
    path = source_common.eval_path(source["root"], unit)
    trace = source_common.read_cache(path, source_experiment.eval_meta(plan, unit, source["models"]))
    count = len(entry["frames"]) - 1
    if trace is None or set(trace) != set(source_experiment.FIELDS):
        raise ValueError(f"Missing Stage-74 trace or fields: {path}")
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, count + 1))
    boxes = trace["candidate_boxes"]
    actions = trace["selected"]
    if (boxes.shape != (count, 4, 4) or actions.shape != (count,) or actions.dtype.kind not in "iu"
            or np.any((actions < 0) | (actions > 3)) or not np.isfinite(boxes).all() or np.any(boxes[..., 2:] <= 0)):
        raise ValueError(f"Invalid Stage-74 candidates or actions: {path}")
    np.testing.assert_array_equal(trace["boxes"], boxes[np.arange(count), actions])
    return trace


def frame_category(ious, action):
    chosen = ious[action]
    if chosen >= DESIGN["good_iou"]:
        return "selected_good"
    if max(ious) < DESIGN["good_iou"]:
        return "all_low" if max(ious) < DESIGN["low_iou"] else "no_good_candidate"
    if ious[0] >= DESIGN["good_iou"] and chosen < DESIGN["low_iou"]:
        return "good_q0_replaced_by_low"
    return "missed_good_candidate"


def resolve_case(spec, unit, entry, trace, reference_events, head):
    if spec["reference_event"] == "first_failure":
        index = reference_events[0]["start"] if reference_events else None
    elif spec["reference_event"] == "first_recovery":
        index = next((event["recovery"] for event in reference_events if event["recovery"] is not None), None)
    else:
        raise ValueError("Unknown Stage-75 reference event")
    if index is None or index + 1 != spec["reference_frame"]:
        raise ValueError(f"Source event differs for {spec['case_id']}")
    frame = spec["reference_frame"] + spec["offset"]
    last_frame = len(entry["frames"]) - 1
    if not 1 <= frame <= last_frame:
        raise ValueError(f"Stage-75 anchor outside source: {spec['case_id']}")
    horizon = min(DESIGN["horizon"], last_frame - frame + 1)
    legal = np.flatnonzero(source_common.action_mask(trace["candidate_boxes"][frame - 1])).tolist()
    action = int(trace["selected"][frame - 1])
    if action not in legal:
        raise ValueError(f"Source PPO action is not canonical: {spec['case_id']}")
    return dict(spec, unit=unit, frame=frame, horizon=horizon, legal_actions=legal,
                recorded_action=action, head_sha256=head,
                search_calls=4 * (1 + len(legal) * (horizon - 1)))


def cached_case_rows(case, traces, overlaps, ppo, myopic, design):
    frame = case["frame"]
    index = frame - 1
    start = max(1, frame - DESIGN["lookback"])
    end = frame + case["horizon"] - 1
    identity = {key: case[key] for key in ("case_id", "sequence", "seed", "frame", "purpose")}
    own_key = ("ppo", case["seed"])
    own = traces[own_key]
    states = []
    timeline = []
    for (policy, seed), trace in traces.items():
        context = dict(identity, history_policy=policy, history_seed=seed)
        y = overlaps[policy, seed]
        boxes = trace["candidate_boxes"][index]
        features = trace["choice_features"][index]
        ppo_score = source_common.gate(features, boxes, ppo, design)
        sl_score = source_common.gate(features, boxes, myopic, design)
        choices = dict(recorded=int(trace["selected"][index]), q0=0, quality=int(trace["quality_choice"][index]),
                       ppo=int(ppo_score["probabilities"].argmax()), myopic=int(sl_score["probabilities"].argmax()),
                       best_current=source_common.best_label(y[index], ppo_score["action_mask"], design["tie_epsilon"]))
        if seed == case["seed"] and policy in ("ppo", "myopic"):
            score = ppo_score if policy == "ppo" else sl_score
            for name in ("action_mask", "logits", "probabilities"):
                np.testing.assert_array_equal(score[name], trace[name][index])
            if choices[policy] != choices["recorded"]:
                raise ValueError("Cached actor decision differs from the frozen model")
        states.append(dict(context, legal_actions="|".join(map(str, np.flatnonzero(ppo_score["action_mask"]))),
                           ppo_category=frame_category(y[index], choices["ppo"]),
                           **{f"candidate_{q}_iou": float(y[index, q]) for q in range(4)},
                           **{f"{name}_action": action for name, action in choices.items()},
                           **{f"{name}_iou": float(y[index, action]) for name, action in choices.items()},
                           **{f"{name}_p{q}": float(score["probabilities"][q])
                              for name, score in (("ppo", ppo_score), ("myopic", sl_score)) for q in range(4)}))
        for row in range(start - 1, end):
            action = int(trace["selected"][row])
            timeline.append(dict(context, tracked_frame=row + 1, relative_frame=row - index, selected=action,
                                 selected_iou=float(y[row, action]), best_iou=float(y[row].max()),
                                 category=frame_category(y[row], action),
                                 **{f"q{q}_iou": float(y[row, q]) for q in range(4)}))

    divergence = []
    own_good = overlaps[own_key].max(axis=1) >= DESIGN["good_iou"]
    for reference in (("q0", 0), ("quality", 0), ("myopic", case["seed"]), ("ppo", 2171 if case["seed"] == 2172 else 2172)):
        other = traces[reference]
        other_good = overlaps[reference].max(axis=1) >= DESIGN["good_iou"]
        gaps = np.flatnonzero((~own_good & other_good)[start - 1:frame]) + start
        last_both = np.flatnonzero((own_good & other_good)[start - 1:index]) + start
        divergence.append(dict(identity, reference_policy=reference[0], reference_seed=reference[1],
                               first_output_difference=first_difference(own["boxes"], other["boxes"], frame),
                               first_history_difference=first_difference(own["candidate_parents"], other["candidate_parents"], frame),
                               first_crop_difference=first_difference(own["crop_sha256"], other["crop_sha256"], frame),
                               first_candidate_difference=first_difference(own["candidate_boxes"], other["candidate_boxes"], frame),
                               window_first_own_missing_reference_good=int(gaps[0]) if len(gaps) else None,
                               window_own_missing_reference_good_frames=len(gaps),
                               window_last_both_good_before_anchor=int(last_both[-1]) if len(last_both) else None,
                               anchor_own_has_good=bool(own_good[index]), anchor_reference_has_good=bool(other_good[index])))
    y = overlaps[own_key]
    categories = [frame_category(y[row], int(own["selected"][row])) for row in range(start - 1, index)]
    anchor = dict(identity, reference_frame=case["reference_frame"], offset=case["offset"],
                  horizon=case["horizon"], window_start=start, recorded_action=case["recorded_action"],
                  legal_actions="|".join(map(str, case["legal_actions"])), search_calls=case["search_calls"],
                  own_anchor_category=frame_category(y[index], case["recorded_action"]),
                  own_anchor_selected_iou=float(y[index, case["recorded_action"]]),
                  own_anchor_best_iou=float(y[index].max()),
                  before_no_good=sum(name in ("no_good_candidate", "all_low") for name in categories),
                  before_missed_good=categories.count("missed_good_candidate"),
                  before_good_q0_replaced_by_low=categories.count("good_q0_replaced_by_low"))
    return anchor, states, divergence, timeline


def first_difference(a, b, through):
    different = (a[:through] != b[:through]).reshape(through, -1).any(axis=1)
    rows = np.flatnonzero(different)
    return int(rows[0] + 1) if len(rows) else None


def cached_tables(source):
    plan = source["plan"]
    seeds = plan["design"]["seeds"]
    if seeds != [2171, 2172]:
        raise ValueError("Stage-75 requires both Stage-74 seeds")
    providers = [("q0", 0), ("quality", 0), ("future", 0)]
    providers += [(policy, seed) for policy in ("ppo", "myopic") for seed in seeds]
    cases = []
    tables = dict(anchors=[], same_state=[], divergence=[], timeline=[])
    for sequence in dict.fromkeys(spec["sequence"] for spec in CASES):
        entry = source_common.entry_for(plan, sequence)
        units = {(u["policy"], u["seed"]): u for u in plan["units"] if u["sequence"] == sequence}
        traces = {key: read_trace(source, units[key]) for key in providers}
        gt = np.asarray(entry["ground_truth"], float)[1:, None]
        overlaps = {key: tracking_iou(data["candidate_boxes"], gt) for key, data in traces.items()}
        events = {key: failure_events(overlaps[key][np.arange(len(data["frame_id"])), data["selected"]])
                  for key, data in traces.items()}
        for spec in (spec for spec in CASES if spec["sequence"] == sequence):
            unit = units["ppo", spec["seed"]]
            case = resolve_case(spec, unit, entry, traces["ppo", spec["seed"]],
                                events[spec["reference_policy"], spec["reference_seed"]],
                                source["run"]["observed_heads"][source_common.key(unit)])
            cases.append(case)
            ppo = source_experiment.learned_pack(source["models"], unit, source["prior"])
            myopic = source_experiment.learned_pack(source["models"], dict(unit, policy="myopic"), source["prior"])
            anchor, states, divergence, timeline = cached_case_rows(case, traces, overlaps, ppo, myopic, plan["design"])
            tables["anchors"].append(anchor)
            tables["same_state"].extend(states)
            tables["divergence"].extend(divergence)
            tables["timeline"].extend(timeline)
    return cases, tables


def make_plan(source, cases):
    if len(cases) != len(CASES) or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Incomplete fixed Stage-75 case set")
    return seal(dict(schema=SCHEMA, role="consumed_development_failure_and_matched_replay", design=dict(DESIGN),
                     source_input=str(source["root"]), source_report_fingerprint=source["report"]["fingerprint"],
                     source_training_fingerprint=source["models"]["fingerprint"], source_file_sha256=source["bindings"],
                     code_sha256={str(Path(__file__).with_name(name).resolve()): file_hash(Path(__file__).with_name(name))
                                  for name in FILES},
                     cases=cases, branch_count=sum(len(case["legal_actions"]) for case in cases),
                     search_calls=sum(case["search_calls"] for case in cases),
                     maximum_search_calls=sum(4 + 16 * (case["horizon"] - 1) for case in cases)))


def publish_tables(root, tables):
    directory = root / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = directory / f"stage75_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = file_hash(path)
    return hashes
