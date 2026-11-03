"""Audit Stage-64 cache on CPU: common-state selection errors and fit-state support."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage65_selection_common as c
from stage4_common import file_hash, fingerprint
from stage27_uav_common import publish_identical, nonempty_path
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from validation_metrics import tracking_iou, IOU_PROTOCOL

c64, e64 = c.c64, c.e64
seal = c64.seal
TABLES64 = {"stage64_"+n+".csv" for n in ("policy_summary", "paired", "scope", "fit_support", "video_summary",
                                                       "seed_video_summary", "focus", "events", "video_influence")}
PAIR_METRICS = ("mean_selected_iou_delta", "selected_good_fraction_delta", "selected_low_fraction_delta",
                "q0_good_to_low_fraction_delta", "choice_disagreement_fraction")


def load_source(root):
    root = Path(root).expanduser().resolve(); plan, old = c64.load_plan(root)
    models = e64.load_models(plan, root); run = e64.completed(plan, root, models)
    report = c64.read_sealed(root/"analysis/stage64_decision.json")
    if (report.get("status") not in ("REVIEW_CANDIDATE_RELIABILITY_PILOT", "PROMISING_CANDIDATE_RELIABILITY_PILOT") or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("training_fingerprint") != models["fingerprint"] or report.get("runtime") != plan["runtime_identity"] or
        report.get("evidence_sha256") != run["cache_sha256"] or set(report.get("table_sha256", {})) != TABLES64 or
        report.get("sgla_training_performed") is not False or report.get("selector_fitting_performed") is not True or
        report.get("fit_origins") != ["native"] or report.get("maximum_source_box_error") != 0 or
        report.get("full_trajectories") != len(plan["units"]) or report.get("formal_search_calls") != plan["formal_search_calls"] or
        report.get("declared_runtime_differences") != plan["declared_runtime_differences"] or
        report.get("new_selectors") != 2*len(plan["folds"]) or models.get("scorers") != 2*len(plan["folds"])):
        raise ValueError("Need complete unmixed Stage-64 result and selectors")
    if set(run.get("observed_heads", {})) != {c64.key(u) for u in plan["units"]}: raise ValueError("Missing frozen head identities")
    l40 = old["l40"]
    for unit in plan["units"]:
        oldunit = e64.source_unit(old, unit, "current")
        identity = c64.prior.prior.source_experiment.model_identity(l40["plan"], l40["records"], oldunit)
        head = identity.split(":", 1)[1] if unit["origin"] == "endpoint" else l40["gate"]["observed_head_sha256"]["native_0"]
        if run["observed_heads"][c64.key(unit)] != head: raise ValueError("Frozen source head identity differs")
    bindings = dict(plan["source_file_sha256"]); bindings.update(plan["code_sha256"]); bindings.update(run["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n, h in report["table_sha256"].items()})
    for name in ("manifest.json", "run.json", "models/training.json", "evaluation/run.json", "smoke/stage64_smoke.json", "analysis/stage64_decision.json"):
        bindings[str(root/name)] = file_hash(root/name)
    e64.VERIFY(bindings)
    return dict(root=root, plan=plan, old=old, models=models, run=run, report=report, bindings=bindings)


def read_unit(source, unit):
    plan, root, models = (source[k] for k in ("plan", "root", "models"))
    entry = next(e for e in plan["entries"] if e["sequence"] == unit["sequence"])
    path = e64.path_for(root, unit); data = c64.read_cache(path, c64.meta(plan, unit, "evaluation", models))
    if data is None or len(data["frame_id"]) != len(entry["frames"])-1: raise ValueError("Missing complete Stage-64 state provider")
    e64.audit(data, entry, unit, plan, models); e64.check_source(data, source["old"], unit)
    return data, entry


def cases(source):
    resolved = []
    for item in c.CASES:
        units = [u for u in source["plan"]["units"] if all(u[k] == item[k] for k in ("sequence", "origin", "seed"))
                 and u["policy"] == item["anchor_policy"]]
        if len(units) != 1: raise ValueError("Missing fixed focus provider: "+item["case_id"])
        unit = units[0]
        if item["anchor"] == "source_current_first_recovery":
            data, entry, _, _ = c64.prior.prior.read_trace(source["old"]["l40"], e64.source_unit(source["old"], unit, "current"))
        else: data, entry = read_unit(source, unit)
        frame = c.anchor_frame(item["anchor"], tracking_iou(data["boxes"], np.asarray(entry["ground_truth"])[1:]), item.get("frame"))
        resolved.append(dict(item, frame=frame))
    if not resolved or len({r["case_id"] for r in resolved}) != len(resolved): raise ValueError("Need unique nonempty focus cases")
    return resolved


def fit_features(data, entry, plan):
    history = [np.asarray(entry["initial_box"], float)]; rows = []
    for b, f, accepted in zip(data["candidate_boxes"], data["features"], data["boxes"]):
        rows.append(c64.choice_features(f, b, history))
        history.append(accepted); history = history[-plan["actor_design"]["history"]:]
    return np.asarray(rows)


def audit(input_root, output_root):
    input_root, output_root = Path(input_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Use a separate Stage-65 output directory")
    source = load_source(input_root); p64 = source["plan"]; models = source["models"]
    focus_cases = cases(source)
    code = {str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in c.FILES}
    plan = seal(dict(schema=c.SCHEMA, role="consumed_development_cached_common_state_selector_audit",
        source_input=str(input_root), source_plan_fingerprint=p64["fingerprint"], source_report_fingerprint=source["report"]["fingerprint"],
        source_training_fingerprint=models["fingerprint"], source_file_sha256=source["bindings"], code_sha256=code,
        design=dict(c.DESIGN), cases=focus_cases, iou_protocol=IOU_PROTOCOL, rules=list(c.RULES),
        unit_definition="four selectors on the same stored predicted history and four public boxes; no state advancement"))
    with writer_lock(output_root):
        if not (output_root/"manifest.json").exists() and any(p.name != "collect.lock" for p in output_root.iterdir()):
            raise ValueError("Use an empty Stage-65 directory")
        publish_identical(output_root/"manifest.json", plan)
        refs = {}; fit_inputs = {}; reconstructed_support = []
        for split in p64["folds"]:
            fold = str(split["fold"]); features, labels, weights, support = e64.examples(p64, source["old"], split["fold"])
            pack = models["models"][fold]["models"]
            for target in c64.TARGETS:
                if c64.fit_selector(features, labels, weights, target, p64["design"]["ridge"]) != pack[target]:
                    raise ValueError("Frozen fit coefficients do not reconstruct")
            refs[fold] = c.reference(features, weights, pack)
            fit_inputs[fold] = (features, labels, support)
            reconstructed_support.extend(support)
            print("[fit reference verified] fold "+fold, flush=True)
        if reconstructed_support != models["support"]: raise ValueError("Frozen fit support differs")
        reference_record = seal(dict(schema=c.SCHEMA, plan_fingerprint=plan["fingerprint"], references=refs,
                                     interpretation="fixed fit 1/99-percentile support description; no online threshold or calibration"))
        publish_identical(output_root/"reference.json", reference_record)
        seeds, shifts, scope, pairs, focus, timelines = [], [], [], [], [], []

        def consume(data, entry, unit, phase, features):
            fold = str(unit["fold"]); pack = models["models"][fold]["models"]
            m, detail, choices, scores = c.metrics(data, entry, features, pack, p64["design"]["tie_epsilon"])
            label = dict(phase=phase, sequence=unit["sequence"], origin=unit["origin"], seed=unit["seed"],
                         fold=unit["fold"], state_policy=unit["policy"])
            count = len(features)
            recorded_rule = unit["policy"] if phase == "held_out" else "hann"
            if not np.array_equal(choices[recorded_rule], data["selected"]): raise ValueError("Recorded policy choice does not reproduce")
            for rule in c.RULES: seeds.append(dict(label, rule=rule, **c.summarize(m[rule])))
            shifts.append(dict(label, frame_states=count, **c.shifted(features, detail["iou"], refs[fold])))
            for left, right in (("quality", "reliability"), ("quality", "hann"), ("reliability", "hann")):
                a, b = m[left], m[right]
                pairs.append(dict(label, comparison=left+"_minus_"+right, frame_states=count,
                    mean_selected_iou_delta=float((a["selected_iou"]-b["selected_iou"]).mean()),
                    selected_good_fraction_delta=float(a["selected_good"].mean()-b["selected_good"].mean()),
                    selected_low_fraction_delta=float(a["selected_low"].mean()-b["selected_low"].mean()),
                    q0_good_to_low_fraction_delta=float(a["q0_good_to_low"].mean()-b["q0_good_to_low"].mean()),
                    choice_disagreement_fraction=float((choices[left]!=choices[right]).mean())))
            scope.append(dict(label, frame_states=count, decoded_boxes=4*count, selector_applications=4*count,
                              actual_choice_reproduced=True, new_network_calls=0, new_image_reads=0))
            if phase != "held_out": return
            for spec in focus_cases:
                if any(spec[k] != unit[k] for k in ("sequence", "origin", "seed")): continue
                start = max(1, spec["frame"]-c.DESIGN["before"]); end = min(count, spec["frame"]+c.DESIGN["after"])
                mask = (data["frame_id"] >= start) & (data["frame_id"] <= end)
                tag = dict(label, case_id=spec["case_id"], anchor_frame=spec["frame"], first_frame=start, last_frame=end)
                for rule in c.RULES: focus.append(dict(tag, rule=rule, **c.summarize(m[rule], mask)))
                for i in np.flatnonzero(mask):
                    row = dict(tag, frame=int(data["frame_id"][i]), recorded_choice=int(data["selected"][i]),
                               best_decoded_iou=float(detail["iou"][i].max()), q0_iou=float(detail["iou"][i, 0]))
                    for q in range(4): row["candidate_"+str(q)+"_iou"] = float(detail["iou"][i, q])
                    for rule in c.RULES:
                        q = int(choices[rule][i]); row[rule+"_choice"] = q; row[rule+"_iou"] = float(detail["iou"][i, q])
                    for target in c64.TARGETS:
                        for q in range(4): row[target+"_score_q"+str(q)] = float(scores[target][i, q])
                    timelines.append(row)

        for split in p64["folds"]:
            fold = str(split["fold"]); features, labels, support = fit_inputs.pop(fold); offset = 0
            for row in support:
                oldunit = next(u for u in source["old"]["l40"]["plan"]["units"] if
                    (u["sequence"], u["origin"], u["policy"]) == (row["sequence"], "native", row["source_policy"]))
                data, entry, _, _ = c64.prior.prior.read_trace(source["old"]["l40"], oldunit)
                f = fit_features(data, entry, p64); n = len(f)
                np.testing.assert_array_equal(f, features[offset:offset+n])
                np.testing.assert_array_equal(tracking_iou(data["candidate_boxes"], np.asarray(entry["ground_truth"])[1:, None]), labels[offset:offset+n])
                consume(data, entry, dict(oldunit, fold=split["fold"]), "fit_reference", f); offset += n
            if offset != len(features): raise ValueError("Missing fit reference states")
        for unit in p64["units"]:
            data, entry = read_unit(source, unit)
            consume(data, entry, unit, "held_out", data["choice_features"])
            print("[same state audited] "+c64.key(unit), flush=True)

        base = ("phase", "origin", "state_policy")
        videos = c.aggregate(seeds, ("sequence",)+base+("rule",), "seeds")
        policies = c.aggregate(videos, base+("rule",), "videos")
        shift_videos = c.aggregate(shifts, ("sequence",)+base, "seeds", c.SHIFT)
        shift_policies = c.aggregate(shift_videos, base, "videos", c.SHIFT)
        pair_videos = c.aggregate(pairs, ("sequence",)+base+("comparison",), "seeds", PAIR_METRICS)
        pair_policies = c.aggregate(pair_videos, base+("comparison",), "videos", PAIR_METRICS)
        expected_focus = sum(sum(all(u[k] == spec[k] for k in ("sequence", "origin", "seed")) for u in p64["units"])*4 for spec in focus_cases)
        if len(focus) != expected_focus: raise ValueError("Missing predeclared focus state provider")
        held = [r for r in scope if r["phase"] == "held_out"]
        if len(held) != len(p64["units"]) or sum(r["decoded_boxes"] for r in held) != p64["formal_search_calls"]:
            raise ValueError("Missing Stage-64 source exposure")
        tables = dict(policy_summary=policies, matched_summary=pair_policies, feature_shift=shift_policies, scope=scope,
                      focus=focus, video_summary=videos, seed_summary=seeds, matched_video=pair_videos,
                      feature_video=shift_videos, feature_seed=shifts, focus_frames=timelines)
        e64.VERIFY(source["bindings"]); e64.VERIFY(code)
        out = output_root/"analysis"; out.mkdir(exist_ok=True)
        with writer_lock(out):
            for name, rows in tables.items(): publish_table(out/("stage65_"+name+".csv"), rows)
            report = seal(dict(schema=c.SCHEMA, status="COMPLETE_CACHED_SAME_STATE_SELECTOR_DIAGNOSTIC", role=plan["role"],
                plan_fingerprint=plan["fingerprint"], source_report_fingerprint=source["report"]["fingerprint"],
                source_training_fingerprint=models["fingerprint"], reference_fingerprint=reference_record["fingerprint"],
                source_bindings_fingerprint=fingerprint(source["bindings"]), source_runtime=p64["runtime_identity"],
                source_trajectories=len(held), held_out_frame_states=sum(r["frame_states"] for r in held),
                fit_reference_frame_states=sum(r["frame_states"] for r in scope if r["phase"] == "fit_reference"),
                cached_state_occurrences=sum(r["frame_states"] for r in scope),
                cached_decoded_box_occurrences=sum(r["decoded_boxes"] for r in scope),
                reported_selector_applications=sum(r["selector_applications"] for r in scope),
                fit_reference_decoded_box_occurrences=sum(r["decoded_boxes"] for r in scope if r["phase"] == "fit_reference"),
                source_search_calls_already_paid=p64["formal_search_calls"], focus_cases=len(focus_cases), focus_rows=len(focus),
                new_network_calls=0, new_image_reads=0, new_full_trajectories=0, new_models=0,
                training_performed=False, old_fit_coefficients_reconstructed=True, offline_development_gt_loaded=True,
                original_checking_labels_loaded=False, counterfactual_continuations_performed=False, novel_method_established=False,
                table_sha256={"stage65_"+n+".csv":file_hash(out/("stage65_"+n+".csv")) for n in tables},
                warnings=["Consumed development states; no method adoption gate or new independent holdout.",
                    "All four rules share each row's actual stored history and boxes; shadow choices never advance a state.",
                    "Same-state IoU or regret differences are not full-trajectory AUC or unobserved future recovery gains.",
                    "Fit-reference metrics are in-sample; held-out own-history metrics and E transfer remain separate.",
                    "E quality/q0 here are shadow choices on stored E states, not newly evaluated E full policies.",
                    "GT defines offline availability, cohorts and focus events only; score computation does not receive GT.",
                    "Seeds average within supported videos, then videos equally; undefined conditional rates remain blank with support.",
                    "Fit 1/99-percentile support and utility ties are descriptive; no new threshold, hyperparameter or online filter is selected.",
                    "Public four-box unavailability is not a test of dense-grid availability or all possible crops."],
                next_step="Use common-state wrong overrides, missed rescue and candidate support to fix a new fit-only protection design"))
            publish_identical(out/"stage65_diagnostic.json", report)
    print("Stage-65: "+report["status"]+"; no training, image reads or GPU inference")
    print("Reports: "+str(out)); return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=nonempty_path); p.add_argument("--output_dir", required=True, type=nonempty_path)
    args = p.parse_args(); audit(args.input, args.output_dir)
