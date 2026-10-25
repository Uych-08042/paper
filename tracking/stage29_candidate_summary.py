"""Report post-intervention persistence; current-frame GT correction is excluded."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import nonempty_path, seal, success_auc
from stage27_uav_summary import write_table
from stage29_candidate_common import (DESIGN, SCHEMA, event_filename, load_event, load_inputs,
                                      load_sealed, validate_stage29_plan)


def first_run(values, threshold, length, below=False, first=0):
    good = np.asarray(values) < threshold if below else np.asarray(values) >= threshold
    for index in range(first, len(good) - length + 1):
        if good[index:index + length].all():
            return index
    return None


def post_metrics(iou, coverage):
    # The injected onset is index 0. It contributes to none of these outcomes.
    future, cov = np.asarray(iou)[1:], np.asarray(coverage)[1:]
    recovery = first_run(future, DESIGN["recovery_iou"], DESIGN["recovery_run"])
    relapse = None if recovery is None else first_run(future, DESIGN["failure_iou"], 3,
                                                    below=True, first=recovery + DESIGN["recovery_run"])
    return dict(post40_mean_iou=float(future.mean()), post20_mean_iou=float(future[:20].mean()),
        post21to40_mean_iou=float(future[20:].mean()), post40_auc=float(success_auc(future)),
        post_low_iou_frames=int((future < DESIGN["failure_iou"]).sum()),
        post_mean_coverage=float(cov.mean()), post_outside_frames=int((cov < .1).sum()),
        post_low_coverage_frames=int((cov < DESIGN["inside_coverage"]).sum()),
        post_recovery_offset=None if recovery is None else recovery + 1,
        post_relapse_offset=None if relapse is None else relapse + 1)


def event_pass(eligible, gain, low_frame_reduction):
    return bool(eligible and gain is not None and low_frame_reduction is not None
                and gain >= DESIGN["minimum_post_iou_gain"] - 1e-12 and low_frame_reduction > 0)


def summarize_event(event, data, trace):
    outcomes = [post_metrics(iou, cov) for iou, cov in zip(data["iou"], data["coverage"])]
    native = outcomes[0]
    rows = []
    for candidate, outcome in zip(event["candidates"], outcomes):
        rows.append(dict(sequence=event["sequence"], source_group_proxy=event["source_group_proxy"],
            event=event["event"], onset_frame=event["onset_frame"], primary_eligible=event["primary_eligible"],
            candidate_rank=candidate["rank"], grid_index=candidate["grid_index"],
            onset_gt_iou=candidate["onset_gt_iou"], onset_weighted_score=candidate["score"], **outcome,
            post40_gain_vs_native=outcome["post40_mean_iou"] - native["post40_mean_iou"],
            post_low_iou_frame_reduction=native["post_low_iou_frames"] - outcome["post_low_iou_frames"]))
    rank = event["onset_gt_best_rank"]
    chosen = None if rank is None else outcomes[rank - 1]
    gain = None if chosen is None else chosen["post40_mean_iou"] - native["post40_mean_iou"]
    reduction = None if chosen is None else native["post_low_iou_frames"] - chosen["post_low_iou_frames"]
    random_mean = float(np.mean([r["post40_mean_iou"] for r in outcomes]))
    frame = event["onset_frame"]
    replay_error = float(np.max(np.abs(data["boxes"][0] - trace["boxes"][frame - 1:frame + DESIGN["horizon"]])))
    event_row = dict(sequence=event["sequence"], source_group_proxy=event["source_group_proxy"],
        event=event["event"], onset_frame=frame, candidates=len(outcomes), primary_eligible=event["primary_eligible"],
        onset_gt_best_rank=rank, native_post40_mean_iou=native["post40_mean_iou"],
        onset_gt_selected_post40_mean_iou=None if chosen is None else chosen["post40_mean_iou"],
        selected_post40_gain_vs_native=gain, native_post_low_iou_frames=native["post_low_iou_frames"],
        selected_post_low_iou_frames=None if chosen is None else chosen["post_low_iou_frames"],
        post_low_iou_frame_reduction=reduction,
        selected_post21to40_mean_iou=None if chosen is None else chosen["post21to40_mean_iou"],
        selected_post_recovery_offset=None if chosen is None else chosen["post_recovery_offset"],
        selected_post_relapse_offset=None if chosen is None else chosen["post_relapse_offset"],
        uniform_candidate_expected_post40_mean_iou=random_mean,
        uniform_non_native_expected_post40_mean_iou=None if len(outcomes) == 1 else
            float(np.mean([r["post40_mean_iou"] for r in outcomes[1:]])),
        selected_minus_uniform_candidate=None if chosen is None else chosen["post40_mean_iou"] - random_mean,
        native_replay_max_box_error=replay_error,
        event_gate_pass=event_pass(event["primary_eligible"], gain, reduction))
    return rows, event_row


def make_decision(plan, rows, missing, run, hashes):
    primary = [r for r in rows if r["primary_eligible"]]
    success = sum(event_pass(True, r["selected_post40_gain_vs_native"], r["post_low_iou_frame_reduction"]) for r in primary)
    complete = not missing
    passed = complete and len(primary) == DESIGN["primary_expected_events"] and success >= DESIGN["minimum_success_events"]
    status = ("PILOT_INCOMPLETE" if not complete else "PASS_CANDIDATE_PERSISTENCE_PILOT" if passed
              else "FAIL_CANDIDATE_PERSISTENCE_PILOT")
    return seal(dict(schema=SCHEMA, role="development_only_spatial_candidate_intervention_mechanism",
        status=status, plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
        source_diagnostic_fingerprint=plan["source_diagnostic_fingerprint"],
        audited_events=len(rows), expected_events=len(plan["events"]), missing_events=missing,
        primary_events_audited=len(primary), expected_primary_events=DESIGN["primary_expected_events"],
        primary_success_events=int(success), persistence_gate_pass=bool(passed),
        primary_rule="At least 2 of the 3 preselected correctable events: post t+1..t+40 mean IoU gain>=0.05 AND fewer IoU<0.2 frames",
        primary_event_results=[dict(sequence=r["sequence"], event=r["event"],
            onset_gt_best_rank=r["onset_gt_best_rank"], gain=r["selected_post40_gain_vs_native"],
            low_iou_frame_reduction=r["post_low_iou_frame_reduction"], pass_event=r["event_gate_pass"]) for r in primary],
        max_native_replay_box_error=max((r["native_replay_max_box_error"] for r in rows), default=None),
        cache_sha256=hashes, analysis_code_sha256=file_hash(Path(__file__)),
        training_performed=False, learned_policy_evaluated=False, statistical_significance_claim=False,
        warnings=["Onsets and the three-event subgroup were chosen using consumed development outcomes.",
            "GT picks an onset candidate offline; it never resets the continuation. This is not a deployable policy.",
            "The primary candidate is fixed by onset IoU, never the best future branch.",
            "Forty continuation frames are not forty independent videos; there are no multi-seed replicas here.",
            "No full-video AUC, external validation, compression or speed improvement is established."],
        next_step=("Complete the identical frozen run before any decision" if not complete else
            "Develop a GT-free lightweight candidate-identity rule, then test it in complete closed-loop videos with matched controls"
            if passed else "Inspect relapse and candidate coverage; this one-output correction did not clear the persistence pilot")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    plan, run = load_sealed(directory / "plan.json"), load_sealed(directory / "run.json")
    inputs = load_inputs(plan["source_input"])
    validate_stage29_plan(plan, inputs)
    if run.get("role") != "one_output_intervention_native_continuation" or run.get("plan") != plan:
        raise ValueError("Run does not match the frozen Stage-29 plan")
    _, source, traces, _ = inputs["source_info"]
    expected = {event_filename(e) for e in plan["events"]}
    extras = {p.name for p in (directory / "branches").glob("*.npz")} - expected
    if extras:
        raise ValueError("Unexpected branch files: " + str(extras))
    branch_rows, step_rows, events, missing, hashes = [], [], [], [], {}
    for event in plan["events"]:
        path = directory / "branches" / event_filename(event)
        if not path.is_file():
            missing.append(dict(sequence=event["sequence"], event=event["event"]))
            continue
        entry, trace = traces[event["sequence"]]
        data = load_event(path, event, entry, trace, source, run)
        rows, event_row = summarize_event(event, data, trace)
        branch_rows.extend(rows)
        events.append(event_row)
        hashes[path.name] = file_hash(path)
        for branch, candidate in enumerate(event["candidates"]):
            for offset in range(DESIGN["horizon"] + 1):
                step_rows.append(dict(sequence=event["sequence"], event=event["event"],
                    candidate_rank=candidate["rank"], offset=offset, included_in_primary_endpoint=offset > 0,
                    iou=float(data["iou"][branch, offset]), native_iou=float(data["iou"][0, offset]),
                    coverage=float(data["coverage"][branch, offset]), native_coverage=float(data["coverage"][0, offset]),
                    parent_to_gt_side_ratio=float(np.sqrt(data["previous"][branch, offset, 2:].prod()
                                                          / data["gt"][offset, 2:].prod())),
                    executed_layer=int(data["layer"][branch, offset])))
    if not events or (missing and not args.allow_incomplete):
        raise ValueError("{} events collected, {} missing; engineering pilot only: --allow_incomplete".format(len(events), len(missing)))
    output = directory / ("analysis_pilot" if args.allow_incomplete else "analysis")
    write_table(output / "stage29_branch_summary.csv", branch_rows)
    write_table(output / "stage29_step_effects.csv", step_rows)
    write_table(output / "stage29_event_summary.csv", events)
    primary = [r for r in events if r["primary_eligible"]]
    write_table(output / "stage29_primary_events.csv", primary, list(events[0]))
    decision = make_decision(plan, events, missing, run, hashes)
    write_json(output / "stage29_decision.json", decision)
    print("Audited {} / {} events; current-frame correction excluded from H40 outcomes.".format(len(events), len(plan["events"])))
    print("sequence                primary  rank  native_IoU  corrected_IoU     gain  low_frames_saved  pass")
    for r in events:
        corrected, gain = r["onset_gt_selected_post40_mean_iou"], r["selected_post40_gain_vs_native"]
        print("{:<24s} {:>6s} {:>5s} {:>10.4f} {:>14s} {:>8s} {:>17s} {:>5s}".format(
            r["sequence"], str(r["primary_eligible"]), str(r["onset_gt_best_rank"] or "n/a"),
            r["native_post40_mean_iou"], "n/a" if corrected is None else "{:.4f}".format(corrected),
            "n/a" if gain is None else "{:+.4f}".format(gain),
            str(r["post_low_iou_frame_reduction"] if gain is not None else "n/a"), str(r["event_gate_pass"])))
    print("Primary successes: {} / {}; decision: {}".format(decision["primary_success_events"],
        DESIGN["primary_expected_events"], decision["status"]))
    print("Reports: " + str(output))
    print("GT-assisted mechanism pilot only; no trained identity policy or speed result.")


if __name__ == "__main__":
    main()
