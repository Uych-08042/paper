"""Read-only audit of cached Stage-23B greedy actions and tracking trajectories."""

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage18_environment import TokenCredit
from stage23_branch_credit import ARMS
from stage23_branch_ppo import single_writer
from stage23_development import CONTROLS, METRIC, SCHEMA, SEEDS, STEPS, read_sealed, sealed
from stage23_development_summary import load_complete


AUDIT_SCHEMA = "stage24_deployment_action_audit_v1"
BOX_TOLERANCE = 1e-6
IOU_TOLERANCE = 1e-12
SAME_REPRESENTATION = ("memory_fast", "l7", "l8", "l9", "l10", "l11", "l12", "random")


def action_summary(rows):
    """Summarize executed actions, not policy probabilities or stochastic entropy."""
    if not rows:
        raise ValueError("Empty deployment trajectory list")
    counts = np.zeros(7, dtype=np.int64)
    opportunities = switches = transitions = multi = single = fast_only = cap = 0
    for row in rows:
        actions = np.asarray(row["actions"])
        if actions.ndim != 1 or not len(actions) or actions.dtype.kind not in "iu" or not np.isin(actions, range(7)).all():
            raise ValueError("Expected nonempty executed Fast/L7-L12 action indices")
        budget = TokenCredit()
        for action in actions:
            budget.advance()
            opportunities += int(budget.credit >= 10)
            budget.spend(int(action))
        counts += np.bincount(actions, minlength=7)
        candidate = actions[actions > 0]
        # Never count a video reset or Fast-to-candidate alternation as a layer switch.
        switches += int(np.sum(candidate[1:] != candidate[:-1]))
        transitions += max(0, len(candidate) - 1)
        active = len(np.unique(candidate))
        multi += int(active > 1)
        single += int(active == 1)
        fast_only += int(active == 0)
        cap += 3 * len(actions) // 10
    calls, frames = int(counts[1:].sum()), int(counts.sum())
    p = counts[1:][counts[1:] > 0] / calls if calls else np.array([])
    return dict(videos=len(rows), tracked_frames=frames, candidate_blocks=calls,
                candidate_ratio=calls / frames, active_layers=int((counts[1:] > 0).sum()),
                fast_only_videos=fast_only, single_layer_videos=single, within_video_multi_layer_videos=multi,
                candidate_layer_switches=switches, within_video_candidate_transitions=transitions,
                candidate_layer_switch_ratio=switches / transitions if transitions else None,
                executed_candidate_layer_entropy_bits=float(-(p * np.log2(p)).sum()) if calls else None,
                legal_candidate_opportunities=opportunities,
                opportunity_execution_ratio=calls / opportunities if opportunities else None,
                prefix_cap_blocks=cap, prefix_cap_utilization=calls / cap if cap else None,
                action_distribution="|".join("{}:{}".format(n, int(counts[i])) for i, n in enumerate(
                    ("fast", "l7", "l8", "l9", "l10", "l11", "l12"))),
                **{"l{}_calls".format(i + 6): int(counts[i]) for i in range(1, 7)})


def compare_trajectories(left, right, same_representation=True):
    """Exact actions, exact floats and tolerance-based output equality stay separate."""
    if not left or len(left) != len(right):
        raise ValueError("Unaligned video lists")
    frames = action_equal = call_equal = joint = layer_equal = 0
    exact_action_videos = exact_output_videos = inconsistent = 0
    box_max = iou_max = 0.
    boxes_exact = True
    for a, b in zip(left, right):
        x, y = np.asarray(a["actions"]), np.asarray(b["actions"])
        valid = np.asarray(a["valid"], bool)
        if (a["sequence"] != b["sequence"] or x.ndim != 1 or not len(x) or y.shape != x.shape
                or valid.shape != x.shape or a["valid"] != b["valid"]):
            raise ValueError("Video/frame/visibility alignment mismatch")
        ax, bx = np.asarray(a["boxes"], float), np.asarray(b["boxes"], float)
        ai, bi = np.asarray(a["overlaps"], float), np.asarray(b["overlaps"], float)
        if (ax.shape != (len(x), 4) or bx.shape != ax.shape or ai.shape != x.shape or bi.shape != x.shape
                or not np.isfinite(ax).all() or not np.isfinite(bx).all()
                or not np.isfinite(ai).all() or not np.isfinite(bi).all()):
            raise ValueError("Invalid comparison arrays")
        bbox_diff = float(np.abs(ax - bx).max())
        iou_diff = float(np.abs(ai[valid] - bi[valid]).max()) if valid.any() else 0.
        output_match = bbox_diff <= BOX_TOLERANCE and iou_diff <= IOU_TOLERANCE
        frames += len(x)
        box_max, iou_max = max(box_max, bbox_diff), max(iou_max, iou_diff)
        boxes_exact = boxes_exact and np.array_equal(ax, bx)
        exact_output_videos += int(output_match)
        if same_representation:
            if not np.isin(x, range(7)).all() or not np.isin(y, range(7)).all():
                raise ValueError("Native selector sentinel is not a known candidate layer")
            equal = x == y
            both = (x > 0) & (y > 0)
            action_equal += int(equal.sum())
            call_equal += int(((x > 0) == (y > 0)).sum())
            joint += int(both.sum())
            layer_equal += int((equal & both).sum())
            exact_action_videos += int(equal.all())
            inconsistent += int(equal.all() and not output_match)
    action_match = action_equal == frames if same_representation else None
    outputs_match = box_max <= BOX_TOLERANCE and iou_max <= IOU_TOLERANCE
    return dict(videos=len(left), tracked_frames=frames, same_tracking_representation=same_representation,
                action_agreement=action_equal / frames if same_representation else None,
                exact_action_match=action_match,
                action_mismatch_frames=frames - action_equal if same_representation else None,
                call_mask_agreement=call_equal / frames if same_representation else None,
                joint_candidate_frames=joint if same_representation else None,
                candidate_layer_agreement_at_joint_calls=layer_equal / joint if joint else None,
                exact_action_match_videos=exact_action_videos if same_representation else None,
                exact_box_match=bool(boxes_exact), max_box_abs_diff=box_max, max_valid_iou_abs_diff=iou_max,
                output_match_within_tolerance=outputs_match, output_match_videos=exact_output_videos,
                same_actions_output_mismatch_videos=inconsistent if same_representation else None,
                trajectory_equivalent=bool(action_match and outputs_match) if same_representation else None,
                auc_delta=float(np.mean([a["auc"] - b["auc"] for a, b in zip(left, right)])))


def classify(summary, comparisons):
    if not summary["candidate_blocks"]:
        return ("MEMORY_FAST_EQUIVALENT" if comparisons["memory_fast"]["trajectory_equivalent"]
                else "FAST_ONLY_OUTPUT_MISMATCH")
    for name in SAME_REPRESENTATION[1:7]:
        if comparisons[name]["trajectory_equivalent"]:
            return "STATIC_{}_P30_EQUIVALENT".format(name.upper())
    if any(comparisons[name]["exact_action_match"] for name in SAME_REPRESENTATION[:7]):
        return "STATIC_ACTIONS_OUTPUT_MISMATCH"
    if summary["within_video_multi_layer_videos"]:
        return "WITHIN_VIDEO_MULTI_LAYER"
    if summary["active_layers"] > 1:
        return "CROSS_VIDEO_SINGLE_LAYER"
    return "SINGLE_LAYER_NONSTATIC_SCHEDULE"


def make_reports(trajectories, metadata):
    policies, video_rows, controls, video_controls, seed_pairs = [], [], [], [], []
    for arm in ARMS:
        for index, seed in enumerate(SEEDS):
            rows = trajectories[arm][index]
            comparisons = {}
            for control in CONTROLS:
                reference = trajectories[control][0]
                same = control in SAME_REPRESENTATION
                comparison = compare_trajectories(rows, reference, same)
                comparisons[control] = comparison
                controls.append(dict(seed=seed, method=arm, control=control, **comparison))
                for row, ref in zip(rows, reference):
                    video_controls.append(dict(seed=seed, method=arm, control=control, sequence=row["sequence"],
                        **compare_trajectories([row], [ref], same)))
            summary = action_summary(rows)
            policy = dict(seed=seed, method=arm, classification=classify(summary, comparisons),
                          auc=float(np.mean([r["auc"] for r in rows])), **summary,
                          static_l9_action_agreement=comparisons["l9"]["action_agreement"],
                          trained_model_sha256=metadata[(seed, arm)]["final_model_sha256"])
            policies.append(policy)
            for video_index, row in enumerate(rows):
                per_video = action_summary([row])
                refs = {name: compare_trajectories([row], [trajectories[name][0][video_index]],
                        name in SAME_REPRESENTATION) for name in CONTROLS}
                video_rows.append(dict(seed=seed, method=arm, sequence=row["sequence"], auc=row["auc"],
                    classification=classify(per_video, refs), **per_video))
        for a, b in combinations(range(len(SEEDS)), 2):
            left, right = metadata[(SEEDS[a], arm)], metadata[(SEEDS[b], arm)]
            seed_pairs.append(dict(method=arm, left_seed=SEEDS[a], right_seed=SEEDS[b],
                same_initial_model_weights=left["initial_model_sha256"] == right["initial_model_sha256"],
                same_trained_model_weights=left["final_model_sha256"] == right["final_model_sha256"],
                same_checkpoint_bytes=left["checkpoint_sha256"] == right["checkpoint_sha256"],
                **compare_trajectories(trajectories[arm][a], trajectories[arm][b])))
    return dict(policy_actions=policies, video_actions=video_rows, control_agreement=controls,
                video_control_agreement=video_controls, seed_agreement=seed_pairs)


def verify_protocol(directory):
    contract = read_sealed(directory / "stage23b_contract.json")
    decision = read_sealed(directory / "analysis" / "stage23b_decision.json")
    if (contract.get("schema") != SCHEMA or contract.get("seeds") != list(SEEDS)
            or contract.get("arms") != list(ARMS) or contract.get("live_steps") != STEPS
            or contract.get("metric") != METRIC or decision.get("schema") != SCHEMA
            or decision.get("contract_fingerprint") != contract["fingerprint"]
            or decision.get("status") not in ("NO_CREDIT_EFFECT_ESTABLISHED", "CREDIT_EFFECT_ONLY_NOT_COMPETITIVE",
                                             "DEVELOPMENT_POLICY_GAIN_SUPPORTED")):
        raise ValueError("Require the completed frozen Stage-23B protocol and its original decision")
    for field in ("fit", "tune"):
        names = contract.get(field)
        if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n
                or n in (".", "..") or "/" in n or "\\" in n for n in names) or len(set(names)) != len(names)):
            raise ValueError("Invalid frozen video identities")
    if set(contract["fit"]) & set(contract["tune"]):
        raise ValueError("Frozen split overlap")
    code = dict(contract["source_code"], **contract["code_sha256"])
    if not {"stage23_development.py", "stage23_development_summary.py"} <= set(code):
        raise ValueError("Missing original development source hashes")
    for name, digest in code.items():
        if Path(name).name != name or "\\" in name or file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Original frozen source changed: " + name)
    return contract, decision, code


def evidence_paths(directory, contract):
    paths = [directory / "stage23b_contract.json", directory / "analysis" / "stage23b_decision.json",
             directory / "controls_complete.json"]
    for name in CONTROLS:
        paths.extend(directory / "controls" / name / (video + ".json") for video in contract["tune"])
    for seed in SEEDS:
        for arm in ARMS:
            base = directory / ("seed" + str(seed)) / arm
            paths.extend(base / name for name in ("complete.json", "last.pt", "history.csv"))
            paths.extend(base / "evaluation" / (video + ".json") for video in contract["tune"])
    return paths


def run_audit(input_dir, output_dir):
    directory, output = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if directory == output or output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("Audit output must be separate from, and not an ancestor of, frozen input")
    if (directory / ".stage23_writer.lock").exists():
        raise RuntimeError("Source training writer lock exists; wait for it to exit before auditing")
    contract, source_decision, code = verify_protocol(directory)
    paths = evidence_paths(directory, contract)
    before = {p.relative_to(directory).as_posix(): file_hash(p) for p in paths}
    trajectories, _ = load_complete(directory, contract)
    metadata = {(seed, arm): read_sealed(directory / ("seed" + str(seed)) / arm / "complete.json")
                for seed in SEEDS for arm in ARMS}
    reports = make_reports(trajectories, metadata)
    audit_contract = sealed(dict(schema=AUDIT_SCHEMA, role="posthoc_descriptive_actions_not_new_policy",
        source_directory=str(directory), source_contract_fingerprint=contract["fingerprint"],
        source_decision_fingerprint=source_decision["fingerprint"], source_status=source_decision["status"],
        source_files_sha256=before, source_code_sha256=code,
        audit_code_sha256=file_hash(Path(__file__)),
        audit_runtime=dict(python=sys.version.split()[0], numpy=np.__version__),
        box_abs_tolerance=BOX_TOLERANCE, valid_iou_abs_tolerance=IOU_TOLERANCE,
        action_matching="exact_integer_indices_on_all_tracked_frames",
        layer_switching="between_successive_candidate_events_within_video_excluding_Fast_and_resets"))
    after = {p.relative_to(directory).as_posix(): file_hash(p) for p in paths}
    if (before != after or any(file_hash(Path(__file__).parent / n) != h for n, h in code.items())
            or file_hash(Path(__file__)) != audit_contract["audit_code_sha256"]):
        raise RuntimeError("Source evidence changed during audit; no reports published")
    policies = reports["policy_actions"]
    h1 = [r for r in policies if r["method"] == "ppo_cf_h1"]
    h20 = [r for r in policies if r["method"] == "ppo_cf_h20"]
    mismatches = [dict(seed=r["seed"], method=r["method"], control=r["control"],
                       videos=r["same_actions_output_mismatch_videos"])
                  for r in reports["control_agreement"] if r["same_actions_output_mismatch_videos"]]
    seed_mismatches = [dict(method=r["method"], left_seed=r["left_seed"], right_seed=r["right_seed"],
                           videos=r["same_actions_output_mismatch_videos"])
                      for r in reports["seed_agreement"] if r["same_actions_output_mismatch_videos"]]
    findings = dict(h1_all_seeds_static_l9_equivalent=all(r["classification"] == "STATIC_L9_P30_EQUIVALENT" for r in h1),
        h1_all_seeds_exact_static_l9_boxes=all(r["exact_box_match"] for r in reports["control_agreement"]
                                             if r["method"] == "ppo_cf_h1" and r["control"] == "l9"),
        h1_all_seed_pairs_same_actions_and_outputs=all(r["trajectory_equivalent"] for r in reports["seed_agreement"]
                                                      if r["method"] == "ppo_cf_h1"),
        h20_memory_fast_equivalent_seeds=[r["seed"] for r in h20 if r["classification"] == "MEMORY_FAST_EQUIVALENT"],
        h20_static_l9_equivalent_seeds=[r["seed"] for r in h20 if r["classification"] == "STATIC_L9_P30_EQUIVALENT"],
        h20_within_video_multi_layer_seeds=[r["seed"] for r in h20 if r["within_video_multi_layer_videos"]],
        same_actions_output_mismatch_pairs=mismatches, same_actions_output_mismatch_seed_pairs=seed_mismatches)
    result = sealed(dict(schema=AUDIT_SCHEMA, audit_contract_fingerprint=audit_contract["fingerprint"],
        status="REQUIRES_OUTPUT_CONSISTENCY_REVIEW" if mismatches or seed_mismatches else "COMPLETE_DESCRIPTIVE_ACTION_AUDIT",
        source_status=source_decision["status"], source_decision_fingerprint=source_decision["fingerprint"],
        source_files_unchanged=True, model_loaded=False, new_predictions_generated=False, new_training=False,
        fit_videos=len(contract["fit"]), tune_videos=len(contract["tune"]), policies=policies, findings=findings,
        warning="Cached greedy deployment only. Executed-layer frequency entropy is not policy entropy. "
                "Multiple layers or output equality do not establish useful adaptation or external generalization. "
                "The original Stage-23B efficacy decision is unchanged."))
    with single_writer(output):
        path = output / "stage24_audit_contract.json"
        if path.exists():
            if read_sealed(path) != audit_contract:
                raise ValueError("Audit inputs/code changed; use a new audit output directory")
        elif any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
            raise ValueError("Unidentified audit outputs; use a new separate directory")
        else:
            write_json(path, audit_contract)
        for name, rows in reports.items():
            write_csv(output / ("stage24_" + name + ".csv"), rows)
        write_json(output / "stage24_action_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if not args.input.strip() or not args.output_dir.strip():
        parser.error("Explicit nonempty input and separate output paths are required")
    result = run_audit(args.input, args.output_dir)
    print("Audited {} final policies on {} frozen tune videos. No model loaded or new predictions.".format(
        len(result["policies"]), result["tune_videos"]))
    print("seed  method                calls% layers multi_videos  class")
    for row in result["policies"]:
        print("{seed}  {method:<19s} {candidate_ratio:>6.2%} {active_layers:>6d} "
              "{within_video_multi_layer_videos:>12d}  {classification}".format(**row))
    print("H1 all seeds equivalent to static L9-P30: " + str(result["findings"]["h1_all_seeds_static_l9_equivalent"]))
    print("H1 all seeds exactly match stored L9 boxes: " + str(result["findings"]["h1_all_seeds_exact_static_l9_boxes"]))
    print("H20 memory-Fast equivalent seeds: " + str(result["findings"]["h20_memory_fast_equivalent_seeds"]))
    print("H20 static L9 equivalent seeds: " + str(result["findings"]["h20_static_l9_equivalent_seeds"]))
    print("Audit status: " + result["status"])
    print("Original Stage-23B status (unchanged): " + result["source_status"])
    print("Reports: " + str(Path(args.output_dir).expanduser().resolve()))


if __name__ == "__main__":
    main()
