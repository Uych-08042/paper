"""CPU-only action/crop-equivalence and immutable-source audit regression tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, tracking_iou, write_csv, write_json
from stage18_environment import TokenCredit
from stage23_branch_credit import ARMS
from stage23_development import CONTROLS, METRIC, SCHEMA, SEEDS, STEPS, read_sealed, sealed, valid_auc
from stage24_deployment_audit import action_summary, classify, compare_trajectories, make_reports, run_audit


def schedule(layer, frames=20):
    budget, actions = TokenCredit(), []
    for _ in range(frames):
        budget.advance()
        action = layer if budget.credit >= 10 else 0
        budget.spend(action)
        actions.append(action)
    return actions


def trajectory(actions, name="a", method="policy", offset=0.):
    actions = list(actions)
    x = np.maximum(actions, 0).cumsum() * .01 + offset
    boxes = np.column_stack((x, np.zeros(len(x)), np.full(len(x), 20.), np.full(len(x), 20.))).tolist()
    overlaps = [float(tracking_iou(box, [0, 0, 20, 20], True)) for box in boxes]
    return dict(sequence=name, method=method, metric=METRIC, frames=len(actions), valid_frames=len(actions),
                actions=actions, boxes=boxes, overlaps=overlaps, valid=[True] * len(actions),
                candidate_blocks=sum(a != 0 for a in actions), auc=valid_auc(overlaps, np.ones(len(actions), bool)),
                mean_crop_model_decode_ms=1.)


def controls(names=("a", "b")):
    output = {}
    for method in CONTROLS:
        actions = ([0] * 20 if method in ("fast", "memory_fast") else [-1] * 20 if method == "sgla"
                   else schedule(int(method[1:]) - 6) if method.startswith("l") else schedule(1))
        output[method] = [[trajectory(actions, name, method, .25 if method == "fast" else 0.) for name in names]]
    return output


def fixture_data():
    rows = controls()
    mixed = schedule(1)
    for event, index in enumerate(np.flatnonzero(mixed)):
        mixed[index] = 1 + event % 2
    for arm in ARMS:
        rows[arm] = []
        for seed in SEEDS:
            actions = [schedule(3), schedule(3)]
            if seed == 2075:
                actions = [[0] * 20, [0] * 20] if arm == "ppo_gae" else actions
            if arm == "ppo_cf_h20" and seed == 2074:
                actions = [[0] * 20, [0] * 20]
            if arm == "ppo_cf_h20" and seed == 2075:
                actions = [mixed, schedule(3)]
            if arm == "ppo_gae" and seed == 2076:
                actions = [schedule(1), schedule(2)]
            rows[arm].append([trajectory(a, n) for a, n in zip(actions, ("a", "b"))])
    return rows


def write_fixture(root, inconsistent_output=False):
    code = {name: file_hash(Path(__file__).parent / name) for name in
            ("stage23_development.py", "stage23_development_summary.py")}
    contract = sealed(dict(schema=SCHEMA, seeds=list(SEEDS), arms=list(ARMS), live_steps=STEPS,
        metric=METRIC, fit=["fit"], tune=["a", "b"], bundle_fingerprint="bundle",
        source_code={"stage18_environment.py": file_hash(Path(__file__).parent / "stage18_environment.py")}, code_sha256=code))
    write_json(root / "stage23b_contract.json", contract)
    write_json(root / "analysis" / "stage23b_decision.json", sealed(dict(schema=SCHEMA,
        contract_fingerprint=contract["fingerprint"], status="NO_CREDIT_EFFECT_ESTABLISHED")))
    data = fixture_data()
    def write_evaluation(directory, videos, policy_id):
        references = []
        for row in videos:
            path = directory / (row["sequence"] + ".json")
            write_json(path, sealed(dict(row, contract_fingerprint=contract["fingerprint"], policy_sha256=policy_id)))
            references.append(dict(sequence=row["sequence"], sha256=file_hash(path)))
        return references
    refs = {name: write_evaluation(root / "controls" / name, data[name][0], "bundle") for name in CONTROLS}
    write_json(root / "controls_complete.json", sealed(dict(contract_fingerprint=contract["fingerprint"], evaluations=refs)))
    for seed_index, seed in enumerate(SEEDS):
        for arm in ARMS:
            base = root / ("seed" + str(seed)) / arm
            write_json(base / "last.pt", dict(opaque_checkpoint_for_byte_hashing_only=True, seed=seed, arm=arm))
            policy_id = file_hash(base / "last.pt")
            videos = data[arm][seed_index]
            if inconsistent_output and seed == 2074 and arm == "ppo_cf_h1":
                videos[0] = trajectory(videos[0]["actions"], videos[0]["sequence"], offset=.001)
            refs = write_evaluation(base / "evaluation", videos, policy_id)
            history = [dict(seed=seed, arm=arm, steps=(i + 1) * 256, anchor_count=4, branch_transitions=1120,
                optimizer_steps=4, valid_transitions=256, likelihood_replay_error=1e-7, probability_replay_error=1e-7,
                chosen_replay_error=0., live_candidate_blocks=70, branch_candidate_blocks=300,
                valid_anchors=4, seconds=1.) for i in range(STEPS // 256)]
            write_csv(base / "history.csv", history)
            write_json(base / "complete.json", sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
                seed=seed, arm=arm, steps=STEPS, initial_model_sha256="initial-" + str(seed),
                final_model_sha256="trained-{}-{}".format(seed, arm), actor_changed=True,
                frozen_representation_unchanged=True, checkpoint_sha256=policy_id,
                history_sha256=file_hash(base / "history.csv"), evaluations=refs)))


class ActionTests(unittest.TestCase):
    def test_budget_alternation_is_not_candidate_layer_switching(self):
        result = action_summary([trajectory(schedule(3))])
        self.assertEqual(result["active_layers"], 1)
        self.assertEqual(result["candidate_layer_switches"], 0)
        self.assertEqual(result["within_video_multi_layer_videos"], 0)
        self.assertEqual(result["executed_candidate_layer_entropy_bits"], 0.)
        self.assertGreater(result["candidate_blocks"], 0)

    def test_resets_and_cross_video_layer_changes_are_not_within_video_switches(self):
        rows = [trajectory(schedule(1), "a"), trajectory(schedule(2), "b")]
        result = action_summary(rows)
        self.assertEqual(result["active_layers"], 2)
        self.assertEqual(result["candidate_layer_switches"], 0)
        self.assertEqual(result["within_video_multi_layer_videos"], 0)
        self.assertAlmostEqual(result["executed_candidate_layer_entropy_bits"], 1.)

    def test_real_candidate_switches_and_fast_only_opportunities(self):
        actions = schedule(1)
        for i, index in enumerate(np.flatnonzero(actions)):
            actions[index] = 1 + i % 2
        result = action_summary([trajectory(actions)])
        self.assertEqual(result["candidate_layer_switches"], result["candidate_blocks"] - 1)
        self.assertEqual(result["within_video_multi_layer_videos"], 1)
        fast = action_summary([trajectory([0] * 20)])
        self.assertEqual(fast["legal_candidate_opportunities"], 17)
        self.assertEqual(fast["fast_only_videos"], 1)
        self.assertIsNone(fast["executed_candidate_layer_entropy_bits"])

    def test_illegal_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            action_summary([trajectory([3] + [0] * 19)])

    def test_equal_auc_does_not_establish_action_or_box_equivalence(self):
        left, right = trajectory(schedule(1)), trajectory(schedule(3))
        self.assertEqual(left["auc"], right["auc"])
        result = compare_trajectories([left], [right])
        self.assertFalse(result["exact_action_match"])
        self.assertFalse(result["trajectory_equivalent"])
        self.assertGreater(result["max_box_abs_diff"], 0.)

    def test_same_layer_but_different_timing_is_not_a_static_control(self):
        delayed = [0] * 20
        for i in np.flatnonzero(schedule(3)):
            if i + 1 < 20:
                delayed[i + 1] = 3
        row = trajectory(delayed)
        refs = {name: compare_trajectories([row], values[0][:1], name not in ("fast", "sgla"))
                for name, values in controls().items()}
        self.assertEqual(classify(action_summary([row]), refs), "SINGLE_LAYER_NONSTATIC_SCHEDULE")

    def test_native_and_different_representation_actions_are_not_equated(self):
        result = compare_trajectories([trajectory([0] * 20)], [trajectory([-1] * 20)], False)
        self.assertIsNone(result["exact_action_match"])
        self.assertIsNone(result["trajectory_equivalent"])
        with self.assertRaises(ValueError):
            compare_trajectories([trajectory([0] * 20)], [trajectory([-1] * 20)])

    def test_action_equality_with_output_difference_requests_review(self):
        left, right = trajectory(schedule(3)), trajectory(schedule(3), offset=.001)
        comparison = compare_trajectories([left], [right])
        self.assertTrue(comparison["exact_action_match"])
        self.assertEqual(comparison["same_actions_output_mismatch_videos"], 1)
        self.assertFalse(comparison["trajectory_equivalent"])

    def test_exact_boxes_and_tolerance_equivalence_are_distinguished(self):
        left, right = trajectory(schedule(3)), trajectory(schedule(3), offset=1e-7)
        comparison = compare_trajectories([left], [right])
        self.assertTrue(comparison["exact_action_match"])
        self.assertFalse(comparison["exact_box_match"])
        self.assertTrue(comparison["output_match_within_tolerance"])
        self.assertTrue(comparison["trajectory_equivalent"])

    def test_misaligned_video_identities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            compare_trajectories([trajectory(schedule(3), "a")], [trajectory(schedule(3), "b")])

    def test_reports_distinguish_collapse_dynamic_layers_and_different_weights(self):
        metadata = {(seed, arm): dict(initial_model_sha256="initial" + str(seed),
            final_model_sha256="trained-{}-{}".format(seed, arm), checkpoint_sha256="file-{}-{}".format(seed, arm))
            for seed in SEEDS for arm in ARMS}
        reports = make_reports(fixture_data(), metadata)
        classifications = {(r["seed"], r["method"]): r["classification"] for r in reports["policy_actions"]}
        self.assertTrue(all(classifications[(s, "ppo_cf_h1")] == "STATIC_L9_P30_EQUIVALENT" for s in SEEDS))
        self.assertEqual(classifications[(2074, "ppo_cf_h20")], "MEMORY_FAST_EQUIVALENT")
        self.assertEqual(classifications[(2075, "ppo_cf_h20")], "WITHIN_VIDEO_MULTI_LAYER")
        self.assertEqual(classifications[(2076, "ppo_gae")], "CROSS_VIDEO_SINGLE_LAYER")
        for pair in reports["seed_agreement"]:
            if pair["method"] == "ppo_cf_h1":
                self.assertTrue(pair["trajectory_equivalent"])
                self.assertFalse(pair["same_trained_model_weights"])


class IntegrationTests(unittest.TestCase):
    def test_complete_read_only_audit_without_torch_and_idempotent_reports(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source, output = Path(temp) / "input", Path(temp) / "audit"
            write_fixture(source)
            before = {p.relative_to(source): file_hash(p) for p in source.rglob("*") if p.is_file()}
            with patch.dict("sys.modules", {"torch": None}):
                result = run_audit(source, output)
                repeated = run_audit(source, output)
            after = {p.relative_to(source): file_hash(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(result, repeated)
            self.assertEqual(result["status"], "COMPLETE_DESCRIPTIVE_ACTION_AUDIT")
            self.assertEqual(result["source_status"], "NO_CREDIT_EFFECT_ESTABLISHED")
            self.assertFalse(result["model_loaded"])
            self.assertTrue(result["findings"]["h1_all_seeds_static_l9_equivalent"])
            self.assertEqual(result["findings"]["h20_memory_fast_equivalent_seeds"], [2074])
            self.assertEqual(len(result["policies"]), 9)
            self.assertEqual(len(list(output.glob("stage24_*.csv"))), 5)

    def test_source_checkpoint_change_is_rejected_without_output(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source, output = Path(temp) / "input", Path(temp) / "audit"
            write_fixture(source)
            write_json(source / "seed2074" / ARMS[0] / "last.pt", {"changed": True})
            with self.assertRaisesRegex(ValueError, "integrity"):
                run_audit(source, output)
            self.assertFalse(output.exists())

    def test_equal_actions_with_inconsistent_outputs_do_not_pass_as_equivalent(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source, output = Path(temp) / "input", Path(temp) / "audit"
            write_fixture(source, inconsistent_output=True)
            result = run_audit(source, output)
            self.assertEqual(result["status"], "REQUIRES_OUTPUT_CONSISTENCY_REVIEW")
            self.assertFalse(result["findings"]["h1_all_seeds_static_l9_equivalent"])
            self.assertTrue(result["findings"]["same_actions_output_mismatch_seed_pairs"])
            self.assertEqual(result["source_status"], "NO_CREDIT_EFFECT_ESTABLISHED")

    def test_unknown_existing_audit_files_are_not_overwritten(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source, output = Path(temp) / "input", Path(temp) / "audit"
            write_fixture(source)
            write_json(output / "keep.json", {"user_data": True})
            original = file_hash(output / "keep.json")
            with self.assertRaisesRegex(ValueError, "Unidentified"):
                run_audit(source, output)
            self.assertEqual(file_hash(output / "keep.json"), original)

    def test_output_inside_or_above_source_and_active_writer_are_rejected(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            source, output = Path(temp) / "input", Path(temp) / "audit"
            write_fixture(source)
            for forbidden in (source, source / "new_audit", source.parent):
                with self.assertRaisesRegex(ValueError, "separate"):
                    run_audit(source, forbidden)
            write_json(source / ".stage23_writer.lock", {"writer": "still active"})
            with self.assertRaisesRegex(RuntimeError, "writer lock"):
                run_audit(source, output)


if __name__ == "__main__":
    unittest.main()
