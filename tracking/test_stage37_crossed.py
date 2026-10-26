"""Channel isolation, actual CPU rollouts, GT isolation and frozen resume tests."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import seal
from stage37_crossed_common import ARMS, DESIGN, SCHEMA, channels, compose, methods, prepare, verify_bindings
import stage37_crossed_common as common
import stage37_crossed_experiment as experiment
import stage37_crossed_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_four_corners_take_only_the_prespecified_channels_without_mutation(self):
        outputs = [dict(score_map=np.full((1, 1, 4, 4), .1 + i * .2),
            size_map=np.full((1, 2, 4, 4), .2 + i * .2), offset_map=np.full((1, 2, 4, 4), .3 + i * .2)) for i in range(2)]
        original = copy.deepcopy(outputs)
        for arm in ARMS:
            result = compose(outputs, arm)
            r, g = channels(arm)
            self.assertIs(result["score_map"], outputs[r]["score_map"])
            self.assertIs(result["size_map"], outputs[g]["size_map"])
            self.assertIs(result["offset_map"], outputs[g]["offset_map"])
        for a, b in zip(outputs, original):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_video_pairing_does_not_count_seeds_as_independent_videos(self):
        rows = []
        for video in range(8):
            for seed in DESIGN["seeds"]:
                for arm in ARMS:
                    gain = dict(student=0., teacher_response=2. if seed == 2110 else 0., teacher_geometry=.5, sgla=2.)[arm]
                    rows.append(dict(sequence=str(video), seed=seed, arm=arm, auc=60 + gain,
                        low_iou_fraction=.1, failure_events=0, student_reliable_auc=70 + gain, sgla_reliable_auc=70 + gain))
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            policies, videos, pairs, stability, influence = summary.analyze(rows)
        primary = next(p for p in pairs if p["metric"] == "auc" and p["contrast"] == "teacher_response_minus_student")
        self.assertEqual(primary["videos"], 8)
        self.assertEqual(primary["mean_delta"], 1.)
        self.assertEqual(primary["ci_low"], 1.)
        self.assertEqual(primary["seed_mean_deltas"], "2.00000000|0.00000000")
        self.assertEqual(len(videos), 32)
        self.assertEqual(len(influence), 8)
        self.assertTrue(all(p["videos"] == 8 for p in policies))
        self.assertFalse(any(r["introduced_failure"] for r in stability))
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            summary.analyze(rows + [rows[0]])


def source_fixture(root, torch):
    """Real eight-model OOF checkpoints; replace only the remote provenance adapter."""
    from stage20_compression_common import state_digest
    from stage20_student_train import atomic_save
    from stage35_preserve_common import DESIGN as training, SCHEMA as old_schema
    from stage35_preserve_experiment import collect_trace as original_trace
    from stage35_preserve_runtime import make_student, trainable_state
    from test_stage36_same_crop import canonical_runtime
    torch.manual_seed(2116)
    rt = canonical_runtime(torch)
    dataset, old_root = root / "dataset", root / "stage35"
    entries = []
    for video in range(8):
        name = "uav{:07d}_00000_s".format(video + 1)
        frames = []
        for index in range(13):
            path = dataset / "sequences" / name / ("img{:07d}.jpg".format(index + 1))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([index + video + 1]))
            frames.append(dict(file=path.relative_to(dataset).as_posix(), sha256=file_hash(path), height=64, width=64))
        annotation = dataset / "annotations" / (name + ".txt")
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text("\n".join(["20,20,12,12"] * len(frames)), encoding="utf-8")
        entries.append(dict(sequence=name, source_group_proxy=name.split("_")[0], initial_box=[20., 20., 12., 12.],
            frames=frames, annotation=annotation.relative_to(dataset).as_posix(), annotation_sha256=file_hash(annotation)))
    names = [e["sequence"] for e in entries]
    folds = [dict(fold=f, held_out=names[2 * f:2 * f + 2], fit=names[:2 * f] + names[2 * f + 2:]) for f in range(4)]
    source = seal(dict(schema=old_schema, fit=entries, folds=folds, root=str(dataset), runtime=rt.identity,
        search_factor=4, search_size=256))
    write_json(old_root / "manifest.json", source)
    for seed in training["seeds"]:
        records = []
        for fold in folds:
            net = make_student(rt)
            initial = state_digest(net)
            with torch.no_grad():
                net.backbone.blocks[4].ffn[0].weight[0, 0] += .02 * (fold["fold"] + 1) * (seed - 2109)
            meta = dict(plan_fingerprint=source["fingerprint"], mode="full", arm="supervised", seed=seed,
                fold=fold["fold"], fit_videos=fold["fit"], held_out_videos=fold["held_out"],
                teacher_sha256=state_digest(rt.net), initial_student_sha256=initial)
            path = old_root / "training" / ("seed{}".format(seed)) / ("fold{}.pt".format(fold["fold"]))
            atomic_save(torch, dict(step=training["updates"], metadata=meta, parameters=trainable_state(net)), path)
            records.append(dict(fold=fold["fold"], arm="supervised", checkpoint=str(path), checkpoint_sha256=file_hash(path)))
        write_json(old_root / "training" / ("seed{}".format(seed)) / "stage35_training_decision.json",
            seal(dict(schema=old_schema, teacher_sha256=state_digest(rt.net), arms=records)))
    models = experiment.load_models(rt, source, old_root)
    def reader(unused_root, frame):
        value = int(Path(frame["file"]).stem[3:])
        image = np.zeros((64, 64, 3), np.uint8)
        image[:, :, 0] = (value * 11) % 255
        image[:, :, 1] = np.arange(64, dtype=np.uint8)[None]
        image[:, :, 2] = np.arange(64, dtype=np.uint8)[:, None]
        return image
    for entry in entries:
        fold = common.fold_for(source, entry)
        for method in ["sgla"] + ["supervised_seed" + str(s) for s in training["seeds"]]:
            data = original_trace(rt, source, entry, method, models.get((fold, method)), reader)
            write_cache(old_root / "oof" / (entry["sequence"] + "__" + method + ".npz"), data, {})
    stage36 = root / "stage36"
    stage36.mkdir()
    (stage36 / "source.txt").write_text("synthetic provenance boundary", encoding="utf-8")
    bindings = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    def adapter(unused):
        verify_bindings(bindings)
        return source, str(old_root), "synthetic-source-onset-fingerprint", dict(bindings)
    return stage36, rt, reader, adapter, bindings


@unittest.skipUnless(TORCH, "PyTorch needed for actual frozen-model rollout tests")
class RolloutTests(unittest.TestCase):
    def test_real_cpu_smoke_full_resume_parity_gt_isolation_and_tamper_rejection(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(DESIGN, smoke_frames=5, bootstrap_samples=64):
            root = Path(temporary)
            prior, rt, reader, adapter, bindings = source_fixture(root, torch)
            output = root / "stage37"
            with mock.patch.object(common, "source_info", side_effect=adapter):
                plan = prepare(prior, output)
                self.assertEqual(plan["methods"], methods())
                with self.assertRaises(FileNotFoundError):
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                experiment.collect(rt, plan, output, "smoke", image_reader=reader)
                experiment.smoke_gate(output, plan)
                experiment.collect(rt, plan, output, "full", plan["fit"][0]["sequence"], reader)
                with self.assertRaisesRegex(ValueError, "all 56"):
                    summary.summarize(output)
                # Change only future labels; the mixed rollout must remain identical.
                models = experiment.load_models(rt, dict(plan, fingerprint=plan["source_plan_fingerprint"]), plan["source_input"])
                entry = plan["fit"][0]
                net = models[(common.fold_for(plan, entry), "supervised_seed2110")]
                method = "teacher_response_seed2110"
                before = experiment.collect_trace(rt, plan, entry, method, net, image_reader=reader)
                annotation = Path(plan["root"]) / "annotations/changed_future.txt"
                annotation.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12), encoding="utf-8")
                changed = dict(entry, annotation="annotations/changed_future.txt", annotation_sha256=file_hash(annotation))
                old_read = experiment.read_boxes
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    def after_forward(path):
                        self.assertEqual(spy.call_count, 12)
                        return old_read(path)
                    with mock.patch.object(experiment, "read_boxes", side_effect=after_forward):
                        after = experiment.collect_trace(rt, plan, changed, method, net, image_reader=reader)
                np.testing.assert_array_equal(before["boxes"], after["boxes"])
                np.testing.assert_array_equal(before["previous"], after["previous"])
                self.assertFalse(np.array_equal(before["gt"], after["gt"]))
                bad = dict(before, previous=before["previous"].copy())
                bad["previous"][1, 0] += 1
                with self.assertRaises(AssertionError):
                    common.audit_trace(bad, plan, entry, method)
                experiment.collect(rt, plan, output, "full", image_reader=reader)
                report = summary.summarize(output)
                self.assertEqual(report["status"], "COMPLETE_FROZEN_CROSSED_DIAGNOSTIC")
                self.assertEqual(report["videos"], 8)
                self.assertEqual(report["unique_full_trajectories"], 56)
                self.assertEqual(report["primary"]["videos"], 8)
                self.assertEqual(len(report["native_source_parity"]), 24)
                self.assertTrue(all(r["max_box_error"] == 0 for r in report["native_source_parity"]))
                self.assertIsNone(report["adoption_gate"])
                self.assertFalse(report["training_performed"])
                verify_bindings(bindings)
                self.assertEqual(report, summary.summarize(output))
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                    self.assertEqual(spy.call_count, 0)
                # A write interrupted between the NPZ and receipt can be audited and committed on resume.
                path = output / "full/traces" / (entry["sequence"] + "__" + method + ".npz")
                path.with_suffix(".json").unlink()
                experiment.status(output, plan)  # An uncommitted file is not a completed trace.
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                self.assertEqual(report, summary.summarize(output))
                old_bytes = path.read_bytes()
                path.write_bytes(old_bytes + b"tamper")
                with self.assertRaisesRegex(ValueError, "trace changed"):
                    summary.summarize(output)
                path.write_bytes(old_bytes)
                checkpoint = Path(plan["source_input"]) / "training/seed2111/fold3.pt"
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    common.load_plan(output)


if __name__ == "__main__":
    unittest.main()
