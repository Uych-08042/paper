"""Test same-candidate permutation semantics and the image-free sealed pipeline."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage111_slot_audit as stage


class SlotAudit(unittest.TestCase):
    def setUp(self):
        rng = np.random.Generator(np.random.PCG64(111))
        self.pack = dict(norm=dict(center=rng.normal(size=256).tolist(), scale=rng.uniform(.5, 2., 256).tolist()), state={})
        for branch, outputs in (("actor", 4), ("critic", 1)):
            for name, value in (("0.weight", rng.normal(0, .07, (32, 256))), ("0.bias", rng.normal(0, .1, 32)),
                                ("2.weight", rng.normal(0, .1, (outputs, 32))), ("2.bias", np.zeros(outputs))):
                self.pack["state"][branch+"."+name] = value.tolist()
        self.design = dict(feature_clip=10.)
        self.history = np.array([[40+i, 30, 12, 20] for i in range(5)], float)
        parents = np.array([[44, 30, 12, 20], [50, 30, 12, 20], [41, 28, 10, 18], [47, 35, 16, 24]], float)
        boxes = np.array([[44, 30, 12, 20], [110, 30, 12, 20], [43, 31, 12, 20], [49, 34, 16, 24]], float)
        self.data = dict(candidate_parents=np.tile(parents, (6, 1, 1)), candidate_boxes=np.tile(boxes, (6, 1, 1)),
                         features=rng.normal(size=(6, 4, 24)), frame_id=np.array(5))
        self.data["features"][:, 0] = self.data["features"][0, 0]

    def constant_actor(self):
        pack = copy.deepcopy(self.pack)
        for name, value in pack["state"].items(): pack["state"][name] = np.zeros_like(value).tolist()
        pack["state"]["actor.2.bias"] = [0., 3., 0., 0.]
        return pack

    def test_identity_exact_actor_parity_and_physical_slot_effect(self):
        result = stage.predict(self.data, self.history, self.pack, self.design)
        for b in range(6):
            for k, order in enumerate(stage.ORDERS):
                f = stage.c74.c66.c64.choice_features(self.data["features"][b, order], self.data["candidate_boxes"][b, order], self.history)
                d = stage.c74.gate(f, self.data["candidate_boxes"][b, order], self.pack, self.design)
                self.assertEqual(result["selected"][b, k], order[np.argmax(d["probabilities"])])
                np.testing.assert_array_equal(result["probabilities"][b, k, order], d["probabilities"])
        biased = stage.predict(self.data, self.history, self.constant_actor(), self.design)
        self.assertEqual(set(biased["selected"][0]), {1, 2, 3})
        ident = dict(cohort="historical", group="uav0000001", sequence="uav0000001_00000_s", seed=1, source_arm="four_matched", frame=5)
        states, rows, _ = stage.labels_and_rows(self.data, biased, self.data["candidate_boxes"][0, 0], ident)
        self.assertTrue(all(r["physical_order_sensitive"] for r in states))
        self.assertAlmostEqual(next(r["iou"] for r in rows if r["bank"] == "fixed" and r["policy"] == "uniform_order_expected"),
                               np.mean(stage.s105.tracking_iou(self.data["candidate_boxes"][0, 1:], self.data["candidate_boxes"][0, 0])))

    def test_symmetrization_equivariant_including_duplicate_outputs_and_ties(self):
        for duplicate in (False, True):
            data = copy.deepcopy(self.data)
            if duplicate: data["candidate_boxes"][:, 2] = data["candidate_boxes"][:, 3]
            for pack in (self.pack, self.constant_actor()):
                original = stage.predict(data, self.history, pack, self.design)
                for order in stage.ORDERS:
                    changed = {k: v.copy() for k, v in data.items()}
                    for name in ("candidate_parents", "candidate_boxes", "features"): changed[name] = changed[name][:, order]
                    result = stage.predict(changed, self.history, pack, self.design)
                    for b in range(6):
                        before = data["candidate_boxes"][b, original["symmetrized_selected"][b]]
                        after = changed["candidate_boxes"][b, result["symmetrized_selected"][b]]
                        np.testing.assert_array_equal(before, after)
                        self.assertAlmostEqual(result["symmetrized_probabilities"][b].sum(), 1.)
        same = copy.deepcopy(self.data); same["candidate_boxes"][:] = same["candidate_boxes"][0, 0]
        result = stage.predict(same, self.history, self.pack, self.design)
        np.testing.assert_array_equal(result["selected"], 0)
        np.testing.assert_array_equal(result["symmetrized_selected"], 0)
        np.testing.assert_array_equal(result["symmetrized_probabilities"], np.tile([1., 0., 0., 0.], (6, 1)))

    def test_canonical_assignment_recovers_fixed_roles_and_gt_independence(self):
        parents, boxes = self.data["candidate_parents"][0], self.data["candidate_boxes"][0]
        for order in stage.ORDERS:
            selected = stage.canonical_order(parents[order], boxes[order], parents, self.history[-1])
            np.testing.assert_array_equal(parents[order][stage.ORDERS[selected]], parents)
        before = stage.predict(self.data, self.history, self.pack, self.design)
        identity = dict(cohort="historical", group="g", sequence="g_s", seed=1, source_arm="four_matched", frame=5)
        for truth in ([44, 30, 12, 20], [1000, 2000, 10, 100], [0, 0, 0, 0]):
            stage.labels_and_rows(self.data, before, truth, identity)
        after = stage.predict(self.data, self.history, self.pack, self.design)
        for key in before: np.testing.assert_array_equal(before[key], after[key])
        trace = dict(boxes=np.tile(self.history[-1], (10, 1)))
        first = stage.e63.history_at(trace, self.history[0], 5, dict(actor_design=dict(history=5)))
        trace["boxes"][4:] = 999
        np.testing.assert_array_equal(first, stage.e63.history_at(trace, self.history[0], 5, dict(actor_design=dict(history=5))))

    def test_norm_clipping_measured_before_clip(self):
        pack = copy.deepcopy(self.pack)
        pack["norm"]["center"] = [1e6]*256
        result = stage.predict(self.data, self.history, pack, self.design)
        np.testing.assert_array_equal(result["clip_fraction"], 1.)
        self.assertGreater(result["maximum_unclipped_z"].min(), self.design["feature_clip"])

    def test_complete_cpu_pipeline_resume_cache_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root, r110, r103 = (Path(directory)/v for v in ("stage111", "stage110", "stage103"))
            for path in (root, r110, r103): path.mkdir()
            unit = dict(sequence="uav0000001_00000_s", seed=2171, origin="native", policy="ppo", fold=0)
            spec = dict(cohort="historical", unit=unit)
            jobs = [dict(spec=spec, source_arm=mode, context=stage.c.key(unit)+"__"+mode,
                         sampling=dict(eval_frames=[5, 7])) for mode in stage.s103.MODES[1:]]
            p110 = stage.io.seal(dict(schema=stage.p.SCHEMA, units=jobs))
            p103 = stage.io.seal(dict(schema=stage.s103.SCHEMA))
            model = stage.io.seal(dict(schema=stage.p.SCHEMA, held_out_group="uav0000001"))
            stage.c.publish_identical(r110/"models/uav0000001.json", model)
            old = dict(design=self.design, actor_design=dict(history=5))
            entry = dict(initial_box=self.history[0], ground_truth=np.tile(self.data["candidate_boxes"][0, 0], (8, 1)))
            traces = {}; supply = []
            for job in jobs:
                trace = dict(boxes=np.tile(self.history[-1], (7, 1)), frame_id=np.arange(1, 8))
                traces[job["source_arm"]] = trace
                stage.c.save(r103/"evaluation"/(job["context"]+".npz"), trace,
                    dict(plan_fingerprint=p103["fingerprint"], unit=unit, arm=job["source_arm"]))
                for frame in job["sampling"]["eval_frames"]:
                    data = dict(self.data, frame_id=np.array(frame))
                    meta = dict(plan_fingerprint=p110["fingerprint"], unit=unit, source_arm=job["source_arm"], frame=frame, model_fingerprint=model["fingerprint"])
                    stem = job["context"]+f"__f{frame:06d}"
                    stage.c.save(r110/"observations"/(stem+".npz"), data, dict(meta, kind="shared_sgla_searches"))
                    stage.c.save(r110/"proposals"/(stem+".npz"), dict(candidate_parents=data["candidate_parents"]), dict(meta, kind="causal_search_proposals"))
                    history = stage.e63.history_at(trace, entry["initial_box"], frame, old)
                    pred = stage.predict(data, history, self.pack, self.design)
                    rows, _, _ = stage.labels_and_rows(data, pred, entry["ground_truth"][frame], stage.s109.s106.identity(spec, job["source_arm"], frame))
                    for b, row in enumerate(rows):
                        supply.append(dict(sequence=unit["sequence"], seed=unit["seed"], source_arm=job["source_arm"], frame=frame, policy=row["bank"],
                                           best_iou=row["best_iou"], q0_iou=row["q0_iou"], shadow_ppo_iou=row["identity_iou"], shadow_selected=int(pred["selected"][b, 0])))
            (r110/"analysis").mkdir()
            with (r110/"analysis/stage110_state_supply.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(supply[0])); writer.writeheader(); writer.writerows(supply)
            source_bytes = {str(q): q.read_bytes() for base in (r110, r103) for q in base.rglob("*") if q.is_file()}
            bindings = {q: stage.c.file_hash(Path(q)) for q in source_bytes}
            plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, units=jobs, groups=["uav0000001"],
                                     source_stage110_fingerprint=stage.SOURCE110, source_sha256=bindings, code_sha256={}))
            stage.c.publish_identical(root/"manifest.json", plan)
            original_labels = stage.labels_and_rows
            def labels(data, pred, truth, ident):
                self.assertTrue(any((root/"predictions").glob(f"*__f{ident['frame']:06d}.receipt.json")))
                return original_labels(data, pred, truth, ident)
            with (patch.object(stage.p, "fit_model", side_effect=AssertionError("New fit forbidden")),
                  patch.object(stage.e63, "Capture", side_effect=AssertionError("GPU forbidden")),
                  patch.object(stage.s105, "checked_image", side_effect=AssertionError("Image reads forbidden")),
                  patch.object(stage, "labels_and_rows", side_effect=labels), contextlib.redirect_stdout(io.StringIO())):
                args = (root, r110, r103, p110, p103, plan, lambda _: entry, lambda _: self.pack, old)
                stage.run(*args)
                first = {str(q): q.read_bytes() for q in root.rglob("*") if q.is_file()}
                stage.run(*args)
                self.assertEqual(first, {str(q): q.read_bytes() for q in root.rglob("*") if q.is_file()})
            stage.check_results(root)
            self.assertEqual(source_bytes, {str(q): q.read_bytes() for base in (r110, r103) for q in base.rglob("*") if q.is_file()})
            report = stage.c.c66.sealed(root/"analysis/stage111_decision.json", stage.SCHEMA)
            self.assertEqual(report["counts"]["state_summary"], 24); self.assertEqual(report["new_actor_evaluations"], 144)
            self.assertEqual(report["maximum_source_iou_error"], 0.)
            cache = next((root/"predictions").glob("*.npz")); saved = cache.read_bytes(); cache.write_bytes(b"tampered")
            with self.assertRaises(ValueError): stage.check_results(root)
            cache.write_bytes(saved)
            source_path = next((r110/"observations").glob("*.npz")); source_path.write_bytes(b"changed source")
            with self.assertRaises(ValueError): stage.check_results(root)


if __name__ == "__main__": unittest.main()
