"""Check fixed-model algebra, leakage boundaries, support selection and cache replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage107_gate_support_audit as stage


def sample(group, frame, value, mode="four_matched", eligible=True):
    x = np.zeros(len(stage.s106.FEATURES)); x[0] = value; x[1] = (-1.)**frame
    return dict(cohort="historical" if group.endswith("1") else "expanded", sequence=group+"_00000_s", group=group,
        seed=1, source_arm=mode, frame=frame, last=frame+3, x=x, eligible=eligible,
        reason="eligible" if eligible else "no_commit_change", roles="offline_fixture", four_selected=1, pair_selected=0,
        future=float(.2*x[0]+.1*x[1]) if eligible else 0., myopic=float(-.15*x[0]+.03*x[1]) if eligible else 0.)


class GateSupport(unittest.TestCase):
    def rows(self):
        return [sample(f"uav{i:07d}", frame, value, mode, eligible=not(i == 1 and frame == 3 and mode == "four_pairwise"))
                for i in range(1, 4) for mode in stage.s106.s103.MODES[1:]
                for frame, value in ((1, -1.), (2, 0.), (3, 1.))]

    def test_frozen_normal_equations_and_heldout_label_isolation(self):
        rows = self.rows(); group = "uav0000001"
        model = stage.s106.fit_gate(rows, group, "future")
        fit, math, residual = stage.audit_model(model, rows)
        self.assertLess(residual, 1e-12); self.assertEqual(len(fit), 12)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["group"] == group:
                row["future"] = 900.; row["myopic"] = -900.; row["x"][:] = 100.
        _, other, _ = stage.audit_model(model, changed)
        np.testing.assert_array_equal(math["matrix"], other["matrix"])
        bad = copy.deepcopy(model); bad["beta"][0] += .1
        with self.assertRaises(ValueError): stage.audit_model(bad, rows)
        bad = copy.deepcopy(model); bad["center"][0] += .1
        with self.assertRaises(AssertionError): stage.audit_model(bad, rows)
        changed = copy.deepcopy(rows); changed[-1]["future"] += 1
        with self.assertRaises(ValueError): stage.audit_model(model, changed)

    def test_feature_source_decomposition_and_label_free_distinct_neighbors(self):
        rows = self.rows(); model = stage.s106.fit_gate(rows, "uav0000001", "future")
        fit, math, _ = stage.audit_model(model, rows)
        query = rows[2]["x"]
        z, contribution, sources, linear = stage.decompose(model, fit, math, query)
        self.assertAlmostEqual(linear, model["intercept"]+contribution.sum())
        self.assertAlmostEqual(linear, sum(r["target_contribution"] for r in sources))
        self.assertAlmostEqual(1., sum(r["influence_weight"] for r in sources))
        self.assertNotIn("uav0000001", {r["fit_source_group"] for r in sources})
        near = stage.nearest_groups(query, fit, model, 5)
        self.assertEqual(len(near), 2)
        self.assertEqual(len({fit[i]["group"] for i, _ in near}), 2)
        changed = copy.deepcopy(fit)
        for r in changed: r.update(future=999., myopic=-999., roles="changed_role")
        self.assertEqual(near, stage.nearest_groups(query, changed, model, 5))
        with self.assertRaises(ValueError): stage.nearest_groups(query, rows, model, 5)

    def test_fit_adequacy_is_distinct_from_heldout_generalization(self):
        train = [sample("uav0000002", 1, -1.), sample("uav0000002", 2, 1.),
                 sample("uav0000003", 1, -1.), sample("uav0000003", 2, 1.)]
        held = [sample("uav0000001", 1, -1.), sample("uav0000001", 2, 1.)]
        for r in train+held:
            r["x"][1] = 0.; r["future"] = float(r["x"][0])
        for r in held: r["future"] *= -1
        model = stage.s106.fit_gate(train+held, "uav0000001", "future")
        a, b = stage.evaluate(model, train), stage.evaluate(model, held)
        self.assertGreater(a["target_r2"], .7); self.assertLess(b["target_r2"], 0.)
        xor = []
        for i, (x, y) in enumerate(((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)), 1):
            r = sample("uav0000002", i, x); r["x"][1] = y; r["future"] = x*y; xor.append(r)
        weak = stage.s106.fit_gate(xor, "uav0000001", "future")
        result = stage.evaluate(weak, xor)
        self.assertAlmostEqual(result["target_mse"], result["intercept_mse"])
        self.assertAlmostEqual(result["target_r2"], 0.)
        self.assertEqual(stage.evaluate(weak, [dict(xor[0], eligible=False)])["eligible_rows"], 0)

    def source(self, root):
        root.mkdir(); (root/"analysis").mkdir(); (root/"models").mkdir()
        rows = self.rows(); groups = sorted({r["group"] for r in rows})
        models = {(g, h): stage.s106.fit_gate(rows, g, h) for g in groups for h in stage.s106.HEADS}
        plan = stage.io.seal(dict(schema=stage.s106.SCHEMA, groups=groups, anchors=len(rows), exposure_contexts=6, exposure_states=30))
        stage.c.publish_identical(root/"manifest.json", plan)
        for (g, h), m in models.items(): stage.c.publish_identical(root/"models"/(g+"__"+h+".json"), m)
        arrays = dict(x=np.asarray([r["x"] for r in rows]), eligible=np.asarray([r["eligible"] for r in rows]),
                      **{h: np.asarray([r[h] for r in rows]) for h in stage.s106.HEADS})
        stage.c.save(root/"samples.npz", arrays, dict(plan_fingerprint=plan["fingerprint"], kind="matched_samples"))
        stage.s106.s103.s86.publish_table(root/"analysis/stage106_samples.csv", [{k:v for k,v in r.items() if k != "x"} for r in rows])
        decisions = stage.s106.matched_decisions(rows, models)
        stage.s106.s103.s86.publish_table(root/"analysis/stage106_sample_decisions.csv", decisions)
        units, evidence = [], {}
        for group in groups:
            for mode in stage.s106.s103.MODES[1:]:
                unit = dict(sequence=group+"_00000_s", seed=1, fold=0, origin="fixture", policy="four_ppo")
                x = np.asarray([sample(group, i+1, v)["x"] for i, v in enumerate((-1., 0., 1., 3., 9.))])
                eligible = np.asarray([True, True, False, True, True])
                data = dict(frame_id=np.arange(1, 6), x=x, eligible=eligible, offline_current_commit_delta=np.ones(5))
                for head in stage.s106.HEADS:
                    predicted = [stage.s106.predict(models[group, head], v, e) for v,e in zip(x, eligible)]
                    for key in predicted[0]: data[head+"_"+key] = np.asarray([r[key] for r in predicted])
                    units.append(dict(sequence=unit["sequence"], seed=1, source_arm=mode, policy=head, group=group,
                        cohort="historical" if group.endswith("1") else "expanded", states=5,
                        eligible=float(eligible.mean()), commit_probability=float(data[head+"_commit"].mean())))
                path = root/"exposure"/(stage.c.key(unit)+"__"+mode+".npz")
                meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, source_arm=mode, kind="frozen_source_exposure")
                stage.c.save(path, data, meta); evidence[path.relative_to(root).as_posix()] = stage.c.file_hash(path)
        stage.s106.s103.s86.publish_table(root/"analysis/stage106_exposure_units.csv", units)
        return plan, dict(artifact_sha256=evidence), rows, models

    def test_complete_cpu_pipeline_resume_exposure_labels_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            base = Path(directory); source = base/"source"; root = base/"run"; root.mkdir()
            old, report, rows, models = self.source(source)
            first = rows[0]
            plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN,
                focus=[["T01", first["sequence"], first["seed"], first["source_arm"], first["frame"]]],
                groups=old["groups"], models=6, samples=len(rows), eligible_samples=sum(r["eligible"] for r in rows),
                exposure_contexts=6, exposure_states=30, source_sha256={}, code_sha256={}))
            stage.c.publish_identical(root/"manifest.json", plan)
            original = {str(p):stage.c.file_hash(p) for p in source.rglob("*") if p.is_file()}
            with patch.object(stage.s106, "fit_gate", side_effect=AssertionError("No new training")), \
                 patch.object(stage.s106.s103, "observe", side_effect=AssertionError("No inference")):
                stage.run(root, source, old, report, plan)
                stage.run(root, source, old, report, plan)
            self.assertEqual(original, {str(p):stage.c.file_hash(p) for p in source.rglob("*") if p.is_file()})
            result = stage.c.c66.sealed(root/"analysis/stage107_decision.json", stage.SCHEMA)
            self.assertEqual(result["new_models"], 0); self.assertEqual(result["exposure_states"], 30)
            self.assertEqual(result["counts"]["state_support"], 36)
            self.assertEqual(result["counts"]["neighbors"], 68)
            normal = stage.exposure_audit(source, old, report, models)
            reader = stage.c.read_cache
            def alter(path, meta):
                data = reader(path, meta)
                if data is not None and "offline_current_commit_delta" in data:
                    data["offline_current_commit_delta"][:] = -900
                return data
            with patch.object(stage.c, "read_cache", side_effect=alter):
                self.assertEqual(normal, stage.exposure_audit(source, old, report, models))
            def drift(path, meta):
                data = reader(path, meta)
                if data is not None and "future_score" in data: data["future_score"][0] += .1
                return data
            with patch.object(stage.c, "read_cache", side_effect=drift), self.assertRaises(AssertionError):
                stage.exposure_audit(source, old, report, models)
            path = root/"analysis/stage107_focus.csv"
            with path.open("ab") as stream: stream.write(b"tampered")
            with self.assertRaises(ValueError): stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
