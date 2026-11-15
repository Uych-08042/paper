"""Verify fixed feature maps, source isolation, factorial controls and CPU pipeline."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage108_relational_commit_gate as stage
import test_stage107_gate_support as fixture


def row(group, frame, value, mode):
    x=np.zeros(len(stage.s106.FEATURES));x[0]=value
    x[stage.s106.FEATURES.index("mode_pairwise")]=mode
    x[stage.s106.FEATURES.index("ppo_value")]=5.
    return dict(cohort="historical" if group.endswith("1") else "expanded",sequence=group+"_00000_s",group=group,
        seed=1,source_arm="four_pairwise" if mode else "four_matched",frame=frame,last=frame+3,x=x,eligible=True,
        reason="eligible",future=float((2*mode-1)*value),myopic=float(.5*(2*mode-1)*value),roles="unused_offline_role")


class RelationalGate(unittest.TestCase):
    def rows(self):
        return [row(f"uav{i:07d}",frame,value,mode) for i in range(1,4) for mode in (0,1) for frame,value in ((1,-1.),(2,1.))]

    def test_feature_maps_fixed_schema_relations_and_constant_interactions(self):
        rows=self.rows();x=rows[0]["x"].copy()
        for name,value in (("anchor_score_delta",.2),("reference_score_delta",-.1),("four_extrapolation_error",2.),
                           ("pair_extrapolation_error",.1),("four_vs_median_log_w",.8),("pair_vs_median_log_w",.2)):
            x[stage.s106.FEATURES.index(name)]=value
        rel=stage.relation_features(x)
        self.assertAlmostEqual(rel[0],np.log(2));self.assertAlmostEqual(rel[4],np.log(2)*.2)
        self.assertAlmostEqual(rel[10],(np.tanh(2.)-np.tanh(.1))*.2)
        mirrored=x.copy()
        for name in ("pair_minus_four_dx","four_move_dx","pair_move_dx","past_velocity_dx"):
            mirrored[stage.s106.FEATURES.index(name)]*=-1
        np.testing.assert_array_equal(rel,stage.relation_features(mirrored))
        for arm,dim in zip(stage.ARMS,(49,65,97,129)):
            mapping=stage.mapping_parameters(rows,arm)
            a=stage.transform(x,arm,mapping);b=stage.transform(np.array([x,x]),arm,mapping)
            self.assertEqual(len(a),dim);self.assertEqual(len(stage.feature_names(arm)),dim)
            self.assertEqual(len(set(stage.feature_names(arm))),dim)
            np.testing.assert_array_equal(a[:49],x);np.testing.assert_array_equal(a,b[0])
            if arm in ("mode","relation_mode"):
                feature=stage.feature_names(arm).index("mode_x_ppo_value")
                self.assertEqual(a[feature],0.)
        bad=x.copy();bad[stage.s106.FEATURES.index("mode_pairwise")]=.5
        with self.assertRaises(ValueError):stage.transform(bad,"mode",stage.mapping_parameters(rows,"mode"))
        bad=x.copy();bad[0]=np.nan
        with self.assertRaises(ValueError):stage.relation_features(bad)

    def test_fit_only_mapping_and_mode_dependent_slopes(self):
        rows=self.rows();group="uav0000001"
        model=stage.fit_gate(rows,group,"mode","future")
        changed=copy.deepcopy(rows)
        for r in changed:
            if r["group"]==group:r["x"][0]=999.;r["future"]=-900.;r["myopic"]=900.;r["roles"]="changed"
        self.assertEqual(model,stage.fit_gate(changed,group,"mode","future"))
        self.assertNotIn(group,model["fit_groups"])
        bad=copy.deepcopy(model);bad["beta"][0]+=.1
        with self.assertRaises(ValueError):stage.audit_fit(bad,stage.fit_inputs(rows,group,"mode","future"),"mode","future")
        baseline=stage.s106.fit_gate(rows,group,"future")
        held=[r for r in rows if r["group"]==group]
        base=stage.evaluate(baseline,held,"baseline");new=stage.evaluate(model,held,"mode")
        self.assertLess(new["target_mse"],base["target_mse"])
        self.assertGreater(new["future_utility"],base["future_utility"])
        self.assertEqual(new["future_positive_acceptance"],1.)
        self.assertEqual(new["future_negative_acceptance"],0.)
        for arm in stage.ARMS[1:]:
            a=stage.fit_gate(rows,group,arm,"future");b=stage.fit_gate(rows,group,arm,"myopic")
            self.assertEqual(a["mapping"],b["mapping"]);self.assertEqual(a["center"],b["center"])
            self.assertEqual(a["fit_rows"],b["fit_rows"])
        masked=stage.predict_batch(model,[held[0]["x"]],[False],"mode")
        self.assertEqual(masked["score"][0],0.);self.assertFalse(masked["commit"][0])

    def test_rate_controls_macro_weighting_and_factorial_identity(self):
        rows=self.rows();models={}
        for group in sorted({r["group"] for r in rows}):
            for arm in stage.ARMS:
                for head in stage.s106.HEADS:
                    models[group,arm,head]=stage.s106.fit_gate(rows,group,head) if arm=="baseline" else stage.fit_gate(rows,group,arm,head)
        predictions=stage.predict_samples(rows,models);decisions=stage.decisions(rows,predictions)
        self.assertEqual(len(decisions),len(rows)*15)
        for arm in stage.ARMS:
            policy=[r for r in decisions if r["policy"]==arm+"__future"]
            expected=[r for r in decisions if r["policy"]==arm+"__rate_expected"]
            self.assertEqual(sum(r["commit_probability"] for r in policy),sum(r["commit_probability"] for r in expected))
        summary,paired=stage.summaries(decisions)
        local={r["policy"]:r["utility"] for r in summary if r["cohort"]==r["source_arm"]=="all"}
        interaction=next(r for r in paired if r["cohort"]==r["source_arm"]=="all" and r["contrast"]=="relation_by_mode_interaction")
        self.assertAlmostEqual(interaction["mean_delta"],local['relation_mode__future']-local['relation__future']-local['mode__future']+local['baseline__future'])
        invalid=dict(rows[0],eligible=False,future=0.,myopic=0.)
        tiny=[invalid];tiny_predictions=stage.predict_samples(tiny,models)
        self.assertTrue(all(r["commit_probability"]==0 for r in stage.decisions(tiny,tiny_predictions)))

    def test_full_pipeline_reuse_no_network_label_isolation_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);source=base/"source";root=base/"run";root.mkdir()
            old,report,rows,_=fixture.GateSupport().source(source)
            first=rows[0]
            plan=stage.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,groups=old["groups"],samples=len(rows),
                eligible_samples=sum(r["eligible"] for r in rows),new_models=6*len(old["groups"]),
                focus=[["T01",first["sequence"],first["seed"],first["source_arm"],first["frame"]]],
                exposure_contexts=old["exposure_contexts"],exposure_states=old["exposure_states"],source_sha256={},code_sha256={}))
            stage.c.publish_identical(root/"manifest.json",plan)
            original={str(p):stage.c.file_hash(p) for p in source.rglob("*") if p.is_file()}
            fit=stage.fit_gate
            with patch.object(stage,"fit_gate",wraps=fit) as fitted,\
                 patch.object(stage.s106,"fit_gate",side_effect=AssertionError("Do not refit original control")),\
                 patch.object(stage.s106.s103,"observe",side_effect=AssertionError("No GPU inference")):
                stage.run(root,source,old,report,plan)
                self.assertEqual(fitted.call_count,18)
                result=stage.c.c66.sealed(root/"analysis/stage108_decision.json",stage.SCHEMA)
                self.assertEqual(result["new_gate_models"],18);self.assertEqual(result["new_search_calls"],0)
                before=fitted.call_count
                reader=stage.c.read_cache
                def altered(path,meta):
                    data=reader(path,meta)
                    if data is not None and "offline_current_commit_delta" in data:data["offline_current_commit_delta"][:]=-999.
                    return data
                with patch.object(stage.c,"read_cache",side_effect=altered):stage.run(root,source,old,report,plan)
                self.assertEqual(fitted.call_count,before)
            self.assertEqual(original,{str(p):stage.c.file_hash(p) for p in source.rglob("*") if p.is_file()})
            path=next((root/"exposure").glob("*.npz"))
            with path.open("ab") as stream:stream.write(b"tampered")
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=="__main__":unittest.main()
