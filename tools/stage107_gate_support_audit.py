"""Audit fixed commit gates: fit adequacy, held-out support and exact linear scores."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage106_commit_gate as s106


c, io = s106.c, s106.io
SCHEMA = "stage107_fixed_commit_gate_support_audit_v1"
STATUS = "COMPLETE_FIXED_COMMIT_GATE_SUPPORT_AUDIT"
SOURCE106 = "644e44f73492caf63b5a8dcecb475b3728f3415b13d01eb9651a482e7d8b06d6"
DESIGN = dict(neighbor_groups=5, verification_atol=1e-10, verification_rtol=1e-9,
              neighbors="fit_only_standardized_RMS_one_anchor_per_distinct_source_group",
              source_decomposition="fixed_normalization_and_normal_matrix_not_group_deletion")
FOCUS = (
    ("F01", "uav0000304_00253_s", 2172, "four_pairwise", 556),
    ("F02", "uav0000304_00253_s", 2172, "four_pairwise", 1083),
    ("F03", "uav0000176_00000_s", 2172, "four_matched", 251),
    ("F04", "uav0000176_00000_s", 2172, "four_matched", 165),
    ("F05", "uav0000176_00000_s", 2172, "four_pairwise", 165),
    ("F06", "uav0000304_00253_s", 2172, "four_matched", 552),
    ("F07", "uav0000091_00460_s", 2171, "four_matched", 87),
    ("F08", "uav0000091_00460_s", 2171, "four_pairwise", 87),
    ("F09", "uav0000348_02415_s", 2171, "four_matched", 364),
    ("F10", "uav0000348_02415_s", 2171, "four_pairwise", 364),
    ("F11", "uav0000003_00000_s", 2171, "four_pairwise", 138),
    ("F12", "uav0000176_00000_s", 2171, "four_matched", 157),
    ("F13", "uav0000176_00000_s", 2171, "four_pairwise", 157))
TABLES = ("model_scope", "prediction_summary", "prediction_macro", "state_support", "neighbors",
          "feature_attribution", "source_contribution", "exposure_units", "exposure_features",
          "focus", "focus_features", "focus_neighbors", "focus_sources")
PACK = tuple(n for n in TABLES if n not in ("neighbors", "feature_attribution", "source_contribution"))
IDENTITY = ("cohort", "sequence", "group", "seed", "source_arm", "frame")
EVAL_METRICS = ("target_mse", "intercept_mse", "target_mae", "target_r2", "target_positive_acceptance",
                "target_negative_acceptance", "future_positive_acceptance", "future_negative_acceptance",
                "future_utility", "always_future_utility", "rate_future_utility",
                "oracle_future_utility", "future_regret", "commit_rate")


def read_csv(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    np.testing.assert_allclose(a, b, atol=DESIGN["verification_atol"], rtol=DESIGN["verification_rtol"], equal_nan=True)


def sample_key(row):
    return row["sequence"], int(row["seed"]), row["source_arm"], int(row["frame"])


def load_samples(root, plan):
    data = s106.required(root/"samples.npz", dict(plan_fingerprint=plan["fingerprint"], kind="matched_samples"))
    rows = read_csv(root/"analysis/stage106_samples.csv")
    n = plan["anchors"]
    if set(data) != {"x", "eligible", *s106.HEADS} or data["x"].shape != (n, len(s106.FEATURES)) or len(rows) != n:
        raise ValueError("Incomplete Stage-106 samples")
    for name in ("eligible", *s106.HEADS):
        if data[name].shape != (n,):
            raise ValueError("Sample alignment changed")
    out = []
    for i, row in enumerate(rows):
        record = dict(row, x=data["x"][i].copy(), seed=int(row["seed"]), frame=int(row["frame"]), last=int(row["last"]),
                      eligible=bool(data["eligible"][i]))
        if row["eligible"] != str(record["eligible"]) or row["group"] != s106.source_group(row["sequence"]):
            raise ValueError("Sample identity/eligibility changed")
        for head in s106.HEADS:
            value = float(data[head][i]); close(value, float(row[head]) if row[head] else np.nan)
            record[head] = value
        if not np.isfinite(record["x"]).all():
            raise ValueError("Invalid cached feature vector")
        out.append(record)
    if len({sample_key(r) for r in out}) != n or sorted({r["group"] for r in out}) != plan["groups"]:
        raise ValueError("Source group/sample coverage changed")
    return out


def common_support(rows):
    return [r for r in rows if r["eligible"] and all(np.isfinite(r[h]) for h in s106.HEADS)]


def audit_model(model, samples):
    """Verify the sealed fit's normal equations; do not optimize or replace it."""
    group, head = model["held_out_group"], model["head"]
    if model["schema"] != s106.SCHEMA or model["design"] != s106.DESIGN or model["features"] != list(s106.FEATURES) or head not in s106.HEADS:
        raise ValueError("Unknown frozen model definition")
    fit = common_support([r for r in samples if r["group"] != group])
    if not fit or not model["supported"] or group in model["fit_groups"]:
        raise ValueError("Missing support or held-out label leakage")
    groups = sorted({r["group"] for r in fit})
    contexts = len({(r["group"], r["seed"], r["source_arm"]) for r in fit})
    if model["fit_groups"] != groups or model["fit_rows"] != len(fit) or model["fit_contexts"] != contexts:
        raise ValueError("Frozen training support differs")
    inputs = [dict(group=r["group"], sequence=r["sequence"], seed=r["seed"], source_arm=r["source_arm"],
                   frame=r["frame"], x=r["x"].tolist(), y=float(r[head])) for r in fit]
    fingerprint = io.seal(dict(design=s106.DESIGN, head=head, held_out_group=group, features=list(s106.FEATURES), fit_rows=inputs))["fingerprint"]
    if model["input_fingerprint"] != fingerprint:
        raise ValueError("Frozen fit input fingerprint differs")
    x = np.asarray([r["x"] for r in fit]); w = s106.balanced_weights(fit)
    mean = w @ x; scale = np.sqrt(w @ ((x-mean)**2)); scale[scale < s106.DESIGN["minimum_scale"]] = 1.
    close(model["center"], mean); close(model["scale"], scale)
    close(model["minimum"], x.min(0)); close(model["maximum"], x.max(0)); close(w.sum(), 1.)
    z = (x-model["center"])/model["scale"]
    y = np.asarray([r[head] for r in fit]); close(model["intercept"], w @ y)
    matrix = z.T @ (w[:, None]*z)+s106.DESIGN["ridge"]*np.eye(len(s106.FEATURES))
    rhs = z.T @ (w*(y-model["intercept"]))
    residual = matrix @ np.asarray(model["beta"])-rhs
    normalized = float(np.abs(residual).max()/max(1., np.abs(rhs).max()))
    if not np.isfinite(normalized) or normalized > DESIGN["verification_rtol"]:
        raise ValueError("Sealed ridge coefficients violate normal equations")
    return fit, dict(x=x, z=z, w=w, y=y, matrix=matrix), normalized


def score_rows(model, rows):
    return np.asarray([s106.predict(model, r["x"], r["eligible"])["score"] for r in rows])


def evaluate(model, rows):
    """Conditional on common eligible support, weighted group/seed/mode/anchor."""
    eligible = common_support(rows)
    result = dict(rows=len(rows), eligible_rows=len(eligible), groups=len({r["group"] for r in eligible}))
    if not eligible:
        return result | {m: None for m in EVAL_METRICS}
    w = s106.balanced_weights(eligible)
    scores = score_rows(model, eligible); head = model["head"]
    target = np.asarray([r[head] for r in eligible]); future = np.asarray([r["future"] for r in eligible])
    p = (scores > s106.DESIGN["threshold"]).astype(float)
    variance = float(w @ ((target-w @ target)**2))
    mse = float(w @ ((scores-target)**2))
    rates = defaultdict(list)
    for i, row in enumerate(eligible):
        rates[row["group"], row["seed"], row["source_arm"]].append(i)
    expected = np.zeros(len(eligible))
    for indices in rates.values():
        expected[indices] = p[indices].mean()
    conditional = lambda mask: float(w[mask] @ p[mask]/w[mask].sum()) if mask.any() else None
    oracle = float(w @ np.maximum(future, 0))
    return result | dict(target_mse=mse, intercept_mse=float(w @ ((model["intercept"]-target)**2)),
        target_mae=float(w @ np.abs(scores-target)), target_r2=1-mse/variance if variance > 0 else None,
        target_positive_acceptance=conditional(target > 0), target_negative_acceptance=conditional(target < 0),
        future_positive_acceptance=conditional(future > 0), future_negative_acceptance=conditional(future < 0),
        future_utility=float(w @ (p*future)), always_future_utility=float(w @ future),
        rate_future_utility=float(w @ (expected*future)), oracle_future_utility=oracle,
        future_regret=oracle-float(w @ (p*future)), commit_rate=float(w @ p))


def nearest_groups(x, fit, model, count):
    """Select indices using X only; attach labels only after ranking is complete."""
    if any(r["group"] == model["held_out_group"] for r in fit):
        raise ValueError("Held-out group entered neighbor support")
    distances = np.sqrt(np.mean(((np.asarray([r["x"] for r in fit])-x)/model["scale"])**2, axis=1))
    best = {}
    for i, (row, distance) in enumerate(zip(fit, distances)):
        group = row["group"]
        if group not in best or distance < best[group][0]:
            best[group] = (float(distance), i)
    ranked = sorted(best.items(), key=lambda item: (item[1][0], item[0]))[:count]
    return [(i, distance) for _, (distance, i) in ranked]


def decompose(model, fit, math, x):
    """Exact affine feature and fit-group contributions, not causal ablations."""
    z = (x-np.asarray(model["center"]))/model["scale"]
    contributions = z*np.asarray(model["beta"])
    linear_score = float(model["intercept"]+contributions.sum())
    # For fixed weighted centering and A, h_i = w_i (1 + z_i' A^-1 z_query).
    h = math["w"]*(1+math["z"] @ np.linalg.solve(math["matrix"], z))
    close(h.sum(), 1.); close(h @ math["y"], linear_score)
    sources = []
    for group in model["fit_groups"]:
        mask = np.asarray([r["group"] == group for r in fit])
        sources.append(dict(fit_source_group=group, fit_rows=int(mask.sum()), influence_weight=float(h[mask].sum()),
                            target_contribution=float(h[mask] @ math["y"][mask])))
    close(sum(r["target_contribution"] for r in sources), linear_score)
    return z, contributions, sources, linear_score


def sample_audit(samples, models, focus):
    tables = {n: [] for n in TABLES if not n.startswith("exposure")}
    focus_map = {tuple(row[1:]): row[0] for row in focus}
    if len(focus_map) != len(focus) or set(focus_map)-{sample_key(r) for r in samples}:
        raise ValueError("Missing or duplicate declared focus state")
    for (group, head), model in sorted(models.items()):
        fit, math, residual = audit_model(model, samples)
        held = [r for r in samples if r["group"] == group]
        tables["model_scope"].append(dict(held_out_group=group, head=head, fit_groups=len(model["fit_groups"]),
            fit_rows=len(fit), fit_contexts=model["fit_contexts"], held_out_rows=len(held),
            input_fingerprint=model["input_fingerprint"], model_fingerprint=model["fingerprint"],
            normalized_normal_equation_residual=residual, normalization_pass=True, group_exclusion_pass=True))
        for phase, rows in (("fit_resubstitution", fit), ("held_out", held)):
            tables["prediction_summary"].append(dict(held_out_group=group, head=head, phase=phase, **evaluate(model, rows)))
        for r in held:
            ident = {k: r[k] for k in IDENTITY} | dict(head=head, case_id=focus_map.get(sample_key(r), ""))
            z, contribution, sources, linear = decompose(model, fit, math, r["x"])
            prediction = s106.predict(model, r["x"], r["eligible"])
            close(prediction["score"], linear if r["eligible"] else 0.)
            outside = (r["x"] < model["minimum"]) | (r["x"] > model["maximum"])
            neighbors = nearest_groups(r["x"], fit, model, DESIGN["neighbor_groups"]) if r["eligible"] else []
            state = ident | dict(eligible=r["eligible"], reason=r["reason"], target=float(r[head]) if np.isfinite(r[head]) else None,
                measured_future=float(r["future"]) if np.isfinite(r["future"]) else None,
                score=prediction["score"], commit=prediction["commit"], intercept=model["intercept"],
                unmasked_linear_score=linear, feature_contribution_sum=float(contribution.sum()),
                outside_features=int(outside.sum()) if r["eligible"] else 0,
                max_abs_z=float(np.abs(z).max()) if r["eligible"] else 0., neighbor_groups=len(neighbors),
                nearest_rms_distance=neighbors[0][1] if neighbors else None,
                neighbor_mean_target=float(np.mean([fit[i][head] for i, _ in neighbors])) if neighbors else None,
                neighbor_mean_future=float(np.mean([fit[i]["future"] for i, _ in neighbors])) if neighbors else None,
                negative_influence_weight=float(sum(v["influence_weight"] for v in sources if v["influence_weight"] < 0)))
            tables["state_support"].append(state)
            if ident["case_id"]:
                tables["focus"].append(state)
            for j, name in enumerate(s106.FEATURES):
                row = ident | dict(feature=name, raw_value=float(r["x"][j]), fit_center=model["center"][j], fit_scale=model["scale"][j],
                    fit_minimum=model["minimum"][j], fit_maximum=model["maximum"][j], standardized_value=float(z[j]),
                    coefficient=model["beta"][j], score_contribution=float(contribution[j]), outside_fit_range=bool(outside[j]))
                tables["feature_attribution"].append(row)
                if ident["case_id"]:
                    tables["focus_features"].append(row)
            for rank, (i, distance) in enumerate(neighbors, 1):
                neighbor = fit[i]
                row = ident | dict(rank=rank, rms_distance=distance, neighbor_group=neighbor["group"],
                    neighbor_sequence=neighbor["sequence"], neighbor_seed=neighbor["seed"], neighbor_source_arm=neighbor["source_arm"],
                    neighbor_frame=neighbor["frame"], neighbor_target=float(neighbor[head]), neighbor_future=float(neighbor["future"]))
                tables["neighbors"].append(row)
                if ident["case_id"]:
                    tables["focus_neighbors"].append(row)
            for source in sources:
                row = ident | source
                tables["source_contribution"].append(row)
                if ident["case_id"]:
                    tables["focus_sources"].append(row)
        print(f"[Stage-107 model] {group} {head}: fit {len(fit)}, held-out {len(held)}", flush=True)
    for head in s106.HEADS:
        for phase in ("fit_resubstitution", "held_out"):
            rows = [r for r in tables["prediction_summary"] if r["head"] == head and r["phase"] == phase]
            tables["prediction_macro"].append(dict(head=head, phase=phase, models=len(rows),
                **{m: s106.mean(rows, m) for m in EVAL_METRICS}))
    return tables


def audit_cached_decisions(root, samples, states):
    old = {(sample_key(r), r["policy"]): r for r in read_csv(root/"analysis/stage106_sample_decisions.csv")}
    if len(old) != len(samples)*len(s106.POLICIES):
        raise ValueError("Incomplete frozen sample decision table")
    for row in states:
        source = old[sample_key(row), row["head"]]
        close(row["score"], float(source[row["head"]+"_score"]))
        if row["commit"] != bool(float(source["commit_probability"])):
            raise ValueError("Frozen gate action drift")


def exposure_audit(root, plan, report, models):
    units, features = [], []
    source_units = read_csv(root/"analysis/stage106_exposure_units.csv")
    lookup = {(r["sequence"], int(r["seed"]), r["source_arm"], r["policy"]): r for r in source_units}
    if len(lookup) != 2*plan["exposure_contexts"]:
        raise ValueError("Duplicate source exposure context")
    for relative in sorted(p for p in report["artifact_sha256"] if p.startswith("exposure/") and p.endswith(".npz")):
        path = root/relative
        receipt = c.c66.sealed(path.with_suffix(".receipt.json"), c.SCHEMA)
        meta = receipt["metadata"]
        if set(meta) != {"plan_fingerprint", "unit", "source_arm", "kind"} or meta["plan_fingerprint"] != plan["fingerprint"] or meta["kind"] != "frozen_source_exposure":
            raise ValueError("Mixed exposure metadata")
        unit, mode = meta["unit"], meta["source_arm"]
        if mode not in s106.s103.MODES[1:] or path.name != c.key(unit)+"__"+mode+".npz":
            raise ValueError("Exposure identity changed")
        data = s106.required(path, meta)
        x, eligible = data["x"], data["eligible"].astype(bool)
        n = len(data["frame_id"])
        if x.shape != (n, len(s106.FEATURES)) or eligible.shape != (n,) or not np.isfinite(x).all():
            raise ValueError("Incomplete exposure features")
        np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
        group = s106.source_group(unit["sequence"])
        for head in s106.HEADS:
            source = lookup[unit["sequence"], unit["seed"], mode, head]
            model = models[group, head]
            z = (x-model["center"])/model["scale"]
            score = np.where(eligible, z @ model["beta"]+model["intercept"], 0.)
            commit = eligible & (score > s106.DESIGN["threshold"])
            outside = eligible[:, None] & ((x < model["minimum"]) | (x > model["maximum"]))
            any_out = outside.any(1); maximum = np.where(eligible, np.abs(z).max(1), 0.)
            close(data[head+"_score"], score); close(data[head+"_max_abs_z"], maximum)
            np.testing.assert_array_equal(data[head+"_commit"], commit)
            np.testing.assert_array_equal(data[head+"_outside_fit_range"], any_out)
            close(float(source["eligible"]), eligible.mean()); close(float(source["commit_probability"]), commit.mean())
            if int(source["states"]) != n or source["group"] != group:
                raise ValueError("Exposure count/group differs")
            ident = dict(cohort=source["cohort"], sequence=unit["sequence"], group=group, seed=unit["seed"], source_arm=mode)
            units.append(ident | dict(head=head, states=n, eligible_states=int(eligible.sum()), commits=int(commit.sum()),
                outside_states=int(any_out.sum()), score_replay_pass=True, range_replay_pass=True))
            if head == s106.HEADS[0]:
                # Both heads must use the identical fitting support/normalization.
                other = models[group, "myopic"]
                for name in ("center", "scale", "minimum", "maximum"):
                    close(model[name], other[name])
                for j, name in enumerate(s106.FEATURES):
                    features.append(ident | dict(feature=name, states=n, eligible_states=int(eligible.sum()),
                        outside_count=int(outside[:, j].sum()),
                        outside_fraction=float(outside[eligible, j].mean()) if eligible.any() else None))
        print(f"[Stage-107 exposure] {path.stem}: {n} original cached states", flush=True)
    if len(units) != 2*plan["exposure_contexts"] or sum(r["states"] for r in units) != 2*plan["exposure_states"]:
        raise ValueError("Incomplete exposure coverage")
    aggregated = []
    for cohort in ("all", "historical", "expanded"):
        for mode in ("all", *s106.s103.MODES[1:]):
            for name in s106.FEATURES:
                rows = [r for r in features if r["feature"] == name and (cohort == "all" or r["cohort"] == cohort) and (mode == "all" or r["source_arm"] == mode)]
                if not rows:
                    continue
                macro = s106.macro(rows, "outside_fraction")
                aggregated.append(dict(cohort=cohort, source_arm=mode, feature=name, groups=len(macro), contexts=len(rows),
                    eligible_states=sum(r["eligible_states"] for r in rows), outside_count=sum(r["outside_count"] for r in rows),
                    macro_outside_fraction=float(np.mean(list(macro.values()))) if macro else None))
    return units, aggregated


def load_models(root, groups):
    models = {(g, h): c.c66.sealed(root/"models"/(g+"__"+h+".json"), s106.SCHEMA) for g in groups for h in s106.HEADS}
    if any(m["held_out_group"] != g or m["head"] != h for (g, h), m in models.items()):
        raise ValueError("Model filename/identity differs")
    return models


def run(root, source_root, source_plan, source_report, plan):
    samples = load_samples(source_root, source_plan)
    if len(common_support(samples)) != plan["eligible_samples"]:
        raise ValueError("Common eligible support changed")
    models = load_models(source_root, source_plan["groups"])
    tables = sample_audit(samples, models, plan["focus"])
    audit_cached_decisions(source_root, samples, tables["state_support"])
    tables["exposure_units"], tables["exposure_features"] = exposure_audit(source_root, source_plan, source_report, models)
    publish(root, plan, tables)


def publish(root, plan, tables):
    if set(tables) != set(TABLES) or len(tables["model_scope"]) != plan["models"] or len(tables["state_support"]) != 2*plan["samples"]:
        raise ValueError("Incomplete fixed gate audit")
    n = plan["samples"]
    expected = dict(prediction_summary=2*plan["models"], prediction_macro=4,
        feature_attribution=2*n*len(s106.FEATURES), focus=2*len(plan["focus"]),
        focus_features=2*len(plan["focus"])*len(s106.FEATURES), exposure_units=2*plan["exposure_contexts"],
        source_contribution=2*n*(len(plan["groups"])-1),
        neighbors=2*plan["eligible_samples"]*min(DESIGN["neighbor_groups"], len(plan["groups"])-1))
    if any(len(tables[name]) != count for name, count in expected.items()):
        raise ValueError("Incomplete score decomposition/focus")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    out = root/"analysis"; out.mkdir(exist_ok=True)
    hashes = {}
    for name in TABLES:
        path = out/f"stage107_{name}.csv"
        s106.s103.s86.publish_table(path, tables[name]); hashes[path.name] = c.file_hash(path)
    c.publish_identical(out/"stage107_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_fixed_gate_fit_and_support_audit", plan_fingerprint=plan["fingerprint"],
        source_stage106_fingerprint=SOURCE106, counts={n: len(tables[n]) for n in TABLES},
        source_models_audited=plan["models"], exposure_states=plan["exposure_states"],
        maximum_normal_equation_residual=max(r["normalized_normal_equation_residual"] for r in tables["model_scope"]),
        new_models=0, training_updates=0, new_image_reads=0, new_search_calls=0, new_full_trajectories=0, confirmation_inference_calls=0,
        frozen_sample_decisions_replayed=True, fit_only_support_pass=True, exact_score_decomposition_pass=True,
        cached_exposure_replayed=True, table_sha256=hashes,
        warnings=["Fit-resubstitution scores reuse training labels and overlapping models; they are not generalization evidence.",
                  "Neighbors rank features only; their labels and scores are offline descriptions, not new gate outputs.",
                  "Feature and source contributions are algebra of the frozen linear model, not causal deletion effects.",
                  "No new full-policy trajectory, AUC, recovery protection or adoption gate; confirmation groups stay sealed."])))
    check_results(root)
    print(f"Stage-107: {STATUS}; {plan['models']} fixed models; no new fitting or GPU inference", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"analysis/stage107_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage106_fingerprint"] != SOURCE106 or report["source_models_audited"] != plan["models"]
            or report["exposure_states"] != plan["exposure_states"] or report["counts"]["model_scope"] != plan["models"]
            or report["counts"]["state_support"] != 2*plan["samples"]
            or report["counts"]["prediction_summary"] != 2*plan["models"] or report["counts"]["prediction_macro"] != 4
            or report["counts"]["feature_attribution"] != 2*plan["samples"]*len(s106.FEATURES)
            or report["counts"]["focus"] != 2*len(plan["focus"])
            or report["counts"]["focus_features"] != 2*len(plan["focus"])*len(s106.FEATURES)
            or report["counts"]["exposure_units"] != 2*plan["exposure_contexts"]
            or report["counts"]["source_contribution"] != 2*plan["samples"]*(len(plan["groups"])-1)
            or report["counts"]["neighbors"] != 2*plan["eligible_samples"]*min(DESIGN["neighbor_groups"], len(plan["groups"])-1)
            or set(report["table_sha256"]) != {f"stage107_{n}.csv" for n in TABLES}
            or not all(report[k] for k in ("frozen_sample_decisions_replayed", "fit_only_support_pass", "exact_score_decomposition_pass", "cached_exposure_replayed"))
            or not np.isfinite(report["maximum_normal_equation_residual"]) or report["maximum_normal_equation_residual"] > DESIGN["verification_rtol"]
            or any(report[k] != 0 for k in ("new_models", "training_updates", "new_image_reads", "new_search_calls", "new_full_trajectories", "confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-107 report")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n): h for n, h in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path); parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root); return
    if not args.input:
        parser.error("Supply the completed Stage-106 directory")
    source = args.input.expanduser().resolve()
    if root == source or root in source.parents or source in root.parents:
        parser.error("Use a separate Stage-107 directory")
    with s106.s93.writer_lock(root):
        if (root/"analysis/stage107_decision.json").exists():
            check_results(root); return
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-107 directory")
        print("Stage-107: verifying frozen source; CPU arithmetic only", flush=True)
        s106.check_results(source)
        source_plan = c.c66.sealed(source/"manifest.json", s106.SCHEMA)
        report = c.c66.sealed(source/"analysis/stage106_decision.json", s106.SCHEMA)
        if report["fingerprint"] != SOURCE106 or report["supported_gate_models"] != 56 or source_plan["anchors"] != 259:
            raise ValueError("Need the declared Stage-106 result")
        bindings = dict(source_plan["source_sha256"])
        for path in (source/"manifest.json", source/"analysis/stage106_decision.json"):
            bindings[str(path)] = c.file_hash(path)
        for name in ("table_sha256", "artifact_sha256"):
            for relative, digest in report[name].items():
                path = source/("analysis/"+relative if name == "table_sha256" else relative)
                bindings[str(path)] = digest
                if path.suffix == ".npz":
                    bindings[str(path.with_suffix(".receipt.json"))] = c.file_hash(path.with_suffix(".receipt.json"))
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage107_gate_support.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, focus=[list(r) for r in FOCUS], models=56, samples=259,
            groups=source_plan["groups"], eligible_samples=224,
            exposure_contexts=source_plan["exposure_contexts"], exposure_states=source_plan["exposure_states"],
            source_sha256=bindings, code_sha256=source_plan["code_sha256"] | {str(p): c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json", plan)
        run(root, source, source_plan, report, plan)


if __name__ == "__main__":
    main()
