"""Vary only the gain model's history features at the frozen Stage-162 switch budget."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage162_history_ablation as history


risk, veto, prior, c, io = history.risk, history.veto, history.prior, history.c, history.io
trained, layers = history.trained, veto.LAYERS
SCHEMA = "stage163_history_gain_v1"
SOURCE162 = "ed7127ced8c2e67070acfdc30de349261dec82764e90a37ac5430412fd10f568"
BUDGETS = {0: 64, 1: 4}
SHORT = "history4_gain"
LONG = "history16_gain"


def attach_control(points, rows):
    control = [r for r in rows if r["arm"] == "history16_trend"]
    by_id = {r["case_id"]: r for r in control}
    if len(control) != len(by_id) or by_id.keys() != {p["case_id"] for p in points}:
        raise ValueError("Incomplete or duplicate Stage-162 control rows")
    for p in points:
        cid = p["case_id"]
        r = by_id[cid]
        layer = int(r["layer"])
        gain, downside, score = map(float, (r["predicted_gain"], r["predicted_downside"], r["ranking_score"]))
        if (r["sequence"] != p["sequence"] or int(r["frame"]) != p["frame"]
                or int(r["native_layer"]) != p["native_layer"] or int(r["proposed_layer"]) != p["future"]
                or layer not in (p["native_layer"], p["future"]) or not np.isfinite([gain, downside, score]).all()
                or downside < 0 or score != gain - risk.PENALTY * downside):
            raise ValueError(f"Stage-162 control differs at {cid}")
        p.update(margin=gain, fixed_downside=downside, control_score=score, control_layer=layer)


def gain_prediction(points, indices, fit, held, groups, features, model, feature_name="history16_trend"):
    short = features[history.BASELINE][held]
    reference = {k: np.asarray(v, float) for k, v in model.items()}
    archived = ((short - reference["mean"]) / reference["scale"]) @ reference["coef"] + reference["bias"]
    held_points = [points[i] for i in indices]
    proposed, margin = veto.predictions(held_points, archived)
    np.testing.assert_array_equal(proposed, [p["future"] for p in held_points])
    np.testing.assert_allclose(margin, [p["margin"] for p in held_points], atol=1e-12, rtol=0)

    treatment = features[feature_name]
    fit_points = [p for p, use in zip(points, fit, strict=True) if use]
    targets = np.stack([p["tail32"] - p["tail32"][layers.index(p["native_layer"])] for p in fit_points])
    counts = Counter(groups[fit])
    weights = np.array([1. / counts[g] for g in groups[fit]])
    mean, scale, bias, coef = trained.ridge(treatment[fit], targets, weights)
    pred = ((treatment[held] - mean) / scale) @ coef + bias
    if not np.isfinite(pred).all():
        raise ValueError("Invalid held-out gain")
    return [float(row[layers.index(p["future"])] - row[layers.index(p["native_layer"])])
            for p, row in zip(held_points, pred, strict=True)]


def evaluate(points, folds, progress, features=None, models=None, *, treatment=LONG,
             feature_name="history16_trend", column="long"):
    groups = np.array([veto.source_group(p["sequence"]) for p in points])
    native = [p["native_layer"] for p in points]
    control = [p["control_layer"] for p in points]
    actions = {"native": native, SHORT: control}
    if features is not None:
        actions[treatment] = native.copy()
    if sorted(f["fold"] for f in folds) != sorted(BUDGETS):
        raise ValueError("Unexpected outer folds")
    coverage = np.zeros(len(points), int)
    scores_by_case = [None] * len(points)
    scope = []
    for fold in folds:
        fold_id = fold["fold"]
        fit, held = np.isin(groups, fold["fit"]), np.isin(groups, fold["held"])
        if not fit.any() or not held.any() or not np.array_equal(fit, ~held):
            raise ValueError("Invalid source partition")
        coverage += held
        indices = np.flatnonzero(held)
        available = [i for i in indices if points[i]["future"] != native[i]]
        budget = BUDGETS[fold_id]
        old = {i for i in indices if control[i] != native[i]}
        old_scores = {i: points[i]["control_score"] for i in indices}
        if len(old) != budget or risk.top(points, available, [old_scores[i] for i in available], budget) != old:
            raise ValueError(f"Stage-162 ranking did not reproduce switches in fold {fold_id}")
        old_cutoff = min(old_scores[i] for i in old)
        old_order = sorted(available, key=lambda i: (-old_scores[i], -points[i]["margin"], points[i]["case_id"]))
        old_rank = {i: rank for rank, i in enumerate(old_order, 1)}
        gains, new_scores, new_rank, new = {}, {}, {}, set()
        if features is not None:
            held_gain = gain_prediction(points, indices, fit, held, groups, features,
                                        models[f"fold{fold_id}_future_supervised"], feature_name)
            gains = dict(zip(indices, held_gain, strict=True))
            new_scores = {i: gains[i] - risk.PENALTY * points[i]["fixed_downside"] for i in indices}
            new = risk.top(points, available, [new_scores[i] for i in available], budget)
            new_order = sorted(available, key=lambda i: (-new_scores[i], -points[i]["margin"], points[i]["case_id"]))
            new_rank = {i: rank for rank, i in enumerate(new_order, 1)}
            for i in new:
                actions[treatment][i] = points[i]["future"]

        for i in indices:
            p = points[i]
            cutoff_gain = old_cutoff + risk.PENALTY * p["fixed_downside"] if i in old_rank else None
            scores_by_case[i] = dict(
                case_id=p["case_id"], sequence=p["sequence"], frame=p["frame"], fold=fold_id,
                switch_budget=budget, native_layer=native[i], proposed_layer=p["future"], fixed_downside=p["fixed_downside"],
                short_gain=p["margin"], short_score=old_scores[i], short_rank=old_rank.get(i), short_layer=control[i],
                short_cutoff_score=old_cutoff, gain_at_short_cutoff=cutoff_gain,
                short_gain_minus_cutoff=None if cutoff_gain is None else p["margin"] - cutoff_gain,
                **{f"{column}_gain": gains.get(i), f"{column}_score": new_scores.get(i),
                   f"{column}_rank": new_rank.get(i),
                   f"{column}_layer": actions[treatment][i] if treatment in actions else None},
                **{f"proposed_{k}": v for k, v in veto.effects(p, p["future"]).items()})
        scope.append(dict(fold=fold_id, fit_cases=int(fit.sum()), held_cases=int(held.sum()), eligible=len(available),
                          switch_budget=budget, control_switches=len(old),
                          **{f"{column}_switches": len(new) if treatment in actions else None},
                          replaced_switches=len(new - old) if treatment in actions else None))
        progress.advance()
    np.testing.assert_array_equal(coverage, np.ones(len(points), int))
    return actions, scores_by_case, scope


def paired_results(points, actions, control):
    cohort = [i for i, p in enumerate(points) if veto.source_group(p["sequence"]) != "uav0000176"]
    subset = history.paired([points[i] for i in cohort], {arm: [a[i] for i in cohort] for arm, a in actions.items()}, control)
    paired = history.paired(points, actions, control)
    paired.extend(dict(r, group="without_0176") for r in subset if r["group"] == "all")
    return paired


def report(root, points, actions, choices, scope, *, stage="stage163", treatment=LONG, replay_control=None,
           feature_audit=None):
    paired = paired_results(points, actions, SHORT)
    focus = [r for r in choices if (veto.source_group(r["sequence"]), r["frame"]) in history.FOCUS]
    tables = dict(summary=veto.summarize(points, actions), paired=paired, choices=choices, folds=scope, focus=focus)
    checks = {}
    if treatment in actions:
        before = [veto.effects(p, a) for p, a in zip(points, actions[SHORT], strict=True)]
        after = [veto.effects(p, a) for p, a in zip(points, actions[treatment], strict=True)]
        control_recoveries = [i for i, d in enumerate(before) if d["new_good"]]
        long_pair = {r["group"]: r for r in paired if r["arm"] == treatment}
        checks = dict(
            retained_control_recoveries=all(after[i]["new_good"] for i in control_recoveries),
            recovered_0176=any(after[i]["new_good"] for i, p in enumerate(points)
                               if (veto.source_group(p["sequence"]), p["frame"]) == ("uav0000176", 163)),
            no_new_low_or_lost_recovery=not any(a["new_low"] or a["lost_good"] for a in after),
            positive_overall_vs_control=long_pair["all"]["mean_tail32_delta"] > veto.EPSILON,
            positive_without_0176_vs_control=long_pair["without_0176"]["mean_tail32_delta"] > veto.EPSILON,
        )
    if replay_control is not None:
        replay = paired_results(points, {treatment: actions[treatment], replay_control: actions[replay_control]}, replay_control)
        paired.extend(replay)
        by_group = {r["group"]: r for r in replay}
        checks.update(positive_overall_vs_replayed_projection=by_group["all"]["mean_tail32_delta"] > veto.EPSILON,
                      positive_without_0176_vs_replayed_projection=by_group["without_0176"]["mean_tail32_delta"] > veto.EPSILON)
    if feature_audit is not None:
        tables["feature_audit"] = feature_audit
    hashes = prior.publish(root, {name: rows for name, rows in tables.items() if rows}, stage=stage)
    return hashes, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in (121, 159, 162):
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--cached-diagnostic", nargs=2, type=Path, metavar=("STAGE159_BUNDLE", "STAGE162_BUNDLE"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    sources = [p.expanduser().resolve() for p in (args.stage121_root, args.stage159_root, args.stage162_root) if p is not None]
    cached = args.cached_diagnostic is not None
    if (cached and sources) or (not cached and len(sources) != 3) or args.progress_interval <= 0:
        parser.error("Provide all three server roots or both result bundles, and a positive progress interval")
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE163_CACHED_RANK_AUDIT" if cached else "COMPLETE_STAGE163_HISTORY_GAIN_ABLATION"
    with prior.prior.writer_lock(root):
        destination = root / "stage163_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage162"] != SOURCE162:
                raise ValueError("Use separate output directories for the rank audit and gain ablation")
            io.verify({str(root / k): h for k, h in saved["table_sha256"].items()})
            OverallProgress("Stage-163", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            _, a = prior.archives.bundle(args.cached_diagnostic[0], 159, veto.SOURCE159)
            d, b = prior.archives.bundle(args.cached_diagnostic[1], 162, SOURCE162)
            points = veto.collect_points(a["stage159_windows.csv"], a["stage159_choices.csv"])
            rows = b["stage162_choices.csv"]
            folds = trained.source.split_sources(sorted({p["sequence"] for p in points}))
        else:
            r121, r159, r162 = sources
            plan, models, points = veto.load(r159)
            folds = plan["folds"]
            d = c.c66.sealed(r162 / "stage162_decision.json", history.SCHEMA)
            if d["fingerprint"] != SOURCE162:
                raise ValueError("Need the exact completed Stage-162 experiment")
            name = "stage162_choices.csv"
            io.verify({str(r162 / name): d["table_sha256"][name]})
            rows = prior.prior.read_table(r162 / name)
        if (d["status"] != "COMPLETE_STAGE162_HISTORY_ABLATION" or d["source_stage159"] != veto.SOURCE159
                or d["source_stage161"] != history.SOURCE161 or d["cases"] != len(points)):
            raise ValueError("Stage-162 and Stage-159 provenance differs")
        attach_control(points, rows)
        work = len(folds) + 1 + (0 if cached else len(points))
        progress = OverallProgress("Stage-163", work, interval=args.progress_interval, unit="work")
        progress.report("Ranking audit" if cached else "Reading causal histories")
        if cached:
            actions, choices, scope = evaluate(points, folds, progress)
        else:
            features, _ = history.prepare(points, plan, r159, r121, progress)
            progress.phase = "Fitting and evaluating gains"
            actions, choices, scope = evaluate(points, folds, progress, features, models)
        hashes, checks = report(root, points, actions, choices, scope)
        c.publish_identical(destination, io.seal(dict(
            schema=SCHEMA, status=status, source_stage159=veto.SOURCE159, source_stage162=SOURCE162,
            cases=len(points), switch_budgets=list(BUDGETS.values()), work=work, table_sha256=hashes,
            fitted_gain_models=0 if cached else len(folds), gpu_searches=0, new_full_trajectories=0,
            checks=checks, ready_for_online_test=all(checks.values()) if checks else None)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
