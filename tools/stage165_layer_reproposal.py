"""Test risk-aware layer proposals at the fixed Stage-163 switch budgets."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage163_history_gain as gain
import stage164_local_match_gain as stage164


history, risk, veto, prior, c, io = gain.history, gain.risk, gain.veto, gain.prior, gain.c, gain.io
LAYERS = gain.layers
SCHEMA = "stage165_layer_reproposal_v1"
SOURCE164 = "4574f92e613bf14106b3193306d2c375bf58b8fe2e12e2ed58e79454e0d97234"
REPROPOSE = "risk_adjusted_reproposal"
SAME_STARTS = "reproposal_same_starts"
ORACLES = ("fixed_proposal", "six_layers_same_eligible", "six_layers_all_starts")
SOURCES = (121, 159, 164)


def attach_control(points, decision, rows):
    if (decision["status"] != stage164.STATUS or decision["source_stage163"] != stage164.SOURCE163
            or decision["cases"] != len(points)):
        raise ValueError("Stage-164 result differs from the completed local-match ablation")
    gain.attach_control(points, [
        dict(r, arm="history16_trend", layer=r["short_layer"], predicted_gain=r["short_gain"],
             predicted_downside=r["fixed_downside"], ranking_score=r["short_score"]) for r in rows])
    fold_of = {r["case_id"]: int(r["fold"]) for r in rows}
    for p in points:
        p["fold"] = fold_of[p["case_id"]]


def safe_gain(outcome):
    return outcome["delta32"] > veto.EPSILON and not (outcome["lost_good"] or outcome["new_low"])


def bounds(points):
    """Retrospective headroom with at most the frozen per-fold number of switches."""
    effects = [{layer: veto.effects(p, layer) for layer in LAYERS} for p in points]
    summary, selections = [], []
    for cohort in ("all", "without_0176"):
        indices = [i for i, p in enumerate(points)
                   if cohort == "all" or veto.source_group(p["sequence"]) != "uav0000176"]
        groups = Counter(veto.source_group(points[i]["sequence"]) for i in indices)
        for arm in ORACLES:
            weighted_gains, support, switches, recoveries = [], 0, 0, 0
            fold_counts = {}
            for fold, budget in gain.BUDGETS.items():
                candidates = []
                for i in indices:
                    p = points[i]
                    if p["fold"] != fold or (arm != "six_layers_all_starts" and p["future"] == p["native_layer"]):
                        continue
                    layers = (p["future"],) if arm == "fixed_proposal" else LAYERS
                    feasible = [layer for layer in layers if safe_gain(effects[i][layer])]
                    if not feasible:
                        continue
                    layer = max(feasible, key=lambda k: (effects[i][k]["delta32"], -k))
                    size = groups[veto.source_group(p["sequence"])]
                    candidates.append((effects[i][layer]["delta32"] / size, i, layer))
                support += len(candidates)
                candidates.sort(key=lambda row: (-row[0], points[row[1]]["case_id"], row[2]))
                count = min(budget, len(candidates))
                fold_counts[f"fold{fold}_switches"] = count
                switches += count
                for value, i, layer in candidates[:count]:
                    p = points[i]
                    outcome = effects[i][layer]
                    weighted_gains.append(value)
                    recoveries += int(outcome["new_good"])
                    selections.append(dict(cohort=cohort, oracle=arm, case_id=p["case_id"], sequence=p["sequence"],
                                           frame=p["frame"], fold=fold, layer=layer, old_proposal=p["future"], **outcome))
            summary.append(dict(cohort=cohort, oracle=arm, cases=len(indices), videos=len(groups),
                                safe_positive_cases=support, switches=switches, new_good=recoveries,
                                mean_delta_vs_native=sum(weighted_gains) / len(groups), **fold_counts))
    return summary, selections


def predict_held(points, indices, fit, held, vectors, model):
    short = vectors[history.BASELINE][held]
    ref = {k: np.asarray(v, float) for k, v in model.items()}
    prediction = ((short - ref["mean"]) / ref["scale"]) @ ref["coef"] + ref["bias"]
    held_points = [points[i] for i in indices]
    frozen_proposals, frozen_margins = veto.predictions(held_points, prediction)
    np.testing.assert_array_equal(frozen_proposals, [p["future"] for p in held_points])
    np.testing.assert_allclose(frozen_margins, [p["margin"] for p in held_points], rtol=0, atol=1e-12)
    downside, _ = risk.downside_matrix(points, vectors["history16_trend"], fit, held)
    proposed = [LAYERS.index(p["future"]) for p in held_points]
    np.testing.assert_allclose(downside[np.arange(len(indices)), proposed],
                               [p["fixed_downside"] for p in held_points], rtol=0, atol=1e-12)
    native = [LAYERS.index(p["native_layer"]) for p in held_points]
    margins = prediction - prediction[np.arange(len(indices)), native, None]
    return margins, downside


def repick(point, margins, downside):
    """Prefer the original proposal on ties and preserve the original eligible start set."""
    if point["future"] == point["native_layer"]:
        return point["native_layer"]
    scores = margins - risk.PENALTY * downside
    nonnative = [k for k in LAYERS if k != point["native_layer"]]
    return max(nonnative, key=lambda k: (scores[LAYERS.index(k)], k == point["future"],
                                         margins[LAYERS.index(k)], -k))


def compare(points, folds, vectors, models, progress):
    actions, choices, scope = gain.evaluate(points, folds, progress)
    native = actions["native"]
    actions[SAME_STARTS] = native.copy()
    actions[REPROPOSE] = native.copy()
    datasets = np.array([veto.source_group(p["sequence"]) for p in points])
    scores_table = []
    for fold, scope_row in zip(folds, scope, strict=True):
        fold_id = fold["fold"]
        fit, held = np.isin(datasets, fold["fit"]), np.isin(datasets, fold["held"])
        indices = np.flatnonzero(held)
        margins, downsides = predict_held(points, indices, fit, held, vectors, models[f"fold{fold_id}_future_supervised"])
        proposals, eligible = {}, []
        for i, candidate_gains, candidate_risks in zip(indices, margins, downsides, strict=True):
            p = points[i]
            new = repick(p, candidate_gains, candidate_risks)
            j = LAYERS.index(new)
            margin, downside = float(candidate_gains[j]), float(candidate_risks[j])
            score = margin - risk.PENALTY * downside
            proposals[i] = (new, margin, downside, score)
            if new != native[i]:
                eligible.append(i)
                if score < p["control_score"] - 1e-12:
                    raise ValueError(f"Adjusted proposal score regressed at {p['case_id']}")
            for k, m, d in zip(LAYERS, candidate_gains, candidate_risks, strict=True):
                scores_table.append(dict(case_id=p["case_id"], sequence=p["sequence"], frame=p["frame"], fold=fold_id,
                                         layer=k, predicted_gain=float(m), predicted_downside=float(d),
                                         ranking_score=float(m - risk.PENALTY * d), **veto.effects(p, k)))
        if len(eligible) != scope_row["eligible"]:
            raise ValueError("The eligible start count changed")
        ranked = sorted(eligible, key=lambda i: (-proposals[i][3], -proposals[i][1], points[i]["case_id"]))
        ranks = {i: rank for rank, i in enumerate(ranked, 1)}
        selected = set(ranked[:gain.BUDGETS[fold_id]])
        old_selected = {i for i in indices if actions[gain.SHORT][i] != native[i]}
        for i in indices:
            p = points[i]
            new, margin, downside, score = proposals[i]
            if i in old_selected:
                actions[SAME_STARTS][i] = new
            if i in selected:
                actions[REPROPOSE][i] = new
            choices[i].update(
                adjusted_proposal=new, adjusted_gain=margin, adjusted_downside=downside, adjusted_score=score,
                adjusted_rank=ranks.get(i), same_starts_layer=actions[SAME_STARTS][i],
                reallocated_layer=actions[REPROPOSE][i],
                **{f"adjusted_{k}": v for k, v in veto.effects(p, new).items()})
        scope_row.update(
            changed_proposals=sum(proposals[i][0] != points[i]["future"] for i in eligible),
            same_starts_changed_layer=sum(actions[SAME_STARTS][i] != actions[gain.SHORT][i] for i in old_selected),
            reallocated_switches=len(selected), reallocated_replaced_starts=len(selected - old_selected),
            reallocated_changed_layer=sum(actions[REPROPOSE][i] != actions[gain.SHORT][i] for i in selected & old_selected))
        progress.advance()
    return actions, choices, scope, scores_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--cached-diagnostic", nargs=2, type=Path, metavar=("STAGE159_BUNDLE", "STAGE164_BUNDLE"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root")) is not None]
    cached = args.cached_diagnostic is not None
    if (cached and paths) or (not cached and len(paths) != len(SOURCES)) or args.progress_interval <= 0:
        parser.error("Provide all server roots or both result bundles, and a positive progress interval")
    if any(root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE165_CACHED_PROPOSAL_BOUNDS" if cached else "COMPLETE_STAGE165_LAYER_REPROPOSAL"
    with prior.prior.writer_lock(root):
        destination = root / "stage165_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage164"] != SOURCE164:
                raise ValueError("Use separate directories for the retrospective bounds and learned policy")
            io.verify({str(root / k): h for k, h in saved["table_sha256"].items()})
            OverallProgress("Stage-165", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            _, a = prior.archives.bundle(args.cached_diagnostic[0], 159, veto.SOURCE159)
            d, b = prior.archives.bundle(args.cached_diagnostic[1], 164, SOURCE164)
            points = veto.collect_points(a["stage159_windows.csv"], a["stage159_choices.csv"])
            folds = gain.trained.source.split_sources(sorted({p["sequence"] for p in points}))
            rows = b["stage164_choices.csv"]
        else:
            r121, r159, r164 = paths
            plan, models, points = veto.load(r159)
            folds = plan["folds"]
            d = c.c66.sealed(r164 / "stage164_decision.json", stage164.SCHEMA)
            if d["fingerprint"] != SOURCE164:
                raise ValueError("Need the frozen Stage-164 result")
            name = "stage164_choices.csv"
            io.verify({str(r164 / name): d["table_sha256"][name]})
            rows = prior.prior.read_table(r164 / name)
        attach_control(points, d, rows)
        expected_folds = {g: f["fold"] for f in folds for g in f["held"]}
        if any(p["fold"] != expected_folds[veto.source_group(p["sequence"])] for p in points):
            raise ValueError("Stage-164 and Stage-159 fold assignments differ")
        work = len(folds) + 1 + (0 if cached else len(points) + len(folds))
        progress = OverallProgress("Stage-165", work, interval=args.progress_interval, unit="work")
        reports, selections = bounds(points)
        if cached:
            progress.report("Checking cached proposal bounds")
            actions, choices, scope = gain.evaluate(points, folds, progress)
            scores = []
        else:
            progress.report("Reading frozen causal features")
            vectors, _ = history.prepare(points, plan, r159, r121, progress)
            progress.report("Comparing layer proposals")
            actions, choices, scope, scores = compare(points, folds, vectors, models, progress)
        hashes, checks = gain.report(root, points, actions, choices, scope, stage="stage165", treatment=REPROPOSE)
        tables = dict(oracle_bounds=reports, oracle_choices=selections)
        if scores:
            tables["layer_scores"] = scores
        hashes.update(prior.publish(root, tables, stage="stage165"))
        c.publish_identical(destination, io.seal(dict(
            schema=SCHEMA, status=status, source_stage159=veto.SOURCE159, source_stage164=SOURCE164,
            source_stage162=gain.SOURCE162, cases=len(points), switch_budgets=list(gain.BUDGETS.values()),
            work=work, refit_downside_models=0 if cached else len(folds), new_gain_models=0, gpu_searches=0,
            new_full_trajectories=0, table_sha256=hashes, checks=checks,
            ready_for_online_test=all(checks.values()) if checks else None)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
