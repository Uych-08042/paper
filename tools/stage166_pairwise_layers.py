"""Learn within-frame layer preferences at the original SGLA switch starts."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiment_progress import OverallProgress
import stage165_layer_reproposal as stage165


gain, veto, prior = stage165.gain, stage165.veto, stage165.prior
c, io, trained, LAYERS = stage165.c, stage165.io, stage165.gain.trained, stage165.LAYERS
SCHEMA = "stage166_pairwise_layer_ranking_v1"
SOURCE165 = "dc02d44ae9215a66c9a8cd34a7d5ec9db7bbbd7ddca0d43f2ed244d9325eec9d"
MARGIN = .01
REGULARIZATION = .01
RAW = "stage159_gain_scores"
ADJUSTED = "stage165_risk_scores"
STATIC = "pairwise_layer_prior"
LEARNED = "pairwise_state_ranker"
DIM = trained.DESIGN["projection"] + trained.HISTORY_FEATURES


def load_baselines(points, decision, choices, layers):
    if (decision["status"] != "COMPLETE_STAGE165_LAYER_REPROPOSAL" or decision["source_stage159"] != veto.SOURCE159
            or decision["source_stage164"] != stage165.SOURCE164 or decision["cases"] != len(points)):
        raise ValueError("Need the formal Stage-165 comparison")
    gain.attach_control(points, [
        dict(r, arm="history16_trend", layer=r["short_layer"], predicted_gain=r["short_gain"],
             predicted_downside=r["fixed_downside"], ranking_score=r["short_score"]) for r in choices])
    by_case = {r["case_id"]: r for r in choices}
    layer_rows = {(r["case_id"], int(r["layer"])): r for r in layers}
    if len(layers) != len(layer_rows) or len(layers) != len(points) * len(LAYERS):
        raise ValueError("Incomplete or duplicate Stage-165 layer scores")
    bank = {RAW: [], ADJUSTED: []}
    for p in points:
        row = by_case[p["case_id"]]
        entries = [layer_rows[p["case_id"], k] for k in LAYERS]
        margins = [float(r["predicted_gain"]) for r in entries]
        adjusted = [float(r["ranking_score"]) for r in entries]
        if (any(int(r["fold"]) != int(row["fold"]) or r["sequence"] != p["sequence"] or int(r["frame"]) != p["frame"]
                or float(r["tail32"]) != p["tail32"][LAYERS.index(int(r["layer"]))] for r in entries)
                or not np.isfinite([margins, adjusted]).all()):
            raise ValueError(f"Stage-165 layer record differs at {p['case_id']}")
        picked = stage165.repick(p, np.array(margins), np.array([float(r["predicted_downside"]) for r in entries]))
        old_action = int(row["short_layer"])
        if picked != int(row["adjusted_proposal"]) or int(row["same_starts_layer"]) != (
                picked if old_action != p["native_layer"] else p["native_layer"]):
            raise ValueError(f"Stage-165 proposal did not replay at {p['case_id']}")
        p["fold"] = int(row["fold"])
        p["adjusted_layer"] = int(row["same_starts_layer"])
        bank[RAW].append(margins)
        bank[ADJUSTED].append(adjusted)
    return {name: np.asarray(values, float) for name, values in bank.items()}


def frame_pairs(point, safety):
    """Return ordered pairs of preferred and disfavored nonnative layer indices."""
    native = LAYERS.index(point["native_layer"])
    good, low, tail = point["good32"], point["low32"], point["tail32"]
    unsafe = (good[native] & ~good) | (low & ~low[native])
    for a, b in combinations((i for i in range(len(LAYERS)) if i != native), 2):
        if safety and (unsafe[a] or unsafe[b]):
            if unsafe[a] != unsafe[b]:
                yield (b, a) if unsafe[a] else (a, b)
        elif abs(tail[a] - tail[b]) >= MARGIN:
            yield (a, b) if tail[a] > tail[b] else (b, a)


def fit_ranker(points, features, fit, held):
    fit_ids = [i for i in np.flatnonzero(fit) if points[i]["future"] != points[i]["native_layer"]]
    frame_weights = Counter(veto.source_group(points[i]["sequence"]) for i in fit_ids)
    weights = np.array([1. / frame_weights[veto.source_group(points[i]["sequence"])] for i in fit_ids])
    if not len(fit_ids):
        raise ValueError("No eligible training frames")
    train = features[fit_ids]
    mean = np.average(train, axis=0, weights=weights)
    scale = np.sqrt(np.average((train - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-9, scale, 1.)
    normalized = np.c_[(features - mean) / scale, np.ones(len(features))]
    pairs = [(i, list(frame_pairs(points[i], safety=True))) for i in fit_ids]
    supported = Counter(veto.source_group(points[i]["sequence"]) for i, rows in pairs if rows)
    count = sum(len(rows) for _, rows in pairs)
    if not count:
        raise ValueError("No supervised layer pairs")
    columns = normalized.shape[1]
    matrix = np.zeros((count, len(LAYERS), columns))
    pair_weights = np.empty(count)
    j = 0
    for i, rows in pairs:
        if not rows:
            continue
        weight = 1. / (supported[veto.source_group(points[i]["sequence"])] * len(rows) * len(supported))
        for positive, negative in rows:
            matrix[j, positive] = normalized[i]
            matrix[j, negative] = -normalized[i]
            pair_weights[j] = weight
            j += 1
    matrix = matrix.reshape(count, -1)

    def objective(params):
        margin = matrix @ params
        loss = pair_weights @ np.logaddexp(0., -margin)
        gradient = -matrix.T @ (pair_weights * np.exp(-np.logaddexp(0., margin)))
        return float(loss + .5 * REGULARIZATION * (params @ params)), gradient + REGULARIZATION * params

    result = minimize(objective, np.zeros(matrix.shape[1]), method="L-BFGS-B", jac=True,
                      options=dict(maxiter=500, ftol=1e-12, gtol=1e-9))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Pairwise optimization failed: {result.message}")
    coef = result.x.reshape(len(LAYERS), columns)
    scores = normalized[held] @ coef.T
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite held-out layer scores")
    audit = dict(fit_sources=len(supported), fit_frames=sum(supported.values()), fit_pairs=count,
                 held_frames=int(held.sum()), iterations=int(result.nit), objective=float(result.fun))
    model = dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist())
    return scores, audit, model


def fit_folds(points, folds, features, progress):
    mode_features = {STATIC: np.empty((len(points), 0))}
    if features is not None:
        mode_features[LEARNED] = features
    groups = np.array([veto.source_group(p["sequence"]) for p in points])
    bank = {arm: np.full((len(points), len(LAYERS)), np.nan) for arm in mode_features}
    audits, models = [], {}
    for fold in folds:
        fold_id = fold["fold"]
        fit, held = np.isin(groups, fold["fit"]), np.isin(groups, fold["held"])
        if not fit.any() or not held.any() or not np.array_equal(fit, ~held):
            raise ValueError("Unexpected source partition")
        np.testing.assert_array_equal(held, [p["fold"] == fold_id for p in points])
        for arm, x in mode_features.items():
            bank[arm][held], audit, model = fit_ranker(points, x, fit, held)
            audits.append(dict(fold=fold_id, arm=arm, feature_dim=x.shape[1], **audit))
            models[f"fold{fold_id}_{arm}"] = model
            progress.advance()
    if any(not np.isfinite(x).all() for x in bank.values()):
        raise ValueError("Incomplete source-held layer predictions")
    return bank, audits, models


def score_quality(points, bank):
    """Measure layer ordering before accepting or rejecting a switch."""
    report = []
    for mode, safety in (("iou_margin", False), ("safety_then_iou", True)):
        pairs = [(i, a, b) for i, p in enumerate(points) if p["future"] != p["native_layer"]
                 for a, b in frame_pairs(p, safety)]
        for arm, scores in bank.items():
            per_source = defaultdict(list)
            for i, a, b in pairs:
                score = scores[i, a] - scores[i, b]
                per_source[veto.source_group(points[i]["sequence"])].append(
                    .5 if abs(score) <= veto.EPSILON else float(score > 0.))
            for cohort in ("all", "without_0176"):
                rows = [values for source, values in per_source.items() if cohort == "all" or source != "uav0000176"]
                if not rows:
                    raise ValueError(f"No evaluable layer pairs for {mode}, {cohort}")
                total = sum(map(len, rows))
                report.append(dict(arm=arm, group=cohort, metric=mode, videos=len(rows), pairs=total,
                                   accuracy=sum(map(sum, rows)) / total,
                                   video_mean_accuracy=float(np.mean([np.mean(values) for values in rows]))))
    return report


def evaluate(points, folds, bank, fitted, progress):
    actions, choices, scope = gain.evaluate(points, folds, progress)
    actions[stage165.SAME_STARTS] = [p["adjusted_layer"] for p in points]
    for arm, scores in fitted.items():
        layers = actions["native"].copy()
        for i, (p, score) in enumerate(zip(points, scores, strict=True)):
            if actions[gain.SHORT][i] != p["native_layer"]:
                # On equal scores, retain the original proposal.
                layers[i] = max((k for k in LAYERS if k != p["native_layer"]),
                                key=lambda k: (score[LAYERS.index(k)], k == p["control_layer"], -k))
            choices[i][arm + "_layer"] = layers[i]
        actions[arm] = layers
    for i, row in enumerate(choices):
        row["stage165_same_starts_layer"] = actions[stage165.SAME_STARTS][i]
    for row in scope:
        ids = [i for i, p in enumerate(points) if p["fold"] == row["fold"]]
        for arm in (stage165.SAME_STARTS, *fitted):
            chosen = {i for i in ids if actions[arm][i] != actions["native"][i]}
            old = {i for i in ids if actions[gain.SHORT][i] != actions["native"][i]}
            if chosen != old:
                raise ValueError(f"Switch starts changed in {arm}")
            row[arm + "_changed_layers"] = sum(actions[arm][i] != actions[gain.SHORT][i] for i in chosen)
    layer_scores = [dict(case_id=p["case_id"], sequence=p["sequence"], frame=p["frame"], fold=p["fold"], layer=k,
                         **{arm: float(x[i, j]) for arm, x in bank.items()}, **veto.effects(p, k))
                    for i, p in enumerate(points) for j, k in enumerate(LAYERS)]
    return actions, choices, scope, layer_scores


def archive_features(root, plan, points, progress):
    rows = []
    for p in points:
        cid = p["case_id"]
        data = c.read_cache(root / f"branches/{cid}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=cid))
        if (data is None or data["features"].shape != (DIM,) or not np.isfinite(data["features"]).all()
                or int(data["frame_id"][0]) != p["frame"] or int(data["anchor_layer"]) != p["native_layer"]):
            raise ValueError(f"Invalid archived feature vector at {cid}")
        rows.append(data["features"])
        progress.advance()
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage159-root", type=Path)
    parser.add_argument("--stage165-root", type=Path)
    parser.add_argument("--cached-diagnostic", nargs=2, type=Path, metavar=("STAGE159_BUNDLE", "STAGE165_BUNDLE"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    sources = [p.expanduser().resolve() for p in (args.stage159_root, args.stage165_root) if p is not None]
    cached = args.cached_diagnostic is not None
    if (cached and sources) or (not cached and len(sources) != 2) or args.progress_interval <= 0:
        parser.error("Provide both server roots or both result bundles, and a positive progress interval")
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE166_CACHED_LAYER_PRIOR" if cached else "COMPLETE_STAGE166_PAIRWISE_LAYER_RANKING"
    with prior.prior.writer_lock(root):
        destination = root / "stage166_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage165"] != SOURCE165:
                raise ValueError("Use separate output directories for the layer prior and full experiment")
            io.verify({str(root / k): h for k, h in
                       (saved["table_sha256"] | {"stage166_models.json": saved["model_sha256"]}).items()})
            OverallProgress("Stage-166", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            _, a = prior.archives.bundle(args.cached_diagnostic[0], 159, veto.SOURCE159)
            d, b = prior.archives.bundle(args.cached_diagnostic[1], 165, SOURCE165)
            points = veto.collect_points(a["stage159_windows.csv"], a["stage159_choices.csv"])
            folds = trained.source.split_sources(sorted({p["sequence"] for p in points}))
        else:
            r159, r165 = sources
            plan, _, points = veto.load(r159)
            folds = plan["folds"]
            d = c.c66.sealed(r165 / "stage165_decision.json", stage165.SCHEMA)
            if d["fingerprint"] != SOURCE165:
                raise ValueError("Need the frozen Stage-165 results")
            names = ("stage165_choices.csv", "stage165_layer_scores.csv")
            io.verify({str(r165 / k): d["table_sha256"][k] for k in names})
            b = {k: prior.prior.read_table(r165 / k) for k in names}
        original = load_baselines(points, d, b["stage165_choices.csv"], b["stage165_layer_scores.csv"])
        work = (2 if cached else 3) * len(folds) + 1 + (0 if cached else len(points))
        progress = OverallProgress("Stage-166", work, interval=args.progress_interval, unit="work")
        features = None
        if not cached:
            progress.report("Reading archived L6 and history features")
            features = archive_features(r159, plan, points, progress)
        progress.report("Fitting held-out layer ranks")
        fitted, fit, models = fit_folds(points, folds, features, progress)
        bank = original | fitted
        metrics = score_quality(points, bank)
        progress.report("Evaluating the fixed switch starts")
        actions, choices, scope, scores = evaluate(points, folds, bank, fitted, progress)
        hashes, checks = gain.report(root, points, actions, choices, scope, stage="stage166", treatment=LEARNED)
        hashes.update(prior.publish(root, dict(pair_quality=metrics, fit=fit, layer_scores=scores), stage="stage166"))
        model_path = root / "stage166_models.json"
        c.publish_identical(model_path, io.seal(dict(schema=SCHEMA, source_stage165=SOURCE165, models=models)))
        hashes[model_path.name] = c.file_hash(model_path)
        if not cached:
            metric = {(r["arm"], r["group"]): r["video_mean_accuracy"] for r in metrics if r["metric"] == "iou_margin"}
            for group in ("all", "without_0176"):
                checks[f"better_pair_ranking_{group}"] = metric[LEARNED, group] > max(
                    metric[arm, group] for arm in (RAW, ADJUSTED, STATIC)) + veto.EPSILON
        c.publish_identical(destination, io.seal(dict(
            schema=SCHEMA, status=status, source_stage159=veto.SOURCE159, source_stage165=SOURCE165,
            cases=len(points), switch_budgets=list(gain.BUDGETS.values()), switch_starts_fixed=True,
            work=work, supervised_pair_margin=MARGIN, regularization=REGULARIZATION,
            gpu_searches=0, new_full_trajectories=0, table_sha256={k: v for k, v in hashes.items() if k.endswith(".csv")},
            model_sha256=hashes[model_path.name], checks=checks, ready_for_online_test=all(checks.values()) if checks else None)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
