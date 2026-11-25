"""Fit-only layer-mean calibration and exact state-correspondence diagnostics.

Consumes the reviewed Stage-197 text bundle only. No training, image access,
model loading, Torch import, or full-video inference is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO / "tools", REPO / "tracking"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical, seal as sealed
from stage4_common import fingerprint, file_hash, write_csv
import stage194_module_common as protocol
import stage195_routing_common as metrics

SCHEMA = "stage198_fit_only_mean_and_state_correspondence_v1"
SOURCE197 = "4b0f88097a9f748eb12d58a53e458b6e53d77d0016da7a001f3b8c7ea026b7b0"
PLAN197 = "43a49f31db5532f7d7a934cdc6aad18f3b9e3a1869dd0c244956db10d137fb53"
BUNDLE_SHA256 = "b824c702af4a80f15ecebd418f4b55f3cdf3cfa8309c780368b1645220cc18b6"
DEFAULT_BUNDLE = REPO / "docs/experiments/06_sgla_uav_tracking/results/stage197_results_20261125T102154Z_1143642.txt"
ARMS = ("ordered", "unordered", "global_stats")
POLICIES = ("raw", "raw_redecoded", "calibrated", "fit_label_mean", "fit_prediction_mean", "native_L9", "oracle")
PREDICTORS = ("raw", "calibrated", "fit_label_mean", "fit_prediction_mean")
SOURCE_TABLES = ("model_scope", "input_audit", "error_summary", "error_video", "policy_summary", "policy_video",
                 "paired", "frame_predictions", "training_history", "source_bridge")
TABLES = ("model_scope", "calibration", "policy_video", "policy_summary", "paired", "error_video",
          "error_summary", "frame_predictions", "source_bridge")


def read_sealed(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint({k: v for k, v in data.items() if k != "fingerprint"}):
        raise ValueError("Changed sealed artifact: " + str(path))
    return data


def load_bundle(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise ValueError("Use the exact reviewed Stage-197 bundle; source bytes changed")
    matches = list(re.finditer(rb"(?:^|\r?\n)===== ([\w.]+) =====\r?\n", raw))
    parts = {}
    for i, match in enumerate(matches):
        name = match[1].decode()
        if name in parts:
            raise ValueError("Duplicate source section: " + name)
        parts[name] = raw[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(raw)]
    expected = {"stage197_decision.json", *(f"stage197_{name}.csv" for name in SOURCE_TABLES)}
    if set(parts) != expected:
        raise ValueError("Incomplete Stage-197 result bundle")
    d = json.loads(parts["stage197_decision.json"])
    if (d["fingerprint"] != fingerprint({k: v for k, v in d.items() if k != "fingerprint"})
            or d["fingerprint"] != SOURCE197 or d["plan_fingerprint"] != PLAN197
            or d["schema"] != "stage197_matched_token_order_routing_v1"
            or d["training_runs"] != 12 or d["training_updates"] != 3072 or d["source_groups"] != 38
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))):
        raise ValueError("Unexpected Stage-197 identity or scope")
    if set(d["table_sha256"]) != expected - {"stage197_decision.json"}:
        raise ValueError("Source table inventory changed")
    tables = {}
    for name, sha in d["table_sha256"].items():
        value = parts[name]
        # The exporter can append one terminal newline, outside the actual CSV.
        candidates = [value]
        if value.endswith(b"\n"):
            candidates.append(value[:-1])
        if value.endswith(b"\r\n"):
            candidates.append(value[:-2])
        data = next((v for v in candidates if hashlib.sha256(v).hexdigest() == sha), None)
        if data is None:
            raise ValueError("Source table checksum mismatch: " + name)
        tables[name.removeprefix("stage197_").removesuffix(".csv")] = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    return d, tables


def source_models(d, tables):
    scopes, frames = tables["model_scope"], tables["frame_predictions"]
    if len(scopes) != 12 or len(frames) != d["predicted_states"]:
        raise ValueError("Incomplete source predictions")
    keys = {(int(r["fold"]), int(r["seed"]), r["arm"]) for r in scopes}
    expected = {(f, s, a) for f in (0, 1) for s in protocol.DESIGN["seeds"] for a in ARMS}
    if keys != expected:
        raise ValueError("Unexpected model grid")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in frames:
        key = int(row["fold"]), int(row["seed"]), row["arm"]
        if key not in keys:
            raise ValueError("Unknown source model")
        grouped[key][row["group"]].append(row)
    all_groups = set(next(iter(grouped.values())))
    if len(all_groups) != 38:
        raise ValueError("Expected 38 historical development groups")
    folds = {r["fold"]: r for r in protocol.split_groups(all_groups)}
    reference_labels, models = {}, []
    for scope in scopes:
        fold_id, seed, arm = int(scope["fold"]), int(scope["seed"]), scope["arm"]
        fold = folds[fold_id]
        if scope["fit_groups"].split("|") != fold["fit"] or scope["held_groups"].split("|") != fold["held"]:
            raise ValueError("Fit/held split changed")
        source = grouped[fold_id, seed, arm]
        if set(source) != all_groups:
            raise ValueError("Missing source group")
        videos = {}
        for group, records in source.items():
            ids = np.asarray([int(r["frame"]) for r in records])
            y = np.asarray([[float(r[f"label_L{k}"]) for k in range(7, 13)] for r in records], float)
            p = np.asarray([[float(r[f"prediction_L{k}"]) for k in range(7, 13)] for r in records], np.float32)
            a = np.asarray([int(r["cpu_layer"]) - 7 for r in records], int)
            metrics.validate_scores(y, p)
            if (len(ids) != len(set(ids)) or (ids <= 0).any() or (np.diff(ids) <= 0).any()
                    or ((a < 0) | (a > 5)).any() or not np.all(p[np.arange(len(p)), a] == p.max(1))):
                raise ValueError("Invalid source state or action")
            for i, row in enumerate(records):
                if row["pool"] != ("fit" if group in fold["fit"] else "held"):
                    raise ValueError("Source pool disagrees with fold")
                if abs(float(row["selected_iou"]) - y[i, a[i]]) > 1e-12:
                    raise ValueError("Archived selected IoU mismatch")
                key = group, int(ids[i])
                value = tuple(y[i])
                if key in reference_labels and reference_labels[key] != value:
                    raise ValueError("True labels changed across models")
                reference_labels[key] = value
            sequence = records[0]["sequence"]
            if any(r["sequence"] != sequence for r in records):
                raise ValueError("Sequence changed inside group")
            videos[group] = dict(sequence=sequence, frame_id=ids, labels=y, prediction=p, choice=a,
                                 original_rows=records)
        draws = protocol.draws_for(fold, seed, {g: len(v["frame_id"]) for g, v in videos.items()})
        if fingerprint(draws) != scope["draws_sha256"]:
            raise ValueError("Training draw identity changed")
        counts = {g: np.zeros(len(v["frame_id"]), int) for g, v in videos.items()}
        for batch in draws:
            for group, i in batch:
                counts[group][i] += 1
        for group, video in videos.items():
            for i, row in enumerate(video["original_rows"]):
                if group in fold["fit"]:
                    if int(row["training_draw_count"]) != counts[group][i]:
                        raise ValueError("Source draw counts changed")
                elif row["training_draw_count"] != "":
                    raise ValueError("Held state was reported as training")
        models.append((scope, fold, seed, arm, videos))
    # Every model must have the same frame IDs, not just the same total count.
    reference = {g: v["frame_id"] for g, v in models[0][-1].items()}
    for *_, videos in models:
        for g, v in videos.items():
            np.testing.assert_array_equal(v["frame_id"], reference[g])
    return models


def fit_means(videos, fit):
    """Exactly preserve Stage-197 fit-only means, including float32 p reductions."""
    if not fit or len(fit) != len(set(fit)) or any(g not in videos for g in fit):
        raise ValueError("Invalid fit source pool")
    mu_y = np.stack([videos[g]["labels"].mean(0) for g in fit]).mean(0)
    mu_p = np.stack([videos[g]["prediction"].mean(0) for g in fit]).mean(0)
    return np.asarray(mu_y, float), np.asarray(mu_p, float)


def calibration_scores(prediction, mu_y, mu_p):
    p = np.asarray(prediction, float)
    if p.ndim != 2 or p.shape[1] != 6 or np.shape(mu_y) != (6,) or np.shape(mu_p) != (6,):
        raise ValueError("Expected six-layer scores and fit means")
    calibrated = np.asarray(mu_y, float) + p - np.asarray(mu_p, float)
    if not np.isfinite(calibrated).all():
        raise ValueError("Nonfinite calibration")
    return calibrated  # Ranking scores: intentionally no sigmoid or clipping.


def correspondence(y, action):
    y = np.asarray(y, float)
    action = np.asarray(action, int)
    ordinary = metrics.policy_metrics(y, action)
    one_hot = np.eye(6)[action]
    frequency = one_hot.mean(0)
    independent = float(frequency @ y.mean(0))
    covariance = float(((one_hot - frequency) * (y - y.mean(0))).mean(0).sum())
    gain = ordinary["mean_iou"] - independent
    error = abs(gain - covariance)
    if error > 1e-12:
        raise ValueError("State-correspondence covariance identity failed")
    return dict(**ordinary, independent_mean_iou=independent, state_alignment_gain=gain,
                covariance_identity_error=error)


def paired_rows(rows):
    result = []
    for arm in ARMS:
        for pool in ("fit", "held"):
            subset = [r for r in rows if r["arm"] == arm and r["pool"] == pool]
            groups = sorted({r["group"] for r in subset})
            if len(groups) != 38:
                raise ValueError("Incomplete paired source pool")
            def values(policy, metric):
                output = []
                for group in groups:
                    cells = [r[metric] for r in subset if r["group"] == group and r["policy"] == policy]
                    if len(cells) != 2:
                        raise ValueError("Expected two seeds for each source")
                    output.append(float(np.mean(cells)))
                return np.asarray(output)
            for control in ("raw", "raw_redecoded", "fit_label_mean", "fit_prediction_mean", "native_L9"):
                for metric in ("mean_iou", "state_alignment_gain"):
                    delta = values("calibrated", metric) - values(control, metric)
                    result.append(dict(arm=arm, pool=pool, contrast="calibrated_minus_" + control, metric=metric,
                                       favorable_direction="positive", **protocol.contrast(delta)))
            for policy in ("raw", "raw_redecoded", "calibrated"):
                result.append(dict(arm=arm, pool=pool, contrast=policy + "_minus_independent_expectation",
                    metric="mean_iou", favorable_direction="positive", **protocol.contrast(values(policy, "state_alignment_gain"))))
    for pool in ("fit", "held"):
        for metric in ("mean_iou", "state_alignment_gain"):
            groups = sorted({r["group"] for r in rows})
            delta = []
            for g in groups:
                get = lambda arm: [r[metric] for r in rows if r["arm"] == arm and r["pool"] == pool
                                  and r["group"] == g and r["policy"] == "calibrated"]
                a, b = get("ordered"), get("unordered")
                if len(a) != 2 or len(b) != 2:
                    raise ValueError("Incomplete interface comparison")
                delta.append(float(np.mean(a) - np.mean(b)))
            result.append(dict(arm="ordered_vs_unordered", pool=pool, contrast="calibrated_ordered_minus_unordered",
                               metric=metric, favorable_direction="positive", **protocol.contrast(delta)))
    return result


def summarize(rows, category, names, fields):
    result = []
    for arm in ARMS:
        for pool in ("fit", "held"):
            for name in names:
                selected = [r for r in rows if r["arm"] == arm and r["pool"] == pool and r[category] == name]
                groups = sorted({r["group"] for r in selected})
                if len(groups) != 38 or len(selected) != 76:
                    raise ValueError("Incomplete summary grid")
                row = dict(arm=arm, pool=pool, **{category: name}, videos=38)
                for field in fields:
                    group_values = []
                    for group in groups:
                        cells = [r[field] for r in selected if r["group"] == group]
                        if all(v is not None for v in cells):
                            group_values.append(float(np.mean(cells)))
                    row[field] = float(np.mean(group_values)) if group_values else None
                    row[field + "_supported_videos"] = len(group_values)
                result.append(row)
    return result


def analyze(d, tables, progress):
    models = source_models(d, tables)
    rows = {name: [] for name in TABLES}
    old = {(int(r["fold"]), int(r["seed"]), r["arm"], r["group"], r["policy"]): r
           for r in tables["policy_video"]}
    for scope, fold, seed, arm, videos in models:
        mu_y, mu_p = fit_means(videos, fold["fit"])
        initial_rows = sum(len(v["frame_id"]) for v in videos.values())
        scope_base = dict(fold=fold["fold"], seed=seed, arm=arm)
        rows["model_scope"].append(dict(scope_base, fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
            source_model_sha256=scope["final_sha256"], source_draws_sha256=scope["draws_sha256"],
            fit_source_groups=len(fold["fit"]), fit_states=sum(len(videos[g]["frame_id"]) for g in fold["fit"]),
            evaluated_states=initial_rows, computed_column_offsets=6, new_training_updates=0, new_model_forwards=0))
        for k in range(6):
            rows["calibration"].append(dict(scope_base, layer=k + 7, fit_label_mean=float(mu_y[k]),
                fit_prediction_mean=float(mu_p[k]), column_offset=float(mu_y[k] - mu_p[k]),
                fit_source_groups=len(fold["fit"]), fit_only=True))
        for group, video in videos.items():
            y, p, raw = video["labels"], video["prediction"], video["choice"]
            score = calibration_scores(p, mu_y, mu_p)
            base = dict(scope_base, group=group, sequence=video["sequence"], pool="fit" if group in fold["fit"] else "held")
            scores = dict(raw=p, calibrated=score, fit_label_mean=np.broadcast_to(mu_y, y.shape),
                          fit_prediction_mean=np.broadcast_to(mu_p, y.shape))
            choices = dict(raw=raw, raw_redecoded=metrics.choose(p), calibrated=metrics.choose(score),
                           fit_label_mean=metrics.choose(scores["fit_label_mean"]),
                           fit_prediction_mean=metrics.choose(scores["fit_prediction_mean"]),
                           native_L9=np.full(len(y), 2), oracle=metrics.choose(y))
            for policy in POLICIES:
                rows["policy_video"].append(dict(base, policy=policy, states=len(y), **correspondence(y, choices[policy])))
            for predictor, values in scores.items():
                rows["error_video"].append(dict(base, predictor=predictor, score_min=float(values.min()),
                    score_max=float(values.max()), **metrics.errors(y, values)))
            for policy, old_policy in (("raw", "trained"), ("fit_prediction_mean", "fit_prediction_mean")):
                expected = old[fold["fold"], seed, arm, group, old_policy]
                observed = metrics.policy_metrics(y, choices[policy])
                if abs(observed["mean_iou"] - float(expected["mean_iou"])) > 1e-12:
                    raise ValueError("Original source control drifted")
            rows["source_bridge"].append(dict(base, states=len(y),
                raw_mean_iou_error=abs(metrics.policy_metrics(y, raw)["mean_iou"] - float(old[fold["fold"], seed, arm, group, "trained"]["mean_iou"])),
                raw_redecode_action_changes=int((raw != choices["raw_redecoded"]).sum()),
                calibrated_action_changes=int((raw != choices["calibrated"]).sum()),
                source_prediction_constant_reproduced=True))
            frequency = {name: np.bincount(a, minlength=6) / len(a) for name, a in choices.items()}
            for i, frame in enumerate(video["frame_id"]):
                rows["frame_predictions"].append(dict(base, frame=int(frame),
                    **{name + "_layer": int(a[i]) + 7 for name, a in choices.items()},
                    raw_iou=float(y[i, raw[i]]), calibrated_iou=float(y[i, choices["calibrated"][i]]),
                    fit_label_mean_iou=float(y[i, choices["fit_label_mean"][i]]),
                    raw_independent_iou=float(y[i] @ frequency["raw"]),
                    calibrated_independent_iou=float(y[i] @ frequency["calibrated"]),
                    **{f"raw_score_L{k}": float(p[i, k - 7]) for k in range(7, 13)},
                    **{f"calibrated_score_L{k}": float(score[i, k - 7]) for k in range(7, 13)},
                    **{f"label_L{k}": float(y[i, k - 7]) for k in range(7, 13)}))
            progress.advance(count=len(y))
    rows["policy_summary"] = summarize(rows["policy_video"], "policy", POLICIES,
        ("mean_iou", "mean_regret", "good_fraction", "modal_fraction", "entropy_bits", "independent_mean_iou", "state_alignment_gain"))
    rows["error_summary"] = summarize(rows["error_video"], "predictor", PREDICTORS,
        ("mse", "relative_mse", "gap_weighted_pair_accuracy", "native_advantage_mse"))
    rows["paired"] = paired_rows(rows["policy_video"])
    return rows


def verify_results(root):
    d, p = read_sealed(root / "stage198_decision.json"), read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or d["status"] != "COMPLETE_FIT_MEAN_STATE_CORRESPONDENCE_DIAGNOSTIC"
            or d["plan_fingerprint"] != p["fingerprint"] or d["source_stage197"] != SOURCE197
            or d["source_bundle_sha256"] != BUNDLE_SHA256 or d["models_audited"] != 12
            or any(d[k] != 0 for k in ("new_training_runs", "new_training_updates", "new_model_forwards", "new_image_reads", "new_full_trajectories", "confirmation_groups_opened"))
            or set(d["table_sha256"]) != {f"stage198_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-198 result")
    for name, sha in d["table_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Stage-198 table changed: " + name)
    print("Stage-198 verified; COMPLETE denotes a diagnostic, not method adoption", flush=True)
    return d


def export_bundle(root, decision):
    name = "stage198_results_" + decision["fingerprint"][:12] + ".txt"
    path = root / name
    content = bytearray()
    for relative in ("stage198_decision.json", "manifest.json", *(f"stage198_{name}.csv" for name in TABLES)):
        content.extend(("\n===== " + relative + " =====\n").encode())
        content.extend((root / relative).read_bytes())
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Committed result bundle changed")
    if not path.exists():
        path.write_bytes(content)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root, bundle = args.output_dir.expanduser().resolve(), args.source_bundle.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    protocol.safe_output(root, [bundle.parent])
    with writer_lock(root):
        if (root / "stage198_decision.json").exists():
            decision = verify_results(root)
            print("Result bundle: " + str(export_bundle(root, decision)), flush=True)
            return
        d197, tables = load_bundle(bundle)
        plan = sealed(dict(schema=SCHEMA, source_stage197=SOURCE197, source_plan197=PLAN197,
            source_bundle_sha256=BUNDLE_SHA256, arms=list(ARMS), policies=list(POLICIES),
            calibration="s=fit_label_mean+p-fit_prediction_mean; fit source groups equally weighted; no clipping or fitted scale",
            precision="source p float32, fit p means float32 as Stage-197; calibrated scores float64",
            tie_rule="raw archived torch choice retained; new scores retain L9 within 1e-8, else first exact max; raw_redecoded control",
            independent="within-video exact product of action frequencies and layer label means; evaluation only",
            primary="ordered held calibrated-minus-fit-label-mean AND calibrated state-alignment gain",
            bootstrap=protocol.DESIGN, evaluation="fixed native-parent cached immediate IoU; no new AUC or external confirmation",
            source_table_sha256=d197["table_sha256"], python=sys.version, numpy=str(np.__version__),
            protocol_sha256=file_hash(REPO / "docs/experiments/06_sgla_uav_tracking/STAGE198_FIT_MEAN_STATE_CORRESPONDENCE.md"),
            code_sha256={Path(p).name: file_hash(p) for p in (__file__, protocol.__file__, metrics.__file__)}))
        publish_identical(root / "manifest.json", plan)
        progress = OverallProgress("Stage-198", d197["predicted_states"] + 2, unit="states")
        progress.report("Fit-only calibration and exact correspondence expectation")
        progress.advance()
        rows = analyze(d197, tables, progress)
        hashes = {}
        for name in TABLES:
            path = root / f"stage198_{name}.csv"
            write_csv(path, rows[name])
            hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete work accounting")
        primaries = [r for r in rows["paired"] if r["arm"] == "ordered" and r["pool"] == "held" and r["metric"] == "mean_iou"
                     and r["contrast"] in ("calibrated_minus_fit_label_mean", "calibrated_minus_independent_expectation")]
        decision = sealed(dict(schema=SCHEMA, status="COMPLETE_FIT_MEAN_STATE_CORRESPONDENCE_DIAGNOSTIC",
            plan_fingerprint=plan["fingerprint"], source_stage197=SOURCE197, source_bundle_sha256=BUNDLE_SHA256,
            source_groups=38, models_audited=12, evaluated_states=len(rows["frame_predictions"]), calibration_constants=72,
            new_training_runs=0, new_training_updates=0, new_model_forwards=0, new_image_reads=0,
            new_full_trajectories=0, confirmation_groups_opened=0, primary_contrasts=primaries,
            maximum_covariance_identity_error=max(r["covariance_identity_error"] for r in rows["policy_video"]),
            source_table_sha256=d197["table_sha256"], table_sha256=hashes,
            scope="reported score/IoU bundle only; raw PT/features not reverified; nominal exploratory source-group intervals",
            next_step="Review both primaries and two-seed consistency before any full-video validation; no automatic adoption"))
        publish_identical(root / "stage198_decision.json", decision)
        progress.phase = "Complete"
        progress.advance()
        progress.report()
        verify_results(root)
        print("Result bundle: " + str(export_bundle(root, decision)), flush=True)


if __name__ == "__main__":
    main()
