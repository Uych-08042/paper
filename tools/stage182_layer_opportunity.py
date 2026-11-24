"""Measure six-layer rescue opportunities on new difficult and stable native histories."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage177_native_proposal_audit as held
import stage181_paired_center_audit as paired


sweep, risk, event, online, previous = held.sweep, held.risk, held.event, held.online, held.previous
base, gate, gain, history, c, io = held.base, held.gate, held.gain, held.history, held.c, held.io
ALIGNED, LAYERS = held.aligned, sweep.LAYERS
SHORT, LONG, HISTORY = held.SHORT, held.LONG, held.HISTORY
SOURCE181 = "a9863249c809b407f0dfd4281a14d8c86248f0a14d72d5397e5cfc96d6bd641a"
SCHEMA = "stage182_layer_opportunity_v1"
SOURCES = (*held.SOURCES, 177, 181)
REFERENCES = {170: (previous.SCHEMA, event.matched.SOURCE170), **held.REFERENCES,
              177: (held.SCHEMA, paired.prior.SOURCE177), 181: (paired.SCHEMA, SOURCE181)}
SEED, MIN_SPACING, MIN_GROUPS = 23182, 32, 3
SIGNALS = ("peak_drop", "center_jump", "size_jump")
NATIVE = "native"


def source_scope(read):
    roster, reserved, _ = held.archive_scope(read)
    native = {r["sequence"]: int(r["frames"]) for r in read(170, "video_summary") if r["arm"] == NATIVE}
    if any(native[s] != n for s, n in roster.items()):
        raise ValueError("Native and cooldown video lengths differ")
    intervals = defaultdict(list)
    cases181 = read(181, "cases")
    ranked177 = {r["case_id"] for r in read(177, "outcomes")}
    held181 = {r["stage177_id"] for r in cases181 if r["archive_stage"] == "177"}
    if (len(cases181) != 61 or len(ranked177) != 35 or held181 != ranked177
            or len({risk.SOURCE_GROUP(s) for s in roster}) != len(roster)):
        raise ValueError("Unexpected source-group or archived continuation roster")
    for row in cases181:
        seq, frame = row["sequence"], int(row["frame"])
        if row["archive_stage"] == "177":
            if seq not in roster or int(row["window"]) != LONG:
                raise ValueError("Stage-177 window differs from the held roster")
            intervals[seq].append((frame - HISTORY, frame + LONG))
        reserved.add((seq, frame))
    return roster, reserved, intervals


def eligible_frames(sequence, frames, reserved, intervals):
    return [f for f in range(HISTORY + 1, frames - LONG + 1) if (sequence, f) not in reserved
            and all(f + LONG < left or f - HISTORY > right for left, right in intervals.get(sequence, ()))]


def causal_features(trace, frames):
    """Use exclusively the committed history through frame f-1."""
    boxes, response = (np.asarray(trace[k], float) for k in ("boxes", "hann_peak"))
    centers = boxes[:, :2] + boxes[:, 2:] / 2.
    moves = np.linalg.norm(np.diff(centers, axis=0) / boxes[:-1, 2:], axis=1)
    size = np.abs(np.diff(np.log(boxes[:, 2:]), axis=0)).max(axis=1)
    values = []
    for frame in frames:
        stop = frame - 1
        recent, earlier = response[stop - 4:stop].mean(), response[stop - HISTORY:stop - 4].mean()
        values.append([max(0., earlier - recent) / max(earlier, .001),
                       moves[stop - 5:stop - 1].max(), size[stop - 5:stop - 1].max()])
    result = np.asarray(values, float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite native history features")
    return result


def sample(roster, traces, reserved, intervals):
    cases, support = [], []
    for sequence, total in sorted(roster.items()):
        group = risk.SOURCE_GROUP(sequence)
        frames = eligible_frames(sequence, total, reserved, intervals)
        row = dict(sequence=sequence, group=group, tracked_frames=total, eligible=len(frames),
                   tail_quota=0, selected=0, separation=None, disjoint=None)
        support.append(row)
        if len(frames) < 4:
            continue
        features = causal_features(traces[sequence], frames)
        scores = np.mean([held.average_ranks(values) for values in features.T], axis=0) / len(frames)
        tie = [sha256(f"{SEED}/{sequence}/{f}".encode()).digest() for f in frames]
        ordered = sorted(range(len(frames)), key=lambda i: (scores[i], tie[i]))
        quota = len(frames) // 4
        row["tail_quota"] = quota
        match = None
        for spacing in (LONG + HISTORY + 1, MIN_SPACING):
            for difficult in reversed(ordered[-quota:]):
                controls = [i for i in ordered[:quota] if scores[i] < scores[difficult]
                            and abs(frames[i] - frames[difficult]) >= spacing]
                if controls:
                    stable = min(controls, key=lambda i: (abs(frames[i] - frames[difficult]), tie[i]))
                    match = difficult, stable
                    break
            if match is not None:
                break
        if match is None:
            continue
        distance = abs(frames[match[0]] - frames[match[1]])
        row.update(selected=2, separation=distance, disjoint=distance > LONG + HISTORY)
        for cohort, j in zip(("difficult", "stable"), match, strict=True):
            frame = frames[j]
            cases.append(dict(id=f"{sequence}__f{frame:06d}", sequence=sequence, group=group,
                              cohort=cohort, frame=frame, end=frame + LONG, source=NATIVE,
                              history_score=float(scores[j]), **dict(zip(SIGNALS, map(float, features[j]), strict=True))))
    return cases, support


def evaluate(case, traces, truth, prefix, archive, exact):
    reference = traces[LAYERS[0]]
    native, ranked = (int(reference[f"anchor_{key}"]) for key in ("native", "ranked"))
    for trace in traces.values():
        for key in reference:
            if key.startswith("anchor_"):
                np.testing.assert_array_equal(trace[key], reference[key])
    start = case["frame"] - 1
    control = traces[native]
    errors = {f"native_{k}_error": float(np.abs(control[k] - archive[k][start:start + LONG + 1]).max())
              for k in ("boxes", "hann_peak", "layer")}
    if exact and any(errors.values()):
        raise ValueError(f"Stage-170 native replay changed at {case['id']}: {errors}")
    ious = {layer: online.tracking_iou(traces[layer]["boxes"], truth) for layer in LAYERS}
    valid = all((v >= 0.).all() for v in ious.values())
    empty = dict.fromkeys(event.event_labels(np.empty(0), np.ones((2, SHORT + 1))))
    rows = []
    for layer, values in ious.items():
        row = dict(case_id=case["id"], layer=layer, is_native=layer == native, is_ranked=layer == ranked)
        if valid:
            delta = values[1:] - ious[native][1:]
            short, long = float(delta[:SHORT].mean()), float(delta.mean())
            flags = event.event_labels(prefix, np.stack((ious[native], values)))
            protected = not flags["event_harm"] and flags["added_failure"] == 0 and min(short, long) >= -sweep.EPSILON
            rescue = protected and (flags["prevented_terminal"] or flags["rescued_recovery"])
            row.update(anchor_iou=float(values[0]), delta32=short, delta_window=long, protected=protected,
                       protected_rescue=rescue, protected_benefit=protected and (long >= sweep.EPSILON or rescue), **flags)
        else:
            row.update(anchor_iou=None, delta32=None, delta_window=None, protected=None,
                       protected_rescue=None, protected_benefit=None, **empty)
        rows.append(row)
    outcome = dict(case_id=case["id"], group=case["group"], cohort=case["cohort"], valid=valid, **errors)
    outcome.update(sweep.headroom(case, rows) if valid else dict(native_layer=native, ranked_layer=ranked))
    native_row = next(row for row in rows if row["is_native"])
    ranked_row = next(row for row in rows if row["is_ranked"])
    recent = prefix[-HISTORY:]
    outcome.update(ranked_rescue=ranked_row["protected_rescue"], ranked_benefit=ranked_row["protected_benefit"],
                   native_terminal=native_row["native_terminal"], native_inherited_failure=native_row["inherited_failure"],
                   native_history_mean_iou=float(recent.mean()) if (recent >= 0.).all() else None,
                   native_future_mean_iou=float(ious[native][1:].mean()) if valid else None)
    return outcome, rows


def summarize(cases, outcomes, layers):
    paired_valid = {r["group"] for r in cases} - {r["group"] for r in outcomes if not r["valid"]}
    rows, by_layer = [], []
    scores = {r["case_id"]: r for r in outcomes}
    for cohort in ("difficult", "stable"):
        sampled = [r for r in cases if r["cohort"] == cohort]
        kept = [scores[r["id"]] for r in sampled if r["group"] in paired_valid]
        ids = {r["case_id"] for r in kept}
        rescue = sum(r["protected_rescue_choices"] > 0 for r in kept)
        benefits = sum(r["improve_native_with_protection"] for r in kept)
        picked = sum(r["ranked_rescue"] for r in kept)
        staged = [r for r in layers if r["case_id"] in ids and not r["is_native"]]
        rows.append(dict(cohort=cohort, sampled=len(sampled), paired_valid=len(kept), rescue_opportunities=rescue,
                         ranked_rescues=picked, rescue_capture=picked / rescue if rescue else None,
                         uniform_rescue_capture=sum(r["protected_rescue"] for r in staged) / (rescue * (len(LAYERS) - 1))
                         if rescue else None, native_terminal_windows=sum(r["native_terminal"] for r in kept),
                         native_inherited_failure_windows=sum(r["native_inherited_failure"] for r in kept),
                         benefit_opportunities=benefits, ranked_benefits=sum(r["ranked_benefit"] for r in kept),
                         harm_opportunities=len({r["case_id"] for r in staged if r["event_harm"]}),
                         ranked_harm=sum(r["ranked_harm"] for r in kept),
                         protected_oracle_mean_gain=float(np.mean([r["protected_oracle_gain"] for r in kept])) if kept else None,
                         ranked_mean_gain=float(np.mean([r["ranked_gain"] for r in kept])) if kept else None))
        for layer in LAYERS:
            arms = [r for r in staged if r["layer"] == layer]
            by_layer.append(dict(cohort=cohort, layer=layer, comparisons=len(arms),
                                 rescues=sum(r["protected_rescue"] for r in arms), harms=sum(r["event_harm"] for r in arms),
                                 benefits=sum(r["protected_benefit"] for r in arms),
                                 mean_delta128=float(np.mean([r["delta_window"] for r in arms])) if arms else None))
    return rows, by_layer


def run(args, root, paths):
    roots = dict(zip(SOURCES, paths, strict=True))
    r76, r89, r93, r121, r159, r165, r166, r167, r170 = (roots[s] for s in (76, 89, 93, 121, 159, 165, 166, 167, 170))
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    refs = {s: c.c66.sealed(roots[s] / f"stage{s}_decision.json", schema) for s, (schema, _) in REFERENCES.items()}
    d170, d181 = refs[170], refs[181]
    if (p170["fingerprint"] != d170["plan_fingerprint"] or p170["runtime"] != d170["runtime"]
            or d167["fingerprint"] != previous.SOURCE167 or d170["source_stage167"] != d167["fingerprint"]
            or d181["source_stage180"] != paired.SOURCE180
            or not d181["association_same_runtime"]
            or any(refs[s]["fingerprint"] != fp or refs[s]["runtime"] != d170["runtime"] for s, (_, fp) in REFERENCES.items())):
        raise ValueError("Need the sealed Stage-167/170/173–177/181 source runs")

    roster, reserved, intervals = source_scope(lambda s, k: risk.read_verified(roots[s], refs[s], s, k))
    p159, points, gains, ranks, held_scores = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Source groups or tracker roster changed")
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    traces = {}
    for seq, count in roster.items():
        trace = previous.archive_trace(r170, p170, d170, seq, NATIVE)
        threshold = thresholds[fold_of[risk.SOURCE_GROUP(seq)]]
        previous.audit(trace, count, trace["parent"][0], NATIVE, threshold)
        traces[seq] = trace
    cases, support = sample(roster, traces, reserved, intervals)
    if not cases:
        raise ValueError("No matched difficult/stable windows remain after the archive exclusions")

    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries = {}
    for seq in sorted({r["sequence"] for r in cases}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        if (entry["sequence"] != seq or len(entry["frames"]) != roster[seq] + 1
                or len(entry["ground_truth"]) != roster[seq] + 1):
            raise ValueError(f"Source image or annotation roster changed for {seq}")
        np.testing.assert_array_equal(entry["initial_box"], traces[seq]["parent"][0])
        entries[seq] = entry

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d170["runtime"], runtime)
    code = (Path(__file__), Path(sweep.__file__), Path(event.__file__), Path(ALIGNED.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, sources={str(s): refs[s]["fingerprint"] for s in refs},
                        cases=cases, support=support, folds=folds, layers=list(LAYERS), source_history=NATIVE,
                        history=HISTORY, horizon=LONG, seed=SEED, signals=list(SIGNALS), min_spacing=MIN_SPACING,
                        minimum_rescue_groups=MIN_GROUPS, protection_tolerance=sweep.EPSILON, runtime=runtime,
                        runtime_differences=differences, code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)

    def location(case, layer):
        return root / "rollouts" / f"{case['id']}__l{layer:02d}.npz", dict(plan_fingerprint=plan["fingerprint"],
                                                                          case_id=case["id"], layer=layer)

    branches = len(cases) * len(LAYERS)
    searches = branches * (LONG + 1)
    resumed = (LONG + 1) * sum(location(case, layer)[0].with_suffix(".receipt.json").exists()
                                for case in cases for layer in LAYERS)
    model_work = len(points) + sum(len(f["fit"]) + 1 for f in folds)
    progress = OverallProgress("Stage-182", searches + model_work + 1, resumed, args.progress_interval, "work")
    progress.report("Restoring frozen Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held_scores, progress)
    for row, archived in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in archived], list(archived.values()), atol=1e-10, rtol=0)
        models[archived["fold"]]["threshold"] = archived["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                  runtime_identity=runtime, runtime=runtime["tracker"]), root)
    outcomes, layer_rows, artifacts = [], [], {}
    progress.report("Evaluating six layers on matched native histories")
    with ExitStack() as stack:
        observer = sweep.SweepObserver(tracker, stack)
        for case in cases:
            seq, start, end = case["sequence"], case["frame"], case["end"]
            entry, archive = entries[seq], traces[seq]
            model = models[fold_of[case["group"]]]

            def read_frame(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            branches_by_layer, template = {}, None
            for layer in LAYERS:
                path, metadata = location(case, layer)
                data = c.read_cache(path, metadata)
                fresh = data is None
                if fresh:
                    if template is None:
                        template = read_frame(0)
                    observer.forced_layer = layer
                    data = ALIGNED.replay(observer, read_frame, template, entry, case, archive, NATIVE, model, progress)
                ALIGNED.audit(dict(data, anchor_probe=np.asarray(layer)), case, archive, "probe")
                if fresh:
                    c.save(path, data, metadata)
                branches_by_layer[layer] = data
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            truth = np.asarray(entry["ground_truth"], float)
            prefix_iou = online.tracking_iou(archive["boxes"][:start - 1], truth[1:start])
            outcome, measured = evaluate(case, branches_by_layer, truth[start:end + 1], prefix_iou, archive, not differences)
            outcomes.append(outcome)
            layer_rows.extend(measured)
    tracker.assert_frozen()
    summary, by_layer = summarize(cases, outcomes, layer_rows)
    columns = dict.fromkeys(key for row in outcomes for key in row)
    tables = dict(cases=cases, support=support, outcomes=[dict(columns, **row) for row in outcomes],
                  layers=layer_rows, summary=summary, by_layer=by_layer)
    if progress.done != progress.total - 1:
        raise ValueError("Incomplete Stage-182 progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], source_stage181=SOURCE181, sampled_groups=len(cases) // 2,
                windows=len(cases), layer_branches=branches, tracking_calls=searches, horizon=LONG,
                annotated_windows=sum(r["valid"] for r in outcomes), replay_matches_source_runtime=not differences,
                runtime=runtime, runtime_differences=differences, main_summary=summary,
                opportunity_for_held_signal_validation=summary[0]["rescue_opportunities"] >= MIN_GROUPS,
                work=progress.total, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage182")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=len(REFERENCES), type=Path,
                        metavar=tuple(f"STAGE{s}" for s in REFERENCES))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage182_layer_opportunity")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for s in SOURCES if (p := getattr(args, f"stage{s}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU check of the source roster")
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1]
                  for p, (s, (_, fp)) in zip(args.bundles, REFERENCES.items(), strict=True)}
        roster, reserved, intervals = source_scope(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        capable = sum(len(eligible_frames(seq, count, reserved, intervals)) >= 4 for seq, count in roster.items())
        print(f"Verified {len(roster)} Stage-176-unused source groups, {capable} with enough frames before history matching, "
              f"{sum(map(len, intervals.values()))} excluded Stage-177 replay windows")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage182_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage181"] != SOURCE181 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-182 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-182", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE182_LAYER_OPPORTUNITY_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
