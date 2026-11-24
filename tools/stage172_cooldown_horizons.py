"""Test action and cooldown effects across horizons on non-targeted Stage-170 states."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage171_matched_cooldown as matched


previous, online, base, gate = matched.previous, matched.online, matched.base, matched.gate
gain, history, c, io = matched.gain, matched.history, matched.c, matched.io
COLD, ARMS = matched.COLD, matched.ARMS
SOURCE171 = "57fe81c845a69c331e20a8ac071dd2242fa647895af4d42817fdf73a2aa1bcdd"
SCHEMA = "stage172_cooldown_horizons_v1"
SOURCES = (*matched.SOURCES, 171)
EXCLUDED = {row[0] for row in matched.FOCUS}
HORIZONS = (32, 64, 128)
SEED = 23172
EPSILON = 0.01
STRATA = ("accepted", "vetoed")
EFFECTS = ("switch_when_open", "switch_when_cooling", "cooldown_after_keep", "cooldown_after_switch", "interaction")


def source_support(rows):
    cold = {r["sequence"]: r for r in rows if r["arm"] == COLD}
    if len(cold) != online.DESIGN["videos"] or not EXCLUDED <= cold.keys():
        raise ValueError("Need the complete Stage-170 development roster")
    return [dict(sequence=seq, frames=int(row["frames"]), accepted_total=int(row["switches"]),
                 vetoed_total=int(row["vetoes"]))
            for seq, row in sorted(cold.items()) if seq not in EXCLUDED]


def sample(roster, sources):
    cases, support = [], []
    for row in roster:
        sequence, frames = row["sequence"], row["frames"]
        data = sources[sequence, COLD]
        for stratum, field in zip(STRATA, ("switched", "vetoed"), strict=True):
            hits = np.flatnonzero(data[field]) + 1
            if len(hits) != row[stratum + "_total"]:
                raise ValueError(f"Stage-170 routing count differs for {sequence} / {stratum}")
            valid = hits[hits + HORIZONS[-1] <= frames]
            frame = min(map(int, valid), key=lambda f: sha256(f"{SEED}/{sequence}/{stratum}/{f}".encode()).digest(),
                        default=None)
            support.append(dict(row, stratum=stratum, full_horizon_opportunities=len(valid), selected=frame))
            if frame is not None:
                prior = np.flatnonzero(data["switched"][:frame - 1])
                cases.append(dict(id=f"{sequence}__{stratum}", sequence=sequence, purpose=stratum, source=COLD,
                                  reference=frame, frame=frame, end=frame + HORIZONS[-1],
                                  last_switch=int(prior[-1] + 1) if prior.size else None))
    return cases, support


def direction(value):
    return int(value >= EPSILON) - int(value <= -EPSILON)


def summaries(cases, effects):
    groups = {r["id"]: (r["sequence"], r["purpose"]) for r in cases}
    values = defaultdict(dict)
    for row in effects:
        horizon = row["horizon"]
        if horizon in HORIZONS and row["valid_successors"] == horizon:
            values[row["case_id"]][horizon] = row
    # A common cohort across all horizons isolates horizon length from label availability.
    complete = {key for key, rows in values.items() if set(rows) == set(HORIZONS)}
    pooled, flips = [], []
    for stratum in ("all", *STRATA):
        keys = sorted(key for key in complete if stratum == "all" or groups[key][1] == stratum)
        for metric in EFFECTS:
            for horizon in HORIZONS:
                by_video = defaultdict(list)
                for key in keys:
                    by_video[groups[key][0]].append(values[key][horizon][metric])
                per_video = [float(np.mean(v)) for v in by_video.values()]
                if per_video:
                    stat = base.s103.contrast(per_video, f"{stratum}_{horizon}_{metric}", metric, online.DESIGN)
                    pooled.append(dict(stratum=stratum, horizon=horizon, anchors=len(keys), **stat))
            for short, long in ((32, 64), (32, 128), (64, 128)):
                pairs = [(direction(values[key][short][metric]), direction(values[key][long][metric])) for key in keys]
                flips.append(dict(stratum=stratum, metric=metric, short=short, long=long, anchors=len(pairs),
                                  both_decisive=sum(a != 0 and b != 0 for a, b in pairs),
                                  opposite=sum(a * b == -1 for a, b in pairs),
                                  short_neutral_long_decisive=sum(a == 0 and b != 0 for a, b in pairs),
                                  short_decisive_long_neutral=sum(a != 0 and b == 0 for a, b in pairs)))
    return sorted(complete), pooled, flips


def anchor_signals(case, source, traces):
    frame = case["frame"]
    online_row = traces["switch_open"]
    signals = history.features_from_history(source["boxes"], source["hann_peak"], frame,
                                            online.DESIGN["history"], True)
    names = ("peak_last", "peak_mean", "peak_std", "peak_change", "shift_last", "shift_mean",
             "scale_last", "scale_mean", "peak_falling_run", "shift_from_median", "scale_std")
    candidate = int(online_row["anchor_choice"][0])
    proposal = int(online_row["proposal"][0])
    return dict(case_id=case["id"], stratum=case["purpose"], native=int(online_row["native_layer"][0]),
                proposal=proposal, candidate=candidate, proposal_matches_candidate=proposal == candidate,
                gate_value=float(online_row["gate_value"][0]),
                frames_since_switch=frame - case["last_switch"] if case["last_switch"] is not None else None,
                **{k: float(v) for k, v in zip(names, signals, strict=True)})


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r171 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    d171 = c.c66.sealed(r171 / "stage171_decision.json", matched.SCHEMA)
    if (d170["fingerprint"] != matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or p170["runtime"] != d170["runtime"] or d167["fingerprint"] != previous.SOURCE167
            or d171["fingerprint"] != SOURCE171 or d171["source_stage170"] != matched.SOURCE170):
        raise ValueError("Need the completed Stage-170 and Stage-171 and their original Stage-167 calibration")
    name = "stage170_video_summary.csv"
    io.verify({str(r170 / name): d170["table_sha256"][name]})
    roster = source_support(gain.prior.prior.read_table(r170 / name))
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"] or not {r["sequence"] for r in roster} <= jobs.keys():
        raise ValueError("Archived source groups differ")
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for row in roster:
        sequence, frames = row["sequence"], row["frames"]
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[sequence]["spec"])[0]
        if (entry["sequence"] != sequence or jobs[sequence]["frames"] != frames
                or len(entry["frames"]) != frames + 1 or len(entry["ground_truth"]) != frames + 1):
            raise ValueError(f"Video roster differs for {sequence}")
        entries[sequence] = entry
        source = previous.archive_trace(r170, p170, d170, sequence, COLD)
        previous.audit(source, frames, entry["initial_box"], COLD, thresholds[fold_of[sequence.split("_", 1)[0]]])
        sources[sequence, COLD] = source
    cases, support = sample(roster, sources)
    if not cases:
        raise ValueError("No complete online-selected anchor windows")
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(matched.__file__), Path(previous.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage171=SOURCE171, source_stage170=matched.SOURCE170,
                        cases=cases, support=support, seed=SEED, horizons=list(HORIZONS), epsilon=EPSILON, arms=list(ARMS),
                        runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)
    def location(case, arm):
        return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(
            plan_fingerprint=plan["fingerprint"], case_id=case["id"], arm=arm)
    per_rollout = HORIZONS[-1] + 1
    searches = len(ARMS) * len(cases) * per_rollout
    reused = per_rollout * sum(location(case, arm)[0].with_suffix(".receipt.json").exists() for case in cases for arm in ARMS)
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-172", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 models")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables = dict(support=support, cases=cases, anchors=[], replays=[], branches=[], frames=[], effects=[])
    artifacts = {}
    progress.report("Running 128-frame matched continuations")
    with ExitStack() as stack:
        observer = matched.MatchedObserver(native, stack)
        for case in cases:
            sequence, anchor, end = case["sequence"], case["frame"], case["end"]
            entry, source = entries[sequence], sources[sequence, COLD]
            model = models[fold_of[sequence.split("_", 1)[0]]]
            traces, template = {}, None

            def image_at(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for arm in ARMS:
                path, meta = location(case, arm)
                data = c.read_cache(path, meta)
                fresh = data is None
                if fresh:
                    if template is None:
                        template = image_at(0)
                    data = matched.replay(observer, image_at, template, entry, case, source, arm, model, progress)
                matched.audit(data, case, source, arm, model["threshold"])
                if fresh:
                    c.save(path, data, meta)
                traces[arm] = data
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            tables["anchors"].append(anchor_signals(case, source, traces))
            report = matched.compare(case, traces, np.asarray(entry["ground_truth"], float)[anchor:end + 1],
                                     sources, not differences)
            for k, rows in report.items():
                tables[k].extend(r for r in rows if k != "effects" or r["horizon"] in HORIZONS)
    native.assert_frozen()
    common, by_video, changes = summaries(cases, tables["effects"])
    tables.update(paired=by_video, horizon_changes=changes)
    if progress.done != work - 1:
        raise ValueError("Incomplete progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], work=work, cases=len(cases), common_horizon_cases=len(common),
                case_videos=len({r["sequence"] for r in cases}), rollouts=len(tables["branches"]), searches=searches,
                runtime=runtime, runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage172")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage172_cooldown_horizons")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU source check")
        _, tables = gain.prior.archives.bundle(args.bundle, 170, matched.SOURCE170)
        roster = source_support(tables["stage170_video_summary.csv"])
        for stratum in STRATA:
            count = sum(row[stratum + "_total"] > 0 for row in roster)
            print(f"{stratum}: {count}/{len(roster)} videos have proposals before applying the 128-frame tail requirement")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all server roots, the original checkpoint and an output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        decision = root / "stage172_decision.json"
        if decision.exists():
            saved = c.c66.sealed(decision, SCHEMA)
            if saved["source_stage171"] != SOURCE171:
                raise ValueError("Stage-172 run uses another source")
            io.verify({str(root / k): h for k, h in (saved["table_sha256"] | saved["artifact_sha256"]).items()})
            OverallProgress("Stage-172", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(decision, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE172_COOLDOWN_HORIZONS",
                                                  source_stage171=SOURCE171, source_stage170=matched.SOURCE170,
                                                  new_full_trajectories=0, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
