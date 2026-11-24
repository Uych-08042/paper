"""Collect same-state layer intervention labels near archived tracking events."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage174_long_horizon_risk as risk


aligned, matched, previous, online = risk.aligned, risk.matched, risk.previous, risk.online
base, gate, gain, history, c, io = risk.base, risk.gate, risk.gain, risk.history, risk.c, risk.io
SOURCE174 = "eba621e11a0f96763b14af9fe588f18dbec74fea2a99af6f8577d004b7306a17"
SCHEMA = "stage175_event_risk_v1"
SOURCES = (*risk.SOURCES, 174)
ARMS = risk.ARMS
HISTORY, SHORT, LONG = online.DESIGN["history"], 32, 128
GRACE = online.DESIGN["recovery_grace"]
OFFSETS = ((1, 4), (5, 16))
SEED = 23175


def archive_sources(video_rows, failure_rows, cases173, cases174, outcomes174):
    source = matched.COLD
    sizes = {r["sequence"]: int(r["frames"]) for r in video_rows if r["arm"] == source}
    events = defaultdict(list)
    for row in failure_rows:
        if row["arm"] == source:
            events[row["sequence"]].append((int(row["onset"]), int(row["end"]),
                                           int(row["recovery"]) if row["recovery"] else None, row["censor"]))
    for values in events.values():
        values.sort()
    index174 = {r["id"]: r for r in cases174}
    if len(sizes) != online.DESIGN["videos"] or not events or not events.keys() <= sizes.keys():
        raise ValueError("Incomplete Stage-170 video or event roster")
    if len(cases173) != 39 or len(index174) != 106 or len(outcomes174) != 106:
        raise ValueError("Incomplete Stage-173 or Stage-174 replay roster")
    reserved = {(r["sequence"], int(r["frame"])) for r in (*cases173, *cases174)}
    known = {risk.SOURCE_GROUP(r["sequence"]) for r in cases173 if r["cohort"] == "targeted"}
    known.update(index174[r["case_id"]]["group"] for r in outcomes174 if r["severe"] == "True")
    return sizes, events, reserved, known


def route_labels(trace):
    return np.select((trace["switched"], trace["vetoed"], trace["proposal"] != trace["native_layer"]),
                     ("accepted", "cooldown_vetoed", "gate_rejected"), default="native_proposal")


def sample(sizes, events, reserved, known, routes):
    cases, support = [], []
    for seq, references in sorted(events.items()):
        size, source_routes = sizes[seq], routes[seq]
        group = risk.SOURCE_GROUP(seq)
        used = {frame for s, frame in reserved if s == seq}
        local = []
        for onset, _, recovered, _ in references:
            endpoints = [("onset", onset, HISTORY + 1)]
            if recovered is not None:
                endpoints.append(("recovery", recovered, onset + online.TRACKING["failure_run"]))
            for kind, reference, first in endpoints:
                for low, high in OFFSETS:
                    eligible = [f for f in range(reference - low, reference - high - 1, -1)
                                if max(HISTORY + 1, first) <= f <= size - SHORT and f not in used
                                and (kind != "recovery" or reference + GRACE + online.TRACKING["recovery_run"] - 1
                                     <= min(size, f + LONG))]
                    frame = next(iter(eligible), None)
                    support.append(dict(sequence=seq, kind=kind, reference=reference, episode_onset=onset,
                                        offset_min=low, offset_max=high, eligible=len(eligible), selected=frame))
                    if frame is None:
                        continue
                    used.add(frame)
                    local.append(dict(id=f"{seq}__f{frame:06d}", sequence=seq, group=group,
                                      known_source=group in known, source=matched.COLD, kind=kind,
                                      reference=reference, episode_onset=onset, frame=frame, end=min(frame + LONG, size),
                                      route=str(source_routes[frame - 1]), match_id=None))
        ordinary = []
        # Match the longest windows first so shorter controls do not exhaust their support.
        for event in sorted(local, key=lambda r: (r["frame"] - r["end"], r["frame"])):
            window = event["end"] - event["frame"]
            pool = [f for f in range(HISTORY + 1, size - window + 1)
                    if f not in used and source_routes[f - 1] == event["route"]
                    and all(f + window < start or f - HISTORY > end for start, end, _, _ in references)]
            frame = min(pool, key=lambda f: sha256(f"{SEED}/{event['id']}/{f}".encode()).digest(), default=None)
            support.append(dict(sequence=seq, kind="ordinary", reference=event["reference"],
                                episode_onset=event["episode_onset"], offset_min=None, offset_max=None,
                                eligible=len(pool), selected=frame))
            if frame is None:
                continue
            used.add(frame)
            case_id = f"{seq}__f{frame:06d}"
            event["match_id"] = case_id
            ordinary.append(dict(event, id=case_id, kind="ordinary", reference=None, episode_onset=None,
                                 frame=frame, end=frame + window, match_id=event["id"]))
        cases.extend(local + ordinary)
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate Stage-175 anchor")
    return cases, support


def event_labels(prefix, continuations):
    start = len(prefix)
    stop = start + continuations.shape[1] - 1
    histories = [online.failure_events(np.r_[prefix, branch]) for branch in continuations]
    local = [[e for e in trace if e["end"] >= start] for trace in histories]
    native, ranked = local

    def new_events(arm, other):
        return [e for e in arm if e["confirmation"] >= start and not any(
            max(start, e["start"], ref["start"]) <= min(e["end"], ref["end"]) for ref in other)]

    added, prevented = new_events(ranked, native), new_events(native, ranked)
    terminal_onsets = [next((e["start"] + 1 for e in branch if e["censor"] == "video_end" and e["end"] == stop), None)
                       for branch in local]
    terminal = [frame is not None for frame in terminal_onsets]
    prior = next((e for e in reversed(online.failure_events(prefix)) if e["censor"] == "video_end"), None)
    recovery = [None, None]
    opportunity = [False, False]
    if prior is not None:
        # An inherited episode has the same onset in both branches, including recoveries
        # whose three-frame confirmation straddles the anchor.
        confirm_tail = online.TRACKING["recovery_run"] - 1
        for i, trace in enumerate(histories):
            same = next(e for e in trace if e["start"] == prior["start"])
            done = same["recovery"]
            if done is not None:
                recovery[i] = done + 1
                opportunity[i] = done + GRACE + confirm_tail <= stop
    lost = opportunity[0] and (recovery[1] is None or recovery[1] > recovery[0] + GRACE)
    rescued = opportunity[1] and (recovery[0] is None or recovery[0] > recovery[1] + GRACE)
    new_terminal = terminal[1] and not terminal[0]
    return dict(inherited_failure=prior is not None, native_recovery_frame=recovery[0], ranked_recovery_frame=recovery[1],
                native_recovery_opportunity=opportunity[0], ranked_recovery_opportunity=opportunity[1],
                lost_recovery=lost, rescued_recovery=rescued, added_failure=len(added), prevented_failure=len(prevented),
                first_added_failure_frame=added[0]["start"] + 1 if added else None,
                first_prevented_failure_frame=prevented[0]["start"] + 1 if prevented else None,
                native_terminal=terminal[0], ranked_terminal=terminal[1], native_terminal_onset=terminal_onsets[0],
                ranked_terminal_onset=terminal_onsets[1],
                new_terminal=new_terminal, prevented_terminal=terminal[0] and not terminal[1],
                event_harm=new_terminal or lost)


def measure(case, traces, truth, archive_iou, source, exact_runtime):
    native, ranked = (traces[k] for k in ARMS)
    for k in native:
        if k.startswith("anchor_"):
            np.testing.assert_array_equal(native[k], ranked[k])
    index = case["frame"] - 1
    reference, recorded = traces["ranked" if source["switched"][index] else "native"], source["layer"][index]
    box_error = float(np.abs(reference["boxes"][0] - source["boxes"][index]).max())
    gate_error = abs(float(native["anchor_legacy"]) - float(source["gate_value"][index]))
    route_matches = (reference["layer"][0] == recorded and native["anchor_native"] == source["native_layer"][index]
                     and native["anchor_proposal"] == source["proposal"][index]
                     and native["anchor_allowed"] == source["would_switch"][index])
    if exact_runtime and (box_error != 0. or gate_error > 1e-12 or not route_matches):
        raise ValueError(f"Stage-170 source action changed at {case['id']}")
    values = np.stack([online.tracking_iou(t["boxes"], truth) for t in (native, ranked)])
    valid = bool((values >= 0.).all())
    report = dict(case_id=case["id"], valid=valid, window=case["end"] - case["frame"],
                  native_layer=int(native["anchor_native"]), ranked_layer=int(native["anchor_ranked"]),
                  proposal_layer=int(native["anchor_proposal"]), frozen_gain32=float(native["anchor_gain32"]),
                  frozen_risk32=float(native["anchor_risk32"]), source_action_box_error=box_error,
                  source_gate_error=gate_error, source_route_matches=bool(route_matches),
                  delta32=None, delta_window=None, mean_iou_risk=None, missed_by_mean05=None, zero_mean_risk=None)
    if not valid:
        # Preserve a uniform schema even when a video contains invalid target annotations.
        return dict(report, **{k: None for k in event_labels(np.empty(0), np.ones((2, SHORT + 1)))})
    flags = event_labels(archive_iou[:index], values)
    delta = values[1, 1:] - values[0, 1:]
    mean = float(delta.mean())
    report.update(delta32=float(delta[:SHORT].mean()), delta_window=mean, mean_iou_risk=max(0., -mean),
                  missed_by_mean05=flags["event_harm"] and mean > -risk.SEVERE,
                  zero_mean_risk=flags["event_harm"] and mean >= 0.)
    return dict(report, **flags)


def summarize(cases, outcomes, folded):
    by_id = {r["id"]: r for r in cases}
    outcome = {r["case_id"]: r for r in outcomes if r["valid"]}
    pairs, summary = [], []
    for case in cases:
        a, b = case["id"], case["match_id"]
        if case["kind"] != "ordinary" and a in outcome and b in outcome:
            event, control = outcome[a], outcome[b]
            pairs.append(dict(event_id=a, ordinary_id=b, group=case["group"], known_source=case["known_source"],
                              kind=case["kind"], full128=event["window"] == LONG,
                              event_harm=event["event_harm"], ordinary_harm=control["event_harm"],
                              event_zero_mean_risk=event["zero_mean_risk"], ordinary_zero_mean_risk=control["zero_mean_risk"]))
    for known in ("all", "new_source"):
        for length in ("all", "full128"):
            for kind in ("events", "onset", "recovery", "ordinary"):
                keys = [k for k, r in outcome.items() if (known == "all" or not by_id[k]["known_source"])
                        and (length == "all" or r["window"] == LONG)
                        and (by_id[k]["kind"] != "ordinary" if kind == "events" else by_id[k]["kind"] == kind)]
                records = [outcome[k] for k in keys]
                hits = {by_id[k]["group"] for k in keys if outcome[k]["event_harm"]}
                summary.append(dict(cohort=known, length=length, kind=kind, cases=len(keys),
                                    source_groups=len({by_id[k]["group"] for k in keys}), harm_groups=len(hits),
                                    harm_folds=len({folded[g] for g in hits}),
                                    **{k: sum(r[k] for r in records) for k in
                                       ("added_failure", "prevented_failure", "new_terminal", "prevented_terminal",
                                        "native_recovery_opportunity", "lost_recovery", "rescued_recovery",
                                        "event_harm", "missed_by_mean05", "zero_mean_risk")}))
    paired = []
    for cohort in ("all", "new_source"):
        for length in ("all", "full128"):
            by_group = defaultdict(list)
            for row in pairs:
                if (cohort == "all" or not row["known_source"]) and (length == "all" or row["full128"]):
                    by_group[row["group"]].append(int(row["event_harm"]) - int(row["ordinary_harm"]))
            if by_group:
                stats = base.s103.contrast([float(np.mean(v)) for v in by_group.values()], "event_minus_ordinary",
                                           "event_harm_enrichment", online.DESIGN)
                paired.append(dict(cohort=cohort, length=length, pairs=sum(map(len, by_group.values())), **stats))
    return summary, pairs, paired


def read_sources(read):
    return archive_sources(read(170, "video_summary"), read(170, "failures"), read(173, "cases"),
                           read(174, "cases"), read(174, "outcomes"))


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r173, r174 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    roots = {170: r170, 173: r173, 174: r174}
    schemas = {170: (previous.SCHEMA, matched.SOURCE170), 173: (aligned.SCHEMA, risk.SOURCE173),
               174: (risk.SCHEMA, SOURCE174)}
    refs = {s: c.c66.sealed(roots[s] / f"stage{s}_decision.json", schema) for s, (schema, _) in schemas.items()}
    d170 = refs[170]
    if (any(refs[s]["fingerprint"] != fp for s, (_, fp) in schemas.items())
            or d170["plan_fingerprint"] != p170["fingerprint"] or d170["runtime"] != p170["runtime"]
            or any(refs[s]["runtime"] != d170["runtime"] for s in (173, 174))
            or d167["fingerprint"] != previous.SOURCE167):
        raise ValueError("Need the frozen Stage-167, Stage-170, Stage-173 and Stage-174 runs")
    roster, events, reserved, known = read_sources(lambda s, k: risk.read_verified(roots[s], refs[s], s, k))
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if set(roster) != set(jobs) or folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Source video roster or folds differ from Stage-170")
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources, truth, source_iou = {}, {}, {}, {}
    for seq, expected in sorted(events.items()):
        size = roster[seq]
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        if (entry["sequence"] != seq or jobs[seq]["frames"] != size or len(entry["frames"]) != size + 1
                or len(entry["ground_truth"]) != size + 1):
            raise ValueError(f"Source sequence or frame count changed at {seq}")
        data = previous.archive_trace(r170, p170, d170, seq, matched.COLD)
        previous.audit(data, size, entry["initial_box"], matched.COLD, thresholds[fold_of[risk.SOURCE_GROUP(seq)]])
        gt = np.asarray(entry["ground_truth"], float)
        iou = online.tracking_iou(data["boxes"], gt[1:])
        observed = [(e["start"] + 1, e["end"] + 1, None if e["recovery"] is None else e["recovery"] + 1, e["censor"])
                    for e in online.failure_events(iou)]
        if expected != observed:
            raise ValueError(f"Archived event labels changed at {seq}")
        entries[seq], sources[seq], truth[seq], source_iou[seq] = entry, data, gt, iou
    cases, support = sample(roster, events, reserved, known, {k: route_labels(v) for k, v in sources.items()})
    if not cases:
        raise ValueError("No eligible Stage-175 event windows")

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(risk.__file__), Path(aligned.__file__), Path(previous.__file__),
            Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_reports={str(s): refs[s]["fingerprint"] for s in roots},
                        cases=cases, support=support, folds=folds, runtime=runtime, runtime_differences=differences,
                        offsets=OFFSETS, min_successors=SHORT, max_successors=LONG, grace=GRACE, seed=SEED,
                        arms=ARMS, known_source_groups=sorted(known), code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)

    def location(case, arm):
        return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(plan_fingerprint=plan["fingerprint"],
                                                                 case_id=case["id"], arm=arm)

    branches = [(case, arm, case["end"] - case["frame"] + 1) for case in cases for arm in ARMS]
    searches = sum(size for _, _, size in branches)
    reused = sum(size for case, arm, size in branches if location(case, arm)[0].with_suffix(".receipt.json").exists())
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-175", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables, artifacts = dict(cases=cases, support=support, outcomes=[]), {}
    progress.report("Replaying event and ordinary windows")
    with ExitStack() as stack:
        observer = risk.RiskObserver(native, stack)
        for case in cases:
            seq = case["sequence"]
            entry, source = entries[seq], sources[seq]
            model = models[fold_of[case["group"]]]
            traces, template = {}, None

            def read(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for arm in ARMS:
                path, metadata = location(case, arm)
                trace = c.read_cache(path, metadata)
                fresh = trace is None
                if fresh:
                    if template is None:
                        template = read(0)
                    trace = aligned.replay(observer, read, template, entry, case, source, arm, model, progress)
                aligned.audit(trace, case, source, arm)
                if fresh:
                    c.save(path, trace, metadata)
                traces[arm] = trace
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            labels = truth[seq][case["frame"]:case["end"] + 1]
            tables["outcomes"].append(measure(case, traces, labels, source_iou[seq], source, not differences))
    native.assert_frozen()
    tables["summary"], tables["matches"], tables["paired"] = summarize(cases, tables["outcomes"], fold_of)
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-175 progress accounting")
    main = next(r for r in tables["summary"] if (r["cohort"], r["length"], r["kind"]) == ("new_source", "full128", "events"))
    return dict(plan_fingerprint=plan["fingerprint"], cases=len(cases), roles=dict(Counter(r["kind"] for r in cases)),
                valid_cases=sum(r["valid"] for r in tables["outcomes"]), main_event_support=main,
                supports_event_supervision_pilot=main["harm_groups"] >= 3 and main["harm_folds"] == len(folds),
                new_full_trajectories=0, rollouts=len(branches), searches=searches, work=work, runtime=runtime,
                runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage175")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=3, type=Path, metavar=("STAGE170", "STAGE173", "STAGE174"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage175_event_risk")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1] for p, (s, fp) in zip(
            args.bundles, ((170, matched.SOURCE170), (173, risk.SOURCE173), (174, SOURCE174)), strict=True)}
        _, events, reserved, known = read_sources(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        print(f"Verified {sum(map(len, events.values()))} archived episodes across {len(events)} videos, "
              f"{len(reserved)} reserved anchors and {len(known)} known source groups")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage175_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage174"] != SOURCE174 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-175 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-175", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE175_EVENT_RISK",
                                              source_stage174=SOURCE174, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
