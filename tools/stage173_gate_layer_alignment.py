"""Measure the originally gated and actually ranked layers on identical tracking states."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage172_cooldown_horizons as windows


matched, previous, online = windows.matched, windows.previous, windows.online
base, gate, gain, history, c, io = windows.base, windows.gate, windows.gain, windows.history, windows.c, windows.io
SOURCE172 = "ea4490ac7845a89fbfbdfa140d0495f24b34e333308c7371266542ba760603ab"
SCHEMA = "stage173_gate_layer_alignment_v1"
SOURCES = (*windows.SOURCES, 172)
HORIZONS = windows.HORIZONS
ARMS = ("native", "proposal", "ranked")
RULES = ("proposal_gate_proposal_action", "proposal_gate_ranked_action", "ranked_gate_ranked_action")


def bind_cases(cases171, frames171, cases172, anchors172):
    prior = {(r["case_id"], int(r["frame"])): r for r in frames171 if r["arm"] == "switch_open"}
    sampled = {r["case_id"]: r for r in anchors172}
    if len(cases171) != 6 or len(cases172) != 33 or len(sampled) != 33:
        raise ValueError("Incomplete Stage-171 or Stage-172 anchors")
    cases = []
    for cohort, group in (("sampled", cases172), ("targeted", cases171)):
        for row in group:
            frame = int(row["frame"])
            if cohort == "sampled":
                expected = sampled[row["id"]]
                actual = expected["candidate"]
            else:
                expected = prior[row["id"], frame]
                actual = expected["layer"]
            cases.append(dict(id=row["id"], sequence=row["sequence"], source=row["source"],
                              stratum=row["purpose"], cohort=cohort, frame=frame, end=frame + HORIZONS[-1],
                              expected_native=int(expected["native"]), expected_proposal=int(expected["proposal"]),
                              expected_ranked=int(actual), expected_gate_value=float(expected["gate_value"])))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate alignment anchors")
    return cases


def anchor_model(models, native, short, long):
    proposal, value, allowed, ranked = gate.online_decision(models, native, short, long, True)
    if not allowed:
        raise ValueError("Source anchor no longer passes the frozen proposal gate")
    no = online.LAYERS.index(native)
    predicted = gate.score(models["gain"], short)
    downside = np.maximum(0., gate.score(models["risk"], long))
    pos = [online.LAYERS.index(layer) for layer in (proposal, ranked)]
    benefits = predicted[pos] - predicted[no]
    risks = downside[pos]
    scores = benefits - gate.risk.PENALTY * risks
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite matched-layer scores")
    np.testing.assert_allclose(scores[0], value, rtol=0, atol=1e-12)
    return dict(native=native, proposal=proposal, ranked=ranked, threshold=float(models["threshold"]),
                gains=benefits, risks=risks, scores=scores)


class AlignmentObserver(online.RoutedObserver):
    def route(self, module, args, output):
        if self.arm == "native":
            return super().route(module, args, output)
        original = self.rt.selected_layer(output)
        if self.tokens is None:
            raise ValueError("Missing anchor feature or tracking history")
        boxes, peaks = np.asarray(self.boxes), np.asarray(self.peaks)
        index = len(boxes) + 1
        visual = online.trained.projected_prefix(self.rt, self.tokens)
        short = np.r_[visual, online.trained.history_features(boxes, peaks, index)]
        long = np.r_[visual, history.features_from_history(boxes, peaks, index, online.DESIGN["history"], True)]
        self.diagnostic = anchor_model(self.model, original, short, long)
        self.original, self.proposal = original, self.diagnostic["proposal"]
        self.value = float(self.diagnostic["scores"][0])
        self.chosen = self.diagnostic[self.action]
        self.switched = self.chosen != original
        if self.switched:
            forced = self.rt.torch.zeros_like(output)
            forced[:, online.LAYERS.index(self.chosen)] = 1
            return forced
        return None


def replay(observer, read, template, entry, case, source, arm, model, progress):
    anchor, end = case["frame"], case["end"]
    observer.initialize(template, entry["initial_box"], matched.OPEN, model)
    observer.action = arm
    start = max(0, anchor - 1 - online.DESIGN["history"])
    observer.boxes.extend(source["boxes"][start:anchor - 1])
    observer.peaks.extend(source["hann_peak"][start:anchor - 1])
    observer.rt.tracker.frame_id = anchor - 1
    parent, rows = source["parent"][anchor - 1], []
    for frame in range(anchor, end + 1):
        if frame > anchor:
            observer.arm = "native"
        row = observer.step(read(frame), parent, frame)
        rows.append(row)
        parent = row["boxes"]
        progress.advance()
    return {**{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
            **{f"anchor_{k}": np.asarray(v) for k, v in observer.diagnostic.items()}}


def audit(trace, case, source, arm):
    axis = np.arange(case["frame"], case["end"] + 1)
    np.testing.assert_array_equal(trace["frame_id"], axis)
    boxes = trace["boxes"]
    if boxes.shape != (len(axis), 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Invalid branch prediction boxes")
    np.testing.assert_array_equal(trace["parent"], np.vstack((source["parent"][axis[0] - 1], boxes[:-1])))
    layers = trace["layer"]
    if not np.isin(layers, online.LAYERS).all():
        raise ValueError("Invalid layer")
    np.testing.assert_equal(layers[0], trace[f"anchor_{arm}"])
    np.testing.assert_array_equal(layers[1:], trace["native_layer"][1:])
    np.testing.assert_array_equal(trace["switched"], layers != trace["native_layer"])
    blocks = np.zeros((len(axis), 12), int)
    blocks[:, :6] = 1
    blocks[np.arange(len(axis)), layers - 1] = 1
    np.testing.assert_array_equal(trace["block_calls"], blocks)
    for k in ("crop_calls", "head_calls", "selector_calls"):
        np.testing.assert_array_equal(trace[k], np.ones(len(axis), int))


def report(case, traces, truth, exact_runtime, source):
    ref = traces["native"]
    for arm in ARMS[1:]:
        for k in ("parent", "gate_value", "proposal", "native_layer"):
            np.testing.assert_array_equal(ref[k][0], traces[arm][k][0])
        for k in ref:
            if k.startswith("anchor_"):
                np.testing.assert_array_equal(ref[k], traces[arm][k])
    same_layers = int(ref["anchor_proposal"]) == int(ref["anchor_ranked"])
    if same_layers:
        for k in ("boxes", "layer", "hann_peak"):
            np.testing.assert_array_equal(traces["proposal"][k], traces["ranked"][k])
    observed = (int(ref["anchor_native"]), int(ref["anchor_proposal"]), int(ref["anchor_ranked"]))
    archived = tuple(case["expected_" + k] for k in ARMS)
    score_error = abs(float(ref["anchor_scores"][0]) - case["expected_gate_value"])
    if exact_runtime and (observed != archived or score_error > 1e-12):
        raise ValueError(f"Anchor decision changed at {case['id']}")
    index = case["frame"] - 1
    natural = [t for t in traces.values() if int(t["layer"][0]) == int(source["layer"][index])]
    box_error = max((float(np.max(np.abs(t["boxes"][0] - source["boxes"][index]))) for t in natural), default=None)
    if exact_runtime and box_error != 0.:
        raise ValueError(f"Stage-170 anchor box changed at {case['id']}")
    anchor = dict(case_id=case["id"], native=observed[0], proposal=observed[1], ranked=observed[2],
                  same_layers=same_layers, exact_source_layers=observed == archived, source_gate_error=score_error,
                  source_action_box_error=box_error,
                  proposal_gain=float(ref["anchor_gains"][0]), ranked_gain=float(ref["anchor_gains"][1]),
                  proposal_risk=float(ref["anchor_risks"][0]), ranked_risk=float(ref["anchor_risks"][1]),
                  proposal_score=float(ref["anchor_scores"][0]), ranked_score=float(ref["anchor_scores"][1]),
                  threshold=float(ref["anchor_threshold"]))
    outcomes = {arm: online.tracking_iou(trace["boxes"], truth) for arm, trace in traces.items()}
    tail = np.array([outcomes[arm][1:] for arm in ARMS])
    rows, events, decisions = [], [], []
    proposed_pass, ranked_pass = ref["anchor_scores"] > ref["anchor_threshold"] + gate.veto.EPSILON
    for horizon in HORIZONS:
        valid = (tail[:, :horizon] >= 0).all(axis=0)
        if not valid.any():
            continue
        native, proposed, ranked = tail[:, :horizon][:, valid].mean(axis=1)
        delta_p, delta_r = float(proposed - native), float(ranked - native)
        downside_p, downside_r = max(0., -delta_p), max(0., -delta_r)
        rows.append(dict(case_id=case["id"], horizon=horizon, valid_successors=int(valid.sum()),
                         proposal_delta=delta_p, ranked_delta=delta_r, ranked_minus_proposal=delta_r - delta_p,
                         observed_proposal_downside=downside_p, observed_ranked_downside=downside_r,
                         ranked_own_risk_error=abs(anchor["ranked_risk"] - downside_r) if horizon == 32 else None,
                         ranked_proposal_risk_error=abs(anchor["proposal_risk"] - downside_r) if horizon == 32 else None))
        for rule, arm, passed, effect in zip(RULES, ("proposal", "ranked", "ranked"),
                (proposed_pass, proposed_pass, ranked_pass), (delta_p, delta_r, delta_r), strict=True):
            decisions.append(dict(case_id=case["id"], horizon=horizon, valid_successors=int(valid.sum()),
                                  rule=rule, executed=arm if passed else "native", gain=effect if passed else 0.,
                                  blocked_positive=bool(not passed and effect >= windows.EPSILON),
                                  accepted_negative=bool(passed and effect <= -windows.EPSILON)))
    for arm, values in outcomes.items():
        found = online.failure_events(values)
        events.append(dict(case_id=case["id"], arm=arm, anchor_iou=float(values[0]), failures=len(found),
                           unrecovered=sum(e["recovery"] is None for e in found),
                           first_failure=case["frame"] + found[0]["start"] if found else None))
    return anchor, rows, decisions, events


def summarize(cases, anchors, outcomes, decisions):
    case_of = {r["id"]: r for r in cases}
    anchor_of = {r["case_id"]: r for r in anchors}
    complete = defaultdict(set)
    for row in outcomes:
        if row["horizon"] == row["valid_successors"]:
            complete[row["case_id"]].add(row["horizon"])
    result, comparisons = [], []
    for cohort in ("sampled", "sampled_mismatch", "targeted"):
        for horizon in HORIZONS:
            rules = {}
            for rule in RULES:
                scored = [r for r in decisions if r["horizon"] == horizon and r["rule"] == rule
                          and complete[r["case_id"]] == set(HORIZONS)
                          and case_of[r["case_id"]]["cohort"] == cohort.replace("_mismatch", "")
                          and (cohort != "sampled_mismatch" or not anchor_of[r["case_id"]]["same_layers"])]
                rules[rule] = {r["case_id"]: r["gain"] for r in scored}
                by_video = defaultdict(list)
                for row in scored:
                    by_video[case_of[row["case_id"]]["sequence"]].append(row["gain"])
                means = [float(np.mean(x)) for x in by_video.values()]
                if means:
                    stats = base.s103.contrast(means, rule, "iou", online.DESIGN)
                    result.append(dict(cohort=cohort, horizon=horizon, rule=rule, cases=len(scored),
                                       switches=sum(r["executed"] != "native" for r in scored),
                                       blocked_positive=sum(r["blocked_positive"] for r in scored),
                                       accepted_negative=sum(r["accepted_negative"] for r in scored), **stats))
            baseline = rules["proposal_gate_ranked_action"]
            for rule in ("proposal_gate_proposal_action", "ranked_gate_ranked_action"):
                candidate = rules[rule]
                if candidate.keys() != baseline.keys():
                    raise ValueError("Unmatched rule outcomes")
                by_video = defaultdict(list)
                for key, value in candidate.items():
                    by_video[case_of[key]["sequence"]].append(value - baseline[key])
                if by_video:
                    stat = base.s103.contrast([float(np.mean(v)) for v in by_video.values()],
                                              rule + "_minus_proposal_gate_ranked_action", "iou", online.DESIGN)
                    comparisons.append(dict(cohort=cohort, horizon=horizon, cases=len(candidate), **stat))
    return result, comparisons


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r171, r172 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    d171 = c.c66.sealed(r171 / "stage171_decision.json", matched.SCHEMA)
    d172 = c.c66.sealed(r172 / "stage172_decision.json", windows.SCHEMA)
    if (d170["fingerprint"] != matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or d170["runtime"] != p170["runtime"] or d167["fingerprint"] != previous.SOURCE167
            or d171["fingerprint"] != windows.SOURCE171 or d172["fingerprint"] != SOURCE172):
        raise ValueError("Need the bound Stage-170, Stage-171 and Stage-172 sources")
    def table(path, decision, name):
        io.verify({str(path / name): decision["table_sha256"][name]})
        return gain.prior.prior.read_table(path / name)
    cases = bind_cases(table(r171, d171, "stage171_cases.csv"), table(r171, d171, "stage171_frames.csv"),
                       table(r172, d172, "stage172_cases.csv"), table(r172, d172, "stage172_anchors.csv"))
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Source groups changed")
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for seq, arm in sorted({(x["sequence"], x["source"]) for x in cases}):
        entry = entries.get(seq)
        if entry is None:
            entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
            entries[seq] = entry
        frames = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["frames"]) != frames + 1 or len(entry["ground_truth"]) != frames + 1
                or max(x["end"] for x in cases if x["sequence"] == seq) > frames):
            raise ValueError(f"Invalid replay horizon at {seq}")
        data = previous.archive_trace(r170, p170, d170, seq, arm)
        previous.audit(data, frames, entry["initial_box"], arm, thresholds[fold_of[seq.split("_", 1)[0]]])
        sources[seq, arm] = data
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(windows.__file__), Path(matched.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage172=SOURCE172, cases=cases, horizons=list(HORIZONS), arms=list(ARMS),
                        runtime=runtime, runtime_differences=differences, code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)
    def location(case, arm):
        return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(plan_fingerprint=plan["fingerprint"],
                                                                 case_id=case["id"], arm=arm)
    size = HORIZONS[-1] + 1
    searches = size * len(cases) * len(ARMS)
    reused = size * sum(location(case, arm)[0].with_suffix(".receipt.json").exists() for case in cases for arm in ARMS)
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-173", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables = dict(cases=cases, anchors=[], outcomes=[], decisions=[], branches=[])
    artifacts = {}
    progress.report("Replaying matched layer choices")
    with ExitStack() as stack:
        observer = AlignmentObserver(native, stack)
        for case in cases:
            seq = case["sequence"]
            entry, data = entries[seq], sources[seq, case["source"]]
            model = models[fold_of[seq.split("_", 1)[0]]]
            traces, template = {}, None

            def read(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for arm in ARMS:
                path, meta = location(case, arm)
                trace = c.read_cache(path, meta)
                fresh = trace is None
                if fresh:
                    if template is None:
                        template = read(0)
                    trace = replay(observer, read, template, entry, case, data, arm, model, progress)
                audit(trace, case, data, arm)
                if fresh:
                    c.save(path, trace, meta)
                traces[arm] = trace
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            truth = np.asarray(entry["ground_truth"], float)[case["frame"]:case["end"] + 1]
            a, outcomes, decisions, events = report(case, traces, truth, not differences, data)
            tables["anchors"].append(a)
            tables["outcomes"].extend(outcomes)
            tables["decisions"].extend(decisions)
            tables["branches"].extend(events)
    native.assert_frozen()
    tables["summary"], tables["paired"] = summarize(cases, tables["anchors"], tables["outcomes"], tables["decisions"])
    if progress.done != work - 1:
        raise ValueError("Incomplete progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], work=work, cases=len(cases), searches=searches,
                rollouts=len(tables["branches"]), runtime=runtime, runtime_differences=differences,
                artifact_sha256=artifacts, table_sha256=gain.prior.publish(root, tables, stage="stage173")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=2, type=Path, metavar=("STAGE171", "STAGE172"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage173_gate_layer_alignment")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        _, a = gain.prior.archives.bundle(args.bundles[0], 171, windows.SOURCE171)
        _, b = gain.prior.archives.bundle(args.bundles[1], 172, SOURCE172)
        cases = bind_cases(a["stage171_cases.csv"], a["stage171_frames.csv"], b["stage172_cases.csv"], b["stage172_anchors.csv"])
        print(f"{len(cases)} sealed anchors, "
              f"{sum(x['expected_proposal'] != x['expected_ranked'] for x in cases)} proposal/ranking disagreements")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, the original checkpoint and an output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage173_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            if saved["source_stage172"] != SOURCE172:
                raise ValueError("Stage-173 run uses another source")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-173", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE173_GATE_LAYER_ALIGNMENT",
                                              source_stage172=SOURCE172, new_full_trajectories=0, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
