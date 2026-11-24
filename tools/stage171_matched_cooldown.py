"""Cross anchor layer choice with the subsequent cooldown on matched Stage-170 histories."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage170_layer_cooldown as previous


online, base, gate = previous.online, previous.base, previous.gate
gain, history, c, io = previous.gain, previous.history, previous.c, previous.io
SOURCE170 = "875e2f065e44a9f9ff61e100f93b85c14179ce67a0dfbe6277a269ff6cd3bee5"
SCHEMA = "stage171_matched_cooldown_v1"
SOURCES = (*online.SOURCES, 167, 170)
OPEN, COLD = "shared_gate_pairwise", "cooldown_pairwise"
ARMS = ("keep_open", "switch_open", "keep_cooldown", "switch_cooldown")
FOCUS = (
    ("uav0000304_00253_s", COLD, COLD, 554, "early_failure"),
    ("uav0000176_00000_s", COLD, OPEN, 158, "prevented_failure"),
    ("uav0000089_00920_s", OPEN, COLD, 1182, "recovery"),
)
LOOKBACK, HORIZON = 32, 64


def windows(videos, events):
    by_arm = {(row["sequence"], row["arm"]): row for row in videos}
    rows = []
    for sequence, source, reference_arm, event, purpose in FOCUS:
        event_field = "recovery" if purpose == "recovery" else "onset"
        if not any(r["sequence"] == sequence and r["arm"] == reference_arm
                   and str(r[event_field]) == str(event) for r in events):
            raise ValueError(f"Stage-170 reference event did not match {sequence}")
        cold = by_arm[sequence, COLD]
        first = int(cold["first_veto_frame"])
        rows.append(dict(sequence=sequence, source=COLD, purpose="first_veto",
                         lower=first, upper=first, reference=event, frames=int(cold["frames"])))
        rows.append(dict(sequence=sequence, source=source, purpose=purpose,
                         lower=max(1, event - LOOKBACK), upper=event if purpose == "recovery" else event - 1,
                         reference=event, frames=int(cold["frames"])))
    return rows


def make_cases(specs, sources, models, fold_of):
    cases, support = [], []
    for spec in specs:
        sequence, source = spec["sequence"], spec["source"]
        data = sources[sequence, source]
        frames = np.arange(spec["lower"], spec["upper"] + 1)
        idx = frames - 1
        cutoff = models[fold_of[sequence.split("_", 1)[0]]]["threshold"] + previous.veto.EPSILON
        eligible = (data["proposal"][idx] != data["native_layer"][idx]) & (data["gate_value"][idx] > cutoff)
        executed = data["switched"][idx]
        if spec["purpose"] == "first_veto" and (not eligible[0] or executed[0] or not data["vetoed"][idx[0]]):
            raise ValueError(f"First veto does not match the source trace for {sequence}")
        candidates = frames[executed] if executed.any() else frames[eligible]
        support.append(dict(spec, eligible=int(eligible.sum()), actual_switches=int(executed.sum()),
                            selected=int(candidates[-1]) if candidates.size else None))
        if not candidates.size:
            continue
        frame = int(candidates[-1])
        prior = np.flatnonzero(data["switched"][:frame - 1])
        last = int(prior[-1] + 1) if prior.size else None
        end = spec["reference"] + LOOKBACK if spec["purpose"] == "first_veto" else frame + HORIZON
        cases.append(dict(spec, frame=frame, end=min(spec["frames"], end),
                          last_switch=last, id=f"{sequence}__{spec['purpose']}"))
    return cases, support


class MatchedObserver(previous.CooldownObserver):
    def step(self, image, parent, frame):
        self.frame = frame
        return super().step(image, parent, frame)

    def route(self, module, args, output):
        if self.frame != self.anchor:
            return super().route(module, args, output)
        routed = online.RoutedObserver.route(self, module, args, output)
        if not self.switched:
            raise ValueError(f"Matched anchor is ineligible on this runtime at frame {self.frame}")
        self.anchor_choice = self.chosen
        self.would_switch = True
        if self.run_anchor_switch:
            return routed
        self.switched, self.chosen = False, self.original
        return None


def replay(observer, images, template, entry, case, source, arm, model, progress):
    anchor, end = case["frame"], case["end"]
    observer.initialize(template, entry["initial_box"], COLD if arm.endswith("cooldown") else OPEN, model)
    observer.anchor = anchor
    observer.run_anchor_switch = arm.startswith("switch")
    observer.last_switch = case["last_switch"]
    start = max(0, anchor - 1 - online.DESIGN["history"])
    observer.boxes.extend(source["boxes"][start:anchor - 1])
    observer.peaks.extend(source["hann_peak"][start:anchor - 1])
    observer.rt.tracker.frame_id = anchor - 1
    parent, records = source["parent"][anchor - 1], []
    for frame in range(anchor, end + 1):
        row = observer.step(images(frame), parent, frame)
        records.append(dict(row, anchor_choice=observer.anchor_choice))
        parent = row["boxes"]
        progress.advance()
    return {k: np.asarray([r[k] for r in records]) for k in records[0]}


def audit(data, case, source, arm, threshold):
    axis = np.arange(case["frame"], case["end"] + 1)
    np.testing.assert_array_equal(data["frame_id"], axis)
    boxes = data["boxes"]
    if boxes.shape != (len(axis), 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Invalid continuation boxes")
    np.testing.assert_array_equal(data["parent"], np.vstack((source["parent"][axis[0] - 1], boxes[:-1])))
    allowed = (data["proposal"] != data["native_layer"]) & (data["gate_value"] > threshold + previous.veto.EPSILON)
    switched = data["layer"] != data["native_layer"]
    prior = np.maximum.accumulate(np.r_[case["last_switch"] or 0, np.where(switched, axis, 0)])[:-1]
    cooling = arm.endswith("cooldown") & (prior > 0) & (axis - prior <= previous.COOLDOWN)
    vetoed = allowed & cooling
    vetoed[0] = False
    expected = allowed & ~cooling
    expected[0] = arm.startswith("switch")
    for name, value in (("would_switch", allowed), ("switched", expected), ("in_cooldown", cooling), ("vetoed", vetoed)):
        np.testing.assert_array_equal(data[name], value)
    np.testing.assert_array_equal(switched, expected)
    if not allowed[0] or data["anchor_choice"][0] == data["native_layer"][0]:
        raise ValueError("The anchor must offer a non-native layer")
    for name in ("layer", "native_layer", "proposal", "anchor_choice"):
        if not np.isin(data[name], online.LAYERS).all():
            raise ValueError(f"Unexpected layer in {name}")
    for name in ("gate_value", "hann_peak"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid {name}")
    expected_layer = data["anchor_choice"][0] if expected[0] else data["native_layer"][0]
    np.testing.assert_equal(data["layer"][0], expected_layer)
    blocks = np.zeros((len(axis), 12), int)
    blocks[:, :6] = 1
    blocks[np.arange(len(axis)), data["layer"] - 1] = 1
    np.testing.assert_array_equal(data["block_calls"], blocks)
    for name in ("crop_calls", "selector_calls", "head_calls"):
        np.testing.assert_array_equal(data[name], np.ones(len(axis), int))


def compare(case, traces, truth, sources, exact_runtime):
    sequence, anchor, end = case["sequence"], case["frame"], case["end"]
    cached = sources[sequence, case["source"]]
    natural = ("switch_" if cached["switched"][anchor - 1] else "keep_") + (
        "cooldown" if case["source"] == COLD else "open")
    archive_arms = {natural: case["source"]}
    if case["purpose"] == "first_veto":
        archive_arms["switch_open"] = OPEN
    segment = slice(anchor - 1, end)
    replay_rows = []
    for arm, ref in archive_arms.items():
        original, run = sources[sequence, ref], traces[arm]
        fields = ("parent", "boxes", "layer", "native_layer", "proposal", "gate_value", "switched", "hann_peak")
        matches = all(np.array_equal(run[k], original[k][segment]) for k in fields)
        if exact_runtime and not matches:
            raise ValueError(f"Stage-170 natural replay differs at {case['id']} / {arm}")
        replay_rows.append(dict(case_id=case["id"], arm=arm, source=ref, exact_match=matches,
                                max_box_error=float(np.abs(run["boxes"] - original["boxes"][segment]).max())))
    for k in ("parent", "native_layer", "proposal", "gate_value", "anchor_choice"):
        for arm in ARMS[1:]:
            np.testing.assert_array_equal(traces[arm][k][0], traces[ARMS[0]][k][0])
    for action in ("keep", "switch"):
        for k in ("boxes", "layer", "hann_peak"):
            np.testing.assert_array_equal(traces[action + "_open"][k][0], traces[action + "_cooldown"][k][0])
    branches, frame_rows, effects = [], [], []
    series = {}
    for arm, data in traces.items():
        iou = online.tracking_iou(data["boxes"], truth)
        valid = iou >= 0
        ev = online.failure_events(iou)
        high = iou >= online.TRACKING["recovery_iou"]
        run = online.TRACKING["recovery_run"]
        offset = max(0, case["reference"] - anchor)
        good = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(high, run).all(axis=1)) if len(high) >= run else np.array([])
        after = good[good >= offset]
        series[arm] = iou
        branches.append(dict(case_id=case["id"], arm=arm, anchor_layer=int(data["layer"][0]),
                             anchor_iou=float(iou[0]), mean_iou=float(iou[valid].mean()) if valid.any() else None,
                             failures=len(ev), unrecovered=sum(e["recovery"] is None for e in ev),
                             first_failure=anchor + ev[0]["start"] if ev else None,
                             failure_at_reference=any(anchor + e["start"] <= case["reference"] <= anchor + e["end"] for e in ev),
                             first_good_run_at_or_after_reference=anchor + int(after[0]) if after.size else None,
                             later_switches=int(data["switched"][1:].sum()), later_vetoes=int(data["vetoed"][1:].sum())))
        for i, frame in enumerate(range(anchor, end + 1)):
            frame_rows.append(dict(case_id=case["id"], arm=arm, frame=frame, iou=float(iou[i]),
                                   layer=int(data["layer"][i]), native=int(data["native_layer"][i]),
                                   proposal=int(data["proposal"][i]), gate_value=float(data["gate_value"][i]),
                                   switched=bool(data["switched"][i]), vetoed=bool(data["vetoed"][i])))
    length = end - anchor
    for horizon in sorted({min(h, length) for h in (8, 16, 32, HORIZON, length)}):
        values = np.array([series[arm][1:horizon + 1] for arm in ARMS])
        valid = (values >= 0).all(axis=0)
        if not valid.any():
            continue
        keep_open, switch_open, keep_cold, switch_cold = values[:, valid].mean(axis=1)
        effects.append(dict(case_id=case["id"], horizon=horizon, valid_successors=int(valid.sum()),
                            switch_when_open=float(switch_open - keep_open), switch_when_cooling=float(switch_cold - keep_cold),
                            cooldown_after_keep=float(keep_cold - keep_open), cooldown_after_switch=float(switch_cold - switch_open),
                            interaction=float((switch_cold - switch_open) - (keep_cold - keep_open))))
    return dict(replays=replay_rows, branches=branches, frames=frame_rows, effects=effects)


def evaluate(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    if (d170["fingerprint"] != SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or d170["runtime"] != p170["runtime"] or d167["fingerprint"] != previous.SOURCE167):
        raise ValueError("Need the completed Stage-170 run and its frozen Stage-167 controls")
    names = ("stage170_video_summary.csv", "stage170_failures.csv")
    io.verify({str(r170 / name): d170["table_sha256"][name] for name in names})
    specs = windows(*(gain.prior.prior.read_table(r170 / name) for name in names))
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Stage-170 development roster or folds changed")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for sequence, *_ in FOCUS:
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[sequence]["spec"])[0]
        if entry["sequence"] != sequence or len(entry["frames"]) != jobs[sequence]["frames"] + 1:
            raise ValueError(f"Incompatible video entry for {sequence}")
        entries[sequence] = entry
        for arm in (OPEN, COLD):
            sources[sequence, arm] = previous.archive_trace(r170, p170, d170, sequence, arm)
    # Anchor eligibility is already committed in Stage-167 and bound to Stage-170.
    thresholds = {m["fold"]: dict(threshold=m["threshold"]) for m in d167["calibration"]}
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    for sequence, *_ in FOCUS:
        for arm in (OPEN, COLD):
            previous.audit(sources[sequence, arm], jobs[sequence]["frames"], entries[sequence]["initial_box"],
                           arm, thresholds[fold_of[sequence.split("_", 1)[0]]]["threshold"])
        previous.audit_until_veto(sources[sequence, OPEN], sources[sequence, COLD])
    cases, support = make_cases(specs, sources, thresholds, fold_of)
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(previous.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage170=SOURCE170, cases=cases, arms=list(ARMS),
                        runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    def location(case, arm):
        return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(
            plan_fingerprint=plan["fingerprint"], case_id=case["id"], arm=arm)
    searches = len(ARMS) * sum(case["end"] - case["frame"] + 1 for case in cases)
    reused = sum(case["end"] - case["frame"] + 1 for case in cases for arm in ARMS
                 if location(case, arm)[0].with_suffix(".receipt.json").exists())
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-171", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 models")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), rtol=0, atol=1e-10)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    (root / "rollouts").mkdir(exist_ok=True)
    tables = dict(windows=support, cases=cases, replays=[], branches=[], frames=[], effects=[])
    hashes = {}
    progress.report("Running matched continuations")
    with ExitStack() as stack:
        observer = MatchedObserver(native, stack)
        for case in cases:
            sequence, anchor, end = case["sequence"], case["frame"], case["end"]
            entry, source = entries[sequence], sources[sequence, case["source"]]
            model = models[fold_of[sequence.split("_", 1)[0]]]
            traces = {}
            template = None

            def image_at(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for arm in ARMS:
                path, metadata = location(case, arm)
                data = c.read_cache(path, metadata)
                new = data is None
                if new:
                    if template is None:
                        template = image_at(0)
                    data = replay(observer, image_at, template, entry, case, source, arm, model, progress)
                audit(data, case, source, arm, model["threshold"])
                if new:
                    c.save(path, data, metadata)
                traces[arm] = data
                for f in (path, path.with_suffix(".receipt.json")):
                    hashes[f.relative_to(root).as_posix()] = c.file_hash(f)
            result = compare(case, traces, np.asarray(entry["ground_truth"], float)[anchor:end + 1],
                             sources, not differences)
            for k, rows in result.items():
                tables[k].extend(rows)
    native.assert_frozen()
    if progress.done != work - 1:
        raise ValueError("Incomplete progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], work=work, runtime=runtime, runtime_differences=differences,
                cases=len(cases), rollouts=len(tables["branches"]), searches=searches,
                artifact_sha256=hashes, table_sha256=gain.prior.publish(root, tables, stage="stage171")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage171_matched_cooldown")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for stage in SOURCES if (path := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU check")
        _, tables = gain.prior.archives.bundle(args.bundle, 170, SOURCE170)
        for spec in windows(tables["stage170_video_summary.csv"], tables["stage170_failures.csv"]):
            print(f"{spec['sequence']} {spec['purpose']}: {spec['lower']}..{spec['upper']}, event {spec['reference']}")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all server roots, the original checkpoint and an output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        decision = root / "stage171_decision.json"
        if decision.exists():
            saved = c.c66.sealed(decision, SCHEMA)
            if saved["source_stage170"] != SOURCE170:
                raise ValueError("Stage-171 run uses another source")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-171", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = evaluate(args, root, paths)
        c.publish_identical(decision, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE171_MATCHED_COOLDOWN",
                                                  source_stage170=SOURCE170, new_full_trajectories=0, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
