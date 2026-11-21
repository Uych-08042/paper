"""Test a single-frame veto and a fixed cooldown on matched Stage-168 states."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage168_matched_layer_rollouts as replay


online, gate, gain = replay.online, replay.gate, replay.gain
history, veto, c, io, base, TRACKING = replay.history, replay.veto, replay.c, replay.io, replay.base, replay.TRACKING
SOURCE168 = "2bb16b118a44a9bb6790c98a37bb9c67e1b2ed3601991e1a09dd461ff94f7153"
SCHEMA = "stage169_targeted_layer_veto_v1"
SOURCES = (*replay.SOURCES, 168)
SUCCESSORS = 128
TRIALS = (
    dict(trial="0304_skip_530", case_id="uav0000304_00253_s__shared_gate_original__000523",
         branch="original_continued", block_from=530, block_until=530, reference=554),
    dict(trial="0089_original_cooldown", case_id="uav0000089_00920_s__shared_gate_pairwise__001182",
         branch="original_continued", block_from=1183, block_until=1198, reference=1182),
    dict(trial="0089_pairwise_cooldown", case_id="uav0000089_00920_s__shared_gate_pairwise__001182",
         branch="pairwise_continued", block_from=1183, block_until=1198, reference=1182),
)
ARMS = ("continued", "blocked")


def protocol(decision, branches):
    if (decision["status"] != "COMPLETE_STAGE168_MATCHED_CONTINUATIONS"
            or decision["source_stage167"] != replay.SOURCE167 or decision["max_replay_box_error"] != 0.
            or decision["original_gate_replays"] != decision["cases"]):
        raise ValueError("Need the reproduced Stage-168 matched continuations")
    index = {(r["case_id"], r["arm"]): r for r in branches}
    if len(index) != len(branches):
        raise ValueError("Duplicate Stage-168 branch")
    rows = []
    for trial in TRIALS:
        case, arm = trial["case_id"], trial["branch"]
        control = index[case, arm]
        once = index[case, arm.replace("_continued", "_once")]
        start, end = int(control["frame"]), int(control["end"])
        if not start < trial["block_from"] <= trial["block_until"] <= end or int(control["later_switches"]) <= 0:
            raise ValueError(f"Unexpected Stage-168 timing for {trial['trial']}")
        if control["anchor_layer"] != once["anchor_layer"]:
            raise ValueError("Single and repeated source branches disagree at the anchor")
        rows.append(dict(trial, sequence=control["sequence"], history=control["history"], frame=start,
                         source_end=end, source_anchor_layer=int(control["anchor_layer"]),
                         source_later_switches=int(control["later_switches"]),
                         source_first_failure_frame=int(control["first_failure_frame"]) if control["first_failure_frame"] else None,
                         source_once_mean_iou=float(once["mean_iou"]), source_continued_mean_iou=float(control["mean_iou"])))
    return rows


class VetoObserver(online.RoutedObserver):
    def step(self, image, parent, frame):
        self.frame = frame
        self.would_switch = self.vetoed = False
        row = super().step(image, parent, frame)
        return dict(row, would_switch=self.would_switch, vetoed=self.vetoed)

    def route(self, module, args, output):
        routed = super().route(module, args, output)
        self.would_switch = self.switched
        self.vetoed = self.would_switch and self.block_from <= self.frame <= self.block_until
        if self.vetoed:
            self.switched, self.chosen = False, self.original
            return None
        return routed


def audit(data, case, source, arm, threshold):
    frame, end = case["frame"], case["end"]
    axis = np.arange(frame, end + 1)
    np.testing.assert_array_equal(data["frame_id"], axis)
    boxes = data["boxes"]
    if boxes.shape != (len(axis), 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Invalid rollout boxes")
    np.testing.assert_array_equal(data["parent"], np.vstack((source["parent"][frame - 1], boxes[:-1])))
    cut = np.full(len(axis), threshold)
    cut[0] = -1e9  # Match the Stage-168 anchor intervention.
    blocked = (arm == "blocked") & (axis >= case["block_from"]) & (axis <= case["block_until"])
    eligible = (data["proposal"] != data["native_layer"]) & (data["gate_value"] > cut + veto.EPSILON)
    np.testing.assert_array_equal(data["would_switch"], eligible)
    np.testing.assert_array_equal(data["vetoed"], eligible & blocked)
    np.testing.assert_array_equal(data["switched"], eligible & ~blocked)
    np.testing.assert_array_equal(data["switched"], data["layer"] != data["native_layer"])
    if not np.isin(data["layer"], online.LAYERS).all() or not np.isfinite(data["gate_value"]).all():
        raise ValueError("Invalid layer or gate score")
    if case["branch"] == "original_continued":
        np.testing.assert_array_equal(data["layer"][data["switched"]], data["proposal"][data["switched"]])
    blocks = np.zeros((len(axis), 12), int)
    blocks[:, :6] = 1
    blocks[np.arange(len(axis)), data["layer"] - 1] = 1
    np.testing.assert_array_equal(data["block_calls"], blocks)
    for name in ("crop_calls", "selector_calls", "head_calls"):
        np.testing.assert_array_equal(data[name], np.ones(len(axis), int))


def compare_traces(case, continued, blocked, source168, truth, exact_runtime):
    first = np.flatnonzero(blocked["vetoed"])
    split = int(first[0]) if first.size else len(blocked["layer"])
    for name in ("parent", "native_layer", "proposal", "gate_value", "would_switch"):
        np.testing.assert_array_equal(continued[name][:split + 1], blocked[name][:split + 1])
    for name in ("boxes", "layer"):
        np.testing.assert_array_equal(continued[name][:split], blocked[name][:split])
    if first.size and not continued["switched"][split]:
        raise ValueError("First veto did not suppress an action in the matched control")
    length = len(source168["boxes"])
    replay_error = float(np.max(np.abs(continued["boxes"][:length] - source168["boxes"])))
    layer_match = bool(np.array_equal(continued["layer"][:length], source168["layer"]))
    if exact_runtime and (replay_error != 0. or not layer_match):
        raise ValueError(f"Stage-168 control did not replay for {case['trial']}")
    branches, frames = [], []
    for arm, data in zip(ARMS, (continued, blocked), strict=True):
        iou = online.tracking_iou(data["boxes"], truth)
        axis = data["frame_id"]
        events = online.failure_events(iou)
        valid = iou >= 0.
        within = (axis >= case["reference"]) & (axis <= case["reference"] + replay.DESIGN["reference_grace"] + 2)
        good = iou[within] >= TRACKING["recovery_iou"]
        size = TRACKING["recovery_run"]
        runs = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(good, size).all(axis=1)) if len(good) >= size else []
        after_resume = (axis > case["block_until"]) & valid
        first_good = int(axis[within][runs[0]]) if len(runs) else None
        branches.append(dict(
            trial=case["trial"], arm=arm, anchor_layer=int(data["layer"][0]), switches=int(data["switched"].sum()),
            vetoes=int(data["vetoed"].sum()), switches_after_resume=int((data["switched"] & (axis > case["block_until"])).sum()),
            mean_iou=float(iou[valid].mean()) if valid.any() else None,
            mean_iou_after_resume=float(iou[after_resume].mean()) if after_resume.any() else None,
            failure_events=len(events), unrecovered_events=sum(e["recovery"] is None for e in events),
            first_failure_frame=int(axis[events[0]["start"]]) if events else None,
            failure_at_reference=any(axis[e["start"]] <= case["reference"] <= axis[e["end"]] for e in events),
            first_good_run_at_or_after_reference=first_good,
            later_failure_after_good_run=first_good is not None and any(axis[e["start"]] >= first_good + size for e in events)))
        frames.extend(dict(trial=case["trial"], arm=arm, frame=int(t), iou=float(u), layer=int(k), native_layer=int(n),
                           gate_value=float(v), would_switch=bool(w), switched=bool(sw), vetoed=bool(z))
                      for t, u, k, n, v, w, sw, z in zip(axis, iou, data["layer"], data["native_layer"],
                          data["gate_value"], data["would_switch"], data["switched"], data["vetoed"], strict=True))
    one, two = branches
    summary = dict(trial=case["trial"], sequence=case["sequence"], frame=case["frame"], end=case["end"],
                   block_from=case["block_from"], block_until=case["block_until"], reference=case["reference"],
                   first_effective_veto_frame=int(blocked["frame_id"][split]) if first.size else None,
                   vetoes=int(first.size), replay_max_box_error=replay_error, replay_layers_match=layer_match,
                   control_failure_at_reference=one["failure_at_reference"], blocked_failure_at_reference=two["failure_at_reference"],
                   control_good_run=one["first_good_run_at_or_after_reference"],
                   blocked_good_run=two["first_good_run_at_or_after_reference"],
                   blocked_later_failure=two["later_failure_after_good_run"],
                   mean_iou_delta=two["mean_iou"] - one["mean_iou"] if one["mean_iou"] is not None else None)
    return summary, branches, frames


def location(root, plan, case, arm):
    return root / "rollouts" / f"{case['trial']}__{arm}.npz", dict(
        plan_fingerprint=plan["fingerprint"], trial=case["trial"], arm=arm)


def rollouts(root, plan, cases, entries, sources, controls, models, native, image_root, progress):
    rows = dict(trials=cases, contrasts=[], branches=[], frames=[])
    artifacts = {}
    fold_of = {g: f["fold"] for f in plan["folds"] for g in f["held"]}
    with ExitStack() as stack:
        observer = VetoObserver(native, stack)
        for case in cases:
            seq = case["sequence"]
            entry, source = entries[seq], sources[seq, case["history"]]
            model = models[fold_of[veto.source_group(seq)]]
            images, traces = None, {}
            for arm in ARMS:
                path, metadata = location(root, plan, case, arm)
                data = c.read_cache(path, metadata)
                if data is None:
                    if images is None:
                        template = online.checked_image(image_root, entry["frames"][0])
                        images = {t: online.checked_image(image_root, entry["frames"][t])
                                  for t in range(case["frame"], case["end"] + 1)}
                    observer.block_from, observer.block_until = (case["block_from"], case["block_until"]) if arm == "blocked" else (0, -1)
                    data = replay.replay_branch(observer, images, template, entry["initial_box"], case, source,
                                                case["branch"], model, progress)
                    audit(data, case, source, arm, model["threshold"])
                    c.save(path, data, metadata)
                else:
                    audit(data, case, source, arm, model["threshold"])
                traces[arm] = data
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            truth = np.asarray(entry["ground_truth"], float)[case["frame"]:case["end"] + 1]
            result, branches, frames = compare_traces(case, traces["continued"], traces["blocked"], controls[case["trial"]],
                                                       truth, exact_runtime=not plan["runtime_differences"])
            rows["contrasts"].append(result)
            rows["branches"].extend(branches)
            rows["frames"].extend(frames)
    native.assert_frozen()
    return rows, artifacts


def execute(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r168 = paths
    p167 = c.c66.sealed(r167 / "manifest.json", online.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    p168 = c.c66.sealed(r168 / "manifest.json", replay.SCHEMA)
    d168 = c.c66.sealed(r168 / "stage168_decision.json", replay.SCHEMA)
    if (d167["fingerprint"] != replay.SOURCE167 or d168["fingerprint"] != SOURCE168
            or d167["plan_fingerprint"] != p167["fingerprint"] or d168["plan_fingerprint"] != p168["fingerprint"]
            or p168["source_stage167"] != replay.SOURCE167 or p167["folds"] != p168["folds"]):
        raise ValueError("Need the bound Stage-167 and Stage-168 runs")
    report = "stage168_branches.csv"
    io.verify({str(r168 / report): d168["table_sha256"][report]})
    cases = protocol(d168, gain.prior.prior.read_table(r168 / report))
    p159, points, gain_models, rank_models, ranks = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    if d121["fingerprint"] != p167["source_stage121"] or p159["folds"] != p167["folds"]:
        raise ValueError("Source group definitions changed")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources, controls = {}, {}, {}
    stage168_cases = {r["case_id"]: r for r in p168["cases"]}
    for case in cases:
        seq, arm = case["sequence"], case["history"]
        if seq not in entries:
            job = jobs[seq]
            entry = s95.load_unit(r93, r89, p93, p89, d89, origin, job["spec"])[0]
            if entry["sequence"] != seq or len(entry["frames"]) != job["frames"] + 1 or len(entry["frames"]) != len(entry["ground_truth"]):
                raise ValueError("Incompatible video entry")
            entries[seq] = entry
            path, meta = online.cache_location(r167, p167, seq, arm)
            io.verify({str(f): d167["artifact_sha256"][f.relative_to(r167).as_posix()]
                       for f in (path, path.with_suffix(".receipt.json"))})
            cache = c.read_cache(path, meta)
            if cache is None:
                raise ValueError("Missing Stage-167 source trajectory")
            sources[seq, arm] = cache
        case["end"] = min(case["frame"] + SUCCESSORS, jobs[seq]["frames"])
        old_case = stage168_cases[case["case_id"]]
        if old_case["end"] != case["source_end"] or old_case["frame"] != case["frame"]:
            raise ValueError("Continuation anchor differs")
        path, meta = replay.rollout_location(r168, p168, old_case, case["branch"])
        io.verify({str(f): d168["artifact_sha256"][f.relative_to(r168).as_posix()]
                   for f in (path, path.with_suffix(".receipt.json"))})
        control = c.read_cache(path, meta)
        if control is None:
            raise ValueError("Missing Stage-168 continued reference")
        controls[case["trial"]] = control
    check = cases[0]
    if not sources[check["sequence"], check["history"]]["switched"][check["block_from"] - 1]:
        raise ValueError("Stage-167 0304/530 is not a logged intervention")
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p168["runtime"], runtime)
    folds = p159["folds"]
    code = (Path(__file__), Path(replay.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage168=SOURCE168, cases=cases, folds=folds, runtime=runtime,
                        runtime_differences=differences, code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)
    searches = len(ARMS) * sum(case["end"] - case["frame"] + 1 for case in cases)
    reused = sum(case["end"] - case["frame"] + 1 for case in cases for arm in ARMS
                 if location(root, plan, case, arm)[0].with_suffix(".receipt.json").exists())
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-169", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 models")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gain_models, rank_models, ranks, progress)
    for row, archived in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in archived], list(archived.values()), rtol=0, atol=1e-10)
        models[archived["fold"]]["threshold"] = archived["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    progress.report("Running targeted paired continuations")
    rows, artifacts = rollouts(root, plan, cases, entries, sources, controls, models, native, old["root"], progress)
    hashes = gain.prior.publish(root, rows, stage="stage169")
    return dict(plan_fingerprint=plan["fingerprint"], runtime=runtime, runtime_differences=differences, trials=len(cases),
                rollouts=len(rows["branches"]), searches=searches, artifact_sha256=artifacts, table_sha256=hashes), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--cached-protocol", type=Path, metavar="STAGE168_BUNDLE")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    parser.set_defaults(mode="stage169_targeted_layer_veto")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [p.expanduser().resolve() for n in SOURCES if (p := getattr(args, f"stage{n}_root")) is not None]
    cached = args.cached_protocol is not None
    if (cached and (paths or args.checkpoint is not None)
            or not cached and (len(paths) != len(SOURCES) or args.checkpoint is None) or args.progress_interval <= 0):
        parser.error("Provide either a Stage-168 bundle or all server roots and the original checkpoint")
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE169_CACHED_PROTOCOL" if cached else "COMPLETE_STAGE169_TARGETED_VETO"
    with gain.prior.prior.writer_lock(root):
        destination = root / "stage169_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage168"] != SOURCE168:
                raise ValueError("Use separate directories for the protocol and trajectory replays")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-169", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            d, b = gain.prior.archives.bundle(args.cached_protocol, 168, SOURCE168)
            progress = OverallProgress("Stage-169", 1, interval=args.progress_interval, unit="work")
            progress.report("Checking targeted replay protocol")
            cases = protocol(d, b["stage168_branches.csv"])
            hashes = gain.prior.publish(root, dict(trials=cases), stage="stage169")
            result = dict(trials=len(cases), rollouts=0, searches=0, artifact_sha256={}, table_sha256=hashes)
        else:
            result, progress = execute(args, root, paths)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete replay progress accounting")
        c.publish_identical(destination, io.seal(dict(schema=SCHEMA, status=status, source_stage168=SOURCE168,
                                                     work=progress.total, new_full_trajectories=0, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
