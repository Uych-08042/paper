"""Evaluate a fixed post-switch cooldown in full SGLA video trajectories."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage169_targeted_layer_veto as targeted


online, base, gate, gain = targeted.online, targeted.base, targeted.gate, targeted.gain
history, veto, c, io = targeted.history, targeted.veto, targeted.c, targeted.io
SOURCE167 = targeted.replay.SOURCE167
SOURCE169 = "024b4c0990a0a7a66f3607d0fb518c9605260c0578ffc5f4a7b83cca2681009e"
SCHEMA = "stage170_full_video_cooldown_v1"
SOURCES = (*online.SOURCES, 167, 169)
COOLDOWN = 16
CONTROLS = dict(cooldown_original=online.ARMS[1], cooldown_pairwise=online.ARMS[2])
ARMS = (*online.ARMS, *CONTROLS)
FOCUS = {"uav0000304", "uav0000080", "uav0000089", "uav0000176"}


def verify_evidence(decision, rows):
    if (decision["status"] != "COMPLETE_STAGE169_TARGETED_VETO"
            or decision["source_stage168"] != targeted.SOURCE168 or decision["rollouts"] != 6):
        raise ValueError("Need the completed targeted veto experiment")
    expected = {r["trial"] for r in targeted.TRIALS}
    by_trial = {r["trial"]: r for r in rows}
    if len(rows) != len(by_trial) or set(by_trial) != expected:
        raise ValueError("Incomplete Stage-169 contrasts")
    first = by_trial["0304_skip_530"]
    if (first["first_effective_veto_frame"] != "530" or first["control_failure_at_reference"] != "True"
            or first["blocked_failure_at_reference"] != "False"):
        raise ValueError("Stage-169 single veto did not reproduce")
    for name in sorted(expected - {"0304_skip_530"}):
        row = by_trial[name]
        if row["blocked_good_run"] != "1194" or row["control_good_run"] or row["blocked_later_failure"] != "False":
            raise ValueError(f"Stage-169 recovery did not reproduce in {name}")
    return rows


class CooldownObserver(online.RoutedObserver):
    def initialize(self, image, box, arm, model):
        self.cooldown = COOLDOWN if arm in CONTROLS else 0
        self.last_switch = None
        super().initialize(image, box, CONTROLS.get(arm, arm), model)

    def step(self, image, parent, frame):
        self.in_cooldown = self.last_switch is not None and frame - self.last_switch <= self.cooldown
        self.would_switch = self.vetoed = False
        row = super().step(image, parent, frame)
        if self.switched:
            self.last_switch = frame
        return dict(row, in_cooldown=self.in_cooldown, would_switch=self.would_switch, vetoed=self.vetoed)

    def route(self, module, args, output):
        routed = super().route(module, args, output)
        self.would_switch = self.switched
        self.vetoed = self.would_switch and self.in_cooldown
        if self.vetoed:
            self.switched, self.chosen = False, self.original
            return None
        return routed


def audit(trace, frames, initial, arm, threshold):
    if arm not in CONTROLS:
        online.audit(trace, frames, initial, arm, threshold)
        return
    ids = np.arange(1, frames + 1)
    last = np.maximum.accumulate(np.r_[0, np.where(trace["switched"], ids, 0)])[:-1]
    blocked = (last > 0) & (ids - last <= COOLDOWN)
    proposed = (trace["proposal"] != trace["native_layer"]) & (trace["gate_value"] > threshold + veto.EPSILON)
    np.testing.assert_array_equal(trace["in_cooldown"], blocked)
    np.testing.assert_array_equal(trace["would_switch"], proposed)
    np.testing.assert_array_equal(trace["vetoed"], blocked & proposed)
    np.testing.assert_array_equal(trace["switched"], proposed & ~blocked)
    executed = dict(trace, proposal=np.where(blocked, trace["native_layer"], trace["proposal"]))
    online.audit(executed, frames, initial, CONTROLS[arm], threshold)


def audit_until_veto(control, cooldown):
    hits = np.flatnonzero(cooldown["vetoed"])
    before = int(hits[0]) if hits.size else len(control["boxes"])
    for k in ("parent", "native_layer", "proposal", "gate_value"):
        np.testing.assert_array_equal(control[k][:before + 1], cooldown[k][:before + 1])
    for k in ("boxes", "layer", "hann_peak", "switched"):
        np.testing.assert_array_equal(control[k][:before], cooldown[k][:before])
    if hits.size and not control["switched"][before]:
        raise ValueError("First veto did not suppress the matching control action")


def report_video(sequence, truth, traces):
    rows, events, routing = online.report_video(sequence, truth, traces)
    for r in rows:
        arm = r["arm"]
        trace = traces[arm]
        hits = np.flatnonzero(trace.get("vetoed", np.zeros(r["frames"], bool)))
        comparison = CONTROLS.get(arm)
        r.update(vetoes=int(hits.size), first_veto_frame=int(hits[0]) + 1 if hits.size else None,
                 stable_own_new_failure=False, missed_own_recoveries=0)
        if comparison is not None:
            other = online.tracking_iou(traces[comparison]["boxes"], truth)
            ref_events = online.failure_events(other)
            iou = online.tracking_iou(trace["boxes"], truth)
            r.update(stable_own_new_failure=not ref_events and bool(r["failure_events"]),
                     missed_own_recoveries=sum(online.misses_recovery(e, iou) for e in ref_events))
    return rows, events, routing


def summary(rows):
    videos = sorted({r["sequence"] for r in rows})
    index = {(r["sequence"], r["arm"]): r for r in rows}
    policies, paired, checks = [], [], {}
    for arm in ARMS:
        group = [index[v, arm] for v in videos]
        policies.append(dict(arm=arm, videos=len(group),
            **{m: float(np.mean([r[m] for r in group])) for m in online.METRICS},
            **{m: sum(r[m] for r in group) for m in ("switches", "vetoes", "head_calls", "block_calls",
                "stable_native_new_failure", "stable_own_new_failure", "missed_native_recoveries", "missed_own_recoveries")}))
    aggregate = {r["arm"]: r for r in policies}
    for arm, own in CONTROLS.items():
        for control in ("native", own):
            for cohort, exclude in (("all", None), ("without_0176", "uav0000176"), ("without_0304", "uav0000304")):
                seqs = [v for v in videos if veto.source_group(v) != exclude]
                if not seqs:
                    continue
                for metric in online.METRICS:
                    values = [index[v, arm][metric] - index[v, control][metric] for v in seqs]
                    paired.append(dict(arm=arm, baseline=control, cohort=cohort,
                        **base.s103.contrast(values, f"{arm}_minus_{control}", metric, online.DESIGN)))
        r = aggregate[arm]
        refs = [aggregate[k] for k in ("native", own)]
        checks[arm] = dict(
            auc_above_native_and_own_control=all(r["auc"] > ref["auc"] for ref in refs),
            low_iou_not_higher=all(r["low_iou_fraction"] <= ref["low_iou_fraction"] for ref in refs),
            unrecovered_not_higher=all(r["unrecovered_events"] <= ref["unrecovered_events"] for ref in refs),
            preserved_stable_videos=r["stable_native_new_failure"] == r["stable_own_new_failure"] == 0,
            preserved_recoveries=r["missed_native_recoveries"] == r["missed_own_recoveries"] == 0,
            made_switches_and_vetoes=r["switches"] > 0 and r["vetoes"] > 0)
    return policies, paired, checks


def archive_trace(root, plan, decision, sequence, arm):
    path, metadata = online.cache_location(root, plan, sequence, arm)
    io.verify({str(f): decision["artifact_sha256"][f.relative_to(root).as_posix()]
               for f in (path, path.with_suffix(".receipt.json"))})
    trace = c.read_cache(path, metadata)
    if trace is None:
        raise ValueError(f"Missing verified Stage-167 control: {path}")
    return trace


def evaluate(root, plan, jobs, entries, native, models, image_root, r167, p167, d167, progress):
    reuse_controls = plan["reused_stage167_controls"]
    tables = dict(video_summary=[], failures=[], routing=[])
    files = {}
    fold_of = {g: f["fold"] for f in plan["folds"] for g in f["held"]}
    with ExitStack() as stack:
        observer = CooldownObserver(native, stack)
        for seq, job in sorted(jobs.items()):
            entry = entries[seq]
            model = models[fold_of[veto.source_group(seq)]]
            traces = {}
            for arm in ARMS:
                if reuse_controls and arm in online.ARMS:
                    trace = archive_trace(r167, p167, d167, seq, arm)
                    audit(trace, job["frames"], entry["initial_box"], arm, model["threshold"])
                else:
                    path, meta = online.cache_location(root, plan, seq, arm)
                    trace = c.read_cache(path, meta)
                    fresh = trace is None
                    if fresh:
                        trace = online.collect(observer, entry["frames"], entry["initial_box"],
                                               image_root, arm, model, progress)
                    audit(trace, job["frames"], entry["initial_box"], arm, model["threshold"])
                    if fresh:
                        c.save(path, trace, meta)
                    for f in (path, path.with_suffix(".receipt.json")):
                        files[f.relative_to(root).as_posix()] = c.file_hash(f)
                traces[arm] = trace
            for arm, control in CONTROLS.items():
                audit_until_veto(traces[control], traces[arm])
            for arm in ARMS[1:]:
                online.audit_shared_state(traces["native"], traces[arm], same_gate=False)
            values = report_video(seq, np.asarray(entry["ground_truth"], float)[1:], traces)
            for label, rows in zip(("video_summary", "failures", "routing"), values, strict=True):
                tables[label].extend(rows)
    native.assert_frozen()
    policies, paired, checks = summary(tables["video_summary"])
    tables.update(policy_summary=policies, paired=paired,
                  focus=[r for r in tables["video_summary"] if veto.source_group(r["sequence"]) in FOCUS])
    return tables, files, checks


def execute(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r169 = paths
    p167 = c.c66.sealed(r167 / "manifest.json", online.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    d169 = c.c66.sealed(r169 / "stage169_decision.json", targeted.SCHEMA)
    if (d167["fingerprint"] != SOURCE167 or d169["fingerprint"] != SOURCE169
            or d167["plan_fingerprint"] != p167["fingerprint"] or d167["runtime"] != p167["runtime"]):
        raise ValueError("Need the frozen Stage-167 and Stage-169 runs")
    name = "stage169_contrasts.csv"
    io.verify({str(r169 / name): d169["table_sha256"][name]})
    evidence = verify_evidence(d169, gain.prior.prior.read_table(r169 / name))
    p159, points, gain_models, rank_models, ranks = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    if (d121["fingerprint"] != p167["source_stage121"] or p159["folds"] != p167["folds"]
            or set(jobs) != {p["sequence"] for p in points} or len(jobs) != online.DESIGN["videos"]
            or sum(j["frames"] for j in jobs.values()) != online.DESIGN["tracked_frames"]):
        raise ValueError("Frozen Stage-167 development roster or source folds differ")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries = {seq: s95.load_unit(r93, r89, p93, p89, d89, origin, job["spec"])[0] for seq, job in jobs.items()}
    for seq, entry in entries.items():
        if entry["sequence"] != seq or len(entry["frames"]) != jobs[seq]["frames"] + 1 or len(entry["frames"]) != len(entry["ground_truth"]):
            raise ValueError(f"Video roster changed at {seq}")
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p167["runtime"], runtime)
    use_archive = not differences
    code = (Path(__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage167=SOURCE167, source_stage169=SOURCE169,
                        cooldown=COOLDOWN, arms=list(ARMS), folds=p159["folds"], runtime=runtime,
                        runtime_differences=differences, reused_stage167_controls=use_archive,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "evaluation").mkdir(exist_ok=True)
    fresh_arms = tuple(CONTROLS) if use_archive else ARMS
    tracked = online.DESIGN["tracked_frames"]
    reused = tracked * (len(ARMS) - len(fresh_arms))
    reused += sum(job["frames"] for seq, job in jobs.items() for arm in fresh_arms
                  if online.cache_location(root, plan, seq, arm)[0].with_suffix(".receipt.json").exists())
    folds = p159["folds"]
    work = tracked * len(ARMS) + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-170", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 models and thresholds")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gain_models, rank_models, ranks, progress)
    for row, archived in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in archived], list(archived.values()), rtol=0, atol=1e-10)
        models[archived["fold"]]["threshold"] = archived["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    progress.report("Tracking complete videos")
    tables, files, checks = evaluate(root, plan, jobs, entries, native, models, old["root"], r167, p167, d167, progress)
    tables.update(source_evidence=evidence)
    hashes = gain.prior.publish(root, tables, stage="stage170")
    return dict(plan_fingerprint=plan["fingerprint"], cooldown=COOLDOWN, runtime=runtime, runtime_differences=differences,
                videos=len(jobs), full_trajectories=len(ARMS) * len(jobs), new_full_trajectories=len(fresh_arms) * len(jobs),
                reused_stage167_controls=use_archive, artifact_sha256=files, table_sha256=hashes,
                checks=checks, passes_development_checks={arm: all(results.values()) for arm, results in checks.items()}), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in SOURCES:
        parser.add_argument(f"--stage{n}-root", type=Path)
    parser.add_argument("--cached-evidence", type=Path, metavar="STAGE169_BUNDLE")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    parser.set_defaults(mode="stage170_layer_cooldown")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [p.expanduser().resolve() for n in SOURCES if (p := getattr(args, f"stage{n}_root")) is not None]
    cached = args.cached_evidence is not None
    if (cached and (paths or args.checkpoint is not None)
            or not cached and (len(paths) != len(SOURCES) or args.checkpoint is None) or args.progress_interval <= 0):
        parser.error("Provide either a Stage-169 bundle or all server roots and the original checkpoint")
    if any(root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE170_CACHED_EVIDENCE" if cached else "COMPLETE_STAGE170_FULL_VIDEO_COOLDOWN"
    with gain.prior.prior.writer_lock(root):
        destination = root / "stage170_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage169"] != SOURCE169:
                raise ValueError("Use separate directories for cached evidence and full trajectories")
            io.verify({str(root / k): h for k, h in (saved["table_sha256"] | saved["artifact_sha256"]).items()})
            OverallProgress("Stage-170", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            data, tables = gain.prior.archives.bundle(args.cached_evidence, 169, SOURCE169)
            evidence = verify_evidence(data, tables["stage169_contrasts.csv"])
            progress = OverallProgress("Stage-170", 1, interval=args.progress_interval, unit="work")
            progress.report("Checking targeted evidence")
            hashes = gain.prior.publish(root, dict(source_evidence=evidence), stage="stage170")
            result = dict(table_sha256=hashes, artifact_sha256={}, new_full_trajectories=0)
        else:
            result, progress = execute(args, root, paths)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete full-video cooldown progress accounting")
        c.publish_identical(destination, io.seal(dict(schema=SCHEMA, status=status, source_stage167=SOURCE167,
            source_stage169=SOURCE169, work=progress.total, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
