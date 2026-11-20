"""Evaluate grayscale box selection with separate reported and search-state boxes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import stage151_write_geometry_cross as geometry
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, failure_events, source_group, success_auc, tracking_iou
from stage27_uav_summary import write_table


prior, c, io, base = geometry.prior, geometry.c, geometry.io, geometry.base
ablation = prior.ablation
appearance, fixed = ablation.appearance, prior.fixed
SCHEMA = "stage152_center_closed_loop_v1"
STATUS = "COMPLETE_CENTER_ONLY_FULL_VIDEO_DEVELOPMENT"
SOURCE151 = "05343065c46229bc2f3133e9f56896951b2d2ecb5afb9718e1bedda24a639f86"
ARMS = ("native", "regression", "gray_full", "gray_center")
DESIGN = dict(videos=28, tracked_frames=29047, threshold=.025, appearance_menu=ablation.DESIGN,
              low_iou=.2, long_failure_frames=32, bootstrap_seed=23152, bootstrap_samples=20000)
SOURCES = ("stage76", "stage89", "stage93", "stage121", "stage143", "stage149", "stage151")
TABLES = ("policy_summary", "paired", "video_summary", "failures", "switches", "native_shadow")
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "failure_events", "unrecovered_events")
EVENT_COLUMNS = ("sequence", "arm", "onset", "confirmation", "end", "recovery", "duration", "censor")
SWITCH_COLUMNS = ("sequence", "arm", "frame", "margin", "regression_iou", "reported_iou", "search_state_iou")
WORK = dict(training_updates=0, confirmation_inference_calls=0)


def gray_proposal(row, scores, image, template, history):
    recent = appearance.recent_rows(row["frame"], history)
    past = [r["descriptor"] for r in recent]
    observed = [appearance.similarities(appearance.candidate_descriptor(image, box, row["usable"]), template, past)
                for box in row["boxes"]]
    cues = {cue:np.array([value[cue] for value in observed]) for cue in ablation.previous.CUES}
    anchor, _, eligible, _ = ablation.previous.candidate_menu(row, cues, scores, design=ablation.DESIGN)
    if not len(eligible):
        return anchor, anchor, 0.
    ncc = cues["template_ncc"]
    margins = ablation.DESIGN["appearance_weight"] * (ncc[eligible] - ncc[anchor])
    best = int(margins.argmax())
    return anchor, int(eligible[best]), float(margins[best])


def state_box(regression, reported, arm):
    state = np.array(reported, float, copy=True)
    if arm == "gray_center":
        state[..., :2] += .5 * (state[..., 2:] - regression[..., 2:])
        state[..., 2:] = regression[..., 2:]
    return state


def collect(observer, frames, initial_box, model, ap, arm, *,
            reader=checked_image, observe=base.e63.actor.observe):
    """Use only initial annotation, current image, and the arm's own prior observations."""
    if arm not in ARMS:
        raise ValueError("Unknown Stage-152 arm")
    first = reader(ap["root"], frames[0])
    observer.initialize(first, initial_box)
    h, w = first.shape[:2]
    template = appearance.candidate_descriptor(first, initial_box, [0., 0., w, h])
    parent = np.array(initial_box, float)
    history, records = {}, []
    for frame, item in enumerate(frames[1:], 1):
        image = reader(ap["root"], item)
        obs = observe(observer, image, parent, frame, ap)
        row, grid, native, regression = prior.decode_search(obs, model, ap)
        proposal, margin = regression, 0.
        if arm != "regression":
            row.update(frame=frame, boxes=grid[row["cells"]], parent=obs["parent"], usable=obs["usable"])
            anchor, idx, margin = gray_proposal(row, fixed.score(model, row["x"]), image, template, history)
            if row["cells"][anchor] != regression:
                raise ValueError("Gray and Stage-143 regression anchors differ")
            proposal = int(row["cells"][idx])
            if appearance.trusted(row):
                row["descriptor"] = appearance.candidate_descriptor(image, grid[native], row["usable"])
                history[frame] = row
            history.pop(frame - appearance.DESIGN["recent_lookback"], None)
        eligible = margin > DESIGN["threshold"]
        switched = arm in ARMS[2:] and eligible
        cell = native if arm == "native" else proposal if switched else regression
        output = grid[cell]
        state = state_box(grid[regression], output, arm)
        records.append(dict(
            frame_id=frame, parent=parent, state=state, boxes=output.copy(), regression=grid[regression].copy(),
            regression_cell=regression, selected_cell=cell, proposal_cell=proposal,
            margin=margin, switched=switched, shadow_accepted=eligible, crop_sha256=obs["crop_sha256"],
            **{k:obs[k] for k in ("head_calls", "selector_calls", "layer", "block_calls")}))
        parent = state
        if frame % 500 == 0:
            print(f"  [{arm}] {frame}/{len(frames)-1} frames", flush=True)
    return {k:np.asarray([r[k] for r in records]) for k in records[0]}


def audit(trace, frames, initial_box, arm):
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, frames+1))
    for name in ("boxes", "parent", "state", "regression"):
        xywh = trace[name]
        if xywh.shape != (frames, 4) or not np.isfinite(xywh).all() or (xywh[:, 2:] <= 0).any():
            raise ValueError(f"Invalid Stage-152 trace field {name}")
    base.e63.source_common.prior.audit_calls(trace, (frames,))
    np.testing.assert_array_equal(trace["parent"], np.vstack((initial_box, trace["state"][:-1])))
    if not np.isfinite(trace["margin"]).all():
        raise ValueError("Nonfinite gray margin")
    shadow = trace["margin"] > DESIGN["threshold"]
    np.testing.assert_array_equal(trace["shadow_accepted"], shadow)
    np.testing.assert_array_equal(trace["switched"], shadow & (arm in ARMS[2:]))
    np.testing.assert_array_equal(trace["state"], state_box(trace["regression"], trace["boxes"], arm))
    if arm != "native":
        chose = np.where(trace["switched"], trace["proposal_cell"], trace["regression_cell"])
        np.testing.assert_array_equal(trace["selected_cell"], chose)
        keep = ~trace["switched"]
        np.testing.assert_array_equal(trace["boxes"][keep], trace["regression"][keep])
    if np.any(shadow & (trace["proposal_cell"] == trace["regression_cell"])):
        raise ValueError("Accepted gray proposal must be distinct")


def audit_paired_prefix(traces):
    control, full, center = (traces[k] for k in ARMS[1:])
    first = np.flatnonzero(full["switched"])
    end = int(first[0]) if first.size else len(full["boxes"])
    for arm in (full, center):
        np.testing.assert_array_equal(arm["boxes"][:end], control["boxes"][:end])
        np.testing.assert_array_equal(arm["state"][:end], control["state"][:end])
    if first.size:
        np.testing.assert_array_equal(full["boxes"][:end+1], center["boxes"][:end+1])
        np.testing.assert_array_equal(full["switched"][:end+1], center["switched"][:end+1])
        np.testing.assert_array_equal(full["parent"][:end+1], center["parent"][:end+1])


def audit_native_shadow(trace, reference, exact_runtime):
    frame = np.array([int(r["frame"])-1 for r in reference])
    archived = np.array([r["dense_template_ncc_only_switch"] == "True" for r in reference])
    expected_cell = np.array([int(r["dense_template_ncc_only_cell"]) for r in reference])
    shadow = trace["shadow_accepted"][frame]
    picked = np.where(shadow, trace["proposal_cell"][frame], trace["regression_cell"][frame])
    switches_differ = int((shadow != archived).sum())
    cells_differ = int((picked != expected_cell).sum())
    if exact_runtime and (switches_differ or cells_differ):
        raise ValueError("Native shadow did not reproduce the frozen Stage-149 choices")
    return dict(sampled=len(frame), archived_switches=int(archived.sum()), reconstructed_switches=int(shadow.sum()),
                switch_disagreements=switches_differ, cell_disagreements=cells_differ)


def report_video(entry, traces):
    truth = np.asarray(entry["ground_truth"], float)[1:]
    iou = {arm:tracking_iou(t["boxes"], truth) for arm,t in traces.items()}
    events = {arm:failure_events(v) for arm,v in iou.items()}
    rows, failures, switches = [], [], []
    for arm, t in traces.items():
        ident = dict(sequence=entry["sequence"], arm=arm)
        values, ev = iou[arm], events[arm]
        valid = values >= 0.
        lengths = [e["end"]-e["start"]+1 for e in ev]
        switched = np.flatnonzero(t["switched"])
        rows.append(dict(ident, tracked_frames=len(values), labeled_frames=int(valid.sum()),
                         new_search_calls=int(t["head_calls"].sum()),
                         auc=success_auc(values[valid]),
                         low_iou_fraction=float((values[valid] < DESIGN["low_iou"]).mean()),
                         longest_failure_fraction=max(lengths, default=0)/len(values),
                         failure_events=len(ev), unrecovered_events=sum(e["recovery"] is None for e in ev),
                         long_events=sum(length >= DESIGN["long_failure_frames"] for length in lengths),
                         native_stable=not events["native"], regression_stable=not events["regression"],
                         introduced_native_failure=not events["native"] and bool(ev),
                         introduced_regression_failure=not events["regression"] and bool(ev),
                         switched=len(switched), first_switch=int(switched[0]+1) if len(switched) else None))
        for e,length in zip(ev,lengths):
            failures.append(dict(ident, onset=e["start"]+1, confirmation=e["confirmation"]+1, end=e["end"]+1,
                                 recovery=None if e["recovery"] is None else e["recovery"]+1,
                                 duration=length, censor=e["censor"]))
        for i in switched:
            switches.append(dict(ident, frame=int(i+1), margin=float(t["margin"][i]),
                                 regression_iou=float(tracking_iou(t["regression"][i], truth[i])),
                                 reported_iou=float(values[i]), search_state_iou=float(tracking_iou(t["state"][i], truth[i]))))
    return rows, failures, switches


def summarize(rows):
    videos = sorted({r["sequence"] for r in rows})
    by_arm = {arm:{r["sequence"]:r for r in rows if r["arm"] == arm} for arm in ARMS}
    policy, pairs = [], []
    for arm, local in by_arm.items():
        values = list(local.values())
        policy.append(dict(arm=arm, videos=len(videos),
            **{k:int(sum(r[k] for r in values)) for k in (
                "new_search_calls", "switched", "long_events", "introduced_native_failure", "introduced_regression_failure")},
            **{m:float(np.mean([r[m] for r in values])) for m in METRICS}))
        for ref in ARMS[:ARMS.index(arm)]:
            for metric in METRICS:
                effects = [local[v][metric] - by_arm[ref][v][metric] for v in videos]
                pairs.append(base.s103.contrast(effects, f"{arm}_minus_{ref}", metric, DESIGN))
    return policy, pairs


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage152_decision.json", SCHEMA)
    if (report["status"] != STATUS or plan["design"] != DESIGN or plan["source_stage151"] != SOURCE151
            or report["plan_fingerprint"] != plan["fingerprint"] or report["videos"] != DESIGN["videos"]
            or report["new_search_calls"] != DESIGN["tracked_frames"]*len(ARMS)
            or report["new_full_trajectories"] != DESIGN["videos"]*len(ARMS)
            or len(report["artifact_sha256"]) != 2*report["new_full_trajectories"]
            or any(report[k] != value for k,value in WORK.items())
            or set(report["table_sha256"]) != {f"stage152_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-152 full-video development comparison")
    io.verify({str(root/k):v for k,v in (report["table_sha256"]|report["artifact_sha256"]).items()})


def run(root, r121, p121, d121, plan, jobs, load_entry, models, shadow, observer, ap):
    tables = {name:[] for name in TABLES}
    artifacts = {}
    (root / "evaluation").mkdir(exist_ok=True)
    exact_native = not plan["stage121_runtime_differences"]
    for job in jobs:
        entry = load_entry(job["spec"])
        seq = job["spec"]["unit"]["sequence"]
        n = job["frames"]
        model = models[source_group(seq)]
        if entry["sequence"] != seq or len(entry["frames"]) != len(entry["ground_truth"]) or len(entry["frames"]) != n+1:
            raise ValueError(f"Development entry changed: {seq}")
        traces = {}
        for arm in ARMS:
            path = root / "evaluation" / f"{seq}__{arm}.npz"
            meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, arm=arm)
            trace = c.read_cache(path, meta)
            fresh = trace is None
            if fresh:
                print(f"[Stage-152] {seq} {arm}: {n} frames", flush=True)
                trace = collect(observer, entry["frames"], entry["initial_box"], model, ap, arm)
            audit(trace, n, entry["initial_box"], arm)
            if arm == "native":
                if exact_native:
                    key = f"evaluation/{base.job_key(job)}"
                    io.verify({str(r121/k):d121["artifact_sha256"][k] for k in (key+".npz", key+".receipt.json")})
                    frozen = prior.source.prior.prior.load_trace(r121, p121, job)
                    np.testing.assert_allclose(trace["boxes"], frozen["boxes"], atol=ap["design"]["box_atol"], rtol=0)
                tables["native_shadow"].append(dict(sequence=seq,
                    **audit_native_shadow(trace, shadow[seq], exact_native)))
            if fresh:
                observer.assert_frozen()
                c.save(path, trace, meta)
            traces[arm] = trace
            for item in (path, path.with_suffix(".receipt.json")):
                artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        audit_paired_prefix(traces)
        for name, rows in zip(("video_summary", "failures", "switches"), report_video(entry, traces)):
            tables[name].extend(rows)
    tables["policy_summary"], tables["paired"] = summarize(tables["video_summary"])
    hashes = {}
    fields = dict(failures=EVENT_COLUMNS, switches=SWITCH_COLUMNS)
    for name, rows in tables.items():
        path = root / f"stage152_{name}.csv"
        write_table(path, rows, fields=fields.get(name))
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage152_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_full_video_comparison",
        plan_fingerprint=plan["fingerprint"], source_stage151=SOURCE151, source_stage149=prior.SOURCE149,
        runtime=plan["runtime"], stage121_runtime_differences=plan["stage121_runtime_differences"],
        videos=len(jobs), new_full_trajectories=len(tables["video_summary"]),
        new_search_calls=sum(r["new_search_calls"] for r in tables["video_summary"]),
        table_sha256=hashes, artifact_sha256=artifacts, **WORK)))
    check_results(root)


def load_frozen(r121, r143, r149, r151):
    p121 = c.c66.sealed(r121 / "manifest.json", prior.source.s121.SCHEMA)
    d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", prior.source.s121.SCHEMA)
    fixed.check_results(r143)
    ablation.check_results(r149)
    geometry.check_results(r151)
    reports = ((r143/"stage143_decision.json", fixed.SCHEMA, ablation.prior.SOURCE143),
               (r149/"stage149_decision.json", ablation.SCHEMA, prior.SOURCE149),
               (r151/"stage151_decision.json", geometry.SCHEMA, SOURCE151))
    if (d121["fingerprint"] != prior.source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]
            or any(c.c66.sealed(p, schema)["fingerprint"] != frozen for p,schema,frozen in reports)):
        raise ValueError("Need the reviewed Stage-121, 143, 149 and 151 runs")
    jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=base.job_key)
    sequences = {j["spec"]["unit"]["sequence"] for j in jobs}
    groups = {source_group(s) for s in sequences}
    if len(jobs) != len(sequences) or len(groups) != len(sequences) or (
            len(sequences), sum(j["frames"] for j in jobs)) != (DESIGN["videos"], DESIGN["tracked_frames"]):
        raise ValueError("Expected one development trajectory per source group")
    models = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
    if set(models) != groups or any(m["held_source"] != g or set(m["fit_sources"]) != groups-{g} for g,m in models.items()):
        raise ValueError("Stage-143 source holdout changed")
    shadow = {seq:[] for seq in sequences}
    with (r149/"stage149_frames.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["threshold_mode"] == "calibrated" and row["state"] == "native":
                if float(row["template_ncc_only_threshold"]) != DESIGN["threshold"]:
                    raise ValueError("Stage-149 acceptance threshold changed")
                shadow[row["sequence"]].append(row)
    if sum(map(len, shadow.values())) != 4514 or not all(shadow.values()):
        raise ValueError("Stage-149 native shadow roster changed")
    return p121, d121, jobs, models, shadow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage152_center_closed_loop")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args, name) is None for name in SOURCES):
        parser.error("Supply the original checkpoint and frozen source directories")
    paths = [getattr(args, name).expanduser().resolve() for name in SOURCES]
    for name, path in zip(SOURCES, paths):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
        if root == path or root in path.parents or path in root.parents:
            parser.error("Use a separate Stage-152 directory")
    r76, r89, r93, r121, r143, r149, r151 = paths
    with writer_lock(root):
        if (root / "stage152_decision.json").is_file():
            check_results(root)
            return
        p121,d121,jobs,models,shadow = load_frozen(r121,r143,r149,r151)
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]

        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,d89,origin,spec)[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        differences = base.s109.check_runtime(p121["runtime"],runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume; set STAGE152_ROOT to a new directory")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage151=SOURCE151, source_stage149=prior.SOURCE149,
                            source_stage121=d121["fingerprint"], source_stage143=ablation.prior.SOURCE143,
                            arms=list(ARMS), jobs=jobs, runtime=runtime, stage121_runtime_differences=differences,
                            code_sha256={Path(p).name:c.file_hash(Path(p)) for p in (
                                __file__, prior.__file__, fixed.__file__, appearance.__file__, appearance.rgb.__file__,
                                ablation.previous.__file__, ablation.__file__)}))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root,r121,p121,d121,plan,jobs,load_entry,models,shadow,observer,base.e63.c.actor_plan(old))
        finally:
            observer.handle.remove()
    print(f"Stage-152: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
