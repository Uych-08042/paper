"""Evaluate an out-of-source localization-quality scorer on frozen SGLA boxes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage142_dense_candidate_recall as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import crop_geometry, source_group, tracking_iou
from stage28_response_common import dense_boxes
from stage46_shift_audit import publish_table


c, io = prior.c, prior.io
SCHEMA = "stage143_fixed_pool_quality_v1"
STATUS = "COMPLETE_FIXED_POOL_QUALITY_DIAGNOSTIC"
SOURCE142 = "c4cacc8a7855a178b8815b28af6e4bb0d16bfd4ef4d4a003527c6c7eedb52f33"
POOLS = ("peak", "dense")
ARMS = ("hann", "raw", "quality", "oracle")
FEATURES = ("raw", "hann", "raw_over_peak", "hann_over_peak", "raw_neighborhood_mean",
            "center_x_change", "center_y_change", "width_change", "height_change")
DESIGN = dict(good=prior.DESIGN["good_iou"], low=prior.DESIGN["low_iou"],
              ridge=.03, stable_window_frames=32, stable_low_run=3, features=list(FEATURES),
              bootstrap_seed=23143, bootstrap_samples=20000)
TABLES = ("policy", "paired", "stable", "fit", "focus", "frames")
WORK = dict(image_reads=0, new_search_calls=0, state_updates=0, sgla_weight_updates=0)


def observables(raw, hann, boxes, parent, cells):
    neighborhoods = np.lib.stride_tricks.sliding_window_view(np.pad(raw, 1, mode="edge"), (3, 3))
    local_mean = neighborhoods.mean(axis=(-1, -2)).ravel()[cells]
    raw_at, hann_at = raw.ravel()[cells], hann.ravel()[cells]
    parent_wh = np.maximum(parent[2:], 1.)
    displacement = (boxes[:, :2] + .5 * boxes[:, 2:] - parent[:2] - .5 * parent[2:]) / parent_wh
    scale = np.log(np.maximum(boxes[:, 2:], 1.) / parent_wh)
    return np.column_stack((raw_at, hann_at, raw_at / max(float(raw.max()), 1e-8),
                            hann_at / max(float(hann.max()), 1e-8), local_mean,
                            np.log1p(np.abs(displacement)), np.abs(scale)))


def extract(raw, size, offset, parent, factor, hw, box, truth, ap):
    raw = np.asarray(raw, dtype=np.float32)
    hann = raw * np.asarray(ap["window"], dtype=np.float32)
    native = int(hann.argmax())
    grid = dense_boxes(size, offset, parent, factor, ap["search_size"], hw)
    if np.abs(grid[native] - box).max() > ap["design"]["box_atol"]:
        raise ValueError("Frozen native box does not match the response decode")
    grid[native] = box
    peak = prior.candidate_pool(raw, hann, grid, native, peaks_only=True)
    dense = prior.candidate_pool(raw, hann, grid, native, peaks_only=False)
    budget = min(len(peak), len(dense))
    pools = dict(peak=np.array(peak[:budget]), dense=np.array(dense[:budget]))
    cells = np.unique(np.concatenate(list(pools.values())))
    indices = {k: np.searchsorted(cells, v) for k, v in pools.items()}
    iou = tracking_iou(grid, truth)
    return dict(cells=cells, pools=indices, x=observables(raw, hann, grid[cells], parent, cells),
                y=iou[cells], native_iou=float(iou[native]), grid_iou=float(iou.max()), budget=budget)


def check_frame(row, reference):
    np.testing.assert_allclose([row["native_iou"], row["grid_iou"]],
                               [float(reference["native_iou"]), float(reference["full_grid_best_iou"])],
                               atol=1e-12, rtol=0)
    for pool in POOLS:
        idx = row["pools"][pool]
        expected = np.array(reference[f"{pool}_cells"].split("|"), dtype=int)
        np.testing.assert_array_equal(row["cells"][idx], expected)
        np.testing.assert_allclose(row["y"][idx].max(), float(reference[f"matched_{pool}_best_iou"]),
                                   atol=1e-12, rtol=0)


def check_native(rows, reference):
    native = np.array([r["native_iou"] for r in rows])
    grid = np.array([r["grid_iou"] for r in rows])
    budgets = np.array([r["budget"] for r in rows])
    peak, dense = [np.array([r["y"][r["pools"][pool]].max() for r in rows]) for pool in POOLS]
    good, low = DESIGN["good"], DESIGN["low"]
    masks = dict(native_all=np.ones(len(rows), dtype=bool), native_below_good=native < good,
                 native_low=native < low, native_low_grid_good=(native < low) & (grid >= good))
    for group, mask in masks.items():
        if not mask.any():
            if group in reference:
                raise ValueError(f"Unexpected Stage-142 support for {group}")
            continue
        p, d = peak[mask] >= good, dense[mask] >= good
        counts = dict(frames=mask.sum(), native_good=(native[mask] >= good).sum(), grid_good=(grid[mask] >= good).sum(),
                      peak_good=p.sum(), dense_good=d.sum(), dense_added_good=(d & ~p).sum(),
                      dense_lost_good=(p & ~d).sum())
        saved = reference[group]
        if any(int(saved[k]) != v for k, v in counts.items()):
            raise ValueError(f"Stage-142 candidate counts changed for {saved['sequence']} / {group}")
        np.testing.assert_allclose(budgets[mask].mean(), float(saved["mean_budget"]), atol=1e-12, rtol=0)


def moments(rows):
    valid = [r for r in rows if r["native_iou"] >= 0.]
    if not valid:
        raise ValueError("An entire fit source has invalid annotations")
    values, weights = [], []
    for row in valid:
        ok = row["y"] >= 0.
        values.append(np.column_stack((row["x"][ok], row["y"][ok], np.ones(int(ok.sum())))))
        weights.append(np.full(ok.sum(), 1. / (ok.sum() * len(valid))))
    values, weights = np.concatenate(values), np.concatenate(weights)
    return (values.T * weights) @ values, len(valid), len(values)


def fit_peer_sources(bank, held):
    peers = sorted(set(bank) - {held})
    if held not in bank or not peers:
        raise ValueError("Held-out source requires separate fit sources")
    m = np.mean([bank[g][0] for g in peers], axis=0)
    means, y_mean = m[:-2, -1], m[-2, -1]
    cov = m[:-2, :-2] - np.outer(means, means)
    scale = np.sqrt(np.maximum(cov.diagonal(), 1e-8))
    gram = cov / np.outer(scale, scale) + DESIGN["ridge"] * np.eye(len(means))
    coef = np.linalg.solve(gram, (m[:-2, -2] - means * y_mean) / scale)
    return dict(held_source=held, fit_sources=peers, fit_frames=sum(bank[g][1] for g in peers),
                fit_boxes=sum(bank[g][2] for g in peers), mean=means.tolist(), scale=scale.tolist(),
                coef=coef.tolist(), intercept=float(y_mean))


def score(model, x):
    return (x - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]


def decide(row, predicted):
    fields = {}
    for pool, idx in row["pools"].items():
        picked = dict(hann=idx[0], raw=idx[row["x"][idx, 0].argmax()],
                      quality=idx[predicted[idx].argmax()], oracle=idx[row["y"][idx].argmax()])
        for arm, choice in picked.items():
            fields[f"{pool}_{arm}_cell"] = int(row["cells"][choice])
            fields[f"{pool}_{arm}_iou"] = float(row["y"][choice])
        fields[f"{pool}_quality_score"] = float(predicted[picked["quality"]])
        fields[f"{pool}_cells"] = "|".join(map(str, row["cells"][idx]))
    return fields


def summarize(decisions):
    good, low = DESIGN["good"], DESIGN["low"]
    valid = [r for r in decisions if r["native_iou"] >= 0.]
    native = [r for r in valid if r["state"] == "native"]
    native_low = [r for r in native if r["native_iou"] < low]
    populations = dict(native_all=native, native_low=native_low,
                       native_low_grid_good=[r for r in native_low if r["grid_iou"] >= good],
                       both_pools_rescuable=[r for r in native_low if min(r[f"{p}_oracle_iou"] for p in POOLS) >= good],
                       dense_only_good=[r for r in native_low if r["dense_oracle_iou"] >= good > r["peak_oracle_iou"]],
                       native_onsets=[r for r in native if r["onset"]],
                       native_stable_windows=[r for r in native if r["stable_window"]],
                       branch_replays=[r for r in valid if r["state"] == "replay"],
                       relapse_replays=[r for r in valid if r["relapse"]])
    policy, paired = [], []
    for name, common in populations.items():
        for pool in POOLS:
            subsets = [(name, common)]
            if name == "native_low":
                subsets.append(("pool_rescuable", [r for r in common if r[f"{pool}_oracle_iou"] >= good]))
            for subset, rows in subsets:
                if not rows:
                    continue
                native_iou = np.array([r["native_iou"] for r in rows])
                groups = defaultdict(list)
                for j, row in enumerate(rows):
                    groups[row["source"]].append(j)
                for arm in ARMS:
                    iou = np.array([r[f"{pool}_{arm}_iou"] for r in rows])
                    iou = np.maximum(iou, 0.)  # an invalid chosen box is a miss on valid GT
                    by_group = [iou[idx] for idx in groups.values()]
                    policy.append(dict(population=subset, pool=pool, arm=arm, sources=len(groups), frames=len(rows),
                                       mean_iou=float(np.mean([a.mean() for a in by_group])),
                                       good_rate=float(np.mean([(a >= good).mean() for a in by_group])),
                                       low_rate=float(np.mean([(a < low).mean() for a in by_group])),
                                       good_frames=int((iou >= good).sum()),
                                       low_to_good=int(((native_iou < low) & (iou >= good)).sum()),
                                       good_to_below_good=int(((native_iou >= good) & (iou < good)).sum()),
                                       good_to_low=int(((native_iou >= good) & (iou < low)).sum())))
                if name.endswith("replays"):
                    continue
                comparisons = [(pool, "hann"), (pool, "raw")]
                if pool == "dense" and subset != "pool_rescuable":
                    comparisons.append(("peak", "quality"))
                for ref_pool, ref_arm in comparisons:
                    a = np.maximum(0., [r[f"{pool}_quality_iou"] for r in rows])
                    b = np.maximum(0., [r[f"{ref_pool}_{ref_arm}_iou"] for r in rows])
                    for metric, delta in (("iou", a-b), ("good_rate", (a >= good).astype(float)-(b >= good))):
                        deltas = [float(delta[idx].mean()) for _, idx in sorted(groups.items())]
                        contrast = prior.base.s103.contrast(deltas, f"{pool}_quality_minus_{ref_pool}_{ref_arm}", metric, DESIGN)
                        paired.append(dict(population=subset, frames=len(rows), **contrast))
    return policy, paired


def mark_stable(rows):
    length = DESIGN["stable_window_frames"]
    for start in range(0, len(rows) - length + 1, length):
        window = rows[start:start+length]
        if all(r["native_iou"] >= DESIGN["good"] for r in window):
            for row in window:
                row["stable_window"] = start + 1


def stable_report(rows):
    windows = defaultdict(list)
    for row in rows:
        if row["stable_window"]:
            windows[row["sequence"], row["stable_window"]].append(row)
    report = []
    for (seq, start), local in sorted(windows.items()):
        fields = dict(sequence=seq, start=start, end=local[-1]["frame"], source=local[0]["source"])
        for pool in POOLS:
            for arm in ("raw", "quality"):
                key = f"{pool}_{arm}"
                bad = np.array([r[key+"_iou"] < DESIGN["low"] for r in local])
                low_run = np.lib.stride_tricks.sliding_window_view(bad, DESIGN["stable_low_run"]).all(axis=1)
                fields[key+"_low_frames"] = int(bad.sum())
                fields[key+"_low_run"] = bool(low_run.any())
        report.append(fields)
    return report


def native_rows(r121, p121, jobs, entry_for, ap, references, onsets):
    by_source = defaultdict(list)
    for job in jobs:
        seq = job["spec"]["unit"]["sequence"]
        entry = entry_for(seq)
        trace = prior.native_audit.source.prior.prior.load_trace(r121, p121, job)
        n = job["frames"]
        if entry["sequence"] != seq or len(entry["ground_truth"]) != n + 1:
            raise ValueError("Native sequence and labels differ")
        np.testing.assert_array_equal(trace["frame_id"], np.arange(1, n+1))
        rows = []
        group = source_group(seq)
        for i in range(n):
            frame = i+1
            parent, hw = trace["candidate_parents"][i, 0], trace["image_hw"][i]
            crop, _, factor = crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])
            np.testing.assert_array_equal(crop, trace["crop"][i, 0])
            maps = (trace[k][i, 0] for k in ("raw", "size", "offset"))
            row = extract(*maps, parent, factor, hw, trace["boxes"][i], entry["ground_truth"][frame], ap)
            ref = onsets.get((seq, frame))
            if ref is not None:
                check_frame(row, ref)
            rows.append(dict(row, source=group, sequence=seq, frame=frame, state="native",
                             case="", branch="", relapse=False, onset=ref is not None, stable_window=0))
        check_native(rows, references[seq])
        mark_stable(rows)
        by_source[group].extend(rows)
        print(f"[Stage-143] {seq}: verified {n} native states", flush=True)
    return by_source


def replay_rows(root, plan, entry_for, ap, reference):
    rows = []
    for case in plan["cases"]:
        cid, seq = case["case_id"], case["sequence"]
        entry = entry_for(seq)
        data = c.read_cache(root / "replays" / f"{cid}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=cid))
        if data is None or entry["sequence"] != seq:
            raise ValueError(f"Missing frozen branch {cid}")
        np.testing.assert_array_equal(data["offsets"], case["offsets"])
        for i, offset in enumerate(case["offsets"]):
            frame = case["frame"] + offset
            for k, branch in enumerate(prior.replay_audit.ARMS):
                obs = [data[name][i, k] for name in ("raw", "size", "offset", "parent", "factor", "image_hw", "boxes")]
                row = extract(*obs, entry["ground_truth"][frame], ap)
                ref = reference[cid, branch, frame]
                check_frame(row, ref)
                rows.append(dict(row, source=source_group(seq), sequence=seq, frame=frame, state="replay", case=cid,
                                 branch=branch, relapse=ref["relapse"] == "True", onset=False, stable_window=0))
    return rows


def check_results(root):
    report = c.c66.sealed(root / "stage143_decision.json", SCHEMA)
    files = {f"stage143_{name}.csv" for name in TABLES} | {"stage143_models.json"}
    if (report["status"] != STATUS or report["source_stage142"] != SOURCE142 or report["design"] != DESIGN
            or report["native_frames"] != 29047 or report["replay_frames"] != 240 or report["native_onsets"] != 19
            or set(report["artifact_sha256"]) != files or any(report[k] != val for k, val in WORK.items())):
        raise ValueError("Incomplete Stage-143 fixed-pool diagnostic")
    io.verify({str(root / k): val for k, val in report["artifact_sha256"].items()})


def run(root, by_source, replays):
    native = [r for rows in by_source.values() for r in rows]
    if (len(by_source), len(native), sum(r["onset"] for r in native), len(replays)) != (28, 29047, 19, 240):
        raise ValueError("Frozen evaluation roster changed")
    bank = {group: moments(rows) for group, rows in by_source.items()}
    models = {group: fit_peer_sources(bank, group) for group in sorted(bank)}
    decisions = []
    for row in native + replays:
        model = models[row["source"]]
        fields = {k: row[k] for k in ("source", "sequence", "frame", "state", "case", "branch", "relapse", "onset",
                                      "stable_window", "native_iou", "grid_iou", "budget")}
        decisions.append(dict(fields, **decide(row, score(model, row["x"]))))
    policy, paired = summarize(decisions)
    stable = stable_report(decisions)
    focus = [r for r in decisions if r["onset"] or (
        r["state"] == "replay" and r["source"] == "uav0000089" and r["frame"] in (1165, 1166))]
    fit = [dict(held_source=k, fit_sources="|".join(v["fit_sources"]), fit_frames=v["fit_frames"],
                fit_boxes=v["fit_boxes"]) for k, v in models.items()]
    paths = []
    for name, rows in zip(TABLES, (policy, paired, stable, fit, focus, decisions)):
        path = root / f"stage143_{name}.csv"
        publish_table(path, rows)
        paths.append(path)
    model_path = root / "stage143_models.json"
    c.publish_identical(model_path, io.seal(dict(schema=SCHEMA, design=DESIGN, models=models)))
    paths.append(model_path)
    c.publish_identical(root / "stage143_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_out_of_source_fixed_state_selection",
        source_stage142=SOURCE142, design=DESIGN, native_frames=len(native), replay_frames=len(replays),
        native_onsets=sum(r["onset"] for r in native), invalid_native_labels=sum(r["native_iou"] < 0 for r in native),
        held_out_sources=len(models), stable_windows=len(stable),
        quality_stable_windows_with_low_run={p:sum(r[f"{p}_quality_low_run"] for r in stable) for p in POOLS},
        code_sha256={Path(__file__).name:c.file_hash(Path(__file__)), Path(prior.__file__).name:c.file_hash(Path(prior.__file__))},
        artifact_sha256={p.name:c.file_hash(p) for p in paths}, **WORK)))
    check_results(root)


def load(root, r76, r89, r93, r121, r133, r139, r142):
    read = prior.replay_audit.prior.read_table
    prior.check_results(r142)
    d142 = c.c66.sealed(r142 / "stage142_decision.json", prior.SCHEMA)
    if d142["fingerprint"] != SOURCE142:
        raise ValueError("Expected the reviewed Stage-142 run")
    prior.native_audit.check_results(r133)
    prior.replay_audit.check_results(r139)
    s121 = prior.native_audit.source.s121
    p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
    d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", s121.SCHEMA)
    d133 = c.c66.sealed(r133 / "stage133_decision.json", prior.native_audit.SCHEMA)
    p139 = c.c66.sealed(r139 / "manifest.json", prior.replay_audit.SCHEMA)
    d139 = c.c66.sealed(r139 / "stage139_decision.json", prior.replay_audit.SCHEMA)
    if any(report["fingerprint"] != d142[f"source_stage{k}"] for k, report in ((121,d121),(133,d133),(139,d139))) or (
            d121["plan_fingerprint"] != p121["fingerprint"]):
        raise ValueError("Stage-142 lineage differs")
    io.verify({str(r121 / k): val for k, val in d133["native_artifact_sha256"].items()})
    jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=prior.base.job_key)
    s95 = prior.base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    specs = {s["unit"]["sequence"]: s for s in p89["units"]}
    specs.update({j["spec"]["unit"]["sequence"]:j["spec"] for j in jobs})
    ap = prior.base.e63.c.actor_plan(origin["original"])

    entries = {}

    def entry_for(seq):
        if seq not in entries:
            entries[seq] = s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]
        return entries[seq]

    references = defaultdict(dict)
    for row in read(r142 / "stage142_video.csv"):
        references[row["sequence"]][row["group"]] = row
    onsets = {(r["sequence"], int(r["frame"])):r for r in read(r142 / "stage142_onsets.csv")}
    branch = {(r["case_id"], r["arm"], int(r["frame"])):r for r in read(r142 / "stage142_branch_replays.csv")}
    native = native_rows(r121, p121, jobs, entry_for, ap, references, onsets)
    run(root, native, replay_rows(r139, p139, entry_for, ap, branch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage139", "stage142")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if any(getattr(args, k) is None for k in names):
        parser.error("Supply the frozen source directories")
    sources = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-143 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage143_decision.json").exists():
            check_results(root)
        else:
            load(root, *sources)
    print(f"Stage-143: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
