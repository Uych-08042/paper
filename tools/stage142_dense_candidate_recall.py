"""Compare peak and dense candidate recall with a matched candidate budget."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage133_native_failure_audit as native_audit
import stage139_post_write_diagnostic as replay_audit
from stage27_uav_collect import writer_lock
from stage27_uav_common import crop_geometry, tracking_iou
from stage28_response_common import DESIGN as RESPONSE_DESIGN, continuous_iou, dense_boxes, local_peak_mask, peak_candidates
from stage46_shift_audit import publish_table


base, c, io = native_audit.base, native_audit.c, native_audit.io
SCHEMA = "stage142_dense_candidate_recall_v1"
STATUS = "COMPLETE_MATCHED_DENSE_CANDIDATE_RECALL"
SOURCE133 = replay_audit.prior.SOURCE133
SOURCE139 = "ff5fba73d06336b895809a5d385d828dddd0e2540e7e81e47ec4846cdd29208b"
DESIGN = dict(max_candidates=8, good_iou=.50, low_iou=.20,
              box_nms_iou=RESPONSE_DESIGN["box_nms_iou"], bootstrap_seed=23142, bootstrap_samples=20000)
TABLES = ("summary", "video", "onsets", "branch_replays", "scope")


def rank_cells(scores, peaks_only):
    flat = scores.ravel()
    eligible = local_peak_mask(scores).ravel() if peaks_only else flat > 0
    cells = np.flatnonzero(eligible)
    order = cells[np.lexsort((cells, -flat[cells]))]
    rank = np.full(flat.size, flat.size + 1, dtype=int)
    rank[order] = np.arange(len(order))
    return rank


def candidate_pool(raw, weighted, boxes, native, peaks_only):
    scores = (rank_cells(raw, peaks_only), rank_cells(weighted, peaks_only))
    best_rank = np.minimum(*scores)
    eligible = np.flatnonzero(best_rank <= raw.size)
    order = eligible[np.lexsort((eligible, scores[1][eligible], best_rank[eligible]))]
    selected = [int(native)]
    threshold = DESIGN["box_nms_iou"]
    suppressed = continuous_iou(boxes[native], boxes) > threshold
    suppressed[native] = True
    for cell in order:
        if suppressed[cell]:
            continue
        selected.append(int(cell))
        if len(selected) == DESIGN["max_candidates"]:
            break
        suppressed |= continuous_iou(boxes[cell], boxes) > threshold
    return selected


def compare(raw, size, offset, parent, factor, hw, box, truth, ap):
    raw = np.asarray(raw, np.float32)
    weighted = raw * np.asarray(ap["window"], np.float32)
    native = int(np.argmax(weighted))
    grid = dense_boxes(size, offset, parent, factor, ap["search_size"], hw)
    error = float(np.abs(grid[native] - box).max())
    if error > ap["design"]["box_atol"]:
        raise ValueError(f"Original response decode disagrees by {error:.8g} pixels")
    grid[native] = box
    iou = tracking_iou(grid, truth)
    legacy = sorted(set(peak_candidates(raw, grid)) | set(peak_candidates(weighted, grid)))
    peaks = candidate_pool(raw, weighted, grid, native, True)
    dense = candidate_pool(raw, weighted, grid, native, False)
    budget = min(len(peaks), len(dense))
    peaks, dense = peaks[:budget], dense[:budget]
    best_cell = int(iou.argmax())
    return dict(native_iou=float(iou[native]), full_grid_best_iou=float(iou.max()),
                legacy_best_iou=float(iou[legacy].max()), legacy_count=len(legacy), matched_budget=budget,
                matched_peak_best_iou=float(iou[peaks].max()), matched_dense_best_iou=float(iou[dense].max()),
                peak_cells="|".join(map(str, peaks)), dense_cells="|".join(map(str, dense)),
                best_grid_cell=best_cell, peak_contains_best_cell=best_cell in peaks, dense_contains_best_cell=best_cell in dense,
                max_decode_error=error)


def tally(rows):
    flags = [(r["matched_peak_best_iou"] >= DESIGN["good_iou"], r["matched_dense_best_iou"] >= DESIGN["good_iou"]) for r in rows]
    return dict(frames=len(rows), native_good=sum(r["native_iou"] >= DESIGN["good_iou"] for r in rows),
                grid_good=sum(r["full_grid_best_iou"] >= DESIGN["good_iou"] for r in rows),
                legacy_good=sum(r["legacy_best_iou"] >= DESIGN["good_iou"] for r in rows),
                peak_good=sum(a for a, _ in flags), dense_good=sum(b for _, b in flags),
                dense_added_good=sum(b and not a for a, b in flags), dense_lost_good=sum(a and not b for a, b in flags),
                mean_budget=float(np.mean([r["matched_budget"] for r in rows])) if rows else None)


def summarize(all_rows, onsets, replays):
    good, low = DESIGN["good_iou"], DESIGN["low_iou"]
    groups = {"native_all": all_rows, "native_below_good": [r for r in all_rows if r["native_iou"] < good],
              "native_low": [r for r in all_rows if r["native_iou"] < low],
              "native_low_grid_good": [r for r in all_rows if r["native_iou"] < low and r["full_grid_best_iou"] >= good],
              "native_onsets": onsets, "stage139_branch_replays": replays}
    summary, video = [], []
    for group, rows in groups.items():
        by_video = defaultdict(list)
        for row in rows:
            by_video[row["sequence"]].append(row)
        counts = []
        for name, subset in sorted(by_video.items()):
            record = tally(subset)
            counts.append((record["dense_good"] - record["peak_good"]) / record["frames"])
            video.append(dict(group=group, sequence=name, **record))
        interval = (base.s103.contrast(counts, "dense_minus_peaks", "candidate_recall", DESIGN)
                    if len(counts) > 1 and group != "stage139_branch_replays" else None)
        summary.append(dict(group=group, videos=len(by_video), **tally(rows),
                            video_mean_recall_delta=float(np.mean(counts)) if counts else None,
                            ci_low=interval["ci_low"] if interval else None, ci_high=interval["ci_high"] if interval else None))
    return summary, video


def native_rows(trace, job, entry, ap):
    seq = job["spec"]["unit"]["sequence"]
    if entry["sequence"] != seq or len(entry["ground_truth"]) != job["frames"] + 1:
        raise ValueError("Frozen Stage-121 native video changed")
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, job["frames"] + 1))
    rows = []
    for i in range(job["frames"]):
        parent, hw = trace["candidate_parents"][i, 0], trace["image_hw"][i]
        crop, _, factor = crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])
        np.testing.assert_array_equal(crop, trace["crop"][i, 0])
        raw, size, offset = (trace[k][i, 0] for k in ("raw", "size", "offset"))
        rows.append(dict(sequence=seq, frame=i+1, **compare(raw, size, offset, parent, factor, hw,
                                                            trace["boxes"][i], entry["ground_truth"][i+1], ap)))
    return rows


def replay_rows(r139, p139, load_entry, ap):
    reference = {(r["case_id"], r["arm"], int(r["offset"])): r
                 for r in replay_audit.prior.read_table(r139 / "stage139_replay.csv")}
    rows = []
    for case in p139["cases"]:
        cid, seq = case["case_id"], case["sequence"]
        entry = load_entry(seq)
        data = c.read_cache(r139 / "replays" / f"{cid}.npz", dict(plan_fingerprint=p139["fingerprint"], case_id=cid))
        if data is None or entry["sequence"] != seq:
            raise ValueError(f"Missing or mismatched Stage-139 replay: {cid}")
        np.testing.assert_array_equal(data["offsets"], case["offsets"])
        for i, j in enumerate(case["offsets"]):
            frame = case["frame"] + j
            for k, arm in enumerate(replay_audit.ARMS):
                raw, size, offset, parent, factor, hw, box = (data[name][i, k] for name in
                    ("raw", "size", "offset", "parent", "factor", "image_hw", "boxes"))
                result = compare(raw, size, offset, parent, float(factor), hw, box, entry["ground_truth"][frame], ap)
                ref = reference[cid, arm, j]
                np.testing.assert_allclose([result[a] for a in ("native_iou", "full_grid_best_iou", "legacy_best_iou")],
                                           [float(ref[a]) for a in ("replay_iou", "dense_best_iou", "peak_pool_best_iou")], rtol=0, atol=1e-12)
                rows.append(dict(case_id=cid, sequence=seq, frame=frame, offset=j, arm=arm,
                                 relapse=ref["in_relapse_neighborhood"] == "True", **result))
    if len(rows) != len(reference) or len(rows) != 240:
        raise ValueError("Missing Stage-139 replay rows")
    return rows


def check_results(root):
    report = c.c66.sealed(root / "stage142_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage139"] != SOURCE139 or report["source_stage133"] != SOURCE133
            or report["design"] != DESIGN or (report["frames"], report["onsets"], report["branch_replays"]) != (29047, 19, 240)
            or set(report["table_sha256"]) != {f"stage142_{key}.csv" for key in TABLES}):
        raise ValueError("Incomplete Stage-142 diagnostic")
    io.verify({str(root / k): v for k, v in report["table_sha256"].items()})


def run(root, r121, p121, jobs, load_entry, ap, load_replays, event_index):
    all_rows, onsets, scope = [], [], []
    for job in jobs:
        seq = job["spec"]["unit"]["sequence"]
        trace = native_audit.source.prior.prior.load_trace(r121, p121, job)
        rows = native_rows(trace, job, load_entry(seq), ap)
        for row in rows:
            ref = event_index.get((seq, row["frame"]))
            if ref is not None:
                np.testing.assert_allclose([row["native_iou"], row["full_grid_best_iou"]],
                                           [float(ref["native_iou"]), float(ref["dense_best_iou"])], rtol=0, atol=1e-12)
                onsets.append(dict(row, category=ref["category"]))
        all_rows.extend(rows)
        scope.append(dict(sequence=seq, frames=len(rows), max_decode_error=max(r["max_decode_error"] for r in rows)))
        print(f"[Stage-142] {seq}: {len(rows)} native frames", flush=True)
    replays = load_replays()
    if (len(all_rows), len(onsets), len(replays)) != (29047, 19, 240):
        raise ValueError("Stage-142 roster changed")
    summary, video = summarize(all_rows, onsets, replays)
    tables = {}
    for name, rows in zip(TABLES, (summary, video, onsets, replays, scope)):
        path = root / f"stage142_{name}.csv"
        publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage142_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_fixed_state_candidate_supply",
        source_stage133=SOURCE133, source_stage139=SOURCE139, source_stage121=native_audit.source.SOURCE121,
        design=DESIGN, videos=len(jobs), frames=len(all_rows), onsets=len(onsets), branch_replays=len(replays),
        image_reads=0, new_search_calls=0, training_updates=0, state_updates=0,
        code_sha256=c.file_hash(Path(__file__)), table_sha256=tables)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage139")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories")
    r76, r89, r93, r121, r133, r139 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-142 directory")
    if not (r139 / "stage139_decision.json").is_file():
        parser.error("Need the completed Stage-139 server replays")
    with writer_lock(root):
        if (root / "stage142_decision.json").is_file():
            check_results(root)
            return
        native_audit.check_results(r133)
        replay_audit.check_results(r139)
        s121 = native_audit.source.s121
        p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
        d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", s121.SCHEMA)
        d133 = c.c66.sealed(r133 / "stage133_decision.json", native_audit.SCHEMA)
        p139 = c.c66.sealed(r139 / "manifest.json", replay_audit.SCHEMA)
        d139 = c.c66.sealed(r139 / "stage139_decision.json", replay_audit.SCHEMA)
        if (d133["fingerprint"] != SOURCE133 or d139["fingerprint"] != SOURCE139
                or d121["fingerprint"] != native_audit.source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]):
            raise ValueError("Need the archived Stage-121/133/139 sources")
        io.verify({str(r121 / k): v for k, v in d133["native_artifact_sha256"].items()})
        jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=base.job_key)
        if len(jobs) != 28:
            raise ValueError("Expected the frozen 28-video roster")
        events = replay_audit.prior.read_table(r133 / "stage133_events.csv")
        event_index = {(r["sequence"], int(r["onset_frame"])): r for r in events}
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        specs = {s["unit"]["sequence"]: s for s in p89["units"]}
        specs.update({j["spec"]["unit"]["sequence"]: j["spec"] for j in jobs})
        ap = base.e63.c.actor_plan(origin["original"])

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        run(root, r121, p121, jobs, load_entry, ap, lambda: replay_rows(r139, p139, load_entry, ap), event_index)
    print(f"Stage-142: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
