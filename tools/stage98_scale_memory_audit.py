"""Audit Stage-97 size writes, references and first divergences from cached branches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

import stage97_trusted_scale as s97
from stage27_uav_common import scene_geometry, tracking_iou


c, io, s95, s93, e63 = s97.c, s97.io, s97.s95, s97.s93, s97.e63
SCHEMA = "stage98_cached_scale_memory_audit_v1"
STATUS = "COMPLETE_CACHED_SCALE_MEMORY_AUDIT"
SOURCE97 = "54a55559198296e8ed7d8ca719dcc4aa5ac5d7929e413c621ebf56ab337aead3"
TABLES = ("cases", "branch_summary", "frames", "references")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_frame(frames, mask):
    found = np.flatnonzero(mask)
    return int(frames[found[0]]) if len(found) else None


def size_error(box, truth):
    if not np.isfinite(truth).all() or np.any(truth[2:] <= 0):
        return None
    return float(np.abs(np.log(np.asarray(box)[2:]/truth[2:])).mean())


def input_geometry(parent, truth, image_hw, ap):
    if not np.isfinite(truth).all() or np.any(truth[2:] <= 0):
        return None, None
    geometry = scene_geometry(parent, truth, image_hw, ap["search_factor"], ap["search_size"])
    return float(geometry["coverage"]), float(np.sqrt(np.prod(geometry["input_wh"])))


def audit_branch(event, data, source, entry, arm, old, design):
    first, last = int(event["first_audit_frame"]), int(event["last_audit_frame"])
    frames = np.arange(first, last+1)
    np.testing.assert_array_equal(data["frame_id"], frames)
    e63.source_common.prior.audit_calls(data, (len(frames), 4))
    ap = e63.c.actor_plan(old)
    history = list(e63.history_at(source, entry["initial_box"], first, old))
    _, sizes = s97.starting_memory(source, first, design)
    truth = np.asarray(entry["ground_truth"], float)
    overlap = tracking_iou(data["boxes"], truth[first:last+1])
    best = tracking_iou(data["candidate_boxes"], truth[first:last+1, None, :]).max(axis=1)
    rows, references = [], []
    ident = dict(case=s97.case_key(event), sequence=event["sequence"], seed=int(event["seed"]),
                 outcome=event["outcome"], arm=arm)
    low, good = design["low_iou"], design["good_iou"]
    for i, frame in enumerate(frames):
        frame = int(frame)
        parents, refs = s97.parents_at(history, sizes, frame, arm, ap, design)
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        np.testing.assert_array_equal(data["scale_refs"][i], refs)
        np.testing.assert_array_equal(data["scale_active"][i], (refs >= 0).any())
        for q, parent in enumerate(parents):
            crop = s97.crop_geometry(parent, data["image_hw"][i], ap["search_factor"], ap["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i, q], crop)
        changes = s97.scale_changes(history, parents, data["image_hw"][i], ap)
        np.testing.assert_array_equal([data["scale_changed"][i], data["scale_crop_changed"][i]], changes)
        decision = {k: data[k][i] for k in ("selected", "ppo_selected", "memory_write")}
        accepted = s97.scale_write(decision, data["queries"][i])
        np.testing.assert_array_equal(data["scale_written"][i], accepted)
        final = {k: data[k][i] for k in ("candidate_boxes", "candidate_parents", "history_latest", "features", "selected")}
        qualified = s93.s92.trusted(s93.action_bank(final), 0, design)
        np.testing.assert_array_equal(data["memory_write"][i], qualified)
        np.testing.assert_array_equal(data["boxes"][i], data["candidate_boxes"][i, int(data["selected"][i])])
        coverage, target_side = input_geometry(parents[1], truth[frame], data["image_hw"][i], ap)
        bad_refs, good_refs, crop_sensitive = 0, 0, 0
        for ref in refs[refs >= 0]:
            ref = int(ref)
            box = source["boxes"][ref-1] if ref < first else data["boxes"][ref-first]
            ref_iou = float(tracking_iou(box, truth[ref]))
            bad_refs += int(0 <= ref_iou < low)
            good_refs += int(ref_iou >= good)
            without = [(f, size) for f, size in sizes if f != ref]
            alternate, alt_refs = s97.parents_at(history, without, frame, arm, ap, design)
            alternative_coverage, _ = input_geometry(alternate[1], truth[frame], data["image_hw"][i], ap)
            changed_crop = any(not np.array_equal(
                s97.crop_geometry(a, data["image_hw"][i], ap["search_factor"], ap["search_size"])[0],
                s97.crop_geometry(b, data["image_hw"][i], ap["search_factor"], ap["search_size"])[0])
                for a, b in zip(parents, alternate))
            crop_sensitive += int(changed_crop)
            references.append(dict(ident, frame=frame, reference_frame=ref, reference_from_prefix=ref < first,
                reference_iou=ref_iou, reference_log_size_error=size_error(box, truth[ref]),
                actual_q1_log_size_error=size_error(parents[1], truth[frame]),
                drop_q1_log_size_error=size_error(alternate[1], truth[frame]),
                actual_q1_gt_coverage=coverage, drop_q1_gt_coverage=alternative_coverage,
                drop_parent_changed=not np.array_equal(parents, alternate), drop_crop_changed=changed_crop,
                drop_refs="|".join(str(f) for f in alt_refs if f >= 0)))
        peaks = data["features"][i, :, e63.source_common.prior.FEATURES.index("hann_peak")]
        selected = int(data["selected"][i])
        rows.append(dict(ident, frame=frame, iou=float(overlap[i]), best_candidate_iou=float(best[i]),
            scale_written=accepted, low_iou_write=bool(accepted and 0 <= overlap[i] < low),
            good_iou_write=bool(accepted and overlap[i] >= good), scale_active=bool((refs >= 0).any()),
            refs="|".join(str(f) for f in refs if f >= 0), low_iou_refs=bad_refs, good_iou_refs=good_refs,
            crop_sensitive_drops=crop_sensitive, output_log_size_error=size_error(data["boxes"][i], truth[frame]),
            q1_parent_log_size_error=size_error(parents[1], truth[frame]),
            q1_input_target_side_px=target_side,
            q1_gt_coverage=coverage, selected_hann_peak=float(peaks[selected]),
            strongest_hann_peak=float(peaks.max()), selected=selected, ppo_selected=int(data["ppo_selected"][i])))
        s97.update_sizes(sizes, frame, data["boxes"][i], accepted, design)
        history = (history+[data["boxes"][i]])[-ap["design"]["history"]:]
    summary = dict(ident, frames=len(frames), writes=sum(r["scale_written"] for r in rows),
        low_iou_writes=sum(r["low_iou_write"] for r in rows), good_iou_writes=sum(r["good_iou_write"] for r in rows),
        frames_reading_low_iou_refs=sum(r["low_iou_refs"] > 0 for r in rows),
        low_iou_reference_reads=sum(r["low_iou_refs"] for r in rows),
        reference_reads=len(references), crop_sensitive_drops=sum(r["drop_crop_changed"] for r in references),
        low_iou_crop_sensitive_drops=sum(0 <= r["reference_iou"] < low and r["drop_crop_changed"] for r in references),
        first_low_iou_write=next((r["frame"] for r in rows if r["low_iou_write"]), None),
        first_low_iou_reference_read=next((r["frame"] for r in rows if r["low_iou_refs"]), None))
    return summary, rows, references


def compare(event, control, trusted, summaries, rows):
    frames = trusted["frame_id"]
    count = len(frames)
    control_iou = np.asarray([r["iou"] for r in rows["control"]])
    trusted_iou = np.asarray([r["iou"] for r in rows["trusted_scale"]])
    control_best = np.asarray([r["best_candidate_iou"] for r in rows["control"]])
    trusted_best = np.asarray([r["best_candidate_iou"] for r in rows["trusted_scale"]])
    good, low = s97.DESIGN["good_iou"], s97.DESIGN["low_iou"]
    differences = dict(
        first_parent_difference=(control["candidate_parents"] != trusted["candidate_parents"]).reshape(count, -1).any(axis=1),
        first_crop_geometry_difference=(control["crop"] != trusted["crop"]).reshape(count, -1).any(axis=1),
        first_crop_pixel_difference=(control["crop_sha256"] != trusted["crop_sha256"]).reshape(count, -1).any(axis=1),
        first_output_difference=(control["boxes"] != trusted["boxes"]).any(axis=1),
        first_integer_output_difference=(np.trunc(control["boxes"]) != np.trunc(trusted["boxes"])).any(axis=1),
        first_slot_difference=control["selected"] != trusted["selected"],
        first_good_candidate_lost=(control_best >= good) & (trusted_best >= 0) & (trusted_best < good),
        first_good_candidate_gained=(trusted_best >= good) & (control_best >= 0) & (control_best < good),
        first_good_output_became_low=(control_iou >= good) & (trusted_iou >= 0) & (trusted_iou < low),
        first_low_output_became_good=(trusted_iou >= good) & (control_iou >= 0) & (control_iou < low))
    outcome = "unchanged"
    a, b = event["control_aligned_recovery"] == "True", event["trusted_scale_aligned_recovery"] == "True"
    if a != b:
        outcome = "gained" if b else "lost"
    result = dict(case=s97.case_key(event), sequence=event["sequence"], seed=int(event["seed"]),
        source_outcome=event["outcome"], transition=outcome, first_frame=int(frames[0]), last_frame=int(frames[-1]),
        ppo_recovery_frame=int(event["ppo_recovery_frame"]),
        control_matched_good_frame=event["control_matched_good_frame"],
        trusted_scale_matched_good_frame=event["trusted_scale_matched_good_frame"],
        **{k: first_frame(frames, mask) for k, mask in differences.items()})
    for arm in s97.ARMS:
        for k in ("writes", "low_iou_writes", "good_iou_writes", "frames_reading_low_iou_refs",
                  "low_iou_reference_reads", "reference_reads", "crop_sensitive_drops",
                  "low_iou_crop_sensitive_drops",
                  "first_low_iou_write", "first_low_iou_reference_read"):
            result[f"{arm}_{k}"] = summaries[arm][k]
    first = result["first_output_difference"]
    for arm in s97.ARMS:
        position = int(first-frames[0]) if first is not None else None
        result[f"{arm}_first_difference_iou"] = rows[arm][position]["iou"] if position is not None else None
        result[f"{arm}_first_difference_best_iou"] = rows[arm][position]["best_candidate_iou"] if position is not None else None
        result[f"{arm}_first_difference_low_iou_refs"] = rows[arm][position]["low_iou_refs"] if position is not None else None
        result[f"{arm}_first_difference_refs"] = rows[arm][position]["refs"] if position is not None else None
    return result


def check_results(root):
    report = c.c66.sealed(root / "stage98_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage97_fingerprint"] != SOURCE97
            or report["cases"] != 25 or report["branches"] != 50 or report["frame_rows"] != 6324
            or set(report["table_sha256"]) != {f"stage98_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-98 audit")
    io.verify(report["source_sha256"])
    io.verify(report["code_sha256"])
    io.verify({str(root / name): sha for name, sha in report["table_sha256"].items()})


def run(root97, root93, root89, plan97, plan93, plan89, report89, source):
    windows = read_csv(root97 / "analysis/stage97_windows.csv")
    by_case = {s97.case_key(e): e for e in windows}
    if len(by_case) != len(plan97["events"]):
        raise ValueError("Incomplete Stage-97 window table")
    specs = {(s["unit"]["sequence"], s["unit"]["seed"]): s for s in plan89["units"]}
    groups = defaultdict(list)
    for event in plan97["events"]:
        groups[event["sequence"], int(event["seed"])].append(event)
    cases, branches, frames, references = [], [], [], []
    for key, events in groups.items():
        spec = specs[key]
        unit = spec["unit"]
        entry, traces = s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
        for event in events:
            name = s97.case_key(event)
            data, stats, local_rows = {}, {}, {}
            for arm in s97.ARMS:
                path = root97 / "branches" / (name+"__"+arm+".npz")
                meta = dict(plan_fingerprint=plan97["fingerprint"], unit=unit, case=name, arm=arm)
                data[arm] = s95.s94.required_cache(path, meta)
                stats[arm], local_rows[arm], refs = audit_branch(event, data[arm], traces["gated"], entry,
                                                                arm, source["original"], plan97["design"])
                metrics, _ = s97.describe(data[arm], entry, event, plan97["design"])
                for field in ("mean_iou", "endpoint_iou", "aligned_recovery", "matched_good_frame", "scale_writes"):
                    actual = metrics[field]
                    recorded = by_case[name][f"{arm}_{field}"]
                    if isinstance(actual, (bool, np.bool_)):
                        matches = str(bool(actual)) == recorded
                    elif actual is None:
                        matches = recorded == ""
                    else:
                        matches = np.isclose(float(recorded), actual, rtol=0, atol=1e-12)
                    if not matches:
                        raise ValueError(f"Stage-97 metric differs: {name} {arm} {field}")
                branches.append(stats[arm])
                frames.extend(local_rows[arm])
                references.extend(refs)
            cases.append(compare(by_case[name], data["control"], data["trusted_scale"], stats, local_rows))
            print(f"[Stage-98] {name}", flush=True)
    return cases, branches, frames, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage97", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage97)):
        parser.error("Supply Stage-76, 89, 93 and 97 caches")
    root93, root76, root89, root97 = [p.expanduser().resolve() for p in
                                     (args.stage93, args.stage76, args.stage89, args.stage97)]
    if any(root == p or root in p.parents or p in root.parents for p in (root93, root76, root89, root97)):
        parser.error("Use a separate Stage-98 output directory")
    with s93.writer_lock(root):
        if (root / "stage98_decision.json").exists():
            check_results(root)
            return
        s97.check_results(root97)
        plan97 = c.c66.sealed(root97 / "manifest.json", s97.SCHEMA)
        report97 = c.c66.sealed(root97 / "analysis/stage97_decision.json", s97.SCHEMA)
        if report97["fingerprint"] != SOURCE97:
            raise ValueError("Need the frozen Stage-97 result")
        plan93, _, plan89, report89, source = s95.s94.load_run(root93, root76, root89)
        paths = [root97 / "manifest.json", root97 / "analysis/stage97_decision.json",
                 root97 / "analysis/stage97_windows.csv"]
        source_hashes = {str(p): c.file_hash(p) for p in paths} | plan97["source_sha256"]
        source_hashes.update({str(root97 / p): sha for p, sha in report97["cache_sha256"].items()})
        code = [Path(__file__), Path(__file__).with_name("run_stage98_scale_memory_audit.sh"), Path(s97.__file__)]
        code_hashes = {str(p.resolve()): c.file_hash(p) for p in code} | plan97["code_sha256"]
        tables = run(root97, root93, root89, plan97, plan93, plan89, report89, source)
        io.verify(source_hashes)
        io.verify(code_hashes)
        hashes = {}
        for name, rows in zip(TABLES, tables):
            path = root / f"stage98_{name}.csv"
            s97.s86.publish_table(path, rows)
            hashes[path.name] = c.file_hash(path)
        c.publish_identical(root / "stage98_decision.json", io.seal(dict(
            schema=SCHEMA, status=STATUS, source_stage97_fingerprint=SOURCE97,
            cases=len(tables[0]), branches=len(tables[1]), frame_rows=len(tables[2]), reference_rows=len(tables[3]),
            source_sha256=source_hashes, code_sha256=code_hashes, table_sha256=hashes,
            image_reads=0, new_search_calls=0, training_updates=0, confirmation_inference_calls=0)))
        check_results(root)
        print(f"Stage-98: {STATUS} ({len(tables[0])} cases, {len(tables[3])} reference drops)")


if __name__ == "__main__":
    main()
