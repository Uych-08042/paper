"""Audit persistence and failure timing in the frozen Stage-188 trajectories."""

import argparse
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
from stage184_center_mapping import cache_reference
import stage188_temporal_support as prior
from stage27_uav_common import inside, read_boxes


c, io, gain, online = prior.c, prior.io, prior.gain, prior.online
SCHEMA = "stage189_continuation_audit_v1"
SOURCE188 = "1c030e7fcec60b73aeefe900ed8310e10ebdabd86b5c68e574131c4eb022e8d8"
VALID = ("valid", "added_failure", "event_harm", "protected_rescue", "delta128_or_shorter", "delta32_or_shorter")


def read_cache(root, decision, kind, key):
    meta = dict(plan_fingerprint=decision["plan_fingerprint"], kind=kind, key=key)
    return cache_reference(root, decision, f"{kind}/{key}.npz", meta)


def native_event_scope(sequence, group, values, cases):
    selected = {r["frame"] for r in cases if r["cohort"] == "onset"}
    rows = []
    for event in online.failure_events(values):
        f = event["start"] + 1
        after = len(values) - f
        enough_history = f > prior.HISTORY
        enough_tail = after >= prior.LONG
        valid_window = enough_history and enough_tail and bool(np.all(values[f - prior.HISTORY - 1:f + prior.LONG] >= 0.))
        rows.append(dict(sequence=sequence, group=group, onset_frame=f, confirmation_frame=event["confirmation"] + 1,
                         event_end_frame=event["end"] + 1,
                         recovery_frame=None if event["recovery"] is None else event["recovery"] + 1,
                         censor=event["censor"], preceding_frames=f - 1, observed_successors=after,
                         enough_history=enough_history, enough_tail=enough_tail,
                         eligible_window=valid_window, selected=f in selected))
    return rows


def first_frame(flags, start):
    hits = np.flatnonzero(flags)
    return start + int(hits[0]) if hits.size else None


def local_metrics(values, prefix, native, frame):
    low = float(online.TRACKING["failure_iou"])
    good = float(online.TRACKING["recovery_iou"])
    short = values >= good
    first_below_good = first_frame(~short, frame)
    initial_good_run = len(values) if first_below_good is None else first_below_good - frame
    delta = values[1:] - native[1:]
    relevant = [e for e in online.failure_events(np.r_[prefix, values]) if e["end"] + 1 >= frame]
    first = relevant[0] if relevant else None
    first_recovery = next((int(e["recovery"] + 1) for e in relevant if e["recovery"] is not None), None)
    relapse = next((int(e["start"] + 1) for e in relevant if first_recovery is not None and e["start"] + 1 > first_recovery), None)
    summary = dict(anchor_iou=float(values[0]), successor1_iou=float(values[1]), endpoint_iou=float(values[-1]),
                   good_run_from_anchor=initial_good_run, good_successors=int(short[1:].sum()),
                   low_successors=int(np.sum(values[1:] < low)), first_low_successor=first_frame(values[1:] < low, frame + 1),
                   first_event_onset=None if first is None else first["start"] + 1,
                   first_event_confirmed=None if first is None else first["confirmation"] + 1,
                   first_recovery=first_recovery, relapse_onset=relapse, intersecting_events=len(relevant),
                   terminal_failure=bool(relevant and relevant[-1]["censor"] == "video_end"),
                   mean_successor_iou=float(values[1:].mean()), tail_delta_native=float(delta.mean()),
                   higher_than_native_successors=int(np.sum(delta > 1e-9)))
    return summary, relevant


def branch_tables(root, decision, case, gt, native, candidates, choices):
    key, f, end = (case[k] for k in ("id", "frame", "end"))
    probe = read_cache(root, decision, "probe", key)
    cache = read_cache(root, decision, "tail", key)
    names = list(probe["names"])
    paths = cache["paths"][cache["inverse"]]
    native_segment = native[f - 1:end]
    if (set(names) != set(candidates) or len(set(names)) != len(names) or set(choices) != set(prior.METHODS)
            or not set(choices.values()) <= set(names) or paths.shape != (len(names), end - f + 1, 4)):
        raise ValueError(f"Inconsistent Stage-188 candidate membership at {key}")
    np.testing.assert_array_equal(paths[:, 0], probe["boxes"])
    paths = np.concatenate((native_segment[None], paths))
    values = online.tracking_iou(paths, gt[f:end + 1])
    if not np.isfinite(paths).all() or (paths[..., 2:] <= 0).any() or (values < 0).any():
        raise ValueError(f"Invalid trajectory at {key}")
    prefix = online.tracking_iou(native[:f - 1], gt[1:f])
    checks = prior.source.tail_summary(paths, gt[f:end + 1], prefix, 0, range(1, len(paths)))
    for i, (name, check) in enumerate(zip(names, checks, strict=True), start=1):
        row = candidates[name]
        if str(float(values[i, 0])) != row["anchor_iou"] or any(str(check[k]) != row[k] for k in VALID):
            raise ValueError(f"Stage-188 outcome did not reproduce for {key}/{name}")

    summaries, events, frames = [], [], []
    short_tail = slice(1, prior.source.SHORT + 1)
    for i, (name, iou, boxes) in enumerate(zip(("native", *names), values, paths, strict=True)):
        roles = "native" if name == "native" else "|".join(k for k in prior.METHODS if choices[k] == name)
        identity = dict(case_id=key, sequence=case["sequence"], anchor_frame=f, branch=str(name), policy_roles=roles)
        outcome, observed = local_metrics(iou, prefix, values[0], f)
        flags = prior.source.event.event_labels(prefix, values[[0, i]])
        summaries.append(dict(identity, **outcome, tail32_delta_native=float((iou[short_tail] - values[0, short_tail]).mean()),
                              inherited_failure=flags["inherited_failure"], prevented_terminal=flags["prevented_terminal"],
                              rescued_recovery=flags["rescued_recovery"], lost_recovery=flags["lost_recovery"],
                              added_failure=flags["added_failure"], event_harm=flags["event_harm"],
                              protected_rescue=checks[i - 1]["protected_rescue"] if i else False))
        for e in observed:
            events.append(dict(identity, event_onset=e["start"] + 1, confirmation=e["confirmation"] + 1,
                               event_end=e["end"] + 1, recovery=None if e["recovery"] is None else e["recovery"] + 1,
                               censor="window_end" if e["censor"] == "video_end" else e["censor"]))
        for j, (score, box, target) in enumerate(zip(iou, boxes, gt[f:end + 1], strict=True)):
            centers = box[:2] + .5 * box[2:] - target[:2] - .5 * target[2:]
            frames.append(dict(identity, frame=f + j, iou=float(score), delta_native=float(score - values[0, j]),
                               center_error_gt_units=float(np.linalg.norm(centers) / np.sqrt(target[2:].prod())),
                               width_ratio=float(box[2] / target[2]), height_ratio=float(box[3] / target[3])))
    return summaries, events, frames


def audit(root, dataset, plan, decision, collection, progress):
    source_tables = {k: prior.risk.read_verified(root, decision, 188, k) for k in ("candidates", "choices", "cases")}
    if [{k: str(v) for k, v in row.items()} for row in collection["cases"]] != source_tables["cases"]:
        raise ValueError("Stage-188 case roster differs between collection and report")
    expected = {"uav0000107_01763_s": 161, "uav0000173_00781_s": 502}
    onsets = {case["sequence"]: case["frame"] for case in collection["cases"] if case["cohort"] == "onset"}
    if onsets != expected or len(plan["clips"]) != decision["additional_groups"]:
        raise ValueError("Unexpected Stage-188 onset or development roster")
    tables = dict(native_events=[], branches=[], branch_events=[], frames=[])
    for clip in plan["clips"]:
        seq = clip["sequence"]
        annotation = inside(dataset, clip["annotation"])
        if c.file_hash(annotation) != clip["annotation_sha256"]:
            raise ValueError(f"Stage-188 annotation hash differs at {seq}")
        gt = read_boxes(annotation)
        trace = read_cache(root, decision, "native", seq)
        if len(gt) != clip["tracked_frames"] + 1:
            raise ValueError(f"Stage-188 annotation length changed at {seq}")
        prior.previous.audit(trace, clip["tracked_frames"], gt[0], "native", 0.)
        values = online.tracking_iou(trace["boxes"], gt[1:])
        cases = prior.pick_windows(seq, values)
        if cases != [case for case in collection["cases"] if case["sequence"] == seq]:
            raise ValueError(f"Stage-188 roster did not reproduce at {seq}")
        tables["native_events"].extend(native_event_scope(seq, clip["group"], values, cases))
        for case in cases:
            if case["cohort"] != "onset":
                continue
            key = case["id"]
            pool = {r["arm"]: r for r in source_tables["candidates"] if r["case_id"] == key}
            choices = {r["method"]: r["arm"] for r in source_tables["choices"] if r["case_id"] == key}
            results = branch_tables(root, decision, case, gt, trace["boxes"], pool, choices)
            for name, rows in zip(("branches", "branch_events", "frames"), results, strict=True):
                tables[name].extend(rows)
        progress.advance()
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage188-root", "dataset-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    source, dataset, target = (p.expanduser().resolve() for p in (args.stage188_root, args.dataset_root, args.output_dir))
    if any(target == p or target in p.parents or p in target.parents for p in (source, dataset)):
        parser.error("Use a separate output directory")
    if not (source / "stage188_decision.json").is_file():
        parser.error(f"Missing completed Stage-188 result directory: {source}")
    decision = c.c66.sealed(source / "stage188_decision.json", prior.SCHEMA)
    plan = c.c66.sealed(source / "manifest.json", prior.SCHEMA)
    collection_path = source / "collection.json"
    io.verify({str(collection_path): decision["artifact_sha256"]["collection.json"]})
    collection = c.c66.sealed(collection_path, prior.SCHEMA)
    if (decision["fingerprint"] != SOURCE188 or plan["fingerprint"] != decision["plan_fingerprint"]
            or collection["plan_fingerprint"] != plan["fingerprint"] or plan["horizon"] != prior.LONG):
        raise ValueError("Need the completed Stage-188 run bound to this audit")
    with gain.prior.prior.writer_lock(target):
        progress = OverallProgress("Stage-189", len(plan["clips"]) + 1, unit="work")
        progress.report("Auditing cached native and candidate trajectories")
        tables = audit(source, dataset, plan, decision, collection, progress)
        hashes = gain.prior.publish(target, tables, stage="stage189")
        c.publish_identical(target / "stage189_decision.json", io.seal(dict(
            schema=SCHEMA, status="COMPLETE_STAGE189_CONTINUATION_AUDIT", source_stage188=SOURCE188,
            source_plan_fingerprint=plan["fingerprint"], audit_sha256=c.file_hash(Path(__file__)),
            new_searches=0, images_read=0, training_performed=False,
            rows={name: len(rows) for name, rows in tables.items()}, table_sha256=hashes)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
