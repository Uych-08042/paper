"""Trace Stage-93 recovery differences and replay read-side memory on fixed candidates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import stage93_appearance_closed_loop as s93
from stage27_uav_common import tracking_iou


c, io, s86, s89 = s93.c, s93.io, s93.s86, s93.s89
SCHEMA = "stage94_appearance_recovery_audit_v1"
STATUS = "COMPLETE_CACHED_APPEARANCE_RECOVERY_AUDIT"
SOURCE = "bf9e1e7f757399ebf4c16ce2a6b691fab80789a7de224e7599b8caff11051000"
LEAD = 32
TABLES = ("events", "frames", "reference_drops")


def first_frame(frames, mask):
    matches = frames[mask]
    return int(matches[0]) if matches.size else None


def memory_replay(gated, frame, design):
    """Rebuild the exact past-write queue, then suppress one read source at a time."""
    i = frame - 1
    row = {key: gated[key][i] for key in ("candidate_boxes", "candidate_parents", "history_latest", "features")}
    row["selected"] = int(gated["ppo_selected"][i])
    memory = []
    for j in range(max(0, i - design["lookback"]), i):
        if gated["memory_write"][j]:
            slot = int(gated["selected"][j] != gated["ppo_selected"][j])
            memory.append((j + 1, gated["queries"][j, slot]))
    queries, anchor = gated["queries"][i], gated["anchor"]
    actual = s93.choose_write(row, queries, anchor, memory.copy(), frame, "gated", design)
    for key, value in actual.items():
        message = f"Stage-93 frame {frame}: {key}"
        if key in ("patch_margin", "margin"):
            np.testing.assert_allclose(value, gated[key][i], rtol=0, atol=1e-12, equal_nan=True, err_msg=message)
        else:
            np.testing.assert_array_equal(value, gated[key][i], err_msg=message)
    template = s93.choose_write(row, queries, anchor, [], frame, "patch", design)
    drops = []
    if actual["used_history"]:
        by_frame = dict(memory)
        for ref in actual["refs"]:
            if ref < 0 or not np.isfinite(by_frame[ref]).all():
                continue
            kept = [(f, feature) for f, feature in memory if f != ref]
            alternative = s93.choose_write(row, queries, anchor, kept, frame, "gated", design)
            drops.append((int(ref), alternative))
    return template, drops


def audit_unit(spec, ppo, gated, truth, design):
    count = len(truth)
    frames = np.arange(1, count + 1)
    for trace in (ppo, gated):
        np.testing.assert_array_equal(trace["frame_id"], frames)
    ii = np.arange(count)
    ppo_iou, gated_iou = (tracking_iou(trace["boxes"], truth) for trace in (ppo, gated))
    ppo_bank, gated_bank = (tracking_iou(trace["candidate_boxes"], truth[:, None, :]) for trace in (ppo, gated))
    ppo_on_gated = gated_bank[ii, gated["ppo_selected"]]
    ppo_best, gated_best = ppo_bank.max(axis=1), gated_bank.max(axis=1)
    box_changed = (ppo["boxes"] != gated["boxes"]).any(axis=1)
    geometry_changed = (ppo["crop"] != gated["crop"]).reshape(count, -1).any(axis=1)
    crop_changed = (ppo["crop_sha256"] != gated["crop_sha256"]).reshape(count, -1).any(axis=1)
    switched = gated["selected"] != gated["ppo_selected"]
    good_iou, low_iou, width = (design[key] for key in ("good_iou", "low_iou", "recovery_run"))
    good = (np.lib.stride_tricks.sliding_window_view(gated_iou >= good_iou, width).all(axis=1)
            if count >= width else np.empty(0, dtype=bool))
    events, rows, drops = [], [], []
    replay_cache = {}
    unit = spec["unit"]
    for event in s93.recovery_comparison(ppo_iou, gated_iou, design):
        onset, recovery = (event[k] for k in ("ppo_failure_frame", "ppo_recovery_frame"))
        ident = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"],
                     **event, outcome="missed" if event["matched_good_frame"] is None else "retained")
        start = max(1, onset - LEAD)
        last = min(count, recovery + design["recovery_window"] + width - 1)
        window = slice(start - 1, last)
        frame_rows, drop_rows = [], []
        for frame in frames[window]:
            frame = int(frame)
            i = frame - 1
            if frame not in replay_cache:
                replay_cache[frame] = memory_replay(gated, frame, design)
            template, alternatives = replay_cache[frame]
            frame_ident = dict(ident, frame=frame)
            refs = [int(ref) for ref in gated["refs"][i] if ref > 0]
            frame_rows.append(dict(
                frame_ident, box_differs=bool(box_changed[i]), crop_geometry_differs=bool(geometry_changed[i]),
                crop_pixels_differ=bool(crop_changed[i]),
                ppo_iou=ppo_iou[i], gated_iou=gated_iou[i], ppo_on_gated_iou=ppo_on_gated[i],
                ppo_best_iou=ppo_best[i], gated_best_iou=gated_best[i],
                ppo_slot=int(ppo["selected"][i]), ppo_on_gated_slot=int(gated["ppo_selected"][i]),
                gated_slot=int(gated["selected"][i]), challenger_slot=int(gated["challenger"][i]),
                gated_switch=bool(switched[i]), in_domain=bool(gated["in_domain"][i]),
                used_history=bool(gated["used_history"][i]), memory_write=bool(gated["memory_write"][i]),
                refs="|".join(map(str, refs)), ref_write_ious="|".join(f"{gated_iou[f-1]:.8f}" for f in refs),
                margin=gated["margin"][i], template_margin=gated["patch_margin"][i],
                template_slot=template["selected"], template_iou=gated_bank[i, template["selected"]]))
            for ref, alt in alternatives:
                alt_iou = gated_bank[i, alt["selected"]]
                drop_rows.append(dict(frame_ident, dropped_write_frame=ref, dropped_write_iou=gated_iou[ref - 1],
                    dropped_write_was_switch=bool(switched[ref - 1]), actual_iou=gated_iou[i],
                    actual_slot=int(gated["selected"][i]), replay_slot=alt["selected"], replay_iou=alt_iou,
                    replay_used_history=bool(alt["used_history"]),
                    replay_refs="|".join(str(f) for f in alt["refs"] if f > 0),
                    slot_changed=alt["selected"] != gated["selected"][i],
                    low_to_good=bool(0 <= gated_iou[i] < low_iou and alt_iou >= good_iou)))
        span = frames[window]
        valid = (ppo_iou[window] >= 0) & (gated_iou[window] >= 0) & (ppo_on_gated[window] >= 0)
        harm = switched & (gated_iou >= 0) & (gated_iou < low_iou) & (ppo_on_gated >= good_iou)
        no_good = (gated_best >= 0) & (gated_best < good_iou) & (ppo_best >= good_iou)
        means = {}
        for name, delta in (("own_iou_gap", gated_iou - ppo_iou),
                            ("same_state_choice_gap", gated_iou - ppo_on_gated),
                            ("state_associated_gap", ppo_on_gated - ppo_iou)):
            means[name] = float(delta[window][valid].mean()) if valid.any() else None
        events.append(dict(ident, first_audit_frame=start, last_audit_frame=last, valid_frames=int(valid.sum()),
            first_switch_full=first_frame(frames, switched), first_box_difference_full=first_frame(frames, box_changed),
            first_crop_geometry_difference_full=first_frame(frames, geometry_changed),
            first_crop_pixel_difference_full=first_frame(frames, crop_changed),
            first_harmful_switch_in_window=first_frame(span, harm[window]),
            first_no_good_in_window=first_frame(span, no_good[window]),
            harmful_switch_frames=int(harm[window].sum()), gated_only_no_good_frames=int(no_good[window].sum()),
            first_template_low_to_good=next((r["frame"] for r in frame_rows
                if 0 <= r["gated_iou"] < low_iou and r["template_iou"] >= good_iou), None),
            first_drop_low_to_good=next((r["frame"] for r in drop_rows if r["low_to_good"]), None),
            gated_first_good_run_from_onset=first_frame(frames[:len(good)], good & (frames[:len(good)] >= onset)),
            gated_iou_at_ppo_recovery=gated_iou[recovery - 1],
            gated_best_at_ppo_recovery=gated_best[recovery - 1],
            ppo_best_at_ppo_recovery=ppo_best[recovery - 1], **means))
        rows.extend(frame_rows)
        drops.extend(drop_rows)
    return events, rows, drops


def load_run(root93, root76, root89):
    s93.check_results(root93)
    plan93 = c.c66.sealed(root93 / "manifest.json", s93.SCHEMA)
    report93 = c.c66.sealed(root93 / "analysis/stage93_decision.json", s93.SCHEMA)
    if report93["fingerprint"] != SOURCE or plan93["design"] != s93.DESIGN:
        raise ValueError("Need the frozen Stage-93 L20 result")
    paths89 = (root89 / "manifest.json", root89 / "analysis/stage89_decision.json")
    io.verify({str(p): plan93["source_sha256"][str(p)] for p in paths89})
    plan89, report89 = (c.c66.sealed(p, s89.SCHEMA) for p in paths89)
    if report89["plan_fingerprint"] != plan89["fingerprint"]:
        raise ValueError("Stage-89 plan and result differ")
    source = s86.previous.load_source(root76)
    roster = c.c66.sealed(s86.ROSTER, s86.confirmation.SCHEMA)
    held = {r["source_group_proxy"] for r in roster["sequences"]}
    units = plan89["units"]
    groups = {s89.source_group(spec["unit"]["sequence"]) for spec in units}
    if (source["report"]["fingerprint"] != plan89["source_stage76_fingerprint"]
            or roster["fingerprint"] != plan89["confirmation_fingerprint"]
            or len(held) != 28 or held & groups or len(groups) != 28 or len(units) != 40):
        raise ValueError("Development source or confirmation boundary differs")
    return plan93, report93, plan89, report89, source


def required_cache(path, meta):
    trace = c.read_cache(path, meta)
    if trace is None:
        raise ValueError(f"Missing sealed cache: {path}")
    return trace


def collect(root93, root76, root89):
    plan93, report93, plan89, report89, source = load_run(root93, root76, root89)
    events, rows, drops = [], [], []
    for spec in plan89["units"]:
        unit = spec["unit"]
        sequence = unit["sequence"]
        name = c.key(unit)
        if spec["cohort"] == "historical":
            entry = c.entry_for(source["original"], sequence)
        else:
            path = f"entries/{sequence}.json"
            io.verify({str(root89 / path): report89["evidence_sha256"][path]})
            bound = c.c66.sealed(root89 / path, s89.SCHEMA)
            if bound["plan_fingerprint"] != plan89["fingerprint"] or bound["entry"]["sequence"] != sequence:
                raise ValueError("Stage-89 expanded entry changed")
            entry = bound["entry"]
        meta = dict(plan_fingerprint=plan93["fingerprint"], unit=unit)
        gated = required_cache(root93 / f"evaluation/{name}__gated.npz", dict(meta, arm="gated"))
        if not plan93["reuse_ppo"]:
            ppo = required_cache(root93 / f"evaluation/{name}__ppo.npz", dict(meta, arm="ppo"))
        elif spec["cohort"] == "historical" and plan89["reuse_historical"]:
            _, ppo = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        else:
            path = f"baseline/{name}.npz"
            io.verify({str(root89 / path): report89["evidence_sha256"][path]})
            ppo = required_cache(root89 / path, dict(plan_fingerprint=plan89["fingerprint"], unit=unit))
        result = audit_unit(spec, ppo, gated, np.asarray(entry["ground_truth"], float)[1:], plan93["design"])
        for target, values in zip((events, rows, drops), result):
            target.extend(values)

    with (root93 / "analysis/stage93_recovery.csv").open(encoding="utf-8", newline="") as handle:
        expected = [r for r in csv.DictReader(handle) if r["arm"] == "gated"]
    keys = ("cohort", "sequence", "seed", "ppo_failure_frame", "ppo_recovery_frame", "matched_good_frame")
    observed = [tuple(str(r[k]) if r[k] is not None else "" for k in keys) for r in events]
    if (sorted(observed) != sorted(tuple(r[k] for k in keys) for r in expected)
            or len(events) != 25 or sum(r["outcome"] == "missed" for r in events) != 4):
        raise ValueError("Stage-93 recovery records do not reconstruct")
    return events, rows, drops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage93", "stage76", "stage89", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    roots = [p.expanduser().resolve() for p in (args.stage93, args.stage76, args.stage89)]
    output = args.output_dir.expanduser().resolve()
    if any(output == p or output in p.parents or p in output.parents for p in roots):
        parser.error("Use a separate Stage-94 output directory")
    with s93.writer_lock(output):
        tables = collect(*roots)
        hashes = {}
        for name, rows in zip(TABLES, tables):
            path = output / f"stage94_{name}.csv"
            s86.publish_table(path, rows)
            hashes[path.name] = c.file_hash(path)
        code = (Path(__file__), Path(s93.__file__), Path(s93.s92.__file__), Path(s93.s88.__file__))
        c.publish_identical(output / "stage94_decision.json", io.seal(dict(
            schema=SCHEMA, status=STATUS, role="consumed_development_conditional_replay",
            stage93_fingerprint=SOURCE, lead_frames=LEAD, events=len(tables[0]),
            missed_events=sum(r["outcome"] == "missed" for r in tables[0]),
            event_frame_rows=len(tables[1]), reference_drop_rows=len(tables[2]), image_reads=0,
            new_search_calls=0, training_updates=0, confirmation_inference_calls=0,
            table_sha256=hashes, code_sha256={p.name: c.file_hash(p) for p in code})))
    print(f"Stage-94: {STATUS} ({len(tables[0])} recoveries, {len(tables[2])} reference drops)")


if __name__ == "__main__":
    main()
