"""Cross original and limited search crops with fixed first actions, then replay PPO."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage78_scale_experiment as s78
import stage79_confirmation as confirmation
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, failure_events, tracking_iou
from stage68_advantage_summary import publish_table


c, io, s76, e63, e74 = s78.c, s78.io, s78.s76, s78.e63, s78.e74
SCHEMA = "stage79_scale_action_cross_v1"
STATUS = "COMPLETE_SCALE_ACTION_CROSS_DIAGNOSTIC"
ANCHORS = dict(recovery_0204=(664, (0, 2), (2, 0)), late_2172=(1075, (2,), (2, 2)))
SCALES = ("native", "search_scale")
TABLES = ("candidates", "branches", "timeline")
FIELDS = s78.FIELDS + ("probabilities", "action_mask", "logits")


def load_stage78(root, report77):
    p78, r78, files = s78.completed(root, s78.SCHEMA, s78.STATUS, tuple(f"stage78_{n}.csv" for n in s78.TABLES))
    if p78["source_diagnostic_fingerprint"] != report77["fingerprint"] or r78["source_max_box_error"] != 0 or any(
            r78[name] for name in ("source_action_disagreements", "source_crop_disagreements")):
        raise ValueError("Need Stage-78 with an exact source bridge to the supplied Stage-77")
    bindings = {}
    for case in p78["cases"]:
        for scale in SCALES:
            s76.binding(bindings, root / "collection" / case["case"] / (scale + ".npz"))
    if c.fingerprint(bindings) != r78["replay_evidence_fingerprint"]:
        raise ValueError("Stage-78 continuation evidence changed")
    files.update(bindings)
    return p78, r78, files


def starting_state(source78, root78, case, old, prefix, unit, prior, pack):
    refs, histories = {}, {}
    anchor = case["anchor"]
    frames = list(range(case["first"], case["last"] + 1))
    pos = anchor - case["first"]
    for scale in SCALES:
        original_case = {key: value for key, value in case.items() if key not in ("anchor", "actions", "source_actions")}
        meta = dict(plan_fingerprint=source78["fingerprint"], case=original_case, arm=scale)
        path = root78 / "collection" / case["case"] / (scale + ".npz")
        data = c.read_cache(path, meta)
        s78.audit(data, prefix, frames, unit, old, prior, pack, source78["design"], scale == "search_scale")
        raw, limited = list(prefix.copy()), list(prefix.copy())
        for box in data["boxes"][:pos]:
            s78.advance(raw, limited, box, old["actor_design"], source78["design"])
        refs[scale] = {k: v[pos:] for k, v in data.items()}
        histories[scale] = (raw, limited)
    for first, second in zip(histories["native"], histories["search_scale"]):
        np.testing.assert_array_equal(first, second, err_msg="Stage-78 raw history diverged before the anchor")
    if np.array_equal(refs["native"]["crop"][0], refs["search_scale"]["crop"][0]):
        raise ValueError("The planned first crop intervention is absent")
    if [int(refs[scale]["selected"][0]) for scale in SCALES] != case["source_actions"]:
        raise ValueError("Stage-78 source actions do not match the anchors")
    return histories["native"], refs


def observe(observer, image, frame, raw, limited, scale, ap, choose):
    parents = s76.branches.prior.history_parents(limited if scale == "search_scale" else raw, ap["design"])
    rows = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents]
    inputs = dict(candidate_boxes=np.asarray([r["boxes"] for r in rows]), candidate_parents=parents,
                  features=np.asarray([e63.actor.feature_vector(r, raw, ap) for r in rows]))
    decision = choose(inputs, raw)
    return dict(decision, frame_id=frame, image_hw=np.asarray(image.shape[:2]),
                decode_error=max(r["decode_error"] for r in rows),
                **{k: np.asarray([r[k] for r in rows]) for k in ("crop", "crop_sha256", "block_calls", "head_calls", "selector_calls", "layer")})


def collect(observer, entry, case, history, scale, ap, choose, design, anchor=None, action=None):
    # Images and histories are the only actor inputs. The forced action is fixed in the plan.
    raw, limited = (list(boxes) for boxes in history)
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), entry["initial_box"])
    first = case["anchor"]
    if anchor is not None:
        box = anchor["candidate_boxes"][action]
        s78.advance(raw, limited, box, ap["design"], design)
        observer.rt.tracker.state = box.tolist()
        first += 1
    rows = []
    for frame in range(first, (first if anchor is None else case["last"]) + 1):
        row = observe(observer, checked_image(ap["root"], entry["frames"][frame]), frame, raw, limited, scale, ap, choose)
        rows.append(row)
        s78.advance(raw, limited, row["boxes"], ap["design"], design)
        observer.rt.tracker.state = row["boxes"].tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in FIELDS}


def cached(root, plan, name, frames, fn, evidence):
    path = root / "collection" / (name + ".npz")
    meta = dict(plan_fingerprint=plan["fingerprint"], record=name)
    data = c.read_cache(path, meta)
    new = data is None
    if new:
        data = fn()
    if set(data) != set(FIELDS):
        raise ValueError("Incomplete scale/action replay")
    np.testing.assert_array_equal(data["frame_id"], frames)
    s76.branches.prior.audit_calls(data, (len(frames), 4))
    if not np.isfinite(data["decode_error"]).all():
        raise ValueError("Nonfinite decode audit")
    if new:
        c.save(path, data, meta)
    s76.binding(evidence, path)
    return data


def alias(boxes, action):
    """Apply the original first-exact-box duplicate convention."""
    return next(i for i, box in enumerate(boxes) if np.array_equal(box, boxes[action]))


def replay(root, plan, source, actor, p78, root78, observer):
    evidence, tables = {}, {n: [] for n in TABLES}
    old = source["original"]
    ap = e74.e67.actor_plan(old)
    with e63.source_experiment.installed(observer, [], plan["cases"][0]["unit"]) as head:
        if head != source["report"]["native_head_sha256"]:
            raise ValueError("Native SGLA head changed")
        for case in plan["cases"]:
            name, first, last = (case[key] for key in ("case", "anchor", "last"))
            entry, trace = s78.previous.read_source_trace(source, dict(unit=case["unit"], frames=[first, last], expected_first_failure=None))
            n = max(ap["design"]["history"], p78["design"]["scale_history"])
            start = case["first"] - 1
            prefix = np.vstack([entry["initial_box"], trace["boxes"][max(0, start-n):start]])[-n:]
            unit, pack = s78.actor_pack(source, actor, case["unit"])
            history, refs = starting_state(p78, root78, case, old, prefix, unit, actor["prior"], pack)

            def choose(row, raw):
                return e74.choose(row, raw, unit, old, actor["prior"], pack)

            for scale in SCALES:
                ref = refs[scale]
                first_row = cached(root, plan, f"{name}__{scale}__anchor", [first],
                    lambda: collect(observer, entry, case, history, scale, ap, choose, plan["design"]), evidence)
                anchor = {k: v[0] for k, v in first_row.items()}
                legal = anchor["action_mask"]
                boxes = anchor["candidate_boxes"]
                gt = np.asarray(entry["ground_truth"], float)
                anchor_ious = tracking_iou(boxes, gt[first])
                for q, box in enumerate(boxes):
                    tables["candidates"].append(dict(case=name, frame=first, scale=scale, candidate=q, iou=float(anchor_ious[q]),
                        x=float(box[0]), y=float(box[1]), width=float(box[2]), height=float(box[3]),
                        parent=anchor["candidate_parents"][q].tolist(), crop=anchor["crop"][q].tolist(),
                        crop_sha256=str(anchor["crop_sha256"][q]), logit=float(anchor["logits"][q]),
                        probability=float(anchor["probabilities"][q]), legal=bool(legal[q]), alias=alias(boxes, q),
                        ppo_selected=q == anchor["selected"], source_selected=q == ref["selected"][0]))
                anchor_exact = all(np.array_equal(anchor[k], ref[k][0]) for k in
                                   ("candidate_boxes", "candidate_parents", "crop", "crop_sha256", "features", "selected"))
                for requested in case["actions"]:
                    action = alias(boxes, requested)
                    tail = cached(root, plan, f"{name}__{scale}__q{requested}", list(range(first+1, last+1)),
                        lambda: collect(observer, entry, case, history, scale, ap, choose, plan["design"], anchor, action), evidence)
                    iou = np.r_[anchor_ious[action], tracking_iou(tail["boxes"], gt[first+1:last+1])]
                    best = np.r_[anchor_ious.max(), tracking_iou(tail["candidate_boxes"], gt[first+1:last+1, None]).max(axis=1)]
                    events = failure_events(iou)
                    natural = action == int(anchor["selected"])
                    ref_match = (anchor_exact and natural and all(np.array_equal(tail[k], ref[k][1:]) for k in
                                 ("boxes", "selected", "crop", "crop_sha256")))
                    ident = dict(case=name, scale=scale, requested_action=requested, executed_action=action)
                    tables["branches"].append(dict(ident, anchor=first, last=last, ppo_action=int(anchor["selected"]),
                        anchor_exact_stage78=anchor_exact, tail_matches_stage78=ref_match, natural=natural,
                        anchor_iou=float(iou[0]), mean_iou=float(iou.mean()), endpoint_iou=float(iou[-1]),
                        good_frames=int((iou >= .5).sum()), candidate_good_frames=int((best >= .5).sum()),
                        low_frames=int((iou < .2).sum()), first_failure=first + events[0]["start"] if events else None,
                        unrecovered_events=sum(e["recovery"] is None for e in events), tail_search_calls=int(tail["head_calls"].sum())))
                    for i, (value, top, q) in enumerate(zip(iou, best, np.r_[action, tail["selected"]])):
                        tables["timeline"].append(dict(ident, frame=first+i, action=int(q), iou=float(value), best_candidate_iou=float(top)))
            print(f"[scale/action cross] {name}", flush=True)
    calls = len(tables["candidates"]) + sum(r["tail_search_calls"] for r in tables["branches"])
    if calls != plan["expected_search_calls"]:
        raise ValueError("Unexpected Stage-79 search budget")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    (root / "analysis").mkdir(exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = root / "analysis" / f"stage79_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    natural = [r for r in tables["branches"] if r["natural"]]
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        source_stage78_fingerprint=plan["source_stage78_fingerprint"], confirmation_fingerprint=plan["confirmation_fingerprint"],
        runtime=plan["runtime"], branches=len(tables["branches"]), physical_search_calls=calls, block_calls=7*calls,
        natural_exact_branches=sum(r["tail_matches_stage78"] for r in natural),
        training_updates=0, confirmation_inference_calls=0, evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(root / "analysis/stage79_decision.json", report)
    print(f"Stage-79: {STATUS} ({calls} searches)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage77", "stage78", "confirmation", "output_dir"):
        p.add_argument("--" + name, type=Path, required=name == "output_dir")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--check-results", action="store_true")
    p.set_defaults(mode="stage79")
    args = p.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        plan = c.c66.sealed(root / "manifest.json", SCHEMA)
        report = c.c66.sealed(root / "analysis/stage79_decision.json", SCHEMA)
        if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
                or report["physical_search_calls"] != plan["expected_search_calls"]
                or set(report["table_sha256"]) != {f"stage79_{name}.csv" for name in TABLES}):
            raise ValueError("Incomplete Stage-79 report")
        io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
        return
    if not all((args.stage76, args.stage77, args.stage78, args.confirmation, args.checkpoint)):
        p.error("Supply Stage-76/77/78, the confirmation roster and the native checkpoint")
    root76, root77, root78 = (path.expanduser().resolve() for path in (args.stage76, args.stage77, args.stage78))
    if any(root == path or root in path.parents or path in root.parents for path in (root76, root77, root78)):
        p.error("Use a separate Stage-79 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
            raise ValueError("Use an empty Stage-79 directory")
        source, actor, r77 = s78.load_sources(root76, root77)
        p78, r78, files = load_stage78(root78, r77)
        if p78["cases"] != s78.cases(source) or p78["design"] != s78.DESIGN:
            raise ValueError("Stage-78 windows or design changed")
        for name, digest in source["evidence"].items():
            if p78["source_sha256"].get(name) != digest:
                raise ValueError("Stage-78 lineage differs from Stage-76/77")
        roster_path = args.confirmation.resolve()
        roster = c.c66.sealed(roster_path, confirmation.SCHEMA)
        held = {r["source_group_proxy"] for r in roster["sequences"]}
        if len(held) < 12 or held & {confirmation.source_group(u["sequence"]) for u in source["plan"]["units"]}:
            raise ValueError("Confirmation roster overlaps development or has too few groups")
        cases = [dict(case, anchor=ANCHORS[case["case"]][0], actions=list(ANCHORS[case["case"]][1]),
                      source_actions=list(ANCHORS[case["case"]][2]))
                 for case in p78["cases"] if case["case"] in ANCHORS]
        if {case["case"] for case in cases} != set(ANCHORS) or any(not case["first"] < case["anchor"] < case["last"] for case in cases):
            raise ValueError("Missing or invalid fixed anchors")
        budget = sum(8 * (1 + len(case["actions"]) * (case["last"] - case["anchor"])) for case in cases)
        if budget != 672:
            raise ValueError("Fixed Stage-79 budget changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        source_hashes = dict(files, **p78["source_sha256"])
        source_hashes[str(roster_path)] = c.file_hash(roster_path)
        code = [Path(__file__), Path(s78.__file__), Path(confirmation.__file__), Path(e74.__file__)]
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_scale_action_cross", cases=cases,
            confirmation_fingerprint=roster["fingerprint"], source_stage78_fingerprint=r78["fingerprint"],
            runtime=runtime, design=p78["design"], expected_search_calls=budget, source_sha256=source_hashes,
            code_sha256={str(path.resolve()): c.file_hash(path) for path in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            replay(root, plan, source, actor, p78, root78, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
