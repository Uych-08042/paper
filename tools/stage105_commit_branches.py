"""Separate one-step geometry and appearance commits in matched SGLA continuations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage104_replacement_audit as prior
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


s103, c, io, s93, e63 = prior.source103, prior.c, prior.io, prior.s93, prior.e63
SCHEMA = "stage105_matched_geometry_memory_commit_v1"
STATUS = "COMPLETE_MATCHED_GEOMETRY_MEMORY_COMMIT_DIAGNOSTIC"
SOURCE104 = "f923f8d0138e1e77458ce8f54f1f1058518f321d1980fb0f66cfa72f0ae6e2da"
DESIGN = dict(horizon=32)
TREATMENTS = {"keep": (False, False), "geometry_pair": (True, False),
              "memory_pair": (False, True), "both_pair": (True, True)}
POINT_FIELDS = s103.s86.FIELDS+("all_queries",)
EXTRA = ("geometry_selected", "geometry_box", "memory_selected", "memory_committed", "pair_memory_write",
         "geometry_commit_changed", "memory_commit_changed", "commit_intervention")
FIELDS = s103.FIELDS+EXTRA
METRICS = ("future_mean_iou", "future_low_iou_fraction", "future_good_candidate_fraction",
           "future_missed_good_fraction", "endpoint_iou")
TABLES = ("summary", "paired", "scope", "branches", "recovery", "focus", "branch_frames")
CONTRASTS = {"geometry_pair_minus_keep": (-1, 1, 0, 0), "memory_pair_minus_keep": (-1, 0, 1, 0),
             "both_pair_minus_keep": (-1, 0, 0, 1), "commit_interaction": (1, -1, -1, 1),
             "geometry_order_average": (-.5, .5, -.5, .5), "memory_order_average": (-.5, -.5, .5, .5)}


def select_anchors(trace, entry, cases, design):
    """Freeze diagnostic roles from consumed sources before any new branch outcome."""
    n, anchors = len(trace["frame_id"]), {}
    def add(frame, role, target, reference_arm="", event=""):
        if frame is None:
            return
        frame = int(frame)
        if not 1 <= frame <= n:
            raise ValueError("Anchor outside its source trajectory")
        anchor = anchors.setdefault(frame, dict(frame=frame, last=min(n, frame+design["horizon"]-1), roles=[]))
        record = dict(role=role, target_frame=int(target), reference_arm=reference_arm, reference_event=event)
        if record not in anchor["roles"]:
            anchor["roles"].append(record)
    direct = {"first_box_replacement", "first_output_difference_from_full_pair", "first_good_lost", "first_low_introduced"}
    for row in cases:
        target, role = int(row["frame"]), row["role"]
        if role in direct:
            add(target, role, target)
        elif role in ("first_failure", "new_failure_on_pair_stable"):
            add(row["last_replacement"] or target, "last_replacement_before_"+role, target)
            add(max(1, target-1), "previous_frame_before_"+role, target)
        elif role in ("newly_lost_ppo_recovery", "newly_lost_pair_recovery"):
            reference = "ppo" if role == "newly_lost_ppo_recovery" else "pair"
            add(max(1, target-1), "previous_frame_before_lost_recovery", target, reference, row["reference_event"])
            add(row["last_good_lost"] or row["last_replacement"] or target,
                "earlier_replacement_before_lost_recovery", target, reference, row["reference_event"])
        else:
            raise ValueError("Unknown Stage-104 role: "+role)
    truth = np.asarray(entry["ground_truth"], float)[1:]
    bank = tracking_iou(trace["candidate_boxes"], truth[:, None])
    index = np.arange(n)
    valid = np.isfinite(truth).all(1) & (truth[:, 2:] > 0).all(1)
    rescue = valid & (bank[index, trace["selected"]] >= s103.DESIGN["good_iou"]) & (bank[index, trace["pair_selected"]] < s103.DESIGN["good_iou"])
    positions = np.flatnonzero(rescue)
    if len(positions):
        add(int(positions[0])+1, "first_good_rescued", int(positions[0])+1)
    return [anchors[f] for f in sorted(anchors)]


def initial_state(trace, entry, first, old, design):
    history = list(e63.history_at(trace, entry["initial_box"], first, old))
    memory = [(f, trace["all_queries"][f-1, int(trace["selected"][f-1])].copy())
              for f in range(max(1, first-design["lookback"]), first) if trace["memory_write"][f-1]]
    return dict(history=history, memory=memory)


def advance(observation, state, anchor, frame, first, treatment, mode, old, choose, design):
    """Publish the same native verifier output at the intervention; vary only commits."""
    if treatment not in TREATMENTS or mode not in s103.MODES[1:]:
        raise ValueError("Unknown commit treatment/source verifier")
    ap = e63.c.actor_plan(old)
    parents = e63.source_common.prior.history_parents(state["history"], ap["design"])
    np.testing.assert_array_equal(observation["candidate_parents"], parents)
    np.testing.assert_array_equal(observation["history_latest"], state["history"][-1])
    np.testing.assert_array_equal(observation["crop"], [crop_geometry(p, observation["image_hw"], ap["search_factor"], ap["search_size"])[0] for p in parents])
    row = choose({k: observation[k].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, state["history"])
    row["history_latest"] = state["history"][-1]
    queries = observation["all_queries"]
    actual = s103.decide(row, queries, anchor, list(state["memory"]), frame, mode, design)
    pair = s103.decide(row, queries, anchor, list(state["memory"]), frame, "pair", design)
    geometry_pair, memory_pair = TREATMENTS[treatment] if frame == first else (False, False)
    geometry_selected = pair["selected"] if geometry_pair else actual["selected"]
    memory_decision = pair if memory_pair else actual
    memory_selected, written = memory_decision["selected"], memory_decision["memory_write"]
    geometry_box = row["candidate_boxes"][geometry_selected].copy()
    box = row["candidate_boxes"][actual["selected"]].copy()
    memory_changed = written != actual["memory_write"] or bool(written and actual["memory_write"] and
        not np.array_equal(queries[memory_selected], queries[actual["selected"]], equal_nan=True))
    result = dict(observation, **actual, boxes=box, geometry_selected=geometry_selected, geometry_box=geometry_box,
        memory_selected=memory_selected, memory_committed=bool(written), pair_memory_write=bool(pair["memory_write"]),
        geometry_commit_changed=not np.array_equal(geometry_box, box), memory_commit_changed=bool(memory_changed),
        commit_intervention=bool(frame == first and (geometry_pair or memory_pair)))
    if written:
        state["memory"].append((frame, queries[memory_selected].copy()))
    state["memory"][:] = [(f, q) for f, q in state["memory"] if frame-f < design["lookback"]]
    state["history"] = (state["history"]+[geometry_box])[-ap["design"]["history"]:]
    return result


def observe(observer, image, state, frame, old, choose, design):
    ap = e63.c.actor_plan(old)
    parents = e63.source_common.prior.history_parents(state["history"], ap["design"])
    row, queries, _ = s103.observe(observer, image, state["history"], parents, frame, ap, choose, design)
    return row | dict(all_queries=queries)


def point(observer, trace, entry, first, old, choose, design, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
    state = initial_state(trace, entry, first, old, design)
    row = observe(observer, image_reader(ap["root"], entry["frames"][first]), state, first, old, choose, design)
    return {k: np.asarray(row[k])[None] for k in POINT_FIELDS} | dict(anchor=trace["anchor"].copy())


def collect(observer, trace, shared, entry, first, last, treatment, mode, old, choose, design, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
    state, rows = initial_state(trace, entry, first, old, design), []
    for frame in range(first, last+1):
        observation = ({k: shared[k][0].copy() for k in POINT_FIELDS} if frame == first else
                       observe(observer, image_reader(ap["root"], entry["frames"][frame]), state, frame, old, choose, design))
        row = advance(observation, state, shared["anchor"], frame, first, treatment, mode, old, choose, design)
        rows.append(row)
        observer.rt.tracker.state = row["geometry_box"].tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in FIELDS} | dict(anchor=shared["anchor"].copy())


def compare_fields(a, b, names):
    for key in names:
        if key in ("scores", "gains"):
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12, equal_nan=True, err_msg=key)
        else:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def audit_point(data, trace, entry, first, mode, old, choose, design):
    if set(data) != set(POINT_FIELDS) | {"anchor"} or data["all_queries"].shape != (1, 4, len(data["anchor"])):
        raise ValueError("Incomplete common first observation")
    np.testing.assert_array_equal(data["frame_id"], [first])
    np.testing.assert_array_equal(data["anchor"], trace["anchor"])
    e63.source_common.prior.audit_calls(data, (1, 4))
    np.testing.assert_array_equal(data["image_hw"][0], [entry["frames"][first]["height"], entry["frames"][first]["width"]])
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > old["actor_design"]["box_atol"]:
        raise ValueError("Common observation decode drift")
    state = initial_state(trace, entry, first, old, design)
    observation = {k: data[k][0].copy() for k in POINT_FIELDS}
    predicted = choose({k: observation[k].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, state["history"])
    np.testing.assert_array_equal(observation["selected"], predicted["selected"])
    np.testing.assert_array_equal(observation["boxes"], observation["candidate_boxes"][int(predicted["selected"])])
    advance(observation, state, data["anchor"], first, first, "keep", mode, old, choose, design)


def audit(data, trace, shared, entry, first, last, treatment, mode, old, choose, design):
    count = last-first+1
    if set(data) != set(FIELDS) | {"anchor"} or data["all_queries"].shape != (count, 4, len(data["anchor"])):
        raise ValueError("Incomplete matched commit branch")
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, last+1))
    np.testing.assert_array_equal(data["anchor"], trace["anchor"])
    e63.source_common.prior.audit_calls(data, (count, 4))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > old["actor_design"]["box_atol"]:
        raise ValueError("Branch decode drift")
    compare_fields({k: data[k][:1] for k in POINT_FIELDS if k not in ("selected", "boxes")}, shared,
                   [k for k in POINT_FIELDS if k not in ("selected", "boxes")])
    state = initial_state(trace, entry, first, old, design)
    for i, frame in enumerate(range(first, last+1)):
        row = {k: data[k][i].copy() for k in FIELDS}
        np.testing.assert_array_equal(row["image_hw"], [entry["frames"][frame]["height"], entry["frames"][frame]["width"]])
        expected = advance(row, state, data["anchor"], frame, first, treatment, mode, old, choose, design)
        compare_fields(row, expected, FIELDS)


def first_difference(a, b, key):
    changed = np.flatnonzero(np.any((a[key] != b[key]).reshape(len(a["frame_id"]), -1), axis=1))
    return int(a["frame_id"][changed[0]]) if len(changed) else None


def metrics(data, entry, design):
    truth = np.asarray(entry["ground_truth"], float)[data["frame_id"]]
    overlap = tracking_iou(data["boxes"], truth)
    bank = tracking_iou(data["candidate_boxes"], truth[:, None])
    pair = bank[np.arange(len(bank)), data["pair_selected"]]
    valid = np.isfinite(truth).all(1) & (truth[:, 2:] > 0).all(1) & (overlap >= 0)
    future = valid.copy(); future[0] = False
    best = bank.max(1)
    return dict(current_iou=float(overlap[0]) if valid[0] else None,
        current_pair_iou=float(pair[0]) if valid[0] else None,
        future_valid_frames=int(future.sum()),
        future_mean_iou=float(overlap[future].mean()) if future.any() else None,
        future_low_iou_fraction=float((overlap[future] < design["low_iou"]).mean()) if future.any() else None,
        future_good_candidate_fraction=float((best[future] >= design["good_iou"]).mean()) if future.any() else None,
        future_missed_good_fraction=float(((best[future] >= design["good_iou"]) & (overlap[future] < design["good_iou"])).mean()) if future.any() else None,
        endpoint_iou=float(overlap[-1]) if valid[-1] else None,
        first_geometry_commit_changed=bool(data["geometry_commit_changed"][0]),
        first_memory_commit_changed=bool(data["memory_commit_changed"][0]))


def aggregate(branches, anchors):
    """Unique anchors overall; each diagnostic role is a separate conditional stratum."""
    role_map = {(r["sequence"], r["seed"], r["source_arm"], r["frame"]): sorted({x["role"] for x in r["roles"]}) for r in anchors}
    indexed = defaultdict(dict)
    for row in branches:
        indexed[row["cohort"], row["sequence"], row["seed"], row["source_arm"], row["frame"]][row["treatment"]] = row
    bins = defaultdict(list)
    for key, bank in sorted(indexed.items()):
        if set(bank) != set(TREATMENTS):
            raise ValueError("Incomplete factorial branch set")
        cohort, sequence, seed, source_arm, frame = key
        for stratum in ["all_unique_anchors"]+role_map[sequence, seed, source_arm, frame]:
            for treatment, row in bank.items():
                bins[cohort, source_arm, stratum, treatment, sequence, seed].append(row)
    units = {k: {m: prior.mean(v, m) for m in METRICS} | dict(anchors=len(v)) for k, v in bins.items()}
    summary, paired = [], []
    for cohort in ("all", "historical", "expanded"):
        settings = sorted({(k[1], k[2]) for k in units if cohort == "all" or k[0] == cohort})
        for source_arm, stratum in settings:
            local = {k: v for k, v in units.items() if (cohort == "all" or k[0] == cohort) and k[1:3] == (source_arm, stratum)}
            for treatment in TREATMENTS:
                rows = defaultdict(list)
                for k, v in local.items():
                    if k[3] == treatment:
                        rows[k[4]].append(v)
                summary.append(dict(cohort=cohort, source_arm=source_arm, stratum=stratum, treatment=treatment,
                    videos=len(rows), units=sum(len(v) for v in rows.values()), anchors=sum(r["anchors"] for v in rows.values() for r in v),
                    **{m: prior.mean([dict(value=prior.mean(v, m)) for v in rows.values()], "value") for m in METRICS},
                    **{m+"_supported_videos": sum(prior.mean(v, m) is not None for v in rows.values()) for m in METRICS}))
            for label, weights in CONTRASTS.items():
                by_video = defaultdict(list)
                owners = sorted({(k[0], k[4], k[5]) for k in local})
                for owner_cohort, sequence, seed in owners:
                    bank = [local[owner_cohort, source_arm, stratum, t, sequence, seed] for t in TREATMENTS]
                    by_video[sequence].append({m: sum(w*r[m] for w, r in zip(weights, bank) if w)
                        if all(r[m] is not None for w, r in zip(weights, bank) if w) else None for m in METRICS})
                values = {m: [prior.mean(v, m) for v in by_video.values() if prior.mean(v, m) is not None] for m in METRICS}
                primary = np.asarray(values["future_mean_iou"], float)
                paired.append(dict(cohort=cohort, source_arm=source_arm, stratum=stratum, contrast=label,
                    videos=len(by_video), units=len(owners),
                    **{m+"_delta": float(np.mean(v)) if v else None for m, v in values.items()},
                    **{m+"_supported_videos": len(v) for m, v in values.items()},
                    future_mean_iou_median_delta=float(np.median(primary)) if len(primary) else None,
                    future_mean_iou_wins=int((primary > 1e-12).sum()),
                    future_mean_iou_ties=int((np.abs(primary) <= 1e-12).sum()),
                    future_mean_iou_losses=int((primary < -1e-12).sum())))
    return summary, paired


def frame_rows(data, entry, identity, treatment, design):
    truth = np.asarray(entry["ground_truth"], float)
    rows = []
    for i, frame in enumerate(data["frame_id"]):
        bank = prior.source103.source102.quality(data["candidate_boxes"][i], truth[frame])
        selected, base = int(data["selected"][i]), int(data["pair_selected"][i])
        rows.append(dict(identity, treatment=treatment, output_frame=int(frame),
            iou=float(bank[selected]) if bank is not None else None,
            same_state_pair_iou=float(bank[base]) if bank is not None else None,
            best_iou=float(bank.max()) if bank is not None else None,
            selected=selected, ppo_selected=int(data["ppo_selected"][i]),
            geometry_selected=int(data["geometry_selected"][i]), memory_selected=int(data["memory_selected"][i]),
            verifier_memory_write=bool(data["memory_write"][i]), actual_memory_committed=bool(data["memory_committed"][i]),
            geometry_commit_changed=bool(data["geometry_commit_changed"][i]),
            memory_commit_changed=bool(data["memory_commit_changed"][i]),
            refs=prior.encoded(int(f) for f in data["refs"][i] if f >= 0),
            candidate_iou=prior.encoded(bank) if bank is not None else ""))
    return rows


def budget(units):
    anchors = [a for u in units for a in u["anchors"]]
    frames = sum(a["last"]-a["frame"]+1 for a in anchors)
    return dict(anchors=len(anchors), short_branches=4*len(anchors), branch_frames=4*frames,
                logical_search_calls=16*frames, new_search_calls=4*(4*frames-3*len(anchors)), caches=5*len(anchors),
                recovery_rows=4*sum(bool(r["reference_event"]) for a in anchors for r in a["roles"]))


def prepare_units(root103, plan103, cases, load_unit, choose_for, old):
    units = []
    for spec in plan103["units"]:
        entry = load_unit(spec)
        choose = choose_for(spec["unit"])
        for mode in s103.MODES[1:]:
            trace = prior.required_trace(root103, plan103, spec["unit"], mode)
            s103.audit(trace, entry, old, choose, mode, plan103["design"])
            local = [r for r in cases if r["sequence"] == spec["unit"]["sequence"] and int(r["seed"]) == spec["unit"]["seed"] and r["arm"] == mode]
            anchors = select_anchors(trace, entry, local, DESIGN)
            if not anchors:
                raise ValueError("A source verifier has no diagnostic anchor")
            units.append(dict(spec=spec, source_arm=mode, anchors=anchors))
            print(f"[Stage-105 prepare] {c.key(spec['unit'])} {mode}: {len(anchors)} frozen anchors", flush=True)
    return units


def run(root, root103, plan103, plan, load_unit, choose_for, old, observer):
    design, evidence = plan103["design"], {}
    branches, scopes, anchor_rows, recovery, frames = [], [], [], [], []
    for job in plan["units"]:
        spec, mode = job["spec"], job["source_arm"]
        unit = spec["unit"]
        entry, choose = load_unit(spec), choose_for(unit)
        trace = prior.required_trace(root103, plan103, unit, mode)
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != plan["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            for case in job["anchors"]:
                first, last = case["frame"], case["last"]
                ident = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"], source_arm=mode, frame=first, last=last)
                anchor_rows.append(dict(ident, roles=case["roles"]))
                stem = f"{c.key(unit)}__{mode}__f{first:06d}"
                meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, source_arm=mode, first=first, last=last)
                path = root/"points"/(stem+".npz")
                shared = c.read_cache(path, dict(meta, kind="point"))
                fresh = shared is None
                if fresh:
                    shared = point(observer, trace, entry, first, old, choose, design)
                audit_point(shared, trace, entry, first, mode, old, choose, design)
                if fresh:
                    c.save(path, shared, dict(meta, kind="point"))
                evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                local = {}
                for treatment in TREATMENTS:
                    path = root/"branches"/(stem+"__"+treatment+".npz")
                    metadata = dict(meta, kind="branch", treatment=treatment)
                    data = c.read_cache(path, metadata)
                    fresh = data is None
                    if fresh:
                        data = collect(observer, trace, shared, entry, first, last, treatment, mode, old, choose, design)
                    audit(data, trace, shared, entry, first, last, treatment, mode, old, choose, design)
                    if fresh:
                        c.save(path, data, metadata)
                    evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                    local[treatment] = data
                keep = local["keep"]
                source = {k: trace[k][first-1:last] for k in s103.FIELDS}
                error = float(np.abs(keep["boxes"]-source["boxes"]).max())
                if not plan["runtime_differences"]:
                    compare_fields(keep, source, s103.FIELDS)
                for treatment, data in local.items():
                    compare_fields({k: data[k][:1] for k in s103.FIELDS}, {k: keep[k][:1] for k in s103.FIELDS}, s103.FIELDS)
                    row = dict(ident, treatment=treatment, **metrics(data, entry, design))
                    for key in ("boxes", "candidate_parents", "crop", "ppo_selected", "memory_committed", "refs"):
                        row["first_"+key+"_difference_from_keep"] = first_difference(data, keep, key)
                    branches.append(row)
                    frames.extend(frame_rows(data, entry, ident, treatment, design))
                    for role in case["roles"]:
                        if not role["reference_event"]:
                            continue
                        matched = s103.source102.recovery_match(data, entry, role["target_frame"], design)
                        outcome = ("recovered" if matched["aligned_recovery"] else "not_recovered_complete_window"
                                   if matched["complete_search_window"] else "right_censored")
                        recovery.append(dict(ident, treatment=treatment, **role, **matched, outcome=outcome))
                scopes.append(dict(ident, roles=prior.encoded(sorted({r["role"] for r in case["roles"]})),
                    targets=prior.encoded(sorted({r["target_frame"] for r in case["roles"]})),
                    source_keep_box_error_px=error, shared_first_pass=True,
                    geometry_intervention_effective=bool(local["geometry_pair"]["geometry_commit_changed"][0]),
                    memory_intervention_effective=bool(local["memory_pair"]["memory_commit_changed"][0]),
                    new_search_calls=4+16*(last-first), logical_search_calls=16*(last-first+1)))
                print(f"[Stage-105] {stem}: four branches {first}..{last} complete", flush=True)
    summary, paired = aggregate(branches, anchor_rows)
    focus = [r for r in frames if r["output_frame"] in (r["frame"], r["last"])]
    tables = dict(zip(TABLES, (summary, paired, scopes, branches, recovery, focus, frames)))
    publish(root, plan, tables, evidence)


def publish(root, plan, tables, evidence):
    expected = budget(plan["units"])
    counts = {n: len(v) for n, v in tables.items()}
    if (set(tables) != set(TABLES) or counts["scope"] != expected["anchors"] or counts["branches"] != expected["short_branches"]
            or counts["branch_frames"] != expected["branch_frames"] or len(evidence) != expected["caches"]
            or counts["recovery"] != expected["recovery_rows"]
            or sum(r["new_search_calls"] for r in tables["scope"]) != expected["new_search_calls"]
            or sum(r["logical_search_calls"] for r in tables["scope"]) != expected["logical_search_calls"]):
        raise ValueError("Incomplete matched commit coverage")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    out = root/"analysis"; out.mkdir(exist_ok=True)
    hashes = {}
    for name in TABLES:
        path = out/f"stage105_{name}.csv"
        s103.s86.publish_table(path, tables[name]); hashes[path.name] = c.file_hash(path)
    c.publish_identical(out/"stage105_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_matched_one_step_commit_diagnostic", plan_fingerprint=plan["fingerprint"],
        source_stage104_fingerprint=SOURCE104, runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        counts=counts, budget=expected, new_block_calls=7*expected["new_search_calls"],
        maximum_keep_source_box_error_px=max(r["source_keep_box_error_px"] for r in tables["scope"]),
        matched_first_output_pass=True, past_only_state_replay_pass=True, one_step_intervention_only=True,
        weights_frozen=True, training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        table_sha256=hashes, cache_sha256=evidence)))
    check_results(root)
    print(f"Stage-105: {STATUS}; {expected['short_branches']} short branches, {expected['new_search_calls']} searches", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"analysis/stage105_decision.json", SCHEMA)
    expected = budget(plan["units"])
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage104_fingerprint"] != SOURCE104 or report["runtime"] != plan["runtime"]
            or report["runtime_differences"] != plan["runtime_differences"] or report["budget"] != expected
            or report["new_block_calls"] != 7*expected["new_search_calls"]
            or (not plan["runtime_differences"] and report["maximum_keep_source_box_error_px"] != 0.)
            or report["counts"]["scope"] != expected["anchors"] or report["counts"]["branches"] != expected["short_branches"]
            or report["counts"]["branch_frames"] != expected["branch_frames"] or len(report["cache_sha256"]) != expected["caches"]
            or report["counts"]["recovery"] != expected["recovery_rows"]
            or set(report["table_sha256"]) != {f"stage105_{n}.csv" for n in TABLES}
            or not all(report[k] for k in ("matched_first_output_pass", "past_only_state_replay_pass", "one_step_intervention_only", "weights_frozen"))
            or any(report[k] != 0 for k in ("training_updates", "new_full_trajectories", "confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-105 diagnostic")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/name): digest for name, digest in report["table_sha256"].items()})
    io.verify({str(root/name): digest for name, digest in report["cache_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage103", "stage104", "output_dir"):
        parser.add_argument("--"+name, type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint"); parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage105")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root); return
    names = ("stage76", "stage89", "stage93", "stage103", "stage104")
    if not all(getattr(args, n) for n in names) or not args.checkpoint:
        parser.error("Supply Stage-76/89/93/103/104 and original checkpoint")
    root76, root89, root93, root103, root104 = sources = [getattr(args, n).expanduser().resolve() for n in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-105 directory")
    with s93.writer_lock(root):
        if (root/"analysis/stage105_decision.json").exists():
            check_results(root); return
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-105 directory")
        print("Stage-105: checking frozen sources and preparing matched anchors before GPU inference", flush=True)
        prior.check_results(root104); s103.check_results(root103)
        report104 = c.c66.sealed(root104/"analysis/stage104_decision.json", prior.SCHEMA)
        plan104 = c.c66.sealed(root104/"manifest.json", prior.SCHEMA)
        plan103 = c.c66.sealed(root103/"manifest.json", s103.SCHEMA)
        report103 = c.c66.sealed(root103/"analysis/stage103_decision.json", s103.SCHEMA)
        if report104["fingerprint"] != SOURCE104 or report103["fingerprint"] != prior.SOURCE103:
            raise ValueError("Need frozen Stage-103/104 results")
        plan93, _, plan89, report89, source = s103.prior.s95.s94.load_run(root93, root76, root89)
        if plan103["units"] != plan89["units"]:
            raise ValueError("Development units changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch
        torch.set_num_threads(1)
        def load_unit(spec):
            return s103.prior.s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)[0]
        def choose_for(unit):
            pack = s103.e74.learned_pack(actor["models"], unit, actor["prior"])
            return lambda row, history: s103.e74.choose(row, history, unit, source["original"], actor["prior"], pack)
        cases = s103.prior.read_csv(root104/"analysis/stage104_cases.csv")
        if len(cases) != 289 or len(plan103["units"]) != 40:
            raise ValueError("Source coverage changed")
        units = prepare_units(root103, plan103, cases, load_unit, choose_for, source["original"])
        runtime = s103.s76.device_runtime(source["original"], torch, args.gpu)
        differences = s103.s76.startup.differences(report103["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Frozen software, weights or configuration changed")
        bindings = plan104["source_sha256"] | actor["bindings"]
        for p in (root104/"manifest.json", root104/"analysis/stage104_decision.json", root103/"manifest.json", root103/"analysis/stage103_decision.json"):
            bindings[str(p)] = c.file_hash(p)
        bindings.update({str(root104/"analysis"/p): h for p, h in report104["table_sha256"].items()})
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage105_commit_branches.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, units=units, runtime=runtime, runtime_differences=differences,
            native_head_sha256=source["report"]["native_head_sha256"], source_sha256=bindings,
            code_sha256=plan104["code_sha256"] | {str(p): c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json", plan)
        print("Stage-105 frozen:", budget(units), "runtime differences:", differences, flush=True)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s103.s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, root103, plan103, plan, load_unit, choose_for, source["original"], observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
