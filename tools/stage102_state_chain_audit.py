"""Trace candidate, choice and memory dependencies using sealed Stage-100/101 caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage101_trigger_replay as prior
from stage27_uav_common import crop_geometry, scene_geometry, tracking_iou


c, io, s93, s97, s100 = prior.c, prior.io, prior.s93, prior.s97, prior.s100
SCHEMA = "stage102_cached_search_state_chain_v1"
STATUS = "COMPLETE_CACHED_SEARCH_STATE_CHAIN_AUDIT"
SOURCE101 = "979f901ca4e7884821a5f32f06bdbdd35c579f47a1fa79b83b4bf0a9d18135ce"
DESIGN = dict(prior.DESIGN, adequate_coverage=.9)
TABLES = ("summary", "unit_summary", "cases", "trigger_summary", "recovery", "branches",
          "trigger_details", "reference_edges", "branch_frames", "state_frames", "chain_excerpt", "reference_summary")
CATEGORIES = ("invalid_gt", "skip_removes_good_supply", "skip_restores_good_supply",
              "neither_has_good_supply", "skip_rescues_choice", "skip_loses_choice",
              "both_miss_available_good", "both_select_good")
RATES = ("no_good_supply", "ppo_miss_available_good", "appearance_loses_good",
         "appearance_rescues_good", "final_miss_available_good", "low_output",
         "low_scale_write", "reads_low_scale", "reads_low_appearance")


def scalar(value):
    return float(value) if np.isfinite(value) else None


def quality(boxes, truth):
    truth = np.asarray(truth, float)
    if not np.isfinite(truth).all() or np.any(truth[2:] <= 0):
        return None
    values = np.asarray(tracking_iou(boxes, truth), float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid output geometry")
    return values


def frame_metrics(data, index, truth, design):
    bank = quality(data["candidate_boxes"][index], truth)
    selected, ppo = int(data["selected"][index]), int(data["ppo_selected"][index])
    if selected not in range(4) or ppo not in range(4):
        raise ValueError("Invalid selected candidate")
    best, final, proposed = (None, None, None) if bank is None else (float(bank.max()), float(bank[selected]), float(bank[ppo]))
    valid, good, low = bank is not None, design["good_iou"], design["low_iou"]
    category = ("invalid_gt" if not valid else "selected_good" if final >= good else
                "no_good_supply" if best < good else "appearance_loses_good" if proposed >= good else
                "ppo_and_final_miss")
    return dict(gt_valid=valid, iou=final, best_iou=best, ppo_iou=proposed, selected=selected, ppo_selected=ppo,
        state=category, no_good_supply=valid and best < good,
        ppo_miss_available_good=valid and best >= good and proposed < good,
        appearance_loses_good=valid and proposed >= good and final < good,
        appearance_rescues_good=valid and proposed < good and final >= good,
        final_miss_available_good=valid and best >= good and final < good,
        low_output=valid and final < low,
        candidate_iou="|".join(f"{v:.12g}" for v in bank) if valid else "")


def cached_row(data, prefix, first, frame):
    if frame < 1:
        raise ValueError("Memory cannot read the initialization as a predicted write")
    trace, index = (prefix, frame-1) if frame < first else (data, frame-first)
    if trace is None or index >= len(trace["frame_id"]) or int(trace["frame_id"][index]) != frame:
        raise ValueError("Reference not covered by its own trajectory")
    return trace, index


def describe_trace(data, entry, unit, cohort, arm, old, design, prefix=None, skip_first=False, drops=True):
    """GT describes already-fixed rows; it never enters state replay or reference deletion."""
    frames = np.asarray(data["frame_id"], int)
    first = int(frames[0])
    np.testing.assert_array_equal(frames, np.arange(first, first+len(frames)))
    prior.e63.source_common.prior.audit_calls(data, (len(frames), 4))
    ap = prior.e63.c.actor_plan(old)
    source = data if prefix is None else prefix
    state = prior.initial_state(source, entry, first, old, design)
    rows, edges = [], []
    truth = np.asarray(entry["ground_truth"], float)
    ident = dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"], arm=arm, start=first)
    for i, frame in enumerate(frames):
        frame = int(frame)
        row = {k: data[k][i] for k in s100.FIELDS}
        mode = "gated" if arm == "gated" or (skip_first and frame == first) else "pulse"
        parents, scale_refs, request, episode = s100.parents_for(state["history"], state["sizes"], frame,
            mode, state["previous"], row["image_hw"], ap, design)
        for key, value in dict(candidate_parents=parents, scale_refs=scale_refs, scale_request=request,
                scale_episode_start=episode, scale_active=bool((scale_refs >= 0).any()),
                history_latest=state["history"][-1]).items():
            np.testing.assert_array_equal(row[key], value, err_msg=f"{frame}:{key}")
        np.testing.assert_array_equal(row["crop"], [crop_geometry(p, row["image_hw"], ap["search_factor"], ap["search_size"])[0] for p in parents])
        chooser = {k: row[k].copy() for k in ("features", "candidate_boxes", "candidate_parents", "history_latest")}
        chooser["selected"] = int(row["ppo_selected"])
        after_memory = list(state["memory"])
        decision = s93.choose_write(chooser, row["queries"], data["anchor"], after_memory, frame, "gated", design)
        for key, value in decision.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-12, equal_nan=True, err_msg=f"{frame}:{key}")
        np.testing.assert_array_equal(row["boxes"], row["candidate_boxes"][int(row["selected"] )])
        accepted = s97.scale_write(decision, row["queries"])
        np.testing.assert_array_equal(row["scale_written"], accepted)
        metrics = frame_metrics(data, i, truth[frame], design)
        reads = dict(scale=[], appearance=[])
        effective_appearance = []
        for kind, references in (("scale", scale_refs),
                                 ("appearance", row["refs"] if row["used_history"] else [])):
            for rf in references:
                rf = int(rf)
                if rf < 0:
                    continue
                if rf >= frame:
                    raise ValueError("Current/future reference in cached state")
                ref_trace, ri = cached_row(data, prefix, first, rf)
                if not ref_trace["memory_write"][ri] or (kind == "scale" and not ref_trace["scale_written"][ri]):
                    raise ValueError("Reference was not an accepted own write")
                if kind == "appearance":
                    slot = int(ref_trace["selected"][ri] != ref_trace["ppo_selected"][ri])
                    if not np.isfinite(ref_trace["queries"][ri, slot]).all():
                        continue  # choose_write excludes these descriptors from its mean.
                    effective_appearance.append(rf)
                value = quality(ref_trace["boxes"][ri], truth[rf])
                ref_iou = float(value) if value is not None else None
                low = ref_iou is not None and ref_iou < design["low_iou"]
                reads[kind].append(low)
                if not drops:
                    continue
                edge = dict(ident, frame=frame, kind=kind, reference_frame=rf, reference_from_prefix=rf < first,
                    reference_iou=ref_iou, reference_low=low,
                    reference_good=ref_iou is not None and ref_iou >= design["good_iou"],
                    reference_log_size_error=(float(np.abs(np.log(ref_trace["boxes"][ri, 2:]/truth[rf, 2:])).mean())
                                              if ref_iou is not None else None),
                    actual_selected=int(row["selected"]), actual_iou=metrics["iou"],
                    drop_parent_changed=None, drop_crop_changed=None, drop_selected=None,
                    drop_iou=None, drop_used_history=None)
                if kind == "scale":
                    without = [(f, size) for f, size in state["sizes"] if f != rf]
                    alternate, _, _, _ = s100.parents_for(state["history"], without, frame, mode,
                        state["previous"], row["image_hw"], ap, design)
                    edge.update(drop_parent_changed=not np.array_equal(parents, alternate),
                        drop_crop_changed=any(not np.array_equal(a, crop_geometry(p, row["image_hw"], ap["search_factor"], ap["search_size"])[0])
                                              for a, p in zip(row["crop"], alternate)))
                else:
                    without = [(f, q) for f, q in state["memory"] if f != rf]
                    changed = s93.choose_write(chooser, row["queries"], data["anchor"], without, frame, "gated", design)
                    value = quality(row["candidate_boxes"][changed["selected"]], truth[frame])
                    edge.update(drop_selected=int(changed["selected"]), drop_iou=float(value) if value is not None else None,
                                drop_used_history=bool(changed["used_history"]))
                edges.append(edge)
        selected = int(row["selected"])
        coverage = []
        if metrics["gt_valid"]:
            coverage = [scene_geometry(p, truth[frame], row["image_hw"], ap["search_factor"], ap["search_size"])["coverage"] for p in parents]
        previous_iou = quality(state["history"][-1], truth[frame-1])
        rows.append(dict(ident, frame=frame, **metrics,
            previous_iou=float(previous_iou) if previous_iou is not None else None,
            history_frames="|".join(str(f) for f in range(max(0, frame-ap["design"]["history"]), frame)),
            scale_active=bool(row["scale_active"]), scale_request=bool(request), scale_written=accepted,
            memory_write=bool(row["memory_write"]), used_history=bool(row["used_history"]),
            scale_refs="|".join(str(int(f)) for f in scale_refs if f >= 0),
            appearance_refs="|".join(str(int(f)) for f in row["refs"] if f >= 0),
            effective_appearance_refs="|".join(str(f) for f in effective_appearance),
            reads_low_scale=any(reads["scale"]), reads_low_appearance=any(reads["appearance"]),
            low_scale_write=bool(accepted and metrics["low_output"]),
            low_appearance_write=bool(row["memory_write"] and metrics["low_output"]),
            best_coverage=max(coverage) if coverage else None,
            selected_coverage=coverage[selected] if coverage else None,
            covered_without_good=bool(metrics["no_good_supply"] and max(coverage) >= design["adequate_coverage"]),
            raw_peak=float(row["features"][selected, prior.e63.source_common.prior.FEATURES.index("raw_peak")]),
            hann_peak=float(row["features"][selected, prior.e63.source_common.prior.FEATURES.index("hann_peak")]),
            appearance_margin=scalar(row["margin"])))
        state["memory"] = after_memory
        s97.update_sizes(state["sizes"], frame, row["boxes"], accepted, design)
        state["history"] = (state["history"]+[row["boxes"]])[-ap["design"]["history"]:]
        state["previous"] = request
    return rows, edges


def summarize_units(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["sequence"], row["seed"], row["arm"]].append(row)
    units = []
    for (cohort, sequence, seed, arm), local in sorted(groups.items()):
        valid = [r for r in local if r["gt_valid"]]
        units.append(dict(cohort=cohort, sequence=sequence, seed=seed, arm=arm, frames=len(local), valid_frames=len(valid),
            **{k+"_frames": sum(r[k] for r in valid) for k in RATES},
            **{k+"_fraction": float(np.mean([r[k] for r in valid])) if valid else None for k in RATES}))
    summary = []
    for cohort in ("all", "historical", "expanded"):
        for arm in ("gated", "pulse"):
            local = [r for r in units if r["arm"] == arm and (cohort == "all" or r["cohort"] == cohort)]
            if not local:
                continue
            videos = defaultdict(list)
            for row in local:
                videos[row["sequence"]].append(row)
            result = dict(cohort=cohort, arm=arm, units=len(local), videos=len(videos), frames=sum(r["frames"] for r in local))
            for k in RATES:
                means = [[r[k+"_fraction"] for r in group if r[k+"_fraction"] is not None] for group in videos.values()]
                means = [float(np.mean(v)) for v in means if v]
                result[k+"_fraction_video"] = float(np.mean(means)) if means else None
            summary.append(result)
    return summary, units


def mechanism(keep, skip, design):
    if not keep["gt_valid"] or not skip["gt_valid"]:
        return "invalid_gt"
    k, s = keep["best_iou"] >= design["good_iou"], skip["best_iou"] >= design["good_iou"]
    if k != s:
        return "skip_removes_good_supply" if k else "skip_restores_good_supply"
    if not k:
        return "neither_has_good_supply"
    k, s = keep["iou"] >= design["good_iou"], skip["iou"] >= design["good_iou"]
    if k != s:
        return "skip_loses_choice" if k else "skip_rescues_choice"
    return "both_select_good" if k else "both_miss_available_good"


def trigger_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["sequence"], row["seed"]].append(row)
    result = []
    for cohort in ("all", "historical", "expanded"):
        local = {k: v for k, v in groups.items() if cohort == "all" or k[0] == cohort}
        if not local:
            continue
        for category in CATEGORIES:
            videos = defaultdict(list)
            total, supported = 0, set()
            for (_, sequence, _), values in local.items():
                count = sum(r["mechanism"] == category for r in values)
                videos[sequence].append(count/len(values))
                total += count
                if count:
                    supported.add(sequence)
            result.append(dict(cohort=cohort, mechanism=category, states=total, supported_videos=len(supported),
                all_videos=len(videos), fraction_video=float(np.mean([np.mean(v) for v in videos.values()]))))
    return result


def reference_summary(edges):
    groups = defaultdict(list)
    for r in edges:
        label = "invalid_gt" if r["reference_iou"] is None else "low" if r["reference_low"] else "good" if r["reference_good"] else "intermediate"
        groups[r["kind"], label].append(r)
    result = []
    for (kind, label), rows in sorted(groups.items()):
        changed = [r["drop_selected"] != r["actual_selected"] for r in rows if r["drop_selected"] is not None]
        gains = [r["drop_iou"]-r["actual_iou"] for r in rows if r["drop_iou"] is not None and r["actual_iou"] is not None]
        result.append(dict(kind=kind, reference_quality=label, reference_reads=len(rows),
            states=len({(r["sequence"],r["seed"],r["arm"],r["frame"]) for r in rows}),
            videos=len({r["sequence"] for r in rows}),
            drop_crop_changed=sum(bool(r["drop_crop_changed"]) for r in rows),
            drop_choice_changed=sum(changed), drop_choice_evaluated=len(changed),
            drop_iou_wins=sum(g > 1e-12 for g in gains), drop_iou_ties=sum(abs(g) <= 1e-12 for g in gains),
            drop_iou_losses=sum(g < -1e-12 for g in gains)))
    return result


def first_difference(a, b, key, stop):
    diff = np.asarray(a[key][:stop] != b[key][:stop]).reshape(stop, -1).any(1)
    found = np.flatnonzero(diff)
    return int(found[0])+1 if len(found) else None


def context_case(unit, cohort, anchor, role, traces, state_rows, trigger_rows, branch):
    target = int(role["target_frame"])
    pulse, gated = state_rows["pulse"], state_rows["gated"]
    if target < 1 or target > len(pulse):
        raise ValueError("Invalid source target")
    last_good = next((r["frame"] for r in reversed(pulse[:target-1]) if r["gt_valid"] and r["iou"] >= DESIGN["good_iou"]), 0)
    context = pulse[last_good:target]
    previous = [r for r in trigger_rows if r["frame"] <= target]
    result = dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"], role=role["role"],
        anchor_frame=anchor["frame"], target_frame=target, context_start=last_good+1, last_good_before_target=last_good or None,
        target_pulse_state=pulse[target-1]["state"], target_gated_state=gated[target-1]["state"],
        target_pulse_iou=pulse[target-1]["iou"], target_gated_iou=gated[target-1]["iou"],
        last_trigger_before_target=previous[-1]["frame"] if previous else None,
        triggers_before_target=len(previous),
        first_context_no_good=next((r["frame"] for r in context if r["no_good_supply"]), None),
        first_context_missed_good=next((r["frame"] for r in context if r["final_miss_available_good"]), None),
        first_context_low_write=next((r["frame"] for r in context if r["low_scale_write"]), None),
        first_context_low_scale_read=next((r["frame"] for r in context if r["reads_low_scale"]), None),
        first_context_low_appearance_read=next((r["frame"] for r in context if r["reads_low_appearance"]), None),
        context_low_writes=sum(r["low_scale_write"] for r in context),
        branch_mean_skip_gain=branch["skip_mean_iou"]-branch["keep_mean_iou"] if branch["skip_mean_iou"] is not None and branch["keep_mean_iou"] is not None else None,
        branch_first_output_difference=branch["first_output_difference"])
    for label, field in (("parent", "candidate_parents"), ("crop", "crop"), ("output", "boxes"),
                         ("ppo", "ppo_selected"), ("choice", "selected"), ("write", "memory_write")):
        result["first_source_"+label+"_difference"] = first_difference(traces["pulse"], traces["gated"], field, target)
    result["first_source_good_supply_lost"] = next((r["frame"] for r, g in zip(pulse[:target], gated[:target])
        if r["gt_valid"] and g["gt_valid"] and r["no_good_supply"] and not g["no_good_supply"]), None)
    return result


def recovery_match(data, entry, reference, design):
    """Preserve Stage-93's actual inclusive start range [reference, reference+15]."""
    frames = np.asarray(data["frame_id"], int)
    if frames[0] > reference:
        raise ValueError("Recovery reference precedes matched branch")
    truth = np.asarray(entry["ground_truth"], float)[frames]
    overlap = tracking_iou(data["boxes"], truth)
    valid = np.isfinite(truth).all(1) & (truth[:, 2:] > 0).all(1) & (overlap >= 0)
    good = valid & (overlap >= design["good_iou"])
    width, end = design["recovery_run"], reference+design["recovery_window"]
    eligible = [f for f in range(reference, end+1) if f+width-1 <= int(frames[-1])]
    matched = next((f for f in eligible if good[f-int(frames[0]):f-int(frames[0])+width].all()), None)
    return dict(reference_recovery=reference, allowed_start_last=end, observed_start_last=eligible[-1] if eligible else None,
        matched_good_frame=matched, aligned_recovery=matched is not None,
        complete_search_window=int(frames[-1]) >= end+width-1)


def chain_excerpt(cases, state_rows, points):
    """Return milestones and one level of actual past dependencies, not a new cohort."""
    out = []
    fields = ("cohort", "sequence", "seed", "arm", "frame", "state", "iou", "best_iou", "ppo_iou", "selected",
              "ppo_selected", "scale_active", "scale_written", "memory_write", "used_history", "scale_refs",
              "appearance_refs", "effective_appearance_refs", "reads_low_scale", "reads_low_appearance", "best_coverage", "selected_coverage")
    for case in cases:
        target = case["target_frame"]
        selected = defaultdict(set)
        for key, value in case.items():
            if key in ("anchor_frame", "target_frame", "last_good_before_target") or key.startswith(("first_context_", "first_source_")):
                if value is not None:
                    selected[int(value)].add(key)
        for frame in list(selected):
            earlier = [p for p in points if p <= frame]
            if earlier:
                selected[earlier[-1]].add("preceding_trigger")
            for lag in (1, 2):
                if frame > lag:
                    selected[frame-lag].add("preceding_prediction")
        # Include every actually consulted appearance/scale reference of these milestones once.
        for frame in list(selected):
            for rows in state_rows.values():
                row = rows[frame-1]
                reads = row["scale_refs"].split("|") + row["effective_appearance_refs"].split("|")
                for text in reads:
                    if text:
                        rf = int(text)
                        if not 1 <= rf < frame:
                            raise ValueError("Non-past excerpt dependency")
                        selected[rf].add("read_dependency")
        if any(f > target for f in selected):
            raise ValueError("Source excerpt crossed its target")
        for frame, tags in sorted(selected.items()):
            for rows in state_rows.values():
                row = rows[frame-1]
                out.append(dict(case_role=case["role"], case_anchor=case["anchor_frame"], case_target=target,
                    evidence_roles="|".join(sorted(tags)), **{k: row[k] for k in fields}))
    return out


def load_matched(root, plan, unit, kind, first, last, treatment):
    path = root / kind / f"{c.key(unit)}__f{first:06d}__{treatment}.npz"
    return prior.s95.s94.required_cache(path, dict(plan_fingerprint=plan["fingerprint"], unit=unit,
        kind=kind, first=first, last=last, treatment=treatment))


def compare_recorded(expected, actual):
    if len(expected) != len(actual):
        raise ValueError("Stage-101 table support differs")
    for recorded, row in zip(expected, actual):
        for key, text in recorded.items():
            if key not in row:
                raise ValueError("Missing Stage-101 metric: "+key)
            value = row[key]
            if value is None:
                matched = text == ""
            elif isinstance(value, str):
                matched = value == text
            elif isinstance(value, (bool, np.bool_)):
                matched = str(bool(value)) == text
            else:
                matched = np.isclose(float(text), value, rtol=0, atol=1e-12, equal_nan=True)
            if not matched:
                raise ValueError("Stage-101 table differs: "+key)


def run(root101, root100, plan101, plan100, root93, root89, plan93, plan89, report89, source, actor, design):
    full_rows, branch_frames, edges, cases, branches, recovery, triggers, excerpts = [], [], [], [], [], [], [], []
    old = source["original"]
    for spec101 in plan101["units"]:
        spec, unit = spec101["spec"], spec101["spec"]["unit"]
        cohort = spec["cohort"]
        entry, traces = prior.load_unit(root100, plan100, root93, root89, plan93, plan89, report89, source, spec)
        pack = prior.e74.learned_pack(actor["models"], unit, actor["prior"])
        choose = lambda row, history: prior.e74.choose(row, history, unit, old, actor["prior"], pack)
        state_rows = {}
        for arm, data in traces.items():
            s100.audit(data, entry, prior.e63.c.actor_plan(old), choose, arm, plan100["design"])
            state_rows[arm], refs = describe_trace(data, entry, unit, cohort, arm, old, design)
            full_rows.extend(state_rows[arm])
            edges.extend(refs)
        np.testing.assert_array_equal(spec101["points"], np.flatnonzero(traces["pulse"]["scale_active"])+1)
        point_rows = []
        for frame in spec101["points"]:
            actual = {t: load_matched(root101, plan101, unit, "points", frame, frame, t) for t in prior.TREATMENTS}
            metrics = {}
            for t, data in actual.items():
                prior.audit(data, traces["pulse"], entry, frame, frame, t, old, choose, plan101["design"])
                metrics[t] = frame_metrics(data, 0, entry["ground_truth"][frame], design)
            for key in ("candidate_parents", "candidate_boxes", "features", "crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls"):
                np.testing.assert_array_equal(actual["keep"][key][0, [0, 2]], actual["skip"][key][0, [0, 2]])
            ident = dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"], frame=frame)
            row = dict(ident, mechanism=mechanism(metrics["keep"], metrics["skip"], design),
                skip_minus_keep_iou=(metrics["skip"]["iou"]-metrics["keep"]["iou"] if metrics["keep"]["gt_valid"] else None),
                reads_low_scale=state_rows["pulse"][frame-1]["reads_low_scale"],
                reads_low_appearance=state_rows["pulse"][frame-1]["reads_low_appearance"])
            for t, values in metrics.items():
                row.update({t+"_"+k: v for k, v in values.items()})
            point_rows.append(row)
        triggers.extend(point_rows)
        local_cases = []
        for anchor in spec101["anchors"]:
            first, last = anchor["frame"], anchor["last"]
            actual = {t: load_matched(root101, plan101, unit, "branches", first, last, t) for t in prior.TREATMENTS}
            branch = dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"], frame=first, last=last,
                roles="|".join(sorted({r["role"] for r in anchor["roles"]})))
            for t, data in actual.items():
                prior.audit(data, traces["pulse"], entry, first, last, t, old, choose, plan101["design"])
                point = load_matched(root101, plan101, unit, "points", first, first, t)
                for key in s100.FIELDS:
                    np.testing.assert_array_equal(data[key][0], point[key][0])
                local, _ = describe_trace(data, entry, unit, cohort, t, old, design, prefix=traces["pulse"], skip_first=t == "skip", drops=False)
                branch_frames.extend(local)
                branch.update({t+"_"+k: v for k, v in prior.window_metrics(data, entry, design).items()})
                for role in anchor["roles"]:
                    if role["reference_recovery"] is not None:
                        recovery.append(dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"],
                            frame=first, last=last, treatment=t, **recovery_match(data, entry, role["reference_recovery"], design)))
            for label, field in (("parent", "candidate_parents"), ("crop", "crop"), ("output", "boxes"),
                    ("ppo", "ppo_selected"), ("choice", "selected"), ("write", "memory_write"),
                    ("appearance_ref", "refs"), ("scale_ref", "scale_refs")):
                f = first_difference(actual["keep"], actual["skip"], field, last-first+1)
                branch["first_"+label+"_difference"] = f+first-1 if f is not None else None
            branches.append(branch)
            for role in anchor["roles"]:
                local_cases.append(context_case(unit, cohort, anchor, role, traces, state_rows, point_rows, branch))
        cases.extend(local_cases)
        excerpts.extend(chain_excerpt(local_cases, state_rows, spec101["points"]))
        print(f"[Stage-102] {c.key(unit)}: {len(traces['pulse']['frame_id'])} source frames, {len(point_rows)} triggers", flush=True)
    # Reproduce the old trigger summary from raw caches before reporting new mechanisms.
    summary101, videos101, _ = prior.aggregate(triggers, [])
    for name, rows in (("trigger_summary", summary101), ("trigger_video", videos101), ("branches", branches)):
        compare_recorded(prior.read_csv(root101 / f"analysis/stage101_{name}.csv"), rows)
    summary, units = summarize_units(full_rows)
    return (summary, units, cases, trigger_summary(triggers), recovery, branches, triggers, edges, branch_frames, full_rows, excerpts, reference_summary(edges))


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage102_decision.json", SCHEMA)
    counts, expected = report["counts"], plan["expected"]
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage101_fingerprint"] != SOURCE101
            or report["source_runtime"] != plan["source_runtime"]
            or counts["unit_summary"] != 2*expected["units"] or counts["state_frames"] != expected["state_frames"]
            or counts["trigger_details"] != expected["points"] or counts["branches"] != expected["anchors"]
            or counts["branch_frames"] != expected["branch_frames"] or counts["cases"] != expected["roles"]
            or counts["recovery"] != expected["recovery_rows"]
            or not all(report[k] for k in ("all_development_units_audited", "source_ppo_and_state_replay_pass",
                                           "trigger_and_branch_summary_reproduction_pass", "matched_q0_q2_and_shared_first_pass"))
            or set(report["table_sha256"]) != {f"stage102_{name}.csv" for name in TABLES}
            or any(report[k] != 0 for k in ("image_reads", "new_search_calls", "new_block_calls", "training_updates", "confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-102 cached audit")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    io.verify({str(root / "analysis" / name): sha for name, sha in report["table_sha256"].items()})


def publish(root, plan, tables):
    expected = plan["expected"]
    counts = {name: len(rows) for name, rows in zip(TABLES, tables)}
    if (counts["unit_summary"] != 2*expected["units"] or counts["state_frames"] != expected["state_frames"]
            or counts["trigger_details"] != expected["points"] or counts["branches"] != expected["anchors"]
            or counts["branch_frames"] != expected["branch_frames"] or counts["cases"] != expected["roles"]):
        raise ValueError("Cached state/trigger/branch coverage differs")
    if counts["recovery"] != expected["recovery_rows"]:
        raise ValueError("Incomplete matched recovery replay")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, tables):
        path = out / f"stage102_{name}.csv"
        prior.s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(out / "stage102_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage101_fingerprint=SOURCE101,
        source_runtime=plan["source_runtime"], counts=counts, all_development_units_audited=True,
        source_ppo_and_state_replay_pass=True, trigger_and_branch_summary_reproduction_pass=True,
        matched_q0_q2_and_shared_first_pass=True, table_sha256=hashes,
        image_reads=0, new_search_calls=0, new_block_calls=0, training_updates=0, confirmation_inference_calls=0)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage100", "stage101", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    names = ("stage76", "stage89", "stage93", "stage100", "stage101")
    if not all(getattr(args, k) for k in names):
        parser.error("Supply completed Stage-76, 89, 93, 100 and 101 directories")
    root76, root89, root93, root100, root101 = sources = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-102 directory")
    with s93.writer_lock(root):
        if (root / "analysis/stage102_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-102 directory")
        prior.check_results(root101)
        prior.compat.check_results(root100)
        plan101 = c.c66.sealed(root101 / "manifest.json", prior.SCHEMA)
        report101 = c.c66.sealed(root101 / "analysis/stage101_decision.json", prior.SCHEMA)
        if report101["fingerprint"] != SOURCE101 or report101["source_stage100_fingerprint"] != prior.SOURCE100:
            raise ValueError("Need the frozen Stage-101 result")
        plan100 = c.c66.sealed(root100 / "manifest.json", s100.SCHEMA)
        report100 = c.c66.sealed(root100 / "analysis/stage100_decision.json", s100.SCHEMA)
        if report100["fingerprint"] != prior.SOURCE100:
            raise ValueError("Stage-100 source changed")
        plan93, _, plan89, report89, source = prior.s95.s94.load_run(root93, root76, root89)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch
        torch.set_num_threads(1)
        paths = [root101 / "manifest.json", root101 / "analysis/stage101_decision.json"]
        bindings = plan101["source_sha256"] | {str(p): c.file_hash(p) for p in paths}
        bindings.update({str(root101 / p): h for p, h in report101["cache_sha256"].items()})
        bindings.update({str(root101 / "analysis" / p): h for p, h in report101["table_sha256"].items()})
        units = plan101["units"]
        expected = dict(units=len(units), points=sum(len(u["points"]) for u in units),
            anchors=sum(len(u["anchors"]) for u in units), roles=sum(len(a["roles"]) for u in units for a in u["anchors"]),
            recovery_rows=sum(2 for u in units for a in u["anchors"] for r in a["roles"] if r["reference_recovery"] is not None),
            branch_frames=sum(2*(a["last"]-a["frame"]+1) for u in units for a in u["anchors"]), state_frames=85144)
        if (expected["units"], expected["points"], expected["anchors"], expected["branch_frames"]) != (40, 1834, 61, 1954):
            raise ValueError("Frozen diagnostic coverage changed")
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage102_state_chain.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_cached_state_dependency_diagnostic",
            design=DESIGN, expected=expected, source_runtime=report101["runtime"],
            source_sha256=bindings, code_sha256=plan101["code_sha256"] | {str(p): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        print("Stage-102: cached audit; no image reads, GPU inference or training", flush=True)
        tables = run(root101, root100, plan101, plan100, root93, root89, plan93, plan89, report89, source, actor, DESIGN)
        publish(root, plan, tables)
        print(f"Stage-102: {STATUS}; {expected['points']} triggers, {expected['units']} development units")


if __name__ == "__main__":
    main()
