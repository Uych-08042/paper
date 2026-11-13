"""Test causal, one-frame scale pulses over complete SGLA development videos."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage99_scale_pulse as s99
from stage27_uav_common import checked_image, crop_geometry, failure_events, tracking_iou


s97, s93, s95, c, io = s99.s97, s99.s93, s99.s95, s99.c, s99.io
e63, e74, s76, s86 = s99.e63, s99.e74, s99.s76, s99.s86
SCHEMA = "stage100_online_scale_episode_pulse_v1"
SOURCE99 = "ccc86856f8b6fdd7a3c195a373cbb10b32c6890ed54ea184db4d1383e7161310"
ARMS = ("native", "ppo", "gated", "continuous", "pulse")
DESIGN = dict(s97.DESIGN, bootstrap_seed=2200, bootstrap_samples=10000)
FIELDS = s97.FIELDS + ("scale_request", "scale_episode_start")
METRICS = s93.METRICS + ("failure_events", "good_candidate_fraction")
FLAGS = ("new_on_native_stable", "new_on_ppo_stable", "additional_on_native_stable",
         "new_on_gated_stable", "ppo_recoveries_lost", "gated_recoveries_lost")
TABLES = ("policy_summary", "paired", "video_summary", "seed_video", "scope", "recovery")
STATUSES = ("REVIEW_ONLINE_SCALE_PULSE_DEVELOPMENT", "PROMISING_ONLINE_SCALE_PULSE_DEVELOPMENT")


def activation(mode, has_refs, request, previous_request):
    if mode not in ("gated", "continuous", "pulse"):
        raise ValueError("Unknown scale controller")
    if request and not has_refs:
        raise ValueError("A crop disagreement requires past scale references")
    episode = bool(request and not previous_request)
    enabled = bool(has_refs and (mode == "continuous" or (mode == "pulse" and episode)))
    return enabled, episode


def parents_for(history, sizes, frame, mode, previous_request, hw, ap, design):
    ordinary, _ = s97.parents_at(history, sizes, frame, "control", ap, design)
    trusted, refs = s97.parents_at(history, sizes, frame, "trusted_scale", ap, design)
    has_refs = bool((refs >= 0).any())
    request = bool(has_refs and s97.scale_changes(history, trusted, hw, ap)[1])
    enabled, episode = activation(mode, has_refs, request, previous_request)
    return (trusted if enabled else ordinary,
            refs if enabled else np.full(design["scale_max_refs"], -1, int), request, episode)


def collect(observer, entry, ap, choose, mode, design, image_reader=checked_image):
    initial = np.asarray(entry["initial_box"], float)
    template = image_reader(ap["root"], entry["frames"][0])
    observer.initialize(template, initial)
    history, memory, sizes, rows, tokens = [initial], [], [], [], []
    anchor, previous_request = None, False

    def capture(_module, _args, output):
        grid = s93.s91.cpu_array(output)
        if grid.ndim != 3 or grid.shape[0] != 1:
            raise ValueError("Expected one patch grid")
        tokens.append(grid[0])

    with observer.rt.net.backbone.patch_embed.register_forward_hook(capture):
        for frame, image_entry in enumerate(entry["frames"][1:], 1):
            image = image_reader(ap["root"], image_entry)
            parents, refs, request, episode = parents_for(history, sizes, frame, mode, previous_request,
                                                         image.shape[:2], ap, design)
            tokens.clear()
            row = choose(s97.probe(observer, image, history, parents, frame, ap), history)
            if len(tokens) != 8 or any(len(t) != (observer.rt.lens_z if q % 2 == 0 else observer.rt.lens_x)
                                       for q, t in enumerate(tokens)):
                raise ValueError("Expected four template/search patch pairs")
            if anchor is None:
                params = observer.rt.tracker.params
                anchor = s93.pool(template, initial, initial, params.template_factor, params.template_size,
                                  tokens[0], design)
            alt, _ = s93.appearance_pair(row, design)
            queries = np.full((2, len(anchor)), np.nan)
            for slot, q in enumerate((int(row["selected"]), alt)):
                if q >= 0:
                    queries[slot] = s93.pool(image, row["candidate_boxes"][q], parents[q],
                        ap["search_factor"], ap["search_size"], tokens[2*q+1], design)
            decision = s93.choose_write(row, queries, anchor, memory, frame, "gated", design)
            box = row["candidate_boxes"][decision["selected"]].copy()
            accepted = s97.scale_write(decision, queries)
            changed, changed_crop = s97.scale_changes(history, parents, image.shape[:2], ap)
            rows.append(row | decision | dict(boxes=box, queries=queries, scale_refs=refs,
                scale_active=bool((refs >= 0).any()), scale_written=accepted,
                scale_changed=changed, scale_crop_changed=changed_crop,
                scale_request=request, scale_episode_start=episode))
            s97.update_sizes(sizes, frame, box, accepted, design)
            history = (history+[box])[-ap["design"]["history"]:]
            observer.rt.tracker.state = box.tolist()
            previous_request = request
            if frame % 250 == 0:
                print(f"  [Stage-100] {mode} {frame}/{len(entry['frames'])-1}", flush=True)
    return {k: np.asarray([row[k] for row in rows]) for k in FIELDS} | dict(anchor=anchor)


def decorate_control(trace, entry, ap, design):
    n = len(entry["frames"])-1
    data = s97.control_slice(trace, 1, n)
    history, sizes, previous, requests, episodes = [np.asarray(entry["initial_box"], float)], [], False, [], []
    for i, box in enumerate(data["boxes"]):
        _, _, request, episode = parents_for(history, sizes, i+1, "gated", previous, data["image_hw"][i], ap, design)
        requests.append(request)
        episodes.append(episode)
        s97.update_sizes(sizes, i+1, box, bool(data["scale_written"][i]), design)
        history = (history+[box])[-ap["design"]["history"]:]
        previous = request
    return data | dict(scale_request=np.asarray(requests), scale_episode_start=np.asarray(episodes))


def audit(data, entry, ap, choose, mode, design):
    n = len(entry["frames"])-1
    if set(data) != set(FIELDS) | {"anchor"}:
        raise ValueError("Incomplete online controller trace")
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
    e63.source_common.prior.audit_calls(data, (n, 4))
    if (not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]
            or data["anchor"].ndim != 1 or data["queries"].shape != (n, 2, len(data["anchor"]))):
        raise ValueError("Invalid decoder or appearance shape")
    history, sizes, memory, previous = [np.asarray(entry["initial_box"], float)], [], [], False
    for i, box in enumerate(data["boxes"]):
        frame = i+1
        hw = entry["frames"][frame]["height"], entry["frames"][frame]["width"]
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        parents, refs, request, episode = parents_for(history, sizes, frame, mode, previous, hw, ap, design)
        for key, value in dict(candidate_parents=parents, history_latest=history[-1], scale_refs=refs,
                scale_request=request, scale_episode_start=episode, scale_active=bool((refs >= 0).any())).items():
            np.testing.assert_array_equal(data[key][i], value, err_msg=f"{frame}:{key}")
        np.testing.assert_array_equal([data["scale_changed"][i], data["scale_crop_changed"][i]],
                                     s97.scale_changes(history, parents, hw, ap))
        np.testing.assert_array_equal(data["crop"][i], [crop_geometry(p, hw, ap["search_factor"], ap["search_size"])[0]
                                                        for p in parents])
        row = choose({k: data[k][i].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, history)
        row["history_latest"] = history[-1]
        decision = s93.choose_write(row, data["queries"][i], data["anchor"], memory, frame, "gated", design)
        for key, value in decision.items():
            if key in ("margin", "patch_margin"):
                np.testing.assert_allclose(data[key][i], value, rtol=0, atol=1e-12, equal_nan=True)
            else:
                np.testing.assert_array_equal(data[key][i], value)
        np.testing.assert_array_equal(box, row["candidate_boxes"][decision["selected"]])
        accepted = s97.scale_write(decision, data["queries"][i])
        np.testing.assert_array_equal(data["scale_written"][i], accepted)
        s97.update_sizes(sizes, frame, box, accepted, design)
        history = (history+[box])[-ap["design"]["history"]:]
        previous = request


def load_native(root93, plan93, source, spec):
    unit = dict(spec["unit"], policy="q0", seed=0)
    if spec["cohort"] == "historical" and plan93["reuse_native_historical"]:
        _, data = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
    else:
        data = s95.s94.required_cache(root93 / f"evaluation/{c.key(unit)}__native.npz",
                                     dict(plan_fingerprint=plan93["fingerprint"], unit=unit, arm="native"))
    return unit, data


def shared_prefix(data, baseline):
    different = (data["candidate_parents"] != baseline["candidate_parents"]).reshape(len(data["frame_id"]), -1).any(1)
    index = int(np.flatnonzero(different)[0]) if different.any() else len(different)
    for key in ("boxes", "candidate_boxes", "features", "queries", "memory_write", "selected", "ppo_selected"):
        np.testing.assert_array_equal(data[key][:index], baseline[key][:index], err_msg=f"Pre-intervention drift:{key}")
    np.testing.assert_array_equal(data["anchor"], baseline["anchor"])
    return index+1 if index < len(different) else None


def summarize(seeds, design):
    groups = defaultdict(list)
    for row in seeds:
        groups[row["cohort"], row["sequence"], row["arm"]].append(row)
    videos, policies, paired = [], [], []
    for (cohort, sequence, arm), rows in sorted(groups.items()):
        videos.append(dict(cohort=cohort, sequence=sequence, arm=arm,
            **{k: float(np.mean(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in METRICS},
            **{k: (sum(r[k] for r in rows if r[k] is not None) if k.endswith("recoveries_lost") else
                   any(r[k] for r in rows if r[k] is not None)) if any(r[k] is not None for r in rows) else None for k in FLAGS}))
    contrasts = (("pulse", "gated"), ("pulse", "continuous"), ("pulse", "ppo"),
                 ("pulse", "native"), ("continuous", "gated"))
    for cohort in ("all", "historical", "expanded"):
        local = [r for r in videos if cohort == "all" or r["cohort"] == cohort]
        indexed = {(r["sequence"], r["arm"]): r for r in local}
        for arm in ARMS:
            rows = [r for r in local if r["arm"] == arm]
            if not rows:
                continue
            policies.append(dict(cohort=cohort, arm=arm, videos=len(rows),
                **{k: float(np.mean(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in METRICS},
                **{k: sum(v) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in FLAGS}))
        for a, b in contrasts:
            for metric in METRICS:
                delta = [indexed[n, a][metric]-indexed[n, b][metric] for n in sorted({r["sequence"] for r in local})
                         if indexed[n, a][metric] is not None and indexed[n, b][metric] is not None]
                if delta:
                    paired.append(dict(s86.contrast(delta, f"{a}_minus_{b}", metric, design,
                        family=(cohort == "all" and metric == "auc" and b in ("gated", "continuous") and a == "pulse")), cohort=cohort))
    return policies, paired, videos


def newly_lost_ppo_recoveries(recoveries):
    lost = {arm: {(r["sequence"], r["seed"], r["ppo_failure_frame"], r["ppo_recovery_frame"])
                  for r in recoveries if r["reference_arm"] == "ppo" and r["arm"] == arm
                  and r["matched_good_frame"] is None} for arm in ("gated", "pulse")}
    return lost["pulse"] - lost["gated"]


def publish(root, plan, evidence, seeds, scope, recoveries):
    if len(seeds) != plan["trajectories"] or len(scope) != len(seeds) or len(evidence) != plan["trajectories"]:
        raise ValueError("Incomplete full-video cohort")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    policies, paired, videos = summarize(seeds, plan["design"])
    macro = {r["arm"]: r for r in policies if r["cohort"] == "all"}
    auc = {r["contrast"]: r for r in paired if r["cohort"] == "all" and r["metric"] == "auc"}
    own, baseline = macro["pulse"], macro["gated"]
    checks = dict(practical_auc_vs_gated=auc["pulse_minus_gated"]["mean_delta"] >= plan["design"]["practical_auc"],
        two_primary_family_ci_positive=all(auc[f"pulse_minus_{b}"]["family_ci_low"] > 0 for b in ("gated", "continuous")),
        low_iou_not_worse=own["low_iou_fraction"] <= baseline["low_iou_fraction"],
        longest_failure_not_worse=own["longest_failure_fraction"] <= baseline["longest_failure_fraction"],
        reliable_auc_not_worse=own["reliable_auc"] >= baseline["reliable_auc"],
        no_new_on_gated_stable=own["new_on_gated_stable"] == 0,
        no_new_on_ppo_stable=own["new_on_ppo_stable"] == 0,
        no_gated_recovery_lost=own["gated_recoveries_lost"] == 0,
        no_new_ppo_recovery_loss=not newly_lost_ppo_recoveries(recoveries),
        nontrivial_usage=sum(r["scale_crop_changed_frames"] for r in scope if r["arm"] == "pulse") > 0)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (policies, paired, videos, seeds, scope, recoveries)):
        path = out / f"stage100_{name}.csv"
        s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    calls = sum(r["new_search_calls"] for r in scope)
    logical = sum(r["logical_search_calls"] for r in scope)
    drift = [r["source_box_error_px"] for r in scope if r["source_box_error_px"] is not None]
    report = io.seal(dict(schema=SCHEMA, status=STATUSES[int(all(checks.values()))],
        role="consumed_development_full_video_online_scale_controller", plan_fingerprint=plan["fingerprint"],
        source_stage99_fingerprint=SOURCE99, runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        reuse_sources=plan["reuse_sources"], trajectories=len(scope), new_full_trajectories=sum(r["new_search_calls"] > 0 for r in scope),
        logical_search_calls=logical, new_search_calls=calls, new_block_calls=calls*7,
        maximum_source_control_box_error_px=max(drift), source_correspondence_pass=max(drift) <= plan["box_atol"],
        shared_pre_intervention_checks_pass=True, new_ppo_recoveries_lost_vs_gated=len(newly_lost_ppo_recoveries(recoveries)),
        training_updates=0, confirmation_inference_calls=0,
        development_checks=checks, development_pass=all(checks.values()), cache_sha256=evidence, table_sha256=hashes))
    c.publish_identical(out / "stage100_decision.json", report)
    check_results(root)
    for arm in ARMS:
        print(f"{arm:12s} AUC={macro[arm]['auc']:.4f} low-IoU={100*macro[arm]['low_iou_fraction']:.2f}%")
    print(f"Stage-100: {report['status']} ({calls} new searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage100_decision.json", SCHEMA)
    if (report["status"] not in STATUSES or report["plan_fingerprint"] != plan["fingerprint"]
            or report["trajectories"] != plan["trajectories"] or len(report["cache_sha256"]) != plan["trajectories"]
            or report["new_full_trajectories"] != plan["new_trajectories"]
            or report["new_search_calls"] != plan["new_search_calls"] or report["new_block_calls"] != 7*plan["new_search_calls"]
            or report["logical_search_calls"] != plan["logical_search_calls"]
            or set(report["table_sha256"]) != {f"stage100_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-100 result")
    io.verify({str(root / "evaluation" / name): sha for name, sha in report["cache_sha256"].items()})
    io.verify({str(root / "analysis" / name): sha for name, sha in report["table_sha256"].items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def run(root, root93, root89, plan, plan93, plan89, report89, source, actor, observer):
    old, design = source["original"], plan["design"]
    ap = e63.c.actor_plan(old)
    groups = defaultdict(list)
    for spec in plan89["units"]:
        groups[spec["unit"]["sequence"]].append(spec)
    seeds, scope, recoveries, evidence = [], [], [], {}
    for sequence, specs in groups.items():
        native_unit, source_native = load_native(root93, plan93, source, specs[0])
        native = None
        for spec in specs:
            unit, cohort = spec["unit"], spec["cohort"]
            entry, references = s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
            frames = np.arange(1, len(entry["frames"]))
            pack = e74.learned_pack(actor["models"], unit, actor["prior"])

            def choose(row, history):
                return e74.choose(row, history, unit, old, actor["prior"], pack)

            arms = ARMS if native is None else ARMS[1:]
            traces = {}
            with e63.source_experiment.installed(observer, [], unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                for arm in arms:
                    owner = native_unit if arm == "native" else unit
                    path = root / "evaluation" / f"{c.key(owner)}__{arm}.npz"
                    meta = dict(plan_fingerprint=plan["fingerprint"], unit=owner, arm=arm)
                    data = c.read_cache(path, meta)
                    fresh = data is None
                    reused = plan["reuse_sources"] and arm in ("native", "ppo", "gated")
                    if fresh:
                        if reused:
                            original = source_native if arm == "native" else references[arm]
                            data = decorate_control(original, entry, ap, design) if arm == "gated" else original
                        elif arm == "native":
                            data = s93.s85.collect(observer, entry, "native_sgla", ap, None, {})
                        elif arm == "ppo":
                            data = s86.collect(observer, entry, np.asarray([entry["initial_box"]], float), frames, 0., ap, choose)
                        else:
                            data = collect(observer, entry, ap, choose, arm, design)
                    if arm == "native":
                        # Historical one-search sources may contain additional receipt fields.
                        e63.source_common.prior.audit_calls(data, (len(frames), 1))
                        np.testing.assert_array_equal(data["frame_id"], frames)
                        if np.any(data["selected"] != 0):
                            raise ValueError("Native baseline changed its single search")
                        if not reused:
                            s93.s85.audit(data, entry, "native_sgla", ap, None, {})
                        native = data
                    elif arm == "ppo":
                        s86.audit({k: data[k] for k in s86.FIELDS}, np.asarray([entry["initial_box"]], float), frames, 0., ap, choose)
                    else:
                        audit(data, entry, ap, choose, arm, design)
                    if fresh:
                        c.save(path, data, meta)
                    evidence[path.name] = c.file_hash(path)
                    traces[arm] = data
                    first = shared_prefix(data, traces["gated"]) if arm in ("continuous", "pulse") else None
                    original = source_native if arm == "native" else references.get(arm)
                    drift = float(np.abs(data["boxes"]-original["boxes"]).max()) if original is not None else None
                    if reused and drift != 0:
                        raise ValueError("Reused baseline drifted")
                    scope.append(dict(cohort=cohort, sequence=sequence, seed=owner["seed"], arm=arm,
                        frames=len(frames), logical_search_calls=len(frames)*(1 if arm == "native" else 4),
                        new_search_calls=0 if reused else len(frames)*(1 if arm == "native" else 4),
                        source_box_error_px=drift, first_parent_difference_from_gated=first,
                        **{k+"_frames": int(np.count_nonzero(data.get(k, False))) for k in
                           ("scale_request", "scale_episode_start", "scale_active", "scale_changed", "scale_crop_changed")},
                        appearance_switches=int(np.count_nonzero(data["selected"] != data.get("ppo_selected", data["selected"])))))
            traces["native"] = native
            truth = np.asarray(entry["ground_truth"], float)[1:]
            gated_iou = tracking_iou(traces["gated"]["boxes"], truth)
            for arm in arms:
                data = traces[arm]
                owner = native_unit if arm == "native" else unit
                row, events = s93.describe_arm(entry, cohort, owner, arm, data, native, traces["ppo"], design)
                overlaps = tracking_iou(data["boxes"], truth)
                own_events = s93.recovery_comparison(gated_iou, overlaps, design) if arm != "native" else []
                row.update(failure_events=len(failure_events(overlaps)), good_candidate_fraction=float(
                    (tracking_iou(data["candidate_boxes"], truth[:, None]).max(1) >= design["good_iou"]).mean()),
                    new_on_gated_stable=not failure_events(gated_iou) and bool(failure_events(overlaps)) if arm != "native" else None,
                    gated_recoveries_lost=sum(r["matched_good_frame"] is None for r in own_events) if arm != "native" else None)
                seeds.append(row)
                recoveries.extend(dict(e, reference_arm="ppo") for e in events)
                recoveries.extend(dict(cohort=cohort, sequence=sequence, seed=owner["seed"], arm=arm,
                                       reference_arm="gated", **e) for e in own_events)
            print(f"[Stage-100] {sequence} seed{unit['seed']} complete", flush=True)
    publish(root, plan, evidence, seeds, scope, recoveries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage99", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage100")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage99, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 99 and the original checkpoint")
    root76, root89, root93, root99 = [p.expanduser().resolve() for p in
                                   (args.stage76, args.stage89, args.stage93, args.stage99)]
    if any(root == p or root in p.parents or p in root.parents for p in (root76, root89, root93, root99)):
        parser.error("Use a separate Stage-100 directory")
    with s93.writer_lock(root):
        if (root / "analysis/stage100_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-100 directory")
        s99.check_results(root99)
        report99 = c.c66.sealed(root99 / "analysis/stage99_decision.json", s99.SCHEMA)
        if report99["fingerprint"] != SOURCE99:
            raise ValueError("Need the frozen Stage-99 result")
        plan99 = c.c66.sealed(root99 / "manifest.json", s99.SCHEMA)
        plan93, report93, plan89, report89, source = s95.s94.load_run(root93, root76, root89)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO source changed")
        import torch

        torch.set_num_threads(1)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report93["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Frozen software or weights changed")
        reuse = not differences
        paths = [root99 / "manifest.json", root99 / "analysis/stage99_decision.json",
                 root93 / "manifest.json", root93 / "analysis/stage93_decision.json"]
        code = [Path(__file__), Path(__file__).with_name("run_stage100_online_scale.sh"), Path(s97.__file__)]
        bindings = {str(p): c.file_hash(p) for p in paths} | plan99["source_sha256"] | plan93["source_sha256"] | actor["bindings"]
        bindings.update({str(root93 / "evaluation" / n): h for n, h in report93["cache_sha256"].items()})
        # Roster and counts are fixed by Stage-93, without reading recovery outcomes into the controller.
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_online_search_geometry", design=DESIGN,
            source_stage99_fingerprint=SOURCE99, runtime=runtime, runtime_differences=differences, reuse_sources=reuse,
            units=plan89["units"], trajectories=188, new_trajectories=80 if reuse else 188,
            logical_search_calls=710199, new_search_calls=340576 if reuse else 710199,
            box_atol=source["original"]["actor_design"]["box_atol"],
            source_sha256=bindings, code_sha256={str(p.resolve()): c.file_hash(p) for p in code}
                | plan99["code_sha256"] | plan93["code_sha256"]))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, root93, root89, plan, plan93, plan89, report89, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
