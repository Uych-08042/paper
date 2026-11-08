"""Compare original and mixed SGLA crops on every active frozen development state."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage81_mixed_scale as s81
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


s80, s78, s76, c, io, e63, e74 = s81.s80, s81.s78, s81.s76, s81.c, s81.io, s81.e63, s81.e74
SCHEMA = "stage82_same_history_scale_supply_v1"
STATUS = "COMPLETE_SAME_HISTORY_SCALE_DIAGNOSTIC"
PROVIDERS = ("original_ppo", "mixed_ppo")
TABLES = ("provider_summary", "paired", "video_summary", "scope", "active_frames")
DESIGN = dict(s81.DESIGN, bootstrap_seed=2182, minimum_supported_videos=8, minimum_oracle_wins=7)
DELTAS = ("best_iou_gain", "good_support_gain", "selected_iou_gain", "regret_reduction")


def verified_stage81(root, source):
    p81 = c.c66.sealed(root / "manifest.json", s81.SCHEMA)
    r81 = c.c66.sealed(root / "analysis/stage81_decision.json", s81.SCHEMA)
    if (r81["status"] != s81.STATUS or r81["plan_fingerprint"] != p81["fingerprint"]
            or r81["trajectories"] != 84 or r81["confirmation_inference_calls"] != 0
            or r81["confirmation_fingerprint"] != p81["confirmation_fingerprint"]
            or r81["primary_development_pass"] or r81["secondary_development_pass"]
            or p81["design"] != s81.DESIGN or set(r81["table_sha256"]) != {f"stage81_{n}.csv" for n in s81.TABLES}
            or any(p81["source_sha256"].get(k) != v for k, v in source["evidence"].items())):
        raise ValueError("Need the completed Stage-81 development run on the supplied sources")
    bindings = {}
    for unit in p81["units"]:
        s76.binding(bindings, trace_path(root, unit))
    if c.fingerprint(bindings) != r81["evaluation_evidence_fingerprint"]:
        raise ValueError("Stage-81 full-track evidence changed")
    bindings.update({str(root / "analysis" / name): digest for name, digest in r81["table_sha256"].items()})
    bindings.update({str(p): c.file_hash(p) for p in (root / "manifest.json", root / "analysis/stage81_decision.json")})
    io.verify(bindings)
    return p81, r81, bindings


def source_unit(unit):
    return {key: value for key, value in unit.items() if key != "arm"}


def trace_path(root, unit):
    return root / "evaluation" / (c.key(source_unit(unit)) + "__" + unit["arm"] + ".npz")


def active_states(entry, data, provider, ap, design):
    """Replay only recorded history updates; never use the paired outputs as successors."""
    n = len(entry["frames"]) - 1
    if data is None or data["boxes"].shape != (n, 4) or data["candidate_boxes"].shape != (n, 4, 4):
        raise ValueError("Missing complete Stage-81 history")
    initial = np.asarray(entry["initial_box"], float)
    raw, limited = [initial.copy()], [initial.copy()]
    states = []
    for i, box in enumerate(data["boxes"]):
        frame = entry["frames"][i + 1]
        hw = (frame["height"], frame["width"])
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        mixed, changed = s81.mixed_parents(raw, limited, hw, ap)
        if provider == "mixed_ppo":
            np.testing.assert_array_equal(changed, data["mixed"][i])
        if changed.any():
            native = s76.branches.prior.history_parents(raw, ap["design"])
            actual = mixed if provider == "mixed_ppo" else native
            rects = [crop_geometry(p, hw, ap["search_factor"], ap["search_size"])[0] for p in actual]
            np.testing.assert_array_equal(rects, data["crop"][i])
            states.append(dict(frame=i+1, history=np.array(raw), parents=(native, mixed), changed=changed,
                               source_boxes=data["candidate_boxes"][i].copy(), source_action=int(data["selected"][i])))
        s78.advance(raw, limited, box, ap["design"], design)
    return states


def prepare(root81, p81, source):
    ap = e74.e67.actor_plan(source["original"])
    records = []
    for unit in p81["units"]:
        if unit["arm"] not in PROVIDERS:
            continue
        entry = c.entry_for(source["original"], unit["sequence"])
        trace = c.read_cache(trace_path(root81, unit), dict(plan_fingerprint=p81["fingerprint"], unit=unit))
        states = active_states(entry, trace, unit["arm"], ap, DESIGN)
        records.append((unit, entry, states))
    keys = [(u["sequence"], u["seed"], u["arm"]) for u, _, _ in records]
    videos = {key[0] for key in keys}
    if len(videos) != 12 or len(keys) != 48 or set(keys) != {(v, s, a) for v in videos for s in (2171, 2172) for a in PROVIDERS}:
        raise ValueError("Need 12 development videos × two seeds × two history providers")
    return records


def collect(observer, entry, states, ap, choose):
    if states:
        observer.initialize(checked_image(ap["root"], entry["frames"][0]), entry["initial_box"])
    rows = []
    for state in states:
        frame, history = state["frame"], state["history"]
        image = checked_image(ap["root"], entry["frames"][frame])
        parents, changed = state["parents"], state["changed"]
        native = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents[0]]
        mixed = native.copy()
        for slot in np.flatnonzero(changed):
            mixed[slot] = e63.actor.observe(observer, image, parents[1][slot], frame, ap)
        boxes, actions = [], []
        for actual, observed in zip(parents, (native, mixed)):
            candidates = np.asarray([r["boxes"] for r in observed])
            row = choose(dict(candidate_boxes=candidates, candidate_parents=actual,
                              features=np.asarray([e63.actor.feature_vector(r, history, ap) for r in observed])), history)
            boxes.append(candidates)
            actions.append(int(row["selected"]))
        rows.append(dict(frame=frame, boxes=boxes, actions=actions, calls=4+int(changed.sum())))
    n = len(rows)
    return dict(frame=np.array([r["frame"] for r in rows], int), calls=np.array([r["calls"] for r in rows], int),
                boxes=np.array([r["boxes"] for r in rows]).reshape(n, 2, 4, 4),
                actions=np.array([r["actions"] for r in rows], int).reshape(n, 2))


def measure(unit, entry, states, data, design):
    np.testing.assert_array_equal(data["frame"], [s["frame"] for s in states])
    np.testing.assert_array_equal(data["calls"], [4+int(s["changed"].sum()) for s in states])
    if (data["boxes"].shape != (len(states), 2, 4, 4) or data["actions"].shape != (len(states), 2)
            or not np.isfinite(data["boxes"]).all() or np.any(data["boxes"][..., 2:] <= 0)):
        raise ValueError("Incomplete paired search results")
    factual = PROVIDERS.index(unit["arm"])
    gt = np.asarray(entry["ground_truth"], float)
    rows = []
    for state, boxes, actions in zip(states, data["boxes"], data["actions"]):
        for b, q in zip(boxes, actions):
            if q not in range(4) or not c.action_mask(b)[q]:
                raise ValueError("Invalid PPO action")
        np.testing.assert_array_equal(boxes[0][~state["changed"]], boxes[1][~state["changed"]])
        y = tracking_iou(boxes, gt[state["frame"]])
        best = y.max(axis=1)
        chosen = y[np.arange(2), actions]
        available = best >= design["good_iou"]
        regret = best - chosen
        rows.append(dict(**unit, frame=state["frame"], changed_slots="|".join(map(str, np.flatnonzero(state["changed"]))),
            original_candidate_ious=y[0].tolist(), mixed_candidate_ious=y[1].tolist(),
            original_action=int(actions[0]), mixed_action=int(actions[1]), original_best=float(best[0]), mixed_best=float(best[1]),
            original_selected=float(chosen[0]), mixed_selected=float(chosen[1]),
            original_good_support=bool(available[0]), mixed_good_support=bool(available[1]),
            best_iou_gain=float(best[1]-best[0]), good_support_gain=int(available[1])-int(available[0]),
            selected_iou_gain=float(chosen[1]-chosen[0]), regret_reduction=float(regret[0]-regret[1]),
            factual_box_error=float(np.abs(boxes[factual]-state["source_boxes"]).max()),
            factual_action_changed=int(actions[factual]) != state["source_action"]))
    return rows


def summarize(scope, design):
    video, providers, paired = [], [], []
    for provider in PROVIDERS:
        provider_rows = [r for r in scope if r["arm"] == provider]
        summaries = []
        for seq in sorted({r["sequence"] for r in provider_rows}):
            seeds = [r for r in provider_rows if r["sequence"] == seq]
            if len(seeds) != 2:
                raise ValueError("Incomplete history provider seeds")
            row = dict(provider=provider, sequence=seq, active_seed_count=sum(r["active_frames"] > 0 for r in seeds),
                       active_frames=sum(r["active_frames"] for r in seeds))
            for metric in DELTAS:
                row[metric+"_full"] = float(np.mean([r[metric+"_full"] for r in seeds]))
                active = [r[metric+"_active"] for r in seeds if r["active_frames"]]
                row[metric+"_active"] = float(np.mean(active)) if active else None
            summaries.append(row)
        video.extend(summaries)
        supported = [r for r in summaries if r["active_frames"]]
        result = dict(provider=provider, videos=len(summaries), active_videos=len(supported),
                      active_frames=sum(r["active_frames"] for r in summaries),
                      gained_good_support_frames=sum(r["gained_good_support_frames"] for r in provider_rows),
                      lost_good_support_frames=sum(r["lost_good_support_frames"] for r in provider_rows))
        for metric in DELTAS:
            for suffix, cohort in (("full", summaries), ("active", supported)):
                key = f"{metric}_{suffix}"
                values = [r[key] for r in cohort]
                result[key] = float(np.mean(values)) if values else None
                if values:
                    paired.append(s80.contrast(values, provider + "_mixed_minus_original", key, design))
        providers.append(result)
    return providers, paired, video


def replay(root, plan, source, actor, records, observer):
    ap = e74.e67.actor_plan(source["original"])
    evidence, frames, scope = {}, [], []
    for (unit, entry, states), planned in zip(records, plan["units"], strict=True):
        old_unit, pack = s78.actor_pack(source, actor, source_unit(unit))

        def choose(row, history):
            return e74.choose(row, history, old_unit, source["original"], actor["prior"], pack)

        path = root / "collection" / (c.key(old_unit) + "__" + unit["arm"] + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=planned)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            with e63.source_experiment.installed(observer, [], old_unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                data = collect(observer, entry, states, ap, choose)
        rows = measure(unit, entry, states, data, plan["design"])
        if int(data["calls"].sum()) != planned["search_calls"] or len(rows) != planned["active_frames"]:
            raise ValueError("Paired search budget changed")
        if new:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        frames.extend(rows)
        row = dict(planned, factual_box_error=max((r["factual_box_error"] for r in rows), default=0.),
                   factual_action_changes=sum(r["factual_action_changed"] for r in rows),
                   gained_good_support_frames=sum(r["good_support_gain"] > 0 for r in rows),
                   lost_good_support_frames=sum(r["good_support_gain"] < 0 for r in rows))
        for metric in DELTAS:
            total = sum(r[metric] for r in rows)
            row[metric+"_full"] = total / planned["tracked_frames"]
            row[metric+"_active"] = total / len(rows) if rows else None
        scope.append(row)
        print(f"[Stage-82 common history] {unit['sequence']} {unit['arm']} seed{unit['seed']} "
              f"{len(rows)} active states / {row['search_calls']} searches", flush=True)
    providers, paired, video = summarize(scope, plan["design"])
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (providers, paired, video, scope, frames)):
        path = analysis / f"stage82_{name}.csv"
        s80.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    lookup = {(r["contrast"], r["metric"]): r for r in paired}
    checks = {}
    for row in providers:
        provider = row["provider"]
        oracle = lookup.get((provider + "_mixed_minus_original", "best_iou_gain_active"), {})
        checks[provider] = dict(
            sufficient_active_videos=row["active_videos"] >= plan["design"]["minimum_supported_videos"],
            oracle_gain_supported=oracle.get("ci_low", 0.) > 0
                and oracle.get("wins", 0) >= plan["design"]["minimum_oracle_wins"],
            more_good_support=(row["good_support_gain_active"] or 0.) > 0,
            selected_iou_worse=(row["selected_iou_gain_active"] or 0.) < 0,
            regret_increased=(row["regret_reduction_active"] or 0.) < 0)
    box_error = max(r["factual_box_error"] for r in scope)
    action_changes = sum(r["factual_action_changes"] for r in scope)
    exact_bridge = box_error == 0. and action_changes == 0
    retraining_supported = exact_bridge and all(all(r.values()) for r in checks.values())
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"],
        source_stage81_fingerprint=plan["source_stage81_fingerprint"], confirmation_fingerprint=plan["confirmation_fingerprint"],
        videos=plan["videos"], histories=len(scope), scanned_states=sum(r["tracked_frames"] for r in scope),
        active_states=len(frames), search_calls=sum(r["search_calls"] for r in scope),
        block_calls=7*sum(r["search_calls"] for r in scope),
        max_factual_box_error=box_error, factual_action_changes=action_changes, factual_exact_bridge=exact_bridge,
        checks=checks, selector_retraining_supported=retraining_supported,
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        collection_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(analysis / "stage82_decision.json", report)
    print(f"Stage-82: {STATUS} selector_retraining_supported={retraining_supported}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage77", "stage81", "output_dir"):
        p.add_argument("--" + name, type=Path, required=name == "output_dir")
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-results", action="store_true")
    p.set_defaults(mode="stage82")
    args = p.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        plan = c.c66.sealed(root / "manifest.json", SCHEMA)
        report = c.c66.sealed(root / "analysis/stage82_decision.json", SCHEMA)
        if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
                or report["search_calls"] != sum(u["search_calls"] for u in plan["units"])
                or report["active_states"] != sum(u["active_frames"] for u in plan["units"])
                or set(report["table_sha256"]) != {f"stage82_{n}.csv" for n in TABLES}):
            raise ValueError("Stage-82 result is incomplete")
        io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
        return
    if not all((args.stage76, args.stage77, args.stage81, args.checkpoint)):
        p.error("Supply Stage-76/77/81 and the native checkpoint")
    paths = [path.expanduser().resolve() for path in (args.stage76, args.stage77, args.stage81)]
    if any(root == path or root in path.parents or path in root.parents for path in paths):
        p.error("Use a separate Stage-82 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
            raise ValueError("Use an empty Stage-82 directory")
        source, actor, _ = s78.load_sources(*paths[:2])
        p81, r81, evidence = verified_stage81(paths[2], source)
        records = prepare(paths[2], p81, source)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        units = [dict(u, tracked_frames=len(e["frames"])-1, active_frames=len(states),
                      search_calls=sum(4+int(s["changed"].sum()) for s in states)) for u, e, states in records]
        if sum(u["tracked_frames"] for u in units) != 54100:
            raise ValueError("The fixed development frame inventory changed")
        bindings = dict(source["evidence"], **evidence)
        code = (__file__, s81.__file__, s80.__file__, s78.__file__, e74.__file__, Path(__file__).with_name("run_stage82_same_history.sh"))
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_same_history_layout_audit", videos=12,
            units=units, runtime=runtime, source_stage81_fingerprint=r81["fingerprint"],
            confirmation_fingerprint=r81["confirmation_fingerprint"], design=DESIGN, source_sha256=bindings,
            code_sha256={str(Path(module).resolve()): c.file_hash(module) for module in code}))
        c.publish_identical(root / "manifest.json", plan)
        print(f"Stage-82 frozen: {sum(u['active_frames'] for u in units)} active states, "
              f"{sum(u['search_calls'] for u in units)} searches", flush=True)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            replay(root, plan, source, actor, records, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
