"""Match four-action supervision on immediate IoU and an eight-frame rollout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage75_l20_replay as replay
import stage75_ppo_failure_common as source_io
import stage75_ppo_failure_experiment as branches
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, tracking_iou
from stage51_state_summary import describe
from stage61_l40_summary import contrast
from stage68_advantage_summary import publish_table


SCHEMA = "stage76_four_candidate_horizon_supervision_v1"
DESIGN = dict(horizon=8, anchors_per_video=48, epochs=16, minibatch=64)
SUPERVISED = ("immediate_ce", "horizon_ce")
CONTROLS = ("q0", "quality", "future")
POLICIES = CONTROLS + ("ppo",) + SUPERVISED
e = branches.e74
c = source_io.source_common
startup = replay.startup
binding = e.e63.source_experiment.bind


def fixed_frames(tracked, design):
    last = tracked - design["horizon"] + 1
    if last < 1:
        raise ValueError("Fit video is shorter than the fixed label horizon")
    return np.linspace(1, last, min(last, design["anchors_per_video"]), dtype=int).tolist()


def device_runtime(plan, torch, index):
    available = startup.gpu_inventory(plan, torch)["visible_cuda_devices"]
    gpu = next((row for row in available if row["index"] == index), None)
    if gpu is None or gpu["name"] not in ("NVIDIA L20", "NVIDIA L40") or gpu["uuid"] in ("", "None", "unavailable"):
        raise ValueError(f"Need a visible L20 or L40 with UUID at logical index {index}: {available}")
    old = plan["runtime_identity"]
    return dict(old, tracker=dict(old["tracker"], device=gpu["name"]), gpu_uuid=gpu["uuid"])


def make_plan(source, runtime):
    old = source["plan"]
    fit = []
    units = []
    for job in old["jobs"]:
        fold, seed = job["fold"], job["seed"]
        split = next(row for row in old["folds"] if row["fold"] == fold)
        last_episode = {episode["sequence"]: i for i, episode in enumerate(job["episodes"])}
        if set(last_episode) != set(split["fit"]) or set(split["fit"]) & set(split["held_out"]):
            raise ValueError("Stage-74 fit groups differ from the sealed split")
        for seq in split["fit"]:
            frames = len(c.entry_for(old, seq)["frames"]) - 1
            fit.append(dict(sequence=seq, origin="native", policy="ppo", fold=fold, seed=seed,
                            episode=last_episode[seq], frames=fixed_frames(frames, DESIGN)))
        units.extend(dict(sequence=seq, origin="native", fold=fold, seed=seed, policy=policy)
                     for seq in split["held_out"] for policy in ("ppo",) + SUPERVISED)
    units.extend(dict(sequence=seq, origin="native", fold=split["fold"], seed=0, policy=policy)
                 for split in old["folds"] for seq in split["held_out"] for policy in CONTROLS)
    if len({c.key(unit) for unit in units}) != len(units):
        raise ValueError("Duplicated held-out trajectory")
    fit_states = sum(len(group["frames"]) for group in fit)
    code = [Path(__file__), Path(__file__).with_name("run_stage76_horizon.sh")]
    code += [Path(m.__file__) for m in (replay, branches, source_io)]
    return source_io.seal(dict(schema=SCHEMA, role="consumed_development_matched_four_candidate_label_comparison",
        source_input=str(source["root"]), source_report_fingerprint=source["report"]["fingerprint"],
        source_training_fingerprint=source["models"]["fingerprint"],
        source_bindings_fingerprint=c.fingerprint(source["bindings"]), code_sha256={str(p.resolve()): c.file_hash(p) for p in code},
        runtime=runtime, declared_runtime_differences=startup.differences(old["runtime_identity"], runtime),
        design=dict(DESIGN), fit_groups=fit, units=units, fit_states=fit_states,
        fit_search_cap=4 * fit_states * (1 + 4 * (DESIGN["horizon"] - 1)),
        eval_search_calls=4 * sum(len(c.entry_for(old, u["sequence"])["frames"]) - 1 for u in units)))


def fit_trace(source, group):
    old = source["plan"]
    job = next(job for job in old["jobs"] if c.job_key(job) == c.job_key(group))
    index = group["episode"]
    if job["episodes"][index]["sequence"] != group["sequence"]:
        raise ValueError("Wrong fit history episode")
    c.check_fit(old, job, group["sequence"])
    before = c.sealed(c.step_path(source["root"], job, index))
    trace = c.read_cache(c.trace_path(source["root"], job, index), e.train_meta(old, job, index, before))
    if trace is None or set(trace) != set(e.TRAIN_FIELDS):
        raise ValueError("Missing sealed Stage-74 fit history")
    return trace


def collect_fit(source, group, design, observer, image_reader=checked_image):
    trace = fit_trace(source, group)
    old = source["plan"]
    entry = c.entry_for(old, group["sequence"])
    unit = {k: group[k] for k in ("sequence", "origin", "policy", "fold", "seed")}
    pack = e.learned_pack(source["models"], unit, source["prior"])
    horizon = design["horizon"]
    rows = []
    for frame in group["frames"]:
        case = dict(frame=frame, horizon=horizon, case_id=f"{group['sequence']}:{frame}", unit=unit)
        anchor = branches.observe_anchor(observer, entry, trace, case, source, pack, image_reader)
        case = replay.audit_anchor(anchor, trace, case, entry, source, pack)
        truth = np.asarray(entry["ground_truth"], float)[frame:frame+horizon]
        outcomes = np.zeros((4, horizon), dtype=float)
        outcomes[:, 0] = tracking_iou(anchor["candidate_boxes"][0], truth[0])
        calls = int(anchor["head_calls"].sum())
        audit_case = dict(case, recorded_action=None)
        for action in case["legal_actions"]:
            tail = branches.collect_tail(observer, entry, trace, anchor, case, action, source, pack, image_reader)
            branches.audit_tail(tail, trace, anchor, entry, audit_case, action, source, pack)
            outcomes[action, 1:] = tracking_iou(tail["boxes"], truth[1:])
            calls += int(tail["head_calls"].sum())
        rows.append(dict(frame=frame, x=anchor["x"][0], action_mask=anchor["action_mask"][0],
                         baseline_action=case["recorded_action"], path_iou=outcomes, search_calls=calls))
    return {k: np.asarray([row[k] for row in rows]) for k in rows[0]}


def labels(data, frames, design, epsilon):
    n, h = len(frames), design["horizon"]
    mask, iou, chosen = data["action_mask"], data["path_iou"], data["baseline_action"]
    if (not np.array_equal(data["frame"], frames) or data["x"].shape != (n, 256)
            or not np.isfinite(data["x"]).all() or mask.shape != (n, 4) or mask.dtype != bool
            or not mask[:, 0].all() or iou.shape != (n, 4, h) or not np.isfinite(iou).all()
            or np.any((iou < 0) | (iou > 1)) or chosen.shape != (n,) or chosen.dtype.kind not in "iu"
            or np.any((chosen < 0) | (chosen >= 4)) or not mask[np.arange(n), chosen].all()
            or not np.array_equal(data["search_calls"], 4 * (1 + mask.sum(1) * (h - 1)))):
        raise ValueError("Invalid four-candidate fit receipt")
    means = iou.mean(axis=-1)
    now = np.array([c.best_label(y, m, epsilon) for y, m in zip(iou[:, :, 0], mask)])
    future = np.array([c.best_label(y, m, epsilon) for y, m in zip(means, mask)])
    return dict(immediate_ce=now, horizon_ce=future), means


def train_pair(torch, chunks, job, norm, design):
    x = np.concatenate([data["x"] for data, _, _ in chunks])
    mask = np.concatenate([data["action_mask"] for data, _, _ in chunks])
    y = {p: np.concatenate([target[p] for _, target, _ in chunks]) for p in SUPERVISED}
    baseline = np.concatenate([mean[np.arange(len(mean)), data["baseline_action"]] for data, _, mean in chunks])
    initial = e.learning.initial(torch, job["seed"], norm, design)
    nets = {p: e.learning.restore(torch, initial["state"], design) for p in SUPERVISED}
    optimizers = {p: e.learning.optimizer(torch, net, None, design) for p, net in nets.items()}
    inputs = torch.as_tensor(x, dtype=torch.float64)
    legal = torch.as_tensor(mask)
    returns = torch.as_tensor(baseline, dtype=torch.float64)
    targets = {p: torch.as_tensor(t) for p, t in y.items()}
    rng = np.random.default_rng(job["seed"] + 100003 * job["fold"])
    updates = 0
    for _ in range(design["epochs"]):
        order = rng.permutation(len(x))
        for start in range(0, len(order), design["minibatch"]):
            idx = order[start:start + design["minibatch"]]
            for p in SUPERVISED:
                net, opt = nets[p], optimizers[p]
                batch = {"x": inputs[idx], "action_mask": legal[idx], "label": targets[p][idx], "return": returns[idx]}
                opt.zero_grad(set_to_none=True)
                loss, _ = e.learning.objective(torch, net, batch, design, "myopic")
                loss.backward()
                e.learning.clipping.clip_branches(torch, net, design)
                opt.step()
            updates += 1
    return {p: dict(state=e.learning.export(net), norm=norm) for p, net in nets.items()}, updates


def fit(root, plan, source, observer, torch):
    lookup = {c.job_key(job): job for job in source["plan"]["jobs"]}
    chunks = {k: [] for k in lookup}
    hashes, support = {}, []
    calls = 0
    epsilon = source["plan"]["design"]["tie_epsilon"]
    for group in plan["fit_groups"]:
        key = c.job_key(group)
        path = root / "fit" / key / (group["sequence"] + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], group=group)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            data = collect_fit(source, group, plan["design"], observer)
        target, means = labels(data, group["frames"], plan["design"], epsilon)
        if new:
            c.save(path, data, meta)
        binding(hashes, path)
        chunks[key].append((data, target, means))
        n = len(group["frames"])
        count = int(data["search_calls"].sum())
        calls += count
        i, h = target["immediate_ce"], target["horizon_ce"]
        support.append(dict(fold=group["fold"], seed=group["seed"], sequence=group["sequence"], states=n,
                            labels_differ=int((i != h).sum()), search_calls=count,
                            mean_oracle_horizon_gain=float((means[np.arange(n), h] - means[np.arange(n), i]).mean())))
        print(f"[fit labels] {key} {group['sequence']} {n} states", flush=True)
    if calls > plan["fit_search_cap"] or sum(row["states"] for row in support) != plan["fit_states"]:
        raise ValueError("Fit budget changed")
    models = {}
    design = dict(source["plan"]["design"], **plan["design"])
    for key, job in lookup.items():
        path = root / "models" / (key + ".json")
        inputs = dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], job=key,
                      labels_fingerprint=c.fingerprint({p: d for p, d in hashes.items() if Path(p).parent.name == key}))
        if path.exists():
            record = c.c66.sealed(path, SCHEMA)
            if any(record[k] != v for k, v in inputs.items()):
                raise ValueError("Committed supervised controller inputs changed")
        else:
            norm = source["models"]["controllers"][key]["ppo"]["norm"]
            packs, updates = train_pair(torch, chunks[key], job, norm, design)
            record = source_io.seal(dict(inputs, packs=packs, updates_per_arm=updates))
            c.publish_identical(path, record)
        models[key] = record
    return models, support, hashes, calls


def evaluate(root, plan, source, models, observer):
    old = source["plan"]
    traces, hashes, scope = {}, {}, []
    for unit in plan["units"]:
        entry = c.entry_for(old, unit["sequence"])
        policy = unit["policy"]
        if policy in SUPERVISED:
            record = models[c.job_key(unit)]
            actor, pack, controller = dict(unit, policy="myopic"), record["packs"][policy], record["fingerprint"]
        else:
            actor = unit
            pack = None if policy in CONTROLS else e.learned_pack(source["models"], actor, source["prior"])
            controller = source["models"]["fingerprint"]
        key = c.key(unit)
        path = root / "evaluation" / (key + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, controller=controller)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            data = e.run_actor(observer, entry, actor, old, source["prior"], pack)
        count = len(entry["frames"]) - 1
        branches.prior.audit_trace(data, count, 4)
        if set(data) != set(e.FIELDS) or not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > old["actor_design"]["box_atol"]:
            raise ValueError(f"Invalid full trajectory: {key}")
        if policy in ("q0", "quality"):
            expected = np.zeros(count, int) if policy == "q0" else data["quality_choice"]
            np.testing.assert_array_equal(data["selected"], expected)
        if policy not in CONTROLS:
            e.audit(data, entry, actor, old, source["prior"], pack)
        if new:
            c.save(path, data, meta)
        binding(hashes, path)
        traces[key] = data["boxes"]
        scope.append(dict(**unit, frames=count, search_calls=int(data["head_calls"].sum()),
                          block_calls=int(data["block_calls"].sum())))
        print(f"[full evaluation] {key}", flush=True)
    calls = sum(row["search_calls"] for row in scope)
    if calls != plan["eval_search_calls"] or sum(row["block_calls"] for row in scope) != 7 * calls:
        raise ValueError("Full evaluation search budget changed")
    return traces, scope, hashes


def evaluate_tables(plan, source, traces):
    old = source["plan"]
    units = plan["units"]
    names = [entry["sequence"] for entry in old["entries"]]
    reference = {u["sequence"]: traces[c.key(u)] for u in units if u["policy"] == "q0"}
    seed_rows = []
    for unit in units:
        seq = unit["sequence"]
        entry = c.entry_for(old, seq)
        native = reference[seq]
        native_low = tracking_iou(native, np.asarray(entry["ground_truth"])[1:]) < .2
        metrics, _ = describe(dict(boxes=traces[c.key(unit)]), entry, native, native_low)
        seed_rows.append(dict(**unit, **metrics))
    keys = ("auc", "low_iou_fraction", "reliable_auc", "longest_failure_fraction", "unrecovered_events")
    video = {}
    for seq in names:
        for p in POLICIES:
            group = [row for row in seed_rows if row["sequence"] == seq and row["policy"] == p]
            expected = 1 if p in CONTROLS else len(old["design"]["seeds"])
            if len(group) != expected:
                raise ValueError("Incomplete same-card comparison")
            video[seq, p] = dict(sequence=seq, policy=p, introduced_failure=any(r["introduced_failure"] for r in group))
            for k in keys:
                values = [r[k] for r in group if r[k] is not None]
                video[seq, p][k] = float(np.mean(values)) if values else None
    macro = []
    for p in POLICIES:
        group = [video[n, p] for n in names]
        macro.append(dict(policy=p, videos=len(group), introduced_failure_videos=sum(r["introduced_failure"] for r in group)))
        for k in keys:
            values = [r[k] for r in group if r[k] is not None]
            macro[-1][k] = float(np.mean(values)) if values else None
    pairs = []
    primary = (("horizon_ce", "immediate_ce"), ("ppo", "horizon_ce"))
    comparisons = primary + tuple((a, b) for a in ("horizon_ce", "ppo") for b in ("quality", "future", "q0"))
    for a, b in comparisons:
        for metric in keys[:-1]:
            supported = [n for n in names if video[n, a][metric] is not None and video[n, b][metric] is not None]
            if supported:
                difference = [video[n, a][metric] - video[n, b][metric] for n in supported]
                row = contrast(difference, f"{a}_minus_{b}", metric, old["design"], family=(a, b) in primary and metric == "auc")
                pairs.append(dict(row, supported_videos="|".join(supported)))
    return dict(policy_summary=macro, paired=pairs, video_summary=list(video.values()), seed_video_summary=seed_rows)


def run(root, plan, source, observer, torch):
    heads = set(source["run"]["observed_heads"].values())
    if len(heads) != 1:
        raise ValueError("Need one frozen native SGLA head")
    head, = heads
    with e.e63.source_experiment.installed(observer, [], plan["units"][0]) as actual:
        if actual != head:
            raise ValueError("Native SGLA head changed")
        models, support, fit_hashes, fit_calls = fit(root, plan, source, observer, torch)
        traces, scope, eval_hashes = evaluate(root, plan, source, models, observer)
    tables = evaluate_tables(plan, source, traces)
    tables.update(scope=scope, fit_support=support)
    table_hashes = {}
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    for name, rows in tables.items():
        path = directory / f"stage76_{name}.csv"
        publish_table(path, rows)
        table_hashes[path.name] = c.file_hash(path)
    source_io.verify(source["bindings"])
    source_io.verify(plan["code_sha256"])
    retention = {}
    for p in ("horizon_ce", "ppo"):
        row = next(r for r in tables["policy_summary"] if r["policy"] == p)
        native = next(r for r in tables["policy_summary"] if r["policy"] == "q0")
        reliable = next((r for r in tables["paired"] if r["contrast"] == f"{p}_minus_q0" and r["metric"] == "reliable_auc"), None)
        retention[p] = dict(introduced_failure_videos=row["introduced_failure_videos"],
                            low_iou_delta=row["low_iou_fraction"] - native["low_iou_fraction"],
                            reliable_auc_ci_low=None if reliable is None else reliable["ci_low"])
        guard = retention[p]
        guard["pass"] = bool(guard["introduced_failure_videos"] == 0 and guard["low_iou_delta"] <= 0
                             and reliable is not None and reliable["ci_low"] >= source["plan"]["design"]["reliable_floor"])
    report = source_io.seal(dict(schema=SCHEMA, status="COMPLETE_MATCHED_HORIZON_SUPERVISION_DEVELOPMENT",
        plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"], declared_runtime_differences=plan["declared_runtime_differences"],
        source_report_fingerprint=plan["source_report_fingerprint"], source_training_fingerprint=plan["source_training_fingerprint"],
        design=plan["design"],
        trajectories=len(scope), fit_states=plan["fit_states"], fit_search_calls=fit_calls, evaluation_search_calls=plan["eval_search_calls"],
        total_block_calls=7 * (fit_calls + plan["eval_search_calls"]), retention=retention,
        model_fingerprints={k: v["fingerprint"] for k, v in models.items()}, supervised_updates_per_job={k: v["updates_per_arm"] for k, v in models.items()},
        native_head_sha256=head, fit_evidence_fingerprint=c.fingerprint(fit_hashes), evaluation_evidence_fingerprint=c.fingerprint(eval_hashes),
        sgla_training_performed=False, new_ppo_training=False, new_recovery_search_calls=0, table_sha256=table_hashes))
    c.publish_identical(directory / "stage76_decision.json", report)
    for row in tables["policy_summary"]:
        print(f"{row['policy']:15s} AUC={row['auc']:.4f} low-IoU={row['low_iou_fraction']:.2%}", flush=True)
    print("Stage-76: " + report["status"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.set_defaults(mode="stage76")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Supply the original frozen SGLA checkpoint")
    root, parent = (Path(value).expanduser().resolve() for value in (args.output_dir, args.input))
    if root == parent or root in parent.parents or parent in root.parents:
        parser.error("Use a Stage-76 output separate from Stage-74")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-76 directory")
        print("[Stage-76] Verifying the sealed Stage-74 source", flush=True)
        source = source_io.load_source(parent)
        runtime = device_runtime(source["plan"], torch, args.gpu)
        plan = make_plan(source, runtime)
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["plan"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = startup.checked_runtime(args, frozen, root)
        observer = e.e63.Capture(native)
        try:
            run(root, plan, source, observer, torch)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
