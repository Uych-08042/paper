"""Live fit-only PPO interaction, matched supervised updates, full held-out policies."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import numpy as np
import stage74_ppo_common as c
import stage74_ppo_runtime as learning
import stage73_ppo_experiment as source_experiment
from stage27_uav_common import checked_image, crop_geometry, nonempty_path, tracking_iou
from stage27_uav_collect import writer_lock

e70, e68, e67, e63 = c.e70, c.prior.e68, c.prior.e67, c.e63
FIELDS = e68.FIELDS + ("x", "action_mask", "logits", "probabilities", "value", "eligible", "action", "log_prob")
TRAIN_FIELDS = FIELDS + ("reward", "myopic_label", "candidate_iou")


def prepare(source, root, hardware):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents: raise ValueError("Use separate Stage-74 directory")
    plan, _ = c.make_plan(source, hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()): raise ValueError("Need empty Stage-74 root")
        c.publish_identical(root/"manifest.json", plan)
    print("Stage-74 prepared: {} live fit steps / {} searches; {} held-out trajectories / {} searches".format(
        plan["training_environment_steps"], plan["training_search_calls"], len(plan["units"]), plan["formal_eval_calls"]), flush=True)
    return plan


def eval_meta(plan, unit, models, smoke=False):
    return dict(plan_fingerprint=plan["fingerprint"], unit=unit, training_fingerprint=models["fingerprint"], smoke=smoke)


def train_meta(plan, job, index, before):
    return dict(plan_fingerprint=plan["fingerprint"], fold=job["fold"], seed=job["seed"], index=index,
                episode=job["episodes"][index], before_fingerprint=before["fingerprint"])


def control_pack(source, unit): return c.prior.model_pack(source["models"], unit)
def learned_pack(models,unit,source):
    if unit["policy"] in c.REUSED:return source_experiment.learned_pack(source["stage73"]["models"],c.source_unit(unit),source["stage73"]["old"])
    return models["controllers"][c.job_key(unit)][unit["policy"]]


def choose(row,history,unit,plan,source,pack=None,rng=None):
    if unit["policy"] in c.CONTROLS+c.REUSED:
        observed=source_experiment.choose(row,history,c.source_unit(unit),plan,source,pack)
        action=int(observed["selected"]);p=np.eye(4)[action]
        observed.update(x=np.zeros(256),action_mask=np.ones(4,bool),logits=np.zeros(4),probabilities=p,
            value=0.,eligible=False,action=action,log_prob=0.)
        return observed
    features=c.c66.c64.choice_features(row["features"],row["candidate_boxes"],history)
    d=c.c66.decide("quality",row["features"],features,c.fit_pack(source,unit["fold"]),plan["controller_design"])
    info=c.gate(features,row["candidate_boxes"],pack,plan["design"])
    action=int(rng.choice(4,p=info["probabilities"])) if rng is not None and info["eligible"] else int(np.argmax(info["probabilities"]))
    d.update(selected=action,veto=bool(d["quality_choice"] and action!=d["quality_choice"]),immediate_gain=0.,future_gain=0.)
    row.update(d,**info,action=action,log_prob=c.log_probability(action,info["logits"],info["action_mask"]),choice_features=features,
        boxes=row["candidate_boxes"][action].copy(),parents=row["candidate_parents"][action].copy())
    return row


def run_actor(observer, entry, unit, plan, source, pack=None, rng=None, limit=None, prefix=None, image_reader=checked_image, fit=False):
    # No GT enters this loop's actor. Labels are attached only AFTER the action/trajectory exists.
    observer.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
    history = [np.asarray(entry["initial_box"], float)]; rows = []; start = 1
    if prefix is not None:
        start = len(prefix["frame_id"])+1
        for b in prefix["boxes"]: history.append(b.copy()); history = history[-plan["actor_design"]["history"]:]
    limit = len(entry["frames"])-1 if limit is None else limit
    for frame in range(start, limit+1):
        row = e63.probe(observer, image_reader(plan["root"], entry["frames"][frame]), history, frame, "history", e67.actor_plan(plan))
        row = choose(row, history, unit, plan, source, pack, rng)
        rows.append(row); history.append(row["boxes"].copy()); history = history[-plan["actor_design"]["history"]:]
        observer.rt.tracker.state = row["boxes"].tolist()
    data = {k: np.asarray([r[k] for r in rows]) for k in FIELDS} if rows else {k: prefix[k][:0] for k in FIELDS}
    if prefix is not None: data = {k: np.concatenate([prefix[k], data[k]]) for k in FIELDS}
    if fit:
        c.check_fit(plan, unit, entry["sequence"])
        gt = np.asarray(entry["ground_truth"], float)[1:limit+1]
        overlaps = tracking_iou(data["candidate_boxes"], gt[:, None]); q = data["quality_choice"].astype(int)
        data.update(candidate_iou=overlaps, reward=overlaps[np.arange(limit), data["selected"].astype(int)],
                    myopic_label=np.array([c.best_label(y,m,plan["design"]["tie_epsilon"]) for y,m in zip(overlaps,data["action_mask"])],int))
    return data


def audit(data,entry,unit,plan,source,pack=None,stochastic=False,fit=False):
    n=len(data["frame_id"]);e63.source_common.prior.audit_trace(data,n,4)
    if not 0<n<=len(entry["frames"])-1 or data["decode_error"].max()>plan["actor_design"]["box_atol"]:raise ValueError("Invalid full search trace")
    history=[np.asarray(entry["initial_box"],float)]
    checked=("choice_features","selected","scores","quality_choice","quality_gain","q0_pred_iou","q0_qualified","veto","immediate_gain","future_gain",
             "x","action_mask","logits","probabilities","value","eligible","action","log_prob","boxes","parents")
    for i in range(n):
        parents=e63.source_common.prior.history_parents(history,plan["actor_design"])
        np.testing.assert_array_equal(data["candidate_parents"][i],parents);np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        for q,p in enumerate(parents):np.testing.assert_array_equal(data["crop"][i,q],crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0])
        row={k:data[k][i].copy() for k in ("features","candidate_boxes","candidate_parents")};row=choose(row,history,unit,plan,source,pack)
        if stochastic:
            action=int(data["action"][i]);lp=c.log_probability(action,row["logits"],row["action_mask"])
            row.update(action=action,selected=action,log_prob=lp,veto=bool(row["quality_choice"] and action!=row["quality_choice"]),
                boxes=row["candidate_boxes"][action].copy(),parents=row["candidate_parents"][action].copy())
        for k in checked:np.testing.assert_array_equal(data[k][i],row[k],err_msg="Categorical decision changed: "+k)
        history.append(data["boxes"][i]);history=history[-plan["actor_design"]["history"]:]
    if fit:
        c.check_fit(plan,unit,entry["sequence"])
        if n!=len(entry["frames"])-1:raise ValueError("Fit episode must terminate naturally, without GT resets")
        y=tracking_iou(data["candidate_boxes"],np.asarray(entry["ground_truth"],float)[1:,None])
        np.testing.assert_array_equal(data["candidate_iou"],y)
        np.testing.assert_array_equal(data["reward"],y[np.arange(n),data["selected"].astype(int)])
        np.testing.assert_array_equal(data["myopic_label"],[c.best_label(v,m,plan["design"]["tie_epsilon"]) for v,m in zip(y,data["action_mask"])])
    if unit["policy"] in c.CONTROLS+c.REUSED:
        ref=c.source_trace(source,unit)
        for k in e68.FIELDS:
            if k!="decode_error":np.testing.assert_array_equal(data[k],ref[k][:n],err_msg="Immutable Stage-73 source differs: "+k)


def source_bridge(root, plan, source, observer, identity, image_reader=checked_image):
    if identity != plan["runtime_identity"]: raise ValueError("Need bound L40 runtime")
    hashes = {}; calls = 0
    old = source["upstream"]["s68"]
    for split in plan["folds"]:
        case = next(x for x in old["plan"]["pairs"] if x["fold"] == split["fold"])
        trace, entry, pack = e68.context(old["plan"], old["old"], case)
        ref = c.c68.read_cache(c.c68.pair_path(old["root"], case), c.c68.pair_meta(old["plan"], case))
        path = Path(root)/"smoke/bridge"/("fold{}.npz".format(split["fold"]))
        meta = dict(plan_fingerprint=plan["fingerprint"], case=case); data = c.read_cache(path, meta)
        with e63.source_experiment.installed(observer, [], case["unit"]) as head:
            if head != case["head_sha256"]: raise ValueError("Bridge head changed")
            if data is None:
                observer.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
                history = e63.history_at(trace, entry["initial_box"], case["frame"], plan)
                row = e67.observe(observer, image_reader(plan["root"], entry["frames"][case["frame"]]), history, case["frame"], "quality", plan, pack)
                data = dict(e67.stack([row]), initial_history=history.copy())
                e68.audit_pair(data, {}, trace, entry, case, plan, pack)
                for k in e67.FIELDS:
                    if k != "decode_error": np.testing.assert_array_equal(data[k], ref[k])
                c.save(path, data, meta)
            e68.audit_pair(data, {}, trace, entry, case, plan, pack)
            for k in e67.FIELDS:
                if k != "decode_error": np.testing.assert_array_equal(data[k], ref[k])
        e63.source_experiment.bind(hashes, path); calls += int(data["head_calls"].sum())
    if calls != plan["source_bridge_search_calls"]: raise ValueError("Bridge budget changed")
    observer.assert_frozen()
    result = c.seal(dict(schema=c.SCHEMA, status="PASS_SOURCE_BRIDGE", plan_fingerprint=plan["fingerprint"], runtime=identity,
                         search_calls=calls, cache_sha256=hashes))
    c.publish_identical(Path(root)/"smoke/bridge.json", result); return result


def bridge_gate(root, plan):
    r = c.sealed(Path(root)/"smoke/bridge.json")
    if r.get("status") != "PASS_SOURCE_BRIDGE" or r.get("runtime") != plan["runtime_identity"] or r.get("plan_fingerprint") != plan["fingerprint"] or r.get("search_calls") != plan["source_bridge_search_calls"]:
        raise ValueError("Need exact source bridge")
    c.VERIFY(r["cache_sha256"]); return r


def initial_step(torch,root,plan,source,job):
    norm=c.fit_normalization(plan,source,job["fold"]);pack=learning.initial(torch,job["seed"],norm,plan["design"])
    before=c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],fold=job["fold"],seed=job["seed"],index=0,
        packs=dict(ppo=pack,myopic=copy.deepcopy(pack))))
    c.publish_identical(c.step_path(root,job,0),before);return before


def check_step(after, before, meta, path, index, design):
    if (after.get("before_fingerprint") != before["fingerprint"] or after.get("metadata") != meta or
        after.get("trace_sha256") != c.file_hash(path) or after.get("index") != index+1 or
        after.get("plan_fingerprint") != before["plan_fingerprint"] or after.get("fold") != before["fold"] or after.get("seed") != before["seed"]):
        raise ValueError("Training checkpoint chain changed")
    if set(after.get("packs", {})) != {"ppo", "myopic"}: raise ValueError("Missing matched controller")
    n = after["stats"]["frames"]; updates = design["epochs"]*int(np.ceil(n/design["minibatch"]))
    if any(after["stats"][arm]["updates"] != updates for arm in ("ppo", "myopic")): raise ValueError("Unmatched optimizer budgets")
    for arm in ("ppo","myopic"):
        for name in ("actor","critic"):
            post=after["stats"][arm][name+"_post_clip_gradient"]
            coefficient=after["stats"][arm][name+"_clip_coefficient"]
            if not np.isfinite([post,coefficient]).all() or not 0<=post<=design["gradient_clip"]+1e-10 or not 0<coefficient<=1:
                raise ValueError("Invalid submitted branch clipping diagnostics")


def calibration(plan,source,job,pack):
    raise ValueError("Four-candidate experiment has no rate-control calibration or extra action")


def train(root, plan, source, observer, identity, smoke=False, image_reader=checked_image):
    if identity != plan["runtime_identity"]: raise ValueError("Need bound L40 runtime")
    bridge_gate(root, plan)
    if not smoke: train_smoke_gate(root, plan)
    import torch
    torch.set_num_threads(1)
    for job in plan["jobs"][:1] if smoke else plan["jobs"]:
        before = initial_step(torch, root, plan, source, job)
        for index, episode in enumerate(job["episodes"][:1] if smoke else job["episodes"]):
            seq = episode["sequence"]; c.check_fit(plan, job, seq); entry = c.entry_for(plan, seq)
            unit = dict(sequence=seq, origin="native", fold=job["fold"], seed=job["seed"], policy="ppo")
            path = c.trace_path(root, job, index); meta = train_meta(plan, job, index, before)
            data = c.read_cache(path, meta)
            with e63.source_experiment.installed(observer, [], unit) as head:
                if head != c.head_for(source, unit): raise ValueError("Training changed original native head")
                if data is None:
                    rng = np.random.default_rng(job["seed"]+100003*job["fold"]+1009*index)
                    data = run_actor(observer, entry, unit, plan, source, before["packs"]["ppo"], rng, image_reader=image_reader, fit=True)
                    audit(data, entry, unit, plan, source, before["packs"]["ppo"], stochastic=True, fit=True); c.save(path, data, meta)
                audit(data, entry, unit, plan, source, before["packs"]["ppo"], stochastic=True, fit=True)
            step_path = c.step_path(root, job, index+1)
            if step_path.exists():
                after = c.sealed(step_path); check_step(after, before, meta, path, index, plan["design"])
            else:
                packs, stats = learning.update(torch, before["packs"], data, plan["design"], job["seed"]+7919*index)
                stats.update(frames=len(data["frame_id"]), eligible=int(data["eligible"].sum()), accepted=int((data["action"]!=0).sum()),
                             mean_reward=float(data["reward"].mean()), myopic_positive=int((data["myopic_label"]!=0).sum()),
                             search_calls=int(data["head_calls"].sum()), head_sha256=head)
                stats.update({"action_q"+str(q):int((data["action"]==q).sum()) for q in range(4)})
                stats.update({"label_q"+str(q):int((data["myopic_label"]==q).sum()) for q in range(4)})
                after = c.seal(dict(schema=c.SCHEMA, plan_fingerprint=plan["fingerprint"], fold=job["fold"], seed=job["seed"],
                    index=index+1, before_fingerprint=before["fingerprint"], metadata=meta, trace_sha256=c.file_hash(path), packs=packs, stats=stats))
                check_step(after, before, meta, path, index, plan["design"]); c.publish_identical(step_path, after)
            before = after
            print("[on-policy fit] {} episode {}/{} {} reward={:.4f}".format(c.job_key(job), index+1, len(job["episodes"]), seq, after["stats"]["mean_reward"]), flush=True)
    observer.assert_frozen(); c.VERIFY(plan["source_file_sha256"]); c.VERIFY(plan["code_sha256"])
    if smoke:
        job = plan["jobs"][0]; path = c.trace_path(root, job, 0)
        r = c.seal(dict(schema=c.SCHEMA, status="PASS_LIVE_TRAINING_SMOKE", plan_fingerprint=plan["fingerprint"],
            search_calls=plan["train_smoke_calls_reused"], trace_sha256=c.file_hash(path), step_sha256=c.file_hash(c.step_path(root, job, 1))))
        c.publish_identical(Path(root)/"smoke/train.json", r); return r
    models = build_models(root, plan, source)
    c.publish_identical(Path(root)/"models/training.json", models); return models


def train_smoke_gate(root, plan):
    r = c.sealed(Path(root)/"smoke/train.json"); job = plan["jobs"][0]
    if (r.get("status") != "PASS_LIVE_TRAINING_SMOKE" or r.get("plan_fingerprint") != plan["fingerprint"] or
        r.get("search_calls") != plan["train_smoke_calls_reused"] or r.get("trace_sha256") != c.file_hash(c.trace_path(root, job, 0)) or
        r.get("step_sha256") != c.file_hash(c.step_path(root, job, 1))): raise ValueError("Need committed live training smoke")
    return r


def build_models(root, plan, source):
    bridge_gate(root, plan); train_smoke_gate(root, plan)
    hashes = {}; controllers = {}; summaries = []; calls = updates = 0
    for job in plan["jobs"]:
        before = c.sealed(c.step_path(root, job, 0))
        if before.get("plan_fingerprint") != plan["fingerprint"] or before.get("index") != 0 or before.get("fold") != job["fold"] or before.get("seed") != job["seed"] or before["packs"]["ppo"] != before["packs"]["myopic"]:
            raise ValueError("Initial matched controller changed")
        expected_norm=c.fit_normalization(plan,source,job["fold"])
        if before["packs"]["ppo"]["norm"]!=expected_norm:raise ValueError("Fit-only normalization changed")
        hashes[str(c.step_path(root, job, 0))] = c.file_hash(c.step_path(root, job, 0))
        for index, episode in enumerate(job["episodes"]):
            seq = episode["sequence"]; c.check_fit(plan, job, seq); entry = c.entry_for(plan, seq)
            unit = dict(sequence=seq, origin="native", fold=job["fold"], seed=job["seed"], policy="ppo")
            path = c.trace_path(root, job, index); meta = train_meta(plan, job, index, before); data = c.read_cache(path, meta)
            if data is None: raise ValueError("Incomplete live training; resume the fixed budget")
            audit(data, entry, unit, plan, source, before["packs"]["ppo"], stochastic=True, fit=True)
            after = c.sealed(c.step_path(root, job, index+1)); check_step(after, before, meta, path, index, plan["design"])
            n = len(data["frame_id"]); n_calls = int(data["head_calls"].sum())
            if (after["stats"]["frames"] != n or after["stats"]["search_calls"] != n_calls or after["stats"]["head_sha256"] != c.head_for(source, unit) or
                after["stats"]["eligible"] != int(data["eligible"].sum()) or after["stats"]["accepted"] != int((data["action"]!=0).sum()) or
                after["stats"]["mean_reward"] != float(data["reward"].mean())): raise ValueError("Training scope differs")
            if any(after["stats"]["action_q"+str(q)]!=int((data["action"]==q).sum()) or
                   after["stats"]["label_q"+str(q)]!=int((data["myopic_label"]==q).sum()) for q in range(4)):
                raise ValueError("Training action/label support differs")
            calls += n_calls; updates += after["stats"]["ppo"]["updates"]
            e63.source_experiment.bind(hashes, path); hashes[str(c.step_path(root, job, index+1))] = c.file_hash(c.step_path(root, job, index+1))
            summaries.append(dict(fold=job["fold"], seed=job["seed"], episode=index, sequence=seq, pass_index=episode["pass_index"],
                **{k:v for k,v in after["stats"].items() if k not in ("ppo", "myopic")},
                **{arm+"_"+k:v for arm in ("ppo", "myopic") for k,v in after["stats"][arm].items()}))
            before = after
        split = next(s for s in plan["folds"] if s["fold"] == job["fold"])
        packs = {arm: {k:v for k,v in before["packs"][arm].items() if k != "optimizer"} for arm in ("ppo", "myopic")}
        controllers[c.job_key(job)] = dict(**packs, fit=split["fit"], held_out=split["held_out"], final_step_fingerprint=before["fingerprint"])
    if calls != plan["training_search_calls"] or updates != plan["optimizer_updates_per_arm"]: raise ValueError("Training budget differs")
    return c.seal(dict(schema=c.SCHEMA, status="COMPLETE_BOUNDED_LIVE_PPO", plan_fingerprint=plan["fingerprint"],
        source_training_fingerprint=source["stage73"]["models"]["fingerprint"], controllers=controllers, training_evidence_sha256=hashes,
        episodes=summaries, environment_steps=calls//4, search_calls=calls, block_calls=7*calls,
        optimizer_updates_per_arm=updates, sgla_training_performed=False, reinforcement_learning_performed=True))


def load_models(root, plan, source):
    models = c.sealed(Path(root)/"models/training.json")
    if models != build_models(root, plan, source): raise ValueError("Trained controller/calibration no longer reconstructs from immutable steps")
    return models


def eval_gate(root, plan, models):
    r = c.sealed(Path(root)/"smoke/evaluate.json")
    if (r.get("status") != "PASS_EVALUATION_SMOKE" or r.get("plan_fingerprint") != plan["fingerprint"] or
        r.get("training_fingerprint") != models["fingerprint"] or r.get("search_calls") != plan["eval_smoke_calls_reused"]): raise ValueError("Need full-policy smoke")
    c.VERIFY(r["cache_sha256"]); return r


def evaluate(root, plan, source, observer, identity, smoke=False, image_reader=checked_image):
    if identity != plan["runtime_identity"]: raise ValueError("Need bound L40 runtime")
    models = load_models(root, plan, source)
    if not smoke: eval_gate(root, plan, models)
    hashes = {}; heads = {}; calls = 0
    units = [u for u in plan["units"] if not smoke or u["sequence"] == plan["entries"][0]["sequence"]]
    for unit in units:
        entry = c.entry_for(plan, unit["sequence"]); limit = len(entry["frames"])-1
        if smoke: limit = min(limit, plan["design"]["smoke_frames"])
        pack = None if unit["policy"] in c.CONTROLS else learned_pack(models, unit, source)
        path = c.eval_path(root, unit, smoke); meta = eval_meta(plan, unit, models, smoke); data = c.read_cache(path, meta)
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != c.head_for(source, unit): raise ValueError("Evaluation native head changed")
            if data is None:
                prefix = None
                if not smoke and unit["sequence"] == plan["entries"][0]["sequence"]:
                    prefix = c.read_cache(c.eval_path(root, unit, True), eval_meta(plan, unit, models, True))
                    if prefix is None: raise ValueError("Missing reusable evaluation smoke")
                data = run_actor(observer, entry, unit, plan, source, pack, limit=limit, prefix=prefix, image_reader=image_reader)
                audit(data, entry, unit, plan, source, pack); c.save(path, data, meta)
            audit(data, entry, unit, plan, source, pack)
        e63.source_experiment.bind(hashes, path); heads[c.key(unit)] = head; calls += int(data["head_calls"].sum())
        print("[PPO evaluated] "+c.key(unit), flush=True)
    observer.assert_frozen(); c.VERIFY(plan["source_file_sha256"]); c.VERIFY(plan["code_sha256"])
    if calls != (plan["eval_smoke_calls_reused"] if smoke else plan["formal_eval_calls"]): raise ValueError("Evaluation budget changed")
    report = c.seal(dict(schema=c.SCHEMA, status="PASS_EVALUATION_SMOKE" if smoke else "COMPLETE_BOUNDED_PPO_EVALUATION",
        plan_fingerprint=plan["fingerprint"], training_fingerprint=models["fingerprint"], runtime=identity, search_calls=calls,
        block_calls=7*calls, trajectories=len(units), cache_sha256=hashes, observed_heads=heads, maximum_source_control_error=0.))
    c.publish_identical(Path(root)/("smoke/evaluate.json" if smoke else "evaluation/run.json"), report); return report


def completed(root, plan, source, models):
    eval_gate(root, plan, models); r = c.sealed(Path(root)/"evaluation/run.json")
    expected = {str(c.eval_path(root, u).with_suffix(s).resolve()) for u in plan["units"] for s in (".npz", ".receipt.json")}
    if (r.get("status") != "COMPLETE_BOUNDED_PPO_EVALUATION" or r.get("plan_fingerprint") != plan["fingerprint"] or
        r.get("training_fingerprint") != models["fingerprint"] or r.get("runtime") != plan["runtime_identity"] or
        r.get("search_calls") != plan["formal_eval_calls"] or r.get("block_calls") != 7*plan["formal_eval_calls"] or
        r.get("trajectories") != len(plan["units"]) or set(r.get("cache_sha256", {})) != expected or
        r.get("observed_heads") != {c.key(u):c.head_for(source,u) for u in plan["units"]} or r.get("maximum_source_control_error") != 0.): raise ValueError("Need all complete same-device policies including exact original PPO controls")
    c.VERIFY(r["cache_sha256"]); return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("prepare", "source_bridge", "train_smoke", "train", "smoke", "evaluate"))
    p.add_argument("--input", required=True, type=nonempty_path); p.add_argument("--output_dir", type=nonempty_path)
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT")); p.add_argument("--gpu", type=int, default=0); p.add_argument("--parameter", default="deit_distilled")
    args = p.parse_args()
    if args.mode == "prepare":
        if not args.output_dir: p.error("prepare requires --output_dir")
        import torch
        source = c.load_source(args.input)
        hardware = c.c66.c64.prior.prior.c.prior.device_binding(source["plan"], args.gpu, torch)
        prepare(args.input, args.output_dir, hardware); return
    root = Path(args.input).resolve(); plan, source = c.load_plan(root)
    if not args.checkpoint: p.error("Need original frozen SGLA V1 checkpoint")
    with writer_lock(root):
        native, identity = e63.startup.checked_runtime(args, plan, root); observer = e63.Capture(native)
        try:
            if args.mode == "source_bridge": source_bridge(root, plan, source, observer, identity)
            elif args.mode in ("train", "train_smoke"): train(root, plan, source, observer, identity, args.mode == "train_smoke")
            else: evaluate(root, plan, source, observer, identity, args.mode == "smoke")
        finally: observer.handle.remove()


if __name__ == "__main__": main()
