"""Bounded on-policy search-gate PPO; frozen SGLA and fit-local Q proposals."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import stage70_history_common as prior
import stage70_history_experiment as e70
from stage4_common import file_hash, fingerprint

SCHEMA = "stage71_bounded_on_policy_search_ppo_v1"
FILES = ("stage71_ppo_common.py", "stage71_ppo_runtime.py", "stage71_ppo_experiment.py",
         "stage71_ppo_summary.py", "test_stage71_ppo.py", "run_stage71_ppo.sh")
DESIGN = dict(seeds=[2171, 2172], passes=4, hidden=32, epochs=4, minibatch=256,
              gamma=.99, gae_lambda=.95, clip=.2, learning_rate=3e-4, adam_epsilon=1e-5,
              value_coefficient=.5, entropy_coefficient=.01, gradient_clip=1.,
              feature_clip=10., tie_epsilon=1e-6, smoke_frames=32,
              bootstrap_seed=2171, bootstrap_samples=20000, practical_gain=.5, reliable_floor=-1.)
CONTROLS = ("q0", "hann", "quality", "future", "mixed_future")
LEARNED = ("ppo", "myopic", "ppo_rate")
c68, c66, e63 = prior.c68, prior.c66, prior.e63
seal, VERIFY, publish_identical = prior.seal, prior.VERIFY, prior.publish_identical


def sealed(path): return c66.sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve()
    plan, upstream = prior.load_plan(root)
    models = e70.load_models(plan, root, upstream)
    run = e70.completed(plan, root, upstream, models)
    report = prior.sealed(root / "analysis/stage70_decision.json")
    names = ("policy_summary", "paired", "factor_summary", "calibration", "fit_summary", "pair_support",
             "history_support", "paired_probe", "focus", "scope", "video_influence", "fit_pairs", "pair_scope", "video_summary", "events")
    if (report.get("status") not in ("REVIEW_FIT_HISTORY_SOURCE_PILOT", "PROMISING_FIT_HISTORY_SOURCE_PILOT") or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("training_fingerprint") != models["fingerprint"] or
        report.get("run_fingerprint") != run["fingerprint"] or report.get("runtime") != plan["runtime_identity"] or
        report.get("sgla_training_performed") is not False or report.get("reinforcement_learning_performed") is not False or
        report.get("maximum_source_control_error") != 0 or report.get("source_quality_unchanged") is not True or
        set(report.get("table_sha256", {})) != {"stage70_" + n + ".csv" for n in names}):
        raise ValueError("Need complete Stage-70 result and frozen source chain")
    bindings = dict(plan["source_file_sha256"])
    bindings.update(plan["code_sha256"])
    bindings.update(run["cache_sha256"])
    bindings.update({str(root / "analysis" / n): h for n, h in report["table_sha256"].items()})
    for name in ("manifest.json", "models/training.json", "evaluation/run.json", "analysis/stage70_decision.json",
                 "collection/pair_plan.json", "pairs/run.json", "history/run.json", "smoke/stage70_bridge.json"):
        bindings[str(root / name)] = file_hash(root / name)
    VERIFY(bindings)
    return dict(root=root, plan=plan, upstream=upstream, models=models, run=run, report=report, bindings=bindings)


def entry_for(plan, seq): return next(e for e in plan["entries"] if e["sequence"] == seq)
def fit_pack(source, fold): return prior.fit_pack(source["upstream"], fold)
def head_for(source, unit): return prior.head_for(source["upstream"], dict(unit, seed=0))


def make_plan(root, hardware):
    source = load_source(root); old = source["plan"]
    if set(hardware) != {"device", "gpu_uuid"} or hardware["device"] != "NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need observed physical L40 binding")
    identity = copy.deepcopy(old["runtime_identity"]); identity["gpu_uuid"] = hardware["gpu_uuid"]
    changes = e63.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in changes): raise ValueError("Only physical L40 UUID may change")
    if (len(set(DESIGN["seeds"])) != len(DESIGN["seeds"]) or not DESIGN["seeds"] or
        min(DESIGN[k] for k in ("passes", "epochs", "minibatch", "hidden")) < 1): raise ValueError("Invalid fixed budget")
    jobs = []
    for split in old["folds"]:
        if set(split["fit"]) & set(split["held_out"]): raise ValueError("Overlapping source groups")
        rng = np.random.default_rng(2171 + split["fold"])
        episodes = [dict(sequence=str(seq),pass_index=p) for p in range(DESIGN["passes"])
                    for seq in rng.permutation(split["fit"])]
        for seed in DESIGN["seeds"]: jobs.append(dict(fold=split["fold"], seed=seed, episodes=episodes))
    units = [dict(sequence=e["sequence"], origin="native", fold=next(s["fold"] for s in old["folds"] if e["sequence"] in s["held_out"]),
                  policy=p, seed=seed) for e in old["entries"] for p in CONTROLS + LEARNED
             for seed in ([0] if p in CONTROLS else DESIGN["seeds"])]
    frames = {e["sequence"]: len(e["frames"]) - 1 for e in old["entries"]}
    train_steps = sum(frames[x["sequence"]] for job in jobs for x in job["episodes"])
    updates = sum(DESIGN["epochs"] * int(np.ceil(frames[x["sequence"]] / DESIGN["minibatch"])) for job in jobs for x in job["episodes"])
    return seal(dict(schema=SCHEMA, role="consumed_development_bounded_on_policy_search_gate", source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"], source_training_fingerprint=source["models"]["fingerprint"],
        source_file_sha256=source["bindings"], design=copy.deepcopy(DESIGN), jobs=jobs, units=units,
        controls=list(CONTROLS), learned=list(LEARNED), training_environment_steps=train_steps,
        training_search_calls=4*train_steps, optimizer_updates_per_arm=updates,
        formal_eval_calls=sum(4*frames[u["sequence"]] for u in units), source_bridge_search_calls=4*len(old["folds"]),
        train_smoke_calls_reused=4*frames[jobs[0]["episodes"][0]["sequence"]],
        eval_smoke_calls_reused=4*min(DESIGN["smoke_frames"], frames[old["entries"][0]["sequence"]])*(len(CONTROLS)+len(LEARNED)*len(DESIGN["seeds"])),
        hardware=hardware, runtime=identity["tracker"], runtime_identity=identity, declared_runtime_differences=changes,
        controller_design=copy.deepcopy(old["controller_design"]),
        **{k: copy.deepcopy(old[k]) for k in ("entries", "folds", "root", "search_size", "search_factor", "window", "actor_design")},
        code_sha256={str(Path(__file__).with_name(n)): file_hash(Path(__file__).with_name(n)) for n in FILES})), source


def load_plan(root):
    plan = sealed(Path(root) / "manifest.json"); expected, source = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected: raise ValueError("Frozen Stage-71 source/code/design changed")
    return plan, source


def check_fit(plan, job, seq):
    split = next(s for s in plan["folds"] if s["fold"] == job["fold"])
    if seq not in split["fit"] or seq in split["held_out"]: raise ValueError("Training crosses held-out source group")


def normalized(f, q, norm, design):
    # This API deliberately has no GT, frame index, source name or future input.
    x = np.r_[np.zeros(64), np.asarray(f, float)[0]] if q == 0 else c68.pair_feature(f, q)
    center, scale = np.asarray(norm["center"]), np.asarray(norm["scale"])
    if center.shape != (128,) or scale.shape != (128,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid fit-only feature normalization")
    return np.clip((x-center)/scale, -design["feature_clip"], design["feature_clip"])


def forward(x, state, branch):
    x = np.asarray(x, float)
    h = np.tanh(x @ np.asarray(state[branch+".0.weight"]).T + np.asarray(state[branch+".0.bias"]))
    return np.squeeze(h @ np.asarray(state[branch+".2.weight"]).T + np.asarray(state[branch+".2.bias"]), axis=-1)


def probability(logit, eligible):
    if not np.isfinite(logit): raise ValueError("Nonfinite gate logit")
    return float(np.exp(-np.logaddexp(0., -float(logit)))) if eligible else 0.


def gate(f, q, boxes, pack, design):
    eligible = bool(q and not np.array_equal(boxes[q], boxes[0]))
    x = normalized(f, q, pack["norm"], design)
    logit = float(forward(x, pack["state"], "actor")); value = float(forward(x, pack["state"], "critic"))
    return dict(x=x, eligible=eligible, logit=logit, value=value, probability=probability(logit, eligible))


def log_probability(action, logit, eligible):
    if action not in (0, 1) or (action and not eligible): raise ValueError("Illegal binary action")
    return float(-np.logaddexp(0., -logit if action else logit)) if eligible else 0.


def gae(rewards, values, gamma, lam):
    """Full video episodes terminate naturally; no truncated/reset bootstrap."""
    r, v = np.asarray(rewards, float), np.asarray(values, float)
    if r.ndim != 1 or not len(r) or v.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(v).all(): raise ValueError("Invalid episode")
    a = np.empty_like(r); trace = 0.; next_value = 0.
    for i in range(len(r)-1, -1, -1):
        trace = r[i] + gamma*next_value - v[i] + gamma*lam*trace
        a[i] = trace; next_value = v[i]
    return a, a+v


def key(unit): return c66.key(unit)
def job_key(job): return "f{}__s{}".format(job["fold"], job["seed"])
def trace_path(root, job, index): return Path(root)/"training"/job_key(job)/("episode{:03d}.npz".format(index))
def step_path(root, job, index): return Path(root)/"training"/job_key(job)/("step{:03d}.json".format(index))
def eval_path(root, unit, smoke=False): return Path(root)/("smoke/evaluation" if smoke else "evaluation/traces")/(key(unit)+".npz")
def save(path, data, meta):
    # Identical immutable NPZ+receipt format, in a new namespace/schema.
    path = Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-71 cache")
    if path.exists():
        import json
        with np.load(path, allow_pickle=False) as old:
            if json.loads(str(old["metadata_json"].item())) != meta: raise ValueError("Unknown orphan Stage-71 cache")
    from stage4_common import write_cache
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    path = Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    r = sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata") != meta or file_hash(path) != r["sha256"]: raise ValueError("Stage-71 cache changed")
    import json
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Mixed Stage-71 cache")
        return {k: f[k] for k in f.files if k != "metadata_json"}
