"""One clipping-scope treatment; unchanged SGLA, objectives and final training budget."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import stage71_ppo_common as base
import stage72_ppo_common as diagnostic
import stage70_history_common as prior
import stage70_history_experiment as e70
from stage4_common import file_hash, fingerprint

SCHEMA = "stage73_separate_gradient_clip_ppo_v1"
FILES = ("stage73_ppo_common.py", "stage73_ppo_runtime.py", "stage73_ppo_experiment.py",
         "stage73_ppo_summary.py", "test_stage73_ppo.py", "run_stage73_ppo.sh")
DESIGN = dict(base.DESIGN, gradient_clip_mode="separate_actor_critic")
CONTROLS = ("q0", "hann", "quality", "future", "mixed_future")
LEARNED = ("ppo", "myopic", "ppo_rate")
REUSED = ("joint_ppo", "joint_myopic", "joint_rate")
SOURCE_POLICIES = dict(joint_ppo="ppo", joint_myopic="myopic", joint_rate="ppo_rate")
c68, c66, e63 = prior.c68, prior.c66, prior.e63
seal, VERIFY, publish_identical = prior.seal, prior.VERIFY, prior.publish_identical


def sealed(path): return c66.sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve()
    p72 = c66.sealed(root/"manifest.json", diagnostic.SCHEMA)
    s71 = diagnostic.load_source(p72["source_input"])
    report = c66.sealed(root/"analysis/stage72_diagnostic.json", diagnostic.SCHEMA)
    names = ("same_state_video", "same_state_summary", "matched_video", "matched_summary", "candidate_video", "candidate_summary",
             "focus", "focus_frames", "divergences", "training_episode", "training_summary", "optimization_summary", "anchors", "scope_video", "scope")
    occurrences = sum((len(base.entry_for(s71["plan"], u["sequence"])["frames"])-1)*len(diagnostic.score_seeds(u,s71["plan"])) for u in s71["plan"]["units"])
    expected_code = {str(Path(diagnostic.__file__).with_name(n)):file_hash(Path(diagnostic.__file__).with_name(n)) for n in diagnostic.FILES}
    if (p72.get("design") != diagnostic.DESIGN or p72.get("rules") != list(diagnostic.RULES) or
        p72.get("anchors") != diagnostic.resolve_cases(s71) or p72.get("code_sha256") != expected_code or
        p72.get("source_bindings") != s71["bindings"] or p72.get("expected_scored_eval_state_occurrences") != occurrences or
        report.get("status") != "COMPLETE_CACHED_PPO_DECISION_TRAINING_DIAGNOSTIC" or report.get("plan_fingerprint") != p72["fingerprint"] or
        any(p72.get(k) != v or report.get(k) != v for k,v in dict(source_report_fingerprint=s71["report"]["fingerprint"],
            source_plan_fingerprint=s71["plan"]["fingerprint"],source_training_fingerprint=s71["models"]["fingerprint"]).items()) or
        report.get("source_bindings_fingerprint") != fingerprint(s71["bindings"]) or report.get("source_runtime") != s71["plan"]["runtime_identity"] or
        any(report.get(k) != 0 for k in ("new_models","new_image_reads","new_search_calls","new_full_trajectories","new_counterfactual_branches")) or
        any(report.get(k) is not False for k in ("training_performed","reinforcement_learning_performed","method_adoption_gate")) or
        any(report.get(k) is not True for k in ("cpu_controller_rescoring_performed","source_controls_unchanged","exact_recorded_choices")) or
        set(report.get("table_sha256",{})) != {"stage72_"+n+".csv" for n in names}):
        raise ValueError("Need complete immutable Stage-72 diagnostic and Stage-71 training/evaluation chain")
    bindings = dict(s71["bindings"]); bindings.update(expected_code)
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    bindings.update({str(root/n):file_hash(root/n) for n in ("manifest.json","analysis/stage72_diagnostic.json")})
    VERIFY(bindings)
    source = dict(s71["old"], stage71=s71, diagnostic_plan=p72, diagnostic_report=report, diagnostic_root=root, bindings=bindings)
    return source


def entry_for(plan, seq): return next(e for e in plan["entries"] if e["sequence"] == seq)
def fit_pack(source, fold): return prior.fit_pack(source["upstream"], fold)
def head_for(source, unit): return prior.head_for(source["upstream"], dict(unit, seed=0))


def make_plan(root, hardware):
    source = load_source(root); old = source["stage71"]["plan"]
    if set(hardware) != {"device","gpu_uuid"} or hardware["device"] != "NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need observed physical L40 binding")
    identity = copy.deepcopy(old["runtime_identity"]); identity["gpu_uuid"] = hardware["gpu_uuid"]
    changes = e63.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in changes): raise ValueError("Only physical L40 UUID may change")
    if {k:v for k,v in DESIGN.items() if k != "gradient_clip_mode"} != old["design"] or DESIGN["gradient_clip_mode"] != "separate_actor_critic":
        raise ValueError("Only actor/critic clipping scope may change from the sealed Stage-71 design")
    jobs = copy.deepcopy(old["jobs"])
    for split in old["folds"]:
        if set(split["fit"]) & set(split["held_out"]): raise ValueError("Overlapping source groups")
    units = [dict(sequence=entry["sequence"],origin="native",fold=next(s["fold"] for s in old["folds"] if entry["sequence"] in s["held_out"]),policy=p,seed=seed)
             for entry in old["entries"] for p in CONTROLS+REUSED+LEARNED for seed in ([0] if p in CONTROLS else DESIGN["seeds"])]
    frames = {e["sequence"]:len(e["frames"])-1 for e in old["entries"]}
    return seal(dict(schema=SCHEMA,role="consumed_development_separate_gradient_clip_ppo",source_input=str(Path(root).resolve()),
        source_report_fingerprint=source["diagnostic_report"]["fingerprint"],source_ppo_report_fingerprint=source["stage71"]["report"]["fingerprint"],
        source_training_fingerprint=source["stage71"]["models"]["fingerprint"],source_file_sha256=source["bindings"],
        design=copy.deepcopy(DESIGN),jobs=jobs,units=units,controls=list(CONTROLS),reused=list(REUSED),learned=list(LEARNED),
        training_environment_steps=old["training_environment_steps"],training_search_calls=old["training_search_calls"],
        optimizer_updates_per_arm=old["optimizer_updates_per_arm"],formal_eval_calls=sum(4*frames[u["sequence"]] for u in units),
        source_bridge_search_calls=old["source_bridge_search_calls"],train_smoke_calls_reused=old["train_smoke_calls_reused"],
        eval_smoke_calls_reused=4*min(DESIGN["smoke_frames"],frames[old["entries"][0]["sequence"]])*(len(CONTROLS)+(len(REUSED)+len(LEARNED))*len(DESIGN["seeds"])),
        hardware=hardware,runtime=identity["tracker"],runtime_identity=identity,declared_runtime_differences=changes,
        **{k:copy.deepcopy(old[k]) for k in ("controller_design","entries","folds","root","search_size","search_factor","window","actor_design")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})),source


def load_plan(root):
    plan = sealed(Path(root) / "manifest.json"); expected, source = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected: raise ValueError("Frozen Stage-73 source/code/design changed")
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
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-73 cache")
    if path.exists():
        import json
        with np.load(path, allow_pickle=False) as old:
            if json.loads(str(old["metadata_json"].item())) != meta: raise ValueError("Unknown orphan Stage-73 cache")
    from stage4_common import write_cache
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    path = Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    r = sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata") != meta or file_hash(path) != r["sha256"]: raise ValueError("Stage-73 cache changed")
    import json
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Mixed Stage-73 cache")
        return {k: f[k] for k in f.files if k != "metadata_json"}


def source_unit(unit):
    return dict(unit,policy=SOURCE_POLICIES.get(unit["policy"],unit["policy"]))


def source_trace(source,unit):
    old = source["stage71"]; u = source_unit(unit)
    data = base.read_cache(base.eval_path(old["root"],u),diagnostic.e71.eval_meta(old["plan"],u,old["models"]))
    if data is None: raise ValueError("Missing immutable Stage-71 comparison trajectory")
    return data
