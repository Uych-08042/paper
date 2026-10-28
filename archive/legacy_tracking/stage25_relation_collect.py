"""Fit-only, fixed-call-time H20 layer branches and causal relational states."""

import argparse
from pathlib import Path
import random

import numpy as np

from stage4_common import file_hash, load_manifest, tracking_iou, write_cache, write_json
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_environment import executed_crop_roi
from stage18_freeze_representation import verify_representation_bundle
from stage18_representation_train import CapturedRuntime, load_sources
from stage21_counterfactual_runtime import CoupledEnvironment, assert_snapshot_equal
from stage23_branch_ppo import check_runtime, model_digest, single_writer
from stage23_development import build_model, frozen_split, read_sealed, sealed
from stage25_relation_common import (ANCHORS, CODE_FILES, COLLECTION_SEED, GAMMA, HORIZON, REPEATS,
    SCHEMA, anchor_indices, cache_path, continuation, keyed_seed, load_cache, relation_features, schedule)


def simulate(env, before, after, actual, images, gt, valid, actions):
    """All first layers spend one call; later actions/call slots are matched.

    Labels are scored after execution only. Branches never write into live
    history. Repeat zero's chosen first layer must replay the live trajectory.
    """
    if (actions.shape != (REPEATS, HORIZON) or len(images) != HORIZON
            or np.shape(gt) != (HORIZON, 4) or np.shape(valid) != (HORIZON,)
            or not np.isin(actions, range(7)).all() or np.any(actions[:, 0] == 0)
            or not np.array_equal(actions[0] > 0, actions[1] > 0)):
        raise ValueError("Invalid equal-cost branch schedule")
    torch, actor = env.rt.torch, env.actor_view
    transient = actor.uniform, actor.forced, actor.last
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy, python = np.random.get_state(), random.getstate()
    rewards, costs = np.zeros((REPEATS, 6, HORIZON)), np.zeros((REPEATS, 6), np.int64)
    replay_boxes = np.zeros((HORIZON, 4))
    error = 0.
    try:
        for repeat in range(REPEATS):
            for action in range(1, 7):
                env.restore(before)
                for step, image in enumerate(images):
                    output = env.step(image, action=action if step == 0 else int(actions[repeat, step]))
                    if repeat == 0 and action == actual["action"]:
                        replay_boxes[step] = output["box"]
                    costs[repeat, action - 1] += int(output["action"] > 0)
                    if step == 0 and action == actual["action"]:
                        error = max(error, float(np.abs(np.asarray(output["box"]) - actual["box"]).max()))
                        assert_snapshot_equal(env.snapshot(), after)
                    overlap = float(tracking_iou(output["box"], gt[step], bool(valid[step])))
                    if valid[step] and (not np.isfinite(overlap) or not 0 <= overlap <= 1):
                        raise RuntimeError("Invalid branch label")
                    rewards[repeat, action - 1, step] = overlap if valid[step] else 0.
        if error > 1e-6 or not np.all(costs == np.count_nonzero(actions[0])):
            raise RuntimeError("Replay or equal-cost invariant failed")
    finally:
        env.restore(after)
        actor.uniform, actor.forced, actor.last = transient
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy)
        random.setstate(python)
        assert_snapshot_equal(env.snapshot(), after)
    return rewards, costs, error, replay_boxes


def collect_video(rt, modules, source):
    wrapped = CapturedRuntime(rt)
    env = CoupledEnvironment(wrapped, modules)
    env.reset(rt.read(source["paths"][0]), source["gt"][0])
    name, steps = source["sequence"], len(source["paths"]) - 1
    actions = schedule(steps, keyed_seed(COLLECTION_SEED, name, "behavior"))
    anchors = anchor_indices(actions)
    rows, live_boxes = [], []
    for step, action in enumerate(actions):
        image = rt.read(source["paths"][step + 1])
        before = env.snapshot()
        actual = env.step(image, action=int(action))
        after = env.snapshot()
        live_boxes.append(actual["box"])
        if step not in anchors:
            continue
        roi = executed_crop_roi(before["box"], before["box"], wrapped.factor, rt.tracker.params.search_size)
        obs = {k: v.detach().cpu().numpy()[0].copy() for k, v in actual["observation"].items()}
        raw = np.r_[obs["grid"].ravel(), obs["template"], obs["memory"].ravel(), obs["history"], roi].astype(np.float32)
        relation = relation_features(obs["grid"], obs["template"], roi)
        future = slice(step + 1, step + 1 + HORIZON)
        rewards, costs, error, replay_boxes = simulate(env, before, after, actual,
            [rt.read(path) for path in source["paths"][future]], source["gt"][future], source["valid"][future],
            continuation(actions, step, keyed_seed(COLLECTION_SEED, name, int(step), "continuation")))
        rows.append(dict(raw=raw, relation=relation, rewards=rewards, valid=source["valid"][future].copy(),
                         branch_cost=costs, replay_error=error, replay_boxes=replay_boxes))
        print("[branch] {} {}/{} at tracked frame {}".format(name, len(rows), ANCHORS, step + 1), flush=True)
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays.update(anchor=anchors, live_actions=actions, live_boxes=np.asarray(live_boxes),
                  live_valid=source["valid"][1:].copy())
    arrays["replay_error"] = np.array([np.abs(arrays["replay_boxes"][i] -
        arrays["live_boxes"][anchor:anchor + HORIZON]).max() for i, anchor in enumerate(anchors)])
    return arrays


def lineage(bundle_path, manifest_path, audit_dir):
    _, bundle, repdir = verify_representation_bundle(bundle_path)
    manifest = load_manifest(manifest_path)
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Frozen source manifest differs")
    fit, tune = frozen_split(bundle, manifest)
    audit_dir = Path(audit_dir).expanduser().resolve()
    audit = read_sealed(audit_dir / "stage24_action_audit.json")
    audit_contract = read_sealed(audit_dir / "stage24_audit_contract.json")
    if (audit.get("status") != "COMPLETE_DESCRIPTIVE_ACTION_AUDIT"
            or audit.get("audit_contract_fingerprint") != audit_contract["fingerprint"]):
        raise ValueError("Require a completed Stage-24 audit without output inconsistency")
    source = Path(audit_contract["source_directory"])
    original = read_sealed(source / "stage23b_contract.json")
    if (original["fingerprint"] != audit_contract["source_contract_fingerprint"]
            or original["bundle_fingerprint"] != bundle["fingerprint"]
            or original["fit"] != fit or original["tune"] != tune):
        raise ValueError("Stage-24/23/18 frozen identities differ")
    if (audit_contract.get("source_status") != "NO_CREDIT_EFFECT_ESTABLISHED"
            or not audit["findings"]["h1_all_seeds_static_l9_equivalent"]):
        raise ValueError("This new protocol requires the reported negative/collapsed source experiment")
    return bundle, repdir, manifest, fit, tune, audit["fingerprint"], source.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    for field in ("bundle", "manifest", "stage24_dir", "output_dir"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--sequence", help="One fit video for integrity only; same root can then resume all fit videos")
    args = parser.parse_args()
    if any(not (getattr(args, name) or "").strip() for name in
           ("bundle", "manifest", "stage24_dir", "output_dir", "checkpoint")):
        parser.error("Explicit nonempty frozen input and output paths required")
    bundle, repdir, manifest, fit, tune, audit_id, source_dir = lineage(args.bundle, args.manifest, args.stage24_dir)
    if args.sequence is not None and args.sequence not in fit:
        parser.error("--sequence must belong to frozen fit, not tune or validation")
    output = Path(args.output_dir).expanduser().resolve()
    for original in (repdir, source_dir, Path(args.stage24_dir).resolve()):
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            parser.error("Use a separate new cache directory")
    with single_writer(output):
        rt = Runtime(args.parameter, args.gpu, args.checkpoint)
        check_runtime(rt.identity, bundle["runtime"])
        dependencies = ("stage18_policy_runtime.py", "stage21_counterfactual_runtime.py", "stage23_development.py",
                        "stage23_branch_ppo.py", "stage18_representation_train.py", "stage4_common.py",
                        "stage18_environment.py", "stage18_temporal_model.py", "stage4_runtime.py",
                        "stage8_immediate_gate.py", "stage18_freeze_representation.py")
        contract = sealed(dict(schema=SCHEMA, role="fit_only_relational_state_feasibility_not_PPO",
            bundle_fingerprint=bundle["fingerprint"], stage24_audit_fingerprint=audit_id,
            manifest_fingerprint=manifest["fingerprint"], fit=fit, excluded_tune=tune,
            clip_steps={e["sequence"]: min(128, e["stop"] - e["start"] - 1)
                        for e in manifest["sequences"] if e["sequence"] in fit},
            excluded_validation=[e["sequence"] for e in manifest["sequences"] if e["split"] == "val"],
            fit_clip_maximum=128, horizon=HORIZON, repeats=REPEATS, anchors=ANCHORS, gamma=GAMMA,
            seed=COLLECTION_SEED, embed_dim=rt.identity["embed_dim"], runtime=rt.identity,
            continuation="shared canonical P30 calls; behavior replay and independent uniform later layers",
            target="visibility-normalized H20 discounted IoU; six equal-cost first layers",
            state="compact grid/template/memory/history plus previous predicted ROI; relations are derived from same inputs",
            representation_caveat="Frozen representation previously saw fit videos; OOF refers to the new probe only",
            code_sha256={n: file_hash(Path(__file__).parent / n) for n in CODE_FILES + dependencies}))
        path = output / "stage25_contract.json"
        if path.exists():
            if read_sealed(path) != contract:
                raise ValueError("Frozen collection contract changed; use a new output directory")
        elif any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
            raise ValueError("Unidentified outputs; use an empty new directory")
        else:
            write_json(path, contract)
        modules = build_model(rt, bundle, repdir, COLLECTION_SEED)
        modules.requires_grad_(False)
        initial = model_digest(modules)
        names = [args.sequence] if args.sequence else fit
        print("Fit-only: {} expected videos; tune/validation images and labels are not loaded.".format(len(fit)), flush=True)
        for name in names:
            destination = cache_path(output, name)
            if destination.exists():
                load_cache(destination, contract, name)
                print("[resume] " + name, flush=True)
                continue
            source = load_sources(manifest, [name], 128)[0]
            with rt.torch.no_grad():
                arrays = collect_video(rt, modules, source)
            from stage25_relation_common import validate_cache
            validate_cache(arrays, contract, name)
            if model_digest(modules) != initial:
                raise RuntimeError("Frozen representation changed during collection")
            write_cache(destination, arrays, dict(contract_fingerprint=contract["fingerprint"], sequence=name))
            print("[done] {} -> {}".format(name, destination), flush=True)
        print('Next: python tracking/stage25_relation_screen.py --input "{}" --audit_only'.format(output))


if __name__ == "__main__":
    main()
