"""Replay four choices at one shared state, then follow the unchanged source policy."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage63_choice_common as c
from stage27_uav_common import checked_image, crop_geometry, nonempty_path
from stage27_uav_collect import writer_lock
from stage28_response_common import dense_boxes
from stage47_search_collect import Capture
from stage48_stability_experiment import load_torch
import stage60_state_runtime as actor

source_experiment = c.prior.source_experiment
source_common = c.prior.c.prior.original
startup = c.prior.c.prior.startup
FIELDS = ("frame_id", "boxes", "parents", "history_latest", "candidate_parents", "candidate_boxes", "features",
          "scores", "selected", "crop", "crop_sha256", "image_hw", "layer", "block_calls", "head_calls",
          "selector_calls", "decode_error", "raw", "size", "offset")


def prepare(source, root, hardware):
    source, root = Path(source).expanduser().resolve(), Path(root).expanduser().resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Use a separate new Stage-63 directory")
    plan, _ = c.make_plan(source, hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-63 directory")
        c.publish_identical(root/"manifest.json", plan)
    print("Stage-63 frozen: {} cases / {} short branches / {} searches; smoke {} calls included; no training".format(
        len(plan["cases"]), plan["counterfactual_branches"], plan["formal_search_calls"], plan["smoke_calls_reused"]))
    for case in plan["cases"]:
        print("[anchor] {} {} {} s{} {} f{} q{}".format(case["case_id"], case["sequence"], case["origin"],
              case["seed"], case["policy"], case["frame"], case["recorded_choice"]), flush=True)


def probe(observer, image, history, frame, policy, plan):
    """No GT, source outcomes or future images are accepted by this actor."""
    first = actor.observe(observer, image, history[-1], frame, plan)
    parents = (source_common.recrop_parents(history[-1], first["boxes"]) if policy == "recrop" else
               source_common.prior.history_parents(history, plan["design"]))
    observations = [first]+[actor.observe(observer, image, parent, frame, plan) for parent in parents[1:]]
    features = np.asarray([actor.feature_vector(row, history, plan) for row in observations])
    chosen, scores = source_common.prior.choose("response", features, {}, plan["design"])
    output = np.asarray(observations[chosen]["boxes"], float).copy()
    observer.rt.tracker.state = output.tolist()
    return dict(frame_id=frame, boxes=output, parents=parents[chosen].copy(), history_latest=history[-1].copy(),
        candidate_parents=parents, candidate_boxes=np.asarray([row["boxes"] for row in observations]), features=features,
        scores=np.asarray(scores)[:4], selected=chosen, crop=np.asarray([row["crop"] for row in observations]),
        crop_sha256=np.asarray([row["crop_sha256"] for row in observations]), image_hw=np.asarray(image.shape[:2]),
        decode_error=max(row["decode_error"] for row in observations),
        **{key:np.asarray([row[key] for row in observations]) for key in
           ("layer", "block_calls", "head_calls", "selector_calls", "raw", "size", "offset")})


def stacked(rows):
    return {key:np.asarray([row[key] for row in rows]) for key in FIELDS}


def collect_shared(observer, frames, initial_box, history, frame, policy, plan, image_reader=checked_image):
    observer.initialize(image_reader(plan["root"], frames[0]), initial_box)
    row = probe(observer, image_reader(plan["root"], frames[frame]), history, frame, policy, plan)
    return dict(stacked([row]), initial_history=np.asarray(history, float).copy())


def collect_tail(observer, frames, initial_box, history, shared, forced, frame, horizon, policy, plan, image_reader=checked_image):
    observer.initialize(image_reader(plan["root"], frames[0]), initial_box)
    own = [np.asarray(v, float).copy() for v in history]
    own.append(shared["candidate_boxes"][0, forced].copy())
    own = own[-plan["design"]["history"]:]
    observer.rt.tracker.state = own[-1].tolist()
    rows = []
    for f in range(frame+1, frame+horizon):
        row = probe(observer, image_reader(plan["root"], frames[f]), own, f, policy, plan)
        rows.append(row)
        own.append(row["boxes"].copy())
        own = own[-plan["design"]["history"]:]
    return stacked(rows)


def history_at(trace, initial_box, frame, plan):
    previous = np.vstack([np.asarray(initial_box, float), trace["boxes"][:frame-1]])
    return previous[-plan["actor_design"]["history"]:].copy()


def audit_segment(data, incoming, first, count, policy, plan):
    ap = c.actor_plan(plan)
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, first+count))
    if data["candidate_boxes"].shape != (count, 4, 4) or data["features"].shape != (count, 4, len(source_common.prior.FEATURES)):
        raise ValueError("Incomplete four-choice segment")
    source_common.prior.audit_calls(data, (count, 4))
    for name in ("boxes", "candidate_boxes", "candidate_parents", "features", "raw", "size", "offset", "decode_error"):
        if not np.isfinite(data[name]).all():
            raise ValueError("Nonfinite Stage-63 observations: "+name)
    if np.any(data["candidate_boxes"][:, :, 2:] <= 0):
        raise ValueError("Invalid Stage-63 output size")
    history = [np.asarray(v, float).copy() for v in incoming]
    for i in range(count):
        parents = (source_common.recrop_parents(history[-1], data["candidate_boxes"][i, 0]) if policy == "recrop" else
                   source_common.prior.history_parents(history, ap["design"]))
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        errors = []
        for q in range(4):
            crop, usable, factor = crop_geometry(parents[q], data["image_hw"][i], ap["search_factor"], ap["search_size"])
            np.testing.assert_array_equal(data["crop"][i, q], crop)
            row = dict(parent=parents[q], boxes=data["candidate_boxes"][i, q], usable=usable, crop=crop,
                       image_hw=data["image_hw"][i], **{k:data[k][i, q] for k in ("raw", "size", "offset")})
            np.testing.assert_array_equal(data["features"][i, q], actor.feature_vector(row, history, ap))
            cell = int(np.argmax(row["raw"]*np.asarray(ap["window"], np.float32)))
            dense = dense_boxes(row["size"], row["offset"], parents[q], factor, ap["search_size"], data["image_hw"][i])
            errors.append(float(np.max(np.abs(dense[cell]-row["boxes"]))))
        np.testing.assert_allclose(data["decode_error"][i], max(errors), rtol=0, atol=1e-12)
        if max(errors) > ap["design"]["box_atol"]:
            raise ValueError("Public dense decode failed")
        chosen, scores = source_common.prior.choose("response", data["features"][i], {}, ap["design"])
        if int(data["selected"][i]) != chosen:
            raise ValueError("Continuation selection is not the unchanged source rule")
        np.testing.assert_array_equal(data["scores"][i], scores)
        np.testing.assert_array_equal(data["boxes"][i], data["candidate_boxes"][i, chosen])
        np.testing.assert_array_equal(data["parents"][i], parents[chosen])
        history.append(data["boxes"][i].copy())
        history = history[-ap["design"]["history"]:]


def exact_source(data, trace, first, label):
    count = len(data["frame_id"])
    for name in FIELDS:
        if name not in trace or name == "decode_error":
            continue
        a, b = data[name], trace[name][first-1:first-1+count]
        if not np.array_equal(a, b):
            error = float(np.max(np.abs(a-b))) if np.issubdtype(a.dtype, np.number) else "non-numeric"
            raise ValueError("Exact source replay failed ({} {}), max error {}; do not loosen tolerances".format(label, name, error))


def audit_shared(data, case, entry, trace, plan):
    history = history_at(trace, entry["initial_box"], case["frame"], plan)
    np.testing.assert_array_equal(data["initial_history"], history)
    audit_segment(data, history, case["frame"], 1, case["policy"], plan)
    exact_source(data, trace, case["frame"], case["case_id"]+" shared")
    if int(data["selected"][0]) != case["recorded_choice"]:
        raise ValueError("Anchor selected choice differs from frozen source")


def audit_tail(data, shared, case, entry, trace, plan, forced):
    history = history_at(trace, entry["initial_box"], case["frame"], plan).tolist()
    history.append(shared["candidate_boxes"][0, forced].tolist())
    history = history[-plan["actor_design"]["history"]:]
    audit_segment(data, history, case["frame"]+1, plan["design"]["horizon"]-1, case["policy"], plan)
    if forced == case["recorded_choice"]:
        exact_source(data, trace, case["frame"]+1, case["case_id"]+" recorded branch")


def paths(root, case, forced=None):
    stem = case["case_id"]+("__shared" if forced is None else "__q"+str(forced))
    return Path(root)/"collection/traces"/(stem+".npz")


def run_record(plan):
    return c.seal(dict(schema=c.SCHEMA, role=plan["role"], plan_fingerprint=plan["fingerprint"], runtime=plan["runtime_identity"]))


def smoke_gate(plan, root):
    value = c.read_sealed(Path(root)/"smoke/stage63_smoke.json")
    if (value.get("status") != "PASS_ONE_CHOICE_REPLAY_SMOKE" or value.get("plan_fingerprint") != plan["fingerprint"] or
        value.get("run_fingerprint") != run_record(plan)["fingerprint"] or
        value.get("search_calls") != plan["smoke_calls_reused"] or value.get("maximum_source_box_error") != 0):
        raise ValueError("Need the exact-runtime Stage-63 smoke")
    c.prior.c.prior.verify_files(value["cache_sha256"])
    return value


def load_records(inputs, torch):
    result = []
    for record, _ in inputs["source"]["records"]:
        if record["arm"] != "endpoint_control":
            continue
        payload = load_torch(torch, Path(record["checkpoint"]))
        receipt = c.read_sealed(Path(record["checkpoint"]).with_suffix(".receipt.json"), "stage58_projected_endpoint_gradient_v1")
        if payload["metadata"] != receipt["metadata"] or payload["step"] != receipt["step"]:
            raise ValueError("Frozen E payload differs from its receipt")
        result.append((record, payload["head"]))
    return result


def execute(root, plan, inputs, observer, records, identity, smoke=False, image_reader=checked_image):
    root = Path(root)
    if identity != plan["runtime_identity"]:
        raise ValueError("Need the planned L40 runtime before image replay")
    if not smoke:
        smoke_gate(plan, root)
    c.publish_identical(root/"run.json", run_record(plan))
    bindings, calls, branches, heads = {}, 0, 0, {}
    chosen_cases = plan["cases"][:1] if smoke else plan["cases"]
    for case in chosen_cases:
        unit = case["unit"]
        trace, entry, _, _ = c.prior.read_trace(inputs["source"], unit)
        shared_path = paths(root, case)
        meta = c.metadata(plan, case, "shared_four_probes")
        with source_experiment.installed(observer, records, unit) as head:
            if head != case["head_sha256"]:
                raise ValueError("Installed head differs from source")
            heads[case["case_id"]] = head
            shared = c.read_cache(shared_path, meta)
            if shared is None:
                shared = collect_shared(observer, entry["frames"], entry["initial_box"],
                    history_at(trace, entry["initial_box"], case["frame"], plan), case["frame"], case["policy"], c.actor_plan(plan), image_reader)
                audit_shared(shared, case, entry, trace, plan)
                c.save(shared_path, shared, meta)
            audit_shared(shared, case, entry, trace, plan)
            source_experiment.bind(bindings, shared_path)
            calls += int(shared["head_calls"].sum())
            choices = [case["recorded_choice"]] if smoke else range(4)
            for q in choices:
                path = paths(root, case, q)
                meta = c.metadata(plan, case, "one_forced_choice_then_original_policy", q)
                data = c.read_cache(path, meta)
                if data is None:
                    data = collect_tail(observer, entry["frames"], entry["initial_box"], shared["initial_history"], shared,
                        q, case["frame"], plan["design"]["horizon"], case["policy"], c.actor_plan(plan), image_reader)
                    audit_tail(data, shared, case, entry, trace, plan, q)
                    c.save(path, data, meta)
                audit_tail(data, shared, case, entry, trace, plan, q)
                source_experiment.bind(bindings, path)
                calls += int(data["head_calls"].sum())
                branches += 1
        print("[matched choice audited] {} {} f{} {} branches".format(case["case_id"], case["sequence"], case["frame"], len(choices)), flush=True)
    observer.assert_frozen()
    c.prior.c.prior.verify_files(plan["source_file_sha256"])
    c.prior.c.prior.verify_files(plan["code_sha256"])
    expected = plan["smoke_calls_reused"] if smoke else plan["formal_search_calls"]
    if calls != expected or branches != (1 if smoke else plan["counterfactual_branches"]):
        raise ValueError("Stage-63 charged call/branch budget differs")
    report = c.seal(dict(schema=c.SCHEMA, status="PASS_ONE_CHOICE_REPLAY_SMOKE" if smoke else "COMPLETE_ONE_CHOICE_CONTINUATIONS",
        plan_fingerprint=plan["fingerprint"], run_fingerprint=run_record(plan)["fingerprint"], runtime=identity,
        search_calls=calls, block_calls=7*calls, branches=branches, maximum_source_box_error=0.,
        observed_head_sha256=heads, cache_sha256=bindings, training_performed=False))
    c.publish_identical(root/("smoke/stage63_smoke.json" if smoke else "collection/run.json"), report)
    print("Stage-63: "+report["status"]+"; source choices exactly reproduced")
    return report


def completed(plan, root):
    root = Path(root)
    smoke_gate(plan, root)
    if c.read_sealed(root/"run.json") != run_record(plan):
        raise ValueError("Stage-63 runtime record changed")
    value = c.read_sealed(root/"collection/run.json")
    expected = {str(paths(root, case, q).with_suffix(suffix).resolve()) for case in plan["cases"]
                for q in (None, 0, 1, 2, 3) for suffix in (".npz", ".receipt.json")}
    if (value.get("status") != "COMPLETE_ONE_CHOICE_CONTINUATIONS" or value.get("plan_fingerprint") != plan["fingerprint"] or
        value.get("run_fingerprint") != run_record(plan)["fingerprint"] or value.get("runtime") != plan["runtime_identity"] or
        value.get("search_calls") != plan["formal_search_calls"] or value.get("block_calls") != 7*plan["formal_search_calls"] or
        value.get("branches") != plan["counterfactual_branches"] or value.get("maximum_source_box_error") != 0 or
        value.get("training_performed") is not False or set(value.get("cache_sha256", {})) != expected or
        value.get("observed_head_sha256") != {r["case_id"]:r["head_sha256"] for r in plan["cases"]}):
        raise ValueError("Need complete unmixed Stage-63 branches")
    c.prior.c.prior.verify_files(value["cache_sha256"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "smoke", "collect"))
    parser.add_argument("--input", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        import torch
        inputs = c.load_source(args.input)
        hardware = c.prior.c.prior.device_binding(inputs["source"]["plan"], args.gpu, torch)
        prepare(args.input, args.output_dir, hardware)
        return
    if not args.checkpoint:
        parser.error("Need original V1 CHECKPOINT")
    root = Path(args.input).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    with writer_lock(root):
        native, identity = startup.checked_runtime(args, plan, root)
        records = load_records(inputs, native.rt.torch)
        observer = Capture(native)
        try:
            execute(root, plan, inputs, observer, records, identity, args.mode == "smoke")
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
