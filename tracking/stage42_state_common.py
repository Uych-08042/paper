"""Frozen model x initial predicted-parent factorial after Stage-41."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import crop_geometry, failure_events, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_common import dense_boxes
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage40_crop_common as training
import stage41_transfer_audit as prior


SCHEMA = "stage42_frozen_model_parent_swap_v1"
ARMS = ("clean", "augmented")
DESIGN = dict(followup_frames=40, failure_lags=[0, 5, 10], control_quartiles=[1, 2, 3], smoke_frames=4,
    native_box_atol=.001, native_iou_atol=1e-12, decode_atol=.001,
    bootstrap_samples=5000, bootstrap_seed=2121, reliable_iou=.5, low_iou=.2, stable_run=3,
    comparison="Same augmented model from clean versus own recorded parent; follow-up offsets 1..40",
    continuation="One initial predicted-parent swap; own predictions thereafter; first-frame template only")


def code_hashes():
    names = ("stage42_state_common.py", "stage42_state_experiment.py", "stage42_state_summary.py", "test_stage42_state_swap.py",
        "stage41_transfer_audit.py", "stage40_crop_experiment.py", "stage39_perturb_experiment.py", "stage35_preserve_runtime.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    identity = read_sealed(root / "plan.json", prior.SCHEMA)
    old, decision, run, bindings = prior.load_source(identity["source_stage40"])
    report_path = root / "stage41_diagnostic.json"
    report = read_sealed(report_path, prior.SCHEMA)
    evidence_path = root / "stage41_evidence.json"
    evidence = read_sealed(evidence_path, prior.SCHEMA)
    if (report.get("status") != "COMPLETE_CACHED_TRANSFER_AUDIT" or report.get("source_status") != decision["status"] or
            report.get("source_decision_fingerprint") != decision["fingerprint"] or report.get("configuration") != prior.CONFIG or
            report.get("code_sha256") != prior.code_hashes() or report.get("training_performed") is not False or
            report.get("new_inference_performed") is not False or report.get("original_checking_labels_loaded") is not False or
            report.get("videos") != 8 or report.get("seeds") != list(training.SEEDS) or
            identity.get("configuration") != prior.CONFIG or identity.get("code_sha256") != prior.code_hashes() or
            identity.get("source_decision_fingerprint") != decision["fingerprint"] or
            identity.get("source_bindings_fingerprint") != fingerprint(bindings) or
            file_hash(evidence_path) != report.get("evidence_sha256") or evidence["source_file_sha256"] != bindings or
            evidence.get("table_sha256") != report["table_sha256"]):
        raise ValueError("Need the unchanged complete Stage-41 audit and original Stage-40 result")
    bindings.update({str(root / n): d for n, d in report["table_sha256"].items()})
    for path in (root / "plan.json", report_path, evidence_path):
        bindings[str(path)] = file_hash(path)
    verify_bindings(bindings)
    return old, identity["source_stage40"], run, report, bindings


def source_traces(plan, entry, seed):
    original = read_sealed(Path(plan["source_stage40"]) / "manifest.json", training.SCHEMA)
    run = read_sealed(Path(plan["source_stage40"]) / "evaluation/run.json", training.SCHEMA)
    traces = {}
    for arm in ARMS:
        method = arm + "_seed" + str(seed)
        traces[arm] = training.load_result(training.result_path(plan["source_stage40"], entry, method), original, run, entry, method)
    return traces


def video_tasks(entry, seed, traces):
    n, grouped, skipped = len(entry["frames"]) - 1, {}, []
    def add(frame, membership):
        if frame > n:
            skipped.append(dict(sequence=entry["sequence"], seed=seed, requested_frame=frame,
                reason="after_video_end", membership=membership))
            return
        grouped.setdefault(frame, []).append(membership)
    for arm in ARMS:
        events = failure_events(traces[arm]["iou"])
        if events:
            onset = int(events[0]["start"] + 1)
            for lag in DESIGN["failure_lags"]:
                add(onset + lag, dict(kind="first_failure", owner=arm, onset_frame=onset, lag=lag))
    for quartile in DESIGN["control_quartiles"]:
        add(max(1, n * quartile // 4), dict(kind="fixed_time", quartile=quartile))
    tasks = [dict(sequence=entry["sequence"], seed=seed, frame_id=frame,
        frames=min(DESIGN["followup_frames"] + 1, n - frame + 1), memberships=grouped[frame]) for frame in sorted(grouped)]
    return tasks, skipped


def make_plan(root):
    old, original_root, run, report, bindings = source_info(root)
    base = dict(source_stage40=str(Path(original_root).resolve()))
    tasks, skipped = [], []
    for entry in old["fit"]:
        for seed in training.SEEDS:
            values, omitted = video_tasks(entry, seed, source_traces(base, entry, seed))
            tasks.extend(values)
            skipped.extend(omitted)
    smoke = next((t for t in tasks if t["frames"] >= DESIGN["smoke_frames"] and any(m["kind"] == "fixed_time" for m in t["memberships"])), None)
    if smoke is None:
        raise ValueError("No fixed-time task with enough smoke frames")
    return seal(dict(schema=SCHEMA, role="posthoc_frozen_model_by_predicted_parent_intervention",
        source_stage41=str(Path(root).resolve()), source_stage40=base["source_stage40"], source_run_fingerprint=run["fingerprint"],
        source_audit_fingerprint=report["fingerprint"], source_file_sha256=bindings,
        source_runtime=run["runtime"], source_model_states=run["model_state_sha256"], source_model_files=run["model_file_sha256"],
        source_teacher_sha256=run["teacher_sha256"], window=old["window"], runtime_seed=old["runtime_seed"],
        fit=old["fit"], folds=old["folds"], root=old["root"], runtime=old["runtime"],
        search_factor=old["search_factor"], search_size=old["search_size"], design=DESIGN, code_sha256=code_hashes(),
        tasks=tasks, skipped_after_end=skipped, smoke_task=smoke, training_performed=False,
        original_checking_labels_loaded=False, adoption_gate=None))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = make_plan(root)
    for frozen in (root, Path(plan["source_stage40"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-42 output tree")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-42 output directory")
    publish_identical(output / "manifest.json", plan)
    print("Stage-42: {} deduplicated anchors, {} branches, {} requested model frames".format(
        len(plan["tasks"]), 4 * len(plan["tasks"]), 4 * sum(t["frames"] for t in plan["tasks"])))
    print("Short horizons are censored; no short branch is treated as 40-frame follow-up")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes() or make_plan(plan["source_stage41"]) != plan:
        raise ValueError("Frozen Stage-42 code/design/source changed")
    return plan


def task_path(root, mode, task):
    return Path(root) / mode / "branches" / (task["sequence"] + "__seed{}__f{:06d}.npz".format(task["seed"], task["frame_id"]))


def metadata(plan, run, task):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], task=task)


def audit_data(data, plan, entry, task, traces, limit=None):
    horizon = min(task["frames"], limit) if limit else task["frames"]
    side = int(plan["runtime"]["lens_x"] ** .5)
    shape = (2, 2, horizon)
    shapes = dict(previous=shape + (4,), boxes=shape + (4,), raw=shape + (side, side),
        size=shape + (2, side, side), offset=shape + (2, side, side), factor=shape, layer=shape, decode_error=shape,
        head_calls=(2, 2), window=(side, side), image_hw=(horizon, 2), gt=(horizon, 4), iou=shape)
    if set(data) != set(shapes) or any(data[k].shape != v for k, v in shapes.items()):
        raise ValueError("State-swap branch shape/coverage changed")
    if (not all(np.isfinite(v).all() for v in data.values()) or (data["boxes"][..., 2:] <= 0).any() or
            (data["raw"] <= 0).any() or (data["raw"] > 1).any() or (data["size"] < 0).any() or
            (data["factor"] <= 0).any() or data["layer"].any()):
        raise ValueError("Invalid frozen state-swap arrays")
    np.testing.assert_array_equal(data["head_calls"], np.full((2, 2), horizon))
    np.testing.assert_array_equal(data["window"], np.asarray(plan["window"], np.float32))
    start = task["frame_id"]
    np.testing.assert_array_equal(data["image_hw"], [[entry["frames"][f]["height"], entry["frames"][f]["width"]]
        for f in range(start, start + horizon)])
    parity = []
    for m, model in enumerate(ARMS):
        for p, parent_arm in enumerate(ARMS):
            expected = np.vstack([traces[parent_arm]["previous"][start - 1], data["boxes"][m, p, :-1]])
            np.testing.assert_array_equal(data["previous"][m, p], expected)
            np.testing.assert_allclose(data["iou"][m, p], tracking_iou(data["boxes"][m, p], data["gt"]), atol=1e-12, rtol=0)
            for t in range(horizon):
                factor = crop_geometry(expected[t], data["image_hw"][t], plan["search_factor"], plan["search_size"])[2]
                np.testing.assert_allclose(data["factor"][m, p, t], factor, atol=1e-12, rtol=0)
                dense = dense_boxes(data["size"][m, p, t], data["offset"][m, p, t], expected[t], factor,
                    plan["search_size"], data["image_hw"][t])
                cell = int(np.argmax(data["raw"][m, p, t] * data["window"]))
                error = float(np.max(np.abs(dense[cell] - data["boxes"][m, p, t])))
                np.testing.assert_allclose(data["decode_error"][m, p, t], error, atol=1e-12, rtol=0)
                if error > DESIGN["decode_atol"]:
                    raise ValueError("State-swap native/dense decoder mismatch")
        sl = slice(start - 1, start - 1 + horizon)
        box_error = float(np.max(np.abs(data["boxes"][m, m] - traces[model]["boxes"][sl])))
        iou_error = float(np.max(np.abs(data["iou"][m, m] - traces[model]["iou"][sl])))
        if box_error > DESIGN["native_box_atol"] or iou_error > DESIGN["native_iou_atol"]:
            raise ValueError("Native state-swap diagonal differs from frozen Stage-40 trajectory")
        parity.append(dict(model=model, max_box_error=box_error, max_iou_error=iou_error))
    return parity


def load_result(path, plan, run, entry, task, traces, limit=None, allow_orphan=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != metadata(plan, run, task):
            raise ValueError("Mixed Stage-42 branch metadata")
        data = {k: a[k] for k in a.files if k != "metadata_json"}
    receipt = seal(dict(schema=SCHEMA, metadata=metadata(plan, run, task), trace_sha256=file_hash(path)))
    if path.with_suffix(".json").exists():
        if read_sealed(path.with_suffix(".json"), SCHEMA) != receipt:
            raise ValueError("Completed Stage-42 branch changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted Stage-42 branch; resume collector")
    parity = audit_data(data, plan, entry, task, traces, limit)
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], read_boxes(annotation)[task["frame_id"]:task["frame_id"] + len(data["gt"])])
    if not path.with_suffix(".json").exists():
        publish_identical(path.with_suffix(".json"), receipt)
    return data, parity
