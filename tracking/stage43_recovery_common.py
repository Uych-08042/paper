"""Frozen, GT-free recovery pilot after the Stage-42 parent intervention."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import publish_identical, seal, tracking_iou
from stage28_response_common import clip_boxes
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage42_state_common as source


SCHEMA = "stage43_online_history_recovery_v1"
MODELS, SEEDS = source.ARMS, source.training.SEEDS
POLICIES = ("native", "motion_hold", "local_search", "history_search")
DESIGN = dict(history=8, score_ratio=.5, innovation_limit=1., velocity_cap=.5,
    extrapolation_cap=10, search_expansion=2., cooldown=10, extra_call_percent=10,
    smoke_frames=64, bootstrap_samples=5000, bootstrap_seed=2122,
    primary_model="augmented", minimum_auc_gain=.5, minimum_positive_videos=5, reliable_mean_floor=-.5,
    reliable_ci_floor=-1., minimum_reliable_frames=10, minimum_reliable_videos=4,
    minimum_changed_ratio=.001, minimum_changed_videos=2,
    primary="History-search minus native full-video AUC; seeds averaged inside each video",
    trigger="After eight accepted predictions: raw response < half historical median OR center innovation > one historical side",
    choice="Alternative passes historical score floor and has strictly smaller forecast-center error than native",
    budget="At most floor(10% * elapsed frames) interventions and >=10 frames between interventions",
    history_role="Model-consistent predictions, never GT-certified reliable states")


def center(box):
    b = np.asarray(box, float)
    return b[:2] + b[2:] / 2


class Controller:
    """Only current model observations and past accepted predictions enter here."""
    def __init__(self, initial_box, policy):
        if policy not in POLICIES:
            raise ValueError("Unknown recovery policy")
        self.policy, self.history = policy, []
        self.initial = np.asarray(initial_box, float).copy()
        self.used, self.last = 0, -DESIGN["cooldown"]

    def forecast(self, frame):
        if not self.history:
            return self.initial.copy()
        last, box, _ = self.history[-1]
        velocity = np.zeros(2)
        if len(self.history) > 1:
            old, before, _ = self.history[-2]
            velocity = (center(box) - center(before)) / (last - old)
        side = float(np.sqrt(np.prod(box[2:])))
        velocity *= min(1., DESIGN["velocity_cap"] * side / max(np.linalg.norm(velocity), 1e-12))
        return np.r_[center(box) + velocity * min(frame - last, DESIGN["extrapolation_cap"]) - box[2:] / 2, box[2:]]

    def inspect(self, frame, previous, native, score, hw):
        forecast = clip_boxes(self.forecast(frame), hw)
        side = float(np.sqrt(np.prod(forecast[2:])))
        innovation = float(np.linalg.norm(center(native) - center(forecast)) / side)
        floor = DESIGN["score_ratio"] * float(np.median([x[2] for x in self.history])) if self.history else 0.
        ready = len(self.history) >= DESIGN["history"]
        alarm = ready and (score < floor or innovation > DESIGN["innovation_limit"])
        cap = frame * DESIGN["extra_call_percent"] // 100
        eligible = bool(alarm and self.policy != "native" and self.used < cap and frame - self.last >= DESIGN["cooldown"])
        parent = np.asarray(previous, float).copy() if self.policy == "local_search" else forecast.copy()
        parent = np.r_[center(parent) - parent[2:] * DESIGN["search_expansion"] / 2,
            parent[2:] * DESIGN["search_expansion"]]
        return dict(alarm=bool(alarm), eligible=eligible, score_floor=floor, innovation=innovation,
            forecast=forecast, alternate_parent=parent, cap=cap)

    def finish(self, frame, previous, native, score, hw, alternative=None):
        view = self.inspect(frame, previous, native, score, hw)
        selected, chosen_score, choice = np.asarray(native, float).copy(), score, 0
        needs_extra = view["eligible"] and self.policy in ("local_search", "history_search")
        if needs_extra != (alternative is not None):
            raise ValueError("Alternative call does not match causal trigger/budget")
        if view["eligible"]:
            self.used += 1
            self.last = frame
            if self.policy == "motion_hold":
                selected, choice = view["forecast"].copy(), 2
            elif alternative is not None:
                box, confidence = alternative
                side = float(np.sqrt(np.prod(view["forecast"][2:])))
                innovation = np.linalg.norm(center(box) - center(view["forecast"])) / side
                if confidence >= view["score_floor"] and innovation < view["innovation"]:
                    selected, chosen_score, choice = np.asarray(box, float).copy(), confidence, 1
        # A motion-only output is never treated as new visual evidence.
        side = float(np.sqrt(np.prod(view["forecast"][2:])))
        consistent = (chosen_score >= view["score_floor"] and
            np.linalg.norm(center(selected) - center(view["forecast"])) / side <= DESIGN["innovation_limit"])
        update = choice != 2 and (len(self.history) < DESIGN["history"] or consistent)
        if update:
            self.history.append((frame, selected.copy(), float(chosen_score)))
            self.history = self.history[-DESIGN["history"]:]
        return selected, dict(view, choice=choice, history_update=bool(update), interventions=self.used)


def code_hashes():
    names = ("stage43_recovery_common.py", "stage43_recovery_experiment.py", "stage43_recovery_summary.py",
        "test_stage43_recovery.py", "stage42_state_common.py", "stage42_state_experiment.py",
        "stage42_state_summary.py", "stage40_crop_experiment.py", "stage35_preserve_runtime.py", "stage4_runtime.py")
    return {n:file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    old = source.load_plan(root)
    report_path, evidence_path = root / "analysis/stage42_diagnostic.json", root / "analysis/stage42_evidence.json"
    report, evidence = read_sealed(report_path, source.SCHEMA), read_sealed(evidence_path, source.SCHEMA)
    from stage42_state_experiment import smoke_gate, validate_run
    smoke_gate(root, old)
    run = read_sealed(root / "full/run.json", source.SCHEMA)
    validate_run(run, old, "full")
    if (report.get("status") != "COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC" or
            report.get("plan_fingerprint") != old["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("videos") != 8 or report.get("seeds") != list(SEEDS) or
            report.get("unique_anchor_packs") != len(old["tasks"]) or report.get("branches") != 4 * len(old["tasks"]) or
            report.get("training_performed") is not False or file_hash(evidence_path) != report.get("evidence_sha256") or
            evidence.get("table_sha256") != report.get("table_sha256")):
        raise ValueError("Need the unchanged complete Stage-42 diagnostic")
    bindings = dict(old["source_file_sha256"])
    bindings.update(evidence["source_file_sha256"])
    for task in old["tasks"]:
        path = source.task_path(root, "full", task)
        if str(path) not in bindings or str(path.with_suffix(".json")) not in bindings:
            raise ValueError("Incomplete Stage-42 branch evidence")
    bindings.update({str(root / "analysis" / n):d for n,d in report["table_sha256"].items()})
    for p in (root / "manifest.json", root / "full/run.json", report_path, evidence_path):
        bindings[str(p)] = file_hash(p)
    verify_bindings(bindings)
    return old, report, bindings


def make_plan(root):
    old, report, bindings = source_info(root)
    return seal(dict(schema=SCHEMA, role="consumed_development_online_recovery_feasibility",
        source_stage42=str(Path(root).resolve()), source_stage40=old["source_stage40"],
        source_diagnostic_fingerprint=report["fingerprint"], source_file_sha256=bindings,
        **{k:old[k] for k in ("fit", "folds", "root", "runtime", "source_runtime", "source_model_states",
            "source_model_files", "source_teacher_sha256", "runtime_seed", "window", "search_factor", "search_size")},
        design=DESIGN, code_sha256=code_hashes(), models=list(MODELS), policies=list(POLICIES), seeds=list(SEEDS),
        training_performed=False, original_checking_labels_loaded=False))


def prepare(root, output):
    plan = make_plan(root)
    output = Path(output).resolve()
    for frozen in (Path(root).resolve(), Path(plan["source_stage40"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-43 output tree")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-43 output directory")
    publish_identical(output / "manifest.json", plan)
    frames = sum(len(e["frames"]) - 1 for e in plan["fit"])
    print("Stage-43: 8 consumed videos x 2 seeds x 2 frozen models x 4 policies = 128 full traces")
    print("Base forwards: {}; extra forwards capped per prefix and video at 10% for search arms".format(frames * 16))
    print("No new training; no GT failure times or other-model parents enter the controller")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes() or make_plan(plan["source_stage42"]) != plan:
        raise ValueError("Frozen Stage-43 source/code/design changed")
    return plan


def tasks(plan, mode):
    if mode == "smoke":
        return [(plan["fit"][0], SEEDS[0], m, p) for m in MODELS for p in POLICIES]
    return [(e, s, m, p) for e in plan["fit"] for s in SEEDS for m in MODELS for p in POLICIES]


def result_path(root, mode, entry, seed, model, policy):
    return Path(root) / mode / "traces" / (entry["sequence"] + "__seed{}__{}__{}.npz".format(seed, model, policy))


def metadata(run, entry, seed, model, policy):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], seed=seed, model=model, policy=policy)


def audit_control(data, entry, policy):
    n = len(data["boxes"])
    floats = ("previous", "boxes", "native_boxes", "native_score", "alternate_boxes", "alternate_score",
        "alternate_parent", "forecast", "score_floor", "innovation", "native_decode_error", "alternate_decode_error")
    flags = ("alarm", "eligible", "choice", "history_update", "interventions", "head_calls")
    shapes = {k:(n, 4) if k in ("previous", "boxes", "native_boxes", "alternate_boxes", "alternate_parent", "forecast") else (n,) for k in floats + flags}
    shapes.update(image_hw=(n, 2), gt=(n, 4), iou=(n,))
    if set(data) != set(shapes) or any(data[k].shape != v for k,v in shapes.items()) or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid Stage-43 trace shapes/finite values")
    if (data["boxes"][:, 2:] <= 0).any() or (data["native_score"] <= 0).any() or (data["native_score"] > 1).any():
        raise ValueError("Invalid online output/response")
    previous = np.asarray(entry["initial_box"], float)
    controller = Controller(previous, policy)
    for i in range(n):
        np.testing.assert_array_equal(data["previous"][i], previous)
        view = controller.inspect(i + 1, previous, data["native_boxes"][i], data["native_score"][i], data["image_hw"][i])
        extra = view["eligible"] and policy in ("local_search", "history_search")
        alternative = (data["alternate_boxes"][i], data["alternate_score"][i]) if extra else None
        box, log = controller.finish(i + 1, previous, data["native_boxes"][i], data["native_score"][i], data["image_hw"][i], alternative)
        np.testing.assert_array_equal(data["boxes"][i], box)
        for k in ("alarm", "eligible", "choice", "history_update", "interventions", "score_floor", "innovation", "forecast"):
            np.testing.assert_array_equal(data[k][i], log[k])
        if extra:
            np.testing.assert_array_equal(data["alternate_parent"][i], view["alternate_parent"])
            if not 0 < data["alternate_score"][i] <= 1:
                raise ValueError("Invalid recovery response")
        elif any(np.any(data[k][i]) for k in ("alternate_parent", "alternate_boxes", "alternate_score", "alternate_decode_error")):
            raise ValueError("Unused alternate slot must remain zero")
        if data["head_calls"][i] != 1 + int(extra) or controller.used > view["cap"]:
            raise ValueError("Online call/prefix intervention budget violated")
        previous = box
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], data["gt"]), atol=1e-12, rtol=0)
    if max(data["native_decode_error"].max(), data["alternate_decode_error"].max()) > source.DESIGN["decode_atol"]:
        raise ValueError("Online native/dense decode mismatch")


def load_result(path, plan, run, entry, seed, model, policy, allow_orphan=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != metadata(run, entry, seed, model, policy):
            raise ValueError("Mixed Stage-43 trace metadata")
        data = {k:a[k] for k in a.files if k != "metadata_json"}
    receipt = seal(dict(schema=SCHEMA, metadata=metadata(run, entry, seed, model, policy), trace_sha256=file_hash(path)))
    if path.with_suffix(".json").exists():
        if read_sealed(path.with_suffix(".json"), SCHEMA) != receipt:
            raise ValueError("Completed Stage-43 trace changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted Stage-43 trace; resume collector")
    n = min(len(entry["frames"]) - 1, DESIGN["smoke_frames"]) if run["mode"] == "smoke" else len(entry["frames"]) - 1
    if len(data["boxes"]) != n:
        raise ValueError("Incomplete Stage-43 video")
    audit_control(data, entry, policy)
    from stage27_uav_common import read_boxes
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation):entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], read_boxes(annotation)[1:n + 1])
    np.testing.assert_array_equal(data["image_hw"], [[f["height"], f["width"]] for f in entry["frames"][1:n + 1]])
    if not path.with_suffix(".json").exists():
        publish_identical(path.with_suffix(".json"), receipt)
    return data
