"""Frozen, outcome-conditioned four-case replay of existing Stage-35 models."""

from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import failure_events, publish_identical, seal
from stage28_response_common import DESIGN as RESPONSE_DESIGN
from stage34_identity_common import load_sealed as read_sealed
from stage35_preserve_common import DESIGN as SOURCE_DESIGN
from stage35_supervised_audit import AUDIT as BASELINE_AUDIT_CONFIG, SCHEMA as AUDIT_SCHEMA, load_source, verify_bindings


SCHEMA = "stage36_same_crop_mechanism_v1"
METHODS = ("fast_initial", "sgla", "supervised_seed2110", "supervised_seed2111")
DESIGN = dict(cases=["uav0000016_00000_s", "uav0000085_00000_s", "uav0000089_00920_s", "uav0000090_00276_s"],
    before=10, after=30, discrepancy_iou=.2, discrepancy_run=3,
    control_quartiles=[1, 2, 3], replay_box_atol=.001, replay_iou_atol=1e-12,
    decode_box_atol=.001, seed=2114)


def load_sealed(path):
    return read_sealed(path, SCHEMA)


def code_hashes():
    names = ("stage36_same_crop_common.py", "stage36_same_crop_collect.py", "stage36_same_crop_summary.py",
        "stage28_response_common.py", "stage27_uav_review.py", "stage35_supervised_audit.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def source_info(root):
    root = Path(root).resolve()
    source, original, _, bindings = load_source(root)
    audit_path = root / "analysis/supervised_baseline_audit_v1/stage35b_audit.json"
    audit = read_sealed(audit_path, AUDIT_SCHEMA)
    if (audit.get("status") != "COMPLETE_POSTHOC_SUPERVISED_AUDIT" or
            audit.get("source_plan_fingerprint") != source["fingerprint"] or
            audit.get("original_decision_fingerprint") != original["fingerprint"] or
            audit.get("original_status") != "NO_PRESERVE_CORRECT_GAIN_ESTABLISHED" or
            audit.get("configuration") != BASELINE_AUDIT_CONFIG or audit.get("original_checking_labels_loaded") is not False or
            audit.get("training_performed") is not False or audit.get("new_inference_performed") is not False):
        raise ValueError("Need the frozen completed Stage-35B audit following the Stage-35 failure")
    verify_bindings(audit["source_file_sha256"])
    for name, digest in audit["code_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Frozen Stage-35B code changed")
    bindings.update(audit["source_file_sha256"])
    bindings[str(audit_path)] = file_hash(audit_path)
    entries = {e["sequence"]: e for e in source["fit"]}
    if len(DESIGN["cases"]) != 4 or len(set(DESIGN["cases"])) != 4 or not set(DESIGN["cases"]) <= set(entries):
        raise ValueError("The four outcome-selected cases must exist in the frozen fit-only OOF source")
    traces = {}
    for name in DESIGN["cases"]:
        traces[name] = {}
        for method in METHODS:
            with np.load(root / "oof" / (name + "__" + method + ".npz"), allow_pickle=False) as archive:
                traces[name][method] = {k: archive[k] for k in archive.files if k != "metadata_json"}
    return source, traces, bindings


def first_run(mask, length):
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        if count == length:
            return i - length + 1
    return None


def video_plan(entry, traces):
    n = len(traces["fast_initial"]["iou"])
    anchors = []
    for method in METHODS:
        events = failure_events(traces[method]["iou"])
        if events:
            anchors.append(dict(kind="first_failure", owner=method, reference="", direction="low_iou",
                frame_id=int(events[0]["start"] + 1)))
    for student in METHODS[2:]:
        for baseline in METHODS[:2]:
            delta = traces[student]["iou"] - traces[baseline]["iou"]
            runs = [(first_run(delta >= DESIGN["discrepancy_iou"], DESIGN["discrepancy_run"]), "student_higher"),
                    (first_run(delta <= -DESIGN["discrepancy_iou"], DESIGN["discrepancy_run"]), "student_lower")]
            runs = [(row, direction) for row, direction in runs if row is not None]
            if runs:
                row, direction = min(runs)
                anchors.append(dict(kind="first_sustained_discrepancy", owner=student, reference=baseline,
                    direction=direction, frame_id=int(row + 1)))
    frames = {}
    for quartile in DESIGN["control_quartiles"]:
        frame = max(1, n * quartile // 4)
        frames.setdefault(frame, dict(frame_id=frame, controls=[], anchors=[]))["controls"].append(quartile)
    for number, anchor in enumerate(anchors):
        anchor["anchor_id"] = number
        for frame in range(max(1, anchor["frame_id"] - DESIGN["before"]), min(n, anchor["frame_id"] + DESIGN["after"]) + 1):
            frames.setdefault(frame, dict(frame_id=frame, controls=[], anchors=[]))["anchors"].append(
                dict(anchor_id=number, offset=frame - anchor["frame_id"]))
    return dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
        anchors=anchors, frames=[frames[f] for f in sorted(frames)])


def make_plan(root, info=None):
    root = Path(root).resolve()
    source, traces, bindings = source_info(root) if info is None else info
    bindings = dict(bindings)
    entries = {e["sequence"]: e for e in source["fit"]}
    videos = [video_plan(entries[n], traces[n]) for n in DESIGN["cases"]]
    images = {}
    for video in videos:
        entry = entries[video["sequence"]]
        for frame in {0} | {f["frame_id"] for f in video["frames"]}:
            info = entry["frames"][frame]
            images[str(Path(source["root"]) / info["file"])] = info["sha256"]
    verify_bindings(images)
    bindings.update(images)
    return seal(dict(schema=SCHEMA, role="posthoc_development_crossed_crop_replay_plan", source_input=str(root),
        source_plan_fingerprint=source["fingerprint"], source_file_sha256=bindings,
        design=DESIGN, candidate_rules=RESPONSE_DESIGN, code_sha256=code_hashes(), videos=videos,
        methods=list(METHODS), parent_methods=list(METHODS), runtime=source["runtime"],
        search_factor=source["search_factor"], search_size=source["search_size"],
        unique_frame_images=sum(len(v["frames"]) for v in videos),
        planned_model_forwards=16 * sum(len(v["frames"]) for v in videos),
        interpretation="Four cases and all windows use consumed outcomes; independent single-state replays, not new policy rollouts",
        training_performed=False, adoption_gate=None))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Use a separate Stage-36 directory, outside the frozen Stage-35 tree")
    plan = make_plan(root)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "plan.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty output directory")
    publish_identical(output / "plan.json", plan)
    print("Stage-36 frozen plan: 4 cases, {} images, {} model forwards".format(plan["unique_frame_images"], plan["planned_model_forwards"]))
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_inputs(output):
    output = Path(output).resolve()
    plan = load_sealed(output / "plan.json")
    if plan["design"] != DESIGN or plan["candidate_rules"] != RESPONSE_DESIGN or plan["code_sha256"] != code_hashes():
        raise ValueError("Stage-36 code or diagnostic rules changed")
    info = source_info(plan["source_input"])
    if plan != make_plan(plan["source_input"], info):
        raise ValueError("Frozen replay plan/source changed")
    source, traces, _ = info
    return plan, source, traces


def load_models(rt, source, source_root):
    from stage20_compression_common import state_digest
    from stage20_student_train import load_checkpoint
    from stage35_preserve_common import load_sealed as load_training_report
    from stage35_preserve_runtime import make_student, restore_state
    models = {}
    for seed in SOURCE_DESIGN["seeds"]:
        report = load_training_report(Path(source_root) / "training" / ("seed{}".format(seed)) / "stage35_training_decision.json")
        if report["teacher_sha256"] != state_digest(rt.net):
            raise ValueError("Original teacher differs from frozen Stage-35 training")
        for fold in source["folds"]:
            record = next(r for r in report["arms"] if r["fold"] == fold["fold"] and r["arm"] == "supervised")
            path = Path(record["checkpoint"])
            if file_hash(path) != record["checkpoint_sha256"]:
                raise ValueError("Frozen supervised checkpoint changed")
            state = load_checkpoint(rt.torch, path)
            meta = state["metadata"]
            if (state["step"] != SOURCE_DESIGN["updates"] or meta["plan_fingerprint"] != source["fingerprint"] or
                    meta["mode"] != "full" or meta["arm"] != "supervised" or meta["seed"] != seed or
                    meta["fold"] != fold["fold"] or meta["fit_videos"] != fold["fit"] or meta["held_out_videos"] != fold["held_out"] or
                    meta["teacher_sha256"] != state_digest(rt.net)):
                raise ValueError("Wrong OOF source/seed/fold checkpoint")
            net = make_student(rt)
            if state_digest(net) != meta["initial_student_sha256"]:
                raise ValueError("Student must be initialized from the original Fast path")
            restore_state(net, state["parameters"])
            models[(fold["fold"], "supervised_seed" + str(seed))] = net.eval().requires_grad_(False)
    return models
