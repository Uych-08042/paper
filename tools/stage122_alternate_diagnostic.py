"""Separate alternate-slot supply from score changes using sealed Stage-121 caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

import stage121_search_state as source
from stage27_uav_common import success_auc, tracking_iou

base, c, io = source.base, source.c, source.io
SCHEMA = "stage122_cached_alternate_reranking_v1"
STATUS = "COMPLETE_CACHED_ALTERNATE_DIAGNOSTIC"
SOURCE121 = "a83daf985bd2d0fd2ff00dd91793c12bb4960936db881400186b6abf8ac0d142"
CONTROL = "generated_shared"
SLOT = source.DESIGN["slot"]
RELATIONS = {"mean_overlap": 24, "max_overlap": 25, "mean_center_distance": 29,
             "mean_size_difference": 30, "hann_rank": 31}
OBSERVATIONS = ("candidate_parents", "candidate_boxes", "features", "crop", "crop_sha256",
                "layer", "raw", "size", "offset")
DESIGN = dict(alternate_slot=SLOT, good_iou=.5, future_frames=31, relations=RELATIONS)
TABLES = ("summary", "trajectories", "activations", "retained_scores", "relation_swaps")
WORK = dict(new_training_updates=0, new_search_calls=0, new_image_reads=0, confirmation_inference_calls=0)


def first(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def history(data, initial, index, length):
    if index < length:
        return np.vstack((initial, data["boxes"][:index]))
    return data["boxes"][index-length:index]


def same_input(ref, alt, index, h_ref, h_alt):
    return (np.array_equal(h_ref, h_alt)
            and np.array_equal(ref["forecast_x"][index], alt["forecast_x"][index])
            and bool(ref["generated_active"][index]) == bool(alt["generated_active"][index]))


def matching_observations(ref, alt, index):
    return all(np.array_equal(ref[k][index, :SLOT], alt[k][index, :SLOT]) for k in OBSERVATIONS)


def shared_scores(ref, alt, index, h, scorer):
    common = base.common
    boxes = ref["candidate_boxes"][index, :SLOT]
    x_ref, x_alt = (common.features(arm["features"][index], arm["candidate_boxes"][index], h)
                    for arm in (ref, alt))
    score_ref, score_alt = (common.scores(scorer, x) for x in (x_ref, x_alt))
    for arm, scores in ((ref, score_ref), (alt, score_alt)):
        np.testing.assert_allclose(scores, arm["scores"][index], rtol=1e-9, atol=1e-12)
        np.testing.assert_equal(common.physical_choice(scores, arm["candidate_boxes"][index]), arm["selected"][index])

    columns = {i for column in RELATIONS.values() for i in (column, column+32)}
    fixed = sorted(set(range(64)) - columns)
    np.testing.assert_array_equal(x_ref[:SLOT, fixed], x_alt[:SLOT, fixed])
    choose = lambda scores: common.physical_choice(scores[:SLOT], boxes)
    before, after = choose(score_ref), choose(score_alt)
    scores = [dict(slot=q, old_score=float(score_ref[q]), new_score=float(score_alt[q]),
                   score_delta=float(score_alt[q]-score_ref[q]),
                   **{name+"_delta": float(x_alt[q, j]-x_ref[q, j]) for name, j in RELATIONS.items()})
              for q in range(SLOT)]
    swaps = []
    for name, j in RELATIONS.items():
        cols = [j, j+32]
        inserted, restored = x_ref.copy(), x_alt.copy()
        inserted[:SLOT, cols] = x_alt[:SLOT, cols]
        restored[:SLOT, cols] = x_ref[:SLOT, cols]
        score_inserted = common.scores(scorer, inserted)
        score_restored = common.scores(scorer, restored)
        only, omit = choose(score_inserted), choose(score_restored)
        swaps.append(dict(relation=name, retained_old_winner=before, retained_new_winner=after,
                          inserted_winner=only, restored_winner=omit,
                          insertion_matches_new=after != before and np.array_equal(boxes[only], boxes[after]),
                          restoration_matches_old=after != before and np.array_equal(boxes[omit], boxes[before]),
                          **{f"q{q}_insertion_delta": float(score_inserted[q]-score_ref[q]) for q in range(SLOT)}))
    return before, after, not np.array_equal(boxes[before], boxes[after]), scores, swaps


def mechanism(changed, control_slot, alternate_slot, matched):
    if not matched:
        return "unmatched_context"
    if not changed:
        return "same_output"
    if control_slot == SLOT and alternate_slot == SLOT:
        return "both_choose_slot4"
    if alternate_slot == SLOT:
        return "alternate_box_selected"
    if control_slot == SLOT:
        return "control_slot4_displaced"
    return "retained_boxes_reranked"


def diagnose(ref, alt, entry, scorer, length, identity):
    gt = np.asarray(entry["ground_truth"], float)[1:]
    n = len(gt)
    for data in (ref, alt):
        np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
        np.testing.assert_array_equal(data["boxes"], data["candidate_boxes"][np.arange(n), data["selected"]])
    np.testing.assert_array_equal(ref["image_hw"], alt["image_hw"])
    first_activation = source.paired_prefix(alt, ref)
    changed = (alt["boxes"] != ref["boxes"]).any(axis=1)
    first_output = first(changed)
    first_crop = first(np.any(alt["crop"][:, :SLOT] != ref["crop"][:, :SLOT], axis=(1, 2)))
    first_score = first(np.any(alt["scores"][:, :SLOT] != ref["scores"][:, :SLOT], axis=1))
    valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
    iou_ref, iou_alt = (tracking_iou(data["candidate_boxes"], gt[:, None, :]) for data in (ref, alt))
    selected_ref, selected_alt = (ious[np.arange(n), data["selected"]]
                                  for ious, data in ((iou_ref, ref), (iou_alt, alt)))
    future = np.zeros(n, bool)
    if first_output is not None:
        future[first_output:first_output+DESIGN["future_frames"]] = True
    future &= valid
    activations, detail, swaps = [], [], []
    for index in np.flatnonzero(alt["alternate_used"]):
        frame = int(index) + 1
        h_ref, h_alt = (history(data, entry["initial_box"], index, length) for data in (ref, alt))
        shared_input = same_input(ref, alt, index, h_ref, h_alt)
        shared_observation = matching_observations(ref, alt, index)
        matched = shared_input and shared_observation
        if np.array_equal(ref["crop"][index, SLOT], alt["crop"][index, SLOT]) and matched:
            raise ValueError("Matched activation did not replace the fourth crop")
        if frame == first_activation and not matched:
            raise ValueError("First substitution does not have a common state and retained observation")
        q_ref, q_alt = int(ref["selected"][index]), int(alt["selected"][index])
        label = mechanism(changed[index], q_ref, q_alt, matched)
        before = after = reranked = None
        ident = dict(identity, frame=frame)
        if matched:
            before, after, reranked, scores, changes = shared_scores(ref, alt, index, h_ref, scorer)
            if label == "retained_boxes_reranked" and not reranked:
                label = "retained_selection_changed_only_in_full_bank"
            detail.extend(dict(ident, **r) for r in scores)
            swaps.extend(dict(ident, **r) for r in changes)

        valid_frame = bool(valid[index])
        retained_good = bool(iou_alt[index, :SLOT].max() >= DESIGN["good_iou"]) if valid_frame else None
        alt_only_good = (bool(iou_alt[index, SLOT] >= DESIGN["good_iou"] and not retained_good)
                         if valid_frame else None)
        activations.append(dict(ident, first_activation=frame == first_activation, same_input=shared_input,
                                same_retained_observations=shared_observation, matched=matched, mechanism=label,
                                first_output_difference=frame == first_output, box_changed=bool(changed[index]),
                                control_slot=q_ref, alternate_slot=q_alt, retained_old_winner=before,
                                retained_new_winner=after, retained_box_reranked=reranked, gt_valid=valid_frame,
                                control_iou=float(selected_ref[index]) if valid_frame else None,
                                alternate_iou=float(selected_alt[index]) if valid_frame else None,
                                control_slot4_iou=float(iou_ref[index, SLOT]) if valid_frame else None,
                                alternate_slot4_iou=float(iou_alt[index, SLOT]) if valid_frame else None,
                                retained_good=retained_good, alternate_only_good=alt_only_good,
                                net_good_bank_added=(bool(iou_ref[index].max() < DESIGN["good_iou"]
                                                         <= iou_alt[index].max()) if matched and valid_frame else None),
                                net_good_bank_removed=(bool(iou_alt[index].max() < DESIGN["good_iou"]
                                                           <= iou_ref[index].max()) if matched and valid_frame else None)))
    first_mechanism = next((r["mechanism"] for r in activations if r["frame"] == first_output), None)
    if first_output is not None and first_mechanism in (None, "unmatched_context"):
        first_mechanism = "first_output_without_matched_substitution"
    ctrl_auc, alt_auc = (success_auc(ious) for ious in (selected_ref, selected_alt))
    row = dict(identity, frames=n, first_activation=first_activation, first_retained_score_change=first_score,
               first_retained_crop_change=first_crop, first_output_change=first_output, first_output_mechanism=first_mechanism,
               activations=len(activations), matched_activations=sum(r["matched"] for r in activations),
               alternate_slot4_selections=sum(r["alternate_slot"] == SLOT for r in activations),
               control_auc=ctrl_auc, alternate_auc=alt_auc, auc_delta=alt_auc-ctrl_auc,
               subsequent_valid_frames=int(future.sum()),
               subsequent_control_iou=float(selected_ref[future].mean()) if future.any() else None,
               subsequent_alternate_iou=float(selected_alt[future].mean()) if future.any() else None)
    return row, activations, detail, swaps


def source_rows(root, table):
    with (root / "analysis" / f"stage121_{table}.csv").open(encoding="utf-8", newline="") as stream:
        return {(r["sequence"], int(r["seed"]), r["arm"]): r for r in csv.DictReader(stream)}


def check_source_row(row, scope, metrics):
    seq, seed = row["sequence"], row["seed"]
    treatment = (seq, seed, source.TREATMENT)
    saved = scope[treatment]
    first_frame = int(saved["first_alternate_frame"]) if saved["first_alternate_frame"] else None
    if (row["frames"], row["activations"], row["alternate_slot4_selections"], row["first_activation"]) != (
            int(saved["frames"]), int(saved["alternate_used_frames"]), int(saved["alternate_selected_frames"]), first_frame):
        raise ValueError("Stage-121 reported activation counts changed")
    np.testing.assert_allclose([row["control_auc"], row["alternate_auc"]],
                               [float(metrics[seq, seed, CONTROL]["auc"]), float(metrics[treatment]["auc"])],
                               rtol=0, atol=1e-10)


def publish(root, plan, tables):
    exposure = tables["activations"]
    matched = [r for r in exposure if r["matched"]]
    first_output = [r for r in tables["trajectories"] if r["first_output_change"] is not None]
    summary = []
    for group, labels in (("activations", [r["mechanism"] for r in exposure]),
                           ("first_output_changes", [r["first_output_mechanism"] for r in first_output])):
        summary.extend(dict(group=group, mechanism=k, count=v) for k, v in sorted(Counter(labels).items()))
    tables["summary"] = summary
    counts = {k:len(v) for k, v in tables.items()}
    if (counts["trajectories"], counts["activations"]) != (plan["pairs"], plan["activations"]):
        raise ValueError("Diagnosis does not cover the frozen source population")
    if counts["retained_scores"] != len(matched)*SLOT or counts["relation_swaps"] != len(matched)*len(RELATIONS):
        raise ValueError("Incomplete matched scoring diagnostic")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for key in TABLES:
        path = out / f"stage122_{key}.csv"
        base.s103.s86.publish_table(path, tables[key])
        hashes[path.name] = c.file_hash(path)
    proof = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
                         source_stage121_fingerprint=plan["source_stage121_fingerprint"], row_counts=counts,
                         matched_activations=len(matched), identical_retained_boxes_reranked=sum(
                             r["retained_box_reranked"] for r in matched),
                         full_output_changes=sum(r["box_changed"] for r in matched),
                         first_output_mechanisms=dict(Counter(r["first_output_mechanism"] for r in first_output)),
                         table_sha256=hashes, **WORK))
    c.publish_identical(out / "stage122_decision.json", proof)
    check_results(root)
    print(f"Stage-122: {STATUS}. {len(matched)}/{len(exposure)} matched activations", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage122_decision.json", SCHEMA)
    counts = report["row_counts"]
    evidence = {f"stage122_{key}.csv" for key in TABLES}
    if (plan["design"] != DESIGN or plan["source_stage121_fingerprint"] != SOURCE121
            or report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage121_fingerprint"] != SOURCE121 or set(report["table_sha256"]) != evidence
            or set(counts) != set(TABLES) or counts["trajectories"] != plan["pairs"]
            or counts["activations"] != plan["activations"] or any(report[k] != v for k, v in WORK.items())):
        raise ValueError("Incomplete Stage-122 diagnosis")
    io.verify(plan["inputs"])
    io.verify({str(root / "analysis" / k): v for k, v in report["table_sha256"].items()})


def load_trace(root, plan, job):
    path = root / "evaluation" / (base.job_key(job) + ".npz")
    data = c.read_cache(path, dict(plan_fingerprint=plan["fingerprint"], job=job))
    if data is None:
        raise ValueError(f"Missing Stage-121 cache: {path}")
    return data


def run(root, stage121, plan, jobs, models, entry_for, history_length, source_plan, scope, metrics):
    tables = {k: [] for k in TABLES}
    for (sequence, seed), arms in sorted(jobs.items()):
        job = arms[CONTROL]
        _, scorer = base.job_models(job, models)
        ident = dict(cohort=job["spec"]["cohort"], sequence=sequence, seed=seed, fold=job["outer_fold"])
        entry = entry_for(job["spec"])
        if entry["sequence"] != sequence or len(entry["frames"]) != job["frames"] + 1:
            raise ValueError("Stage-121 sequence/length differs from labeled data")
        data = [load_trace(stage121, source_plan, arms[arm]) for arm in (CONTROL, source.TREATMENT)]
        row, activations, scores, swaps = diagnose(*data, entry, scorer, history_length, ident)
        check_source_row(row, scope, metrics)
        tables["trajectories"].append(row)
        tables["activations"].extend(activations)
        tables["retained_scores"].extend(scores)
        tables["relation_swaps"].extend(swaps)
        print(f"[Stage-122] {sequence} s{seed}: {row['matched_activations']}/{row['activations']} matched", flush=True)
    publish(root, plan, tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage112", "stage121")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all(getattr(args, name) for name in names):
        parser.error("Supply Stage-76/89/93/112/121 directories")
    r76, r89, r93, r112, r121 = paths = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Choose a separate Stage-122 directory")
    if not (r121 / "manifest.json").is_file():
        parser.error(f"Stage-121 manifest not found: {r121 / 'manifest.json'}")
    with base.s109.s106.s93.writer_lock(root):
        if (root / "analysis/stage122_decision.json").exists():
            check_results(root)
            return
        source.check_results(r121)
        p121 = c.c66.sealed(r121 / "manifest.json", source.SCHEMA)
        r121_report = c.c66.sealed(r121 / "analysis/stage121_decision.json", source.SCHEMA)
        if r121_report["fingerprint"] != SOURCE121 or p121["reuse_controls"]:
            raise ValueError("Need the completed Stage-121 L40 run with same-card controls")
        p112 = c.c66.sealed(r112 / "manifest.json", base.common.SCHEMA)
        model_files = {str(p): p121["source_sha256"][str(p)] for p in (
            [r112 / "manifest.json"] + sorted((r112 / "models").glob("*__shared__*.json"))
            + sorted((r112 / "models").glob("generator_*.json")))}
        io.verify(model_files)
        models, _ = base.load_frozen_models(r112, p112)
        s94 = base.s103.prior.s95.s94
        _, _, p89, report89, origin = s94.load_run(r93, r76, r89)

        def entry_for(spec):
            seq = spec["unit"]["sequence"]
            if spec["cohort"] == "historical":
                return c.entry_for(origin["original"], seq)
            if spec["cohort"] != "expanded":
                raise ValueError("Unknown development cohort")
            path = f"entries/{seq}.json"
            io.verify({str(r89 / path): report89["evidence_sha256"][path]})
            metadata = c.c66.sealed(r89 / path, s94.s89.SCHEMA)
            if metadata["plan_fingerprint"] != p89["fingerprint"]:
                raise ValueError("Stage-89 entry changed")
            return metadata["entry"]

        jobs = defaultdict(dict)
        for job in p121["jobs"]:
            if job["arm"] in (CONTROL, source.TREATMENT):
                jobs[job["spec"]["unit"]["sequence"], job["seed"]][job["arm"]] = job
        sequences = {sequence for sequence, _ in jobs}
        if set(jobs) != {(sequence, seed) for sequence in sequences for seed in base.DESIGN["seeds"]} or len(sequences) != 28:
            raise ValueError("Expected all 28 development videos and both seeds")
        if any(set(arms) != {CONTROL, source.TREATMENT}
               or dict(arms[source.TREATMENT], arm=CONTROL) != arms[CONTROL] for arms in jobs.values()):
            raise ValueError("Expected matched Stage-121 trajectory identities")
        scope, metrics = (source_rows(r121, name) for name in ("scope", "seed_video"))
        activations = sum(int(r["alternate_used_frames"]) for r in scope.values())
        inputs = p121["source_sha256"] | p121["code_sha256"] | model_files
        for arms in jobs.values():
            for job in arms.values():
                for ext in ("npz", "receipt.json"):
                    key = f"evaluation/{base.job_key(job)}.{ext}"
                    inputs[str(r121 / key)] = r121_report["artifact_sha256"][key]
        bound = (r121 / "manifest.json", r121 / "analysis/stage121_decision.json", Path(__file__).resolve(),
                 Path(__file__).with_name("run_stage122_alternate_diagnostic.sh").resolve())
        inputs.update({str(p):c.file_hash(p) for p in bound})
        inputs.update({str(r121 / "analysis" / key):value for key, value in r121_report["table_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage121_fingerprint=SOURCE121,
                            source_plan_fingerprint=p121["fingerprint"], pairs=len(jobs),
                            activations=activations, inputs=inputs))
        c.publish_identical(root / "manifest.json", plan)
        length = base.e63.c.actor_plan(origin["original"])["design"]["history"]
        run(root, r121, plan, jobs, models, entry_for, length, p121, scope, metrics)


if __name__ == "__main__":
    main()
