"""Compare full rescoring with frozen retained scores on the same alternate bank."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

import stage122_alternate_diagnostic as prior
from stage27_uav_common import tracking_iou

s121, base, c, io = prior.source, prior.base, prior.c, prior.io
SCHEMA = "stage123_frozen_retained_scores_v1"
STATUS = "COMPLETE_FROZEN_RETAINED_SCORE_DIAGNOSTIC"
SOURCE122 = "6ddf479079eab2e4ffbf439b26230ca4a5e1b54d0247218a069450801dc75a5f"
TABLES = ("summary", "paired", "video_effects", "decisions")
DESIGN = dict(points=80, added_good_points=9, bootstrap_seed=23123, bootstrap_samples=20000,
              good_iou=.5, low_iou=.2)
WORK = dict(prior.WORK, new_full_trajectories=0)


def compare_scores(old_scores, new_scores, boxes):
    frozen = np.asarray(new_scores, float).copy()
    frozen[:prior.SLOT] = old_scores[:prior.SLOT]
    choose = base.common.physical_choice
    return choose(new_scores, boxes), choose(frozen, boxes), frozen


def compare_point(ref, alt, entry, length, event, retained_scores):
    frame = int(event["frame"])
    i = frame - 1
    if not 0 <= i < len(alt["frame_id"]) or not alt["alternate_used"][i]:
        raise ValueError("Stage-122 point does not match a recorded substitution")
    np.testing.assert_equal(ref["frame_id"][i], frame)
    np.testing.assert_equal(alt["frame_id"][i], frame)
    histories = [prior.history(data, entry["initial_box"], i, length) for data in (ref, alt)]
    if not prior.same_input(ref, alt, i, *histories) or not prior.matching_observations(ref, alt, i):
        raise ValueError("Stage-122 matched context changed")
    if np.array_equal(ref["crop"][i, prior.SLOT], alt["crop"][i, prior.SLOT]):
        raise ValueError("No replacement crop at matched point")

    old_scores, new_scores = ref["scores"][i], alt["scores"][i]
    boxes = alt["candidate_boxes"][i]
    for q, row in enumerate(retained_scores):
        np.testing.assert_equal(q, int(row["slot"]))
        np.testing.assert_allclose([old_scores[q], new_scores[q]],
                                   [float(row["old_score"]), float(row["new_score"])], rtol=1e-9, atol=1e-12)
    full, frozen, hybrid = compare_scores(old_scores, new_scores, boxes)
    np.testing.assert_equal(full, int(alt["selected"][i]))
    np.testing.assert_equal(full, int(event["alternate_slot"]))
    old_selected = int(ref["selected"][i])
    np.testing.assert_equal(old_selected, int(event["control_slot"]))

    gt = np.asarray(entry["ground_truth"][frame], float)
    original_iou, iou = [tracking_iou(data["candidate_boxes"][i], gt) for data in (ref, alt)]
    if (original_iou < 0).any() or (iou < 0).any():
        raise ValueError("Unusable diagnostic ground truth")
    good, low = DESIGN["good_iou"], DESIGN["low_iou"]
    added = bool(original_iou.max() < good <= iou.max())
    if added != (event["net_good_bank_added"] == "True"):
        raise ValueError("Stage-122 good-candidate label changed")
    np.testing.assert_allclose([original_iou[old_selected], iou[full], iou[prior.SLOT]],
                               [float(event[k]) for k in ("control_iou", "alternate_iou", "alternate_slot4_iou")],
                               rtol=0, atol=1e-12)
    return dict(cohort=event["cohort"], sequence=event["sequence"], seed=int(event["seed"]),
                fold=int(event["fold"]), frame=frame, stage122_mechanism=event["mechanism"],
                first_output_difference=event["first_output_difference"] == "True",
                net_good_bank_added=added, full_slot=full, frozen_slot=frozen,
                physical_choice_changed=not np.array_equal(boxes[full], boxes[frozen]),
                full_iou=float(iou[full]), frozen_iou=float(iou[frozen]), iou_delta=float(iou[frozen]-iou[full]),
                good_gained=bool(iou[full] < good <= iou[frozen]),
                good_lost=bool(iou[frozen] < good <= iou[full]),
                severe_loss=bool(iou[frozen] < low and iou[full] >= good),
                full_scores="|".join(map(str, new_scores)), frozen_scores="|".join(map(str, hybrid)),
                candidate_ious="|".join(map(str, iou)))


def summarize(decisions):
    populations = {"all": decisions, "new_good_added": [r for r in decisions if r["net_good_bank_added"]],
                   "rest": [r for r in decisions if not r["net_good_bank_added"]]}
    summary = []
    for name, rows in populations.items():
        summary.append(dict(population=name, points=len(rows), videos=len({r["sequence"] for r in rows}),
                            choice_changes=sum(r["physical_choice_changed"] for r in rows),
                            full_alternate_selections=sum(r["full_slot"] == prior.SLOT for r in rows),
                            frozen_alternate_selections=sum(r["frozen_slot"] == prior.SLOT for r in rows),
                            full_good=sum(r["full_iou"] >= DESIGN["good_iou"] for r in rows),
                            frozen_good=sum(r["frozen_iou"] >= DESIGN["good_iou"] for r in rows),
                            good_gained=sum(r["good_gained"] for r in rows), good_lost=sum(r["good_lost"] for r in rows),
                            severe_losses=sum(r["severe_loss"] for r in rows),
                            full_mean_iou=float(np.mean([r["full_iou"] for r in rows])) if rows else None,
                            frozen_mean_iou=float(np.mean([r["frozen_iou"] for r in rows])) if rows else None))
    videos = defaultdict(lambda: defaultdict(list))
    for row in decisions:
        videos[row["sequence"]][row["seed"]].append(row)
    effects = []
    for seq, groups in sorted(videos.items()):
        means = {metric:float(np.mean([np.mean([r[metric] for r in group]) for group in groups.values()]))
                 for metric in ("full_iou", "frozen_iou", "iou_delta")}
        effects.append(dict(sequence=seq, seeds=len(groups), points=sum(map(len, groups.values())), **means))
    paired = [base.s103.contrast([r["iou_delta"] for r in effects], "frozen_minus_full", "iou", DESIGN)]
    return dict(summary=summary, paired=paired, video_effects=effects, decisions=decisions)


def read_table(root, name):
    with (root / "analysis" / f"stage122_{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def get_event_key(row):
    return row["sequence"], int(row["seed"]), int(row["frame"])


def load_support(root):
    events = {get_event_key(r):r for r in read_table(root, "activations") if r["matched"] == "True"}
    grouped_scores = defaultdict(list)
    for row in read_table(root, "retained_scores"):
        grouped_scores[get_event_key(row)].append(row)
    if (set(events) != set(grouped_scores) or len(events) != DESIGN["points"]
            or sum(row["net_good_bank_added"] == "True" for row in events.values()) != DESIGN["added_good_points"]):
        raise ValueError("Stage-122 matched support differs")
    grouped_events = defaultdict(list)
    for key, event in sorted(events.items()):
        scores = sorted(grouped_scores[key], key=lambda r: int(r["slot"]))
        if [int(r["slot"]) for r in scores] != list(range(prior.SLOT)):
            raise ValueError("Stage-122 retained score support differs")
        grouped_events[key[:2]].append((event, scores))
    return grouped_events


def read_entries(r76, r89, r93):
    s94 = base.s103.prior.s95.s94
    _, _, stage89_plan, report89, origin = s94.load_run(r93, r76, r89)

    def entry_for(spec):
        name = spec["unit"]["sequence"]
        if spec["cohort"] == "historical":
            return c.entry_for(origin["original"], name)
        if spec["cohort"] != "expanded":
            raise ValueError("Unexpected development cohort")
        relative = f"entries/{name}.json"
        io.verify({str(r89 / relative):report89["evidence_sha256"][relative]})
        entry = c.c66.sealed(r89 / relative, s94.s89.SCHEMA)
        if entry["plan_fingerprint"] != stage89_plan["fingerprint"]:
            raise ValueError("Stage-89 entry changed")
        return entry["entry"]

    return entry_for, base.e63.c.actor_plan(origin["original"])["design"]["history"]


def run(r121, p121, support, entry_for, length):
    jobs = {(j["spec"]["unit"]["sequence"], j["seed"], j["arm"]):j for j in p121["jobs"]}
    decisions = []
    for (sequence, seed), points in support.items():
        ref_job, alt_job = [jobs[sequence, seed, arm] for arm in (prior.CONTROL, s121.TREATMENT)]
        if dict(alt_job, arm=prior.CONTROL) != ref_job:
            raise ValueError("Need matched Stage-121 trajectory identities")
        entry = entry_for(ref_job["spec"])
        if entry["sequence"] != sequence or len(entry["ground_truth"]) != ref_job["frames"] + 1:
            raise ValueError("Stage-121 data binding differs")
        ref, alt = [prior.load_trace(r121, p121, job) for job in (ref_job, alt_job)]
        for event, scores in points:
            if int(event["fold"]) != ref_job["outer_fold"] or event["cohort"] != ref_job["spec"]["cohort"]:
                raise ValueError("Stage-122 event identity changed")
            decisions.append(compare_point(ref, alt, entry, length, event, scores))
    return summarize(decisions)


def publish(root, plan, tables):
    if len(tables["decisions"]) != DESIGN["points"] or tables["summary"][1]["points"] != DESIGN["added_good_points"]:
        raise ValueError("Incomplete fixed cohort")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for key in TABLES:
        path = out / f"stage123_{key}.csv"
        base.s103.s86.publish_table(path, tables[key])
        hashes[path.name] = c.file_hash(path)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, source_stage122_fingerprint=SOURCE122,
                         plan_fingerprint=plan["fingerprint"], row_counts={k:len(v) for k,v in tables.items()},
                         primary=tables["paired"][0], added_good=tables["summary"][1],
                         all_points=tables["summary"][0], table_sha256=hashes, **WORK))
    c.publish_identical(out / "stage123_decision.json", report)
    check_results(root)
    print(f"Stage-123: {STATUS}. Added good candidates selected: "
          f"{report['added_good']['full_good']} -> {report['added_good']['frozen_good']}/{DESIGN['added_good_points']}",
          flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage123_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or plan["source_stage122_fingerprint"] != SOURCE122
            or report["source_stage122_fingerprint"] != SOURCE122 or report["status"] != STATUS
            or report["plan_fingerprint"] != plan["fingerprint"] or report["row_counts"]["decisions"] != DESIGN["points"]
            or set(report["table_sha256"]) != {f"stage123_{k}.csv" for k in TABLES}
            or any(report[k] != v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-123 diagnostic")
    io.verify(plan["inputs"])
    io.verify({str(root / "analysis" / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage122")
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
        parser.error("Supply Stage-76/89/93/121/122 directories")
    r76, r89, r93, r121, r122 = paths = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Choose a separate Stage-123 directory")
    if not (r122 / "manifest.json").is_file() or not (r121 / "manifest.json").is_file():
        parser.error("Need Stage-121 and Stage-122 server manifests and paired Stage-121 NPZ caches")
    with base.s109.s106.s93.writer_lock(root):
        if (root / "analysis/stage123_decision.json").exists():
            check_results(root)
            return
        prior.check_results(r122)
        p122 = c.c66.sealed(r122 / "manifest.json", prior.SCHEMA)
        report122 = c.c66.sealed(r122 / "analysis/stage122_decision.json", prior.SCHEMA)
        p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
        if report122["fingerprint"] != SOURCE122 or p121["fingerprint"] != p122["source_plan_fingerprint"]:
            raise ValueError("Need the archived Stage-122 support and its same-card Stage-121 source")
        support = load_support(r122)
        entry_for, length = read_entries(r76, r89, r93)
        inputs = {str(r121 / "manifest.json"):p122["inputs"][str(r121 / "manifest.json")]}
        for seq, seed in support:
            for arm in (prior.CONTROL, s121.TREATMENT):
                for ext in ("npz", "receipt.json"):
                    path = str(r121 / "evaluation" / f"{seq}__{arm}__s{seed}.{ext}")
                    inputs[path] = p122["inputs"][path]
        for path in (r122 / "manifest.json", r122 / "analysis/stage122_decision.json",
                     Path(__file__).resolve(), Path(__file__).with_name("run_stage123_frozen_retained_scores.sh").resolve()):
            inputs[str(path)] = c.file_hash(path)
        inputs.update({str(r122 / "analysis" / k):v for k,v in report122["table_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage122_fingerprint=SOURCE122,
                            source_stage121_plan=p121["fingerprint"], inputs=inputs))
        c.publish_identical(root / "manifest.json", plan)
        publish(root, plan, run(r121, p121, support, entry_for, length))


if __name__ == "__main__":
    main()
