"""Evaluate crop agreement by selecting among sealed Stage-186 continuations."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
from stage184_center_mapping import cache_reference
from stage185_writeback_components import LAYER
import stage186_crop_quantization as prior
from stage28_response_common import continuous_iou


c, gain, io, risk = prior.c, prior.gain, prior.io, prior.risk
SOURCE186 = "318e1a59f5322d3458f59c5b3fca8bb1240d7d648db7301e8ebbdedda0cb5741"
SCHEMA = "stage187_crop_consensus_v1"
POOL = prior.ARMS[1:]
METHODS = ("current", "highest_hann", "crop_consensus")
LONG = prior.source.LONG


def bind(read):
    cases = [dict(r, frame=int(r["frame"]), end=int(r["end"]), earlier=int(r["earlier"]),
                  prior_event=r["prior_event"] == "True", selected=r["selected"] == "True") for r in read("cases")]
    probe_rows, outcome_rows = read("probes"), read("windows")
    probes = {(r["case_id"], r["arm"]): r for r in probe_rows}
    outcomes = {(r["case_id"], r["arm"]): r for r in outcome_rows}
    expected = {(p["id"], arm) for p in cases
                for arm in (*prior.ARMS, "size_control") if arm != "size_control" or p["id"] == prior.SCALE_POINT}
    if (len(cases) != 27 or sum(p["cohort"] == "stable" for p in cases) != 8
            or len(probes) != len(probe_rows) or len(outcomes) != len(outcome_rows)
            or set(probes) != expected or set(outcomes) != expected):
        raise ValueError("Expected all Stage-186 cases and matching observed branches")
    for p in cases:
        key = p["id"]
        for arm in (*prior.ARMS, "size_control") if key == prior.SCALE_POINT else prior.ARMS:
            x, z = probes[key, arm], outcomes[key, arm]
            if (int(x["frame"]) != p["frame"] + 1 or int(z["successors"]) != p["end"] - p["frame"]
                    or any(x[k] != p[k] or z[k] != p[k] for k in ("group", "sequence", "cohort"))):
                raise ValueError(f"Stage-186 outcome metadata differs at {key} / {arm}")
    return cases, probes, outcomes


def candidate_pool(case, probes):
    crops = defaultdict(list)
    for arm in POOL:
        row = probes[case["id"], arm]
        rectangle = tuple(int(row[k]) for k in ("crop_left", "crop_top", "crop_side"))
        crops[rectangle].append(arm)

    def preference(a):
        r = probes[case["id"], a]
        return math.hypot(float(r["dx"]), float(r["dy"])), POOL.index(a)

    groups = {min(arms, key=preference): arms for arms in crops.values()}
    if "current" not in groups or len({int(probes[case["id"], a]["crop_side"]) for a in groups}) != 1:
        raise ValueError("Current crop missing, or fixed-size candidate pool changed")
    return groups


def rank_arms(pool, probes, key, boxes=None):
    """Scores use image-coordinate predictions; targets and outcomes stay outside selection."""
    names = list(pool)
    peaks = {a: float(probes[key, a]["hann_peak"]) for a in names}
    if not all(np.isfinite(v) and v >= 0. for v in peaks.values()):
        raise ValueError("Invalid sealed Hann peaks")

    def tie(a):
        r = probes[key, a]
        return a == "current", -math.hypot(float(r["dx"]), float(r["dy"])), -POOL.index(a)

    selection = dict(current="current", highest_hann=max(names, key=lambda a: (peaks[a], *tie(a))))
    if boxes is None:
        return selection, {}
    image_boxes = np.asarray([boxes[a] for a in names], float)
    if image_boxes.shape != (len(names), 4) or not np.isfinite(image_boxes).all() or (image_boxes[:, 2:] <= 0.).any():
        raise ValueError("Expected finite positive image-coordinate candidate boxes")
    overlaps = np.asarray([continuous_iou(box, image_boxes) for box in image_boxes])
    np.fill_diagonal(overlaps, 0.)
    scores = dict(zip(names, map(float, overlaps.sum(axis=1) / max(len(names) - 1, 1)), strict=True))
    selection["crop_consensus"] = max(names, key=lambda a: (
        scores[a], a == "current", peaks[a], *tie(a)[1:]))
    return selection, scores


def label_choice(case, method, arm, probes, outcomes, pool, score):
    key = case["id"]
    r, selection = outcomes[key, arm], probes[key, arm]
    valid = r["valid"] == "True"
    current = outcomes[key, "current"]
    delta = float(r["delta128_or_shorter"]) if valid else None
    return dict(case_id=key, sequence=case["sequence"], cohort=case["cohort"], group=case["group"],
                method=method, arm=arm, candidates=len(pool), agreement=score, selected_hann=float(selection["hann_peak"]),
                anchor_iou=float(selection["output_iou"]), anchor_delta=float(selection["output_iou"]) - float(probes[key, "current"]["output_iou"]),
                successors=int(r["successors"]), valid=valid, tail_delta_vs_native=delta,
                tail_delta_vs_current=delta - float(current["delta128_or_shorter"]) if valid and current["valid"] == "True" else None,
                protected_rescue=r["protected_rescue"] == "True" if valid else None,
                event_harm=r["event_harm"] == "True" if valid else None,
                missed_current_rescue=(current["protected_rescue"] == "True" and r["protected_rescue"] != "True") if valid else None,
                harm_absent_in_current=(r["event_harm"] == "True" and current["event_harm"] != "True") if valid else None)


def read_boxes(root, receipt, case, probes):
    key = case["id"]
    kwargs = dict(plan_fingerprint=receipt["plan_fingerprint"], case_id=key)
    probe = cache_reference(root, receipt, f"probe/{key}.npz", dict(kwargs, kind="probe"))
    tail = cache_reference(root, receipt, f"tail/{key}.npz", dict(kwargs, kind="tail"))
    arms = (*prior.ARMS, "size_control") if key == prior.SCALE_POINT else prior.ARMS
    boxes = probe["boxes"]
    indices, paths = tail["inverse"], tail["paths"]
    if (boxes.shape != (len(arms), 4) or indices.shape != (len(arms),)
            or paths.shape[1:] != (case["end"] - case["frame"], 4)
            or not np.issubdtype(indices.dtype, np.integer) or not ((indices >= 0) & (indices < len(paths))).all()):
        raise ValueError(f"Invalid Stage-186 stored probe or continuation at {key}")
    np.testing.assert_array_equal(paths[indices, 0], boxes)
    np.testing.assert_array_equal(probe["layer"], np.full(len(arms), LAYER, int))
    np.testing.assert_array_equal(probe["hann_peak"], [float(probes[key, a]["hann_peak"]) for a in arms])
    for i, arm in enumerate(arms):
        row = probes[key, arm]
        observed = probe["parent"][i, :2] - probe["parent"][1, :2]
        np.testing.assert_array_equal(observed, [float(row["dx"]), float(row["dy"])])
    return dict(zip(arms, boxes, strict=True))


def report_case(case, probes, outcomes, boxes):
    key = case["id"]
    pool = candidate_pool(case, probes)
    selection, scores = rank_arms(pool, probes, key, boxes)
    candidates = []
    for arm, aliases in pool.items():
        row = probes[key, arm]
        x, y, w, h = map(float, boxes[arm])
        candidates.append(dict(case_id=key, cohort=case["cohort"], group=case["group"], arm=arm, aliases="|".join(aliases),
                               crop_left=int(row["crop_left"]), crop_top=int(row["crop_top"]), crop_side=int(row["crop_side"]),
                               box_x=x, box_y=y, box_width=w, box_height=h, hann_peak=float(row["hann_peak"]),
                               average_peer_iou=scores[arm], anchor_iou=float(row["output_iou"])))
    choices = [label_choice(case, method, arm, probes, outcomes, pool, scores[arm]) for method, arm in selection.items()]
    raw_rescues = [a for a in POOL if outcomes[key, a]["protected_rescue"] == "True"]
    available = [a for a in raw_rescues if a in pool]
    scope = dict(case_id=key, cohort=case["cohort"], group=case["group"], candidates=len(pool),
                 removed_duplicate_arms=len(POOL) - len(pool), available_rescue=bool(available),
                 available_rescue_arms="|".join(available), raw_rescue=bool(raw_rescues),
                 discarded_rescue_without_canonical_rescue=bool(raw_rescues and not available),
                 no_geometric_support=max(scores.values()) == 0.,
                 agreement_with_current=selection["crop_consensus"] == "current")
    return candidates, choices, scope


def summarize(cases, choices, support):
    by_arm = {(r["case_id"], r["method"]): r for r in choices}
    scope_by_id = {r["case_id"]: r for r in support}
    tables = []
    paired = []
    for cohort in ("supply_loss", "stable", "supply_loss_without_previous_event"):
        for length in ("all", "full128"):
            sample = [r for r in cases if (r["cohort"] == cohort or cohort == "supply_loss_without_previous_event"
                      and r["cohort"] == "supply_loss" and not r["prior_event"])
                      and (length == "all" or r["end"] - r["frame"] == LONG)
                      and all(by_arm[r["id"], m]["valid"] for m in METHODS)]
            for method in METHODS:
                selected = [by_arm[p["id"], method] for p in sample]
                opportunities = sum(scope_by_id[p["id"]]["available_rescue"] for p in sample)
                rescues = sum(p["protected_rescue"] for p in selected)
                tables.append(dict(cohort=cohort, length=length, method=method, windows=len(selected),
                                   sources=len({r["group"] for r in selected}), available_rescues=opportunities,
                                   selected_rescues=rescues, rescue_capture=rescues / opportunities if opportunities else None,
                                   rescue_groups=len({r["group"] for r in selected if r["protected_rescue"]}),
                                   event_harm_windows=sum(r["event_harm"] for r in selected),
                                   event_harm_groups=len({r["group"] for r in selected if r["event_harm"]}),
                                   missed_current_rescues=sum(r["missed_current_rescue"] for r in selected),
                                   harm_absent_in_current=sum(r["harm_absent_in_current"] for r in selected),
                                   changed_from_current=sum(r["arm"] != "current" for r in selected)))
            for treatment, control in (("crop_consensus", "current"), ("highest_hann", "current"),
                                       ("crop_consensus", "highest_hann")):
                for metric in ("anchor_iou", "tail_delta_vs_native"):
                    group_values = defaultdict(list)
                    for p in sample:
                        a, b = (by_arm[p["id"], m] for m in (treatment, control))
                        group_values[p["group"]].append(a[metric] - b[metric])
                    means = [float(np.mean(x)) for x in group_values.values()]
                    paired.append(dict(cohort=cohort, length=length, treatment=treatment, control=control,
                                       metric=metric, windows=len(sample), groups=len(means),
                                       mean_group_delta=float(np.mean(means)) if means else None,
                                       median_group_delta=float(np.median(means)) if means else None,
                                       mean_without_best_group=float(np.mean(sorted(means)[:-1])) if len(means) > 1 else None,
                                       wins=sum(x > 1e-12 for x in means), losses=sum(x < -1e-12 for x in means),
                                       ties=sum(abs(x) <= 1e-12 for x in means)))
    return tables, paired


def preflight(bundle):
    _, tables = gain.prior.archives.bundle(bundle, 186, SOURCE186)
    cases, probes, outcomes = bind(lambda k: tables[f"stage186_{k}.csv"])
    groups = [candidate_pool(p, probes) for p in cases]
    print(f"Verified {len(cases)} cached windows; {sum(map(len, groups))} distinct eligible crops "
          f"from {len(POOL) * len(cases)} eligible arms")
    for study, cohort, full in (("Complete supply-loss", "supply_loss", True), ("Stable", "stable", False)):
        picks = [(p, pool, rank_arms(pool, probes, p["id"])[0])
                 for p, pool in zip(cases, groups, strict=True) if p["cohort"] == cohort
                 and (not full or p["end"] - p["frame"] == LONG)]
        opportunities = sum(any(outcomes[p["id"], a]["protected_rescue"] == "True" for a in pool) for p, pool, _ in picks)
        print(f"{study}: {len(picks)} windows, {opportunities} with a protected-rescue candidate")
        for method in ("current", "highest_hann"):
            selected = [outcomes[p["id"], choose[method]] for p, _, choose in picks]
            print(f"  {method}: {sum(r['protected_rescue'] == 'True' for r in selected)} rescues, "
                  f"{sum(r['event_harm'] == 'True' for r in selected)} event harms")


def run(root, r186):
    receipt = c.c66.sealed(r186 / "stage186_decision.json", prior.SCHEMA)
    manifest = c.c66.sealed(r186 / "manifest.json", prior.SCHEMA)
    if (receipt["fingerprint"] != SOURCE186 or receipt["plan_fingerprint"] != manifest["fingerprint"]
            or receipt["source_stage185"] != prior.SOURCE185 or receipt["runtime"] != manifest["runtime"]):
        raise ValueError("Need sealed Stage-186 probes and continuations")
    cases, probes, outcomes = bind(lambda k: risk.read_verified(r186, receipt, 186, k))
    if cases != manifest["cases"]:
        raise ValueError("Stage-186 planned and measured cases differ")
    code = (Path(__file__), Path(continuous_iou.__code__.co_filename))
    plan = io.seal(dict(schema=SCHEMA, source_stage186=SOURCE186, cases=cases, pool=list(POOL),
                        methods=list(METHODS), ranker="mean_pairwise_image_box_iou",
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    progress = OverallProgress("Stage-187", len(cases) + 1, unit="work")
    progress.report("Scoring verified cached crops and paired continuations")
    candidates, choices, support = [], [], []
    for case in cases:
        boxes = read_boxes(r186, receipt, case, probes)
        pool, picked, scope = report_case(case, probes, outcomes, boxes)
        candidates.extend(pool)
        choices.extend(picked)
        support.append(scope)
        progress.advance()
    summary, paired = summarize(cases, choices, support)
    tables = dict(cases=cases, candidates=candidates, support=support, choices=choices, summary=summary, paired=paired)
    return dict(schema=SCHEMA, status="COMPLETE_STAGE187_CROP_CONSENSUS_DIAGNOSTIC", source_stage186=SOURCE186,
                plan_fingerprint=plan["fingerprint"], cases=len(cases), candidate_boxes=len(candidates),
                cached_choices=len(choices), stable_controls=sum(p["cohort"] == "stable" for p in cases),
                new_searches=0, training_performed=False, work=progress.total,
                table_sha256=gain.prior.publish(root, tables, stage="stage187")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Verify the export and compute baseline counts without box caches")
    parser.add_argument("--stage186-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bundle:
        if args.stage186_root or args.output_dir:
            parser.error("--bundle is a standalone CPU check")
        preflight(args.bundle)
        return
    if args.stage186_root is None or args.output_dir is None:
        parser.error("Need the complete Stage-186 directory and a separate output directory")
    upstream, root = (p.expanduser().resolve() for p in (args.stage186_root, args.output_dir))
    if root == upstream or root in upstream.parents or upstream in root.parents:
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        decision = root / "stage187_decision.json"
        if decision.exists():
            saved = c.c66.sealed(decision, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage186"] != SOURCE186 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-187 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in saved["table_sha256"].items()})
            OverallProgress("Stage-187", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(root, upstream)
        c.publish_identical(decision, io.seal(result))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
