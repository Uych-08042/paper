"""Evaluate causal reopening scores where the frozen gate proposes the native layer."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage176_layer_event_sweep as sweep


event, risk, aligned, online, previous = sweep.event, sweep.risk, sweep.aligned, sweep.online, sweep.previous
base, gate, gain, history, c, io = sweep.base, sweep.gate, sweep.gain, sweep.history, sweep.c, sweep.io
SOURCE176 = "7eb638828bb53591223c9d30beb5d38a3c23d0a26aee3d1a07f3ca700108ac57"
SCHEMA = "stage177_native_proposal_audit_v1"
SOURCES = (*sweep.SOURCES, 176)
REFERENCES = {**sweep.REFERENCES, 176: (sweep.SCHEMA, SOURCE176)}
POLICIES = ("uniform", "selector_margin", "response_history", "joint")
HISTORY, SHORT, LONG = event.HISTORY, event.SHORT, event.LONG
SCAN, QUOTA, SEED = 32, 2, 23177
EPSILON = sweep.EPSILON


def archive_scope(read):
    roster = {r["sequence"]: int(r["frames"]) for r in read(170, "video_summary") if r["arm"] == event.matched.COLD}
    designed = read(176, "cases")
    if len(roster) != online.DESIGN["videos"] or len(designed) != 26:
        raise ValueError("Incomplete Stage-170 or Stage-176 roster")
    used_groups = {r["group"] for r in designed}
    held = {s: n for s, n in roster.items() if risk.SOURCE_GROUP(s) not in used_groups}
    reserved = {(r["sequence"], int(r["frame"])) for stage in (173, 174, 175, 176) for r in read(stage, "cases")}
    cases175 = {r["id"]: r for r in read(175, "cases")}
    watch = {r["id"] for r in designed if r["archived_rescue"] == "True" and int(r["stage"]) == 175}
    watch.add("uav0000348_02415_s__f000362")
    sentinels = []
    for key in sorted(watch):
        r = cases175[key]
        if r["route"] != "native_proposal":
            raise ValueError("A Stage-177 sentinel did not originally propose the native layer")
        sentinels.append(dict(id=key, sequence=r["sequence"], group=r["group"], frame=int(r["frame"]),
                              end=int(r["end"]), source=r["source"], cohort="sentinel"))
    return held, reserved, sentinels


def pool(held, sources, reserved, sentinels):
    points, support = [], []
    for seq, size in sorted(held.items()):
        source = sources[seq, event.matched.COLD]
        mandatory = np.flatnonzero(source["proposal"] == source["native_layer"]) + 1
        eligible = [int(f) for f in mandatory if HISTORY < f <= size - LONG and (seq, int(f)) not in reserved]
        picked = [min(map(int, bucket), key=lambda f: sha256(f"{SEED}/{seq}/{f}".encode()).digest())
                  for bucket in np.array_split(eligible, min(len(eligible), SCAN))] if eligible else []
        group = risk.SOURCE_GROUP(seq)
        points.extend(dict(id=f"{seq}__f{f:06d}", sequence=seq, group=group, frame=f, end=f + LONG,
                           source=event.matched.COLD, cohort="held") for f in picked)
        support.append(dict(sequence=seq, group=group, frames=size, native_proposal_frames=len(mandatory),
                            eligible=len(eligible), scanned=len(picked)))
    points.extend(sentinels)
    if len({r["id"] for r in points}) != len(points):
        raise ValueError("Duplicate Stage-177 scan point")
    return points, support


class ReopenObserver(risk.RiskObserver):
    def route(self, module, args, output):
        routed = super().route(module, args, output)
        if self.arm != "native":
            ordered = np.sort(output.detach().cpu().numpy().reshape(-1).astype(float))
            if ordered.shape != (len(online.LAYERS),) or not np.isfinite(ordered).all():
                raise ValueError("Invalid native selector output")
            self.diagnostic["selector_gap"] = float(ordered[-1] - ordered[-2])
        return routed


def scan_row(case, trace, source, exact_runtime):
    i = case["frame"] - 1
    native, proposal, ranked = (int(trace[f"anchor_{name}"]) for name in ("native", "proposal", "ranked"))
    if source["proposal"][i] != source["native_layer"][i] or source["switched"][i]:
        raise ValueError(f"Scan point did not originally keep the native layer: {case['id']}")
    error = float(np.abs(trace["boxes"][0] - source["boxes"][i]).max())
    gate_error = abs(float(trace["anchor_legacy"]) - float(source["gate_value"][i]))
    if exact_runtime and (error != 0. or gate_error > 1e-12 or native != source["native_layer"][i] or proposal != native):
        raise ValueError(f"Stage-170 native scan diverged at {case['id']}")
    p = np.asarray(source["hann_peak"][i - HISTORY:i], float)
    if len(p) != HISTORY or not np.isfinite(p).all():
        raise ValueError(f"Missing causal response history at {case['id']}")
    recent, earlier = float(p[-4:].mean()), float(p[:-4].mean())
    weak_or_falling = 1. - recent + max(0., earlier - recent)
    return dict(case_id=case["id"], sequence=case["sequence"], group=case["group"], cohort=case["cohort"],
                eligible=proposal == native, native=native, proposal=proposal, ranked=ranked,
                selector_gap=float(trace["anchor_selector_gap"]), recent_hann=recent, earlier_hann=earlier,
                response_difficulty=weak_or_falling, source_box_error=error, source_gate_error=gate_error,
                in_cooldown=bool(source["in_cooldown"][i]))


def average_ranks(values):
    _, bucket, sizes = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(sizes) - (sizes + 1.) / 2.)[bucket]


def select(scans):
    groups = defaultdict(list)
    for row in scans:
        if row["eligible"] and row["cohort"] == "held":
            groups[row["group"]].append(row)
    choices, support = [], []
    for group, rows in sorted(groups.items()):
        if len(rows) < QUOTA:
            support.append(dict(group=group, eligible=len(rows), quota=0, distinct_replays=0))
            continue
        # Pair the allocation counts by source; percentile ranks form an offline quota comparison.
        ids = [r["case_id"] for r in rows]
        identity = [sha256(f"{SEED}/uniform/{k}".encode()).hexdigest() for k in ids]
        margin = average_ranks([r["selector_gap"] for r in rows])
        response = average_ranks([-r["response_difficulty"] for r in rows])
        picked = set()
        for policy, scores in zip(POLICIES, (identity, margin, response, (margin + response) / 2.), strict=True):
            order = sorted(range(len(ids)), key=lambda j: (scores[j], identity[j]))
            for position, j in enumerate(order[:QUOTA], 1):
                choices.append(dict(case_id=ids[j], group=group, policy=policy, position=position,
                                    selector_rank=float(margin[j]), response_rank=float(response[j])))
                picked.add(ids[j])
        support.append(dict(group=group, eligible=len(rows), quota=QUOTA, distinct_replays=len(picked)))
    return choices, support


def report(case, traces, truth, prefix, source, exact_runtime, scan):
    d = event.measure(case, traces, truth, prefix, source, exact_runtime)
    # Ensure the scored anchor and the replays refer to the same causal state and fixed candidate.
    for k in ("native", "proposal", "ranked", "selector_gap"):
        if float(traces["native"]["anchor_" + k]) != scan[k]:
            raise ValueError(f"Scanned signal or candidate differs on replay: {case['id']} / {k}")
    if not d["valid"]:
        return dict(d, protected=None, benefit=None, protected_benefit=None)
    protect = not d["event_harm"] and d["added_failure"] == 0 and min(d["delta32"], d["delta_window"]) >= -EPSILON
    benefit = d["delta_window"] >= EPSILON or d["prevented_terminal"] or d["rescued_recovery"]
    return dict(d, protected=protect, benefit=benefit, protected_benefit=protect and benefit)


def summarize(choices, outcomes):
    labels = {r["case_id"]: r for r in outcomes}
    lost_groups = {p["group"] for p in choices if not labels[p["case_id"]]["valid"]}
    counts = ("protected_benefit", "event_harm", "new_terminal", "lost_recovery", "added_failure")
    metrics = ("delta32", "delta_window", "protected_benefit", "event_harm")
    summary, paired, decisions, group_values = [], [], [], {}
    for policy in POLICIES:
        groups = defaultdict(list)
        for pick in choices:
            if pick["policy"] != policy:
                continue
            key = pick["case_id"]
            result = labels[key]
            keep = pick["group"] not in lost_groups
            decisions.append(dict(pick, common_valid_group=keep, **{k: result[k] for k in (*counts, "delta32", "delta_window")}))
            if keep:
                groups[pick["group"]].append(result)
        if any(len(rr) != QUOTA for rr in groups.values()):
            raise ValueError("Matched held-group quota changed")
        group_values[policy] = {g: {k: float(np.mean([r[k] for r in rows])) for k in metrics} for g, rows in groups.items()}
        selected = [r for rows in groups.values() for r in rows]
        summary.append(dict(policy=policy, source_groups=len(groups), selections=len(selected),
                            benefit_groups=sum(any(r["protected_benefit"] for r in rows) for rows in groups.values()),
                            harm_groups=sum(any(r["event_harm"] for r in rows) for rows in groups.values()),
                            **{k: sum(r[k] for r in selected) for k in counts},
                            **{f"mean_{k}": float(np.mean([v[k] for v in group_values[policy].values()])) if groups else None
                               for k in metrics[:2]}))
    for policy in POLICIES[1:]:
        control, active = group_values["uniform"], group_values[policy]
        if control.keys() != active.keys():
            raise ValueError("Policy comparisons must have the same source groups")
        if active:
            for k in metrics:
                sign = -1 if k == "event_harm" else 1
                delta = [sign * (active[g][k] - control[g][k]) for g in sorted(active)]
                paired.append(dict(policy=policy, **base.s103.contrast(delta, f"{policy}_vs_uniform",
                              "harm_reduction" if k == "event_harm" else k, online.DESIGN)))
    return dict(summary=summary, paired=paired, decisions=decisions)


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r173, r174, r175, r176 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    roots = {170: r170, 173: r173, 174: r174, 175: r175, 176: r176}
    schemas = {170: (previous.SCHEMA, event.matched.SOURCE170), **REFERENCES}
    refs = {stage: c.c66.sealed(roots[stage] / f"stage{stage}_decision.json", schema)
            for stage, (schema, _) in schemas.items()}
    d170 = refs[170]
    if (d170["plan_fingerprint"] != p170["fingerprint"] or p170["runtime"] != d170["runtime"]
            or d167["fingerprint"] != previous.SOURCE167
            or any(refs[s]["fingerprint"] != fp or refs[s]["runtime"] != d170["runtime"]
                   for s, (_, fp) in schemas.items())):
        raise ValueError("Need the bound Stage-167, Stage-170 and Stage-173 to Stage-176 sources")

    def read(stage, table):
        return risk.read_verified(roots[stage], refs[stage], stage, table)

    held_roster, reserved, sentinels = archive_scope(read)
    sentinel_labels = {r["case_id"]: r for r in read(175, "outcomes")}
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Stage-170 source groups or fit partition changed")
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    needed = {(s, event.matched.COLD) for s in held_roster} | {(p["sequence"], p["source"]) for p in sentinels}
    for seq, arm in sorted(needed):
        size = jobs[seq]["frames"]
        if seq not in entries:
            entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
            if (entry["sequence"] != seq or len(entry["frames"]) != size + 1 or len(entry["ground_truth"]) != size + 1
                    or held_roster.get(seq, size) != size):
                raise ValueError(f"Video roster changed at {seq}")
            entries[seq] = entry
        data = previous.archive_trace(r170, p170, d170, seq, arm)
        previous.audit(data, size, entries[seq]["initial_box"], arm, thresholds[fold_of[risk.SOURCE_GROUP(seq)]])
        sources[seq, arm] = data
    candidates, support = pool(held_roster, sources, reserved, sentinels)

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(sweep.__file__), Path(event.__file__), Path(risk.__file__), Path(aligned.__file__),
            Path(previous.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_reports={str(s): refs[s]["fingerprint"] for s in refs},
                        candidates=candidates, support=support, folds=folds, runtime=runtime, runtime_differences=differences,
                        scan_limit=SCAN, quota=QUOTA, seed=SEED, protection_tolerance=EPSILON, horizon=LONG,
                        policies=list(POLICIES), code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "scan").mkdir(exist_ok=True)
    (root / "rollouts").mkdir(exist_ok=True)

    def scan_location(case):
        return root / "scan" / f"{case['id']}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=case["id"])

    scan_reused = sum(scan_location(case)[0].with_suffix(".receipt.json").exists() for case in candidates)
    fit_work = len(points) + sum(len(f["fit"]) + 1 for f in folds)
    max_replays = len(held_roster) * len(POLICIES) * QUOTA * len(risk.ARMS) * (LONG + 1)
    progress = OverallProgress("Stage-177", len(candidates) + fit_work + max_replays + 1, scan_reused,
                               args.progress_interval, "work")
    progress.report("Restoring Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables = dict(scan_support=support, scans=[], choices=[], outcomes=[], sentinels=[])
    evidence, templates = {}, {}

    def record(path):
        for f in (path, path.with_suffix(".receipt.json")):
            evidence[f.relative_to(root).as_posix()] = c.file_hash(f)

    def read_frame(seq, frame):
        return online.checked_image(old["root"], entries[seq]["frames"][frame])

    def template(seq):
        if seq not in templates:
            templates[seq] = read_frame(seq, 0)
        return templates[seq]

    with ExitStack() as stack:
        observer = ReopenObserver(native, stack)
        progress.report("Scanning causal signals on native proposals")
        for case in candidates:
            seq = case["sequence"]
            archive = sources[seq, case["source"]]
            model = models[fold_of[case["group"]]]
            one = dict(case, end=case["frame"])
            path, metadata = scan_location(case)
            data = c.read_cache(path, metadata)
            fresh = data is None
            if fresh:
                data = aligned.replay(observer, lambda f: read_frame(seq, f), template(seq), entries[seq], one,
                                      archive, "native", model, progress)
            aligned.audit(data, one, archive, "native")
            row = scan_row(case, data, archive, not differences)
            if fresh:
                c.save(path, data, metadata)
            record(path)
            tables["scans"].append(row)
            if case["cohort"] == "sentinel":
                previous_label = sentinel_labels[case["id"]]
                tables["sentinels"].append(dict(row, frame=case["frame"], archived_event_harm=previous_label["event_harm"],
                                                archived_gain128=previous_label["delta_window"],
                                                archived_prevented_terminal=previous_label["prevented_terminal"]))

        tables["choices"], tables["quota"] = select(tables["scans"])
        selected = {r["case_id"] for r in tables["choices"]}
        replays = [r for r in candidates if r["id"] in selected]
        scan_index = {r["case_id"]: r for r in tables["scans"]}
        collection = io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], scan_sha256=dict(evidence),
                                  replays=replays, choices=tables["choices"]))
        c.publish_identical(root / "collection.json", collection)

        def location(case, arm):
            return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(collection_fingerprint=collection["fingerprint"],
                                                                     case_id=case["id"], arm=arm)

        searches = len(replays) * len(risk.ARMS) * (LONG + 1)
        progress.total = len(candidates) + fit_work + searches + 1
        resumed = sum((LONG + 1) for case in replays for arm in risk.ARMS
                      if location(case, arm)[0].with_suffix(".receipt.json").exists())
        progress.reused += resumed
        progress.advance(count=resumed)
        progress.report("Replaying matched reopening choices")
        labels, archive_iou = {}, {}
        for case in replays:
            seq, key = case["sequence"], (case["sequence"], case["source"])
            archive, entry = sources[key], entries[seq]
            if key not in archive_iou:
                labels[key] = np.asarray(entry["ground_truth"], float)
                archive_iou[key] = online.tracking_iou(archive["boxes"], labels[key][1:])
            model = models[fold_of[case["group"]]]
            traces = {}
            for arm in risk.ARMS:
                path, metadata = location(case, arm)
                data = c.read_cache(path, metadata)
                fresh = data is None
                if fresh:
                    data = aligned.replay(observer, lambda f: read_frame(seq, f), template(seq), entry, case,
                                          archive, arm, model, progress)
                aligned.audit(data, case, archive, arm)
                if fresh:
                    c.save(path, data, metadata)
                record(path)
                traces[arm] = data
            gt = labels[key][case["frame"]:case["end"] + 1]
            tables["outcomes"].append(report(case, traces, gt, archive_iou[key], archive, not differences, scan_index[case["id"]]))
    native.assert_frozen()
    tables.update(summarize(tables["choices"], tables["outcomes"]))
    if progress.done != progress.total - 1:
        raise ValueError("Incomplete Stage-177 progress accounting")
    stage_label = {r["policy"]: r for r in tables["summary"]}
    joint, uniform = stage_label["joint"], stage_label["uniform"]
    supports = (joint["benefit_groups"] >= 3 and joint["protected_benefit"] > uniform["protected_benefit"]
                and joint["event_harm"] <= uniform["event_harm"] and joint["new_terminal"] == 0
                and joint["added_failure"] <= uniform["added_failure"] and joint["lost_recovery"] <= uniform["lost_recovery"]
                and joint["mean_delta_window"] > uniform["mean_delta_window"] and joint["mean_delta_window"] > 0.)
    evidence["collection.json"] = c.file_hash(root / "collection.json")
    return dict(plan_fingerprint=plan["fingerprint"], collection_fingerprint=collection["fingerprint"],
                stage176_unused_groups=len({risk.SOURCE_GROUP(s) for s in held_roster}), scan_points=len(candidates),
                held_scan_points=sum(r["cohort"] == "held" for r in candidates),
                sentinel_points=len(sentinels), replay_points=len(replays),
                valid_replay_points=sum(r["valid"] for r in tables["outcomes"]), scan_searches=len(candidates),
                replay_searches=searches, supports_reopening_pilot=supports, new_full_trajectories=0,
                joint=joint, work=progress.total, runtime=runtime, runtime_differences=differences, artifact_sha256=evidence,
                table_sha256=gain.prior.publish(root, tables, stage="stage177")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=5, type=Path, metavar=("STAGE170", "STAGE173", "STAGE174", "STAGE175", "STAGE176"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage177_native_proposal_audit")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        bindings = {170: (previous.SCHEMA, event.matched.SOURCE170), **REFERENCES}
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1]
                  for p, (s, (_, fp)) in zip(args.bundles, bindings.items(), strict=True)}
        held, reserved, sentinels = archive_scope(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        capable = sum(n > HISTORY + LONG for n in held.values())
        print(f"Verified {len(held)} Stage-176-unused videos, {capable} with a possible 128-frame window, "
              f"{len(reserved)} reserved anchors and {len(sentinels)} sentinels")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage177_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage176"] != SOURCE176 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-177 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-177", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE177_NATIVE_PROPOSAL_AUDIT",
                                              source_stage176=SOURCE176, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
