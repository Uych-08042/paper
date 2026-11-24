"""Compare short and long downside supervision at the same layer-switch budget."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage173_gate_layer_alignment as aligned


base, gate, gain, history, c, io = aligned.base, aligned.gate, aligned.gain, aligned.history, aligned.c, aligned.io
matched, previous, online, windows = aligned.matched, aligned.previous, aligned.online, aligned.windows
SOURCE173 = "10e874a80e41905eb159898d875800822b83339639c387403fb9ebbd78bad85f"
SCHEMA = "stage174_long_horizon_risk_v1"
SOURCES = (76, 89, 93, 121, 159, 165, 166, 167, 170, 173)
ARMS = ("native", "ranked")
POLICIES = ("frozen32", "refit32", "refit128")
STRATA = ("accepted", "cooldown_vetoed", "gate_rejected")
HORIZONS = (32, 128)
LIMIT, SPACING, SEED = 2, 32, 23174
FRACTION, SEVERE, BENEFICIAL = 0.40, 0.05, 0.01
SOURCE_GROUP = gate.veto.source_group


def sealed_cases(cases, anchors):
    by_id = {r["case_id"]: r for r in anchors}
    seen = {(r["sequence"], int(r["frame"])) for r in cases}
    known = [r for r in cases if r["cohort"] == "targeted"]
    if len(cases) != 39 or len(by_id) != 39 or len(seen) != 39 or len(known) != 6:
        raise ValueError("Need all 39 sealed Stage-173 anchors and six targeted checks")
    for row in cases:
        anchor = by_id[row["id"]]
        if (anchor["exact_source_layers"] != "True" or float(anchor["source_action_box_error"]) != 0.
                or any(int(anchor[k]) != int(row["expected_" + k]) for k in aligned.ARMS)):
            raise ValueError("Stage-173 archive does not have exact source anchors")
    return seen, known


def sample(roster, sources, held_fold, reserved, targeted):
    excluded = {SOURCE_GROUP(r["sequence"]) for r in targeted}
    support, cases = [], []
    for seq, size in sorted(roster.items()):
        group = SOURCE_GROUP(seq)
        if group in excluded:
            continue
        data = sources[seq, matched.COLD]
        rejected = (data["proposal"] != data["native_layer"]) & ~data["would_switch"]
        for stratum, mask in zip(STRATA, (data["switched"], data["vetoed"], rejected), strict=True):
            frames = [int(i) for i in np.flatnonzero(mask) + 1]
            eligible = [i for i in frames if online.trained.DESIGN["history"] < i <= size - HORIZONS[-1]
                        and (seq, i) not in reserved]
            ordered = sorted(eligible, key=lambda i: sha256(f"{SEED}/{seq}/{stratum}/{i}".encode()).digest())
            picked = []
            for frame in ordered:
                if all(abs(frame - other) >= SPACING for other in picked):
                    picked.append(frame)
                    cases.append(dict(id=f"{seq}__f{frame:06d}", sequence=seq, group=group, fold=held_fold[group],
                                      stratum=stratum, source=matched.COLD, frame=frame, end=frame + HORIZONS[-1]))
                    if len(picked) == LIMIT:
                        break
            support.append(dict(sequence=seq, group=group, fold=held_fold[group], stratum=stratum,
                                opportunities=len(frames), eligible=len(eligible), sampled=len(picked)))
    for ref in targeted:
        seq, frame = ref["sequence"], int(ref["frame"])
        cases.append(dict(id=ref["id"], sequence=seq, group=SOURCE_GROUP(seq), fold=held_fold[SOURCE_GROUP(seq)],
                          stratum="targeted", source=ref["source"], frame=frame, end=frame + HORIZONS[-1]))
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate Stage-174 anchor")
    return cases, support


def anchor_scores(models, native, short, long):
    proposal, legacy, allowed, _ = gate.online_decision(models, native, short, long, True)
    scores = gate.score(models["pairwise"], short, pairwise=True)
    ranked = gate.choose_pairwise(scores, native, proposal)
    n, k = (online.LAYERS.index(layer) for layer in (native, ranked))
    values = gate.score(models["gain"], short)
    risk = max(0., gate.score(models["risk"], long)[k])
    ranked_gain = float(values[k] - values[n])
    if not np.isfinite([legacy, risk, ranked_gain, *long]).all():
        raise ValueError("Nonfinite anchor score or feature")
    return dict(native=native, ranked=ranked, proposal=proposal, legacy=float(legacy), allowed=allowed,
                gain32=ranked_gain, risk32=float(risk), features=long)


class RiskObserver(aligned.AlignmentObserver):
    def route(self, module, args, output):
        if self.arm == "native":
            return super().route(module, args, output)
        self.original = self.rt.selected_layer(output)
        if self.tokens is None:
            raise ValueError("Missing L6 anchor features")
        boxes, peaks = np.asarray(self.boxes), np.asarray(self.peaks)
        frame = len(boxes) + 1
        visual = online.trained.projected_prefix(self.rt, self.tokens)
        short = np.r_[visual, online.trained.history_features(boxes, peaks, frame)]
        long = np.r_[visual, history.features_from_history(boxes, peaks, frame, online.DESIGN["history"], True)]
        self.diagnostic = anchor_scores(self.model, self.original, short, long)
        self.proposal, self.value = self.diagnostic["proposal"], self.diagnostic["legacy"]
        self.chosen = self.diagnostic[self.action]
        self.switched = self.chosen != self.original
        if self.switched:
            forced = self.rt.torch.zeros_like(output)
            forced[:, online.LAYERS.index(self.chosen)] = 1
            return forced
        return None


def measure(case, traces, truth, source, exact_runtime, prior):
    native, ranked = (traces[k] for k in ARMS)
    for k in native:
        if k.startswith("anchor_"):
            np.testing.assert_array_equal(native[k], ranked[k])
    a = {k.removeprefix("anchor_"): v.item() for k, v in native.items()
         if k.startswith("anchor_") and k != "anchor_features"}
    i = case["frame"] - 1
    would_switch = source.get("would_switch", source["switched"])
    same_route = (a["native"] == source["native_layer"][i] and a["proposal"] == source["proposal"][i]
                  and a["allowed"] == would_switch[i] and abs(a["legacy"] - source["gate_value"][i]) <= 1e-12)
    natural = [t for t in traces.values() if t["layer"][0] == source["layer"][i]]
    box_error = max((float(np.abs(t["boxes"][0] - source["boxes"][i]).max()) for t in natural), default=None)
    if exact_runtime and (not same_route or box_error != 0.):
        raise ValueError(f"Stage-170 anchor replay differs at {case['id']}")
    values = np.stack([online.tracking_iou(t["boxes"], truth) for t in (native, ranked)])
    complete = bool((values >= 0.).all())
    outcome = dict(case_id=case["id"], **a, valid=complete, source_action_box_error=box_error,
                   same_source_gate=bool(same_route), delta32=None, delta128=None, severe=None, new_terminal=None,
                   beneficial=None, native_terminal=None, ranked_terminal=None)
    if not complete:
        if prior is not None and exact_runtime:
            raise ValueError(f"Stage-173 sentinel lost valid targets at {case['id']}")
        return outcome, None
    effects = {f"delta{h}": float((values[1, 1:h + 1] - values[0, 1:h + 1]).mean()) for h in HORIZONS}
    n_terminal, r_terminal = (sum(e["recovery"] is None for e in online.failure_events(v)) for v in values)
    terminal = r_terminal > n_terminal
    severe = terminal or effects["delta128"] <= -SEVERE
    if prior is not None and exact_runtime:
        for k in aligned.ARMS:
            if a[k] != int(prior[0][k]):
                raise ValueError(f"Stage-173 sentinel layer changed at {case['id']}")
        np.testing.assert_allclose(list(effects.values()), prior[1], rtol=0., atol=1e-12)
    outcome.update(**effects, severe=severe, new_terminal=terminal,
                   beneficial=effects["delta128"] >= BENEFICIAL and not severe,
                   native_terminal=n_terminal, ranked_terminal=r_terminal)
    long = native["anchor_features"]
    if long.shape != (27,):
        raise ValueError("Unexpected risk input dimension")
    identity = np.eye(len(online.LAYERS))
    features = np.r_[long, identity[online.LAYERS.index(a["native"])], identity[online.LAYERS.index(a["ranked"])]]
    return outcome, features


def compare(cases, outcomes, features, folds):
    rows = {r["case_id"]: r for r in outcomes if r["valid"]}
    trainable = [r for r in cases if r["stratum"] != "targeted" and r["id"] in rows]
    directions, fits, checks, summaries, paired = [], [], [], [], []
    metric_rows = defaultdict(lambda: defaultdict(list))
    for fold in folds:
        number = fold["fold"]
        fit = [r for r in trainable if r["group"] in fold["fit"]]
        held = [r for r in trainable if r["group"] in fold["held"]]
        sentinel = [r for r in cases if r["stratum"] == "targeted" and r["fold"] == number and r["id"] in rows]
        counts = Counter(r["group"] for r in fit)
        if not counts or not held or set(counts) & {r["group"] for r in held + sentinel}:
            raise ValueError("Insufficient support or overlapping fit/held source groups")
        x = np.stack([features[r["id"]] for r in fit])
        y = np.array([[max(0., -rows[r["id"]][f"delta{h}"]) for h in HORIZONS] for r in fit])
        weights = np.array([1. / counts[r["group"]] for r in fit])
        mean, scale, bias, coef = online.trained.ridge(x, y, weights)
        quota = max(1, int(FRACTION * len(held) + .5))
        scores, risks = {}, {}
        for case in held + sentinel:
            key = case["id"]
            row = rows[key]
            pred = np.maximum(0., ((features[key] - mean) / scale) @ coef + bias)
            risks[key] = dict(zip(POLICIES, (row["risk32"], *map(float, pred)), strict=True))
            scores[key] = {p: float(row["gain32"] - gate.risk.PENALTY * risk) for p, risk in risks[key].items()}

        fits.append(dict(fold=number, fit_points=len(fit), fit_groups=len(counts), held_points=len(held), quota=quota,
                         fit_group_ids="|".join(sorted(counts)),
                         held_group_ids="|".join(sorted({r["group"] for r in held})),
                         fit_severe=sum(rows[r["id"]]["severe"] for r in fit),
                         held_severe=sum(rows[r["id"]]["severe"] for r in held)))
        for policy in POLICIES:
            def position(case):
                return -scores[case["id"]][policy], case["id"]

            order = sorted(held, key=position)
            cutoff = position(order[quota - 1])
            acc = 0
            for case in order + sentinel:
                key = case["id"]
                row = rows[key]
                selected = position(case) <= cutoff
                if case["stratum"] != "targeted":
                    acc += selected
                target = 128 if policy == "refit128" else 32
                rec = dict(case_id=key, sequence=case["sequence"], stratum=case["stratum"], fold=number, policy=policy,
                           selected=selected, risk=risks[key][policy], score=scores[key][policy], cutoff=-cutoff[0],
                           risk_target=target, risk_absolute_error=abs(risks[key][policy] - max(0., -row[f"delta{target}"])),
                           severe_accepted=selected and row["severe"], terminal_accepted=selected and row["new_terminal"],
                           beneficial_accepted=selected and row["beneficial"], beneficial_blocked=not selected and row["beneficial"],
                           gain32=row["delta32"] if selected else 0., gain128=row["delta128"] if selected else 0.)
                if case["stratum"] == "targeted":
                    checks.append(rec)
                else:
                    directions.append(rec)
                    metric_rows[policy][case["group"]].append(rec)
            if acc != quota:
                raise ValueError("Matched switch quota changed")
    if len(directions) != len(POLICIES) * len(trainable):
        raise ValueError("Incomplete outer held decisions")
    metrics = ("gain32", "gain128", "severe_accepted", "terminal_accepted", "beneficial_accepted")
    group_means = {}
    for policy, groups in metric_rows.items():
        group_means[policy] = {k: {m: float(np.mean([r[m] for r in rr])) for m in metrics} for k, rr in groups.items()}
        rr = [r for r in directions if r["policy"] == policy]
        summaries.append(dict(policy=policy, source_groups=len(groups), anchors=len(rr),
                              **{k: sum(r[k] for r in rr) for k in
                                 ("selected", "severe_accepted", "terminal_accepted", "beneficial_accepted", "beneficial_blocked")},
                              mean_gain32=float(np.mean([v["gain32"] for v in group_means[policy].values()])),
                              mean_gain128=float(np.mean([v["gain128"] for v in group_means[policy].values()]))))
    for control in POLICIES[:2]:
        for metric in metrics:
            long, short = group_means["refit128"], group_means[control]
            delta = [long[k][metric] - short[k][metric] for k in sorted(long)]
            if metric in ("severe_accepted", "terminal_accepted"):
                paired.append(dict(control=control, **base.s103.contrast([-v for v in delta],
                    f"{control}_minus_refit128", metric + "_reduction", online.DESIGN)))
                continue
            stats = base.s103.contrast(delta, f"refit128_minus_{control}", metric, online.DESIGN)
            paired.append(dict(control=control, **stats))
    return dict(fits=fits, decisions=directions, targeted=checks, summary=summaries, paired=paired)


def read_verified(root, decision, stage, table):
    name = f"stage{stage}_{table}.csv"
    path = root / name
    io.verify({str(path): decision["table_sha256"][name]})
    return gain.prior.prior.read_table(path)


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r173 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    d173 = c.c66.sealed(r173 / "stage173_decision.json", aligned.SCHEMA)
    if (d170["fingerprint"] != matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or p170["runtime"] != d170["runtime"] or d167["fingerprint"] != previous.SOURCE167
            or d173["fingerprint"] != SOURCE173 or d173["runtime"] != d170["runtime"]):
        raise ValueError("Need the sealed Stage-167, Stage-170 and Stage-173 sources")
    prior_cases = read_verified(r173, d173, 173, "cases")
    prior_anchors = read_verified(r173, d173, 173, "anchors")
    reserved, targeted = sealed_cases(prior_cases, prior_anchors)
    prior_outcomes = {(r["case_id"], int(r["horizon"])): float(r["ranked_delta"])
                      for r in read_verified(r173, d173, 173, "outcomes")}
    prior = {r["case_id"]: (r, [prior_outcomes[r["case_id"], h] for h in HORIZONS])
             for r in prior_anchors if r["case_id"] in {p["id"] for p in targeted}}
    roster = {r["sequence"]: int(r["frames"]) for r in read_verified(r170, d170, 170, "video_summary")
              if r["arm"] == matched.COLD}
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if (len(roster) != online.DESIGN["videos"] or set(roster) != set(jobs) or folds != p170["folds"]
            or d121["fingerprint"] != p159["source_stage121"]):
        raise ValueError("Stage-170 video roster or fit partition changed")
    held_fold = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    paths170 = {(seq, matched.COLD) for seq in roster} | {(r["sequence"], r["source"]) for r in targeted}
    for seq, arm in sorted(paths170):
        entry = entries.get(seq)
        if entry is None:
            entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
            entries[seq] = entry
        size = roster[seq]
        if (entry["sequence"] != seq or jobs[seq]["frames"] != size or len(entry["frames"]) != size + 1
                or len(entry["ground_truth"]) != size + 1):
            raise ValueError(f"Frame roster changed at {seq}")
        data = previous.archive_trace(r170, p170, d170, seq, arm)
        previous.audit(data, size, entry["initial_box"], arm, thresholds[held_fold[SOURCE_GROUP(seq)]])
        sources[seq, arm] = data
    cases, support = sample(roster, sources, held_fold, reserved, targeted)
    if any(r["end"] > roster[r["sequence"]] for r in cases):
        raise ValueError("A Stage-174 anchor exceeds its video")
    if {r["fold"] for r in cases if r["stratum"] != "targeted"} != set(thresholds):
        raise ValueError("Missing expandable states in a held fold")

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(aligned.__file__), Path(previous.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage170=d170["fingerprint"], source_stage173=SOURCE173, cases=cases,
                        support=support, folds=folds, horizons=list(HORIZONS), arms=list(ARMS), policies=list(POLICIES),
                        seed=SEED, fraction=FRACTION, severe=SEVERE, beneficial=BENEFICIAL, max_per_stratum=LIMIT,
                        spacing=SPACING, runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)

    def location(case, arm):
        return root / "rollouts" / f"{case['id']}__{arm}.npz", dict(plan_fingerprint=plan["fingerprint"],
                                                                 case_id=case["id"], arm=arm)

    calls = HORIZONS[-1] + 1
    searches = calls * len(cases) * len(ARMS)
    reused = calls * sum(location(case, arm)[0].with_suffix(".receipt.json").exists()
                         for case in cases for arm in ARMS)
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-174", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables = dict(cases=cases, support=support, outcomes=[])
    artifacts, features = {}, {}
    progress.report("Replaying native and ranked actions")
    with ExitStack() as stack:
        observer = RiskObserver(native, stack)
        for case in cases:
            seq = case["sequence"]
            entry, source = entries[seq], sources[seq, case["source"]]
            traces, template = {}, None

            def read(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for arm in ARMS:
                path, metadata = location(case, arm)
                trace = c.read_cache(path, metadata)
                fresh = trace is None
                if fresh:
                    if template is None:
                        template = read(0)
                    trace = aligned.replay(observer, read, template, entry, case, source, arm, models[case["fold"]], progress)
                aligned.audit(trace, case, source, arm)
                if fresh:
                    c.save(path, trace, metadata)
                traces[arm] = trace
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            truth = np.asarray(entry["ground_truth"], float)[case["frame"]:case["end"] + 1]
            row, vector = measure(case, traces, truth, source, not differences, prior.get(case["id"]))
            tables["outcomes"].append(row)
            if vector is not None:
                features[case["id"]] = vector
    native.assert_frozen()
    progress.report("Comparing matched held-out budgets")
    tables.update(compare(cases, tables["outcomes"], features, folds))
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-174 progress accounting")
    main_cases = {r["id"]: r for r in cases if r["stratum"] != "targeted"}
    severe_groups = {main_cases[r["case_id"]]["group"] for r in tables["outcomes"]
                     if r["case_id"] in main_cases and r["severe"]}
    cmp = {r["policy"]: r for r in tables["summary"]}
    long = cmp["refit128"]
    advance = len(severe_groups) >= 3 and all(long["severe_accepted"] < cmp[p]["severe_accepted"]
                 and long["terminal_accepted"] <= cmp[p]["terminal_accepted"]
                 and long["beneficial_accepted"] >= cmp[p]["beneficial_accepted"]
                 and long["mean_gain128"] > cmp[p]["mean_gain128"] for p in POLICIES[:2])
    return dict(plan_fingerprint=plan["fingerprint"], cases=len(cases), expanded_cases=len(main_cases),
                sampled_strata=dict(Counter(r["stratum"] for r in main_cases.values())), valid_cases=len(features),
                severe_source_groups=len(severe_groups), supports_closed_loop_pilot=advance, searches=searches,
                rollouts=len(cases) * len(ARMS), new_full_trajectories=0, work=work, runtime=runtime,
                runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage174")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Verify the Stage-173 result bundle without CUDA")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage174_long_horizon_risk")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU source check")
        _, tables = gain.prior.archives.bundle(args.bundle, 173, SOURCE173)
        reserved, targeted = sealed_cases(tables["stage173_cases.csv"], tables["stage173_anchors.csv"])
        print(f"Verified {len(reserved)} archived anchors and {len(targeted)} targeted checks")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage174_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage173"] != SOURCE173 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-174 completed run has a different source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-174", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE174_LONG_HORIZON_RISK",
                                              source_stage173=SOURCE173, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
