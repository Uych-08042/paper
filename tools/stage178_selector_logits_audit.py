"""Capture native selector logits and compare them with frozen continuation labels."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage177_native_proposal_audit as reopening


sweep, event, risk, online = reopening.sweep, reopening.event, reopening.risk, reopening.online
previous, base, gain, history, c, io = reopening.previous, reopening.base, reopening.gain, reopening.history, reopening.c, reopening.io
SOURCE177 = "a149550250865cb0f22fe2eff6d6e99da80360dbaefc60a44c1237303695f624"
SCHEMA = "stage178_selector_logits_audit_v1"
SOURCES = (76, 89, 93, 121, 167, 170, 176, 177)
LAYERS = online.LAYERS


def source_cases(read):
    scans = read(177, "scans")
    cases176 = read(176, "cases")
    anchors176 = {r["case_id"]: r for r in read(176, "anchors")}
    if len(scans) != 243 or len(cases176) != 26 or len(anchors176) != 26:
        raise ValueError("Incomplete Stage-176 or Stage-177 roster")
    states = {}
    for row in scans:
        seq, suffix = row["case_id"].rsplit("__f", 1)
        if seq != row["sequence"] or not suffix.isdecimal():
            raise ValueError("Unexpected Stage-177 scan key")
        frame = int(suffix)
        key = seq, frame
        if key in states:
            raise ValueError("Duplicate Stage-177 state")
        states[key] = dict(id=row["case_id"], sequence=seq, frame=frame, source=event.matched.COLD, group=row["group"],
                           stage177_id=row["case_id"], stage176_id=None, native=int(row["native"]),
                           ranked=int(row["ranked"]), scan_cohort=row["cohort"])
    seen176 = set()
    for row in cases176:
        seq, frame = row["sequence"], int(row["frame"])
        key = seq, frame
        if key in seen176 or row["source"] != event.matched.COLD:
            raise ValueError("Ambiguous Stage-176 archived state")
        seen176.add(key)
        a = anchors176[row["id"]]
        state = states.setdefault(key, dict(id=row["id"], sequence=seq, frame=frame, source=row["source"],
                                           group=row["group"], stage177_id=None, stage176_id=None,
                                           native=int(a["native_layer"]), ranked=int(a["ranked_layer"]), scan_cohort=None))
        if (state["native"], state["ranked"]) != (int(a["native_layer"]), int(a["ranked_layer"])):
            raise ValueError(f"Archived roles do not agree at {state['id']}")
        state["stage176_id"] = row["id"]
    if len({r["id"] for r in states.values()}) != len(states):
        raise ValueError("Nonunique Stage-178 state ID")
    return sorted(states.values(), key=lambda r: (r["sequence"], r["frame"]))


class LogitObserver(online.RoutedObserver):
    def __init__(self, native, stack):
        self.logits = self.probabilities = None
        super().__init__(native, stack)
        stack.callback(self.rt.backbone.MLP.fc2.register_forward_hook(self.capture_logits).remove)

    def capture_logits(self, module, args, output):
        if self.logits is not None:
            raise ValueError("Selector fc2 executed more than once")
        self.logits = output.detach()

    def route(self, module, args, output):
        if self.logits is None or self.probabilities is not None:
            raise ValueError("Missing or repeated selector output")
        self.rt.torch.testing.assert_close(self.logits.sigmoid(), output, rtol=0, atol=0)
        self.probabilities = output.detach()
        return super().route(module, args, output)

    def observe(self, template, image, initial_box, parent, frame):
        self.logits = self.probabilities = None
        self.initialize(template, initial_box, "native", None)
        self.rt.tracker.frame_id = frame - 1
        self.logits = self.probabilities = None
        output = self.step(image, parent, frame)
        return dict(output, logits=self.logits.cpu().numpy().reshape(-1),
                    probabilities=self.probabilities.cpu().numpy().reshape(-1))


def cache_reference(root, decision, relative, metadata):
    path = root / relative
    refs = (path, path.with_suffix(".receipt.json"))
    io.verify({str(p): decision["artifact_sha256"][p.relative_to(root).as_posix()] for p in refs})
    out = c.read_cache(path, metadata)
    if out is None:
        raise ValueError(f"Missing verified anchor reference {path}")
    return out


def references(case, r176, r177, d176, d177):
    stage177, stage176 = case["stage177_id"], case["stage176_id"]
    if stage177 is not None:
        yield cache_reference(r177, d177, f"scan/{stage177}.npz",
                              dict(plan_fingerprint=d177["plan_fingerprint"], case_id=stage177))
    if stage176 is not None:
        layer = case["native"]
        yield cache_reference(r176, d176, f"rollouts/{stage176}__l{layer:02d}.npz",
                              dict(plan_fingerprint=d176["plan_fingerprint"], case_id=stage176, layer=layer))


def signal_row(case, trace, archived, exact_runtime):
    logits = np.asarray(trace["logits"], float)
    post = np.asarray(trace["probabilities"], float)
    if (logits.shape != (len(LAYERS),) or post.shape != logits.shape or not np.isfinite(logits).all()
            or not np.isfinite(post).all() or (post < 0.).any() or (post > 1.).any()):
        raise ValueError(f"Invalid selector scores at {case['id']}")
    if int(trace["frame_id"]) != case["frame"] or int(trace["layer"]) != int(trace["native_layer"]):
        raise ValueError("Native route was not preserved")
    for k in ("crop_calls", "head_calls", "selector_calls"):
        np.testing.assert_equal(trace[k], 1)
    blocks = np.zeros(12, int)
    blocks[:6] = 1
    blocks[int(trace["layer"]) - 1] = 1
    np.testing.assert_array_equal(trace["block_calls"], blocks)
    source_error, peak_error, same = 0., 0., True
    for old in archived:
        if int(old["frame_id"][0]) != case["frame"]:
            raise ValueError("Archived anchor frame changed")
        source_error = max(source_error, float(np.abs(trace["boxes"] - old["boxes"][0]).max()))
        peak_error = max(peak_error, abs(float(trace["hann_peak"]) - float(old["hann_peak"][0])))
        same = same and int(trace["layer"]) == int(old["layer"][0]) and np.array_equal(trace["parent"], old["parent"][0])
    native, ranked = case["native"], case["ranked"]
    n, k = LAYERS.index(native), LAYERS.index(ranked)
    post_gap = float(post[n] - np.max(np.delete(post, n)))
    if exact_runtime and (not same or source_error != 0. or peak_error != 0. or int(trace["layer"]) != native
                          or (case["stage177_id"] is not None and post_gap != 1.)):
        raise ValueError(f"Stage-176/177 native anchor changed at {case['id']}")
    return dict(case_id=case["id"], sequence=case["sequence"], group=case["group"], frame=case["frame"],
                scan_cohort=case["scan_cohort"], native=int(trace["layer"]), ranked=ranked,
                max_raw_layer=LAYERS[int(logits.argmax())],
                selector_gap_raw=float(logits[n] - np.max(np.delete(logits, n))), selector_gap_post=post_gap,
                ranked_minus_native_raw=float(logits[k] - logits[n]), ranked_minus_native_post=float(post[k] - post[n]),
                native_box_error=source_error, hann_peak_error=peak_error, same_native_state=bool(same),
                **{f"logit_l{layer}": float(v) for layer, v in zip(LAYERS, logits, strict=True)},
                **{f"post_l{layer}": float(v) for layer, v in zip(LAYERS, post, strict=True)})


def join_effects(cases, signals, read):
    by_id = {r["case_id"]: r for r in signals}
    by176 = {p["stage176_id"]: p for p in cases if p["stage176_id"]}
    by177 = {p["stage177_id"]: p for p in cases if p["stage177_id"]}
    roster176 = {p["id"]: p for p in read(176, "cases")}
    outcomes, alternatives, sentinels = [], [], []
    for row in read(176, "layers"):
        p = by176[row["case_id"]]
        s = by_id[p["id"]]
        layer = int(row["layer"])
        if layer == p["native"]:
            continue
        ref = roster176[row["case_id"]]
        candidate = dict(case_id=p["id"], group=p["group"],
                         cohort="stage176_known" if ref["known_source"] == "True" else "stage176_other",
                         layer=layer, window=int(ref["end"]) - int(ref["frame"]),
                         candidate_raw=s[f"logit_l{layer}"] - s[f"logit_l{p['native']}"],
                         candidate_post=s[f"post_l{layer}"] - s[f"post_l{p['native']}"],
                         benefit=row["protected_benefit"] == "True", harm=row["event_harm"] == "True",
                         delta128_or_shorter=float(row["delta_window"]))
        alternatives.append(candidate)
        if layer == p["ranked"]:
            outcomes.append(dict(candidate, selector_raw=-s["selector_gap_raw"], selector_post=-s["selector_gap_post"]))
    for row in read(177, "outcomes"):
        p = by177[row["case_id"]]
        s = by_id[p["id"]]
        if row["valid"] != "True":
            continue
        outcomes.append(dict(case_id=p["id"], group=p["group"], cohort="stage177_held", layer=p["ranked"], window=event.LONG,
                             candidate_raw=s["ranked_minus_native_raw"], candidate_post=s["ranked_minus_native_post"],
                             benefit=row["protected_benefit"] == "True", harm=row["event_harm"] == "True",
                             delta128_or_shorter=float(row["delta_window"]), selector_raw=-s["selector_gap_raw"],
                             selector_post=-s["selector_gap_post"]))
    for row in read(177, "sentinels"):
        p = by177[row["case_id"]]
        s = by_id[p["id"]]
        sentinels.append(dict(s, archived_harm=row["archived_event_harm"],
                              archived_prevented_terminal=row["archived_prevented_terminal"],
                              archived_delta=row["archived_gain128"]))
    return outcomes, alternatives, sentinels


def pair_concordance(rows, score, endpoint, block):
    pairs, by_source = 0, defaultdict(list)
    groups = defaultdict(list)
    for row in rows:
        groups[row[block]].append(row)
    for group in groups.values():
        values = []
        for a, b in combinations(group, 2):
            truth = float(a[endpoint]) - float(b[endpoint])
            if abs(truth) <= 1e-12:
                continue
            diff = a[score] - b[score]
            if endpoint == "harm":
                diff = -diff
            values.append(.5 if abs(diff) <= 1e-12 else float(diff * truth > 0.))
        if values:
            pairs += len(values)
            by_source[group[0]["group"]].append(float(np.mean(values)))
    return pairs, len(by_source), (float(np.mean([np.mean(v) for v in by_source.values()])) if by_source else None)


def analyze(signals, outcomes, alternatives, exact_runtime):
    banks, support = [], []
    for cohort in ("all", "stage177_scans", "stage176_only"):
        rows = [r for r in signals if cohort == "all" or (cohort == "stage177_scans") == (r["scan_cohort"] is not None)]
        for name in ("selector_gap_raw", "selector_gap_post", "ranked_minus_native_raw", "ranked_minus_native_post"):
            values = np.array([r[name] for r in rows], float)
            banks.append(dict(cohort=cohort, signal=name, states=len(rows), distinct=len(np.unique(values)),
                              minimum=float(values.min()), median=float(np.median(values)), maximum=float(values.max()),
                              std=float(values.std())))
    for cohort, a, b in (("stage176_all", outcomes, alternatives), ("stage176_other", outcomes, alternatives),
                         ("stage177_held", outcomes, [])):
        cohort_keys = {"stage176_known", "stage176_other"} if cohort == "stage176_all" else {cohort}
        for kind, rows, metrics, block in (("ranked_across_states", a, ("selector_raw", "selector_post", "candidate_raw", "candidate_post"), "group"),
                                           ("alternatives_same_state", b, ("candidate_raw", "candidate_post"), "case_id")):
            rows = [r for r in rows if r["window"] == event.LONG and r["cohort"] in cohort_keys]
            if not rows:
                continue
            for score in metrics:
                for endpoint in ("benefit", "harm", "delta128_or_shorter"):
                    pairs, groups, concordance = pair_concordance(rows, score, endpoint, block) if exact_runtime else (0, 0, None)
                    support.append(dict(cohort=cohort, kind=kind, signal=score, endpoint=endpoint, rows=len(rows),
                                        benefit_labels=sum(r["benefit"] for r in rows), harm_labels=sum(r["harm"] for r in rows),
                                        comparable_pairs=pairs, supported_source_groups=groups, concordance=concordance))
    return banks, support


def run(args, root, directories):
    r76, r89, r93, r121, r167, r170, r176, r177 = directories
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    p177 = c.c66.sealed(r177 / "manifest.json", reopening.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d176 = c.c66.sealed(r176 / "stage176_decision.json", sweep.SCHEMA)
    d177 = c.c66.sealed(r177 / "stage177_decision.json", reopening.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    if (d170["fingerprint"] != event.matched.SOURCE170 or d176["fingerprint"] != reopening.SOURCE176
            or d177["fingerprint"] != SOURCE177 or d167["fingerprint"] != previous.SOURCE167
            or d170["plan_fingerprint"] != p170["fingerprint"] or d177["plan_fingerprint"] != p177["fingerprint"]
            or any(x["runtime"] != d170["runtime"] for x in (p170, d176, d177))):
        raise ValueError("Need the sealed Stage-167, Stage-170, Stage-176 and Stage-177 sources")
    roots, decisions = {176: r176, 177: r177}, {176: d176, 177: d177}

    def read(stage, table):
        return risk.read_verified(roots[stage], decisions[stage], stage, table)

    cases = source_cases(read)
    scan_plan = {r["id"]: r for r in p177["candidates"]}
    for case in cases:
        if case["stage177_id"]:
            prior = scan_plan[case["stage177_id"]]
            if any(case[k] != prior[k] for k in ("sequence", "frame", "source", "group")):
                raise ValueError(f"Archived scan identity changed at {case['id']}")
    _, _, jobs = history.native_jobs(r121)
    fold_of = {g: f["fold"] for f in p170["folds"] for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for seq in sorted({r["sequence"] for r in cases}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        size = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["frames"]) != size + 1
                or any(not 1 <= r["frame"] <= size for r in cases if r["sequence"] == seq)):
            raise ValueError(f"Source image roster differs at {seq}")
        arm = event.matched.COLD
        data = previous.archive_trace(r170, p170, d170, seq, arm)
        previous.audit(data, size, entry["initial_box"], arm, thresholds[fold_of[risk.SOURCE_GROUP(seq)]])
        entries[seq], sources[seq] = entry, data

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    plan = io.seal(dict(schema=SCHEMA, source_stage170=d170["fingerprint"], source_stage176=d176["fingerprint"],
                        source_stage177=SOURCE177, cases=cases, runtime=runtime, runtime_differences=differences,
                        hook="backbone.MLP.fc2", code_sha256=c.file_hash(Path(__file__))))
    c.publish_identical(root / "manifest.json", plan)
    (root / "observed").mkdir(exist_ok=True)

    def location(case):
        return root / "observed" / f"{case['id']}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=case["id"])

    reused = sum(location(case)[0].with_suffix(".receipt.json").exists() for case in cases)
    progress = OverallProgress("Stage-178", len(cases) + 1, reused, args.progress_interval, "work")
    progress.report("Loading frozen native tracker")
    args.parameter = runtime["tracker"]["parameter"]
    bound = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, bound, root)
    signals, manifests, templates = [], {}, {}
    progress.report("Capturing pre-Sigmoid logits")
    with ExitStack() as stack:
        observer = LogitObserver(native, stack)
        for case in cases:
            seq, frame = case["sequence"], case["frame"]
            entry = entries[seq]
            path, metadata = location(case)
            out = c.read_cache(path, metadata)
            fresh = out is None
            if fresh:
                if seq not in templates:
                    templates[seq] = online.checked_image(old["root"], entry["frames"][0])
                image = online.checked_image(old["root"], entry["frames"][frame])
                out = observer.observe(templates[seq], image, entry["initial_box"], sources[seq]["parent"][frame - 1], frame)
                progress.advance()
            refs = references(case, r176, r177, d176, d177)
            row = signal_row(case, out, refs, not differences)
            if fresh:
                c.save(path, {k: np.asarray(v) for k, v in out.items()}, metadata)
            for p in (path, path.with_suffix(".receipt.json")):
                manifests[p.relative_to(root).as_posix()] = c.file_hash(p)
            signals.append(row)
    native.assert_frozen()
    outcomes, alternatives, sentinels = join_effects(cases, signals, read)
    dispersion, association = analyze(signals, outcomes, alternatives, not differences)
    tables = dict(cases=cases, signals=signals, dispersion=dispersion, ranked_outcomes=outcomes,
                  alternative_outcomes=alternatives, association=association, sentinels=sentinels)
    if progress.done != len(cases):
        raise ValueError("Incomplete Stage-178 capture progress")
    captured = [r for r in signals if r["scan_cohort"] is not None]
    return dict(plan_fingerprint=plan["fingerprint"], unique_anchors=len(cases), stage177_anchors=len(captured),
                stage176_anchors=sum(r["stage176_id"] is not None for r in cases), native_searches=len(cases),
                stage177_distinct_raw_gaps=len({r["selector_gap_raw"] for r in captured}),
                stage177_distinct_post_gaps=len({r["selector_gap_post"] for r in captured}),
                association_same_runtime=not differences, training_performed=False, new_rollouts=0, work=len(cases) + 1,
                runtime=runtime, runtime_differences=differences, artifact_sha256=manifests,
                table_sha256=gain.prior.publish(root, tables, stage="stage178")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=2, type=Path, metavar=("STAGE176", "STAGE177"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage178_selector_logits_audit")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        values = ((176, reopening.SOURCE176), (177, SOURCE177))
        tabs = {s: gain.prior.archives.bundle(path, s, fp)[1] for (s, fp), path in zip(values, args.bundles, strict=True)}
        cases = source_cases(lambda s, name: tabs[s][f"stage{s}_{name}.csv"])
        print(f"Verified {len(cases)} unique states for single-frame capture, including "
              f"{sum(p['stage176_id'] is not None and p['stage177_id'] is not None for p in cases)} shared states")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage178_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage177"] != SOURCE177 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-178 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-178", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE178_SELECTOR_LOGITS_AUDIT",
                                              source_stage177=SOURCE177, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
