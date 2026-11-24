"""Compare native and fixed alternative CENTER responses on a shared L6 state."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_progress import OverallProgress
import stage180_center_match_audit as center
from stage28_response_common import continuous_iou


prior, event, risk, online, previous = center.prior, center.event, center.risk, center.online, center.previous
base, gain, history, c, io = center.base, center.gain, center.history, center.c, center.io
SOURCE180 = "47a3ca149e922fc3f25ce11fb976c57ad0752aa99bbc3a9810db42c8e25be3f2"
SCHEMA = "stage181_paired_center_audit_v1"
SOURCES = (76, 89, 93, 121, 167, 170, 176, 177, 180)
COMPONENTS = ("peak_competition", "regression_drop", "neighbor_disagreement")
FEATURES = ("native_peak_competition", "native_ambiguity", "candidate_clarity", "structure_gain",
            *(f"gain_{k}" for k in COMPONENTS), "hann_peak_gain", "parent_iou_gain", "branch_iou", "uniform")
WATCH = {("uav0000089_00920_s", 1177), ("uav0000204_00000_s", 663)}


def select_cases(cases, labels):
    roster = {r["id"]: r for r in center.parse_cases(cases)}
    if len(labels) != 61 or len({r["case_id"] for r in labels}) != len(labels):
        raise ValueError("Need all 61 ranked outcomes of Stage-180")
    result = []
    for label in labels:
        case = roster[label["case_id"]]
        if label["group"] != case["group"] or int(label["layer"]) != case["ranked"] or case["ranked"] == case["native"]:
            raise ValueError("Stage-180 ranked outcome is not matched to the archived state")
        stage = 177 if label["cohort"] == "stage177_held" else 176
        archive_id = case[f"stage{stage}_id"]
        if not archive_id:
            raise ValueError("Missing the original alternative replay")
        result.append(dict(case, archive_stage=stage, archive_id=archive_id, cohort=label["cohort"],
                           window=int(label["window"]), watch=(case["sequence"], case["frame"]) in WATCH))
    if sum(r["watch"] for r in result) != len(WATCH):
        raise ValueError("Missing the archived positive/negative replay checks")
    return result


class SharedCenterObserver(center.CenterObserver):
    def __init__(self, native, stack):
        self.shared = None
        super().__init__(native, stack)
        stack.callback(self.rt.backbone.blocks[5].register_forward_hook(self.save_shared).remove)

    def save_shared(self, module, args, output):
        if self.shared is not None:
            raise ValueError("Expected a single L6 state on the original crop")
        self.shared = output.detach().clone()

    def observe_pair(self, template, image, initial_box, parent, frame, layer):
        self.shared = None
        self.maps = None
        self.initialize(template, initial_box, "native", None)
        tracker = self.rt.tracker
        tracker.frame_id = frame - 1
        namespace = tracker.track.__func__.__globals__
        sample = namespace["sample_target"]
        factors = []

        def record_crop(*args, **kwargs):
            np.testing.assert_array_equal(args[1], parent)
            result = sample(*args, **kwargs)
            factors.append(float(result[1]))
            return result

        with patch.dict(namespace, sample_target=record_crop):
            row = self.step(image, parent, frame)
            if self.shared is None or self.maps is None or len(factors) != 1:
                raise ValueError("Missing shared L6 state, native maps or source crop")
            search_size = int(tracker.params.search_size)
            image_hw = np.asarray(image.shape[:2])
            crop, _, scale = center.crop_geometry(parent, image_hw, tracker.params.search_factor, search_size)
            window = tracker.output_window.detach().float().cpu().numpy().reshape(self.maps["raw"].shape)
            native = dict(row, **self.maps, window=window, factor=scale, search_size=search_size, image_hw=image_hw, crop=crop)
            np.testing.assert_allclose(factors[0], scale, rtol=0., atol=1e-12)
            alternate = self.observe_alternate(image, parent, layer, factors[0], native)
            if len(factors) != 1 or tracker.frame_id != frame:
                raise ValueError("The extra head changed the source crop or frame index")
            np.testing.assert_array_equal(tracker.state, native["boxes"])
        return {**{f"native_{k}": np.asarray(v) for k, v in native.items()},
                **{f"alternate_{k}": np.asarray(v) for k, v in alternate.items()}}

    def observe_alternate(self, image, parent, layer, factor, native):
        rt = self.rt
        modules = [*rt.backbone.blocks, rt.backbone.MLP, rt.net.box_head]
        counts = np.zeros(len(modules), int)

        def count(index):
            def hook(*_):
                counts[index] += 1
            return hook

        tracked = rt.tracker.state
        with ExitStack() as stack:
            for idx, module in enumerate(modules):
                stack.callback(module.register_forward_hook(count(idx)).remove)
            stack.enter_context(rt.torch.no_grad())
            out = rt.candidate(self.shared, layer)
            try:
                rt.tracker.state = np.asarray(parent, float).tolist()
                box = rt.decode(out, factor, image)
            finally:
                rt.tracker.state = tracked
            side = rt.search_side
            maps = {k: out[name].detach().float().cpu().numpy().reshape(shape)
                    for k, name, shape in (("raw", "score_map", (side, side)),
                                           ("size", "size_map", (2, side, side)),
                                           ("offset", "offset_map", (2, side, side)))}
            peak = float((out["score_map"] * rt.tracker.output_window).amax().item())
        return dict(maps, layer=layer, factor=factor, boxes=np.asarray(box, float), hann_peak=peak,
                    parent=np.asarray(parent, float), window=native["window"], search_size=native["search_size"],
                    image_hw=native["image_hw"], counts=counts)


def compare(case, pair, archived_native, archived_alternative, exact_runtime):
    native = {k.removeprefix("native_"): v for k, v in pair.items() if k.startswith("native_")}
    alternative = {k.removeprefix("alternate_"): v for k, v in pair.items() if k.startswith("alternate_")}
    n = center.audit(case, native, archived_native, exact_runtime)
    expected = np.zeros(14, int)
    expected[case["ranked"] - 1], expected[-1] = 1, 1
    np.testing.assert_array_equal(alternative["counts"], expected)
    np.testing.assert_array_equal(alternative["parent"], archived_alternative["parent"][0])
    if (int(alternative["layer"]) != case["ranked"] or int(archived_alternative["layer"][0]) != case["ranked"]
            or int(archived_alternative["frame_id"][0]) != case["frame"]):
        raise ValueError(f"Wrong alternative branch at {case['id']}")
    native_map_error = max(float(np.abs(native[k] - archived_native[k]).max()) for k in ("raw", "size", "offset"))
    alt_box_error = float(np.abs(alternative["boxes"] - archived_alternative["boxes"][0]).max())
    alt_peak_error = abs(float(alternative["hann_peak"]) - float(archived_alternative["hann_peak"][0]))
    if exact_runtime and (native_map_error != 0. or alt_box_error > center.BOX_ATOL or alt_peak_error > 1e-6):
        raise ValueError(f"Archived CENTER branch changed at {case['id']}: {native_map_error}, {alt_box_error}, {alt_peak_error}")
    a = center.center_features(alternative)
    peak_gain = (a["hann_peak"] - n["hann_peak"]) / max(a["hann_peak"], n["hann_peak"])
    parent_gain = float(continuous_iou(alternative["boxes"], native["parent"]) - continuous_iou(native["boxes"], native["parent"]))
    branch_iou = float(continuous_iou(native["boxes"], alternative["boxes"]))
    return dict(case_id=case["id"], group=case["group"], sequence=case["sequence"], frame=case["frame"], cohort=case["cohort"],
                window=case["window"], native=int(native["layer"]), alternative=case["ranked"],
                native_map_error=native_map_error, native_box_error=n["native_box_error"],
                alternative_box_error=alt_box_error, alternative_peak_error=alt_peak_error,
                native_ambiguity=n["combined"], candidate_clarity=-a["combined"], structure_gain=n["combined"] - a["combined"],
                hann_peak_gain=peak_gain, parent_iou_gain=parent_gain, branch_iou=branch_iou, uniform=0.,
                **{f"native_{k}": n[k] for k in (*COMPONENTS, "hann_peak", "native_cell")},
                **{f"alternate_{k}": a[k] for k in (*COMPONENTS, "hann_peak", "native_cell")},
                **{f"gain_{k}": n[k] - a[k] for k in COMPONENTS})


def analyze(pairs, labels, exact_runtime):
    by_id = {r["case_id"]: r for r in labels}
    matches = []
    for p in pairs:
        r = by_id[p["case_id"]]
        matches.append(dict(p, **{k: r[k] == "True" for k in ("benefit", "harm", "protected_rescue", "new_terminal_without_inherited_loss")},
                            delta128=float(r["delta128"])))
    summary = []
    for scope, cohorts in (("stage176_all", {"stage176_known", "stage176_other"}), ("stage176_other", {"stage176_other"}),
                           ("stage177_held", {"stage177_held"})):
        rows = [r for r in matches if r["cohort"] in cohorts and r["window"] == event.LONG]
        for score in FEATURES:
            for endpoint in ("benefit", "protected_rescue", "harm", "new_terminal_without_inherited_loss", "delta128"):
                target = "harm" if endpoint in ("harm", "new_terminal_without_inherited_loss") else endpoint
                # More overlap means motion continuity; test it as protection, like the other relative scores.
                scored = rows if target == endpoint else [dict(r, harm=r[endpoint]) for r in rows]
                comparisons, groups, value = prior.pair_concordance(scored, score, target, "group") if exact_runtime else (0, 0, None)
                summary.append(dict(cohort=scope, signal=score, outcome=endpoint, records=len(rows),
                                    positive_labels=sum(r[endpoint] for r in rows) if endpoint != "delta128" else None,
                                    pairs=comparisons, source_groups=groups, concordance=value))
    return matches, summary


def archived_alternative(case, roots, decisions):
    stage, name = case["archive_stage"], case["archive_id"]
    receipt = decisions[stage]
    if stage == 176:
        layer = case["ranked"]
        suffix = f"l{layer:02d}"
        meta = dict(plan_fingerprint=receipt["plan_fingerprint"], case_id=name, layer=layer)
    else:
        suffix = "ranked"
        meta = dict(collection_fingerprint=receipt["collection_fingerprint"], case_id=name, arm=suffix)
    return prior.cache_reference(roots[stage], receipt, f"rollouts/{name}__{suffix}.npz", meta)


def run(args, root, paths):
    r76, r89, r93, r121, r167, r170, r176, r177, r180 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    p180 = c.c66.sealed(r180 / "manifest.json", center.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    roots = {176: r176, 177: r177, 180: r180}
    specs = {176: (prior.sweep.SCHEMA, prior.reopening.SOURCE176),
             177: (prior.reopening.SCHEMA, prior.SOURCE177), 180: (center.SCHEMA, SOURCE180)}
    decisions = {s: c.c66.sealed(roots[s] / f"stage{s}_decision.json", schema) for s, (schema, _) in specs.items()}
    d180 = decisions[180]
    if (d170["fingerprint"] != event.matched.SOURCE170 or d167["fingerprint"] != previous.SOURCE167
            or d170["plan_fingerprint"] != p170["fingerprint"] or d180["plan_fingerprint"] != p180["fingerprint"]
            or p170["runtime"] != d170["runtime"] or not d180["association_same_runtime"]
            or any(decisions[s]["fingerprint"] != fp or decisions[s]["runtime"] != d170["runtime"]
                   for s, (_, fp) in specs.items())):
        raise ValueError("Need the sealed same-runtime Stage-170/176/177/180 sources")
    labels = risk.read_verified(r180, d180, 180, "ranked_outcomes")
    cases = select_cases(risk.read_verified(r180, d180, 180, "cases"), labels)
    _, _, jobs = history.native_jobs(r121)
    fold_of = {g: f["fold"] for f in p170["folds"] for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for seq in sorted({r["sequence"] for r in cases}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        frames = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["frames"]) != frames + 1
                or any(not 1 <= r["frame"] <= frames for r in cases if r["sequence"] == seq)):
            raise ValueError(f"Source video or frame count differs at {seq}")
        data = previous.archive_trace(r170, p170, d170, seq, event.matched.COLD)
        previous.audit(data, frames, entry["initial_box"], event.matched.COLD, thresholds[fold_of[risk.SOURCE_GROUP(seq)]])
        entries[seq], sources[seq] = entry, data

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d180["runtime"], runtime)
    code = (Path(__file__), Path(center.__file__), Path(prior.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage170=d170["fingerprint"], cases=cases,
                        source_reports={str(s): decisions[s]["fingerprint"] for s in decisions},
                        runtime=runtime, runtime_differences=differences, score_signals=list(FEATURES),
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "observed").mkdir(exist_ok=True)

    def location(case):
        return root / "observed" / f"{case['id']}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=case["id"])

    reused = sum(location(case)[0].with_suffix(".receipt.json").exists() for case in cases)
    progress = OverallProgress("Stage-181", len(cases) + 1, reused, args.progress_interval, "work")
    progress.report("Loading frozen native tracker")
    args.parameter = runtime["tracker"]["parameter"]
    native, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                  runtime_identity=runtime, runtime=runtime["tracker"]), root)
    pairs, artifacts, templates = [], {}, {}
    progress.report("Probing the original layer and one alternative on the same L6 state")
    with ExitStack() as stack:
        observer = SharedCenterObserver(native, stack)
        for case in cases:
            seq, frame, key = case["sequence"], case["frame"], case["id"]
            entry = entries[seq]
            path, metadata = location(case)
            pair = c.read_cache(path, metadata)
            fresh = pair is None
            if fresh:
                if seq not in templates:
                    templates[seq] = online.checked_image(old["root"], entry["frames"][0])
                image = online.checked_image(old["root"], entry["frames"][frame])
                pair = observer.observe_pair(templates[seq], image, entry["initial_box"],
                                             sources[seq]["parent"][frame - 1], frame, case["ranked"])
                progress.advance()
            prior_native = prior.cache_reference(r180, d180, f"observed/{key}.npz",
                                                 dict(plan_fingerprint=d180["plan_fingerprint"], case_id=key))
            prior_alternative = archived_alternative(case, roots, decisions)
            pairs.append(compare(case, pair, prior_native, prior_alternative, not differences))
            if fresh:
                c.save(path, pair, metadata)
            for f in (path, path.with_suffix(".receipt.json")):
                artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
    native.assert_frozen()
    outcomes, summary = analyze(pairs, labels, not differences)
    tables = dict(cases=cases, signals=pairs, outcomes=outcomes, association=summary,
                  sentinels=[r for r in outcomes if (r["sequence"], r["frame"]) in WATCH])
    if progress.done != len(cases):
        raise ValueError("Incomplete Stage-181 progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], states=len(cases),
                full128_states=sum(r["window"] == event.LONG for r in cases),
                crops=len(cases), native_searches=len(cases), extra_blocks=len(cases), extra_heads=len(cases),
                new_rollouts=0, training_performed=False, association_same_runtime=not differences,
                work=len(cases) + 1, runtime=runtime, runtime_differences=differences,
                artifact_sha256=artifacts, table_sha256=gain.prior.publish(root, tables, stage="stage181")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Verify the Stage-180 export without CUDA")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage181_paired_center_audit")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU source check")
        _, tables = gain.prior.archives.bundle(args.bundle, 180, SOURCE180)
        cases = select_cases(tables["stage180_cases.csv"], tables["stage180_ranked_outcomes.csv"])
        print(f"Verified {len(cases)} paired states, {sum(r['window'] == event.LONG for r in cases)} with full 128-frame labels")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage181_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage180"] != SOURCE180 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-181 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-181", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE181_PAIRED_CENTER_AUDIT",
                                             source_stage180=SOURCE180, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
