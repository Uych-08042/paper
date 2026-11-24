"""Audit current-frame native CENTER evidence on previously labeled layer interventions."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage178_selector_logits_audit as prior
from stage28_response_common import continuous_iou, dense_boxes, peak_candidates
from stage27_uav_common import crop_geometry


event, risk, online, previous = prior.event, prior.risk, prior.online, prior.previous
base, gain, history, c, io = prior.base, prior.gain, prior.history, prior.c, prior.io
SOURCE178 = "941317bf0105565e62e01ee709be16c10ea595cfc6d7ac56403b3d7e9cb72630"
SCHEMA = "stage180_center_match_audit_v1"
SOURCES = (76, 89, 93, 121, 167, 170, 176, 177, 178)
FEATURES = ("peak_competition", "regression_drop", "neighbor_disagreement", "combined", "uniform")
NEAR_SUPPORT = .5
BOX_ATOL = .001


def parse_cases(rows):
    cases = [dict(r, frame=int(r["frame"]), native=int(r["native"]), ranked=int(r["ranked"])) for r in rows]
    if (len(cases) != 265 or len({r["id"] for r in cases}) != len(cases)
            or any(r["source"] != event.matched.COLD for r in cases)):
        raise ValueError("Need the 265 sealed matched Stage-178 anchor states")
    return cases


class CenterObserver(online.RoutedObserver):
    def capture_peak(self, module, args, output):
        if self.maps is not None:
            raise ValueError("Expected exactly one native CENTER output")
        super().capture_peak(module, args, output)
        side = self.rt.search_side
        self.maps = {name: output[key].detach().float().cpu().numpy().reshape(shape)
                     for name, key, shape in (("raw", "score_map", (side, side)),
                                              ("size", "size_map", (2, side, side)),
                                              ("offset", "offset_map", (2, side, side)))}

    def observe(self, template, image, initial_box, parent, frame):
        self.maps = None
        self.initialize(template, initial_box, "native", None)
        self.rt.tracker.frame_id = frame - 1
        row = self.step(image, parent, frame)
        if self.maps is None:
            raise ValueError("Missing native CENTER maps")
        tracker = self.rt.tracker
        search_size = int(tracker.params.search_size)
        image_hw = np.asarray(image.shape[:2])
        crop, _, scale = crop_geometry(parent, image_hw, tracker.params.search_factor, search_size)
        window = tracker.output_window.detach().float().cpu().numpy().reshape(self.maps["raw"].shape)
        return dict(row, **self.maps, window=window,
                    factor=scale, search_size=search_size, image_hw=image_hw, crop=crop)


def interpolate(matrix, x, y):
    h, w = matrix.shape
    if not 0 <= x <= w - 1 or not 0 <= y <= h - 1:
        return 0.
    i, j = int(y), int(x)
    dy, dx = y - i, x - j
    q, r = min(i + 1, h - 1), min(j + 1, w - 1)
    return float((1. - dy) * ((1. - dx) * matrix[i, j] + dx * matrix[i, r])
                 + dy * ((1. - dx) * matrix[q, j] + dx * matrix[q, r]))


def center_features(data):
    raw, window = (np.asarray(data[k], np.float32) for k in ("raw", "window"))
    if (raw.ndim != 2 or raw.shape != window.shape or not np.isfinite(raw).all() or not np.isfinite(window).all()
            or (raw < 0.).any() or (window < 0.).any()):
        raise ValueError("Invalid native CENTER response or Hann window")
    weighted = raw * window
    flat = weighted.ravel()
    grid = dense_boxes(data["size"], data["offset"], data["parent"], float(data["factor"]), int(data["search_size"]), data["image_hw"])
    top = int(flat.argmax())
    peak = float(flat[top])
    if peak <= 0. or abs(peak - float(data["hann_peak"])) > 1e-6:
        raise ValueError("Native Hann peak differs from the tracked result")
    decode_error = float(np.max(np.abs(grid[top] - data["boxes"])))
    if decode_error > BOX_ATOL:
        raise ValueError(f"Dense CENTER box differs from the native decoder: {decode_error}")
    grid[top] = data["boxes"]
    peaks = peak_candidates(weighted, grid)
    second = peaks[1] if len(peaks) > 1 else None
    competition = float(flat[second]) / peak if second is not None else 0.
    y, x = np.unravel_index(top, weighted.shape)
    dx, dy = map(float, data["offset"][:, y, x])
    refined_score = interpolate(weighted, x + dx, y + dy)
    regression_drop = float(np.clip(1. - refined_score / peak, 0., 1.))
    yy, xx = np.indices(raw.shape)
    mask = (np.abs(xx - x) <= 1) & (np.abs(yy - y) <= 1) & (weighted >= NEAR_SUPPORT * peak)
    mask[y, x] = False
    nearby = np.flatnonzero(mask)
    disagreement = float(np.average(np.clip(1. - continuous_iou(grid[top], grid[nearby]), 0., 1.),
                                    weights=flat[nearby])) if len(nearby) else 0.
    raw_top = int(raw.argmax())
    raw_y, raw_x = np.unravel_index(raw_top, raw.shape)
    return dict(native_cell=top, raw_cell=raw_top, second_cell=second, hann_peak=peak,
                second_peak=float(flat[second]) if second is not None else None,
                peak_competition=competition, main_second_iou=float(continuous_iou(grid[top], grid[second])) if second is not None else None,
                regression_drop=regression_drop, offset_cells=float(np.hypot(dx, dy)),
                neighbor_support=len(nearby), neighbor_disagreement=disagreement,
                raw_hann_separation_cells=float(np.hypot(raw_x - x, raw_y - y)),
                combined=(competition + regression_drop + disagreement) / 3., uniform=0., dense_decode_error=decode_error)


def audit(case, trace, archived, exact_runtime):
    layer = int(trace["layer"])
    if int(trace["frame_id"]) != case["frame"] or layer != int(trace["native_layer"]) or layer not in online.LAYERS:
        raise ValueError("The single-frame probe did not follow native routing")
    for k in ("crop_calls", "head_calls", "selector_calls"):
        np.testing.assert_equal(trace[k], 1)
    blocks = np.zeros(12, int)
    blocks[:6] = 1
    blocks[layer - 1] = 1
    np.testing.assert_array_equal(trace["block_calls"], blocks)
    np.testing.assert_array_equal(trace["parent"], archived["parent"])
    error = float(np.max(np.abs(trace["boxes"] - archived["boxes"])))
    peak_error = abs(float(trace["hann_peak"]) - float(archived["hann_peak"]))
    if exact_runtime and (error != 0. or peak_error != 0. or layer != case["native"]):
        raise ValueError(f"Stage-178 original prediction changed at {case['id']}")
    return dict(case_id=case["id"], sequence=case["sequence"], group=case["group"], frame=case["frame"],
                scan_cohort=case["scan_cohort"], layer=layer, native_box_error=error, hann_peak_error=peak_error,
                **center_features(trace))


def archived_labels(cases, ranked, alternatives, layers176, outcomes177):
    states = {p["id"]: p for p in cases}
    layer_flags = {(r["case_id"], int(r["layer"])): r for r in layers176}
    flags177 = {r["case_id"]: r for r in outcomes177}
    if len(ranked) != 61 or len(alternatives) != 130 or len(layer_flags) != 156 or len(flags177) != 35:
        raise ValueError("Incomplete Stage-176/177/178 label roster")
    def typed(row):
        p = states[row["case_id"]]
        if row["group"] != p["group"]:
            raise ValueError("Label source group changed")
        stage177 = row["cohort"] == "stage177_held"
        flag = flags177[p["stage177_id"]] if stage177 else layer_flags[p["stage176_id"], int(row["layer"])]
        benefit, harm = row["benefit"] == "True", row["harm"] == "True"
        if benefit != (flag["protected_benefit"] == "True") or harm != (flag["event_harm"] == "True"):
            raise ValueError("Archived event outcome changed")
        return dict(case_id=p["id"], group=p["group"], cohort=row["cohort"], window=int(row["window"]),
                    layer=int(row["layer"]), benefit=benefit, harm=harm,
                    protected_rescue=benefit and (flag["prevented_terminal"] == "True" or flag["rescued_recovery"] == "True"),
                    new_terminal_without_inherited_loss=flag["new_terminal"] == "True" and flag["inherited_failure"] == "False",
                    delta128=float(row["delta128_or_shorter"]))

    choices, others = list(map(typed, ranked)), list(map(typed, alternatives))
    pools = {}
    for r in others:
        p = pools.setdefault(r["case_id"], dict(case_id=r["case_id"], group=r["group"], cohort=r["cohort"],
                                                  window=r["window"], layers=0, benefit_available=False, rescue_available=False))
        p["layers"] += 1
        p["benefit_available"] |= r["benefit"]
        p["rescue_available"] |= r["protected_rescue"]
    if len(pools) != 26 or any(r["layers"] != 5 for r in pools.values()):
        raise ValueError("Need all five nonnative alternatives in each Stage-176 state")
    return choices, list(pools.values())


def analyze(signals, choices, pools, exact_runtime):
    index = {r["case_id"]: r for r in signals}
    def merge(rows):
        return [dict(r, **{k: index[r["case_id"]][k] for k in FEATURES}) for r in rows]

    choices, pools = merge(choices), merge(pools)
    stats = []
    for scope, cohorts in (("stage176_all", ("stage176_other", "stage176_known")), ("stage176_other", ("stage176_other",)),
                           ("stage177_held", ("stage177_held",))):
        for kind, labels, metrics in (("ranked", choices, ("benefit", "protected_rescue", "harm",
                           "new_terminal_without_inherited_loss", "delta128")),
                                     ("available", pools, ("benefit_available", "rescue_available"))):
            rows = [r for r in labels if r["cohort"] in cohorts and r["window"] == event.LONG]
            if not rows:
                continue
            for score in FEATURES:
                for k in metrics:
                    # High ambiguity motivates trying a backup. For harm the hypothesis is reversed.
                    target = "harm" if k in ("harm", "new_terminal_without_inherited_loss") else k
                    inputs = rows if target == k else [dict(r, harm=r[k]) for r in rows]
                    pairs, groups, value = prior.pair_concordance(inputs, score, target, "group") if exact_runtime else (0, 0, None)
                    stats.append(dict(cohort=scope, kind=kind, signal=score, outcome=k, rows=len(rows),
                                      positive_labels=sum(r[k] for r in rows) if k != "delta128" else None,
                                      pairs=pairs, source_groups=groups, concordance=value))
    dispersion = []
    for name in ("peak_competition", "regression_drop", "neighbor_disagreement", "combined", "raw_hann_separation_cells"):
        x = np.asarray([r[name] for r in signals], float)
        dispersion.append(dict(signal=name, states=len(x), distinct=len(np.unique(x)), minimum=float(x.min()),
                               median=float(np.median(x)), maximum=float(x.max())))
    return choices, pools, stats, dispersion


def input_labels(read):
    cases = parse_cases(read(178, "cases"))
    choices, pools = archived_labels(cases, read(178, "ranked_outcomes"), read(178, "alternative_outcomes"),
                                     read(176, "layers"), read(177, "outcomes"))
    sentinels = {r["case_id"]: r for r in read(178, "sentinels")}
    if len(sentinels) != 4:
        raise ValueError("Need all four archived sentinel labels")
    return cases, choices, pools, sentinels


def run(args, root, directories):
    r76, r89, r93, r121, r167, r170, r176, r177, r178 = directories
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    p178 = c.c66.sealed(r178 / "manifest.json", prior.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    roots = {176: r176, 177: r177, 178: r178}
    schemas = {176: (prior.sweep.SCHEMA, prior.reopening.SOURCE176), 177: (prior.reopening.SCHEMA, prior.SOURCE177),
               178: (prior.SCHEMA, SOURCE178)}
    receipts = {s: c.c66.sealed(roots[s] / f"stage{s}_decision.json", schema) for s, (schema, _) in schemas.items()}
    d178 = receipts[178]
    if (d170["fingerprint"] != event.matched.SOURCE170 or d167["fingerprint"] != previous.SOURCE167
            or d170["plan_fingerprint"] != p170["fingerprint"] or d178["plan_fingerprint"] != p178["fingerprint"]
            or p170["runtime"] != d170["runtime"] or not d178["association_same_runtime"]
            or any(receipts[s]["fingerprint"] != fp or receipts[s]["runtime"] != d170["runtime"]
                   for s, (_, fp) in schemas.items())):
        raise ValueError("Need the sealed same-runtime Stage-167/170/176/177/178 sources")

    def read(stage, name):
        return risk.read_verified(roots[stage], receipts[stage], stage, name)

    cases, choices, pools, sentinels = input_labels(read)
    keys = {k["id"]: k for k in p178["cases"]}
    if len(keys) != len(cases) or any(any(p[k] != keys[p["id"]][k] for k in ("sequence", "frame", "group", "source"))
                                      for p in cases):
        raise ValueError("Stage-178 measurement roster differs from the manifest")
    _, _, jobs = history.native_jobs(r121)
    fold = {g: row["fold"] for row in p170["folds"] for g in row["held"]}
    threshold = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources = {}, {}
    for seq in sorted({r["sequence"] for r in cases}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        size = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["frames"]) != size + 1
                or any(not 1 <= r["frame"] <= size for r in cases if r["sequence"] == seq)):
            raise ValueError(f"Invalid source image roster at {seq}")
        data = previous.archive_trace(r170, p170, d170, seq, event.matched.COLD)
        previous.audit(data, size, entry["initial_box"], event.matched.COLD, threshold[fold[risk.SOURCE_GROUP(seq)]])
        entries[seq], sources[seq] = entry, data

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d178["runtime"], runtime)
    code = (Path(__file__), Path(prior.__file__), Path(dense_boxes.__code__.co_filename))
    plan = io.seal(dict(schema=SCHEMA, source_stage170=d170["fingerprint"],
                        source_reports={str(s): receipts[s]["fingerprint"] for s in receipts}, cases=cases,
                        runtime=runtime, runtime_differences=differences, features=list(FEATURES),
                        near_support=NEAR_SUPPORT, decoder_atol=BOX_ATOL,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "observed").mkdir(exist_ok=True)

    def location(case):
        return root / "observed" / f"{case['id']}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=case["id"])

    reused = sum(location(case)[0].with_suffix(".receipt.json").exists() for case in cases)
    progress = OverallProgress("Stage-180", len(cases) + 1, reused, args.progress_interval, "work")
    progress.report("Loading frozen native tracker")
    args.parameter = runtime["tracker"]["parameter"]
    native, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                  runtime_identity=runtime, runtime=runtime["tracker"]), root)
    signals, artifacts, templates = [], {}, {}
    progress.report("Capturing current-frame CENTER outputs")
    with ExitStack() as stack:
        observer = CenterObserver(native, stack)
        for case in cases:
            seq, frame, name = case["sequence"], case["frame"], case["id"]
            entry = entries[seq]
            path, metadata = location(case)
            observed = c.read_cache(path, metadata)
            fresh = observed is None
            if fresh:
                if seq not in templates:
                    templates[seq] = online.checked_image(old["root"], entry["frames"][0])
                image = online.checked_image(old["root"], entry["frames"][frame])
                observed = observer.observe(templates[seq], image, entry["initial_box"], sources[seq]["parent"][frame - 1], frame)
                progress.advance()
            archived = prior.cache_reference(r178, d178, f"observed/{name}.npz",
                                            dict(plan_fingerprint=d178["plan_fingerprint"], case_id=name))
            signals.append(audit(case, observed, archived, not differences))
            if fresh:
                c.save(path, {k: np.asarray(v) for k, v in observed.items()}, metadata)
            for f in (path, path.with_suffix(".receipt.json")):
                artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
    native.assert_frozen()
    ranked, available, association, dispersion = analyze(signals, choices, pools, not differences)
    watch = [dict(r, archived_harm=sentinels[r["case_id"]]["archived_harm"],
                  archived_prevented_terminal=sentinels[r["case_id"]]["archived_prevented_terminal"],
                  archived_delta=sentinels[r["case_id"]]["archived_delta"]) for r in signals if r["case_id"] in sentinels]
    tables = dict(cases=cases, signals=signals, dispersion=dispersion, ranked_outcomes=ranked,
                  alternative_availability=available, association=association, sentinels=watch)
    if progress.done != len(cases):
        raise ValueError("Incomplete Stage-180 progress accounting")
    return dict(plan_fingerprint=plan["fingerprint"], states=len(cases), native_searches=len(cases),
                new_rollouts=0, training_performed=False, association_same_runtime=not differences,
                work=len(cases) + 1, runtime=runtime, runtime_differences=differences,
                artifact_sha256=artifacts, table_sha256=gain.prior.publish(root, tables, stage="stage180")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=3, type=Path, metavar=("STAGE176", "STAGE177", "STAGE178"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage180_center_match_audit")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        bindings = ((176, prior.reopening.SOURCE176), (177, prior.SOURCE177), (178, SOURCE178))
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1] for p, (s, fp) in zip(args.bundles, bindings, strict=True)}
        cases, choices, pools, sentinels = input_labels(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        print(f"Verified {len(cases)} capture states, {len(choices)} ranked labels, "
              f"{len(pools)} all-candidate states and {len(sentinels)} sentinels")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage180_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage178"] != SOURCE178 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-180 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-180", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE180_CENTER_MATCH_AUDIT",
                                              source_stage178=SOURCE178, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
