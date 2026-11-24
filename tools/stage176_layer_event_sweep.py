"""Replay all six SGLA anchor layers and measure protected long-window headroom."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage175_event_risk as event


risk, aligned, previous, online = event.risk, event.aligned, event.previous, event.online
base, gate, gain, history, c, io = event.base, event.gate, event.gain, event.history, event.c, event.io
SOURCE175 = "33214a0995ab082783b6016db51369d3baed77410a1d21b9b9e376c1edebc5c8"
SCHEMA = "stage176_layer_event_sweep_v1"
SOURCES = (*event.SOURCES, 175)
LAYERS = online.LAYERS
SHORT, LONG = event.SHORT, event.LONG
EPSILON, SEED = 0.01, 23176
REFERENCES = {173: (aligned.SCHEMA, risk.SOURCE173), 174: (risk.SCHEMA, event.SOURCE174),
              175: (event.SCHEMA, SOURCE175)}


def select_cases(read):
    prior = read(175, "cases")
    outcomes = {r["case_id"]: r for r in read(175, "outcomes")}
    if len(prior) != 139 or len(outcomes) != 139:
        raise ValueError("Need the sealed Stage-175 case and outcome roster")
    ordinary = defaultdict(list)
    for row in prior:
        if row["kind"] == "ordinary":
            ordinary[row["group"]].append(row["id"])
    sampled = {min(ids, key=lambda k: sha256(f"{SEED}/{k}".encode()).digest()) for ids in ordinary.values()}

    cases = []
    for row in prior:
        key = row["id"]
        result = outcomes[key]
        if result["valid"] != "True" or result["source_route_matches"] != "True":
            raise ValueError("Need complete and matched Stage-175 outcomes")
        harmful = result["event_harm"] == "True"
        rescued = result["prevented_terminal"] == "True" or result["rescued_recovery"] == "True"
        if harmful or rescued or key in sampled:
            cases.append(dict(id=key, sequence=row["sequence"], group=row["group"], frame=int(row["frame"]),
                              end=int(row["end"]), source=row["source"], stage=175, kind=row["kind"],
                              known_source=row["known_source"] == "True", sampled_ordinary=key in sampled,
                              archived_harmful=harmful, archived_rescue=rescued, expected_native=int(result["native_layer"]),
                              expected_ranked=int(result["ranked_layer"]), expected_delta32=float(result["delta32"]),
                              expected_delta=float(result["delta_window"])))

    reference = next(r for r in read(173, "cases") if r["id"] == "uav0000304_00253_s__early_failure")
    tail = {int(r["horizon"]): float(r["ranked_delta"]) for r in read(173, "outcomes") if r["case_id"] == reference["id"]}
    legacy = next(r for r in read(174, "cases") if r["sequence"] == "uav0000348_02415_s" and int(r["frame"]) == 438)
    legacy_outcome = next(r for r in read(174, "outcomes") if r["case_id"] == legacy["id"])
    sentinels = ((173, reference, reference["expected_native"], reference["expected_ranked"], tail[SHORT], tail[LONG]),
                 (174, legacy, legacy_outcome["native"], legacy_outcome["ranked"], legacy_outcome["delta32"],
                  legacy_outcome["delta128"]))
    for stage, row, native, ranked, short, long in sentinels:
        cases.append(dict(id=row["id"], sequence=row["sequence"], group=risk.SOURCE_GROUP(row["sequence"]),
                          frame=int(row["frame"]), end=int(row["end"]), source=row["source"], stage=stage,
                          kind="sentinel", known_source=True, sampled_ordinary=False, archived_harmful=True,
                          archived_rescue=False, expected_native=int(native), expected_ranked=int(ranked),
                          expected_delta32=float(short), expected_delta=float(long)))
    if len({r["id"] for r in cases}) != len(cases) or len({(r["sequence"], r["frame"]) for r in cases}) != len(cases):
        raise ValueError("Stage-176 selected a duplicate state")
    return cases


class SweepObserver(risk.RiskObserver):
    def route(self, module, args, output):
        result = super().route(module, args, output)
        if self.arm == "native" or self.forced_layer == self.original:
            return result
        self.chosen, self.switched = self.forced_layer, True
        forced = self.rt.torch.zeros_like(output)
        forced[:, LAYERS.index(self.chosen)] = 1
        return forced


def stage175_cache(root, receipt, case, arm):
    path = root / "rollouts" / f"{case['id']}__{arm}.npz"
    files = (path, path.with_suffix(".receipt.json"))
    io.verify({str(f): receipt["artifact_sha256"][f.relative_to(root).as_posix()] for f in files})
    metadata = dict(plan_fingerprint=receipt["plan_fingerprint"], case_id=case["id"], arm=arm)
    trace = c.read_cache(path, metadata)
    if trace is None:
        raise ValueError(f"Missing verified Stage-175 branch: {path}")
    return trace


def measure(case, traces, truth, source, archive_iou, exact, saved):
    reference = traces[LAYERS[0]]
    native, ranked = int(reference["anchor_native"]), int(reference["anchor_ranked"])
    for layer in LAYERS:
        for k in reference:
            if k.startswith("anchor_"):
                np.testing.assert_array_equal(reference[k], traces[layer][k])
    pair = {"native": traces[native], "ranked": traces[ranked]}
    diagonal = event.measure(case, pair, truth, archive_iou, source, exact)
    if not diagonal["valid"]:
        raise ValueError(f"Incomplete evaluation targets at {case['id']}")
    if exact:
        if (native, ranked) != (case["expected_native"], case["expected_ranked"]):
            raise ValueError(f"Archived candidate layers changed at {case['id']}")
        np.testing.assert_allclose([diagonal["delta32"], diagonal["delta_window"]],
                                   [case["expected_delta32"], case["expected_delta"]], rtol=0, atol=1e-12)
        for k in ("event_harm", "lost_recovery", "new_terminal", "prevented_terminal", "rescued_recovery"):
            if saved is not None and diagonal[k] != (saved[k] == "True"):
                raise ValueError(f"Stage-175 event label changed at {case['id']} / {k}")
    ious = {layer: online.tracking_iou(traces[layer]["boxes"], truth) for layer in LAYERS}
    prefix = archive_iou[:case["frame"] - 1]
    rows = []
    for layer in LAYERS:
        delta = ious[layer][1:] - ious[native][1:]
        short, long = float(delta[:SHORT].mean()), float(delta.mean())
        flags = event.event_labels(prefix, np.stack((ious[native], ious[layer])))
        protected = not flags["event_harm"] and flags["added_failure"] == 0 and min(short, long) >= -EPSILON
        benefit = protected and (long >= EPSILON or flags["prevented_terminal"] or flags["rescued_recovery"])
        rows.append(dict(case_id=case["id"], layer=layer, is_native=layer == native, is_ranked=layer == ranked,
                         anchor_iou=float(ious[layer][0]), delta32=short, delta_window=long,
                         protected=protected, protected_benefit=benefit, **flags))
    return diagonal, rows


def headroom(case, rows):
    native = next(r for r in rows if r["is_native"])
    ranked = next(r for r in rows if r["is_ranked"])
    safe = [r for r in rows if r["protected"]]
    alternatives = [r for r in safe if not r["is_native"] and not r["is_ranked"]]
    # Prefer the unchanged native route on an exact tie in measured mean IoU.
    best = max(rows, key=lambda r: (r["delta_window"], r["is_native"], -r["layer"]))
    best_safe = max(safe, key=lambda r: (r["delta_window"], r["is_native"], -r["layer"]))
    return dict(case_id=case["id"], native_layer=native["layer"], ranked_layer=ranked["layer"],
                ranked_harm=ranked["event_harm"], ranked_protected=ranked["protected"],
                ranked_gain=ranked["delta_window"], oracle_layer=best["layer"], oracle_gain=best["delta_window"],
                protected_oracle_layer=best_safe["layer"], protected_oracle_gain=best_safe["delta_window"],
                protected_oracle_minus_ranked=best_safe["delta_window"] - ranked["delta_window"],
                safe_alternatives=len(alternatives), protected_rescue_choices=sum(
                    r["prevented_terminal"] or r["rescued_recovery"] for r in safe if not r["is_native"]),
                protected_benefit_layers="|".join(str(r["layer"]) for r in safe if r["protected_benefit"]),
                undo_harm_with_alternative=ranked["event_harm"] and bool(alternatives),
                improve_native_with_protection=any(r["protected_benefit"] for r in safe))


def summarize(cases, headroom_rows):
    outcomes = {r["case_id"]: r for r in headroom_rows}
    summary, paired = [], []
    for scope in ("all", "new_source", "sampled_ordinary", "historical_sentinels"):
        for length in ("all", "full128"):
            roster = [r for r in cases if (length == "all" or r["end"] - r["frame"] == LONG)
                      and (scope == "all" or (scope == "new_source" and not r["known_source"])
                           or (scope == "sampled_ordinary" and r["sampled_ordinary"])
                           or (scope == "historical_sentinels" and r["stage"] != 175))]
            rows = [outcomes[r["id"]] for r in roster]
            harm = {r["group"] for r in roster if outcomes[r["id"]]["ranked_harm"]}
            repaired = {r["group"] for r in roster if outcomes[r["id"]]["undo_harm_with_alternative"]}
            benefit = {r["group"] for r in roster if outcomes[r["id"]]["improve_native_with_protection"]}
            summary.append(dict(scope=scope, length=length, anchors=len(rows), groups=len({r["group"] for r in roster}),
                                ranked_harm_groups=len(harm), alternative_repair_groups=len(repaired),
                                protected_gain_groups=len(benefit),
                                **{k: sum(r[k] for r in rows) for k in
                                   ("ranked_harm", "undo_harm_with_alternative", "improve_native_with_protection")}))
            for metric in ("ranked_gain", "oracle_gain", "protected_oracle_gain", "protected_oracle_minus_ranked"):
                grouped = defaultdict(list)
                for r in roster:
                    grouped[r["group"]].append(outcomes[r["id"]][metric])
                if grouped:
                    paired.append(dict(scope=scope, length=length, anchors=len(rows), **base.s103.contrast(
                        [float(np.mean(v)) for v in grouped.values()], metric, "iou", online.DESIGN)))
    return summary, paired


def run(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167, r170, r173, r174, r175 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    roots = {173: r173, 174: r174, 175: r175}
    refs = {stage: c.c66.sealed(roots[stage] / f"stage{stage}_decision.json", schema)
            for stage, (schema, _) in REFERENCES.items()}
    if (d170["fingerprint"] != event.matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or d170["runtime"] != p170["runtime"] or d167["fingerprint"] != previous.SOURCE167
            or any(refs[s]["fingerprint"] != fingerprint or refs[s]["runtime"] != d170["runtime"]
                   for s, (_, fingerprint) in REFERENCES.items())):
        raise ValueError("Need the frozen Stage-167, Stage-170 and Stage-173 to Stage-175 runs")
    read = lambda s, k: risk.read_verified(roots[s], refs[s], s, k)
    cases = select_cases(read)
    expected = {r["case_id"]: r for r in read(175, "outcomes")}
    p159, points, gains, ranks, held = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    folds = p159["folds"]
    if folds != p170["folds"] or d121["fingerprint"] != p159["source_stage121"]:
        raise ValueError("Source groups or tracker roster changed")
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, sources, truths, source_iou = {}, {}, {}, {}
    for seq, arm in sorted({(r["sequence"], r["source"]) for r in cases}):
        size = jobs[seq]["frames"]
        if seq not in entries:
            entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
            if (entry["sequence"] != seq or len(entry["frames"]) != size + 1 or len(entry["ground_truth"]) != size + 1
                    or max(r["end"] for r in cases if r["sequence"] == seq) > size):
                raise ValueError(f"Invalid source sequence or frame count at {seq}")
            entries[seq], truths[seq] = entry, np.asarray(entry["ground_truth"], float)
        data = previous.archive_trace(r170, p170, d170, seq, arm)
        previous.audit(data, size, entries[seq]["initial_box"], arm, thresholds[fold_of[risk.SOURCE_GROUP(seq)]])
        sources[seq, arm] = data
        source_iou[seq, arm] = online.tracking_iou(data["boxes"], truths[seq][1:])

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p170["runtime"], runtime)
    code = (Path(__file__), Path(event.__file__), Path(risk.__file__), Path(aligned.__file__),
            Path(previous.__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage170=d170["fingerprint"],
                        source_reports={str(stage): refs[stage]["fingerprint"] for stage in refs}, cases=cases, folds=folds,
                        layers=list(LAYERS), epsilon=EPSILON, seed=SEED, runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)

    def location(case, layer):
        return root / "rollouts" / f"{case['id']}__l{layer:02d}.npz", dict(plan_fingerprint=plan["fingerprint"],
                                                                          case_id=case["id"], layer=layer)

    reuse = {(case["id"], case["expected_" + arm]): arm for case in cases for arm in risk.ARMS
             if case["stage"] == 175 and not differences}
    branches = [(case, layer, case["end"] - case["frame"] + 1) for case in cases for layer in LAYERS]
    searches = sum(size for _, _, size in branches)
    source_reused = sum(size for case, layer, size in branches if (case["id"], layer) in reuse)
    reused = sum(size for case, layer, size in branches if (case["id"], layer) in reuse
                 or location(case, layer)[0].with_suffix(".receipt.json").exists())
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-176", work, reused, args.progress_interval, "work")
    progress.report("Restoring Stage-167 calibration")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gains, ranks, held, progress)
    for row, ref in zip(calibration, d167["calibration"], strict=True):
        np.testing.assert_allclose([row[k] for k in ref], list(ref.values()), atol=1e-10, rtol=0)
        models[ref["fold"]]["threshold"] = ref["threshold"]
    args.parameter = runtime["tracker"]["parameter"]
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    tables = dict(cases=cases, anchors=[], layers=[], headroom=[])
    artifacts = {}
    progress.report("Replaying six anchor layers")
    with ExitStack() as stack:
        observer = SweepObserver(native, stack)
        for case in cases:
            seq = case["sequence"]
            source_key = seq, case["source"]
            entry, source = entries[seq], sources[source_key]
            model = models[fold_of[case["group"]]]
            traces, template = {}, None

            def read_frame(frame):
                return online.checked_image(old["root"], entry["frames"][frame])

            for layer in LAYERS:
                path, metadata = location(case, layer)
                trace = c.read_cache(path, metadata)
                fresh = trace is None
                if fresh:
                    role = reuse.get((case["id"], layer))
                    if role is None:
                        if template is None:
                            template = read_frame(0)
                        observer.forced_layer = layer
                        trace = aligned.replay(observer, read_frame, template, entry, case, source, "native", model, progress)
                    else:
                        trace = stage175_cache(r175, refs[175], case, role)
                aligned.audit(dict(trace, anchor_probe=np.asarray(layer)), case, source, "probe")
                if fresh:
                    c.save(path, trace, metadata)
                traces[layer] = trace
                for f in (path, path.with_suffix(".receipt.json")):
                    artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
            target = truths[seq][case["frame"]:case["end"] + 1]
            diag, rows = measure(case, traces, target, source, source_iou[source_key], not differences,
                                 expected.get(case["id"]) if case["stage"] == 175 else None)
            tables["anchors"].append(diag)
            tables["layers"].extend(rows)
            tables["headroom"].append(headroom(case, rows))
    native.assert_frozen()
    tables["summary"], tables["paired"] = summarize(cases, tables["headroom"])
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-176 progress accounting")
    main = next(r for r in tables["summary"] if (r["scope"], r["length"]) == ("new_source", "full128"))
    return dict(plan_fingerprint=plan["fingerprint"], cases=len(cases), layer_branches=len(branches),
                sampled_ordinary=sum(r["sampled_ordinary"] for r in cases),
                archived_harm=sum(r["archived_harmful"] for r in cases), archived_rescue=sum(r["archived_rescue"] for r in cases),
                source_stage175_reused_searches=source_reused, logical_searches=searches,
                required_new_searches=searches - source_reused, new_full_trajectories=0, main_headroom=main,
                work=work, runtime=runtime, runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage176")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=3, type=Path, metavar=("STAGE173", "STAGE174", "STAGE175"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage176_layer_event_sweep")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU source check")
        tables = {stage: gain.prior.archives.bundle(p, stage, fp)[1]
                  for p, (stage, (_, fp)) in zip(args.bundles, REFERENCES.items(), strict=True)}
        cases = select_cases(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        print(f"Verified {len(cases)} states across {len({r['group'] for r in cases})} source groups, "
              f"{sum(r['sampled_ordinary'] for r in cases)} sampled ordinary states, "
              f"{sum(r['archived_harmful'] for r in cases)} archived harmful states, "
              f"{sum(r['archived_rescue'] for r in cases)} archived rescue states")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        path = root / "stage176_decision.json"
        if path.exists():
            saved = c.c66.sealed(path, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage175"] != SOURCE175 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-176 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-176", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE176_LAYER_EVENT_SWEEP",
                                              source_stage175=SOURCE175, **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
