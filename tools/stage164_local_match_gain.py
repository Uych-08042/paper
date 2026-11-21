"""Compare two L6 representations for gain prediction on the same SGLA crops."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from math import isqrt
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_progress import OverallProgress
import stage163_history_gain as gain
from stage27_uav_common import checked_image
from stage28_response_common import cpu_array


history, veto, prior, c, io = gain.history, gain.veto, gain.prior, gain.c, gain.io
trained, base = gain.trained, gain.trained.base
SCHEMA = "stage164_l6_local_match_gain_v1"
STATUS = "COMPLETE_STAGE164_L6_LOCAL_MATCH_GAIN_ABLATION"
SOURCE163 = "6a8f4361ef426d140e946d52a47dd5872cfc6250b8f169d5282328962fb33974"
LOCAL = "local_match_gain"
REPLAY = "replayed_projection_gain"
SOURCES = (76, 89, 93, 121, 159, 163)
GRID = 4
VISUAL_DIM = GRID * GRID
TOTAL_DIM = VISUAL_DIM + trained.HISTORY_FEATURES
PREFIX_CALLS = (1,) * 6 + (0,) * 8


def local_match(tokens, template_count, search_side):
    """Max cosine similarity to central template tokens, pooled on a 4x4 search grid."""
    side = isqrt(template_count)
    if (tokens.ndim != 2 or tokens.shape[0] != template_count + search_side ** 2 or tokens.shape[1] == 0
            or side ** 2 != template_count or side % 4 or search_side % GRID
            or side == 0 or search_side < GRID or not np.isfinite(tokens).all()):
        raise ValueError("Invalid L6 token layout")
    start, end = side // 4, 3 * side // 4
    template = tokens[:template_count].reshape(side, side, -1)[start:end, start:end]
    template = template.reshape(-1, tokens.shape[1])
    search = tokens[template_count:]
    template = template / np.maximum(np.linalg.norm(template, axis=1, keepdims=True), 1e-12)
    search = search / np.maximum(np.linalg.norm(search, axis=1, keepdims=True), 1e-12)
    similarity = (search @ template.T).max(axis=1)
    tile = search_side // GRID
    return similarity.reshape(GRID, tile, GRID, tile).max(axis=(1, 3)).reshape(VISUAL_DIM)


def capture(rt, image, parent, frame):
    rt.tracker.state = np.asarray(parent, float).tolist()
    rt.tracker.frame_id = frame - 1
    calls = np.zeros(len(PREFIX_CALLS), int)
    crop_hashes = []
    sample = rt.s1.sample_target

    def sample_once(*args, **kwargs):
        np.testing.assert_array_equal(args[1], parent)
        crop = sample(*args, **kwargs)
        crop_hashes.append(sha256(np.ascontiguousarray(crop[0]).tobytes()).hexdigest())
        return crop

    def count(index):
        def hook(*_):
            calls[index] += 1
        return hook

    with ExitStack() as stack:
        for index, module in enumerate([*rt.backbone.blocks, rt.backbone.MLP, rt.net.box_head]):
            stack.callback(module.register_forward_hook(count(index)).remove)
        stack.enter_context(patch.object(rt.s1, "sample_target", sample_once))
        stack.enter_context(rt.torch.no_grad())
        crop, _ = rt.prepare(image)
        tokens = rt.prefix(crop)
        projection = trained.projected_prefix(rt, tokens)
        match = local_match(cpu_array(tokens)[0].astype(float), rt.lens_z, rt.search_side)
    np.testing.assert_array_equal(calls, PREFIX_CALLS)
    np.testing.assert_array_equal(rt.tracker.state, parent)
    if len(crop_hashes) != 1:
        raise ValueError("Expected one native search crop per L6 capture")
    return dict(projection=projection, match=match, calls=calls, crop_sha256=np.array(crop_hashes[0]),
                image_hw=np.array(image.shape[:2]))


def load_control(root, points):
    decision = c.c66.sealed(root / "stage163_decision.json", gain.SCHEMA)
    if (decision["fingerprint"] != SOURCE163 or decision["status"] != "COMPLETE_STAGE163_HISTORY_GAIN_ABLATION"
            or decision["source_stage159"] != veto.SOURCE159 or decision["source_stage162"] != gain.SOURCE162
            or decision["cases"] != len(points) or decision["switch_budgets"] != list(gain.BUDGETS.values())):
        raise ValueError("Need the frozen Stage-163 result")
    name = "stage163_choices.csv"
    io.verify({str(root / name): decision["table_sha256"][name]})
    rows = prior.prior.read_table(root / name)
    gain.attach_control(points, [
        dict(r, arm="history16_trend", layer=r["short_layer"], predicted_gain=r["short_gain"],
             predicted_downside=r["fixed_downside"], ranking_score=r["short_score"]) for r in rows])


def cached_cases(root, r159, plan, p159, points):
    pending, records = {}, {}
    for p in points:
        cid = p["case_id"]
        archive = c.read_cache(r159 / f"branches/{cid}.npz", dict(plan_fingerprint=p159["fingerprint"], case_id=cid))
        if (archive is None or archive["features"].shape != (TOTAL_DIM,) or not np.isfinite(archive["features"]).all()
                or int(archive["frame_id"][0]) != p["frame"] or int(archive["anchor_layer"]) != p["native_layer"]):
            raise ValueError(f"Invalid Stage-159 cache for {cid}")
        path = root / f"features/{cid}.npz"
        metadata = dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        feature = c.read_cache(path, metadata)
        records[cid] = (archive, feature)
        if feature is None:
            pending[cid] = (path, metadata)
    return records, pending


def features_for_run(points, records, pending, native, jobs, load_entry, image_root, progress):
    vectors = {name: [] for name in (history.BASELINE, REPLAY, LOCAL)}
    audits = []
    sequence = None
    for p in points:
        cid = p["case_id"]
        archive, feature = records[cid]
        if cid in pending:
            if p["sequence"] != sequence:
                sequence = p["sequence"]
                entry = load_entry(jobs[sequence]["spec"])
                if entry["sequence"] != sequence:
                    raise ValueError(f"Unexpected template sequence for {cid}")
                native.initialize(checked_image(image_root, entry["frames"][0]), entry["initial_box"])
            image = checked_image(image_root, entry["frames"][p["frame"]])
            feature = capture(native.rt, image, archive["anchor_parent"], p["frame"])
        np.testing.assert_array_equal(feature["image_hw"], archive["anchor_hw"])
        np.testing.assert_array_equal(feature["calls"], PREFIX_CALLS)
        if str(feature["crop_sha256"]) != str(archive["anchor_crop"]):
            raise ValueError(f"Stage-159 image crop differs at {cid}")
        projection, match = feature["projection"], feature["match"]
        if any(v.shape != (VISUAL_DIM,) or not np.isfinite(v).all() for v in (projection, match)):
            raise ValueError(f"Invalid L6 feature vector at {cid}")
        if cid in pending:
            path, metadata = pending[cid]
            c.save(path, feature, metadata)
            progress.advance()
        original = archive["features"]
        past = original[VISUAL_DIM:]
        vectors[history.BASELINE].append(original)
        vectors[REPLAY].append(np.r_[projection, past])
        vectors[LOCAL].append(np.r_[match, past])
        audits.append(dict(case_id=cid, sequence=p["sequence"], frame=p["frame"],
                           projection_max_abs_delta=float(np.max(np.abs(projection - original[:VISUAL_DIM]))),
                           local_min=float(match.min()), local_max=float(match.max()),
                           crop_sha256=str(feature["crop_sha256"])))
    return {name: np.stack(rows) for name, rows in vectors.items()}, audits


def compare(points, folds, vectors, models, progress):
    actions, choices, scope = gain.evaluate(points, folds, progress, vectors, models, treatment=LOCAL,
                                           feature_name=LOCAL, column="local")
    control, replay, replay_scope = gain.evaluate(points, folds, progress, vectors, models, treatment=REPLAY,
                                                 feature_name=REPLAY, column="replay")
    actions[REPLAY] = control[REPLAY]
    for current, baseline in zip(choices, replay, strict=True):
        if current["case_id"] != baseline["case_id"] or current["short_layer"] != baseline["short_layer"]:
            raise ValueError("Replayed projection uses a different evaluation case")
        current.update({key: value for key, value in baseline.items() if key.startswith("replay_")})
    for current, baseline in zip(scope, replay_scope, strict=True):
        if current["fold"] != baseline["fold"]:
            raise ValueError("Replayed projection uses a different evaluation fold")
        current.update(replay_switches=baseline["replay_switches"],
                       replay_replaced_switches=baseline["replaced_switches"])
    return actions, choices, scope


def start(args, root, paths):
    r76, r89, r93, r121, r159, r163 = paths
    p159, models, points = veto.load(r159)
    load_control(r163, points)
    _, d121, jobs = history.native_jobs(r121)
    if p159["source_stage121"] != d121["fingerprint"] or set(jobs) != {p["sequence"] for p in points}:
        raise ValueError("Stage-159 and Stage-121 source rosters differ")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]

    def load_entry(spec):
        return s95.load_unit(r93, r89, p93, p89, d89, origin, spec)[0]

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p159["runtime"], runtime)
    code = (Path(__file__), Path(gain.__file__), Path(trained.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage159=veto.SOURCE159, source_stage163=SOURCE163,
                        source_stage121=d121["fingerprint"], runtime=runtime, runtime_differences=differences,
                        switch_budgets=list(gain.BUDGETS.values()), folds=p159["folds"], visual_dim=VISUAL_DIM,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "features").mkdir(exist_ok=True)
    records, pending = cached_cases(root, r159, plan, p159, points)
    work = len(points) + 2 * len(p159["folds"]) + 1
    progress = OverallProgress("Stage-164", work, reused=len(points) - len(pending),
                               interval=args.progress_interval, unit="work")
    native = None
    progress.report("Capturing L6 features")
    if pending:
        args.parameter = runtime["tracker"]["parameter"]
        snapshot = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, snapshot, root)
    vectors, audit = features_for_run(points, records, pending, native, jobs, load_entry, old["root"], progress)
    if native is not None:
        native.assert_frozen()
    progress.report("Fitting and comparing gains")
    actions, choices, scope = compare(points, p159["folds"], vectors, models, progress)
    hashes, checks = gain.report(root, points, actions, choices, scope, stage="stage164", treatment=LOCAL,
                                replay_control=REPLAY, feature_audit=audit)
    artifacts = {}
    for p in points:
        path = root / f"features/{p['case_id']}.npz"
        for file in (path, path.with_suffix(".receipt.json")):
            artifacts[file.relative_to(root).as_posix()] = c.file_hash(file)
    c.publish_identical(root / "stage164_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=runtime,
        runtime_differences=differences, work=work, source_stage163=SOURCE163,
        cases=len(points), prefix_only_searches=len(points), selector_calls=0, head_calls=0, new_full_trajectories=0,
        fitted_gain_models=2 * len(p159["folds"]), maximum_projection_abs_delta=max(r["projection_max_abs_delta"] for r in audit),
        table_sha256=hashes, feature_sha256=artifacts, checks=checks, ready_for_online_test=all(checks.values()))))
    progress.phase = "Complete"
    progress.advance()
    progress.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    parser.set_defaults(mode="stage164_local_match_gain")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [getattr(args, f"stage{stage}_root").expanduser().resolve() for stage in SOURCES]
    if args.progress_interval <= 0 or any(root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use separate source and output directories and a positive progress interval")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    with prior.prior.writer_lock(root):
        decision = root / "stage164_decision.json"
        if decision.exists():
            saved = c.c66.sealed(decision, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["status"] != STATUS or saved["plan_fingerprint"] != plan["fingerprint"] or saved["source_stage163"] != SOURCE163:
                raise ValueError("Unexpected Stage-164 result")
            io.verify({str(root / name): digest for name, digest in (saved["table_sha256"] | saved["feature_sha256"]).items()})
            OverallProgress("Stage-164", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        start(args, root, paths)


if __name__ == "__main__":
    main()
