"""Engineering smoke, fit-only native candidate collection, and convex identity ranking."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage28_response_collect import validate_runtime
from stage31_identity_common import (FEATURES, METHODS, SCHEMA, fit_ranker, ground_truth,
                                      load_contract, load_sealed)
from stage31_identity_engine import NativeObserver, audit_video, cache_metadata, load_video, run_video


def load_model(directory, contract, verify_caches=True):
    directory = Path(directory)
    model = load_sealed(directory / "model.json")
    run = load_sealed(directory / "fit_run.json")
    if (model.get("role") != "frozen_fit_only_identity_ranker" or model.get("contract_fingerprint") != contract["fingerprint"]
            or model.get("fit_run_fingerprint") != run["fingerprint"] or run.get("contract_fingerprint") != contract["fingerprint"]
            or model.get("feature_names") != list(FEATURES) or model.get("checking_labels_used") is not False):
        raise ValueError("Frozen identity model/source mismatch")
    for key in ("weights", "feature_scale", "static_weights", "static_feature_scale"):
        value = np.asarray(model[key])
        if value.shape != (len(FEATURES),) or not np.isfinite(value).all():
            raise ValueError("Invalid model parameter: " + key)
    if (np.asarray(model["feature_scale"]) <= 0).any() or (np.asarray(model["static_feature_scale"]) <= 0).any():
        raise ValueError("Invalid feature scales")
    if np.any(np.asarray(model["static_weights"])[[1, 2, 3, 4, 5, 6]] != 0):
        raise ValueError("Static control must not use history or geometry features")
    if verify_caches:
        expected = {e["sequence"] + ".npz" for e in contract["sequences"] if e["split"] == "fit"}
        if {p.name for p in (directory / "fit_cache").glob("*.npz")} != expected:
            raise ValueError("Fit cache contains missing or unexpected videos")
        hashes = {name: file_hash(directory / "fit_cache" / name) for name in expected}
        if model.get("fit_cache_sha256") != hashes:
            raise ValueError("Fit cache hashes changed")
    return model


def initialize_observer(contract, parameter, gpu, checkpoint):
    if file_hash(Path(checkpoint).expanduser()) != contract["runtime"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen original SGLA")
    import random
    import torch
    from stage4_runtime import Runtime
    random.seed(contract["design"]["seed"])
    np.random.seed(contract["design"]["seed"])
    torch.manual_seed(contract["design"]["seed"])
    torch.cuda.manual_seed_all(contract["design"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    rt = Runtime(parameter, gpu, checkpoint)
    validate_runtime(rt, contract)
    if rt.net.training or any(p.requires_grad for p in rt.net.parameters()):
        raise ValueError("Original network must remain frozen and in eval mode")
    return NativeObserver(rt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-31 root with manifest.json")
    parser.add_argument("--mode", choices=("smoke", "collect", "train"), required=True)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="Collect one complete fit video; never selects checking data")
    args = parser.parse_args()
    if args.sequence and args.mode != "collect":
        parser.error("--sequence only applies to full fit-video collection")
    directory = Path(args.input).expanduser().resolve()
    contract = load_contract(directory / "manifest.json")
    fit_entries = [e for e in contract["sequences"] if e["split"] == "fit"]
    with writer_lock(directory):
        if args.mode == "train":
            if args.sequence:
                parser.error("--sequence is only available during collection")
            run = load_sealed(directory / "fit_run.json")
            if run.get("role") != "native_fit_candidate_features" or run.get("contract_fingerprint") != contract["fingerprint"]:
                raise ValueError("Fit run identity mismatch")
            if (directory / "model.json").exists():
                model = load_model(directory, contract)
                print("[reuse, audited] Frozen fit-only model: " + model["fingerprint"])
                return
            expected = {e["sequence"] + ".npz" for e in fit_entries}
            if {p.name for p in (directory / "fit_cache").glob("*.npz")} != expected:
                raise ValueError("Need all 8 complete, unmixed fit video caches before training")
            records, hashes = [], {}
            for entry in fit_entries:
                path = directory / "fit_cache" / (entry["sequence"] + ".npz")
                records.append(load_video(path, contract, entry, "sgla", run, fit_collect=True))
                hashes[path.name] = file_hash(path)
            result = fit_ranker(records)
            static_records = []
            for record in records:
                features = np.zeros_like(record["features"])
                features[..., [0, 7, 8]] = record["features"][..., [0, 7, 8]]
                static_records.append(dict(record, features=features))
            static_result = fit_ranker(static_records)
            model = seal(dict(schema=SCHEMA, role="frozen_fit_only_identity_ranker", **result,
                static_weights=static_result["weights"], static_feature_scale=static_result["feature_scale"],
                static_fit_pair_accuracy=static_result["fit_pair_accuracy"],
                static_control="Same ranking data/optimizer; initial identity and current response only, all history/geometry features zeroed",
                contract_fingerprint=contract["fingerprint"], fit_run_fingerprint=run["fingerprint"],
                fit_cache_sha256=hashes, checking_labels_used=False, hyperparameter_search_performed=False,
                training="Video/frame-balanced pairwise logistic ranking, fixed L2; frozen descriptors; no backbone optimizer",
                diagnostic_role="Fit pair accuracy is resubstitution, not held-out or closed-loop evidence"))
            publish_identical(directory / "model.json", model)
            print("Frozen fit-only model: {} frames / {} videos / {} pairs; fit accuracy {:.4f}".format(
                result["fit_pair_frames"], result["fit_pair_videos"], result["fit_pairs"], result["fit_pair_accuracy"]))
            print("Model fingerprint: " + model["fingerprint"])
            print("Checking labels used: no. Next: one frozen checking evaluation.")
            return
        if not args.checkpoint:
            parser.error("--checkpoint is required for smoke/collect")
        observer = initialize_observer(contract, args.parameter, args.gpu, args.checkpoint)
        if args.mode == "smoke":
            entry = fit_entries[0]
            gt = ground_truth(contract, entry)
            small = dict(entry, frames=entry["frames"][:65])
            gt = gt[:len(small["frames"])]
            bootstrap = dict(feature_scale=[1.] * len(FEATURES), weights=[1., 1., 1., -1., 1., -.25, -.1, .1, 0.],
                static_feature_scale=[1.] * len(FEATURES), static_weights=[1., 0., 0., 0., 0., 0., 0., .1, 0.])
            methods = []
            for method in METHODS:
                data = run_video(observer, contract, small, method, bootstrap, labels=gt)
                audit_video(data, contract, small, method, bootstrap, labels=gt)
                methods.append(dict(method=method, frames=len(data["frame_id"]),
                    alternative_frames=int((data["chosen"] > 0).sum()), finite_features=True, causal_state_continuity=True))
            decision = seal(dict(schema=SCHEMA, role="untrained_identity_execution_only", status="PASS_CAUSAL_IDENTITY_SMOKE",
                contract_fingerprint=contract["fingerprint"], sequence=entry["sequence"], methods=methods,
                checking_labels_used=False, model_trained=False,
                warning="Bootstrap weights only; no learned association, AUC gain or speed claim"))
            publish_identical(directory / "smoke" / "stage31_smoke_decision.json", decision)
            print("PASS_CAUSAL_IDENTITY_SMOKE: first fit video only; no checking labels loaded.")
            return
        entries = [e for e in fit_entries if args.sequence is None or e["sequence"] == args.sequence]
        if not entries:
            raise ValueError("--sequence is not a fit video")
        run = seal(dict(schema=SCHEMA, role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"],
                        runtime=observer.identity, trajectory="native SGLA own predictions; GT labels logged after state advance"))
        if not (directory / "fit_run.json").exists() and any((directory / "fit_cache").glob("*.npz")):
            raise ValueError("Unidentified fit caches; preserve them and use a new run directory")
        publish_identical(directory / "fit_run.json", run)
        for entry in entries:
            path = directory / "fit_cache" / (entry["sequence"] + ".npz")
            if path.exists():
                load_video(path, contract, entry, "sgla", run, fit_collect=True)
                print("[reuse, audited] " + path.name, flush=True)
                continue
            data = run_video(observer, contract, entry, "sgla", fit_collect=True)
            audit_video(data, contract, entry, "sgla", fit_collect=True)
            write_cache(path, data, cache_metadata(run, entry, "sgla"))
            load_video(path, contract, entry, "sgla", run, fit_collect=True)
            print("[fit cache] " + path.name, flush=True)


if __name__ == "__main__":
    main()
