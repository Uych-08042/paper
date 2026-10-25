"""Audit-only float32-score replay compatibility for the frozen Stage-31 scripts.

NativeObserver produces float32 scores. run_video stores them losslessly in a
float64 buffer, but cached replay must retain the original division precision.
No observer, feature generator, policy, model, cache or original source is edited.
"""

import argparse
import importlib
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage31_identity_common import FEATURES, load_contract
import stage31_identity_engine as engine


SCHEMA = "stage31_float32_replay_compat_v1"
TASKS = dict(smoke="stage31_identity_fit", collect="stage31_identity_fit", train="stage31_identity_fit",
             evaluate="stage31_identity_evaluate", summary="stage31_identity_summary")


def replay_view(data):
    """Lossless dtype restoration only; never edits cached arrays or features."""
    scores = np.asarray(data["scores"])
    if scores.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(scores).all():
        raise ValueError("Compatibility audit requires finite float32/float64 native scores")
    source_scores = scores.astype(np.float32)
    if not np.array_equal(scores, source_scores.astype(scores.dtype)):
        raise ValueError("Scores are not losslessly representable as native float32; refusing audit compatibility")
    return dict(data, scores=source_scores)


@contextmanager
def precision_replay_auditor():
    """Patch only auditor entry points for this process; leave all online math intact."""
    original = engine.audit_video
    aliases = {}
    stats = dict(audited_arrays=0, feature_arrays=0, max_float64_relative_response_delta=0.,
                 max_float32_relative_response_delta=0., audited_sequences=set())

    def compatible_audit(data, contract, entry, method, model=None, fit_collect=False, labels=None):
        view = replay_view(data)
        # The original audit retains every shape, history, label, feature and choice check,
        # including its original atol=1e-9 and rtol=0 for all nine features.
        original(view, contract, entry, method, model, fit_collect, labels)
        stats["audited_arrays"] += 1
        stats["audited_sequences"].add(entry["sequence"])
        if fit_collect or method != "sgla":
            stats["feature_arrays"] += 1
            column = FEATURES.index("relative_response")
            actual = data["features"][..., column]
            mask = data["valid"]
            s32, s64 = view["scores"], data["scores"].astype(np.float64)
            ratio32 = s32 / np.maximum(s32.max(1, keepdims=True), np.float32(1e-12))
            ratio64 = s64 / np.maximum(s64.max(1, keepdims=True), 1e-12)
            for key, ratio in (("max_float32_relative_response_delta", ratio32),
                               ("max_float64_relative_response_delta", ratio64)):
                stats[key] = max(stats[key], float(np.max(np.abs(actual[mask] - ratio[mask]))))

    engine.audit_video = compatible_audit
    for name in set(TASKS.values()):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "audit_video"):
            aliases[name] = module.audit_video
            module.audit_video = compatible_audit
    try:
        yield stats
    finally:
        engine.audit_video = original
        for name in set(TASKS.values()):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "audit_video"):
                module.audit_video = aliases.get(name, original)


def compatibility_record(contract):
    return seal(dict(schema=SCHEMA, role="explicit_audit_precision_compatibility_supplement",
        contract_fingerprint=contract["fingerprint"], original_code_sha256=contract["code_sha256"],
        compatibility_code_sha256=file_hash(Path(__file__)), numpy_version=np.__version__,
        source_score_dtype="float32", storage_score_dtype="float64", replay_score_dtype="float32",
        score_roundtrip_required="exact float64 -> float32 -> float64 equality",
        original_feature_atol=1e-9, original_feature_rtol=0,
        scope="Audit input view only; no tolerance relaxation, online feature change, cache rewrite or contract resealing"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--input", type=nonempty_path, required=True)
    args, forwarded = parser.parse_known_args()
    if "--mode" in forwarded or any(value.startswith("--mode=") for value in forwarded):
        parser.error("The positional task supplies --mode; do not pass it again")
    directory = Path(args.input).expanduser().resolve()
    contract = load_contract(directory / "manifest.json")  # All frozen source/hash checks remain active.
    record = compatibility_record(contract)
    with writer_lock(directory):
        publish_identical(directory / "stage31_precision_compat.json", record)
    print("Float32-score replay auditor enabled; original feature tolerance unchanged; contract "
          + contract["fingerprint"], flush=True)
    argv = ["--input", str(directory)] + forwarded
    if args.task in ("smoke", "collect", "train"):
        argv += ["--mode", args.task]
    original_argv = sys.argv
    status, error = "ERROR", None
    with precision_replay_auditor() as stats:
        try:
            module = importlib.import_module(TASKS[args.task])
            sys.argv = [module.__file__] + argv
            module.main()
            status = "OK"
        except BaseException as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            raise
        finally:
            sys.argv = original_argv
            details = dict(stats, audited_sequences=sorted(stats["audited_sequences"]))
            artifacts = {}
            for relative in ("manifest.json", "fit_run.json", "model.json", "checking_run.json",
                             "analysis/stage31_decision.json", "analysis_pilot/stage31_decision.json"):
                path = directory / relative
                if path.is_file():
                    artifacts[relative] = file_hash(path)
            report = seal(dict(schema=SCHEMA, role="compatibility_audit_execution_record", task=args.task,
                status=status, error=error, finished_utc=datetime.now(timezone.utc).isoformat(),
                compatibility_fingerprint=record["fingerprint"], contract_fingerprint=contract["fingerprint"],
                original_artifact_sha256=artifacts, **details))
            write_json(directory / "precision_compat" / (args.task + "_audit.json"), report)
    print("Compatibility audit: {} arrays; max relative-response delta float64={:.3g}, source float32={:.3g}".format(
        stats["audited_arrays"], stats["max_float64_relative_response_delta"], stats["max_float32_relative_response_delta"]), flush=True)


if __name__ == "__main__":
    main()
