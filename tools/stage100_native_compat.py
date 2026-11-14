"""Resume frozen Stage-100 using an audited q0 view of four-slot native sources.

Only native-source representation is adapted. The original experiment, its
manifest, all upstream code, source files and committed caches stay unchanged.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage100_online_scale as original


c, io = original.c, original.io
SCHEMA = "stage100_native_q0_cache_compat_v1"
RECEIPT = "stage100_native_compat.json"
ORIGINAL_LOAD_NATIVE = original.load_native
ORIGINAL_RUN = original.run
ORIGINAL_CHECK = original.check_results
ORIGINAL_READ_CACHE = c.read_cache
ORIGINAL_PUBLISH = c.publish_identical


def native_view(data, entry, ap):
    """Audit every recorded call, then copy the native q0 slot without inference."""
    n = len(entry["frames"])-1
    shape = np.asarray(data["block_calls"]).shape
    if shape not in ((n, 1, 12), (n, 4, 12)):
        raise ValueError(f"Unsupported native call shape {shape}; expected ({n}, 1|4, 12)")
    slots = shape[1]
    original.e63.source_common.prior.audit_calls(data, (n, slots))
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
    if np.asarray(data["selected"]).shape != (n,) or np.any(data["selected"] != 0):
        raise ValueError("A native q0 source must always select slot zero")
    for key in ("candidate_boxes", "candidate_parents"):
        bank = np.asarray(data[key])
        if bank.shape != (n, slots, 4) or not np.isfinite(bank).all() or np.any(bank[..., 2:] <= 0):
            raise ValueError("Malformed native candidate bank: "+key)
    np.testing.assert_array_equal(data["boxes"], data["candidate_boxes"][:, 0])
    result = {key: np.asarray(data[key]).copy() for key in
              ("frame_id", "boxes", "selected", "image_hw", "decode_error")}
    for key in ("candidate_parents", "candidate_boxes") + original.s93.s85.MODEL_FIELDS:
        array = np.asarray(data[key])
        if array.ndim < 2 or array.shape[:2] != (n, slots):
            raise ValueError("Native candidate axis differs: "+key)
        result[key] = array[:, :1].copy()
    if "crop_sha256" in data:
        if np.asarray(data["crop_sha256"]).shape != (n, slots):
            raise ValueError("Malformed native crop hashes")
        result["crop_sha256"] = np.asarray(data["crop_sha256"])[:, :1].copy()
    result["changed_crop"] = np.zeros(n, bool)
    # Keep old source computation distinct from the one-slot evaluation view.
    source_slots = int(np.asarray(data.get("native_source_slots", slots)).item())
    source_calls = int(np.asarray(data.get("native_source_head_calls", np.sum(data["head_calls"]))).item())
    if source_slots not in (1, 4) or source_calls != n*source_slots or (slots == 4 and source_slots != 4):
        raise ValueError("Native source accounting provenance differs")
    result.update(native_source_slots=np.asarray(source_slots), native_source_head_calls=np.asarray(source_calls))
    original.s93.s85.audit(result, entry, "native_sgla", ap, None, {})
    return result


def receipt_for(plan):
    here = Path(__file__).resolve()
    runner = here.with_name("run_stage100_native_compat.sh")
    return io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"],
        operation="audit_all_native_source_slots_then_project_q0_for_single_search_baseline",
        original_code_sha256=plan["code_sha256"],
        adapter_code_sha256={str(p): c.file_hash(p) for p in (here, runner)},
        selected_source_slot=0, manifest_edited=False, source_artifacts_edited=False,
        committed_cache_bytes_edited=False, policy_design_changed=False, adapter_inference_calls=0))


def compatibility_binding(root, receipt):
    return dict(schema=SCHEMA, fingerprint=receipt["fingerprint"],
                receipt_sha256=c.file_hash(root / RECEIPT), selected_source_slot=0)


def check_results(root):
    root = Path(root).resolve()
    ORIGINAL_CHECK(root)
    plan = c.c66.sealed(root / "manifest.json", original.SCHEMA)
    receipt = c.c66.sealed(root / RECEIPT, SCHEMA)
    if receipt != receipt_for(plan):
        raise ValueError("Native compatibility receipt or adapter changed")
    io.verify(receipt["adapter_code_sha256"])
    report = c.c66.sealed(root / "analysis/stage100_decision.json", original.SCHEMA)
    if report.get("native_compatibility") != compatibility_binding(root, receipt):
        raise ValueError("Stage-100 report is not bound to the native compatibility receipt")
    expected = {spec["unit"]["sequence"] for spec in plan["units"]}
    views = report.get("native_source_views", [])
    if len(views) != len(expected) or {r["sequence"] for r in views} != expected:
        raise ValueError("Incomplete native source accounting")
    for row in views:
        if (row["source_slots"] not in (1, 4) or row["source_head_calls"] != row["frames"]*row["source_slots"]
                or row["q0_view_head_calls"] != row["frames"]):
            raise ValueError("Native source/view call totals differ")


def run(root, root93, root89, plan, plan93, plan89, report89, source, actor, observer):
    root = Path(root).resolve()
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    receipt = receipt_for(plan)
    ORIGINAL_PUBLISH(root / RECEIPT, receipt)
    ap = original.e63.c.actor_plan(source["original"])
    provenance, entries = {}, {}

    def load_native(*args):
        unit, data = ORIGINAL_LOAD_NATIVE(*args)
        # Resolve entries through the exact same verified source loader as the experiment.
        spec = args[-1]
        entry, _ = original.s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
        entries[unit["sequence"]] = entry
        normalized = native_view(data, entry, ap)
        n = len(entry["frames"])-1
        provenance[unit["sequence"]] = dict(sequence=unit["sequence"], frames=n,
            source_slots=int(normalized["native_source_slots"]),
            source_head_calls=int(normalized["native_source_head_calls"]), q0_view_head_calls=n)
        return unit, normalized

    def read_cache(path, metadata):
        data = ORIGINAL_READ_CACHE(path, metadata)
        if (data is not None and Path(path).resolve().parent == root / "evaluation"
                and metadata.get("plan_fingerprint") == plan["fingerprint"] and metadata.get("arm") == "native"):
            # Existing receipts are checked before creating this in-memory view.
            data = native_view(data, entries[metadata["unit"]["sequence"]], ap)
        return data

    def publish(path, value):
        if Path(path).resolve() == root / "analysis/stage100_decision.json":
            body = {k: v for k, v in value.items() if k != "fingerprint"}
            value = io.seal(body | dict(native_compatibility=compatibility_binding(root, receipt),
                                       native_source_views=[provenance[k] for k in sorted(provenance)]))
        return ORIGINAL_PUBLISH(path, value)

    with (patch.object(original, "load_native", side_effect=load_native),
          patch.object(c, "read_cache", side_effect=read_cache),
          patch.object(c, "publish_identical", side_effect=publish),
          patch.object(original, "check_results", side_effect=check_results)):
        return ORIGINAL_RUN(root, root93, root89, plan, plan93, plan89, report89, source, actor, observer)


def main():
    # Delegate argument parsing, source/hardware checks and the unchanged manifest to Stage-100.
    with (patch.object(original, "run", side_effect=run),
          patch.object(original, "check_results", side_effect=check_results)):
        original.main()


if __name__ == "__main__":
    main()
