"""Resume frozen Stage-67 summary with a uniform optional anchor frame column."""
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import patch
import stage67_veto_common as c
import stage67_veto_summary as original
from stage27_uav_common import nonempty_path
from stage27_uav_collect import writer_lock
from stage4_common import file_hash

SCHEMA = "stage67_anchor_csv_export_repair_v1"
ANCHOR_FIELDS = ("case_id", "sequence", "selector", "frame", "status", "resolved_frame")
REQUIRED_FIELDS = set(ANCHOR_FIELDS)-{"frame"}
ORIGINAL_SUMMARIZE = original.summarize
ORIGINAL_PUBLISH = original.publish_table


def anchor_rows(rows):
    """Fill only the optional requested frame; keep the resolved anchor distinct."""
    if not rows: raise ValueError("Missing frozen anchor requests")
    result = []
    for row in rows:
        if not REQUIRED_FIELDS.issubset(row) or set(row)-set(ANCHOR_FIELDS):
            raise ValueError("Unexpected Stage-67 anchor fields")
        result.append({k:row.get(k) for k in ANCHOR_FIELDS})
    return result


def publish_table(path, rows):
    normalized = anchor_rows(rows) if Path(path).name == "stage67_anchors.csv" else rows
    return ORIGINAL_PUBLISH(path, normalized)


def summarize(root):
    root = Path(root).expanduser().resolve()
    # Run every original source, manifest/code, receipt, diagonal and budget check.
    # Only the publisher is scoped to this call; no source gate is patched.
    with patch.object(original, "publish_table", side_effect=publish_table):
        report = ORIGINAL_SUMMARIZE(root)
    plan = c.sealed(root/"manifest.json")
    receipt = c.seal(dict(schema=SCHEMA, operation="fill_optional_requested_frame_in_anchor_csv_only",
        plan_fingerprint=report["plan_fingerprint"], run_fingerprint=report["run_fingerprint"],
        original_code_sha256=plan["code_sha256"],
        repair_code_sha256={str(Path(__file__).resolve()):file_hash(__file__)},
        anchor_columns=list(ANCHOR_FIELDS), anchor_rows=len(plan["requests"]),
        diagnostic_sha256=file_hash(root/"analysis/stage67_diagnostic.json"),
        table_sha256=report["table_sha256"],
        manifest_edited=False, source_hash_checks_relaxed=False, cache_metadata_edited=False,
        training_performed=False, new_image_reads=0, new_search_calls=0,
        interpretation="Export repair only; the original frozen summary revalidates all recorded branches"))
    with writer_lock(root/"analysis"):
        c.publish_identical(root/"analysis/stage67_summary_repair.json",receipt)
    print("Stage-67 summary repaired from committed caches; no image reads or GPU inference")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
