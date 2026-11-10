"""Evaluate Stage-88's fixed write signals on the sealed Stage-89 result bundle."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import io as streams
import json
from pathlib import Path
import re

import numpy as np

import stage89_conflict_sampling as s89
from stage27_uav_collect import writer_lock


s88, s86, c, io = s89.s88, s89.s86, s89.c, s89.io
SCHEMA = "stage90_frozen_write_direction_audit_v1"
STATUS = "COMPLETE_FROZEN_WRITE_DIRECTION_AUDIT"
SOURCE_FINGERPRINT = "3bfe9d9b7e5d2a0a90558d800142174789dcd7581e61271b0aaeb3beabbaccfd"
DESIGN = dict(s89.DESIGN, minimum_source_groups_per_sign=s88.DESIGN["minimum_source_groups_per_sign"],
              switch_margin=1e-12, bootstrap_seed=2190)
TABLES = ("signal_support", "paired", "decisions")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "docs/experiments/06_sgla_uav_tracking/results/stage89_results_20261110T043903Z_1088951.txt"


def read_bundle(path):
    raw = path.read_bytes()
    pieces = re.split(rb"^===== (stage89_\S+) =====\r?\n", raw, flags=re.M)
    sections = {k.decode(): v.strip(b"\r\n") for k, v in zip(pieces[1::2], pieces[2::2])}
    if len(sections) != 5 or len(pieces) != 11 or set(sections) != {
            "stage89_decision.json", *(f"stage89_{n}.csv" for n in s89.TABLES)}:
        raise ValueError("Expected the original Stage-89 bundle sections")
    report = json.loads(sections["stage89_decision.json"])
    fingerprint = c.fingerprint({k: v for k, v in report.items() if k != "fingerprint"})
    expected_tables = {f"stage89_{n}.csv" for n in s89.TABLES}
    if (report["schema"] != s89.SCHEMA or report["status"] != s89.STATUS
            or fingerprint != report["fingerprint"] or fingerprint != SOURCE_FINGERPRINT
            or set(report["table_sha256"]) != expected_tables
            or report["training_updates"] or report["confirmation_inference_calls"]):
        raise ValueError("Unexpected Stage-89 source decision")
    for name, digest in report["table_sha256"].items():
        if hashlib.sha256(sections[name] + b"\r\n").hexdigest() != digest:
            raise ValueError("Stage-89 table bytes differ: " + name)
    rows = list(csv.DictReader(streams.StringIO(sections["stage89_windows.csv"].decode())))
    roster = c.c66.sealed(s86.ROSTER, s86.confirmation.SCHEMA)
    held = {r["source_group_proxy"] for r in roster["sequences"]}
    cases, groups = set(), set()
    for r in rows:
        r["source_group"] = s89.source_group(r["sequence"])
        if (r["case"] in cases or r["source_group"] in held or r["stratum"] not in s89.STRATA
                or r["cohort"] not in ("expanded", "historical") or r["ppo_action"] == r["challenger_action"]):
            raise ValueError("Unexpected Stage-89 window identity")
        cases.add(r["case"])
        groups.add(r["source_group"])
        for k in ("anchor_iou", "followup_gain", *(name + "_margin" for name in s88.SIGNALS)):
            r[k] = float(r[k])
            if not np.isfinite(r[k]):
                raise ValueError("Nonfinite Stage-89 value")
        for k in ("ppo_return", "challenger_return"):
            if r[k] not in ("True", "False"):
                raise ValueError("Invalid recovery flag")
            r[k] = r[k] == "True"
        r["low_iou_delta_frames"] = int(r["low_iou_delta_frames"])
    if len(rows) != report["windows"] or len(groups) != 28 or len(held) != 28:
        raise ValueError("Unexpected Stage-89 support")
    return rows, report, hashlib.sha256(raw).hexdigest()


def decisions(rows, design):
    result = []
    for r in rows:
        gain, low = r["followup_gain"], r["low_iou_delta_frames"]
        initially_low = r["anchor_iou"] < design["low_iou"]
        result.append(dict({k: r[k] for k in ("case", "sequence", "source_group", "seed", "first", "cohort", "stratum",
                            "anchor_iou", "ppo_action", "challenger_action", "followup_gain", "low_iou_delta_frames")},
            **{name + "_margin": r[name + "_margin"] for name in s88.SIGNALS},
            **{name + "_switch": r[name + "_margin"] > design["switch_margin"] for name in s88.SIGNALS},
            material_help=gain >= design["material_gain"], material_harm=gain <= -design["material_gain"],
            tail_help=low <= -design["material_low_frames"], tail_harm=low >= design["material_low_frames"],
            ppo_only_recovery=initially_low and r["ppo_return"] and not r["challenger_return"],
            challenger_only_recovery=initially_low and r["challenger_return"] and not r["ppo_return"]))
    return result


def summarize(rows, design):
    support, paired = [], []
    flags = ("material_help", "material_harm", "tail_help", "tail_harm", "ppo_only_recovery", "challenger_only_recovery")
    for cohort in ("all", "expanded", "historical"):
        for stratum in s89.STRATA:
            selected = [r for r in rows if r["stratum"] == stratum and (cohort == "all" or r["cohort"] == cohort)]
            if not selected:
                continue
            outcomes = {flag: [r for r in selected if r[flag]] for flag in flags}
            groups = {flag: len({r["source_group"] for r in subset}) for flag, subset in outcomes.items()}
            rank_support = min(groups["material_help"], groups["material_harm"]) >= design["minimum_source_groups_per_sign"]
            for method in s88.METHODS:
                take = {r["case"] for r in selected if method == "always_challenger" or r[method + "_switch"]}
                auc = None
                if rank_support and method in s88.SIGNALS:
                    margin = method + "_margin"
                    diff = np.array([r[margin] for r in outcomes["material_help"]])[:, None] - [
                        r[margin] for r in outcomes["material_harm"]]
                    auc = float(np.mean((diff > 1e-12) + .5 * (np.abs(diff) <= 1e-12)))
                by_video = defaultdict(list)
                for r in selected:
                    gain, low = (r["followup_gain"], r["low_iou_delta_frames"] / design["horizon"]) if r["case"] in take else (0., 0.)
                    by_video[r["source_group"]].append((gain, low))
                values = np.array([np.mean(by_video[key], axis=0) for key in sorted(by_video)])
                support.append(dict(cohort=cohort, stratum=stratum, method=method, windows=len(selected),
                    source_groups=len(values), switches=len(take), rank_support=rank_support, rank_auc=auc,
                    **{flag + "_available": len(outcomes[flag]) for flag in flags},
                    **{flag + "_groups": groups[flag] for flag in flags},
                    **{flag + "_switched": sum(r["case"] in take for r in outcomes[flag]) for flag in flags}))
                for i, metric in enumerate(("followup_mean_iou", "low_iou_fraction")):
                    row = s86.contrast(values[:, i], method + "_minus_ppo", metric, design)
                    paired.append(dict(row, cohort=cohort, stratum=stratum, windows=len(selected)))
    return support, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=REPO / "output/stage90/frozen_signal_audit_v1")
    args = parser.parse_args()
    rows, source, digest = read_bundle(args.bundle)
    selected = decisions(rows, DESIGN)
    support, paired = summarize(selected, DESIGN)
    root = args.output_dir.resolve()
    with writer_lock(root):
        hashes = {}
        for name, table in zip(TABLES, (support, paired, selected)):
            path = root / f"stage90_{name}.csv"
            s86.publish_table(path, table)
            hashes[path.name] = c.file_hash(path)
        c.publish_identical(root / "stage90_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
            source_stage89_fingerprint=source["fingerprint"], source_bundle_sha256=digest,
            analysis_sha256=c.file_hash(Path(__file__)), windows=len(rows), design=DESIGN,
            training_updates=0, new_search_calls=0, confirmation_inference_calls=0, table_sha256=hashes)))
    print(f"Stage-90: {STATUS} ({len(rows)} cached windows)\nReports: {root}")


if __name__ == "__main__":
    main()
