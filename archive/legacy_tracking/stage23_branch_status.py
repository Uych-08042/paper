"""Read atomic Stage-23 progress reports without importing PyTorch."""

import argparse
import csv
import json
from pathlib import Path

from stage23_branch_credit import ARMS


def read_status(directory):
    directory = Path(directory).expanduser().resolve()
    contract_path = directory / "stage23_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError("Stage-23 contract missing: " + str(contract_path))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = []
    for arm in ARMS:
        path = directory / arm / "stage23_history.csv"
        history = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        last = history[-1] if history else None
        rows.append(dict(arm=arm, reported_steps=int(last["steps"]) if last else 0,
                         branch_transitions=sum(int(r["branch_transitions"]) for r in history),
                         latest_iou=float(last["mean_valid_iou"]) if last else None,
                         maximum_replay_error=max((max(float(r[k]) for k in (
                             "likelihood_replay_error", "probability_replay_error", "chosen_replay_error"))
                             for r in history), default=None),
                         checkpoint_exists=(directory / arm / "last.pt").is_file()))
    decision_path = directory / "stage23_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else None
    if decision is not None and decision.get("contract_fingerprint") != contract.get("fingerprint"):
        raise ValueError("Decision belongs to a different contract")
    return rows, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    if not args.input.strip():
        parser.error("input must not be empty")
    rows, decision = read_status(args.input)
    print("arm               reported_live  branch_frames  latest_fit_IoU  replay_error")
    for r in rows:
        iou = "n/a" if r["latest_iou"] is None else "{:.4f}".format(r["latest_iou"])
        error = "n/a" if r["maximum_replay_error"] is None else "{:.2g}".format(r["maximum_replay_error"])
        print("{arm:<19s} {reported_steps:>10d} {branch_transitions:>14d} {iou:>15s} {error:>13s}".format(
            **r, iou=iou, error=error))
    print("Decision: " + (decision["status"] if decision else "PENDING_MATCHED_SMOKE"))
    print("CSV progress can lag one atomic checkpoint. Resume the same command to reconcile it.")
    print("Latest fit IoU is a training diagnostic, not a paired policy comparison or validation result.")


if __name__ == "__main__":
    main()
