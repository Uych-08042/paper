"""Freeze fresh source-group 8-fit / 6-checking videos before any Stage-31 prediction."""

import argparse
from pathlib import Path

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical
from stage27_uav_summary import write_table
from stage31_identity_common import build_contract, load_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=nonempty_path, required=True, help="Complete Stage-30 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Stage-31 output must be separate from the frozen evidence")
    with writer_lock(output):
        path = output / "manifest.json"
        if path.exists():
            contract = load_contract(path)
            if contract["evidence_input"] != str(source):
                raise ValueError("Existing manifest has a different source")
            print("Existing frozen contract reused; no resampling.")
        else:
            if any(p.name != "collect.lock" for p in output.iterdir()):
                raise ValueError("Use a new empty output directory")
            contract = build_contract(source)
            publish_identical(path, contract)
        rows = [dict(sequence=e["sequence"], split=e["split"], source_group_proxy=e["source_group_proxy"],
                     tracked_frames=len(e["frames"]) - 1) for e in contract["sequences"]]
        write_table(output / "stage31_split.csv", rows)
        print("Manifest: " + str(path))
        print("8 fit / 6 checking; all prior diagnostic groups excluded; full videos.")
        print("Fingerprint: " + contract["fingerprint"])
        print("Dataset audit only. No model inference or training.")


if __name__ == "__main__":
    main()
