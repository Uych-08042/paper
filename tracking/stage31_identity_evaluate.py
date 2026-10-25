"""One frozen full-video checking evaluation; no GT-dependent online decisions."""

import argparse
from pathlib import Path

from stage4_common import write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage31_identity_common import METHODS, SCHEMA, load_contract
from stage31_identity_engine import audit_video, cache_metadata, load_video, run_video
from stage31_identity_fit import initialize_observer, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--checkpoint", type=nonempty_path, required=True)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="Resume/debug one checking video; no parameter changes")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    contract = load_contract(directory / "manifest.json")
    model = load_model(directory, contract)
    entries = [e for e in contract["sequences"] if e["split"] == "checking"]
    if args.sequence and args.sequence not in {e["sequence"] for e in entries}:
        raise ValueError("--sequence is not in the frozen checking set")
    with writer_lock(directory):
        observer = initialize_observer(contract, args.parameter, args.gpu, args.checkpoint)
        run = seal(dict(schema=SCHEMA, role="frozen_full_video_identity_checking", contract_fingerprint=contract["fingerprint"],
            model_fingerprint=model["fingerprint"], runtime=observer.identity, methods=list(METHODS),
            method_order="Rotate the fixed method list by checking-video index; reset all state/template per method",
            gt_use="First-frame initialization; future labels logged only after actor state advance",
            resource_role="Descriptive crop/model/decode/association wall time; image reads and label logging excluded; not end-to-end FPS"))
        if not (directory / "checking_run.json").exists() and any((directory / "checking").glob("*.npz")):
            raise ValueError("Unidentified checking caches; preserve them and use a new run directory")
        publish_identical(directory / "checking_run.json", run)
        for index, entry in enumerate(entries):
            if args.sequence and entry["sequence"] != args.sequence:
                continue
            offset = index % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                path = directory / "checking" / (entry["sequence"] + "__" + method + ".npz")
                if path.exists():
                    load_video(path, contract, entry, method, run, model)
                    print("[reuse, audited] " + path.name, flush=True)
                    continue
                data = run_video(observer, contract, entry, method, model)
                audit_video(data, contract, entry, method, model)
                write_cache(path, data, cache_metadata(run, entry, method))
                load_video(path, contract, entry, method, run, model)
                print("[checking] " + path.name, flush=True)
    print('Next: python tracking/stage31_identity_summary.py --input "{}"'.format(directory))


if __name__ == "__main__":
    main()
