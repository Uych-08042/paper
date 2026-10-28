# Stage-11: Frozen External Full-Sequence Validation

## Purpose

Stage-11 is the final experiment, not another policy-search stage. It evaluates
the already selected Stage-5 operating points on untouched OTB-2015 full
sequences. The frozen policy set is:

- `fast`: no candidate block;
- `periodic_l9_20`: secondary sparse baseline;
- `periodic_l9_30`: predeclared primary policy;
- `sgla`: one selector-chosen candidate block on every tracked frame.

The primary decision was fixed before observing OTB outcomes:

1. `P30 - Fast` paired sequence-bootstrap 95% CI lower bound is above zero;
2. `P30 - SGLA` paired 95% CI lower bound is above `-0.5` AUC;
3. P30 executes approximately 30% candidate blocks;
4. measured P30 model/decode latency is below SGLA.

Do not add rates, layers, schedules, gates, or subsets after this run.

## 1. Prepare OTB-2015

Extract the complete OTB-2015 archive to a directory whose immediate children
include `Basketball`, `Biker`, `Bird1`, and the other OTB sequence folders.
The repository loader represents the benchmark as 100 sequence objects over
98 physical videos (two videos have two annotation tracks). Point estimates
use the standard 100-object macro average; confidence intervals bootstrap the
98 physical-video clusters.

The dataset is available from the OTB2015 Figshare record:
`https://figshare.com/articles/dataset/OTB2015/24427468`. To inspect the current
archive filename, checksum, size, and download URL before downloading, run:

```bash
python - <<'PY'
import requests

metadata = requests.get(
    "https://api.figshare.com/v2/articles/24427468", timeout=60
).json()
for item in metadata["files"]:
    print("name:", item["name"])
    print("size_GiB:", round(item["size"] / 1024**3, 3))
    print("download_url:", item["download_url"])
    print("md5:", item["supplied_md5"])
PY
```

Keep roughly 6 GB free while the archive and extracted dataset coexist. After
the extraction audit succeeds, the archive itself can be removed if space is
tight.

From the CUDA server repository root:

```bash
cd /home/u25600003160604/workspace/paper

export OTB_ROOT="$HOME/workspace/datasets/OTB100"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE11_DIR="$PWD/output/stage11/otb_frozen_external_v1"
```

Audit the extraction without generating predictions or viewing outcomes:

```bash
python tracking/stage11_external_validate.py \
  --dataset_root "$OTB_ROOT" \
  --audit_only
```

The audit must report exactly 100 sequence objects, 98 physical videos, and no
missing frames.

## 2. Run the frozen experiment

There is deliberately no one-sequence OTB sanity run. The execution path has
already been validated on UAV123 and GOT-10k clips, so OTB remains an untouched
full benchmark.

Single GPU:

```bash
python tracking/stage11_external_validate.py \
  --dataset_root "$OTB_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --output_dir "$STAGE11_DIR"
```

The runner is resumable. Reissuing the same command skips completed files and
refuses any changed dataset, checkpoint, code, policy set, or shard count.

For two GPUs, start two processes with the same output directory:

```bash
python tracking/stage11_external_validate.py \
  --dataset_root "$OTB_ROOT" --checkpoint "$CHECKPOINT" \
  --gpu 0 --num_shards 2 --shard_id 0 --output_dir "$STAGE11_DIR" &

python tracking/stage11_external_validate.py \
  --dataset_root "$OTB_ROOT" --checkpoint "$CHECKPOINT" \
  --gpu 1 --num_shards 2 --shard_id 1 --output_dir "$STAGE11_DIR" &

wait
```

## 3. Run the frozen analysis once

```bash
python tracking/stage11_external_summary.py \
  --input "$STAGE11_DIR" \
  --bootstrap_samples 10000 \
  --seed 2041

cat "$STAGE11_DIR/analysis/stage11_external_policy_summary.csv"
cat "$STAGE11_DIR/analysis/stage11_external_paired.csv"
cat "$STAGE11_DIR/analysis/stage11_external_decision.json"
```

The summary refuses incomplete coverage and emits one of three terminal states:

- `PASS`: the frozen P30 Pareto claim transfers;
- `NO_CONFIRMED_HEADROOM`: SGLA itself does not establish improvement over
  Fast on OTB, so recovery cannot be confirmed on this benchmark;
- `FAIL_TRANSFER`: SGLA has confirmed headroom but frozen P30 misses the
  predeclared criteria.

All three states end policy experimentation. Report the result as observed;
do not use OTB to choose a replacement schedule.
