"""Replay selective serial depth on the paired Stage-155 development windows."""

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, quantiles

import numpy as np


FINGERPRINTS = {137: "a092d092f5fc2ff684f888c243197eefe6a8c70f430268d820c8475d2c38e89c",
                155: "1ba5ee6090a88adc6adbb908f2b2a7ee2d88a8e88f73164954479371ce5ebd4d"}
ARMS = ("frozen_l10", "adapted_l10")
SIGNALS = ("selector_margin", "native_hann_peak", "native_peak_centrality")
GAIN = "predicted_32_frame_gain"


def bundle(path, stage, expected_fingerprint=None):
    parts = re.split(rb"\n===== ([\w.]+) =====\r?\n", path.read_bytes())
    sections = {name.decode(): content for name, content in zip(parts[1::2], parts[2::2], strict=True)}
    decision = json.loads(sections[f"stage{stage}_decision.json"])
    fingerprint = decision.pop("fingerprint")
    digest = sha256(json.dumps(decision, sort_keys=True, allow_nan=False).encode()).hexdigest()
    expected = FINGERPRINTS[stage] if expected_fingerprint is None else expected_fingerprint
    if fingerprint != digest or fingerprint != expected:
        raise ValueError(f"Unexpected Stage-{stage} decision")
    tables = {}
    for name, digest in decision["table_sha256"].items():
        content = sections[name]
        if sha256(content).hexdigest() != digest:
            raise ValueError(f"Stage-{stage} CSV bytes differ from the decision receipt: {name}")
        tables[name] = list(csv.DictReader(content.decode().splitlines()))
    return decision, tables


def matched_points(path137, path155):
    decision137, s137 = bundle(path137, 137)
    decision155, s155 = bundle(path155, 155)
    if decision137["runtime"] != decision155["runtime"]:
        raise ValueError("Native signal and continuation runtimes differ")
    fits = {int(row["fold"]): (set(row["fit"].split("|")), set(row["held"].split("|")))
            for row in s155["stage155_training.csv"]}
    roster = {r["case_id"]: r for r in s137["stage137_cases.csv"]}
    signals = {r["case_id"]: r for r in s137["stage137_layers.csv"] if r["native"] == "True"}
    anchors = {(r["case_id"], r["contrast"]): r for r in s155["stage155_anchors.csv"]}
    tails = {(r["case_id"], r["contrast"], int(r["horizon"])): r for r in s155["stage155_windows.csv"]}
    if (len(roster) != 47 or set(roster) != set(signals)
            or len(anchors) != 3*len(roster) or len(tails) != 6*len(roster)):
        raise ValueError("Incomplete paired-window roster")
    points = []
    for cid, case in roster.items():
        signal = signals[cid]
        r, c = divmod(int(signal["hann_cell"]), 16)
        point = dict(case_id=cid, sequence=case["sequence"], frame=int(case["frame"]), kind=case["kind"],
                     selector_margin=float(case["raw_selector_score_gap"]), native_hann_peak=float(signal["hann_peak"]),
                     native_peak_centrality=-((r-8)**2+(c-8)**2), outcomes={})
        source = case["sequence"].split("_", 1)[0]
        for arm in ARMS:
            contrast = arm+"_minus_native"
            a = anchors[cid, contrast]
            fold = int(a["fold"])
            point.setdefault("fold", fold)
            if (fold != point["fold"] or source in fits[fold][0] or source not in fits[fold][1]
                    or a["evaluated"] != "True" or a["primary_layer"] != "9" or a["extra_layer"] != "10"
                    or any(a[k] != case[k] or signal[k] != case[k] for k in ("sequence", "frame", "group"))
                    or abs(float(signal["top1_iou"])-float(a["reference_iou"])) > 1e-12):
                raise ValueError(f"Unmatched native state or fit source for {cid}")
            effects = dict(anchor_delta=float(a["anchor_delta"]), new_anchor_good=a["new_good"] == "True",
                           lost_anchor_good=a["lost_good"] == "True")
            for h in (8, 32):
                t = tails[cid, contrast, h]
                if int(t["valid_successors"]) != h or int(t["fold"]) != fold:
                    raise ValueError(f"Incomplete successor window for {cid}")
                effects[f"tail{h}"] = float(t["tail_mean_delta"])
            reference_good, treatment_good = (bool(t[f"{k}_first_good_run"]) for k in ("reference", "treatment"))
            effects.update(new_good_run=treatment_good and not reference_good,
                           lost_good_run=reference_good and not treatment_good,
                           new_low_run=bool(t["treatment_first_low_run"]) and not t["reference_first_low_run"])
            point["outcomes"][arm] = effects
        points.append(point)
    if Counter(p["kind"] for p in points) != {"onset": 19, "healthy": 28}:
        raise ValueError("Expected 19 onset and 28 healthy anchors")
    return points


def routed(points, signal):
    if signal != GAIN:
        thresholds = {fold: quantiles([p[signal] for p in points if p["fold"] != fold], n=4, method="inclusive")[0]
                      for fold in (0, 1)}
        values = [p[signal] for p in points]
        return [float(v < thresholds[p["fold"]]) for p, v in zip(points, values)], thresholds, values
    # Train only on frozen-L10 outcomes to avoid indirect dependence on the held-out group's labels
    # through the adapted models. Use the same gate with both L10 versions.
    features = np.array([[p[k] for k in SIGNALS[1:]] for p in points])
    targets = np.array([p["outcomes"]["frozen_l10"]["tail32"] for p in points])
    values, thresholds = np.empty(len(points)), {}
    for fold in (0, 1):
        fit = np.array([p["fold"] != fold for p in points])
        mu, scale = features[fit].mean(axis=0), features[fit].std(axis=0)
        x = (features-mu)/np.where(scale > 0, scale, 1.)
        design, target = x[fit], targets[fit]
        intercept = target.mean()
        weights = np.linalg.solve(design.T@design+len(design)*np.eye(x.shape[1]), design.T@(target-intercept))
        thresholds[fold] = max(0., float(np.quantile(intercept+design@weights, .75)))
        values[~fit] = intercept+x[~fit]@weights
    return [float(v > thresholds[p["fold"]]) for p, v in zip(points, values)], thresholds, values.tolist()


def score(points, arm, policy, use, limit):
    # Random controls pass the per-fold selection probabilities, so all means and counts are expectations.
    rows = []
    for kind in ("all", "onset", "healthy"):
        block = [(p["sequence"], u, p["outcomes"][arm]) for p, u in zip(points, use, strict=True)
                 if kind == "all" or p["kind"] == kind]
        row = dict(arm=arm, policy=policy, kind=kind, points=len(block), extra_blocks=sum(u for _, u, _ in block),
                   budget_reference=limit)
        for key in ("anchor_delta", "tail8", "tail32"):
            videos = defaultdict(list)
            for seq, u, effect in block:
                videos[seq].append(u*effect[key])
            by_video = [mean(v) for v in videos.values()]
            row[key] = mean(by_video)
            if key == "tail32":
                row["worst_leave_one_video_out_tail32"] = (sum(by_video)-max(by_video))/(len(by_video)-1)
        for key in ("new_anchor_good", "lost_anchor_good", "new_good_run", "lost_good_run", "new_low_run"):
            row[key] = sum(u*e[key] for _, u, e in block)
        row["tail32_gain_points"] = sum(u*(e["tail32"] > 1e-8) for _, u, e in block)
        row["tail32_harm_points"] = sum(u*(e["tail32"] < -1e-8) for _, u, e in block)
        rows.append(row)
    return rows


def oracle(points, arm, budget):
    """At most the per-fold quota, using future truth to bound improvement on these paths."""
    counts = Counter(p["sequence"] for p in points)
    use = [0.]*len(points)
    for fold in (0, 1):
        candidates = []
        for i, p in enumerate(points):
            e = p["outcomes"][arm]
            if (p["fold"] == fold and e["tail32"] > 0 and not e["lost_good_run"]
                    and not e["new_low_run"] and not e["lost_anchor_good"]):
                candidates.append((e["tail32"]/counts[p["sequence"]], i))
        for _, i in sorted(candidates, reverse=True)[:budget[fold]]:
            use[i] = 1.
    return use


def audit(points):
    rows, decisions = [], []
    n = len(points)
    for arm in ARMS:
        rows.extend(score(points, arm, "no_extra", [0.]*n, "fixed"))
        rows.extend(score(points, arm, "always_extra", [1.]*n, "fixed"))
        for signal in (*SIGNALS, GAIN):
            use, thresholds, values = routed(points, signal)
            budget = {f: int(sum(u for p, u in zip(points, use) if p["fold"] == f)) for f in (0, 1)}
            random = [budget[p["fold"]]/sum(q["fold"] == p["fold"] for q in points) for p in points]
            rows.extend(score(points, arm, signal, use, signal))
            rows.extend(score(points, arm, "matched_random_expected", random, signal))
            rows.extend(score(points, arm, "protected_hindsight", oracle(points, arm, budget), signal))
            decisions.extend(dict(case_id=p["case_id"], arm=arm, fold=p["fold"], signal=signal,
                                  value=v, threshold=thresholds[p["fold"]], add=bool(u), **p["outcomes"][arm])
                             for p, u, v in zip(points, use, values, strict=True))
    return rows, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage137", required=True, type=Path)
    parser.add_argument("--stage155", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    points = matched_points(args.stage137, args.stage155)
    summary, decisions = audit(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary), ("decisions", decisions)):
        with (args.output_dir/f"stage156_{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for r in summary:
        if r["kind"] == "all":
            print(f"{r['arm']:12} {r['policy']:23} {r['budget_reference']:22} "
                  f"extra={r['extra_blocks']:5.1f}  tail32={r['tail32']:+.5f}")


if __name__ == "__main__":
    main()
