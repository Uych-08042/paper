"""Audit selector residuals after removing a source-held common logit direction."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import stage178_selector_logits_audit as source


SOURCE178 = "941317bf0105565e62e01ee709be16c10ea595cfc6d7ac56403b3d7e9cb72630"
SCHEMA = "stage179_selector_residual_audit_v1"
LAYERS = source.LAYERS
POLICIES = ("uniform", "raw", "centered", "residual")
ENDPOINTS = ("benefit", "harm", "delta128_or_shorter")


def load_input(bundle, directory):
    if bundle is not None:
        return source.gain.prior.archives.bundle(bundle, 178, SOURCE178)
    decision = source.c.c66.sealed(directory / "stage178_decision.json", source.SCHEMA)
    if decision["fingerprint"] != SOURCE178:
        raise ValueError("Need the bound Stage-178 report")
    source.io.verify({str(directory / name): h for name, h in decision["table_sha256"].items()})
    tables = {name: source.gain.prior.prior.read_table(directory / name) for name in decision["table_sha256"]}
    return decision, tables


def leave_source_out(signals):
    case_ids = [r["case_id"] for r in signals]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate Stage-178 anchor")
    group = np.array([r["group"] for r in signals])
    x = np.array([[float(r[f"logit_l{layer}"]) for layer in LAYERS] for r in signals])
    unique = sorted(set(group))
    if len(unique) < 2 or not np.isfinite(x).all():
        raise ValueError("Need finite logits from at least two source groups")
    by_id, fits = {}, []
    for held in unique:
        use = group != held
        counts = Counter(group[use])
        weights = np.array([1. / counts[g] for g in group[use]])
        mean = np.average(x[use], axis=0, weights=weights)
        _, singular, basis = np.linalg.svd((x[use] - mean) * np.sqrt(weights[:, None]), full_matrices=False)
        if singular[0] <= 0.:
            raise ValueError(f"Training logits are constant for {held}")
        direction = basis[0].copy()
        if direction[np.abs(direction).argmax()] < 0:
            direction = -direction
        centered = x[~use] - mean
        along = centered @ direction
        residual = centered - np.outer(along, direction)
        np.testing.assert_allclose(residual @ direction, 0., rtol=0., atol=1e-10)
        energy = float(np.square(centered).sum())
        fits.append(dict(held_group=held, fit_groups=len(unique) - 1, fit_states=int(use.sum()),
                         held_states=int((~use).sum()), fit_component_variance_share=float(singular[0]**2 / (singular**2).sum()),
                         held_energy_removed=float((along**2).sum() / energy) if energy else None,
                         **{f"mean_l{k}": float(m) for k, m in zip(LAYERS, mean, strict=True)},
                         **{f"axis_l{k}": float(v) for k, v in zip(LAYERS, direction, strict=True)}))
        for i, raw, ctr, res in zip(np.flatnonzero(~use), x[~use], centered, residual, strict=True):
            by_id[case_ids[i]] = {"raw": raw, "centered": ctr, "residual": res}
    return by_id, fits


def attach(ranked, alternatives, signals, transforms):
    anchors = {r["case_id"]: r for r in signals}
    output = []
    for rows in (ranked, alternatives):
        typed = []
        for row in rows:
            key, layer = row["case_id"], int(row["layer"])
            record = anchors[key]
            if row["group"] != record["group"]:
                raise ValueError(f"Stage-178 label source is inconsistent at {key}")
            n, k = LAYERS.index(int(record["native"])), LAYERS.index(layer)
            if n == k:
                raise ValueError("Expected a non-native alternative")
            result = dict(case_id=key, group=row["group"], cohort=row["cohort"], window=int(row["window"]), layer=layer,
                          benefit=row["benefit"] == "True", harm=row["harm"] == "True",
                          delta128_or_shorter=float(row["delta128_or_shorter"]),
                          candidate_uniform=0., selector_uniform=0.)
            for name, scores in transforms[key].items():
                result[f"candidate_{name}"] = float(scores[k] - scores[n])
                result[f"selector_{name}"] = float(np.max(np.delete(scores, n)) - scores[n])
            np.testing.assert_allclose(result["candidate_raw"], float(row["candidate_raw"]), rtol=0., atol=1e-12)
            if "selector_raw" in row:
                np.testing.assert_allclose(result["selector_raw"], float(row["selector_raw"]), rtol=0., atol=1e-12)
            typed.append(result)
        output.append(typed)
    return output


def association(ranked, alternatives):
    summary, contrast = [], []
    for cohort in ("stage176_all", "stage176_other", "stage177_held"):
        scope = {"stage176_known", "stage176_other"} if cohort == "stage176_all" else {cohort}
        for kind, source_rows, terms, block in (("ranked_across_states", ranked, ("selector", "candidate"), "group"),
                                               ("alternatives_same_state", alternatives, ("candidate",), "case_id")):
            rows = [r for r in source_rows if r["cohort"] in scope and r["window"] == source.event.LONG]
            if not rows:
                continue
            groups = defaultdict(list)
            for r in rows:
                groups[r["group"]].append(r)
            for term in terms:
                for endpoint in ENDPOINTS:
                    values, n_pairs = {}, 0
                    for policy in POLICIES:
                        values[policy] = {}
                        n_pairs = 0
                        for group, sample in sorted(groups.items()):
                            pairs, _, mean = source.pair_concordance(sample, f"{term}_{policy}", endpoint, block)
                            n_pairs += pairs
                            if mean is not None:
                                values[policy][group] = mean
                        means = list(values[policy].values())
                        summary.append(dict(cohort=cohort, kind=kind, term=term, outcome=endpoint, policy=policy,
                                            records=len(rows), positives=sum(r[endpoint] for r in rows)
                                            if endpoint != "delta128_or_shorter" else None,
                                            comparable_pairs=n_pairs, supported_groups=len(means),
                                            concordance=float(np.mean(means)) if means else None))
                    for control in POLICIES[:-1]:
                        if values["residual"].keys() != values[control].keys():
                            raise ValueError("Concordance support differs between scoring rules")
                        delta = [v - values[control][g] for g, v in values["residual"].items()]
                        if delta:
                            stat = source.base.s103.contrast(delta, "residual_minus_" + control,
                                                            "concordance", source.online.DESIGN)
                            contrast.append(dict(cohort=cohort, kind=kind, term=term, outcome=endpoint, control=control,
                                                 comparable_pairs=n_pairs, **stat))
    return summary, contrast


def top_alternative(alternatives):
    states = defaultdict(list)
    for r in alternatives:
        if r["window"] == source.event.LONG:
            states[r["case_id"]].append(r)
    picks = []
    for key, rows in states.items():
        if len(rows) != len(LAYERS) - 1 or len({r["layer"] for r in rows}) != len(rows):
            raise ValueError(f"Incomplete six-layer support at {key}")
        for policy in POLICIES:
            chosen = None if policy == "uniform" else max(rows, key=lambda r: (r[f"candidate_{policy}"], -r["layer"]))
            picks.append(dict(case_id=key, group=rows[0]["group"], cohort=rows[0]["cohort"], policy=policy,
                              selected_layer=None if chosen is None else chosen["layer"],
                              **{k: float(np.mean([r[k] for r in rows])) if chosen is None else float(chosen[k])
                                 for k in ENDPOINTS}))
    summary = []
    for scope in ("all", "other"):
        for policy in POLICIES:
            rows = [r for r in picks if r["policy"] == policy and (scope == "all" or r["cohort"] == "stage176_other")]
            grouped = defaultdict(list)
            for r in rows:
                grouped[r["group"]].append(r)
            summary.append(dict(scope=scope, policy=policy, cases=len(rows), source_groups=len(grouped),
                                **{f"mean_{k}": float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()]))
                                   if grouped else None for k in ENDPOINTS}))
    return picks, summary


def analyze(tables):
    signals = tables["stage178_signals.csv"]
    if len(signals) != 265 or any(r["same_native_state"] != "True" for r in signals):
        raise ValueError("Need all 265 matching native states")
    transforms, fits = leave_source_out(signals)
    ranked, alternative = attach(tables["stage178_ranked_outcomes.csv"], tables["stage178_alternative_outcomes.csv"], signals, transforms)
    summary, paired = association(ranked, alternative)
    picks, top = top_alternative(alternative)
    return dict(fits=fits, ranked_scores=ranked, alternative_scores=alternative, association=summary,
                paired=paired, forced_choices=picks, forced_choice_summary=top)


def publish_bundle(root, table_names):
    bundle = root / "stage179_results.txt"
    payload = b"".join(f"\n===== {name} =====\n".encode() + (root / name).read_bytes()
                       for name in ("stage179_decision.json", *table_names))
    if bundle.exists() and bundle.read_bytes() == payload:
        return
    if bundle.exists():
        raise ValueError("Existing Stage-179 bundle differs from the verified result")
    temporary = bundle.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bundle", type=Path)
    source_group.add_argument("--stage178-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.stage178_root:
        directory = args.stage178_root.expanduser().resolve()
        if directory in (root, *root.parents) or root in directory.parents:
            parser.error("Use a separate output directory")
    decision, tables = load_input(args.bundle, args.stage178_root)
    if not decision["association_same_runtime"]:
        raise ValueError("Need the completed same-runtime Stage-178 capture")
    plan = source.io.seal(dict(schema=SCHEMA, source_stage178=SOURCE178, source_table_sha256=decision["table_sha256"],
                               split="leave_one_source_group_out", fit_group_weight="equal", removed_components=1,
                               policies=list(POLICIES), code_sha256=source.c.file_hash(Path(__file__)), numpy=np.__version__))
    with source.gain.prior.prior.writer_lock(root):
        source.c.publish_identical(root / "manifest.json", plan)
        path = root / "stage179_decision.json"
        if path.exists():
            saved = source.c.c66.sealed(path, SCHEMA)
            if saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Existing Stage-179 decision differs from the plan")
            hashes = saved["table_sha256"]
            source.io.verify({str(root / k): v for k, v in hashes.items()})
        else:
            result = analyze(tables)
            hashes = source.gain.prior.publish(root, result, stage="stage179")
            source.c.publish_identical(path, source.io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE179_SELECTOR_RESIDUAL_AUDIT",
                plan_fingerprint=plan["fingerprint"], source_stage178=SOURCE178, states=len(tables["stage178_signals.csv"]),
                source_groups=len(result["fits"]), new_searches=0, label_fitting=False, table_sha256=hashes)))
        publish_bundle(root, sorted(hashes))
    print(f"Stage-179 complete. Result bundle: {root / 'stage179_results.txt'}", flush=True)


if __name__ == "__main__":
    main()
