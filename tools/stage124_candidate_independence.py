"""Fit paired shared scorers with and without same-frame candidate relations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

import stage123_frozen_retained_scores as previous
from stage27_uav_common import tracking_iou

prior, base, c, io = previous.prior, previous.base, previous.c, previous.io
common, s121 = base.common, previous.s121
SCHEMA = "stage124_candidate_independence_v1"
STATUS = "COMPLETE_MATCHED_CANDIDATE_SCORING_DIAGNOSTIC"
ARMS = ("relation", "independent")
SOURCES = (prior.CONTROL, s121.TREATMENT)
RELATION_COLUMNS = (24, 25, 26, 29, 30, 31)
MASK_COLUMNS = (*RELATION_COLUMNS, *(j+32 for j in RELATION_COLUMNS))
DESIGN = dict(uniform_per_trajectory=64, good=.5, low=.2, bootstrap_seed=23124, bootstrap_samples=20000,
              training=common.DESIGN, mask_columns=list(MASK_COLUMNS))
TABLES = ("model_scope", "policy_summary", "paired", "video_effects", "decisions")
WORK = dict(new_search_calls=0, new_image_reads=0, new_full_trajectories=0,
            sgla_training_updates=0, confirmation_inference_calls=0)
POOLS = ("uniform", "uniform_q0_good", "uniform_reference_rescue",
         "alternate_activation", "matched_activation", "new_good_added", "native_recovery")
METRICS = ("iou", "good", "low")


def inputs(x, arm):
    if arm == "relation":
        return x
    if arm != "independent":
        raise ValueError("Unknown feature condition")
    result = np.array(x, copy=True)
    result[..., list(MASK_COLUMNS)] = 0
    return result


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def uniform_frames(n):
    return set(map(int, np.linspace(1, n, min(n, DESIGN["uniform_per_trajectory"]), dtype=int)))


def recovery_frames(root):
    frames = defaultdict(set)
    for row in read_table(root / "analysis/stage121_recovery.csv"):
        if row["reference_arm"] == "native":
            for source in SOURCES:
                frames[row["sequence"], int(row["seed"]), source].add(int(row["reference_recovery_frame"]))
    return frames


def load_rows(r121, r122, plan, entry_for, length):
    scopes = prior.source_rows(r121, "scope")
    recovery = recovery_frames(r121)
    matched = {previous.get_event_key(r):r
               for r in previous.read_table(r122, "activations") if r["matched"] == "True"}
    rows = defaultdict(list)
    entries = {}
    jobs = [j for j in plan["jobs"] if j["arm"] in SOURCES]
    if len(jobs) != 112:
        raise ValueError("Need all Stage-121 generated and alternate source trajectories")
    seen_matched = set()
    for job in jobs:
        seq, seed, source = job["spec"]["unit"]["sequence"], job["seed"], job["arm"]
        group = base.s109.s106.source_group(seq)
        if seq not in entries:
            entries[seq] = entry_for(job["spec"])
        entry = entries[seq]
        n = job["frames"]
        if entry["sequence"] != seq or len(entry["ground_truth"]) != n + 1:
            raise ValueError("Stage-121 labels do not match trajectory")
        data = prior.load_trace(r121, plan, job)
        np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
        uniform = uniform_frames(n)
        alt = set(map(int, data["frame_id"][data["alternate_used"]])) if source == s121.TREATMENT else set()
        if len(alt) != int(scopes[seq, seed, source]["alternate_used_frames"]):
            raise ValueError("Stage-121 activation support changed")
        recover = recovery[seq, seed, source]
        for frame in sorted(uniform | alt | recover):
            if not 1 <= frame <= n:
                raise ValueError("Selected audit frame is out of bounds")
            i = frame-1
            history = prior.history(data, entry["initial_box"], i, length)
            np.testing.assert_array_equal(history[-1], data["history_latest"][i])
            boxes = data["candidate_boxes"][i]
            x = common.features(data["features"][i], boxes, history)
            y = tracking_iou(boxes, entry["ground_truth"][frame])
            if (y < 0).any():
                raise ValueError("Invalid cached IoU labels")
            selected = int(data["selected"][i])
            event_key = (seq, seed, frame)
            event = matched.get(event_key) if source == s121.TREATMENT else None
            if event is not None:
                seen_matched.add(event_key)
                if frame not in alt or selected != int(event["alternate_slot"]):
                    raise ValueError("Stage-122 action no longer reproduces")
                np.testing.assert_allclose([y[prior.SLOT], y[selected]],
                                           [float(event["alternate_slot4_iou"]), float(event["alternate_iou"])],
                                           rtol=0, atol=1e-12)
            pools = dict(uniform=frame in uniform, alternate_activation=frame in alt,
                         matched_activation=event is not None, native_recovery=frame in recover,
                         new_good_added=event is not None and event["net_good_bank_added"] == "True",
                         uniform_q0_good=frame in uniform and y[0]>=DESIGN["good"],
                         uniform_reference_rescue=frame in uniform and y[0]<DESIGN["low"] and y[selected]>=DESIGN["good"])
            rows[group].append(dict(group=group, sequence=seq, cohort=job["spec"]["cohort"], seed=seed,
                                    source_arm=source, frame=frame, fold=job["outer_fold"], x=x, y=y, boxes=boxes,
                                    recorded_iou=float(y[selected]), recorded_action=selected,
                                    fit_eligible=frame in uniform or frame in alt, **pools))
    if seen_matched != set(matched) or len(rows) != 28:
        raise ValueError("Incomplete Stage-121/122 development support")
    return rows


def training_split(fold, group, provider_folds):
    other = sorted(set(provider_folds[fold]["held_out_groups"]) - {group})
    if len(other) != 6 or set(other + [group]) != set(provider_folds[fold]["held_out_groups"]):
        raise ValueError("Need six peer groups from the same frozen provider fold")
    return dict(fold=fold, train_groups=other, held_out_groups=[group])


def fit(root, rows, split, arm, seed):
    examples = [dict(r, x=inputs(r["x"], arm)) for r in rows]
    _, _, _, signature = common.training_arrays(examples, split)
    group = split["held_out_groups"][0]
    path = root / "models" / f"{group}__{arm}__s{seed}.json"
    expected = dict(split=split, architecture="shared", seed=seed, design=common.DESIGN,
                    input_fingerprint=signature)
    if path.exists():
        model = c.c66.sealed(path, common.SCHEMA)
        if any(model[k] != value for k, value in expected.items()):
            raise ValueError("Existing fitted score model differs")
    else:
        model = common.train(examples, split, "shared", seed, common.DESIGN)
        c.publish_identical(path, model)
    return model, path


def evaluate(row, models, model_seed):
    result = {k:row[k] for k in ("group", "sequence", "cohort", "seed", "source_arm", "frame", "fold", *POOLS)}
    y, boxes = row["y"], row["boxes"]
    result.update(model_seed=model_seed, q0_iou=float(y[0]), best_iou=float(y.max()),
                  candidate_ious="|".join(map(str, y)), recorded_action=row["recorded_action"])
    actual = {"recorded":row["recorded_iou"]}
    for arm, model in models.items():
        score = common.scores(model, inputs(row["x"], arm))
        action = common.physical_choice(score, boxes)
        result[arm+"_action"] = action
        result[arm+"_scores"] = "|".join(map(str, score))
        actual[arm] = float(y[action])
    good, low = DESIGN["good"], DESIGN["low"]
    for arm, iou in actual.items():
        result.update({arm+"_iou":iou, arm+"_good":int(iou>=good), arm+"_low":int(iou<low)})
    result["lost_relation_good"] = actual["relation"]>=good>actual["independent"]
    result["lost_recorded_good"] = actual["recorded"]>=good>actual["independent"]
    result["gained_relation_good"] = actual["independent"]>=good>actual["relation"]
    result["gained_recorded_good"] = actual["independent"]>=good>actual["recorded"]
    return result


def summarize(rows):
    policy, paired, effects = [], [], []
    policies = (*ARMS, "recorded")
    for pool in POOLS:
        local = [r for r in rows if r[pool]]
        grouped = defaultdict(list)
        for row in local:
            grouped[row["group"]].append(row)
        for g, group_rows in sorted(grouped.items()):
            weights = common.p.weights(group_rows)
            means = {arm+"_"+m:float(weights @ np.array([r[arm+"_"+m] for r in group_rows]))
                     for arm in policies for m in METRICS}
            effects.append(dict(group=g, population=pool, evaluations=len(group_rows), **means))
        subset = [r for r in effects if r["population"]==pool]
        for arm in policies:
            policy.append(dict(population=pool, arm=arm, groups=len(grouped), evaluations=len(local),
                               **{m:float(np.mean([r[arm+"_"+m] for r in subset])) if subset else None for m in METRICS},
                               good_actions=sum(r[arm+"_good"] for r in local),
                               lost_relation_good=sum(r["lost_relation_good"] for r in local) if arm=="independent" else None,
                               lost_recorded_good=sum(r["lost_recorded_good"] for r in local) if arm=="independent" else None,
                               gained_relation_good=sum(r["gained_relation_good"] for r in local) if arm=="independent" else None,
                               gained_recorded_good=sum(r["gained_recorded_good"] for r in local) if arm=="independent" else None))
        if subset:
            for ref in ("relation","recorded"):
                values = [r["independent_iou"]-r[ref+"_iou"] for r in subset]
                paired.append(dict(population=pool, model_seed="all",
                                   **base.s103.contrast(values, "independent_minus_"+ref, "iou", DESIGN)))
    for seed in common.DESIGN["seeds"]:
        by_group = defaultdict(list)
        for row in rows:
            if row["uniform"] and row["model_seed"] == seed:
                by_group[row["group"]].append(row)
        if by_group:
            values = [float(common.p.weights(part) @ np.array([r["independent_iou"]-r["relation_iou"] for r in part]))
                      for _, part in sorted(by_group.items())]
            paired.append(dict(population="uniform", model_seed=seed,
                               **base.s103.contrast(values, "independent_minus_relation", "iou", DESIGN)))
    return policy, paired, effects


def run(root, plan, rows, provider_folds):
    scope, decisions, evidence = [], [], {}
    for group, held in sorted(rows.items()):
        fold = held[0]["fold"]
        if {r["fold"] for r in held}!={fold}:
            raise ValueError("Mixed provider folds for one source")
        split = training_split(fold, group, provider_folds)
        training = [r for peer in split["train_groups"] for r in rows[peer] if r["fit_eligible"]]
        if {r["fold"] for r in training} != {fold} or len({r["group"] for r in training}) != 6:
            raise ValueError("Fitting provider isolation differs")
        for seed in common.DESIGN["seeds"]:
            models = {}
            for arm in ARMS:
                model, path = fit(root, training, split, arm, seed)
                models[arm] = model
                evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                scope.append(dict(group=group, provider_fold=fold, arm=arm, model_seed=seed,
                                  examples=model["examples"], parameters=model["parameters"],
                                  optimizer_steps=model["optimizer_steps"], fit_mse=model["final_fit_mse"],
                                  fit_groups="|".join(split["train_groups"]), fingerprint=model["fingerprint"]))
            if any(models["relation"][k]!=models["independent"][k] for k in ("examples","parameters","optimizer_steps")):
                raise ValueError("Score training budgets differ")
            decisions.extend(evaluate(row, models, seed) for row in held)
        print(f"[Stage-124] {group}: {len(training)} train / {len(held)} eval states", flush=True)
    policy, paired, effects = summarize(decisions)
    publish(root, plan, dict(zip(TABLES,(scope,policy,paired,effects,decisions))), evidence)


def publish(root, plan, tables, evidence):
    expected = plan["population_counts"]
    if any(sum(r[k] for r in tables["decisions"]) != expected[k]*len(common.DESIGN["seeds"]) for k in POOLS):
        raise ValueError("Incomplete held-out evaluation")
    if len(tables["model_scope"]) != 112 or len(evidence) != 112 or len(tables["decisions"]) != plan["states"]*2:
        raise ValueError("Incomplete matched model evaluation")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for key in TABLES:
        path = out / f"stage124_{key}.csv"
        base.s103.s86.publish_table(path, tables[key])
        hashes[path.name] = c.file_hash(path)
    primary = next(r for r in tables["paired"] if r["population"]=="uniform"
                   and r["contrast"]=="independent_minus_relation" and r["model_seed"]=="all")
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
                         source_stage122_fingerprint=previous.SOURCE122, primary=primary, population_counts=expected,
                         states=plan["states"], models=len(evidence),
                         row_counts={k:len(v) for k,v in tables.items()},
                         model_sha256=evidence, table_sha256=hashes, **WORK))
    c.publish_identical(out / "stage124_decision.json", report)
    check_results(root)
    print(f"Stage-124: {STATUS}. Uniform paired IoU delta: {primary['mean_delta']:+.6f}", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage124_decision.json", SCHEMA)
    if (plan["design"]!=DESIGN
            or plan["source_stage122_fingerprint"]!=previous.SOURCE122
            or report["source_stage122_fingerprint"]!=previous.SOURCE122 or report["status"]!=STATUS
            or plan["fingerprint"]!=report["plan_fingerprint"] or report["states"]!=plan["states"]
            or report["population_counts"]!=plan["population_counts"] or report["models"]!=112
            or report["row_counts"]["model_scope"]!=112 or report["row_counts"]["decisions"]!=2*plan["states"]
            or len(report["model_sha256"])!=112 or set(report["table_sha256"])!={f"stage124_{k}.csv" for k in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-124 result")
    io.verify(plan["inputs"])
    io.verify({str(root / k):v for k,v in report["model_sha256"].items()})
    io.verify({str(root / "analysis" / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage112", "stage121", "stage122")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all(getattr(args, name) for name in names):
        parser.error("Supply Stage-76/89/93/112/121/122 directories")
    r76,r89,r93,r112,r121,r122 = paths = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Choose a separate Stage-124 directory")
    if not (r121 / "manifest.json").is_file() or not (r122 / "manifest.json").is_file():
        parser.error("Need Stage-121 and Stage-122 server manifests and Stage-121 NPZ caches")
    with base.s109.s106.s93.writer_lock(root):
        if (root / "analysis/stage124_decision.json").exists():
            check_results(root)
            return
        prior.check_results(r122)
        p122 = c.c66.sealed(r122 / "manifest.json", prior.SCHEMA)
        d122 = c.c66.sealed(r122 / "analysis/stage122_decision.json", prior.SCHEMA)
        p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
        p112 = c.c66.sealed(r112 / "manifest.json", common.SCHEMA)
        if (d122["fingerprint"]!=previous.SOURCE122 or p122["source_plan_fingerprint"]!=p121["fingerprint"]
                or p121["reuse_controls"] or len(p112["folds"])!=4):
            raise ValueError("Need the completed frozen Stage-121/122 L40 sources")
        if p122["inputs"][str(r112 / "manifest.json")]!=c.file_hash(r112 / "manifest.json"):
            raise ValueError("Provider fold manifest changed")
        provider_folds = {f["fold"]:f for f in p112["folds"]}
        for fold, split in provider_folds.items():
            path = r112 / "models" / f"generator_f{fold}.json"
            generator = c.c66.sealed(path, common.SCHEMA)
            if generator["split"]!=split or set(split["held_out_groups"]) & set(
                    generator["generator"]["fit_groups"]+generator["generator"]["calibration_groups"]):
                raise ValueError("Upstream provider fold leaks the held-out pools")
            for seed in common.DESIGN["seeds"]:
                model = c.c66.sealed(base.s112.model_path(r112,fold,"generated_4","shared",seed), common.SCHEMA)
                if model["split"]!=split or model["seed"]!=seed:
                    raise ValueError("Upstream scoring split changed")
        entry_for, length = previous.read_entries(r76, r89, r93)
        rows = load_rows(r121,r122,p121,entry_for,length)
        counts = {k:sum(r[k] for group in rows.values() for r in group) for k in POOLS}
        states = sum(map(len, rows.values()))
        if (states,counts["uniform"],counts["alternate_activation"],counts["matched_activation"],
                counts["new_good_added"],counts["native_recovery"]) != (7561,7168,358,80,9,52):
            raise ValueError("Frozen archive coverage changed")
        source_bindings = dict(p122["inputs"])
        bind = (r122 / "manifest.json", r122 / "analysis/stage122_decision.json", Path(__file__).resolve(),
                Path(__file__).with_name("run_stage124_candidate_independence.sh").resolve())
        source_bindings.update({str(p):c.file_hash(p) for p in bind})
        source_bindings.update({str(r122 / "analysis" / k):v for k,v in d122["table_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage122_fingerprint=previous.SOURCE122,
                            states=states, population_counts=counts, inputs=source_bindings))
        c.publish_identical(root / "manifest.json", plan)
        run(root, plan, rows, provider_folds)


if __name__ == "__main__":
    main()
