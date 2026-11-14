"""Cross-fit a deployable-input commit gate on sealed matched continuation evidence.

CPU diagnostic only: decisions on old full traces are exposure, not new trajectories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage105_commit_branches as s105
from stage27_uav_common import source_group, tracking_iou


c, io, s103, s93, e63 = s105.c, s105.io, s105.s103, s105.s93, s105.e63
SCHEMA = "stage106_cross_fitted_commit_gate_v1"
STATUS = "COMPLETE_CROSS_FITTED_COMMIT_GATE_DIAGNOSTIC"
SOURCE105 = "a01a169a4cb8d0e9211f2252cddfe7ff739e9a1758ed8f53789ecdba91c57363"
HEADS = ("future", "myopic")
POLICIES = ("keep", "always_pair", "myopic", "future", "rate_expected", "future_oracle")
DESIGN = dict(ridge=1., threshold=0., minimum_scale=1e-8, split="leave_one_source_group_out",
              weighting="group_then_seed_then_source_arm_then_anchor", target="both_pair_minus_keep",
              horizon=32, feature_version=1)
FEATURES = (
    "pair_minus_four_dx", "pair_minus_four_dy", "pair_over_four_log_w", "pair_over_four_log_h",
    "four_move_dx", "four_move_dy", "pair_move_dx", "pair_move_dy", "past_velocity_dx", "past_velocity_dy",
    "four_extrapolation_error", "pair_extrapolation_error", "four_vs_median_log_w", "four_vs_median_log_h",
    "pair_vs_median_log_w", "pair_vs_median_log_h", "four_raw_peak", "pair_raw_peak", "raw_peak_delta",
    "four_hann_peak", "pair_hann_peak", "hann_peak_delta", "four_anchor_score", "pair_anchor_score",
    "anchor_score_delta", "four_reference_score", "pair_reference_score", "reference_score_delta",
    "reference_delta_min", "reference_delta_max", "reference_delta_std", "refs_prefer_pair_fraction",
    "reference_count_fraction", "four_uses_history", "pair_uses_history", "four_memory_write", "pair_memory_write",
    "four_ppo_probability", "pair_ppo_probability", "ppo_probability_delta", "ppo_equals_four", "ppo_equals_pair",
    "ppo_entropy", "ppo_value", "candidate_center_spread", "candidate_log_size_spread", "mode_pairwise",
    "four_velocity_cosine", "pair_velocity_cosine")
TABLES = ("samples", "fit_support", "matched_summary", "paired", "sample_decisions", "exposure_units", "exposure_summary")


def online_features(row, queries, anchor, history, memory, frame, mode, actual, pair, design):
    """Explicit online whitelist. No entry/GT/role/sequence/seed/future argument."""
    if any(f >= frame for f, _ in memory):
        raise ValueError("Current or future memory is not an online feature")
    if mode not in s103.MODES[1:]:
        raise ValueError("Unknown source verifier")
    a, b = int(actual["selected"]), int(pair["selected"])
    boxes, past = np.asarray(row["candidate_boxes"], float), np.asarray(history, float)
    q, anchor = np.asarray(queries, float), np.asarray(anchor, float)
    zero = np.zeros(len(FEATURES))
    if (boxes.shape != (4, 4) or past.ndim != 2 or past.shape[1] != 4 or len(past) == 0
            or q.shape != (4, len(anchor)) or not 0 <= a < 4 or not 0 <= b < 4
            or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any()
            or not np.isfinite(past).all() or (past[:, 2:] <= 0).any()
            or not np.isfinite(q[[a, b]]).all() or not np.isfinite(anchor).all()):
        return zero, False, "invalid_input"
    geometry = not np.array_equal(boxes[a], boxes[b])
    writes = (bool(actual["memory_write"]), bool(pair["memory_write"]))
    changed = geometry or writes[0] != writes[1] or (all(writes) and not np.array_equal(q[a], q[b]))
    center = lambda v: v[..., :2]+v[..., 2:]/2
    latest, ca, cb = center(past[-1]), center(boxes[a]), center(boxes[b])
    scale = np.sqrt(np.prod(past[-1, 2:]))
    velocity = (latest-center(past[-2]))/scale if len(past) >= 2 else np.zeros(2)
    ma, mb = (ca-latest)/scale, (cb-latest)/scale
    median_size = np.median(past[:, 2:], axis=0)
    values = list((cb-ca)/scale)+list(np.log(boxes[b, 2:]/boxes[a, 2:]))
    values += list(ma)+list(mb)+list(velocity)+[np.linalg.norm(ma-velocity), np.linalg.norm(mb-velocity)]
    values += list(np.log(boxes[a, 2:]/median_size))+list(np.log(boxes[b, 2:]/median_size))
    for name in ("raw_peak", "hann_peak"):
        column = e63.source_common.prior.FEATURES.index(name)
        peaks = row["features"][[a, b], column]
        values += [peaks[0], peaks[1], peaks[1]-peaks[0]]
    scores = q[[a, b]] @ anchor
    values += [scores[0], scores[1], scores[1]-scores[0]]
    refs = [v for _, v in s93.select_memory(memory, frame, design) if np.isfinite(v).all()]
    reference_scores = np.asarray(refs) @ q[[a, b]].T if refs else scores[None]
    avg = reference_scores.mean(0); delta = reference_scores[:, 1]-reference_scores[:, 0]
    values += [avg[0], avg[1], avg[1]-avg[0], delta.min(), delta.max(), delta.std(), (delta > 0).mean(),
               len(refs)/design["max_refs"], actual["used_history"], pair["used_history"], *writes]
    prob = np.asarray(row["probabilities"], float)
    if prob.shape != (4,) or not np.isfinite(prob).all() or (prob < 0).any() or abs(prob.sum()-1) > 1e-6:
        return zero, False, "invalid_probability"
    entropy = -np.sum(prob[prob > 0]*np.log(prob[prob > 0]))/np.log(4)
    values += [prob[a], prob[b], prob[b]-prob[a], row["selected"] == a, row["selected"] == b, entropy, row["value"],
               np.linalg.norm(center(boxes)-center(boxes).mean(0), axis=1).mean()/scale,
               np.linalg.norm(np.log(boxes[:, 2:])-np.log(boxes[:, 2:]).mean(0), axis=1).mean(),
               mode == "four_pairwise"]
    for move in (ma, mb):
        denom = np.linalg.norm(move)*np.linalg.norm(velocity)
        values.append(float(move @ velocity/denom) if denom > 0 else 0.)
    x = np.asarray(values, float)
    if x.shape != (len(FEATURES),):
        raise ValueError("Feature schema changed")
    if not np.isfinite(x).all():
        return zero, False, "nonfinite_feature"
    return x, bool(changed), "eligible" if changed else "no_commit_change"


def decision_features(observation, state, anchor, frame, mode, choose, design):
    row = choose({k: observation[k].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, state["history"])
    row["history_latest"] = state["history"][-1]
    q = observation["all_queries"]
    actual = s103.decide(row, q, anchor, list(state["memory"]), frame, mode, design)
    pair = s103.decide(row, q, anchor, list(state["memory"]), frame, "pair", design)
    x, eligible, reason = online_features(row, q, anchor, state["history"], state["memory"], frame, mode, actual, pair, design)
    return x, eligible, reason, actual, pair


def identity(spec, mode, frame):
    unit = spec["unit"]
    return dict(cohort=spec["cohort"], sequence=unit["sequence"], group=source_group(unit["sequence"]),
                seed=unit["seed"], source_arm=mode, frame=int(frame))


def balanced_weights(rows):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, r in enumerate(rows):
        tree[r["group"]][r["seed"]][r["source_arm"]].append(i)
    weights = np.zeros(len(rows))
    for seeds in tree.values():
        for modes in seeds.values():
            for indices in modes.values():
                weights[indices] = 1/(len(tree)*len(seeds)*len(modes)*len(indices))
    return weights


def fit_gate(samples, held_out, head):
    if head not in HEADS:
        raise ValueError("Unknown gate target")
    fit = [r for r in samples if r["group"] != held_out and r["eligible"] and
           all(np.isfinite(r[h]) for h in HEADS)]
    # The fitting receipt deliberately excludes every held-out label/feature.
    inputs = [dict(group=r["group"], sequence=r["sequence"], seed=r["seed"], source_arm=r["source_arm"],
                   frame=r["frame"], x=r["x"].tolist(), y=float(r[head])) for r in fit]
    receipt = io.seal(dict(design=DESIGN, head=head, held_out_group=held_out, features=list(FEATURES), fit_rows=inputs))["fingerprint"]
    model = dict(schema=SCHEMA, kind="ridge_gate", head=head, held_out_group=held_out,
                 features=list(FEATURES), design=DESIGN, input_fingerprint=receipt, fit_rows=len(fit),
                 fit_groups=sorted({r["group"] for r in fit}), fit_contexts=len({(r["group"], r["seed"], r["source_arm"]) for r in fit}),
                 supported=bool(fit), center=[0.]*len(FEATURES), scale=[1.]*len(FEATURES),
                 minimum=[0.]*len(FEATURES), maximum=[0.]*len(FEATURES), beta=[0.]*len(FEATURES), intercept=0.)
    if fit:
        x, y, w = np.asarray([r["x"] for r in fit]), np.asarray([r[head] for r in fit]), balanced_weights(fit)
        if x.shape != (len(fit), len(FEATURES)) or not np.isfinite(x).all():
            raise ValueError("Invalid eligible fitting features")
        mean = w @ x; scale = np.sqrt(w @ ((x-mean)**2))
        scale[scale < DESIGN["minimum_scale"]] = 1.
        z = (x-mean)/scale; intercept = float(w @ y)
        beta = np.linalg.solve(z.T @ (w[:, None]*z)+DESIGN["ridge"]*np.eye(len(FEATURES)), z.T @ (w*(y-intercept)))
        model.update(center=mean.tolist(), scale=scale.tolist(), minimum=x.min(0).tolist(), maximum=x.max(0).tolist(),
                     beta=beta.tolist(), intercept=intercept)
    return io.seal(model)


def predict(model, x, eligible):
    x = np.asarray(x, float)
    if not eligible or not model["supported"] or x.shape != (len(FEATURES),) or not np.isfinite(x).all():
        return dict(score=0., commit=False, outside_fit_range=False, max_abs_z=0.)
    z = (x-np.asarray(model["center"]))/model["scale"]
    score = float(z @ model["beta"]+model["intercept"])
    return dict(score=score, commit=bool(score > DESIGN["threshold"]),
                outside_fit_range=bool(np.any(x < model["minimum"]) or np.any(x > model["maximum"])),
                max_abs_z=float(np.abs(z).max()))


def matched_decisions(samples, models):
    bins = defaultdict(list)
    for r in samples:
        bins[r["group"], r["sequence"], r["seed"], r["source_arm"]].append(r)
    out = []
    for rows in bins.values():
        future = [predict(models[r["group"], "future"], r["x"], r["eligible"]) for r in rows]
        denominator = sum(r["eligible"] for r in rows)
        rate = sum(d["commit"] for d in future)/denominator if denominator else 0.
        for r, decision in zip(rows, future):
            myopic = predict(models[r["group"], "myopic"], r["x"], r["eligible"])
            valid = bool(np.isfinite(r["future"]))
            choices = (0., float(r["eligible"]), float(myopic["commit"]), float(decision["commit"]),
                       rate*float(r["eligible"]), float(valid and r["eligible"] and r["future"] > 0))
            for policy, p in zip(POLICIES, choices):
                delta = float(r["future"]) if valid else None
                out.append({k: r[k] for k in ("cohort", "sequence", "group", "seed", "source_arm", "frame")} |
                    dict(policy=policy, eligible=r["eligible"], commit_probability=p, future_score=decision["score"],
                         myopic_score=myopic["score"], measured_delta=delta,
                         utility=None if delta is None else p*delta,
                         harmful_commit=None if delta is None else p*float(delta < 0),
                         missed_positive_utility=None if delta is None else (1-p)*max(delta, 0.),
                         oracle_regret=None if delta is None else float(r["eligible"])*max(delta, 0.)-p*delta))
    return out


def mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
    return float(np.mean(v)) if v else None


def macro(rows, metric):
    # Anchors/frames -> seed and mode contexts -> seed -> source group, no frame pooling.
    bins = defaultdict(list)
    for r in rows:
        bins[r["group"], r["seed"], r["source_arm"]].append(r)
    seeds = defaultdict(list)
    for (group, seed, _), values in bins.items():
        value = mean(values, metric)
        if value is not None:
            seeds[group, seed].append(value)
    groups = defaultdict(list)
    for (group, _), values in seeds.items():
        groups[group].append(float(np.mean(values)))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def summarize(rows, policies, metrics):
    summary, paired = [], []
    for cohort in ("all", "historical", "expanded"):
        for mode in ("all", *s103.MODES[1:]):
            local = [r for r in rows if (cohort == "all" or r["cohort"] == cohort) and (mode == "all" or r["source_arm"] == mode)]
            if not local:
                continue
            for policy in policies:
                part = [r for r in local if r["policy"] == policy]
                result = dict(cohort=cohort, source_arm=mode, policy=policy, states=sum(r.get("states", 1) for r in part), groups=len({r["group"] for r in part}))
                for metric in metrics:
                    values = macro(part, metric)
                    result[metric] = float(np.mean(list(values.values()))) if values else None
                    result[metric+"_supported_groups"] = len(values)
                summary.append(result)
            if "utility" not in metrics:
                continue
            target = macro([r for r in local if r["policy"] == "future"], "utility")
            for control in ("keep", "always_pair", "myopic", "rate_expected"):
                baseline = macro([r for r in local if r["policy"] == control], "utility")
                delta = np.asarray([target[g]-baseline[g] for g in sorted(target.keys() & baseline.keys())])
                paired.append(dict(cohort=cohort, source_arm=mode, contrast="future_minus_"+control, groups=len(delta),
                    mean_delta=float(delta.mean()) if len(delta) else None, median_delta=float(np.median(delta)) if len(delta) else None,
                    wins=int((delta > 1e-12).sum()), ties=int((np.abs(delta) <= 1e-12).sum()), losses=int((delta < -1e-12).sum()),
                    worst_group_delta=float(delta.min()) if len(delta) else None,
                    minimum_leave_one_group_out=float(min((delta.sum()-v)/(len(delta)-1) for v in delta)) if len(delta) > 1 else None))
    return summary, paired


def required(path, meta):
    data = c.read_cache(path, meta)
    if data is None:
        raise ValueError("Missing sealed evidence: "+str(path))
    return data


def collect_samples(root105, root103, plan105, plan103, load_unit, choose_for, old):
    samples, evidence = [], {}
    for job in plan105["units"]:
        spec, mode = job["spec"], job["source_arm"]
        unit, entry, choose = spec["unit"], load_unit(spec), choose_for(spec["unit"])
        trace = s105.prior.required_trace(root103, plan103, unit, mode)
        for case in job["anchors"]:
            first, last = case["frame"], case["last"]
            stem = f"{c.key(unit)}__{mode}__f{first:06d}"
            meta = dict(plan_fingerprint=plan105["fingerprint"], unit=unit, source_arm=mode, first=first, last=last)
            point_path = root105/"points"/(stem+".npz")
            shared = required(point_path, dict(meta, kind="point"))
            s105.audit_point(shared, trace, entry, first, mode, old, choose, plan103["design"])
            state = s105.initial_state(trace, entry, first, old, plan103["design"])
            observation = {k: shared[k][0] for k in s105.POINT_FIELDS}
            x, eligible, reason, actual, pair = decision_features(observation, state, shared["anchor"], first, mode, choose, plan103["design"])
            values = {}
            for treatment in ("keep", "both_pair"):
                path = root105/"branches"/(stem+"__"+treatment+".npz")
                branch = required(path, dict(meta, kind="branch", treatment=treatment))
                s105.audit(branch, trace, shared, entry, first, last, treatment, mode, old, choose, plan103["design"])
                values[treatment] = s105.metrics(branch, entry, plan103["design"])
                evidence[str(path)] = c.file_hash(path)
            v0, v1 = values["keep"], values["both_pair"]
            np.testing.assert_equal(v0["current_iou"], v1["current_iou"])
            future = np.nan if any(v["future_mean_iou"] is None for v in (v0, v1)) else v1["future_mean_iou"]-v0["future_mean_iou"]
            myopic = np.nan if v0["current_iou"] is None else v0["current_pair_iou"]-v0["current_iou"]
            samples.append(identity(spec, mode, first) | dict(last=last, roles=s105.prior.encoded(sorted({r["role"] for r in case["roles"]})),
                x=x, eligible=eligible, reason=reason, future=future, myopic=myopic,
                four_selected=actual["selected"], pair_selected=pair["selected"]))
            evidence[str(point_path)] = c.file_hash(point_path)
        print(f"[Stage-106 labels] {c.key(unit)} {mode}: {len(job['anchors'])} matched states", flush=True)
    keys = [(r["sequence"], r["seed"], r["source_arm"], r["frame"]) for r in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate matched anchor")
    return samples, evidence


def fit_models(root, plan, samples):
    models, rows, hashes = {}, [], {}
    for group in plan["groups"]:
        for head in HEADS:
            computed = fit_gate(samples, group, head)
            path = root/"models"/(group+"__"+head+".json")
            if path.exists():
                model = c.c66.sealed(path, SCHEMA)
                for key in ("kind", "head", "held_out_group", "features", "design", "input_fingerprint", "fit_groups", "fit_rows"):
                    if model[key] != computed[key]:
                        raise ValueError("Committed gate fit input changed")
            else:
                path.parent.mkdir(exist_ok=True)
                c.publish_identical(path, computed); model = computed
            if group in model["fit_groups"]:
                raise ValueError("Held-out source group entered gate fitting")
            models[group, head] = model; hashes[path.relative_to(root).as_posix()] = c.file_hash(path)
            held = [r for r in samples if r["group"] == group]
            rows.append(dict(held_out_group=group, head=head, fit_groups=len(model["fit_groups"]), fit_rows=model["fit_rows"],
                fit_contexts=model["fit_contexts"], held_out_rows=len(held), held_out_eligible=sum(r["eligible"] for r in held),
                fitted=model["supported"], group_leakage=False, input_fingerprint=model["input_fingerprint"],
                weighted_target_mean=model["intercept"], coefficient_l2=float(np.linalg.norm(model["beta"])),
                model_fingerprint=model["fingerprint"], fit_group_names=s105.prior.encoded(model["fit_groups"])))
    return models, rows, hashes


def exposure(trace, entry, spec, mode, choose, old, design, models):
    """Replay only the ORIGINAL source history. Gate decisions never alter it."""
    s103.audit(trace, entry, old, choose, mode, design)
    history, memory = [np.asarray(entry["initial_box"], float)], []
    arrays = defaultdict(list)
    group = source_group(spec["unit"]["sequence"])
    for i, box in enumerate(trace["boxes"]):
        observation = {k: trace[k][i] for k in ("features", "candidate_boxes", "candidate_parents", "all_queries")}
        state = dict(history=history, memory=memory)
        x, eligible, reason, actual, pair = decision_features(observation, state, trace["anchor"], i+1, mode, choose, design)
        arrays["x"].append(x); arrays["eligible"].append(eligible); arrays["reason"].append(reason)
        for head in HEADS:
            prediction = predict(models[group, head], x, eligible)
            for key, value in prediction.items():
                arrays[head+"_"+key].append(value)
        # Current truth is attached only after both predictions; never used to replay memory.
        truth = np.asarray(entry["ground_truth"][i+1])
        current = tracking_iou(trace["candidate_boxes"][i], truth)
        arrays["offline_current_commit_delta"].append(current[pair["selected"]]-current[actual["selected"]])
        if actual["memory_write"]:
            memory.append((i+1, trace["all_queries"][i, actual["selected"]].copy()))
        memory[:] = [(f, q) for f, q in memory if i+1-f < design["lookback"]]
        history = (history+[box])[-old["actor_design"]["history"]:]
    return {k: np.asarray(v) for k, v in arrays.items()} | dict(frame_id=trace["frame_id"].copy())


def run(root, root105, root103, plan105, plan103, plan, load_unit, choose_for, old):
    samples, source_evidence = collect_samples(root105, root103, plan105, plan103, load_unit, choose_for, old)
    if len(samples) != plan["anchors"] or sorted({r["group"] for r in samples}) != plan["groups"]:
        raise ValueError("Matched source coverage changed")
    models, fits, evidence = fit_models(root, plan, samples)
    decisions = matched_decisions(samples, models)
    summary, paired = summarize(decisions, POLICIES, ("eligible", "commit_probability", "utility", "harmful_commit", "missed_positive_utility", "oracle_regret"))
    units, states = [], 0
    for spec in plan103["units"]:
        unit, entry, choose = spec["unit"], load_unit(spec), choose_for(spec["unit"])
        for mode in s103.MODES[1:]:
            trace = s105.prior.required_trace(root103, plan103, unit, mode)
            path = root/"exposure"/(c.key(unit)+"__"+mode+".npz")
            meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, source_arm=mode, kind="frozen_source_exposure")
            data = c.read_cache(path, meta)
            if data is None:
                data = exposure(trace, entry, spec, mode, choose, old, plan103["design"], models)
                c.save(path, data, meta)
            if len(data["frame_id"]) != len(trace["frame_id"]) or data["x"].shape != (len(trace["frame_id"]), len(FEATURES)):
                raise ValueError("Incomplete full-source exposure")
            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
            states += len(data["frame_id"])
            for head in HEADS:
                eligible = data["eligible"].astype(bool)
                units.append(identity(spec, mode, 0) | dict(policy=head, states=len(eligible), eligible=float(eligible.mean()),
                    commit_probability=float(data[head+"_commit"].mean()),
                    outside_fit_range=float(data[head+"_outside_fit_range"][eligible].mean()) if eligible.any() else None,
                    mean_max_abs_z=float(data[head+"_max_abs_z"][eligible].mean()) if eligible.any() else None,
                    p95_max_abs_z=float(np.quantile(data[head+"_max_abs_z"][eligible], .95)) if eligible.any() else None))
            print(f"[Stage-106 exposure] {c.key(unit)} {mode}: {len(data['frame_id'])} cached states", flush=True)
    if states != plan["exposure_states"] or len(units) != len(plan103["units"])*4:
        raise ValueError("Exposure budget changed")
    exposure_summary, _ = summarize(units, HEADS, ("eligible", "commit_probability", "outside_fit_range", "mean_max_abs_z", "p95_max_abs_z"))
    sample_rows = [{k: (float(v) if np.isfinite(v) else None) if k in HEADS else v for k, v in r.items() if k != "x"} for r in samples]
    sample_path = root/"samples.npz"
    sample_data = dict(x=np.asarray([r["x"] for r in samples]), eligible=np.asarray([r["eligible"] for r in samples]),
                       future=np.asarray([r["future"] for r in samples]), myopic=np.asarray([r["myopic"] for r in samples]))
    metadata = dict(plan_fingerprint=plan["fingerprint"], kind="matched_samples")
    cached = c.read_cache(sample_path, metadata)
    if cached is None:
        c.save(sample_path, sample_data, metadata)
    else:
        for key in sample_data:
            np.testing.assert_array_equal(cached[key], sample_data[key])
    evidence[sample_path.name] = c.file_hash(sample_path)
    tables = dict(zip(TABLES, (sample_rows, fits, summary, paired, decisions, units, exposure_summary)))
    out = root/"analysis"; out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = out/f"stage106_{name}.csv"
        s103.s86.publish_table(path, rows); hashes[path.name] = c.file_hash(path)
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    c.publish_identical(out/"stage106_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_cross_fitted_commit_gate_learnability", plan_fingerprint=plan["fingerprint"],
        source_stage105_fingerprint=SOURCE105, counts={n: len(v) for n, v in tables.items()},
        new_gate_models=len(models), supported_gate_models=sum(m["supported"] for m in models.values()),
        exposure_states=states, sgla_training_updates=0, ppo_training_updates=0, new_image_reads=0,
        new_search_calls=0, new_full_trajectories=0, confirmation_inference_calls=0,
        held_out_gate_labels_excluded=True, fit_only_normalization=True, frozen_source_exposure_only=True,
        table_sha256=hashes, artifact_sha256=evidence, label_evidence_sha256=source_evidence,
        warnings=["GT-selected development anchors, overlapping windows and repeatedly consumed groups; no external confirmation.",
                  "Only the new gate labels and normalizer are held out by source group; original SGLA/PPO fits are fixed overlapping sources.",
                  "Utility is a one-time both_pair commit followed by the original controller, not repeated gated control.",
                  "Full-cache decisions use the old source trajectory; no new full-policy AUC or retention pass.",
                  "Rate expected and future oracle are offline diagnostics, not deployable controllers."])))
    check_results(root)
    print(f"Stage-106: {STATUS}; {len(models)} CPU gates, {len(samples)} anchors; no new GPU inference", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"analysis/stage106_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or plan["design"] != DESIGN or plan["features"] != list(FEATURES)
            or report["source_stage105_fingerprint"] != SOURCE105 or report["counts"]["samples"] != plan["anchors"]
            or report["new_gate_models"] != 2*len(plan["groups"]) or report["exposure_states"] != plan["exposure_states"]
            or report["counts"]["fit_support"] != report["new_gate_models"]
            or report["counts"]["sample_decisions"] != plan["anchors"]*len(POLICIES)
            or report["counts"]["exposure_units"] != 2*plan["exposure_contexts"]
            or len(report["artifact_sha256"]) != report["new_gate_models"]+plan["exposure_contexts"]+1
            or len(report["label_evidence_sha256"]) != 3*plan["anchors"]
            or set(report["table_sha256"]) != {f"stage106_{n}.csv" for n in TABLES}
            or not all(report[k] for k in ("held_out_gate_labels_excluded", "fit_only_normalization", "frozen_source_exposure_only"))
            or any(report[k] != 0 for k in ("sgla_training_updates", "ppo_training_updates", "new_image_reads", "new_search_calls", "new_full_trajectories", "confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-106 result")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n): h for n, h in report["table_sha256"].items()})
    io.verify({str(root/n): h for n, h in report["artifact_sha256"].items()})
    io.verify(report["label_evidence_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage103", "stage105", "output_dir"):
        parser.add_argument("--"+name, type=Path, required=name == "output_dir")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root); return
    names = ("stage76", "stage89", "stage93", "stage103", "stage105")
    if not all(getattr(args, n) for n in names):
        parser.error("Supply Stage-76/89/93/103/105 sources")
    root76, root89, root93, root103, root105 = sources = [getattr(args, n).expanduser().resolve() for n in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-106 directory")
    with s93.writer_lock(root):
        if (root/"analysis/stage106_decision.json").exists():
            check_results(root); return
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-106 directory")
        print("Stage-106: validating frozen CPU sources; no images or GPU runtime", flush=True)
        s105.check_results(root105); s103.check_results(root103)
        plan105 = c.c66.sealed(root105/"manifest.json", s105.SCHEMA)
        report105 = c.c66.sealed(root105/"analysis/stage105_decision.json", s105.SCHEMA)
        plan103 = c.c66.sealed(root103/"manifest.json", s103.SCHEMA)
        report103 = c.c66.sealed(root103/"analysis/stage103_decision.json", s103.SCHEMA)
        if report105["fingerprint"] != SOURCE105 or report103["fingerprint"] != s105.prior.SOURCE103:
            raise ValueError("Need frozen Stage-103/105 results")
        plan93, _, plan89, report89, source = s103.prior.s95.s94.load_run(root93, root76, root89)
        if plan103["units"] != plan89["units"]:
            raise ValueError("Development roster changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch
        torch.set_num_threads(1)
        def load_unit(spec):
            return s103.prior.s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)[0]
        def choose_for(unit):
            pack = s103.e74.learned_pack(actor["models"], unit, actor["prior"])
            return lambda row, history: s103.e74.choose(row, history, unit, source["original"], actor["prior"], pack)
        groups = sorted({source_group(s["unit"]["sequence"]) for s in plan103["units"]})
        expected = s105.budget(plan105["units"])
        if len(groups) != 28 or len(plan105["units"]) != 80 or expected["anchors"] != 259 or plan105["design"] != s105.DESIGN:
            raise ValueError("Frozen anchor coverage/design changed")
        count = sum(len(s105.prior.required_trace(root103, plan103, spec["unit"], mode)["frame_id"])
                    for spec in plan103["units"] for mode in s103.MODES[1:])
        bindings = plan105["source_sha256"] | actor["bindings"]
        for path in (root105/"manifest.json", root105/"analysis/stage105_decision.json", root103/"manifest.json", root103/"analysis/stage103_decision.json"):
            bindings[str(path)] = c.file_hash(path)
        bindings.update({str(root105/p): h for p, h in report105["cache_sha256"].items()})
        bindings.update({str(root103/p): h for p, h in report103["cache_sha256"].items()})
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage106_commit_gate.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, features=list(FEATURES), groups=groups, anchors=expected["anchors"],
            exposure_contexts=80, exposure_states=count, source_stage105_fingerprint=SOURCE105,
            source_sha256=bindings, code_sha256=plan105["code_sha256"] | {str(p): c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json", plan)
        run(root, root105, root103, plan105, plan103, plan, load_unit, choose_for, source["original"])


if __name__ == "__main__":
    main()
