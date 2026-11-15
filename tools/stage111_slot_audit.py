"""Audit frozen PPO slot dependence using only the complete Stage-110 caches."""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import stage110_search_proposals as s110

p = s110.p
c, io, s103, s105, e63, s109 = p.c, p.io, p.s103, p.s105, p.e63, p.s109
c74 = s103.e74.c
SCHEMA = "stage111_cached_ppo_slot_permutation_v1"
STATUS = "COMPLETE_CACHED_PPO_SLOT_PERMUTATION_DIAGNOSTIC"
SOURCE110 = "dd804631d678a6e112c45af5df33d3ebbbf88bd3113dfdd7af0be7f7fb31b254"
ORDERS = np.asarray([(0, *tail) for tail in itertools.permutations((1, 2, 3))], int)
POLICIES = ("identity", "canonical_geometry", "probability_symmetrized", "uniform_order_expected",
            "q0", "hann", "best_order_offline", "worst_order_offline", "candidate_oracle")
METRICS = ("iou", "good_selection", "missed_good", "harmful_q0_replacement", "oracle_regret")
SENSITIVITY = ("physical_order_sensitive", "order_iou_range", "mean_clip_fraction", "identity_clip_fraction",
               "best_iou", "q0_iou", "good_available")
TABLES = ("scope", "state_summary", "policy_summary", "paired", "permutation_summary", "sensitivity_summary")
CONTRASTS = (("probability_symmetrized", "identity"), ("canonical_geometry", "identity"),
             ("uniform_order_expected", "identity"), ("best_order_offline", "identity"),
             ("probability_symmetrized", "q0"), ("probability_symmetrized", "hann"))
DESIGN = dict(orders=ORDERS.tolist(), q0_fixed=True, tie_atol=1e-12, good_iou=.5, bad_iou=.2,
              primary_bank="learned_calibrated", primary="probability_symmetrized_minus_identity",
              primary_metric="iou", policies=list(POLICIES), candidate_banks=list(p.ARMS),
              canonical_cost="squared_center_in_latest_target_sides_plus_squared_log_width_height",
              duplicate_rule="sum_mapped_probability_by_exact_output_box", history_source="unchanged_stage103")


def physical_choice(scores, boxes):
    """GT-free score tie rule: prefer q0, then lexicographic physical box."""
    scores, boxes = np.asarray(scores, float), np.asarray(boxes, float)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError("Need four finite physical scores")
    tied = np.flatnonzero(scores >= scores.max()-DESIGN["tie_atol"])
    if 0 in tied:
        return 0
    return int(min(tied, key=lambda j: tuple(boxes[j])))


def canonical_order(parents, boxes, reference, latest):
    """Assign three non-q0 inputs to old fixed-search roles without labels."""
    parents, reference, latest = np.asarray(parents, float), np.asarray(reference, float), np.asarray(latest, float)
    side = np.sqrt(latest[2:].prod())
    centers = parents[:, :2]+parents[:, 2:]/2
    ref_centers = reference[:, :2]+reference[:, 2:]/2
    costs = []
    for order in ORDERS:
        cost = float(np.square((centers[order[1:]]-ref_centers[1:])/side).sum()
                     + np.square(np.log(parents[order[1:], 2:]/reference[1:, 2:])).sum())
        # Physical tie key avoids introducing a new dependence on tail labels.
        key = tuple(np.r_[parents[j], boxes[j]] for j in order[1:])
        costs.append((cost, tuple(tuple(v) for v in key)))
    minimum = min(v[0] for v in costs)
    return int(min((i for i, v in enumerate(costs) if v[0] <= minimum+DESIGN["tie_atol"]), key=lambda i: costs[i][1]))


def symmetrize(probabilities, boxes):
    """Probabilities already refer to original physical candidate indices."""
    mean = np.asarray(probabilities, float).mean(0)
    grouped = np.zeros(4)
    for j, box in enumerate(boxes):
        first = next(k for k in range(j+1) if np.array_equal(box, boxes[k]))
        grouped[first] += mean[j]
    np.testing.assert_allclose(grouped.sum(), 1., rtol=0, atol=1e-12)
    # Non-representatives have zero mass and cannot win a positive softmax sum.
    return grouped, physical_choice(grouped, boxes)


def predict(data, history, pack, actor_design):
    """Only public cached observations and past predictions enter this function."""
    parents = np.asarray(data["candidate_parents"], float)
    boxes = np.asarray(data["candidate_boxes"], float)
    base = np.asarray(data["features"], float)
    if parents.shape != (6, 4, 4) or boxes.shape != parents.shape or base.shape != (6, 4, 24):
        raise ValueError("Need six complete Stage-110 candidate banks")
    if not np.isfinite(np.r_[parents.ravel(), boxes.ravel(), base.ravel()]).all() or np.any(parents[..., 2:] <= 0):
        raise ValueError("Invalid cached candidate inputs")
    np.testing.assert_array_equal(boxes[:, 0], np.tile(boxes[0, 0], (6, 1)))
    np.testing.assert_array_equal(parents[:, 0], np.tile(parents[0, 0], (6, 1)))
    probabilities, logits, choices, clips, maxima, canonical, sym_p, sym_a, hann = [], [], [], [], [], [], [], [], []
    column = e63.source_common.prior.FEATURES.index("hann_peak")
    for bank in range(6):
        pp, ll, aa, cc, zz = [], [], [], [], []
        for order in ORDERS:
            features = c74.c66.c64.choice_features(base[bank, order], boxes[bank, order], history)
            decision = c74.gate(features, boxes[bank, order], pack, actor_design)
            mapped = np.zeros(4); mapped[order] = decision["probabilities"]
            z = (features.reshape(-1)-np.asarray(pack["norm"]["center"]))/np.asarray(pack["norm"]["scale"])
            pp.append(mapped); ll.append(decision["logits"]); aa.append(int(order[np.argmax(decision["probabilities"])]))
            cc.append(float(np.mean(np.abs(z) > actor_design["feature_clip"]))); zz.append(float(np.abs(z).max()))
        probabilities.append(pp); logits.append(ll); choices.append(aa); clips.append(cc); maxima.append(zz)
        canonical.append(canonical_order(parents[bank], boxes[bank], parents[0], history[-1]))
        probability, action = symmetrize(pp, boxes[bank]); sym_p.append(probability); sym_a.append(action)
        hann.append(physical_choice(base[bank, :, column], boxes[bank]))
    return dict(probabilities=np.asarray(probabilities), slot_logits=np.asarray(logits), selected=np.asarray(choices, int),
                clip_fraction=np.asarray(clips), maximum_unclipped_z=np.asarray(maxima), canonical_order=np.asarray(canonical, int),
                symmetrized_probabilities=np.asarray(sym_p), symmetrized_selected=np.asarray(sym_a, int), hann_selected=np.asarray(hann, int))


def labels_and_rows(data, prediction, truth, ident):
    """Attach current-frame labels only after immutable predictions have been saved."""
    truth = np.asarray(truth, float)
    valid = bool(truth.shape == (4,) and np.isfinite(truth).all() and np.all(truth[2:] > 0))
    overlaps = s105.tracking_iou(data["candidate_boxes"], truth) if valid else np.full((6, 4), np.nan)
    states, decisions, permutations = [], [], []
    for b, bank in enumerate(p.ARMS):
        a = prediction["selected"][b]; boxes = data["candidate_boxes"][b]
        selected_ious = overlaps[b, a]
        actions = dict(identity=int(a[0]), canonical_geometry=int(a[prediction["canonical_order"][b]]),
                       probability_symmetrized=int(prediction["symmetrized_selected"][b]), q0=0, hann=int(prediction["hann_selected"][b]))
        if valid:
            actions.update(best_order_offline=int(a[np.argmax(selected_ious)]), worst_order_offline=int(a[np.argmin(selected_ious)]),
                           candidate_oracle=int(np.argmax(overlaps[b])))
        best, q0 = (float(overlaps[b].max()), float(overlaps[b, 0])) if valid else (None, None)
        state = dict(ident, bank=bank, valid_gt=valid, best_iou=best, q0_iou=q0,
                     good_available=float(best >= DESIGN["good_iou"]) if valid else None,
                     physical_order_sensitive=float(any(not np.array_equal(boxes[j], boxes[a[0]]) for j in a)),
                     order_iou_range=float(np.ptp(selected_ious)) if valid else None,
                     identity_clip_fraction=float(prediction["clip_fraction"][b, 0]),
                     mean_clip_fraction=float(prediction["clip_fraction"][b].mean()),
                     maximum_unclipped_z=float(prediction["maximum_unclipped_z"][b].max()),
                     order_selected="|".join(map(str, a)), order_ious="|".join(map(str, selected_ious)) if valid else "",
                     canonical_order="|".join(map(str, ORDERS[prediction["canonical_order"][b]])))
        for policy in POLICIES:
            values = selected_ious if policy == "uniform_order_expected" else np.asarray([overlaps[b, actions.get(policy, 0)]])
            row = dict(ident, bank=bank, policy=policy, iou=float(values.mean()) if valid else None,
                       good_selection=float((values >= DESIGN["good_iou"]).mean()) if valid else None,
                       missed_good=float(((best >= DESIGN["good_iou"]) & (values < DESIGN["good_iou"])).mean()) if valid else None,
                       harmful_q0_replacement=float(((q0 >= DESIGN["good_iou"]) & (values < DESIGN["bad_iou"])).mean()) if valid else None,
                       oracle_regret=float(best-values.mean()) if valid else None)
            decisions.append(row)
            state[policy+"_iou"] = row["iou"]; state[policy+"_selected"] = actions.get(policy)
        states.append(state)
        for k, order in enumerate(ORDERS):
            permutations.append(dict(ident, bank=bank, policy="".join(map(str, order)), iou=float(selected_ious[k]) if valid else None,
                                     clip_fraction=float(prediction["clip_fraction"][b, k]),
                                     harmful_q0_replacement=float(q0 >= DESIGN["good_iou"] and selected_ious[k] < DESIGN["bad_iou"]) if valid else None))
    return states, decisions, permutations


def summaries(states, decisions, permutations):
    policies, paired, orders, sensitivity = [], [], [], []
    for bank in p.ARMS:
        local = [r for r in decisions if r["bank"] == bank]
        result, _ = s109.s106.summarize(local, POLICIES, METRICS)
        policies.extend(dict(r, bank=bank) for r in result)
        result, _ = s109.s106.summarize([r for r in permutations if r["bank"] == bank],
            tuple("".join(map(str, o)) for o in ORDERS), ("iou", "clip_fraction", "harmful_q0_replacement"))
        orders.extend(dict(r, bank=bank) for r in result)
        result, _ = s109.s106.summarize([dict(r, policy="all_orders") for r in states if r["bank"] == bank], ("all_orders",), SENSITIVITY)
        sensitivity.extend(dict(r, bank=bank) for r in result)
        for cohort in ("all", "historical", "expanded"):
            for mode in ("all", *s103.MODES[1:]):
                part = [r for r in local if (cohort == "all" or r["cohort"] == cohort) and (mode == "all" or r["source_arm"] == mode)]
                if not part: continue
                for target, control in CONTRASTS:
                    for metric in ("iou", "harmful_q0_replacement", "missed_good"):
                        a = s109.s106.macro([r for r in part if r["policy"] == target], metric)
                        b = s109.s106.macro([r for r in part if r["policy"] == control], metric)
                        groups = sorted(a.keys() & b.keys()); d = np.asarray([a[g]-b[g] for g in groups])
                        paired.append(dict(cohort=cohort, source_arm=mode, bank=bank, contrast=target+"_minus_"+control, metric=metric,
                            groups=len(d), mean_delta=float(d.mean()) if len(d) else None, median_delta=float(np.median(d)) if len(d) else None,
                            wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()), losses=int((d < -1e-12).sum()),
                            minimum_leave_one_group_out=float(min((d.sum()-v)/(len(d)-1) for v in d)) if len(d) > 1 else None))
    return policies, paired, orders, sensitivity


def read_supply(root110):
    with (root110/"analysis/stage110_state_supply.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {(r["sequence"], int(r["seed"]), r["source_arm"], int(r["frame"]), r["policy"]): r for r in rows}
    if len(result) != len(rows): raise ValueError("Duplicate Stage-110 state-supply identity")
    return result


def run(root, root110, root103, p110, p103, plan, load_unit, pack_for, old):
    evidence, states, decisions, permutations, scope = {}, [], [], [], []
    supply = read_supply(root110)
    for job in plan["units"]:
        spec, mode = job["spec"], job["source_arm"]; unit = spec["unit"]
        entry, pack = load_unit(spec), pack_for(unit)
        trace = s105.prior.required_trace(root103, p103, unit, mode)
        group = s109.s106.source_group(unit["sequence"])
        model = c.c66.sealed(root110/"models"/(group+".json"), p.SCHEMA)
        for frame in job["sampling"]["eval_frames"]:
            ident = s109.s106.identity(spec, mode, frame); stem = job["context"]+f"__f{frame:06d}"
            meta110 = dict(plan_fingerprint=p110["fingerprint"], unit=unit, source_arm=mode, frame=frame, model_fingerprint=model["fingerprint"])
            data = s109.s106.required(root110/"observations"/(stem+".npz"), dict(meta110, kind="shared_sgla_searches"))
            proposal = s109.s106.required(root110/"proposals"/(stem+".npz"), dict(meta110, kind="causal_search_proposals"))
            np.testing.assert_array_equal(data["candidate_parents"], proposal["candidate_parents"])
            np.testing.assert_array_equal(data["frame_id"], frame)
            history = e63.history_at(trace, entry["initial_box"], frame, old)
            computed = predict(data, history, pack, old["design"])
            metadata = dict(plan_fingerprint=plan["fingerprint"], unit=unit, source_arm=mode, frame=frame, kind="slot_predictions")
            prediction = s109.cache(root, root/"predictions"/(stem+".npz"), metadata, lambda: computed, evidence)
            if set(prediction) != set(computed): raise ValueError("Prediction schema changed")
            for key in computed:
                if computed[key].dtype.kind in "biu": np.testing.assert_array_equal(prediction[key], computed[key])
                else: np.testing.assert_allclose(prediction[key], computed[key], rtol=1e-9, atol=1e-12)
            # Saved above before accessing evaluation GT or any oracle selections.
            ss, dd, oo = labels_and_rows(data, prediction, entry["ground_truth"][frame], ident)
            maximum_error = 0.
            for b, bank in enumerate(p.ARMS):
                source = supply[(unit["sequence"], unit["seed"], mode, frame, bank)]
                if int(source["shadow_selected"]) != int(prediction["selected"][b, 0]): raise ValueError("Identity PPO action does not reproduce Stage-110")
                for field, actual in (("shadow_ppo_iou", ss[b]["identity_iou"]), ("best_iou", ss[b]["best_iou"]), ("q0_iou", ss[b]["q0_iou"])):
                    if actual is None:
                        if source[field] != "": raise ValueError("GT support changed")
                    else:
                        error = abs(float(source[field])-actual); maximum_error = max(maximum_error, error)
                        if error > 1e-12: raise ValueError("Same candidate/current-IoU parity failed")
            states.extend(ss); decisions.extend(dd); permutations.extend(oo)
            scope.append(dict(ident, banks=6, permutations_per_bank=6, actor_evaluations=36, q0_fixed=True, candidate_set_unchanged=True,
                              predictions_saved_before_labels=True, maximum_source_iou_error=maximum_error))
        print(f"[Stage-111] {job['context']}: {len(job['sampling']['eval_frames'])} states, cached CPU slot audit", flush=True)
    policy, paired, orders, sensitivity = summaries(states, decisions, permutations)
    tables = dict(zip(TABLES, (scope, states, policy, paired, orders, sensitivity)))
    publish(root, plan, tables, evidence)


def expected(plan):
    n = sum(len(j["sampling"]["eval_frames"]) for j in plan["units"])
    settings = {(cohort, mode) for j in plan["units"] if j["sampling"]["eval_frames"]
                for cohort in ("all", j["spec"]["cohort"]) for mode in ("all", j["source_arm"])}
    return dict(scope=n, state_summary=6*n, policy_summary=6*len(POLICIES)*len(settings),
                paired=6*len(CONTRASTS)*3*len(settings), permutation_summary=36*len(settings), sensitivity_summary=6*len(settings))


def publish(root, plan, tables, evidence):
    if set(tables) != set(TABLES) or any(len(tables[k]) != n for k, n in expected(plan).items()) or len(evidence) != 2*expected(plan)["scope"]:
        raise ValueError("Incomplete Stage-111 cached audit")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    out = root/"analysis"; out.mkdir(exist_ok=True); hashes = {}
    for name in TABLES:
        path = out/f"stage111_{name}.csv"; s103.s86.publish_table(path, tables[name]); hashes[path.name] = c.file_hash(path)
    primary = next(r for r in tables["paired"] if r["cohort"] == r["source_arm"] == "all" and r["bank"] == DESIGN["primary_bank"]
                   and r["contrast"] == DESIGN["primary"] and r["metric"] == DESIGN["primary_metric"])
    c.publish_identical(out/"stage111_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        source_stage110_fingerprint=plan["source_stage110_fingerprint"], role="consumed_development_same_candidate_slot_diagnostic",
        counts={k: len(v) for k, v in tables.items()}, primary=primary, new_actor_evaluations=36*len(tables["scope"]),
        new_search_calls=0, image_reads=0, training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        source_parity_pass=True, maximum_source_iou_error=max(r["maximum_source_iou_error"] for r in tables["scope"]),
        table_sha256=hashes, artifact_sha256=evidence, warnings=[
            "All source groups are consumed development; this is a current-frame cache diagnostic, not new full-video AUC.",
            "Six orderings are evaluated per bank with q0 fixed. This extra CPU actor cost is not a speedup claim.",
            "Identity retains source slot normalization and duplicate mask; other orders recompute all causal relational features.",
            "Canonical assignment and probability averaging do not use GT. Best/worst-order and candidate oracle use GT only for offline bounds.",
            "Slot effects do not uniquely explain all loss; changed search geometry, feature support and long-term state remain unresolved.",
            "No method adoption gate, new predictor/PPO fitting, GPU inference, confirmation use or candidate-generation change."])))
    check_results(root)
    print(f"Stage-111: {STATUS}; {len(tables['scope'])} states, {36*len(tables['scope'])} CPU actor evaluations", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA); report = c.c66.sealed(root/"analysis/stage111_decision.json", SCHEMA)
    n = expected(plan)["scope"]
    if (plan["design"] != DESIGN or plan["source_stage110_fingerprint"] != SOURCE110 or report["status"] != STATUS
            or report["plan_fingerprint"] != plan["fingerprint"] or report["source_stage110_fingerprint"] != SOURCE110
            or report["counts"] != expected(plan) or report["new_actor_evaluations"] != 36*n or not report["source_parity_pass"]
            or report["maximum_source_iou_error"] > 1e-12 or len(report["artifact_sha256"]) != 2*n
            or set(report["table_sha256"]) != {f"stage111_{name}.csv" for name in TABLES}
            or any(report[k] != 0 for k in ("new_search_calls", "image_reads", "training_updates", "new_full_trajectories", "confirmation_inference_calls"))):
        raise ValueError("Incomplete or changed Stage-111 result")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/name):h for name, h in report["table_sha256"].items()})
    io.verify({str(root/name):h for name, h in report["artifact_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage103", "stage110")
    for name in (*names, "output_dir"): parser.add_argument("--"+name, type=Path, required=name == "output_dir")
    parser.add_argument("--check-results", action="store_true"); args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results: check_results(root); return
    if not all(getattr(args, name) for name in names): parser.error("Supply Stage-76/89/93/103/110 roots; no GPU/checkpoint needed")
    r76, r89, r93, r103, r110 = sources = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == q or root in q.parents or q in root.parents for q in sources): parser.error("Use a separate Stage-111 directory")
    with s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage111_decision.json").exists(): check_results(root); return
        if not (root/"manifest.json").exists() and any(q.name != "collect.lock" for q in root.iterdir()): raise ValueError("Use an empty Stage-111 directory")
        s110.check_results(r110); s103.check_results(r103)
        plan110 = c.c66.sealed(r110/"manifest.json", p.SCHEMA); report110 = c.c66.sealed(r110/"analysis/stage110_decision.json", p.SCHEMA)
        plan103 = c.c66.sealed(r103/"manifest.json", s103.SCHEMA)
        p93, _, p89, report89, source = s103.prior.s95.s94.load_run(r93, r76, r89)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if (report110["fingerprint"] != SOURCE110 or plan103["units"] != p89["units"] or len(plan103["units"]) != 40
                or len(plan110["groups"]) != 28 or s110.expected(plan110)["scope"] != 1920
                or actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]):
            raise ValueError("Need the complete fixed Stage-110 sources and original frozen PPO")
        bindings = dict(plan110["source_sha256"]) | actor["bindings"]
        for path in (r110/"manifest.json", r110/"analysis/stage110_decision.json", r103/"manifest.json", r103/"analysis/stage103_decision.json"):
            bindings[str(path)] = c.file_hash(path)
        bindings.update({str(r110/name): h for name, h in report110["artifact_sha256"].items()})
        bindings.update({str(r110/"analysis"/name): h for name, h in report110["table_sha256"].items()})
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage111_slot_audit.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, units=plan110["units"], groups=plan110["groups"],
            source_stage110_fingerprint=SOURCE110, source_runtime=plan110["runtime"],
            source_sha256=bindings, code_sha256=plan110["code_sha256"] | {str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json", plan)
        def load_unit(spec): return s103.prior.s95.load_unit(r93, r89, p93, p89, report89, source, spec)[0]
        def pack_for(unit): return s103.e74.learned_pack(actor["models"], unit, actor["prior"])
        run(root, r110, r103, plan110, plan103, plan, load_unit, pack_for, source["original"])


if __name__ == "__main__": main()
