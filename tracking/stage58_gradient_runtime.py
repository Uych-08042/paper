"""Full-cohort endpoint preservation with one-sided gradient projection.

Projection is in parameter-gradient space BEFORE clipping and AdamW. It is not
a guarantee about the optimizer displacement or a later tracking trajectory.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np
from stage48_stability_experiment import load_torch
from stage48_stability_runtime import make_head, digest
from stage52_branch_runtime import configure as branch_configure, frozen_digest
from stage57_quality_runtime import rollout_batch, temporary_head, decode_image_boxes, node_penalty
from stage48_stability_runtime import head_output
from stage18_tracking_loss import tracking_loss
import stage58_gradient_common as c


def configure(head, arm):
    if arm not in c.ARMS: raise ValueError("Unknown gradient arm")
    return branch_configure(head, "geometry_only")


def objective(torch,head,rows,ends,plan,arm):
    if arm not in c.ARMS: raise ValueError("Unknown gradient arm")
    device=next(head.parameters()).device
    # Recompute batch-one heads, matching the native actor's numerical path.
    outputs=[head_output(head,torch.as_tensor(r["features"],device=device)[None]) for r in rows]
    out={k:torch.cat([o[k] for o in outputs],0) for k in outputs[0]}
    response_error=float(np.max(np.abs(out["score_map"].detach().cpu().numpy()[:,0]-np.stack([r["raw"] for r in rows]))))
    if response_error>plan["design"]["frozen_map_atol"]: raise ValueError("Frozen response replay changed")
    target=torch.as_tensor(np.stack([r["target"] for r in rows]),device=device,dtype=out["score_map"].dtype)
    ordinary,stats=tracking_loss(torch,out,target,True)
    decoded=decode_image_boxes(torch,out,rows,plan["search_size"])
    error=float(np.max(np.abs(decoded.detach().cpu().numpy()-np.array([r["boxes"] for r in rows]))))
    if error>plan["design"]["decode_atol"]: raise ValueError("Differentiable endpoint decode differs from the native actor")
    penalty=node_penalty(torch,decoded,ends,plan)
    loss=ordinary+1.*penalty
    return loss,stats,ordinary,penalty,error,response_error


def gradient_vectors(torch, ordinary, penalty, parameters):
    def flat(value):
        grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1) for p, g in zip(parameters, grads)])
    return flat(ordinary), flat(penalty)


def gradient_rule(torch, supervised, anchor, arm, design):
    """Return the applied anchor vector and SAME-state shadow controls."""
    if arm not in c.ARMS or supervised.shape != anchor.shape or supervised.ndim != 1:
        raise ValueError("Invalid gradient vectors/arm")
    if not bool(torch.isfinite(supervised).all() & torch.isfinite(anchor).all()):
        raise ValueError("Non-finite gradient vector")
    s64, a64 = supervised.to(torch.float64), anchor.to(torch.float64)
    s2, a2 = torch.dot(s64, s64), torch.dot(a64, a64)
    dot = torch.dot(s64, a64)
    conflict = bool(dot < 0 and s2 > 0 and a2 > 0)
    projected = anchor - (dot/s2).to(anchor.dtype)*supervised if conflict else anchor.clone()
    # Ratios are detached. Zero vectors are valid empty/coincident-anchor cases.
    p64 = projected.to(torch.float64); pn = torch.linalg.vector_norm(p64)
    sn, an = s2.sqrt(), a2.sqrt()
    ratio = pn/an if bool(an > 0) else an.new_tensor(1.)
    matched = ratio.to(anchor.dtype)*anchor
    applied = {"endpoint_control": anchor, "norm_matched": matched, "projected_anchor": projected}[arm]
    pdot = torch.dot(s64, p64)
    removed = torch.linalg.vector_norm(a64-p64)
    tiny = np.finfo(float).tiny
    violation = max(0., -float(pdot))/max(float(sn*an), tiny)
    norm_error = abs(float(torch.linalg.vector_norm(matched.to(torch.float64))-pn))/max(float(an), tiny)
    if max(violation, norm_error) > design["gradient_audit_rtol"]:
        raise ValueError("Gradient projection/norm matching numerical audit failed")
    modified = arm != "endpoint_control" and bool(torch.any(applied != anchor))
    audit = dict(steps=1, full_gradient_batches=int(an > 0), conflict_batches=int(conflict),
        nontrivial_projection_batches=int(float(removed) > design["numerical_projection_fraction"]*float(an)),
        actual_modified_batches=int(modified), ordinary_norm_sum=float(sn), anchor_norm_sum=float(an),
        projected_norm_sum=float(pn), matched_norm_sum=float(torch.linalg.vector_norm(matched.to(torch.float64))),
        actual_anchor_norm_sum=float(torch.linalg.vector_norm(applied.to(torch.float64))),
        raw_dot_sum=float(dot), projected_dot_sum=float(pdot),
        actual_anchor_dot_sum=float(torch.dot(s64, applied.to(torch.float64))),
        raw_cosine_sum=float(dot)/max(float(sn*an), tiny),
        projected_cosine_sum=float(pdot)/max(float(sn*pn), tiny),
        norm_ratio_sum=float(ratio), removed_norm_sum=float(removed))
    errors = dict(projection_relative_violation=violation, norm_match_relative_error=norm_error)
    return applied.detach(), audit, errors


def assign_gradient(parameters, vector):
    offset = 0
    for p in parameters:
        count = p.numel()
        p.grad = vector[offset:offset+count].reshape_as(p).clone()
        offset += count
    if offset != vector.numel(): raise ValueError("Gradient parameter order changed")


def backward_gradient(loss, parameters, supervised, applied, modified):
    # No-op batches must preserve the ORIGINAL fused backward rounding; even a
    # mathematically equal separately summed gradient can change a later crop.
    loss.backward()
    if modified: assign_gradient(parameters, supervised+applied)


def folder(root, fold, seed, arm, smoke=False):
    return Path(root)/("smoke/training" if smoke else "training")/("seed{}".format(seed))/("fold{}".format(fold))/arm


def save_step(torch, path, state):
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed gradient checkpoint")
    if path.exists() and load_torch(torch, path).get("metadata") != state["metadata"]: raise ValueError("Unidentified orphan gradient state")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as h: temporary = h.name
    try:
        torch.save(state, temporary); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    c.publish_identical(path.with_suffix(".receipt.json"), c.seal(dict(schema=c.SCHEMA, metadata=state["metadata"], step=state["step"], sha256=c.file_hash(path))))


def restore(torch, directory, metadata):
    latest = None
    for receipt in sorted(directory.glob("step*.receipt.json")):
        r = c.read_sealed(receipt); path = directory/("step{:06d}.pt".format(r["step"]))
        if (r.get("metadata") != metadata or not 0 < r["step"] <= metadata["updates"] or
            receipt != path.with_suffix(".receipt.json") or not path.is_file() or c.file_hash(path) != r["sha256"]):
            raise ValueError("Committed gradient checkpoint changed")
        latest = load_torch(torch, path)
        if latest.get("metadata") != metadata or latest.get("step") != r["step"]: raise ValueError("Gradient checkpoint metadata changed")
    return latest


def train_arm(native, plan, root, pool, bindings, fold, seed, arm, smoke=False):
    torch = native.rt.torch; head = make_head(native.rt); _, names = configure(head, arm)
    parameters = [p for p in head.parameters() if p.requires_grad]
    initial = digest(head); frozen = frozen_digest(head, names)
    if initial != c.initial_record(plan, fold, seed)["initial_head_sha256"]: raise ValueError("Original SGLA head initialization changed")
    if dict(fold=fold, pairs=len(pool), fingerprint=c.fingerprint(pool)) != plan["anchor_pools"][fold]: raise ValueError("Fit anchor pool changed")
    training = plan["training"]; updates = training["smoke_updates"] if smoke else training["updates"]
    draws = np.random.default_rng(seed+100*fold).integers(0, len(pool), size=(updates, training["batch_pairs"]))
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold, seed=seed, arm=arm, mode="smoke" if smoke else "full",
        updates=updates, fit=plan["folds"][fold]["fit"], held_out=plan["folds"][fold]["held_out"], feature_sha256=bindings,
        initial_head_sha256=initial, initial_frozen_sha256=frozen, trainable_names=names,
        draws_fingerprint=c.fingerprint(draws.tolist()), anchor_pool=plan["anchor_pools"][fold])
    directory = folder(root, fold, seed, arm, smoke)
    opt = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    entries = {e["sequence"]: e for e in plan["entries"] if e["sequence"] in metadata["fit"]}
    state = restore(torch, directory, metadata); start = 0; history = []; counts = dict(supervised=0, regression=0, center=0)
    totals = dict(reference_calls=0, student_calls=0, head_only_batches=0, head_only_examples=0, successor_parents=0,
        changed_successor_parents=0, changed_successor_crops=0, qualified_endpoints=0, endpoints=0, qualified_first_transitions=0, first_transitions=0)
    _, blank, maximum_gradient_audit = gradient_rule(torch, torch.zeros(1), torch.zeros(1), arm, plan["design"])
    gradient_totals = {k: 0 for k in blank}
    gradient_totals.update(adamw_supervised_opposition_batches=0, adamw_supervised_descent_sum=0.,
        conflict_all_references_better=0, conflict_any_reference_not_better=0)
    quality_totals = dict(eligible_endpoints=0, retained_endpoints=0, quality_better=0, quality_tied=0, quality_worse=0)
    maximum_gradient = dict(response=0., geometry=0.); maximum_endpoint_gradient = 0.
    decode_error = response_error = native_error = 0.
    if state:
        head.load_state_dict(state["head"], strict=True); opt.load_state_dict(state["optimizer"])
        start = state["step"]; history = state["history"]; counts = state["counts"]; totals = state["totals"]
        gradient_totals = state["gradient_totals"]; quality_totals = state["quality_totals"]; maximum_gradient_audit = state["maximum_gradient_audit"]
        maximum_gradient = state["maximum_gradient"]; maximum_endpoint_gradient = state["maximum_endpoint_gradient"]
        decode_error = state["decode_error"]; response_error = state["response_error"]; native_error = state["native_error"]
        if frozen_digest(head, names) != frozen: raise ValueError("Frozen branch changed on resume")
    for step in range(start+1, updates+1):
        opt.zero_grad(set_to_none=True)
        rows, ends, audit, ne = rollout_batch(native, head, plan, entries, [pool[int(i)] for i in draws[step-1]])
        loss, stats, ordinary, penalty, de, re = objective(torch, head, rows, ends, plan, arm)
        gs, gp = gradient_vectors(torch, ordinary, penalty, parameters)
        applied, ga, ge = gradient_rule(torch, gs, gp, arm, plan["design"])
        quality = c.quality_composition(ends)
        ga["conflict_all_references_better"] = int(ga["conflict_batches"] and quality["eligible_endpoints"] == quality["quality_better"])
        ga["conflict_any_reference_not_better"] = ga["conflict_batches"]-ga["conflict_all_references_better"]
        # All arms pay the full scalar backward; preserve control rounding.
        backward_gradient(loss, parameters, gs, applied, ga["actual_modified_batches"])
        if any(p.grad is not None for n, p in head.named_parameters() if n not in names): raise ValueError("Gradient reached frozen response")
        gn = float(torch.nn.utils.clip_grad_norm_(parameters, training["gradient_clip"]))
        if not np.isfinite([float(loss.detach()), gn, de, re, ne]+list(ga.values())).all(): raise ValueError("Non-finite gradient training")
        before = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
        opt.step()
        after = torch.cat([p.detach().reshape(-1) for p in parameters])
        descent = -float(torch.dot(gs.to(torch.float64), (after-before).to(torch.float64)))
        if not np.isfinite(descent): raise ValueError("Non-finite actual AdamW displacement")
        ga["adamw_supervised_descent_sum"] = descent
        ga["adamw_supervised_opposition_batches"] = int(descent < 0)
        for k in gradient_totals: gradient_totals[k] += ga[k]
        for k in quality_totals: quality_totals[k] += quality[k]
        for k in maximum_gradient_audit: maximum_gradient_audit[k] = max(maximum_gradient_audit[k], ge[k])
        maximum_gradient["geometry"] = max(maximum_gradient["geometry"], gn)
        maximum_endpoint_gradient = max(maximum_endpoint_gradient, ga["anchor_norm_sum"])
        decode_error = max(decode_error, de); response_error = max(response_error, re); native_error = max(native_error, ne)
        for k in counts: counts[k] += int(stats[k+"_count"])
        for k in totals: totals[k] += audit[k]
        if step == 1 or step % training["save_every"] == 0 or step == updates:
            history.append(dict(step=step, full_scalar_loss=float(loss.detach()), ordinary_loss=float(ordinary.detach()),
                endpoint_penalty=float(penalty.detach()), clipped_from_gradient_norm=gn,
                **counts, **totals, **gradient_totals, **quality_totals, **maximum_gradient_audit))
            print("[gradient fold{} seed{} {}] {}/{} conflicts={}".format(fold, seed, arm, step, updates, gradient_totals["conflict_batches"]), flush=True)
        if step % training["save_every"] == 0 or step == updates:
            native.assert_frozen()
            if frozen_digest(head, names) != frozen: raise ValueError("Response/BN changed")
            save_step(torch, directory/("step{:06d}.pt".format(step)), dict(metadata=metadata, step=step,
                head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, optimizer=opt.state_dict(),
                counts=counts, totals=totals, history=history, gradient_totals=gradient_totals, quality_totals=quality_totals,
                maximum_gradient_audit=maximum_gradient_audit, maximum_gradient=maximum_gradient,
                maximum_endpoint_gradient=maximum_endpoint_gradient, decode_error=decode_error, response_error=response_error, native_error=native_error))
    expected = updates*training["batch_pairs"]*2*plan["design"]["horizon"]
    if any(totals[k] != expected for k in ("reference_calls", "student_calls", "head_only_batches", "head_only_examples")) or counts["supervised"] != expected:
        raise ValueError("Short-branch supervision/call budget changed")
    if digest(head) == initial or maximum_gradient["geometry"] <= 0 or frozen_digest(head, names) != frozen or totals["qualified_endpoints"] <= 0 or maximum_endpoint_gradient <= 0:
        raise ValueError("No valid short-rollout intervention")
    native.assert_frozen(); c.verify_files(bindings)
    path = directory/("step{:06d}.pt".format(updates))
    control = next(r for r in plan["endpoint_controls"] if (r["fold"], r["seed"]) == (fold, seed))
    reproduced = None
    if arm == "endpoint_control" and not smoke:
        reproduced = digest(head) == control["final_head_sha256"] and counts == control["counts"] and metadata["draws_fingerprint"] == control["draws_fingerprint"]
        if not reproduced: raise ValueError("Stage-57 endpoint control did not reproduce exactly")
    return dict(source_control_sha256=control["checkpoint_sha256"], reproduces_stage57_control=reproduced,
        fold=fold, seed=seed, arm=arm, checkpoint=str(path.resolve()), checkpoint_sha256=c.file_hash(path), receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),
        initial_head_sha256=initial, final_head_sha256=digest(head), initial_frozen_sha256=frozen, final_frozen_sha256=frozen_digest(head, names),
        trainable_names=names, trainable_parameters=sum(p.numel() for p in parameters), draws_fingerprint=metadata["draws_fingerprint"], counts=counts, totals=totals,
        maximum_gradient=maximum_gradient, maximum_endpoint_gradient=maximum_endpoint_gradient, maximum_gradient_audit=maximum_gradient_audit,
        gradient_totals=gradient_totals, quality_totals=quality_totals,
        maximum_frozen_map_error=response_error, maximum_decode_error=decode_error, maximum_native_branch_error=native_error, history=history)
