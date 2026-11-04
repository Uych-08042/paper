"""Separate branch gradient clipping, unchanged Stage-71 Adam and PPO/BCE losses."""
from __future__ import annotations
import numpy as np
import stage73_ppo_common as c
import stage71_ppo_runtime as prior

network,export,restore,initial = prior.network,prior.export,prior.restore,prior.initial
optimizer,export_optimizer,objective = prior.optimizer,prior.export_optimizer,prior.objective


def gradient_norm(torch, parameters):
    values = [p.grad.square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(sum(values))) if values else 0.


def clip_branches(torch, net, design):
    if design["gradient_clip_mode"] != "separate_actor_critic": raise ValueError("Need sealed separate clipping treatment")
    before = {k:gradient_norm(torch,net[k].parameters()) for k in ("actor","critic")}
    for branch in ("actor","critic"):
        value = torch.nn.utils.clip_grad_norm_(net[branch].parameters(),design["gradient_clip"])
        if not torch.isfinite(value): raise ValueError("Nonfinite controller gradient")
    after = {k:gradient_norm(torch,net[k].parameters()) for k in before}
    return dict(gradient_norm=float(np.hypot(before["actor"],before["critic"])),
        **{k+"_gradient":v for k,v in before.items()},**{k+"_post_clip_gradient":v for k,v in after.items()},
        **{k+"_clip_coefficient":min(1.,design["gradient_clip"]/(v+1e-6)) for k,v in before.items()})


def update(torch, packs, data, design, update_seed):
    a,returns = c.gae(data["reward"],data["value"],design["gamma"],design["gae_lambda"])
    mask = data["eligible"].astype(bool); normalized = np.zeros_like(a)
    if mask.any(): normalized[mask] = (a[mask]-a[mask].mean())/max(a[mask].std(),1e-8)
    arrays = dict(x=data["x"],eligible=mask,action=data["action"],log_prob=data["log_prob"],advantage=normalized,
                  label=data["myopic_label"],**{"return":returns})
    batch = {k:torch.tensor(v,dtype=torch.bool if k=="eligible" else torch.float64) for k,v in arrays.items()}
    nets = {arm:restore(torch,packs[arm]["state"],design) for arm in ("ppo","myopic")}
    opts = {arm:optimizer(torch,nets[arm],packs[arm]["optimizer"],design) for arm in nets}
    rows = {arm:[] for arm in nets}; rng = np.random.default_rng(update_seed)
    for _ in range(design["epochs"]):
        order = rng.permutation(len(a))
        for start in range(0,len(a),design["minibatch"]):
            indices = order[start:start+design["minibatch"]]; minibatch = {k:v[indices] for k,v in batch.items()}
            for arm in nets:
                opt = opts[arm]; opt.zero_grad(set_to_none=True)
                loss,metrics = objective(torch,nets[arm],minibatch,design,arm); loss.backward()
                gradients = clip_branches(torch,nets[arm],design)
                opt.step(); rows[arm].append({k:float(v) for k,v in metrics.items()}|gradients)
    result = {arm:dict(packs[arm],state=export(nets[arm]),optimizer=export_optimizer(opts[arm])) for arm in nets}
    stats = {arm:dict(updates=len(rows[arm]),**{k:float(np.mean([r[k] for r in rows[arm]])) for k in rows[arm][0]},
        maximum_actor_gradient=max(r["actor_gradient"] for r in rows[arm]),maximum_critic_gradient=max(r["critic_gradient"] for r in rows[arm])) for arm in nets}
    return result,stats
