"""CPU float64 actor/critic updates; no SGLA parameters enter an optimizer."""
from __future__ import annotations
import copy
import numpy as np
import stage71_ppo_common as c


def network(torch, design):
    def branch(): return torch.nn.Sequential(torch.nn.Linear(128, design["hidden"]), torch.nn.Tanh(), torch.nn.Linear(design["hidden"], 1))
    return torch.nn.ModuleDict(dict(actor=branch(), critic=branch())).to(device="cpu", dtype=torch.float64)


def export(net): return {k: v.detach().cpu().numpy().tolist() for k, v in net.state_dict().items()}


def restore(torch, state, design):
    with torch.random.fork_rng(devices=[]): net = network(torch, design)
    expected = net.state_dict()
    if set(expected) != set(state): raise ValueError("Controller architecture changed")
    values = {k: torch.tensor(v, dtype=torch.float64) for k, v in state.items()}
    if any(values[k].shape != expected[k].shape or not torch.isfinite(values[k]).all() for k in values): raise ValueError("Invalid controller parameters")
    net.load_state_dict(values); return net


def initial(torch, seed, norm, design):
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU controller; torch.manual_seed also reseeds CUDA and
        # must not change the frozen SGLA runtime's random stream.
        torch.random.default_generator.manual_seed(seed); net = network(torch, design)
        for name in ("actor", "critic"):
            torch.nn.init.orthogonal_(net[name][0].weight, gain=np.sqrt(2.)); torch.nn.init.zeros_(net[name][0].bias)
            torch.nn.init.zeros_(net[name][2].weight); torch.nn.init.zeros_(net[name][2].bias)
    return dict(state=export(net), norm=copy.deepcopy(norm), optimizer=None)


def optimizer(torch, net, saved, design):
    opt = torch.optim.Adam(net.parameters(), lr=design["learning_rate"], eps=design["adam_epsilon"])
    if saved is not None:
        state = {int(k): {n: torch.tensor(v, dtype=torch.float32 if n == "step" else torch.float64) for n, v in d.items()} for k, d in saved["state"].items()}
        opt.load_state_dict(dict(state=state, param_groups=saved["param_groups"]))
    return opt


def export_optimizer(opt):
    import json
    d = opt.state_dict()
    return dict(state={str(k): {n: v.detach().cpu().tolist() for n, v in s.items()} for k, s in d["state"].items()}, param_groups=json.loads(json.dumps(d["param_groups"])))


def objective(torch, net, batch, design, arm):
    x = batch["x"]; logits = net["actor"](x).squeeze(-1); values = net["critic"](x).squeeze(-1)
    mask = batch["eligible"]
    entropy = logits.sum()*0.; policy_loss = logits.sum()*0.; kl = logits.sum()*0.; clipped_fraction = logits.sum()*0.
    if mask.any():
        z = logits[mask]; acts = batch["action"][mask]
        lp = -torch.nn.functional.softplus(torch.where(acts.bool(), -z, z))
        p = torch.sigmoid(z)
        entropy = (torch.nn.functional.softplus(z)-p*z).mean()
        if arm == "ppo":
            delta = lp-batch["log_prob"][mask]; ratio = delta.exp(); adv = batch["advantage"][mask]
            policy_loss = -torch.minimum(ratio*adv, ratio.clamp(1-design["clip"], 1+design["clip"])*adv).mean()
            kl = (torch.expm1(delta)-delta).mean()
            clipped_fraction = ((ratio-1).abs()>design["clip"]).double().mean()
        elif arm == "myopic":
            policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(z, batch["label"][mask])
        else: raise ValueError("Unknown training objective")
    value_loss = (values-batch["return"]).square().mean()
    loss = policy_loss+design["value_coefficient"]*value_loss-design["entropy_coefficient"]*entropy
    if not torch.isfinite(loss): raise ValueError("Nonfinite training objective")
    return loss, dict(policy_loss=policy_loss.detach(), value_loss=value_loss.detach(), entropy=entropy.detach(),
                      approximate_kl=kl.detach(), clip_fraction=clipped_fraction.detach())


def update(torch, packs, data, design, update_seed):
    a, returns = c.gae(data["reward"], data["value"], design["gamma"], design["gae_lambda"])
    mask = data["eligible"].astype(bool); normalized = np.zeros_like(a)
    if mask.any(): normalized[mask] = (a[mask]-a[mask].mean())/max(a[mask].std(), 1e-8)
    arrays = dict(x=data["x"], eligible=mask, action=data["action"], log_prob=data["log_prob"], advantage=normalized,
                  label=data["myopic_label"], **{"return": returns})
    batch = {k: torch.tensor(v, dtype=torch.bool if k == "eligible" else torch.float64) for k, v in arrays.items()}
    nets = {arm: restore(torch, packs[arm]["state"], design) for arm in ("ppo", "myopic")}
    opts = {arm: optimizer(torch, nets[arm], packs[arm]["optimizer"], design) for arm in nets}
    rows = {arm: [] for arm in nets}; rng = np.random.default_rng(update_seed)
    for _ in range(design["epochs"]):
        order = rng.permutation(len(a))
        for start in range(0, len(a), design["minibatch"]):
            indices = order[start:start+design["minibatch"]]; minibatch = {k: v[indices] for k, v in batch.items()}
            for arm in nets:
                opt = opts[arm]; opt.zero_grad(set_to_none=True)
                loss, metrics = objective(torch, nets[arm], minibatch, design, arm); loss.backward()
                branch_grads = {name: float(torch.sqrt(sum(p.grad.square().sum() for p in nets[arm][name].parameters() if p.grad is not None))) for name in ("actor", "critic")}
                grad = torch.nn.utils.clip_grad_norm_(nets[arm].parameters(), design["gradient_clip"])
                if not torch.isfinite(grad): raise ValueError("Nonfinite controller gradient")
                opt.step(); rows[arm].append({k: float(v) for k, v in metrics.items()} | dict(gradient_norm=float(grad), **{k+"_gradient":v for k,v in branch_grads.items()}))
    result = {arm: dict(packs[arm], state=export(nets[arm]), optimizer=export_optimizer(opts[arm])) for arm in nets}
    stats = {arm: dict(updates=len(rows[arm]), **{k: float(np.mean([r[k] for r in rows[arm]])) for k in rows[arm][0]},
                       maximum_actor_gradient=max(r["actor_gradient"] for r in rows[arm]), maximum_critic_gradient=max(r["critic_gradient"] for r in rows[arm])) for arm in nets}
    return result, stats
