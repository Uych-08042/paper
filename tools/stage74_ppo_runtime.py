"""Categorical four-candidate PPO/CE, CPU float64, separate gradient clipping."""
from __future__ import annotations
import copy
import numpy as np
import stage74_ppo_common as c
import stage71_ppo_runtime as serialization
import stage73_ppo_runtime as clipping
import stage72_ppo_common as diagnostics
export=serialization.export
optimizer=serialization.optimizer
export_optimizer=serialization.export_optimizer


def network(torch,design):
    def branch(outputs):return torch.nn.Sequential(torch.nn.Linear(design["input_dim"],design["hidden"]),torch.nn.Tanh(),torch.nn.Linear(design["hidden"],outputs))
    return torch.nn.ModuleDict(dict(actor=branch(4),critic=branch(1))).to(device="cpu",dtype=torch.float64)


def restore(torch,state,design):
    with torch.random.fork_rng(devices=[]):net=network(torch,design)
    expected=net.state_dict();values={k:torch.tensor(v,dtype=torch.float64) for k,v in state.items()}
    if set(values)!=set(expected) or any(values[k].shape!=expected[k].shape or not torch.isfinite(values[k]).all() for k in values):raise ValueError("Invalid four-candidate architecture/parameters")
    net.load_state_dict(values);return net


def initial(torch,seed,norm,design):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);net=network(torch,design)
        for name in ("actor","critic"):
            torch.nn.init.orthogonal_(net[name][0].weight,gain=np.sqrt(2.));torch.nn.init.zeros_(net[name][0].bias)
            torch.nn.init.zeros_(net[name][2].weight);torch.nn.init.zeros_(net[name][2].bias)
    return dict(state=export(net),norm=copy.deepcopy(norm),optimizer=None)


def objective(torch,net,batch,design,arm):
    logits=net["actor"](batch["x"]);values=net["critic"](batch["x"]).squeeze(-1);mask=batch["action_mask"]
    if mask.shape!=logits.shape or not mask[:,0].all():raise ValueError("Invalid categorical action mask")
    active=mask.sum(1)>1;lp=torch.log_softmax(logits.masked_fill(~mask,-torch.inf),dim=-1)
    p=lp.exp();safe_lp=torch.where(mask,lp,torch.zeros_like(lp))
    entropy=logits.sum()*0.;policy_loss=logits.sum()*0.;kl=logits.sum()*0.;clipped=logits.sum()*0.
    if active.any():
        entropy=(-(p*safe_lp).sum(1))[active].mean()
        if arm=="ppo":
            actions=batch["action"].long()
            if not mask.gather(1,actions[:,None]).all():raise ValueError("Stored action has become illegal")
            delta=lp.gather(1,actions[:,None]).squeeze(1)[active]-batch["log_prob"][active]
            ratio=delta.exp();adv=batch["advantage"][active]
            policy_loss=-torch.minimum(ratio*adv,ratio.clamp(1-design["clip"],1+design["clip"])*adv).mean()
            kl=(torch.expm1(delta)-delta).mean();clipped=((ratio-1).abs()>design["clip"]).double().mean()
        elif arm=="myopic":
            labels=batch["label"].long()
            if not mask.gather(1,labels[:,None]).all():raise ValueError("Illegal supervised candidate target")
            policy_loss=-lp.gather(1,labels[:,None]).squeeze(1)[active].mean()
        else:raise ValueError("Unknown four-candidate objective")
    value_loss=(values-batch["return"]).square().mean()
    loss=policy_loss+design["value_coefficient"]*value_loss-design["entropy_coefficient"]*entropy
    if not torch.isfinite(loss):raise ValueError("Nonfinite categorical objective")
    return loss,dict(policy_loss=policy_loss.detach(),value_loss=value_loss.detach(),entropy=entropy.detach(),approximate_kl=kl.detach(),clip_fraction=clipped.detach())


def prediction_stats(data,before,after,design,arm,target):
    mask=data["action_mask"].astype(bool);active=mask.sum(1)>1;out={}
    choices={};post_lp=None
    for label,pack in (("before",before),("after",after)):
        logits=c.forward(data["x"],pack["state"],"actor");pairs=[c.distribution(z,m) for z,m in zip(logits,mask)]
        p=np.asarray([v[0] for v in pairs]);lp=np.asarray([v[1] for v in pairs]);selected=p.argmax(1);choices[label]=selected
        value=c.forward(data["x"],pack["state"],"critic");mc=diagnostics.discounted_return(data["reward"],design["gamma"])
        out.update({label+"_mean_q0_probability":diagnostics.mean(p[:,0],active),label+"_mean_max_probability":diagnostics.mean(p.max(1),active),
            label+"_deterministic_nonq0_fraction":float((selected!=0).mean()),label+"_label_accuracy":diagnostics.mean(selected==data["myopic_label"],active),
            label+"_fixed_GAE_mse":float(((value-target)**2).mean()),label+"_behavior_MC_explained_variance":diagnostics.explained_variance(value,mc)})
        if label=="after":post_lp=lp
    for branch in ("actor","critic"):
        delta=np.concatenate([(np.asarray(after["state"][k])-np.asarray(before["state"][k])).ravel() for k in before["state"] if k.startswith(branch+".")])
        out[branch+"_step_l2"]=float(np.linalg.norm(delta))
    out["decision_flip_fraction"]=float((choices["before"]!=choices["after"]).mean())
    diff=post_lp[np.arange(len(mask)),data["action"].astype(int)][active]-data["log_prob"][active]
    out["post_behavior_kl"]=diagnostics.mean(np.expm1(diff)-diff) if arm=="ppo" else None
    out["post_behavior_clip_fraction"]=diagnostics.mean(abs(np.expm1(diff))>design["clip"]) if arm=="ppo" else None
    return out


def update(torch,packs,data,design,update_seed):
    a,target=c.gae(data["reward"],data["value"],design["gamma"],design["gae_lambda"]);active=data["eligible"].astype(bool);adv=np.zeros_like(a)
    if active.any():adv[active]=(a[active]-a[active].mean())/max(a[active].std(),1e-8)
    arrays=dict(x=data["x"],action_mask=data["action_mask"],action=data["action"],log_prob=data["log_prob"],advantage=adv,label=data["myopic_label"],**{"return":target})
    batch={k:torch.tensor(v,dtype=torch.bool if k=="action_mask" else torch.float64) for k,v in arrays.items()}
    nets={arm:restore(torch,packs[arm]["state"],design) for arm in ("ppo","myopic")};opts={arm:optimizer(torch,nets[arm],packs[arm]["optimizer"],design) for arm in nets}
    rows={arm:[] for arm in nets};rng=np.random.default_rng(update_seed)
    for _ in range(design["epochs"]):
        order=rng.permutation(len(a))
        for start in range(0,len(a),design["minibatch"]):
            ids=order[start:start+design["minibatch"]];mini={k:v[ids] for k,v in batch.items()}
            for arm in nets:
                opts[arm].zero_grad(set_to_none=True);loss,metrics=objective(torch,nets[arm],mini,design,arm);loss.backward()
                grads=clipping.clip_branches(torch,nets[arm],design);opts[arm].step()
                rows[arm].append({k:float(v) for k,v in metrics.items()}|grads)
    result={arm:dict(packs[arm],state=export(nets[arm]),optimizer=export_optimizer(opts[arm])) for arm in nets}
    stats={arm:dict(updates=len(rows[arm]),**{k:float(np.mean([r[k] for r in rows[arm]])) for k in rows[arm][0]},
        maximum_actor_gradient=max(r["actor_gradient"] for r in rows[arm]),maximum_critic_gradient=max(r["critic_gradient"] for r in rows[arm]),
        **prediction_stats(data,packs[arm],result[arm],design,arm,target)) for arm in nets}
    return result,stats
