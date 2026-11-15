"""Nested generator/scorer isolation, variable search budgets and shared quality learning."""
from __future__ import annotations

import hashlib
import numpy as np
import stage111_slot_audit as prior

p, c, io, s103, s105, e63, s109 = prior.p, prior.c, prior.io, prior.s103, prior.s105, prior.e63, prior.s109
SCHEMA = "stage112_nested_shared_candidate_scoring_v1"
STATUS = "COMPLETE_NESTED_SHARED_SCORING_BUDGET_PILOT"
SOURCE111 = "61d3d1e808b8df3f317489c2b9f0dc4d69fe7f65a5e7f8e04d2aa492c39b13f5"
DESIGN = dict(outer_folds=4, fit_per_context=24, budgets=[2,4,6,8], scorer_fit_budget=4,
              seeds=[2312,2313], hidden=32, epochs=20, minibatch=128, learning_rate=3e-4,
              gradient_clip=1., feature_clip=10., minimum_scale=1e-6, tie_atol=1e-12,
              primary_condition="generated_4", primary="shared_minus_hann", primary_metric="iou")
CONDITIONS = ("fixed_4","generated_2","generated_4","generated_6","generated_8")


def folds(jobs, count):
    cohorts = {}
    for job in jobs:
        g = s109.s106.source_group(job["spec"]["unit"]["sequence"])
        cohort = job["spec"]["cohort"]
        if g in cohorts and cohorts[g] != cohort: raise ValueError("Mixed source cohort")
        cohorts[g] = cohort
    assignment = {g:i%count for cohort in sorted(set(cohorts.values()))
                  for i,g in enumerate(sorted(g for g in cohorts if cohorts[g] == cohort))}
    groups = sorted(cohorts)
    return [dict(fold=i, held_out_groups=[g for g in groups if assignment[g] == i],
                 train_groups=[g for g in groups if assignment[g] != i]) for i in range(count)]


def frame_selection(job, split, design):
    group = s109.s106.source_group(job["spec"]["unit"]["sequence"])
    if group in split["held_out_groups"]: return "evaluate",list(job["sampling"]["eval_frames"])
    pool = job["sampling"]["fit_frames"]
    if group not in split["train_groups"] or not pool: raise ValueError("Unknown or empty fitting context")
    indices = np.linspace(0,len(pool)-1,min(len(pool),design["fit_per_context"]),dtype=int)
    return "fit",[pool[i] for i in indices]


def fit_generator(rows, split):
    allowed = set(split["train_groups"]); forbidden = set(split["held_out_groups"])
    if allowed & forbidden: raise ValueError("Outer source leakage")
    local = [r for r in rows if r["group"] in allowed]
    model = p.fit_model(local,split["held_out_groups"][0],p.DESIGN)
    if forbidden & set(model["fit_groups"]+model["calibration_groups"]): raise ValueError("Generator leakage")
    return io.seal(dict(schema=SCHEMA,split=split,generator=model))


def search_parents(history, hw, ap, x, generator, maximum):
    mean,sigma = p.forecast(generator,x)
    hypotheses = p.parents_from_states(mean+p.scenarios(p.DESIGN)*sigma*generator["conditional_quantile"],history[-1],hw)
    options = np.vstack((history[-1],p.parents_from_states(mean[None],history[-1],hw),hypotheses))
    generated,indices = p.allocate(options,hypotheses,hw,ap,dict(p.DESIGN,search_budget=maximum))
    fixed = e63.source_common.prior.history_parents(history,ap["design"])
    np.testing.assert_array_equal(fixed[0],generated[0])
    return dict(candidate_parents=np.vstack((fixed,generated)),allocation_indices=indices,
                forecast_mean=mean,forecast_sigma=sigma,conditional_radius=sigma*generator["conditional_quantile"])


def features(base, boxes, history):
    """Original 64-dimensional content/relationship descriptor generalized to K>=2."""
    base,boxes,h = np.asarray(base,float),np.asarray(boxes,float),np.asarray(history,float)
    k = len(boxes)
    if k == 4: return prior.c74.c66.c64.choice_features(base,boxes,h)
    if k < 2 or base.shape != (k,24) or boxes.shape != (k,4): raise ValueError("Invalid variable candidate bank")
    if not all(np.isfinite(v).all() for v in (base,boxes,h)) or (boxes[:,2:]<=0).any() or (h[:,2:]<=0).any(): raise ValueError("Invalid geometry")
    overlap = prior.c74.c66.c64.continuous_iou
    centers = boxes[:,:2]+boxes[:,2:]/2; hc = h[:,:2]+h[:,2:]/2
    predicted = hc[-1]+(hc[-1]-hc[-2] if len(h)>1 else 0)
    unit = np.sqrt(h[-1,2:].prod())
    agreement = np.array([overlap(b,boxes) for b in boxes]); np.fill_diagonal(agreement,0.)
    distance = np.linalg.norm(centers[:,None]-centers[None],axis=2)/unit
    scale = np.abs(np.log(boxes[:,None,2:]/boxes[None,:,2:])).mean(2)
    peak = base[:,1]
    extra = np.column_stack((agreement.sum(1)/(k-1),agreement.max(1),np.array([overlap(b,boxes[:1])[0] for b in boxes]),
        np.linalg.norm(centers-predicted,axis=1)/unit,np.abs(np.log(boxes[:,2:])-np.median(np.log(h[:,2:]),axis=0)).mean(1),
        distance.sum(1)/(k-1),scale.sum(1)/(k-1),(peak[:,None]>peak[None]).mean(1)))
    value = np.clip(np.c_[base,extra],-10.,10.)
    return np.c_[value,value*value]


def physical_choice(scores, boxes):
    scores,boxes = np.asarray(scores,float),np.asarray(boxes,float)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all(): raise ValueError("Invalid candidate scores")
    tied = np.flatnonzero(scores>=scores.max()-DESIGN["tie_atol"])
    return 0 if 0 in tied else int(min(tied,key=lambda j:tuple(boxes[j])))


def normalization(x, weight, design):
    center = np.sum(weight[:,None,None]*x,axis=(0,1))/x.shape[1]
    scale = np.sqrt(np.sum(weight[:,None,None]*(x-center)**2,axis=(0,1))/x.shape[1])
    scale[scale<design["minimum_scale"]] = 1.
    return dict(center=center.tolist(),scale=scale.tolist())


def normalize(x, norm, design):
    x = np.asarray(x,float)
    if x.shape[-1]!=64 or not np.isfinite(x).all() or not np.isfinite(np.r_[norm["center"],norm["scale"]]).all() or np.any(np.asarray(norm["scale"])<=0):
        raise ValueError("Invalid fit-normalized descriptor")
    return np.clip((x-np.asarray(norm["center"]))/np.asarray(norm["scale"]),-design["feature_clip"],design["feature_clip"])


def initial(seed, architecture, design):
    rng = np.random.Generator(np.random.PCG64(seed))
    inputs,outputs = (64,1) if architecture == "shared" else (256,4)
    if architecture not in ("shared","concat"): raise ValueError("Unknown scorer")
    hidden = design["hidden"]
    return dict(w1=rng.normal(0,np.sqrt(2/(inputs+hidden)),(inputs,hidden)),b1=np.zeros(hidden),
                w2=rng.normal(0,np.sqrt(2/(hidden+outputs)),(hidden,outputs)),b2=np.zeros(outputs))


def forward(state, x, architecture):
    x = np.asarray(x,float)
    if architecture == "concat" and x.shape[-2:] != (4,64): raise ValueError("Concatenation control is fixed K=4")
    z = x.reshape(-1,64 if architecture == "shared" else 256)
    hidden = np.tanh(z@state["w1"]+state["b1"])
    output = hidden@state["w2"]+state["b2"]
    score = 1/(1+np.exp(-np.clip(output,-60,60)))
    return score.reshape(x.shape[:-1]),(z,hidden,score)


def loss_gradient(state, x, y, scaled_weights, architecture):
    score,(z,hidden,flat_score) = forward(state,x,architecture)
    difference = score-y
    loss = float(np.mean(scaled_weights[:,None]*difference**2))
    d = (2*scaled_weights[:,None]*difference/np.prod(y.shape)).reshape(flat_score.shape)*flat_score*(1-flat_score)
    gradients = dict(w2=hidden.T@d,b2=d.sum(0))
    dh = (d@state["w2"].T)*(1-hidden*hidden)
    gradients.update(w1=z.T@dh,b1=dh.sum(0))
    return loss,gradients


def training_arrays(rows, split):
    if not rows or any(r["group"] not in split["train_groups"] for r in rows): raise ValueError("Scorer source leakage")
    x = np.asarray([r["x"] for r in rows]); y = np.asarray([r["y"] for r in rows])
    if x.shape[1:] != (4,64) or y.shape != x.shape[:2] or not np.isfinite(x).all() or not np.isfinite(y).all() or ((y<0)|(y>1)).any(): raise ValueError("Invalid scorer examples")
    w = p.weights(rows)
    digest = hashlib.sha256()
    identities = [{k:r[k] for k in ("sequence","group","seed","source_arm","frame")} for r in rows]
    digest.update(io.seal(dict(rows=identities))["fingerprint"].encode())
    for a in (x,y,w): digest.update(np.asarray(a,dtype="<f8").tobytes())
    return x,y,w,digest.hexdigest()


def train(rows, split, architecture, seed, design):
    x,y,w,signature = training_arrays(rows,split)
    norm = normalization(x,w,design); x = normalize(x,norm,design)
    state = initial(seed,architecture,design)
    first_loss = loss_gradient(state,x,y,w*len(w),architecture)[0]
    first = {k:np.zeros_like(v) for k,v in state.items()}; second = {k:v.copy() for k,v in first.items()}
    rng = np.random.Generator(np.random.PCG64(seed+100000)); step = 0
    for _ in range(design["epochs"]):
        order = rng.permutation(len(x))
        for start in range(0,len(x),design["minibatch"]):
            ids = order[start:start+design["minibatch"]]
            _,gradient = loss_gradient(state,x[ids],y[ids],w[ids]*len(w),architecture)
            length = np.sqrt(sum(float((v*v).sum()) for v in gradient.values()))
            if not np.isfinite(length): raise ValueError("Nonfinite scorer gradient")
            multiplier = min(1.,design["gradient_clip"]/(length+1e-12)); step += 1
            for name in state:
                g = gradient[name]*multiplier
                first[name] = .9*first[name]+.1*g; second[name] = .999*second[name]+.001*g*g
                state[name] -= design["learning_rate"]*(first[name]/(1-.9**step))/(np.sqrt(second[name]/(1-.999**step))+1e-8)
    loss = loss_gradient(state,x,y,w*len(w),architecture)[0]
    return io.seal(dict(schema=SCHEMA,design=design,split=split,architecture=architecture,seed=seed,examples=len(x),
        input_fingerprint=signature,norm=norm,state={k:v.tolist() for k,v in state.items()},optimizer_steps=step,
        parameters=sum(v.size for v in state.values()),first_fit_mse=first_loss,final_fit_mse=loss))


def scores(model, x):
    state = {k:np.asarray(v,float) for k,v in model["state"].items()}
    z = normalize(x,model["norm"],model["design"])
    return forward(state,z,model["architecture"])[0]
