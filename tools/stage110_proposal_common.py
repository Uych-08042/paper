"""Fit source-group-isolated state forecasts and allocate a bounded SGLA search bank."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

import stage109_sampled_commit as s109
import stage84_motion_candidates as motion

c, io, s103, s105, e63 = s109.c, s109.io, s109.s103, s109.s105, s109.e63
SCHEMA = "stage110_calibrated_search_proposals_v1"
STATUS = "COMPLETE_LEARNED_SEARCH_PROPOSAL_SUPPLY_PILOT"
SOURCE109 = "1094eac91bf60b49ab1073a2f59779b7956449d15332f260d5b6acbd58702d80"
DESIGN = dict(first_frame=5,fit_per_context=96,eval_per_context=24,sampler_seed=2310,
    ridge=1.,minimum_scale=1e-6,variance_floor=1e-4,nominal_coverage=.9,
    calibration_modulus=3,inner_folds=3,scenario_count=32,search_budget=4,
    core_fraction=.25,feature_clip=10.,primary="learned_calibrated_minus_fixed")
ARMS = ("fixed","motion","learned_constant","learned_uncalibrated","learned_calibrated","calibrated_random")
FEATURES = tuple(f"step{k}_{v}" for k in range(4) for v in ("dx","dy","dlogw","dlogh"))+tuple(
    p+"_"+v for p in ("mean","std","turn") for v in ("dx","dy","dlogw","dlogh"))+("history_fraction","past_raw_peak","past_hann_peak","past_memory_write")


def seed_for(key):
    return int.from_bytes(hashlib.sha256(f"{DESIGN['sampler_seed']}:{key}".encode()).digest()[:16],"big")


def sample_frames(frame_ids,context,design):
    frames=np.asarray(frame_ids)
    np.testing.assert_array_equal(frames,np.arange(1,len(frames)+1))
    pool=frames[frames>=design["first_frame"]].astype(int)
    fit=sorted(int(x) for x in pool[np.linspace(0,len(pool)-1,min(len(pool),design["fit_per_context"]),dtype=int)]) if len(pool) else []
    remaining=sorted(set(pool.tolist())-set(fit))
    seed=seed_for(context)
    evaluation=sorted(int(x) for x in np.random.Generator(np.random.PCG64(seed)).choice(
        remaining,size=min(len(remaining),design["eval_per_context"]),replace=False))
    return dict(fit_frames=fit,eval_frames=evaluation,pool_size=len(pool),eval_pool_size=len(remaining),
        sampler_seed=str(seed),eval_inclusion_probability=len(evaluation)/len(remaining) if remaining else 0.)


def online_input(history,past_features,past_selected,past_written):
    """Only previous own boxes/responses are accepted; no current image, GT or group ID."""
    h=np.asarray(history,float)[-5:]
    f=np.asarray(past_features,float)
    if h.ndim!=2 or h.shape[1]!=4 or len(h)<2 or not np.isfinite(h).all() or (h[:,2:]<=0).any() or f.shape!=(4,24):
        raise ValueError("Need valid past state and response bank")
    center=h[:,:2]+h[:,2:]/2; side=float(np.sqrt(h[-1,2:].prod()))
    increments=np.c_[np.diff(center,axis=0)/side,np.diff(np.log(h[:,2:]),axis=0)]
    steps=np.zeros((4,4));steps[-len(increments):]=increments
    turn=increments[-1]-increments[-2] if len(increments)>1 else np.zeros(4)
    names=e63.source_common.prior.FEATURES
    x=np.r_[steps.ravel(),increments.mean(0),increments.std(0),turn,len(h)/5,
        f[int(past_selected),names.index("raw_peak")],f[int(past_selected),names.index("hann_peak")],float(past_written)]
    if x.shape!=(len(FEATURES),) or not np.isfinite(x).all():
        raise ValueError("Invalid causal forecast features")
    return np.clip(x,-10.,10.)


def input_at(trace,entry,frame,old):
    h=e63.history_at(trace,entry["initial_box"],frame,old)
    i=frame-2
    if i<0:raise ValueError("Need a previous predicted frame")
    x=online_input(h,trace["features"][i],trace["selected"][i],trace["memory_write"][i])
    return x,h


def target(box,latest):
    b,p=np.asarray(box,float),np.asarray(latest,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=0).any():return None
    return np.r_[(b[:2]+b[2:]/2-p[:2]-p[2:]/2)/np.sqrt(p[2:].prod()),np.log(b[2:]/p[2:])]


def weights(rows):
    return s109.s106.balanced_weights(rows)


def ridge_fit(rows,values,design):
    if not rows:raise ValueError("Missing prediction fitting support")
    x=np.asarray([r["x"] for r in rows],float);y=np.asarray(values,float);w=weights(rows)
    mu=w@x;scale=np.sqrt(w@((x-mu)**2));scale[scale<design["minimum_scale"]]=1.
    z=(x-mu)/scale;intercept=w@y
    beta=np.linalg.solve(z.T@(w[:,None]*z)+design["ridge"]*np.eye(x.shape[1]),z.T@(w[:,None]*(y-intercept)))
    return dict(center=mu.tolist(),scale=scale.tolist(),intercept=intercept.tolist(),beta=beta.tolist())


def ridge_predict(model,x):
    x=np.asarray(x,float)
    return ((x-model["center"])/model["scale"])@np.asarray(model["beta"])+model["intercept"]


def quantile(values,weight,level):
    v,w=np.asarray(values,float),np.asarray(weight,float)
    order=np.argsort(v,kind="stable");cumulative=np.cumsum(w[order])/w.sum()
    return float(v[order[min(np.searchsorted(cumulative,level),len(v)-1)]])


def fit_model(rows,held_out,design):
    others=sorted({r["group"] for r in rows}-{held_out})
    if len(others)<5:raise ValueError("Need five or more other source groups")
    calibration=others[::design["calibration_modulus"]]
    fit_groups=[g for g in others if g not in calibration]
    fit=[r for r in rows if r["group"] in fit_groups]
    cal=[r for r in rows if r["group"] in calibration]
    y=np.asarray([r["y"] for r in fit]);x=np.asarray([r["x"] for r in fit])
    folds={g:i%design["inner_folds"] for i,g in enumerate(fit_groups)}
    residual=np.full_like(y,np.nan);inner=[]
    for fold in range(design["inner_folds"]):
        train=[i for i,r in enumerate(fit) if folds[r["group"]]!=fold]
        held=[i for i,r in enumerate(fit) if folds[r["group"]]==fold]
        if not held or not train:raise ValueError("Incomplete fit-only residual folds")
        model=ridge_fit([fit[i] for i in train],y[train],design)
        residual[held]=y[held]-ridge_predict(model,x[held])
        inner.append(dict(fold=fold,fit_groups=sorted({fit[i]["group"] for i in train}),residual_groups=sorted({fit[i]["group"] for i in held})))
    if not np.isfinite(residual).all():raise ValueError("Incomplete own-prediction residual support")
    mean=ridge_fit(fit,y,design)
    variance=ridge_fit(fit,np.log(residual**2+design["variance_floor"]),design)
    constant=np.sqrt(weights(fit)@(residual**2)+design["variance_floor"])
    cx=np.asarray([r["x"] for r in cal]);cy=np.asarray([r["y"] for r in cal]);error=np.abs(cy-ridge_predict(mean,cx))
    scale=np.sqrt(np.exp(np.clip(ridge_predict(variance,cx),-20,20))+design["variance_floor"])
    q_cond=quantile((error/scale).max(1),weights(cal),design["nominal_coverage"])
    q_const=quantile((error/constant).max(1),weights(cal),design["nominal_coverage"])
    inputs=[{k:r[k] for k in ("group","sequence","seed","source_arm","frame","x","y")} for r in rows if r["group"]!=held_out]
    return io.seal(dict(schema=SCHEMA,held_out_group=held_out,design=design,features=list(FEATURES),
        fit_groups=fit_groups,calibration_groups=calibration,fit_rows=len(fit),calibration_rows=len(cal),
        inner_folds=inner,input_fingerprint=io.seal(dict(rows=inputs))["fingerprint"],mean=mean,variance=variance,
        constant_scale=constant.tolist(),conditional_quantile=q_cond,constant_quantile=q_const))


def forecast(model,x):
    mean=ridge_predict(model["mean"],x)
    sigma=np.sqrt(np.exp(np.clip(ridge_predict(model["variance"],x),-20,20))+model["design"]["variance_floor"])
    return mean,sigma


def scenarios(design):
    # Fixed antithetic Gaussian scenarios; dimensions move jointly, common across controls.
    rng=np.random.Generator(np.random.PCG64(design["sampler_seed"]+1))
    half=rng.standard_normal((design["scenario_count"]//2,4))
    return np.vstack((half,-half))


def parents_from_states(states,latest,hw):
    s=np.asarray(states,float);p=np.asarray(latest,float)
    centers=p[:2]+p[2:]/2+s[:,:2]*np.sqrt(p[2:].prod())
    centers=np.clip(centers,0,np.asarray(hw[::-1])-1)
    size=np.clip(p[2:]*np.exp(np.clip(s[:,2:],-20,20)),1,np.asarray(hw[::-1]))
    return np.c_[centers-size/2,size]


def allocate(options,hypotheses,hw,ap,design,random_seed=None):
    rects=np.asarray([s105.crop_geometry(p,hw,ap["search_factor"],ap["search_size"])[0] for p in options])
    point=hypotheses[:,:2]+hypotheses[:,2:]/2
    centers=rects[:,:2]+rects[:,2:]/2
    offset=(centers[:,None]-point[None])/np.maximum(rects[:,None,2:]*design["core_fraction"],1.)
    # Geometric coverage is only a proposal proxy; decoded SGLA boxes are the measured endpoint.
    support=np.exp(-.5*np.sum(offset**2,axis=2))
    ratio=np.abs(np.log(options[:,None,2:]/hypotheses[None,:,2:])).mean(2)
    support*=np.exp(-ratio)
    chosen=[0];used={tuple(rects[0])};covered=support[0].copy()
    rng=np.random.Generator(np.random.PCG64(random_seed)) if random_seed is not None else None
    for _ in range(design["search_budget"]-1):
        available=np.array([i not in chosen and tuple(r) not in used for i,r in enumerate(rects)])
        if not available.any():
            chosen.append(0);continue
        gain=np.maximum(support-covered,0).mean(1)
        index=int(rng.choice(np.flatnonzero(available))) if rng is not None else int(np.argmax(np.where(available,gain,-np.inf)))
        chosen.append(index);used.add(tuple(rects[index]));covered=np.maximum(covered,support[index])
    return np.asarray(options)[chosen].copy(),np.asarray(chosen,int)


def proposal_banks(history,hw,ap,x,model,context,design):
    history=np.asarray(history,float);latest=history[-1]
    fixed=motion.layouts(history,hw,ap,motion.DESIGN)
    mean,sigma=forecast(model,x)
    spreads=(np.asarray(model["constant_scale"])*model["constant_quantile"],sigma,sigma*model["conditional_quantile"])
    banks=[fixed[0],fixed[1]];indices=[np.arange(4),np.full(4,-1)]
    z=scenarios(design)
    for spread in spreads:
        states=mean+z*spread
        hypotheses=parents_from_states(states,latest,hw)
        options=np.vstack((latest,parents_from_states(mean[None],latest,hw),hypotheses))
        bank,index=allocate(options,hypotheses,hw,ap,design)
        banks.append(bank);indices.append(index)
    spread=spreads[-1];hypotheses=parents_from_states(mean+z*spread,latest,hw)
    options=np.vstack((latest,parents_from_states(mean[None],latest,hw),hypotheses))
    bank,index=allocate(options,hypotheses,hw,ap,design,seed_for(context))
    banks.append(bank);indices.append(index)
    out=np.asarray(banks)
    if out.shape!=(len(ARMS),4,4) or not np.isfinite(out).all() or (out[...,2:]<=0).any():raise ValueError("Invalid generated search bank")
    np.testing.assert_array_equal(out[:,0],np.tile(latest,(len(ARMS),1)))
    return dict(candidate_parents=out,allocation_indices=np.asarray(indices),forecast_mean=mean,forecast_sigma=sigma,
        constant_radius=spreads[0],conditional_radius=spreads[-1],x=np.asarray(x))


def model_audit(model,rows):
    g=model["held_out_group"]
    if (g in model["fit_groups"] or g in model["calibration_groups"] or set(model["fit_groups"])&set(model["calibration_groups"])
            or model["features"]!=list(FEATURES)):
        raise ValueError("Prediction/calibration source leakage")
    expected=fit_model(rows,g,model["design"])
    numerical={"mean","variance","constant_scale","conditional_quantile","constant_quantile","fingerprint"}
    if set(model)!=set(expected) or any(model[k]!=expected[k] for k in expected.keys()-numerical):
        raise ValueError("Committed forecast inputs or source split changed")
    try:
        for branch in ("mean","variance"):
            if set(model[branch])!=set(expected[branch]):raise ValueError("Forecast coefficient schema changed")
            for name in expected[branch]:np.testing.assert_allclose(model[branch][name],expected[branch][name],rtol=1e-9,atol=1e-12)
        for name in ("constant_scale","conditional_quantile","constant_quantile"):
            np.testing.assert_allclose(model[name],expected[name],rtol=1e-9,atol=1e-12)
    except AssertionError as error:
        raise ValueError("Committed forecast package differs from fixed inputs") from error
