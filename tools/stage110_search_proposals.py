"""Measure learned search proposals on common development histories before policy adoption."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import stage110_proposal_common as p

c,io,s103,s105,e63=p.c,p.io,p.s103,p.s105,p.e63
s109=p.s109
TABLES=("sampling_scope","fit_support","prediction_quality","prediction_summary","state_supply","policy_summary","paired","scope")
METRICS=("best_iou","good_available","q0_iou","gained_good","lost_good","distinct_crops","maximum_coverage","mean_target_side_px","shadow_ppo_iou","shadow_missed_good")
PRED_METRICS=("center_error_latest","center_error_velocity","center_error_learned","log_size_error_latest","log_size_error_learned","covered_constant","covered_uncalibrated","covered_calibrated","conditional_radius_mean")


def runtime_binding(expected,actual,previous=None):
    differences=s109.check_runtime(expected,actual)
    if previous is not None and previous!=actual:
        raise ValueError("Runtime changed on resume; use a new STAGE110_ROOT for this GPU/runtime")
    return differences


def prepare(root103,plan103,load_unit,old):
    jobs,rows,bindings=[],[],{}
    for spec in plan103["units"]:
        entry=load_unit(spec);unit=spec["unit"]
        for mode in s103.MODES[1:]:
            trace=s105.prior.required_trace(root103,plan103,unit,mode)
            context=c.key(unit)+"__"+mode
            sample=p.sample_frames(trace["frame_id"],context,p.DESIGN)
            job=dict(spec=spec,source_arm=mode,context=context,sampling=sample);jobs.append(job)
            for frame in sample["fit_frames"]:
                x,h=p.input_at(trace,entry,frame,old)
                y=p.target(entry["ground_truth"][frame],h[-1])
                if y is not None:
                    rows.append(s109.s106.identity(spec,mode,frame)|dict(x=x.tolist(),y=y.tolist()))
            path=root103/"evaluation"/(context+".npz")
            for q in (path,path.with_suffix(".receipt.json")):bindings[str(q)]=c.file_hash(q)
            print(f"[Stage-110 prepare] {context}: {len(sample['fit_frames'])} fit / {len(sample['eval_frames'])} evaluation identities",flush=True)
    if not rows or not any(j["sampling"]["eval_frames"] for j in jobs):raise ValueError("Insufficient source support")
    return jobs,rows,bindings


def build_models(root,rows,groups,design):
    bank,support,evidence={},{},{}
    for group in groups:
        path=root/"models"/(group+".json")
        if path.exists():
            model=c.c66.sealed(path,p.SCHEMA);p.model_audit(model,rows)
        else:
            model=p.fit_model(rows,group,design);path.parent.mkdir(exist_ok=True);c.publish_identical(path,model)
        bank[group]=model;evidence[path.relative_to(root).as_posix()]=c.file_hash(path)
        support[group]=dict(group=group,fit_groups="|".join(model["fit_groups"]),calibration_groups="|".join(model["calibration_groups"]),
            fit_rows=model["fit_rows"],calibration_rows=model["calibration_rows"],conditional_quantile=model["conditional_quantile"],
            constant_quantile=model["constant_quantile"],held_out_excluded=True,model_fingerprint=model["fingerprint"])
        print(f"[Stage-110 fit] {group}: {model['fit_rows']} fitting / {model['calibration_rows']} calibration rows",flush=True)
    return bank,list(support.values()),evidence


def observe_banks(observer,image,history,frame,ap,parents):
    views={}
    for parent in parents.reshape(-1,4):
        k=tuple(parent)
        if k not in views:views[k]=e63.actor.observe(observer,image,parent,frame,ap)
    lookup=[[views[tuple(parent)] for parent in bank] for bank in parents]
    data=dict(candidate_parents=parents.copy(),candidate_boxes=np.asarray([[v["boxes"] for v in bank] for bank in lookup]),
        features=np.asarray([[e63.actor.feature_vector(v,history,ap) for v in bank] for bank in lookup]),
        image_hw=np.asarray(image.shape[:2]),frame_id=np.array(frame),unique_search_calls=np.array(len(views)))
    for name in ("crop","crop_sha256","layer","block_calls","head_calls","selector_calls","decode_error","raw","size","offset"):
        data[name]=np.asarray([[v[name] for v in bank] for bank in lookup])
    return data


def audit_observations(data,parents,history,frame,hw,ap):
    a=len(p.ARMS)
    np.testing.assert_array_equal(data["candidate_parents"],parents)
    np.testing.assert_array_equal(data["frame_id"],frame);np.testing.assert_array_equal(data["image_hw"],hw)
    e63.source_common.prior.audit_calls(data,(a,4))
    if data["candidate_boxes"].shape!=(a,4,4) or data["features"].shape!=(a,4,24):raise ValueError("Incomplete candidate observation")
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]:raise ValueError("Proposal decode drift")
    unique={tuple(parent) for parent in parents.reshape(-1,4)}
    if int(data["unique_search_calls"])!=len(unique):raise ValueError("Unique proposal budget differs")
    for ai,bank in enumerate(parents):
        for j,parent in enumerate(bank):
            crop,usable,factor=s105.crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
            np.testing.assert_array_equal(data["crop"][ai,j],crop)
            view=dict(parent=parent,boxes=data["candidate_boxes"][ai,j],crop=crop,usable=usable,image_hw=hw,
                **{k:data[k][ai,j] for k in ("raw","size","offset")})
            np.testing.assert_array_equal(data["features"][ai,j],e63.actor.feature_vector(view,history,ap))
            dense=e63.dense_boxes(view["size"],view["offset"],parent,factor,ap["search_size"],hw)
            cell=int(np.argmax(view["raw"]*np.asarray(ap["window"],np.float32)))
            np.testing.assert_allclose(dense[cell],view["boxes"],rtol=0,atol=ap["design"]["box_atol"])
    np.testing.assert_array_equal(data["candidate_boxes"][:,0],np.tile(data["candidate_boxes"][0,0],(a,1)))


def evaluate(prediction,data,entry,frame,ident,model,history,ap,choose):
    truth=np.asarray(entry["ground_truth"][frame],float);y=p.target(truth,history[-1])
    valid=y is not None
    mean,sigma=prediction["forecast_mean"],prediction["forecast_sigma"]
    velocity=np.r_[(history[-1,:2]+history[-1,2:]/2-history[-2,:2]-history[-2,2:]/2)/np.sqrt(history[-1,2:].prod()),
        np.log(history[-1,2:]/history[-2,2:])]
    quality=dict(ident,policy="forecast",valid_gt=valid,
        center_error_latest=float(np.linalg.norm(y[:2])) if valid else None,
        center_error_velocity=float(np.linalg.norm(y[:2]-velocity[:2])) if valid else None,
        center_error_learned=float(np.linalg.norm(y[:2]-mean[:2])) if valid else None,
        log_size_error_latest=float(np.abs(y[2:]).mean()) if valid else None,
        log_size_error_learned=float(np.abs(y[2:]-mean[2:]).mean()) if valid else None,
        covered_constant=float(np.all(np.abs(y-mean)<=prediction["constant_radius"])) if valid else None,
        covered_uncalibrated=float(np.all(np.abs(y-mean)<=sigma)) if valid else None,
        covered_calibrated=float(np.all(np.abs(y-mean)<=prediction["conditional_radius"])) if valid else None,
        conditional_radius_mean=float(prediction["conditional_radius"].mean()),
        forecast_mean="|".join(map(str,mean)),forecast_sigma="|".join(map(str,sigma)))
    iou=s105.tracking_iou(data["candidate_boxes"],truth) if valid else np.full((len(p.ARMS),4),np.nan)
    fixed_good=bool(iou[0].max()>=.5) if valid else False
    supply=[]
    for ai,arm in enumerate(p.ARMS):
        selected=choose(dict(candidate_boxes=data["candidate_boxes"][ai].copy(),candidate_parents=data["candidate_parents"][ai].copy(),
            features=data["features"][ai].copy()),list(history))["selected"]
        crops=data["crop"][ai];coverage=[];sides=[]
        if valid:
            for parent in data["candidate_parents"][ai]:
                _,usable,factor=s105.crop_geometry(parent,data["image_hw"],ap["search_factor"],ap["search_size"])
                intersection=np.maximum(0,np.minimum(usable[:2]+usable[2:],truth[:2]+truth[2:])-np.maximum(usable[:2],truth[:2]))
                coverage.append(float(intersection.prod()/truth[2:].prod()));sides.append(float(np.sqrt(truth[2:].prod())*factor))
        good=bool(iou[ai].max()>=.5) if valid else False
        supply.append(dict(ident,policy=arm,valid_gt=valid,best_iou=float(iou[ai].max()) if valid else None,
            good_available=float(good) if valid else None,q0_iou=float(iou[ai,0]) if valid else None,
            gained_good=float(good and not fixed_good) if valid else None,lost_good=float(fixed_good and not good) if valid else None,
            distinct_crops=len({tuple(r) for r in crops}),maximum_coverage=max(coverage) if valid else None,
            mean_target_side_px=float(np.mean(sides)) if valid else None,shadow_ppo_iou=float(iou[ai,int(selected)]) if valid else None,
            shadow_missed_good=float(good and iou[ai,int(selected)]<.5) if valid else None,shadow_selected=int(selected)))
    return quality,supply


def summaries(supply,quality):
    summary,_=s109.s106.summarize(supply,p.ARMS,METRICS)
    prediction,_=s109.s106.summarize(quality,("forecast",),PRED_METRICS)
    paired=[]
    for cohort in ("all","historical","expanded"):
        for mode in ("all",*s103.MODES[1:]):
            local=[r for r in supply if (cohort=="all" or r["cohort"]==cohort) and (mode=="all" or r["source_arm"]==mode)]
            if not local:continue
            for control in p.ARMS[:-1]:
                if control=="learned_calibrated":continue
                for metric in ("best_iou","good_available","shadow_ppo_iou"):
                    a=s109.s106.macro([r for r in local if r["policy"]=="learned_calibrated"],metric)
                    b=s109.s106.macro([r for r in local if r["policy"]==control],metric)
                    groups=sorted(a.keys()&b.keys());d=np.asarray([a[g]-b[g] for g in groups])
                    paired.append(dict(cohort=cohort,source_arm=mode,contrast="learned_calibrated_minus_"+control,metric=metric,
                        groups=len(d),mean_delta=float(d.mean()) if len(d) else None,median_delta=float(np.median(d)) if len(d) else None,
                        wins=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),
                        minimum_leave_one_group_out=float(min((d.sum()-v)/(len(d)-1) for v in d)) if len(d)>1 else None))
            # Random allocation uses precisely the calibrated pool and the same novelty constraint.
            for metric in ("best_iou","good_available","shadow_ppo_iou"):
                a=s109.s106.macro([r for r in local if r["policy"]=="learned_calibrated"],metric)
                b=s109.s106.macro([r for r in local if r["policy"]=="calibrated_random"],metric)
                groups=sorted(a.keys()&b.keys());d=np.asarray([a[g]-b[g] for g in groups])
                paired.append(dict(cohort=cohort,source_arm=mode,contrast="learned_calibrated_minus_calibrated_random",metric=metric,
                    groups=len(d),mean_delta=float(d.mean()) if len(d) else None,median_delta=float(np.median(d)) if len(d) else None,
                    wins=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),
                    minimum_leave_one_group_out=float(min((d.sum()-v)/(len(d)-1) for v in d)) if len(d)>1 else None))
    return prediction,summary,paired


def run(root,root103,p103,plan,rows,load_unit,choose_for,old,observer):
    if io.seal(dict(rows=rows))["fingerprint"]!=plan["fit_input_fingerprint"]:raise ValueError("Frozen fitting labels/features changed")
    models,fit_support,evidence=build_models(root,rows,plan["groups"],plan["design"])
    inputs=root/"fit_inputs.json";c.publish_identical(inputs,io.seal(dict(schema=p.SCHEMA,rows=rows)))
    evidence["fit_inputs.json"]=c.file_hash(inputs)
    sampling,quality,supply,scope=[],[],[],[];ap=e63.c.actor_plan(old)
    for job in plan["units"]:
        spec,mode=job["spec"],job["source_arm"];unit=spec["unit"];group=s109.s106.source_group(unit["sequence"])
        sampling.append(s109.s106.identity(spec,mode,0)|dict(context=job["context"],actor_seed=unit["seed"],
            **{k:v for k,v in job["sampling"].items() if k not in ("fit_frames","eval_frames")},
            fit_count=len(job["sampling"]["fit_frames"]),eval_count=len(job["sampling"]["eval_frames"]),
            fit_frames="|".join(map(str,job["sampling"]["fit_frames"])),eval_frames="|".join(map(str,job["sampling"]["eval_frames"]))))
        entry,choose=load_unit(spec),choose_for(unit);trace=s105.prior.required_trace(root103,p103,unit,mode)
        with e63.source_experiment.installed(observer,[],unit) as head:
            if head!=plan["native_head_sha256"]:raise ValueError("Original SGLA head changed")
            for frame in job["sampling"]["eval_frames"]:
                ident=s109.s106.identity(spec,mode,frame);x,history=p.input_at(trace,entry,frame,old)
                hw=(entry["frames"][frame]["height"],entry["frames"][frame]["width"])
                stem=job["context"]+f"__f{frame:06d}"
                meta=dict(plan_fingerprint=plan["fingerprint"],unit=unit,source_arm=mode,frame=frame,model_fingerprint=models[group]["fingerprint"])
                computed=p.proposal_banks(history,hw,ap,x,models[group],stem,plan["design"])
                proposal=s109.cache(root,root/"proposals"/(stem+".npz"),dict(meta,kind="causal_search_proposals"),lambda:computed,evidence)
                if set(proposal)!=set(computed):raise ValueError("Proposal schema changed")
                for k in computed:np.testing.assert_array_equal(proposal[k],computed[k])
                def collect():
                    observer.initialize(s105.checked_image(ap["root"],entry["frames"][0]),entry["initial_box"])
                    image=s105.checked_image(ap["root"],entry["frames"][frame])
                    return observe_banks(observer,image,history,frame,ap,proposal["candidate_parents"])
                data=s109.cache(root,root/"observations"/(stem+".npz"),dict(meta,kind="shared_sgla_searches"),collect,evidence)
                audit_observations(data,proposal["candidate_parents"],history,frame,hw,ap)
                error=float(np.abs(data["candidate_boxes"][0]-trace["candidate_boxes"][frame-1]).max())
                if not plan["runtime_differences"]:
                    for k in ("candidate_boxes","features","crop","crop_sha256","layer"):
                        np.testing.assert_array_equal(data[k][0],trace[k][frame-1])
                q,s=evaluate(proposal,data,entry,frame,ident,models[group],history,ap,choose)
                quality.append(q);supply.extend(s)
                scope.append(dict(ident,source_fixed_box_error_px=error,shared_q0_pass=True,
                    unique_search_calls=int(data["unique_search_calls"]),logical_search_calls=4*len(p.ARMS),
                    proposals_saved_before_inference=True,history_changed=False))
                print(f"[Stage-110] {stem}: six equal-budget banks; {int(data['unique_search_calls'])} unique searches",flush=True)
    prediction,summary,paired=summaries(supply,quality)
    publish(root,plan,dict(zip(TABLES,(sampling,fit_support,quality,prediction,supply,summary,paired,scope))),evidence)


def expected(plan):
    n=sum(len(j["sampling"]["eval_frames"]) for j in plan["units"])
    settings={(cohort,mode) for j in plan["units"] if j["sampling"]["eval_frames"]
        for cohort in ("all",j["spec"]["cohort"]) for mode in ("all",j["source_arm"])}
    return dict(sampling_scope=len(plan["units"]),fit_support=len(plan["groups"]),prediction_quality=n,
        prediction_summary=len(settings),state_supply=len(p.ARMS)*n,policy_summary=len(settings)*len(p.ARMS),paired=len(settings)*15,scope=n)


def publish(root,plan,tables,evidence):
    if set(tables)!=set(TABLES) or any(len(tables[k])!=n for k,n in expected(plan).items()) or len(evidence)!=4*expected(plan)["scope"]+len(plan["groups"])+1:
        raise ValueError("Incomplete learned-proposal experiment")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name in TABLES:
        path=out/f"stage110_{name}.csv";s103.s86.publish_table(path,tables[name]);hashes[path.name]=c.file_hash(path)
    primary=next(r for r in tables["paired"] if r["cohort"]==r["source_arm"]=="all" and r["contrast"]==p.DESIGN["primary"] and r["metric"]=="best_iou")
    calls=sum(r["unique_search_calls"] for r in tables["scope"])
    c.publish_identical(out/"stage110_decision.json",io.seal(dict(schema=p.SCHEMA,status=p.STATUS,plan_fingerprint=plan["fingerprint"],
        role="consumed_development_group_isolated_learned_search_supply",source_stage109_fingerprint=p.SOURCE109,
        runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],counts={n:len(v) for n,v in tables.items()},primary=primary,
        new_prediction_packages=len(plan["groups"]),sgla_training_updates=0,ppo_training_updates=0,new_full_trajectories=0,
        confirmation_inference_calls=0,new_search_calls=calls,new_block_calls=7*calls,logical_search_calls=24*len(tables["scope"]),
        maximum_fixed_source_box_error_px=max(r["source_fixed_box_error_px"] for r in tables["scope"]),
        held_out_excluded=True,calibration_groups_disjoint=True,shared_q0_pass=True,past_only_proposals=True,
        table_sha256=hashes,artifact_sha256=evidence,warnings=[
            "All 28 source groups are consumed development; group isolation is not independent method confirmation.",
            "Fixed source histories and common images; proposals never update future state. No new full-video AUC.",
            "Conditional mean/scale are learned; Gaussian scenario shape, coverage proxy and retained q0 remain declared assumptions.",
            "90% group-balanced empirical interval calibration is not a finite-sample coverage guarantee under temporal dependence.",
            "Frozen PPO outputs are shadow compatibility diagnostics; no matched PPO retraining in this supply experiment.",
            "Four is a fixed SGLA-call budget, not an optimal candidate-count claim; candidate positions and sizes are generated."])))
    check_results(root);print(f"Stage-110: {p.STATUS}; {len(tables['scope'])} states, {calls} unique searches",flush=True)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",p.SCHEMA);r=c.c66.sealed(root/"analysis/stage110_decision.json",p.SCHEMA)
    for job in plan["units"]:
        if job["sampling"]!=p.sample_frames(np.arange(1,job["source_states"]+1),job["context"],plan["design"]):raise ValueError("Sampling changed")
    counts=expected(plan);n=counts["scope"]
    if (plan["design"]!=p.DESIGN or r["status"]!=p.STATUS or r["plan_fingerprint"]!=plan["fingerprint"]
            or r["source_stage109_fingerprint"]!=p.SOURCE109 or r["runtime"]!=plan["runtime"] or r["runtime_differences"]!=plan["runtime_differences"]
            or any(r["counts"][k]!=v for k,v in counts.items()) or r["new_prediction_packages"]!=len(plan["groups"])
            or r["new_block_calls"]!=7*r["new_search_calls"] or not n<=r["new_search_calls"]<=19*n or r["logical_search_calls"]!=24*n
            or len(r["artifact_sha256"])!=4*n+len(plan["groups"])+1 or set(r["table_sha256"])!={f"stage110_{k}.csv" for k in TABLES}
            or (not plan["runtime_differences"] and r["maximum_fixed_source_box_error_px"]!=0.)
            or not all(r[k] for k in ("held_out_excluded","calibration_groups_disjoint","shared_q0_pass","past_only_proposals"))
            or any(r[k]!=0 for k in ("sgla_training_updates","ppo_training_updates","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-110 result")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n):h for n,h in r["table_sha256"].items()});io.verify({str(root/n):h for n,h in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    names=("stage76","stage89","stage93","stage103","stage109")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--checkpoint");parser.add_argument("--parameter",default="deit_distilled");parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true");parser.set_defaults(mode="stage110")
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not all(getattr(args,n) for n in names) or not args.checkpoint:parser.error("Supply Stage-76/89/93/103/109 and original checkpoint")
    r76,r89,r93,r103,r109=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==x or root in x.parents or x in root.parents for x in sources):parser.error("Use a separate Stage-110 directory")
    with s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage110_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(x.name!="collect.lock" for x in root.iterdir()):raise ValueError("Use an empty Stage-110 directory")
        s109.check_results(r109);s103.check_results(r103)
        p109=c.c66.sealed(r109/"manifest.json",s109.SCHEMA);report109=c.c66.sealed(r109/"analysis/stage109_decision.json",s109.SCHEMA)
        p103=c.c66.sealed(r103/"manifest.json",s103.SCHEMA);report103=c.c66.sealed(r103/"analysis/stage103_decision.json",s103.SCHEMA)
        if report109["fingerprint"]!=p.SOURCE109 or report103["fingerprint"]!=s105.prior.SOURCE103 or len(p103["units"])!=40:raise ValueError("Need the fixed complete development sources")
        p93,_,p89,report89,source=s103.prior.s95.s94.load_run(r93,r76,r89)
        actor=io.load_source(Path(source["plan"]["source_input"]))
        if p103["units"]!=p89["units"] or actor["plan"]!=source["original"] or actor["models"]["fingerprint"]!=source["report"]["source_training_fingerprint"]:raise ValueError("PPO/source units changed")
        def load_unit(spec):return s103.prior.s95.load_unit(r93,r89,p93,p89,report89,source,spec)[0]
        def choose_for(unit):
            pack=s103.e74.learned_pack(actor["models"],unit,actor["prior"])
            return lambda row,history:s103.e74.choose(row,history,unit,source["original"],actor["prior"],pack)
        jobs,rows,bindings=prepare(r103,p103,load_unit,source["original"])
        for job in jobs:
            trace=s105.prior.required_trace(r103,p103,job["spec"]["unit"],job["source_arm"]);job["source_states"]=len(trace["frame_id"])
        groups=sorted({s109.s106.source_group(j["spec"]["unit"]["sequence"]) for j in jobs})
        if len(groups)!=28:raise ValueError("Development source group coverage changed")
        import torch
        torch.set_num_threads(1)
        runtime=s103.s76.device_runtime(source["original"],torch,args.gpu)
        previous=c.c66.sealed(root/"manifest.json",p.SCHEMA) if (root/"manifest.json").exists() else None
        differences=runtime_binding(report103["runtime"],runtime,previous["runtime"] if previous else None)
        bindings.update(p109["source_sha256"]);bindings.update(actor["bindings"])
        for base,number in ((r103,103),(r109,109)):
            for q in (base/"manifest.json",base/f"analysis/stage{number}_decision.json"):bindings[str(q)]=c.file_hash(q)
        code=[Path(__file__).resolve(),Path(p.__file__).resolve(),Path(p.motion.__file__).resolve(),Path(__file__).with_name("run_stage110_proposals.sh").resolve()]
        plan=io.seal(dict(schema=p.SCHEMA,design=p.DESIGN,units=jobs,groups=groups,runtime=runtime,runtime_differences=differences,
            native_head_sha256=source["report"]["native_head_sha256"],fit_input_fingerprint=io.seal(dict(rows=rows))["fingerprint"],
            source_sha256=bindings,code_sha256=p109["code_sha256"]|p103["code_sha256"]|{str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json",plan)
        n=sum(len(j["sampling"]["eval_frames"]) for j in jobs)
        print(f"Stage-110 frozen: {n} common states, six four-search banks; at most {19*n} unique searches",flush=True)
        frozen=dict(source["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s103.s76.startup.checked_runtime(args,frozen,root);observer=e63.Capture(native)
        try:run(root,r103,p103,plan,rows,load_unit,choose_for,source["original"],observer)
        finally:observer.handle.remove()


if __name__=="__main__":main()
