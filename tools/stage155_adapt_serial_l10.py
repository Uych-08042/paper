"""Fit an isolated L10 copy on frozen L9 features and evaluate paired continuations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import stage154_forced_depth_continuation as prior
from stage18_environment import executed_crop_roi
from stage18_tracking_loss import _center_heatmap, tracking_loss
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, source_group, tracking_iou
from stage28_response_common import cpu_array
from stage46_shift_audit import publish_table
from stage48_stability_experiment import restore_latest, save_step
from stage48_stability_runtime import clone_without_hooks, digest


base, c, io = prior.base, prior.c, prior.io
source = prior.prior.source
SCHEMA = "stage155_isolated_l10_adaptation_v1"
STATUS = "COMPLETE_ISOLATED_L10_ADAPTATION_DIAGNOSTIC"
SOURCE154 = "8f35d1f3d10abf243ee27056e25dbf0d3f4395650fd66733a9ed821247bb2bb2"
SOURCES = ("stage76", "stage89", "stage93", "stage121", "stage134", "stage154")
ARMS = ("native", "frozen_l10", "adapted_l10")
CONTRASTS = ((0, 1), (0, 2), (1, 2))
DESIGN = dict(folds=2, seed=23155, videos=28, tracked_frames=29047, uniform_views=32, difficult_views=16,
              difficult_iou=.5, updates=192, batch=8, save_every=64,
              learning_rate=1e-5, weight_decay=1e-4, gradient_clip=1., continuation=prior.DESIGN)
TABLES = ("training", "fit_support", "summary", "anchors", "windows")
WORK = dict(new_full_trajectories=0, confirmation_inference_calls=0)


def split_sources(sequences):
    groups = sorted({source_group(seq) for seq in sequences})
    if len(groups) != DESIGN["videos"] or len(sequences) != len(groups):
        raise ValueError("Expected 28 distinct development source groups")
    rng = np.random.default_rng(DESIGN["seed"])
    permuted = [groups[i] for i in rng.permutation(len(groups))]
    return [dict(fold=i, fit=sorted(set(groups)-set(held)), held=sorted(map(str, held)))
            for i, held in enumerate(np.array_split(permuted, DESIGN["folds"]))]


def sampled_views(entry, trace, ap):
    """Sample visible labels on the recorded native search state."""
    truths = np.asarray(entry["ground_truth"], float)[1:]
    n = len(truths)
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, n+1))
    iou = tracking_iou(trace["boxes"], truths)
    targets = {}
    for i in range(n):
        if iou[i] < 0. or trace["layer"][i, 0] != 9:
            continue
        parent = trace["candidate_parents"][i, 0]
        _, usable, factor = crop_geometry(parent, trace["image_hw"][i], ap["search_factor"], ap["search_size"])
        center = truths[i, :2]+.5*truths[i, 2:]
        if not ((center >= usable[:2]) & (center < usable[:2]+usable[2:])).all():
            continue
        roi = executed_crop_roi(truths[i], parent, factor, ap["search_size"])
        center_in_crop = roi[:2]+.5*roi[2:]
        if ((center_in_crop >= 0.) & (center_in_crop < 1.)).all():
            targets[i+1] = roi
    uniform = set(evenly_spaced(list(targets), DESIGN["uniform_views"]))
    difficult = [f for f in targets if f not in uniform and iou[f-1] < DESIGN["difficult_iou"]]
    extra = set(evenly_spaced(difficult, DESIGN["difficult_views"]))
    return [(frame, targets[frame], frame in extra) for frame in sorted(uniform|extra)]


def evenly_spaced(values, count):
    if len(values) <= count:
        return values
    return [values[i] for i in np.linspace(0, len(values)-1, count, dtype=int)]


def collect_features(observer, entry, trace, views, ap, exact_runtime):
    inputs = []

    def capture_l9(module, args, value):
        inputs.append(cpu_array(value)[0])

    handle = observer.blocks[8].register_forward_hook(capture_l9)
    try:
        frames = entry["frames"]
        observer.initialize(checked_image(ap["root"], frames[0]), entry["initial_box"], "native")
        for frame, _, _ in views:
            idx = frame-1
            parent = trace["candidate_parents"][idx, 0]
            row = observer.step(checked_image(ap["root"], frames[frame]), parent, frame, capture_crop=True)
            if row["selected"] != 9 or row["added"] or row["crop_sha256"] != trace["crop_sha256"][idx, 0]:
                raise ValueError("L9 feature collection diverged from the recorded crop or native route")
            if exact_runtime:
                np.testing.assert_allclose(row["boxes"], trace["boxes"][idx], atol=ap["design"]["box_atol"], rtol=0)
    finally:
        handle.remove()
    return dict(frame_id=np.array([f for f,_,_ in views], int), targets=np.stack([roi for _,roi,_ in views]),
                difficult=np.array([hard for _,_,hard in views], bool), tokens=np.stack(inputs))


def feature_bank(root, r121, p121, d121, observer, ap, jobs, entries, plan):
    bank, hashes, rows = {}, {}, []
    source_runtime_matches = not plan["stage121_runtime_differences"]
    (root/"features").mkdir(exist_ok=True)
    for seq, job in jobs.items():
        entry = entries[seq]
        path = root/"features"/f"{seq}.npz"
        key = f"evaluation/{base.job_key(job)}"
        io.verify({str(r121/k):d121["artifact_sha256"][k] for k in (key+".npz", key+".receipt.json")})
        trace = source.prior.prior.load_trace(r121,p121,job)
        views = sampled_views(entry, trace, ap)
        if not views:
            # A source with no supervised views contributes no data to either fit.
            rows.append(dict(sequence=seq, source=source_group(seq), views=0, extra_low_iou_views=0))
            continue
        metadata = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, source_trace=d121["artifact_sha256"][key+".npz"])
        data = c.read_cache(path, metadata)
        fresh = data is None
        if fresh:
            data = collect_features(observer,entry,trace,views,ap,source_runtime_matches)
        np.testing.assert_array_equal(data["frame_id"], [f for f,_,_ in views])
        np.testing.assert_array_equal(data["targets"], np.stack([roi for _,roi,_ in views]))
        np.testing.assert_array_equal(data["difficult"], [hard for _,_,hard in views])
        rt = observer.rt
        if (data["tokens"].shape != (len(views), rt.lens_x+rt.lens_z, rt.identity["embed_dim"])
                or not np.isfinite(data["tokens"]).all()):
            raise ValueError("Invalid frozen L9 tokens")
        if fresh:
            c.save(path,data,metadata)
        rows.append(dict(sequence=seq, source=source_group(seq), views=len(views), extra_low_iou_views=int(data["difficult"].sum())))
        for item in (path,path.with_suffix(".receipt.json")):
            hashes[item.relative_to(root).as_posix()] = c.file_hash(item)
        bank[source_group(seq)] = dict(data, feature_sha256=hashes[path.relative_to(root).as_posix()])
        print(f"[Stage-155] {seq}: {len(views)} supervised views",flush=True)
    observer.native.assert_frozen()
    return bank, rows, hashes


def isolated_l10(rt):
    original = rt.backbone.blocks[9]
    block = clone_without_hooks(original).eval().requires_grad_(True)
    if digest(block) != digest(original) or any(p.requires_grad for p in rt.net.parameters()):
        raise ValueError("Stage-155 requires a frozen native model and an exact L10 copy")
    return block


def train_fold(observer, plan, fold, bank, root):
    rt, torch = observer.rt, observer.rt.torch
    l10 = isolated_l10(rt)
    before = digest(l10)
    buffers = digest(l10, buffers_only=True)
    params = list(l10.parameters())
    opt = torch.optim.AdamW(params,lr=DESIGN["learning_rate"],weight_decay=DESIGN["weight_decay"])
    fit = [g for g in fold["fit"] if g in bank]
    if not fit or set(fit)&set(fold["held"]):
        raise ValueError("Empty fit pool or held-out source leaked into training")
    directory = root/"training"/f"fold{fold['fold']}"
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], fit=fit, held=fold["held"],
                    initial_sha256=before, updates=DESIGN["updates"],
                    fit_feature_sha256={g:bank[g]["feature_sha256"] for g in fit})
    saved = restore_latest(torch,directory,metadata)
    start, log = 0, []
    if saved:
        l10.load_state_dict(saved["l10"],strict=True)
        opt.load_state_dict(saved["optimizer"])
        start,log = saved["step"],saved["history"]
    rng = np.random.default_rng(DESIGN["seed"]+fold["fold"])
    draws = []
    for _ in range(DESIGN["updates"]):
        groups = [fit[i] for i in rng.integers(len(fit),size=DESIGN["batch"])]
        draws.append([(g,int(rng.integers(len(bank[g]["targets"])))) for g in groups])
    device, dtype = params[0].device, params[0].dtype
    side = rt.search_side
    for step in range(start+1, DESIGN["updates"]+1):
        pairs = draws[step-1]
        tokens = torch.as_tensor(np.stack([bank[g]["tokens"][i] for g,i in pairs]),device=device,dtype=dtype)
        targets = torch.as_tensor(np.stack([bank[g]["targets"][i] for g,i in pairs]),device=device,dtype=dtype)
        opt.zero_grad(set_to_none=True)
        # Train the regression maps at the GT heatmap peak, as in the CENTER head.
        gt_score, _ = _center_heatmap(torch,targets,side,side)
        encoded = l10(tokens)
        encoded = rt.s1.recover_tokens(encoded,rt.lens_z,rt.lens_x,mode=rt.backbone.cat_mode)
        output = rt.net.forward_head(rt.backbone.norm(encoded),gt_score[:,0])
        loss, _ = tracking_loss(torch,output,targets)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(params,DESIGN["gradient_clip"],error_if_nonfinite=True))
        value = float(loss.detach())
        if not np.isfinite(value) or any(p.grad is not None for p in rt.net.parameters()):
            raise ValueError("Invalid loss or gradient on an original SGLA parameter")
        opt.step()
        log.append(dict(step=step, loss=value, gradient_norm=grad))
        if step % DESIGN["save_every"] == 0 or step == DESIGN["updates"]:
            observer.native.assert_frozen()
            if digest(l10,buffers_only=True) != buffers:
                raise ValueError("The adapted L10 buffers changed")
            save_step(torch,directory,dict(metadata=metadata,step=step,l10={k:v.detach().cpu().clone()
                      for k,v in l10.state_dict().items()},optimizer=opt.state_dict(),history=log))
            print(f"[Stage-155] fold {fold['fold']} {step}/{DESIGN['updates']} loss={value:.4f}",flush=True)
    after = digest(l10)
    observer.native.assert_frozen()
    if before == after or not any(row["gradient_norm"] > 0. for row in log) or digest(l10,buffers_only=True) != buffers:
        raise ValueError("Fit did not update isolated L10 parameters")
    checkpoint = directory/f"step{DESIGN['updates']:06d}.pt"
    hashes = {p.relative_to(root).as_posix():c.file_hash(p) for p in (checkpoint,checkpoint.with_suffix(".receipt.json"))}
    scope = dict(fold=fold["fold"], fit="|".join(fit), held="|".join(fold["held"]), trainable_parameters=sum(p.numel() for p in params),
                 fit_views=sum(len(bank[g]["targets"]) for g in fit), updates=DESIGN["updates"], first_loss=log[0]["loss"],
                 final_loss=log[-1]["loss"], max_grad=max(r["gradient_norm"] for r in log),
                 original_l10_sha256=before, adapted_l10_sha256=after)
    return l10.requires_grad_(False), scope, hashes


def collect(observer, entry, case, parent, l10, dataset_root, reader=checked_image):
    frames = entry["frames"]
    first = case["frame"]
    observer.initialize(reader(dataset_root,frames[0]),entry["initial_box"],"native")
    image = reader(dataset_root,frames[first])
    paths = [[observer.step(image,parent.copy(),first,force_extra=i>0,capture_crop=True,
                            extra_l10=l10 if i==2 else None)] for i in range(len(ARMS))]
    ids = np.arange(first,first+case["successors"]+1)
    for frame in ids[1:]:
        image = reader(dataset_root,frames[frame])
        for path in paths:
            path.append(observer.step(image,path[-1]["boxes"].copy(),int(frame)))
    data = {k:np.asarray([[row[k] for row in path] for path in paths]) for k in prior.FIELDS}
    return dict(data,frame_id=ids,anchor_crop_sha256=np.array([p[0]["crop_sha256"] for p in paths]))


def pair(data, ref, adapted):
    return {k:(v if k=="frame_id" else v[[ref,adapted]]) for k,v in data.items()}


def evaluate(root, r134, p134, r154, p154, plan, observer, ap, entries, models, stats, hashes):
    anchors, windows = defaultdict(list), defaultdict(list)
    (root/"branches").mkdir(exist_ok=True)
    same_anchor_runtime = not plan["stage134_runtime_differences"]
    same_previous_runtime = not plan["stage154_runtime_differences"]
    held = {g:f["fold"] for f in plan["folds"] for g in f["held"]}
    for case in plan["cases"]:
        cid, seq = case["case_id"],case["sequence"]
        entry = entries[seq]
        name = f"branches/{cid}.npz"
        ref = c.read_cache(r134/name,dict(plan_fingerprint=p134["fingerprint"],case_id=cid))
        prev = c.read_cache(r154/name,dict(plan_fingerprint=p154["fingerprint"],case_id=cid))
        if ref is None or prev is None:
            raise ValueError(f"Missing paired source for {cid}")
        fold = held[source_group(seq)]
        l10 = models[fold]
        model_sha = stats[fold]["adapted_l10_sha256"]
        meta = dict(plan_fingerprint=plan["fingerprint"],case_id=cid,fold=fold,adapted_l10_sha256=model_sha)
        path = root/name
        data = c.read_cache(path,meta)
        fresh = data is None
        if fresh:
            hook = l10.register_forward_hook(observer.native._block(9))
            try:
                data = collect(observer,entry,case,ref["parent"][0,0],l10,ap["root"])
            finally:
                hook.remove()
        np.testing.assert_array_equal(data["selected"][:,0],[9,9,9])
        np.testing.assert_array_equal(data["added"][:,0],[0,10,10])
        for k in (1,2):
            prior.audit(pair(data,0,k),case,ref,same_anchor_runtime,ap["design"]["box_atol"])
        replay_error = float(np.abs(data["boxes"][:2]-prev["boxes"]).max())
        if same_previous_runtime:
            np.testing.assert_allclose(data["boxes"][:2],prev["boxes"],rtol=0,atol=ap["design"]["box_atol"])
            np.testing.assert_array_equal(data["selected"][:2],prev["selected"])
        if fresh:
            c.save(path,data,meta)
        for p in (path,path.with_suffix(".receipt.json")):
            hashes[p.relative_to(root).as_posix()] = c.file_hash(p)
        for ref_index,treat_index in CONTRASTS:
            label = f"{ARMS[treat_index]}_minus_{ARMS[ref_index]}"
            anchor, w, _ = prior.describe(pair(data,ref_index,treat_index),case,entry["ground_truth"],replay_error)
            anchors[label].append(dict(anchor,contrast=label,fold=fold))
            windows[label].extend(dict(row,contrast=label,fold=fold) for row in w)
    observer.native.assert_frozen()
    for fold,l10 in models.items():
        if digest(l10) != stats[fold]["adapted_l10_sha256"]:
            raise ValueError("Evaluation changed the fitted L10")
    summary = [dict(r,contrast=label) for label in anchors for r in prior.summarize(anchors[label],windows[label])]
    # Stage-154's comparison keys are reused internally. Name the two sides by their roles in the exported contrast.
    arows, wrows = [], []
    for label in anchors:
        for r in anchors[label]:
            arows.append({{"native_iou":"reference_iou","serial_iou":"treatment_iou",
                           "native_source_max_box_error":"stage154_control_max_box_error"}.get(k,k):v for k,v in r.items()})
        for r in windows[label]:
            wrows.append({k.replace("native_","reference_").replace("serial_extra_","treatment_"):v for k,v in r.items()})
    return summary,arows,wrows


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"stage155_decision.json",SCHEMA)
    cases = sum(prior.roster.GROUPS.values())
    searches = len(ARMS)*cases*(prior.DESIGN["horizons"][-1]+1)
    if (plan["design"] != DESIGN or plan["source_stage154"] != SOURCE154 or report["status"] != STATUS
            or report["plan_fingerprint"] != plan["fingerprint"] or report["cases"] != cases
            or report["training_updates"] != DESIGN["folds"]*DESIGN["updates"]
            or report["evaluation_searches"] != searches or report["extra_blocks"] != 2*cases
            or any(report[k] != v for k,v in WORK.items())
            or set(report["table_sha256"]) != {f"stage155_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-155 comparison")
    io.verify({str(root/k):v for k,v in (report["table_sha256"]|report["artifact_sha256"]).items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage155_adapt_l10")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args,k) is None for k in SOURCES):
        parser.error("Supply all frozen source directories and the original SGLA checkpoint")
    paths = [getattr(args,k).expanduser().resolve() for k in SOURCES]
    for name,path in zip(SOURCES,paths):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
        if root == path or root in path.parents or path in root.parents:
            parser.error("Use a separate Stage-155 directory")
    r76,r89,r93,r121,r134,r154 = paths
    with writer_lock(root):
        if (root/"stage155_decision.json").is_file():
            check_results(root)
            return
        prior.prior.check_results(r134)
        prior.check_results(r154)
        p134 = c.c66.sealed(r134/"manifest.json",prior.prior.SCHEMA)
        d134 = c.c66.sealed(r134/"stage134_decision.json",prior.prior.SCHEMA)
        p154 = c.c66.sealed(r154/"manifest.json",prior.SCHEMA)
        d154 = c.c66.sealed(r154/"stage154_decision.json",prior.SCHEMA)
        p121 = c.c66.sealed(r121/"manifest.json",source.s121.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json",source.s121.SCHEMA)
        if (d154["fingerprint"] != SOURCE154 or p154["source_stage134"] != d134["fingerprint"]
                or d121["fingerprint"] != source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]):
            raise ValueError("Need sealed Stage-121, Stage-134 and Stage-154 source results")
        jobs = {j["spec"]["unit"]["sequence"]:j for j in p121["jobs"] if j["arm"]=="native"}
        if sum(j["frames"] for j in jobs.values()) != DESIGN["tracked_frames"]:
            raise ValueError("Stage-121 development frame count changed")
        folds = split_sources(jobs)
        cases = p154["cases"]
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]
        ap = base.e63.c.actor_plan(old)
        entries = {seq:s95.load_unit(r93,r89,p93,p89,d89,origin,j["spec"])[0] for seq,j in jobs.items()}
        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        diffs = {stage:base.s109.check_runtime(sealed["runtime"],runtime)
                 for stage,sealed in (("stage121",p121),("stage134",p134),("stage154",p154))}
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume; set STAGE155_ROOT to a new directory")
        script = Path(__file__).resolve()
        code = (script, Path(prior.__file__), script.with_name("stage153_adaptive_depth.py"),
                script.parents[1]/"tracking"/"stage48_stability_runtime.py")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage154=SOURCE154, source_stage121=source.SOURCE121,
                            arms=list(ARMS), folds=folds, cases=cases, runtime=runtime,
                            code_sha256={p.name:c.file_hash(p) for p in code},
                            **{f"{k}_runtime_differences":v for k,v in diffs.items()}))
        c.publish_identical(manifest,plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        with ExitStack() as stack:
            observer = prior.DepthObserver(native,stack)
            bank,support,hashes = feature_bank(root,r121,p121,d121,observer,ap,jobs,entries,plan)
            models,stats = {},{}
            for fold in folds:
                i = fold["fold"]
                models[i],stats[i],files = train_fold(observer,plan,fold,bank,root)
                hashes.update(files)
            summary,anchors,windows = evaluate(root,r134,p134,r154,p154,plan,observer,ap,entries,models,stats,hashes)
        tables = {}
        for name,rows in zip(TABLES,(list(stats.values()),support,summary,anchors,windows)):
            path = root/f"stage155_{name}.csv"
            publish_table(path,rows)
            tables[path.name] = c.file_hash(path)
        c.publish_identical(root/"stage155_decision.json",io.seal(dict(
            schema=SCHEMA,status=STATUS,role="consumed_development_held_out_l10_only",
            plan_fingerprint=plan["fingerprint"],source_stage154=SOURCE154,runtime=runtime,runtime_differences=diffs,
            cases=len(cases),extra_blocks=2*len(cases),training_updates=sum(r["updates"] for r in stats.values()),
            feature_searches=sum(r["views"] for r in support),
            evaluation_searches=len(ARMS)*sum(r["successors"]+1 for r in cases),
            table_sha256=tables,artifact_sha256=hashes,**WORK)))
        check_results(root)
    print(f"Stage-155: {STATUS}",flush=True)


if __name__ == "__main__":
    main()
