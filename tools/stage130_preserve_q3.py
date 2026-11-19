"""Reassemble Stage-129 observations into four-slot banks that preserve q3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage129_coarse_to_fine as source
from stage27_uav_common import crop_geometry, tracking_iou

prior, base, c, io = source.prior, source.base, source.c, source.io
SCHEMA = "stage130_preserve_q3_single_zoom_v1"
STATUS = "COMPLETE_CACHED_Q3_PRESERVATION"
SOURCE129 = "e4a3cd5cb570ae66e123a8c6e49524aab44a73f8b196690304f832f7d29d1ff1"
DESIGN = dict(points=346, good=.5, bootstrap_seed=23130, bootstrap_samples=20000)
ARMS = ("trim", "two_zoom", "rank1", "rank2", "coverage_vote")
TABLES = ("points", "summary", "paired")
WORK = dict(new_search_calls=0, image_reads=0, training_updates=0, new_full_trajectories=0)


def replacement_slot(wide):
    retained = {0,3,wide}
    if len(retained)==2:
        retained.add(1)
    return next(q for q in range(4) if q not in retained)


def candidate_votes(control, zoom_parents, hw, ap):
    boxes = control["candidate_boxes"][[0,3]]
    centers = boxes[:,:2]+boxes[:,2:]/2
    votes = []
    for parent in zoom_parents:
        _,usable,_ = crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
        inside = ((centers>=usable[:2]) & (centers<usable[:2]+usable[2:])).all(axis=1)
        votes.append(int(inside.sum()))
    return votes


def point_key(row):
    return row["sequence"],int(row["source_seed"]),int(row["frame"])


def finite_choices(control, zoom, wide, ordered, hw, ap):
    if not ordered:
        return {arm:(control,None,[]) for arm in ARMS[2:]}
    target = replacement_slot(wide)
    parents = zoom["candidate_parents"][ordered]
    votes = candidate_votes(control,parents,hw,ap)
    choices = (0,min(1,len(ordered)-1),int(np.argmax(votes)))
    banks = {}
    for arm,j in zip(ARMS[2:],choices):
        q = ordered[j]
        bank = {k:control[k].copy() for k in ("candidate_parents","candidate_boxes","features")}
        for k in bank:
            bank[k][target] = zoom[k][q]
            np.testing.assert_array_equal(bank[k][3],control[k][3])
        banks[arm] = bank,j+1,votes
    return banks


def summarize(points):
    summary,paired = [],[]
    for cohort, subset in (("all",points),("active",[p for p in points if p["active"]]),
                           ("small_covered",[p for p in points if p["small_covered"]])):
        by_arm = {arm:{point_key(p):p for p in subset if p["arm"]==arm} for arm in ARMS}
        baseline = by_arm["trim"]
        for arm in ARMS:
            rows = by_arm[arm]
            if set(rows)!=set(baseline):
                raise ValueError("Matched frame inventory differs by arm")
            summary.append(dict(population=cohort,arm=arm,points=len(rows),videos=len({k[0] for k in rows}),
                good_bank=sum(p["best_iou"]>=DESIGN["good"] for p in rows.values()),
                new_good=sum(p["best_iou"]>=DESIGN["good"]>baseline[k]["best_iou"] for k,p in rows.items()),
                lost_good=sum(baseline[k]["best_iou"]>=DESIGN["good"]>p["best_iou"] for k,p in rows.items()),
                selected_good=sum(p["selected_iou"]>=DESIGN["good"] for p in rows.values()),
                lost_selected_good=sum(baseline[k]["selected_iou"]>=DESIGN["good"]>p["selected_iou"] for k,p in rows.items()),
                lost_selected_q3=sum(baseline[k]["selected_slot"]==3 and baseline[k]["selected_iou"]>=DESIGN["good"]>p["selected_iou"]
                                     for k,p in rows.items()),
                second_rank=sum(p["zoom_rank"]==2 for p in rows.values())))
            if arm=="trim":
                continue
            for metric in ("best_iou","selected_iou"):
                units, videos = defaultdict(list),defaultdict(list)
                for k,p in rows.items():
                    units[k[:2]].append(p[metric]-baseline[k][metric])
                for (seq,_),values in sorted(units.items()):
                    videos[seq].append(float(np.mean(values)))
                if videos:
                    paired.append(dict(population=cohort,arm=arm,**base.s103.contrast(
                        [float(np.mean(v)) for _,v in sorted(videos.items())],arm+"_minus_trim",metric,DESIGN)))
    return summary,paired


def extract_archive(path):
    parts = re.split(rb"^===== stage129_([\w.]+) =====\r?\n",path.read_bytes(),flags=re.M)
    files = {k.decode():v.rstrip(b"\r\n") for k,v in zip(parts[1::2],parts[2::2])}
    decision = json.loads(files["decision.json"])
    if decision["fingerprint"]!=SOURCE129 or io.seal({k:v for k,v in decision.items() if k!="fingerprint"})!=decision:
        raise ValueError("Archived Stage-129 report changed")
    tables = {}
    for name in ("points.csv","crops.csv"):
        data = files[name]+b"\r\n"
        if hashlib.sha256(data).hexdigest()!=decision["table_sha256"][f"stage129_{name}"]:
            raise ValueError(f"Archived {name} hash differs")
        tables[name] = list(csv.DictReader(data.decode().splitlines()))
    return tables["points.csv"],tables["crops.csv"]


def archive_support(points, crops):
    """Bound support for each actual cached query bank. No re-scoring from CSV."""
    by_frame = defaultdict(dict)
    for r in crops:
        by_frame[point_key(r)][r["arm"],int(r["slot"])] = r
    values = []
    for p in points:
        rows = by_frame[point_key(p)]
        trim = [float(rows["trim",q]["iou"]) for q in range(4)]
        ranks = [float(rows["zoom",q]["iou"]) for q in range(4) if rows["zoom",q]["local_search"]=="True"]
        keep = trim.copy()
        if ranks:
            keep.pop(replacement_slot(int(p["wide_slot"])))
        values.append(dict(active=bool(ranks),small=p["source_small_covered"]=="True",trim=max(trim),
                           rank1=max([*keep,*ranks[:1]]),rank2=max([*keep,*ranks[-1:]]),
                           either=max([*keep,*ranks])))
    for name, rows in (("all",values),("active",[r for r in values if r["active"]]),
                       ("small_covered",[r for r in values if r["small"]])):
        for arm in ("rank1","rank2","either"):
            yield dict(population=name,arm=arm,points=len(rows),
                good_bank=sum(r[arm]>=DESIGN["good"] for r in rows),
                new_good=sum(r[arm]>=DESIGN["good"]>r["trim"] for r in rows),
                lost_good=sum(r["trim"]>=DESIGN["good"]>r[arm] for r in rows))


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"stage130_decision.json",SCHEMA)
    if (plan["design"]!=DESIGN or report["plan_fingerprint"]!=plan["fingerprint"] or report["status"]!=STATUS
            or report["source_stage129_fingerprint"]!=SOURCE129 or report["row_counts"]["points"]!=346*len(ARMS)
            or set(report["table_sha256"])!={f"stage130_{n}.csv" for n in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-130 result")
    io.verify({str(root/k):v for k,v in report["table_sha256"].items()})


def run(root,r121,p121,r129,p129,plan,models,load_entry,ap):
    results = prior.source.read_table(r129/"stage129_points.csv")
    jobs = prior.reference_jobs(p121)
    groups = defaultdict(list)
    for r in results:
        groups[point_key(r)[:2]].append(r)
    points = []
    for (seq,seed), records in sorted(groups.items()):
        job = jobs[seq,seed]
        entry = load_entry(job["spec"])
        trace = prior.prior.load_trace(r121,p121,job)
        _,scorer = base.job_models(dict(job,arm=prior.prior.CONTROL),models)
        for p in records:
            f = int(p["frame"])
            i = f-1
            if int(trace["frame_id"][i])!=f or not trace["alternate_used"][i]:
                raise ValueError("Source frame is no longer a recorded trigger")
            ident = dict(sequence=seq,source_seed=seed,frame=f)
            stem = f"{seq}__s{seed}__f{f:06d}"
            meta = dict(ident,plan_fingerprint=p129["fingerprint"])
            control = c.read_cache(r129/"frames"/f"{stem}__trim.npz",dict(meta,arm="trim"))
            if control is None:
                raise ValueError("Missing Stage-129 trim observation")
            np.testing.assert_equal(control["frame_id"],f)
            np.testing.assert_array_equal(control["image_hw"],trace["image_hw"][i])
            hw = control["image_hw"]
            original = trace["candidate_parents"][i]
            expect,wide,peaks,_ = source.zoom_layout(control["candidate_parents"],original,control["raw"],hw,ap)
            if wide!=int(p["wide_slot"]) or len(peaks)!=int(p["zoom_search_slots"]):
                raise ValueError("Stage-129 response placement changed")
            zoom = {k:control[k].copy() for k in ("candidate_parents","candidate_boxes","features")}
            order = sorted(peaks)
            if order:
                fine = c.read_cache(r129/"frames"/f"{stem}__zoom.npz",dict(meta,arm="zoom"))
                if fine is None:
                    raise ValueError("Missing Stage-129 local observation")
                np.testing.assert_equal(fine["frame_id"],f)
                np.testing.assert_array_equal(fine["image_hw"],hw)
                np.testing.assert_array_equal(fine["candidate_parents"],expect[order])
                for k in zoom:
                    zoom[k][order] = fine[k]
            banks = {"trim":(control,None,[]),"two_zoom":(zoom,None,[])}
            banks.update(finite_choices(control,zoom,wide,order,hw,ap))
            history = list(prior.prior.history(trace,entry["initial_box"],i,ap["design"]["history"]))
            gt = entry["ground_truth"][f]
            for arm,(bank,rank,votes) in banks.items():
                scores,selected = base.choose(bank,history,"generated_shared",scorer)
                iou = tracking_iou(bank["candidate_boxes"],gt)
                if iou.shape!=(4,) or (iou<0).any():
                    raise ValueError("Invalid Stage-130 evaluation state")
                if arm in ("trim","two_zoom"):
                    ref = "trim" if arm=="trim" else "zoom"
                    np.testing.assert_equal(selected,int(p[ref+"_selected_slot"]))
                    np.testing.assert_allclose([iou.max(),iou[selected]],[float(p[ref+"_best"]),float(p[ref+"_selected"])],
                                               atol=1e-12,rtol=0)
                points.append(dict(ident,arm=arm,active=bool(order),small_covered=p["source_small_covered"]=="True",
                    wide_slot=wide,zoom_rank=rank,replaced_slot=replacement_slot(wide) if order and rank else None,
                    rank1_votes=votes[0] if votes else None,rank2_votes=votes[1] if len(votes)>1 else None,
                    best_iou=float(iou.max()),selected_iou=float(iou[selected]),selected_slot=selected,
                    q3_iou=float(iou[3]),scores="|".join(map(str,scores)),ious="|".join(map(str,iou))))
        print(f"[Stage-130] {seq} s{seed}: {len(records)} cached states",flush=True)
    if len(points)!=DESIGN["points"]*len(ARMS):
        raise ValueError("Incomplete frozen sample")
    summary,paired = summarize(points)
    hashes = {}
    for name,rows in dict(points=points,summary=summary,paired=paired).items():
        path = root/f"stage130_{name}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root/"stage130_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        plan_fingerprint=plan["fingerprint"],source_stage129_fingerprint=SOURCE129,
        row_counts=dict(points=len(points),summary=len(summary),paired=len(paired)),table_sha256=hashes,**WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage127","stage129")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--check-results",action="store_true")
    args = parser.parse_args()
    if args.archive:
        writer = csv.writer(sys.stdout,lineterminator="\n")
        rows = list(archive_support(*extract_archive(args.archive)))
        writer.writerow(rows[0])
        writer.writerows(r.values() for r in rows)
        return
    if args.output_dir is None:
        parser.error("Supply --output-dir or --archive")
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all(getattr(args,k) for k in names):
        parser.error("Supply the frozen source directories")
    r76,r89,r93,r112,r121,r127,r129 = paths = [getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-130 directory")
    if not (r129/"stage129_decision.json").is_file():
        parser.error("Need the server Stage-129 frame caches for re-scoring")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"stage130_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r129)
        p129 = c.c66.sealed(r129/"manifest.json",source.SCHEMA)
        d129 = c.c66.sealed(r129/"stage129_decision.json",source.SCHEMA)
        p127 = c.c66.sealed(r127/"manifest.json",source.source.audit.SCHEMA)
        if d129["fingerprint"]!=SOURCE129 or p127["inputs"].get(str(r121/"manifest.json"))!=c.file_hash(r121/"manifest.json"):
            raise ValueError("Need the frozen Stage-121/127/129 lineage")
        p121 = c.c66.sealed(r121/"manifest.json",prior.s121.SCHEMA)
        p112 = c.c66.sealed(r112/"manifest.json",prior.source.common.SCHEMA)
        paths112 = [r112/"manifest.json",r112/"analysis/stage112_decision.json",
                    *(r112/"models").glob("*__generated_4__shared__*.json")]
        io.verify({str(p):p121["source_sha256"][str(p)] for p in paths112})
        models,_ = base.load_frozen_models(r112,p112)
        traces = {path:sha for path,sha in p127["inputs"].items() if Path(path).parent==r121/"evaluation"}
        if not traces:
            raise ValueError("Missing Stage-121 source binds")
        io.verify(traces)
        s95 = base.s103.prior.s95
        p93,_,p89,report89,origin = s95.s94.load_run(r93,r76,r89)
        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,source_stage129_fingerprint=SOURCE129,
            source_stage127_plan=p127["fingerprint"],code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(root/"manifest.json",plan)
        run(root,r121,p121,r129,p129,plan,models,load_entry,base.e63.c.actor_plan(origin["original"]))


if __name__=="__main__":
    main()
