"""NumPy-only, same-state quality diagnostics for already decoded search candidates."""
from __future__ import annotations
import numpy as np
import stage61_l40_common as prior  # Establishes the frozen tracking import path.
from stage27_uav_common import failure_events
from validation_metrics import IOU_PROTOCOL, tracking_iou

DESIGN=dict(good_iou=.5,low_iou=.2,before=30,after=30,epsilon=1e-6)
COHORTS=("all","native_reliable","native_low","pre_first_failure","first_failure_episode","first_recovery_window")


def case(case_id,sequence,origin,seed,policy,frame,kind,purpose):
    return dict(case_id=case_id,sequence=sequence,origin=origin,seed=seed,anchor_policy=policy,
                frame=frame,anchor_kind=kind,purpose=purpose)


CASES=(
    case("C01","uav0000226_05370_s","native",0,"history",115,"onset","native_lost_recovery"),
    case("C02","uav0000307_04531_s","endpoint",2132,"history",392,"onset","endpoint_lost_recovery"),
    case("C03","uav0000307_04531_s","endpoint",2133,"history",395,"onset","endpoint_lost_recovery"),
    case("C04","uav0000176_00000_s","native",0,"current",162,"onset","native_failure_avoided"),
    case("C05","uav0000204_00000_s","native",0,"current",664,"onset","native_failure_avoided"),
    case("C06","uav0000304_00253_s","endpoint",2133,"current",554,"onset","endpoint_seed2133_gain"),
    case("C07","uav0000304_00253_s","endpoint",2132,"current",1078,"onset","endpoint_seed2132_control_loss"),
    case("C08","uav0000304_00253_s","endpoint",2132,"history",1093,"onset","endpoint_seed2132_history_loss"),
    case("C09","uav0000307_04531_s","endpoint",2132,"current",471,"recovery","endpoint_recovery_opportunity"),
    case("C10","uav0000307_04531_s","endpoint",2133,"current",511,"recovery","endpoint_recovery_opportunity"),
    case("C11","uav0000226_05370_s","native",0,"current",338,"recovery","native_recovery_opportunity"),
)


def first_frame(mask):
    found=np.flatnonzero(mask)
    return int(found[0])+1 if len(found) else None


def frame_metrics(data,entry):
    """GT labels the frozen candidates; it cannot select or advance any state."""
    boxes=np.asarray(data["candidate_boxes"],float)
    count,proposals,_=boxes.shape
    gt=np.asarray(entry["ground_truth"],float)[1:count+1]
    if gt.shape!=(count,4) or not np.isfinite(gt).all() or np.any(gt[:,2:]<=0):
        raise ValueError("Missing or invalid offline ground truth")
    selected=np.asarray(data["selected"],int)
    if selected.shape!=(count,) or np.any((selected<0)|(selected>=proposals)): raise ValueError("Invalid frozen selection")
    idx=np.arange(count)
    np.testing.assert_array_equal(data["boxes"],boxes[idx,selected])
    overlap=tracking_iou(boxes,gt[:,None])
    if not np.isfinite(overlap).all() or np.any(overlap<0): raise ValueError("Invalid candidate IoU")
    best_index=overlap.argmax(1); best=overlap[idx,best_index]; own=overlap[idx,selected]; q0=overlap[:,0]
    good=overlap>=DESIGN["good_iou"]; available=good.any(1); own_good=good[idx,selected]; q0_good=good[:,0]
    parents=np.asarray(data["candidate_parents"],float)
    if parents.shape!=boxes.shape or np.any(parents[:,:,2:]<=0): raise ValueError("Invalid candidate parents")
    crop=np.asarray(data["crop"],float)
    hw=np.asarray(data["image_hw"],float)
    if crop.shape!=boxes.shape or hw.shape!=(count,2): raise ValueError("Invalid cached geometry")
    # Same continuous usable area as crop_geometry, including historical +1 right/bottom padding.
    lo=crop[:,:,:2]; size=crop[:,:,2:]
    left=np.maximum(0.,-lo)
    right=np.maximum(0.,lo+size-hw[:,None,::-1]+1.)
    usable_lo=lo+left; usable_size=np.maximum(0.,size-left-right)
    gtlo=gt[:,None,:2]; gthi=gtlo+gt[:,None,2:]
    intersection=np.maximum(0.,np.minimum(usable_lo+usable_size,gthi)-np.maximum(usable_lo,gtlo)).prod(2)
    coverage=intersection/gt[:,None,2:].prod(2)
    center=gtlo+gt[:,None,2:]/2
    center_inside=((center>=usable_lo)&(center<usable_lo+usable_size)).all(2)
    features=np.asarray(data["features"],float)
    hann=features[:,:,prior.original.prior.FEATURES.index("hann_peak")]
    raw=features[:,:,prior.original.prior.FEATURES.index("raw_peak")]
    parent_center=parents[:,:,:2]+parents[:,:,2:]/2
    output_center=boxes[:,:,:2]+boxes[:,:,2:]/2
    gt_center=gtlo+gt[:,None,2:]/2
    normalized_parent=(parent_center-gt_center)/gt[:,None,2:]
    normalized_output=(output_center-gt_center)/gt[:,None,2:]
    size_ratio=boxes[:,:,2:]/gt[:,None,2:]
    row=dict(frame_id=np.arange(1,count+1),selected=selected,best_index=best_index,
        q0_iou=q0,selected_iou=own,best_decoded_iou=best,decoded_regret=best-own,
        selected_minus_q0_iou=own-q0,best_minus_q0_iou=best-q0,has_good_decoded=available,
        no_good_decoded=~available,available_selected_good=available&own_good,
        available_selected_miss=available&~own_good,q0_good=q0_good,selected_good=own_good,
        selected_low=own<DESIGN["low_iou"],all_decoded_low=best<DESIGN["low_iou"],
        q0_good_lost=q0_good&~own_good,q0_good_to_low=q0_good&(own<DESIGN["low_iou"]),
        rescue_opportunity=~q0_good&available,rescued=~q0_good&own_good,
        q0_low_rescued=(q0<DESIGN["low_iou"])&own_good,noncurrent=selected!=0,
        max_gt_coverage=coverage.max(1),selected_gt_coverage=coverage[idx,selected],q0_gt_coverage=coverage[:,0],
        any_gt_center_inside=center_inside.any(1),selected_gt_center_inside=center_inside[idx,selected],
        q0_hann=hann[:,0],selected_hann=hann[idx,selected],best_iou_candidate_hann=hann[idx,best_index],
        selected_minus_q0_hann=hann[idx,selected]-hann[:,0],
        q0_raw_peak=raw[:,0],selected_raw_peak=raw[idx,selected])
    detail=dict(iou=overlap,coverage=coverage,center_inside=center_inside,hann=hann,raw=raw,
        parent_error=normalized_parent,output_error=normalized_output,size_ratio=size_ratio)
    return row,detail


def cohort_masks(metrics,native_iou):
    count=len(metrics["selected_iou"]); events=failure_events(metrics["selected_iou"])
    if np.asarray(native_iou).shape!=(count,): raise ValueError("Native reference length differs")
    masks=dict(all=np.ones(count,bool),native_reliable=np.asarray(native_iou)>=DESIGN["good_iou"],
        native_low=np.asarray(native_iou)<DESIGN["low_iou"],pre_first_failure=np.zeros(count,bool),
        first_failure_episode=np.zeros(count,bool),first_recovery_window=np.zeros(count,bool))
    if events:
        e=events[0]; masks["pre_first_failure"][max(0,e["start"]-DESIGN["before"]):e["start"]]=True
        masks["first_failure_episode"][e["start"]:e["end"]+1]=True
        if e["recovery"] is not None:
            masks["first_recovery_window"][e["recovery"]:min(count,e["recovery"]+DESIGN["after"])]=True
    return masks,events


FLAGS=("has_good_decoded","no_good_decoded","available_selected_good","available_selected_miss",
       "q0_good","selected_good","selected_low","all_decoded_low","q0_good_lost","q0_good_to_low",
       "rescue_opportunity","rescued","q0_low_rescued","noncurrent","any_gt_center_inside","selected_gt_center_inside")
MEANS=("q0_iou","selected_iou","best_decoded_iou","decoded_regret","selected_minus_q0_iou","best_minus_q0_iou",
       "max_gt_coverage","selected_gt_coverage","q0_gt_coverage","selected_minus_q0_hann")
CONDITIONAL={"miss_given_good_decoded":("available_selected_miss","has_good_decoded"),
    "loss_given_good_q0":("q0_good_lost","q0_good"),
    "low_given_good_q0":("q0_good_to_low","q0_good"),
    "rescue_given_opportunity":("rescued","rescue_opportunity")}
RATE_METRICS=tuple(k+"_fraction" for k in FLAGS)+tuple("mean_"+k for k in MEANS)+tuple(CONDITIONAL)


def summarize_metrics(metrics,mask):
    mask=np.asarray(mask,bool); count=int(mask.sum()); out=dict(frame_states=count)
    if mask.shape!=metrics["selected_iou"].shape: raise ValueError("Cohort length differs")
    for name in FLAGS:
        out[name+"_count"]=int(np.asarray(metrics[name])[mask].sum())
        out[name+"_fraction"]=out[name+"_count"]/count if count else None
    for name in MEANS: out["mean_"+name]=float(np.mean(metrics[name][mask])) if count else None
    for name,(num,den) in CONDITIONAL.items():
        denominator=out[den+"_count"]
        out[name]=out[num+"_count"]/denominator if denominator else None
    return out


def aggregate(rows,group_names,support_label):
    """Equal supported-run means, then equal supported-video means; expose support."""
    groups={}
    for row in rows: groups.setdefault(tuple(row[k] for k in group_names),[]).append(row)
    result=[]
    for values,items in groups.items():
        row=dict(zip(group_names,values)); row[support_label]=len(items)
        row["frame_states_total"]=sum(r.get("frame_states",r.get("frame_states_total",0)) for r in items)
        for metric in RATE_METRICS:
            supported=[r[metric] for r in items if r[metric] is not None]
            row[metric]=float(np.mean(supported)) if supported else None
            row[metric+"_supported_"+support_label]=len(supported)
        result.append(row)
    return result
