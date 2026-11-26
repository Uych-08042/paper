"""Paired executed-crop support and matched sampling for Stage-207."""
import numpy as np
import stage194_module_common as protocol

SCHEMA='stage207_paired_provider_state_training_v1'
TRAINED=('native_all','native_pair','own_pair')
ARMS=('native','frozen_provider',*TRAINED)
DESIGN=dict(protocol.DESIGN)


def covered(truth,parent,factor,image_hw,search_size):
    """The actual sample_target integer window, excluding constant padded pixels.

    Loss targets retain the frozen legacy floating map; support uses the executed
    integer crop (including sample_target's existing right/bottom -1 behavior).
    """
    truth,parent=np.asarray(truth,float),np.asarray(parent,float)
    hw=np.asarray(image_hw,int)
    if (truth.shape!=(4,) or parent.shape!=(4,) or hw.shape!=(2,)
            or not np.isfinite(np.r_[truth,parent,factor,search_size]).all() or float(factor)<=0 or search_size<=0
            or np.any(truth[2:]<=0) or np.any(parent[2:]<=0) or np.any(hw<=0)):raise ValueError('Invalid executed crop geometry')
    side=int(round(float(search_size)/float(factor)))
    if side<=0 or abs(side*float(factor)-search_size)>1e-5:raise ValueError('Noninteger executed crop side')
    start=np.asarray([round(parent[k]+parent[k+2]/2-side/2) for k in range(2)])
    lower=np.maximum(start,0);upper=np.minimum(start+side,hw[::-1]-1)
    return bool(np.all(truth[:2]>=lower) and np.all(truth[:2]+truth[2:]<=upper))


def paired_bank(native,own,search_size):
    if set(native)!=set(own):raise ValueError('Paired banks must contain the same fit sources only')
    native_pair,own_pair,support,sampling={}, {},[],[]
    for g in native:
        a,b=native[g],own[g]
        np.testing.assert_array_equal(a['frame_id'],b['frame_id'])
        np.testing.assert_array_equal(a['truth'],b['truth'])
        masks=[];na=[];nb=[]
        for i,frame in enumerate(a['frame_id']):
            valid=lambda data:covered(data['truth'][i],data['parent'][i],data['factor'][i],data['image_hw'][i],search_size)
            va,vb=valid(a),valid(b);masks.append(va and vb);na.append(va);nb.append(vb)
            sampling.append(dict(group=g,frame=int(frame),native_covered=va,own_covered=vb,paired=va and vb))
        mask=np.asarray(masks,bool)
        native_pair[g]={k:np.asarray(v)[mask] for k,v in a.items()}
        own_pair[g]={k:np.asarray(v)[mask] for k,v in b.items()}
        support.append(dict(group=g,states=len(mask),native_covered=sum(na),own_covered=sum(nb),paired_states=int(mask.sum()),
            changed_parent_frames=int(np.any(a['parent']!=b['parent'],axis=1).sum()),
            changed_token_frames=int(np.any(a['tokens']!=b['tokens'],axis=(1,2)).sum())))
    return native_pair,own_pair,support,sampling


def summarize(rows):
    groups=sorted({r['group'] for r in rows});index={};summary=[]
    for arm in ARMS:
        for g in groups:
            cell=[r for r in rows if r['arm']==arm and r['group']==g]
            seeds=[0] if arm=='native' else DESIGN['seeds']
            if sorted(r['seed'] for r in cell)!=seeds:raise ValueError('Incomplete source/seed result')
            index[arm,g]={k:float(np.mean([r[k] for r in cell])) for k in protocol.METRICS}
            index[arm,g]['introduced']=any(r['introduced_failure'] for r in cell)
        summary.append(dict(arm=arm,videos=len(groups),introduced_failure_videos=sum(index[arm,g]['introduced'] for g in groups),
            **{k:float(np.mean([index[arm,g][k] for g in groups])) for k in protocol.METRICS}))
    pairs=[]
    for a,b in (('own_pair','native_pair'),('own_pair','native_all'),('own_pair','native'),('own_pair','frozen_provider'),
                ('native_pair','native_all'),('native_all','frozen_provider')):
        for metric in ('auc','low_iou_fraction','longest_failure_fraction','missed_native_recoveries','native_good_frame_loss'):
            pairs.append(dict(contrast=a+'_minus_'+b,metric=metric,primary=(a=='own_pair' and b=='native_pair' and metric=='auc'),
                favorable_direction='positive' if metric=='auc' else 'negative',
                **protocol.contrast([index[a,g][metric]-index[b,g][metric] for g in groups])))
    return summary,pairs
