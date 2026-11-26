"""Causal square-search state learning; fixed-budget forecast/coverage comparison."""
import csv,hashlib,io,json,re
from pathlib import Path
import numpy as np
import stage213_output_ceiling as source
old=source.old
protocol=source.previous.c.PROTOCOL
SCHEMA='stage214_learned_search_state_v1'
SOURCE213='27a584ca907ddc0e0584ddfd7e7944535bb694fc57bc831a368ef29f570b1065'
BUNDLE213='4084d3323e1253c7d92b92d3628962553afc7fcd0513f9755ff9f5b628399176'
SEEDS=[2132,2133]
TRAINED=('forecast','coverage')
ARMS=('native','nn','mm',*TRAINED)
DESIGN=dict(updates=256,batch=4,samples_per_video=48,hidden=32,learning_rate=1e-3,weight_decay=1e-4,gradient_clip=1.,coverage_weight=1.)
TABLES=('frozen_scope','fit_support','scope_summary','training_history','smoke','source_replay','policy_summary','paired','seed_video_summary','search_geometry','calls','execution')
METRICS=source.previous.c.METRICS


def source213(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BUNDLE213:raise ValueError('Need exact reviewed Stage-213 result')
    mm=list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n',raw))
    pp={m[1].decode():raw[m.end():mm[i+1].start() if i+1<len(mm) else len(raw)] for i,m in enumerate(mm)}
    d,p=json.loads(pp['stage213_decision.json']),json.loads(pp['manifest.json'])
    if len(pp)!=len(mm) or set(pp)!={'manifest.json','stage213_decision.json',*d['table_sha256']}:raise ValueError('Changed source inventory')
    for v in (d,p):
        if v!=old.sealed({k:w for k,w in v.items() if k!='fingerprint'}):raise ValueError('Changed source fingerprint')
    if d['fingerprint']!=SOURCE213 or d['plan_fingerprint']!=p['fingerprint'] or d['source_groups']!=38 or d['derived_oracle_sequences']!=76 or any(d[k]!=0 for k in ('new_training_updates','new_image_reads','new_gpu_inference_calls','new_full_trajectories','confirmation_groups_opened')):raise ValueError('Unexpected source scope')
    tables={}
    for name,sha in d['table_sha256'].items():
        b=pp[name];v=next((v for v in (b,b[:-1],b[:-2]) if hashlib.sha256(v).hexdigest()==sha),None)
        if v is None:raise ValueError('Changed source table')
        tables[name.removeprefix('stage213_').removesuffix('.csv')]=list(csv.DictReader(io.StringIO(v.decode())))
    if set(tables)!=set(source.TABLES):raise ValueError('Incomplete source')
    return d,p,tables


def features(tokens,lens_z,parent,raw,adapted):
    t=np.asarray(tokens,dtype=np.float32)
    if t.ndim!=3 or t.shape[0]!=1 or not 0<lens_z<t.shape[1]:raise ValueError('Bad prefix tokens')
    p=np.asarray(parent,float);side=np.sqrt(np.prod(p[2:]));center=p[:2]+p[2:]/2
    def geom(b):
        b=np.asarray(b,float)
        return np.r_[(b[:2]+b[2:]/2-center)/side,np.log(b[2:]/p[2:])]
    result=np.r_[t[0,:lens_z].mean(0),t[0,lens_z:].mean(0),geom(raw),geom(adapted)].astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite causal input')
    return result


def parent(raw,delta,image_hw):
    r=np.asarray(raw,float);z=np.asarray(delta,float);hw=np.asarray(image_hw,float)
    if r.shape!=(4,) or z.shape!=(3,) or not np.isfinite(np.r_[r,z,hw]).all() or min(r[2:])<=0:raise ValueError('Bad learned state')
    side=np.sqrt(np.prod(r[2:]));center=r[:2]+r[2:]/2
    shift=np.clip(z[:2]*side,-center,hw[::-1]-center)
    scale=np.exp(np.clip(np.log(side)+z[2],0,np.log(max(hw)))-np.log(side))
    wh=r[2:]*scale
    result=np.r_[r[:2]+shift+(r[2:]-wh)/2,wh]
    if not np.isfinite(result).all() or min(result[2:])<=0:raise ValueError('Invalid next search parent')
    return result


def build(torch,dimension,mean,std,seed):
    class Updater(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
            self.network=torch.nn.Sequential(torch.nn.Linear(dimension,DESIGN['hidden']),torch.nn.Tanh(),torch.nn.Linear(DESIGN['hidden'],3))
            torch.nn.init.zeros_(self.network[-1].weight);torch.nn.init.zeros_(self.network[-1].bias)
        def forward(self,x):return self.network((x-self.mean)/self.std)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return Updater()


def loss(torch,delta,raw,target,image_hw,factor,arm):
    if arm not in TRAINED:raise ValueError('Unknown objective')
    side=raw[:,2:].prod(1).sqrt();rc=raw[:,:2]+raw[:,2:]/2;tc=target[:,:2]+target[:,2:]/2;ts=target[:,2:].prod(1).sqrt()
    bound=image_hw.flip(1);center=torch.minimum(torch.maximum(rc+delta[:,:2]*side[:,None],torch.zeros_like(rc)),bound)
    logside=torch.minimum(torch.clamp(side.log()+delta[:,2],min=0),image_hw.max(1).values.log());newside=logside.exp()
    pred=torch.cat(((center-rc)/side[:,None],(logside-side.log())[:,None]),1)
    label=torch.cat(((tc-rc)/side[:,None],(ts/side).log()[:,None]),1)
    regression=torch.nn.functional.smooth_l1_loss(pred,label)
    # Crop side=factor*sqrt(w*h); edge violations penalize missed target extent.
    overflow=torch.relu(((center-tc).abs()+target[:,2:]/2-factor*newside[:,None]/2)/ts[:,None]).mean()
    total=regression+(overflow if arm=='coverage' else 0.)
    return total,regression,overflow


def audit(v,initial,n,arm):
    if arm=='native':old.audit_trace(v,initial,n);return
    np.testing.assert_array_equal(v['frame_id'],np.arange(1,n+1))
    for k in ('parent','boxes','reference_boxes','next_parent'):
        if v[k].shape!=(n,4) or not np.isfinite(v[k]).all() or (v[k][:,2:]<=0).any():raise ValueError('Bad online geometry')
    np.testing.assert_array_equal(v['parent'],np.vstack((initial,v['next_parent'][:-1])))
    if arm in ('nn','mm'):source.previous.c.audit(v,initial,n,arm)
    else:
        np.testing.assert_allclose(v['next_parent'],np.stack([parent(r,z,hw) for r,z,hw in zip(v['reference_boxes'],v['state_delta'],v['image_hw'])]),rtol=0,atol=0)
        calls=np.zeros((n,12),int);calls[:,:6]=1;calls[np.arange(n),v['layer']-1]=2
        np.testing.assert_array_equal(v['block_calls'],calls)
        for k,num in (('head_calls',2),('selector_calls',1),('crop_calls',1)):np.testing.assert_array_equal(v[k],np.full(n,num))
    if not np.isin(v['layer'],range(7,13)).all():raise ValueError('Invalid chosen layer')


def summarize(rows):
    groups=sorted({r['group'] for r in rows});idx={};summary=[];paired=[]
    for a in ARMS:
        for g in groups:
            rr=[r for r in rows if r['group']==g and r['arm']==a]
            if sorted(r['seed'] for r in rr)!=([0] if a=='native' else SEEDS):raise ValueError('Incomplete seed grid')
            idx[a,g]={m:float(np.mean([r[m] for r in rr])) for m in METRICS}
            idx[a,g]['introduced']=any(r['introduced_failure'] for r in rr);idx[a,g]['introduced_mm']=any(r['introduced_matched_failure'] for r in rr)
        summary.append(dict(arm=a,videos=len(groups),introduced_failure_videos=sum(idx[a,g]['introduced'] for g in groups),introduced_matched_failure_videos=sum(idx[a,g]['introduced_mm'] for g in groups),**{m:float(np.mean([idx[a,g][m] for g in groups])) for m in METRICS}))
    for a,b in [('coverage','forecast')]+[(a,b) for a in TRAINED for b in ('native','nn','mm')]:
        for m in ('auc','low_iou_fraction','missed_native_recoveries','native_good_frame_loss','missed_matched_recoveries','matched_good_frame_loss'):
            paired.append(dict(contrast=a+'_minus_'+b,metric=m,primary=(a,b,m)==('coverage','forecast','auc'),favorable_direction='positive' if m=='auc' else 'negative',**protocol.contrast([idx[a,g][m]-idx[b,g][m] for g in groups])))
    return summary,paired
