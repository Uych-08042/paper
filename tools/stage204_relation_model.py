"""Parameter-matched explicit relation descriptors with fit-only references."""
import hashlib
import numpy as np
import stage199_residual_model as old

ARMS=('concat','matched_relation','mismatched_relation')


def donors(fold, groups):
    fit=list(fold['fit'])
    if len(fit)<2 or set(fit)&set(fold['held']):raise ValueError('Invalid donor pool')
    return {g:fit[(fit.index(g)+1)%len(fit)] if g in fit else fit[int(hashlib.sha256(g.encode()).hexdigest(),16)%len(fit)] for g in groups}


def descriptor(value, arm, reference=None):
    a,b,c,d=np.split(np.asarray(value,dtype=np.float32),4,axis=-1)
    if arm=='concat':return np.asarray(value,dtype=np.float32).copy()
    if arm not in ARMS:raise ValueError('Unknown relation arm')
    if arm=='mismatched_relation':
        if reference is None:raise ValueError('Fit reference required')
        a,b=np.split(np.asarray(reference,dtype=np.float32),2,axis=-1)
    return np.concatenate((a*c,np.abs(a-c),b*d,np.abs(b-d)),axis=-1)


def prepare(bank, fold, arm):
    mapping=donors(fold,bank)
    refs={g:bank[mapping[g]]['pooled'].astype(float).mean(0)[:bank[g]['pooled'].shape[1]//2].astype(np.float32) for g in bank}
    result={g:dict(v,pooled=descriptor(v['pooled'],arm,refs[g]).astype(np.float32),reference=refs[g].astype(np.float32)) for g,v in bank.items()}
    stats=old.fit_statistics(result,fold['fit'])
    variance=np.stack([((v['pooled'].astype(float)-stats['mean_pooled'])**2).mean(0) for g,v in result.items() if g in fold['fit']]).mean(0)
    stats['scale_pooled']=np.where(variance>0,np.sqrt(variance),1.)
    return result,stats,mapping


def build(torch, stats, lens_z, arm):
    if arm not in ARMS:raise ValueError('Unknown relation arm')
    total,dimension=stats['mean_tokens'].shape
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('prior','mean_tokens','mean_pooled','scale_pooled'):
                dtype=torch.float64 if name in ('mean_pooled','scale_pooled') else torch.float32
                self.register_buffer(name,torch.as_tensor(stats[name],dtype=dtype).clone())
            self.projection=torch.nn.Linear(dimension,1,bias=False)
            self.readout=torch.nn.Linear(4*dimension+total,6,bias=False)
            torch.nn.init.zeros_(self.projection.weight);torch.nn.init.zeros_(self.readout.weight)
            with torch.no_grad():self.projection.weight[0,0]=1.

        def pool(self,tokens,reference=None):
            base=old.pooling.pool(torch,tokens,lens_z)
            if arm=='concat':return base
            a,b,c,d=base.chunk(4,-1)
            if arm=='mismatched_relation':
                if reference is None:raise ValueError('Reference required for direct path')
                a,b=reference.chunk(2,-1)
            return torch.cat((a*c,(a-c).abs(),b*d,(b-d).abs()),-1)

        def raw(self,tokens,pooled=None,reference=None):
            value=self.pool(tokens,reference) if pooled is None else pooled
            standardized=((value.to(torch.float64)-self.mean_pooled)/self.scale_pooled).to(tokens.dtype)
            features=torch.cat((standardized,self.projection(tokens-self.mean_tokens).squeeze(-1)),-1)
            return self.readout(features)

        def forward(self,tokens,pooled=None,reference=None):
            raw=self.raw(tokens,pooled,reference)
            return self.prior+(raw-raw.mean(-1,keepdim=True))
    return Router()
