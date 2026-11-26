"""Full-width, single-head spatial residual; native routing reads pre-adapter F6."""
from contextlib import ExitStack
import math
import numpy as np

ARMS = ('search_control','matched_spatial','mismatched_spatial')


def build(torch, dimension, lens_z, search_side, seed, arm):
    if arm not in ARMS or int(math.isqrt(lens_z))**2 != lens_z or search_side != 2*math.isqrt(lens_z):
        raise ValueError('Need square template and twice-resolution search grids')
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query=torch.nn.Linear(dimension,dimension,bias=False)
            self.key=torch.nn.Linear(dimension,dimension,bias=False)
            self.value=torch.nn.Linear(dimension,dimension,bias=False)
            self.output=torch.nn.Linear(dimension,dimension,bias=False)
            torch.nn.init.zeros_(self.output.weight)

        def adapt(self,tokens,reference=None):
            z,x=tokens[:,:lens_z],tokens[:,lens_z:]
            if x.shape[1]!=search_side**2 or tokens.shape[-1]!=dimension:
                raise ValueError('Unexpected spatial token layout')
            if arm=='search_control':
                grid=x.transpose(1,2).reshape(len(x),dimension,search_side,search_side)
                ref=torch.nn.functional.avg_pool2d(grid,2,2).flatten(2).transpose(1,2)
            elif arm=='matched_spatial':ref=z
            else:
                if reference is None or tuple(reference.shape[-2:])!=(lens_z,dimension):
                    raise ValueError('Fit-only spatial template reference required')
                ref=reference.expand(len(tokens),-1,-1)
            norm=lambda t:torch.nn.functional.layer_norm(t,(dimension,))
            q,k,v=self.query(norm(x)),self.key(norm(ref)),self.value(norm(ref))
            attention=(q@k.transpose(-1,-2)/math.sqrt(dimension)).softmax(-1)
            return torch.cat((z,x+self.output(attention@v)),1)
    # Independent of CUDA/global RNG state, identical trainable initialization in all arms.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Adapter()


def chosen(rt,tokens):
    with rt.torch.no_grad():return rt.torch.topk(rt.selector(tokens.detach()),1,dim=1).indices[:,0]+7


def objective(rt,model,tokens,reference,target):
    from stage18_tracking_loss import _center_heatmap,tracking_loss
    torch=rt.torch;layers=chosen(rt,tokens)
    encoded=model.adapt(tokens,reference)
    terms=[];calls=0
    # One native-selected candidate per example; no six-layer hindsight target.
    for layer in sorted(set(layers.tolist())):
        mask=layers==layer
        feature=rt.backbone.blocks[layer-1](encoded[mask])
        feature=rt.s1.recover_tokens(feature,rt.lens_z,rt.lens_x,mode=rt.backbone.cat_mode)
        heatmap,_=_center_heatmap(torch,target[mask],rt.search_side,rt.search_side)
        out=rt.net.forward_head(rt.backbone.norm(feature),heatmap[:,0])
        loss,_=tracking_loss(torch,out,target[mask])
        terms.append(loss*int(mask.sum())/len(tokens));calls+=1
    return sum(terms),layers,calls


class Installed:
    """One original MLP call; replace its input, never its scores/weights."""
    def __init__(self,native,model,reference):
        self.rt,self.model,self.reference=native.rt,model,reference
        self.route_input=None;self.prefix_calls=0;self.route_calls=0

    def prefix(self,module,args,output):
        self.route_input=output[:,:,0].detach().clone()
        self.prefix_calls+=1
        return self.model.adapt(output,self.reference)

    def route(self,module,args):
        if self.route_input is None or args[0].shape!=self.route_input.shape:
            raise ValueError('Missing original pre-adapter route input')
        self.route_calls+=1
        value=self.route_input;self.route_input=None
        return (value,)+args[1:]

    def observed(self,module,args,output):self.last_scores=output.detach().clone()

    def __enter__(self):
        self.stack=ExitStack()
        try:
            self.stack.callback(self.rt.backbone.blocks[5].register_forward_hook(self.prefix).remove)
            self.stack.callback(self.rt.backbone.MLP.register_forward_pre_hook(self.route).remove)
            self.stack.callback(self.rt.backbone.MLP.register_forward_hook(self.observed).remove)
        except BaseException:
            self.stack.close();raise
        return self

    def __exit__(self,*exc):return self.stack.__exit__(*exc)
