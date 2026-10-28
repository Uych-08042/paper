"""Native hard-selector SGLA with exactly the shared L5-L6 adaptation budget."""

import copy
import hashlib

from stage20_compression_common import state_digest
from stage35_preserve_runtime import parameters, restore_state, trainable_state


def make_retained(rt):
    net = copy.deepcopy(rt.net).eval().requires_grad_(False)
    if len(net.backbone.blocks) != 12 or not hasattr(net.backbone,"MLP") or hasattr(net,"spatial_identity"):
        raise ValueError("Initialize retained control from original native SGLA")
    for index in (4,5):
        net.backbone.blocks[index].requires_grad_(True)
    parameters(net)
    if state_digest(net) != state_digest(rt.net):
        raise ValueError("Retained initialization differs from original SGLA")
    return net


def frozen_digest(net):
    trainable = set(parameters(net))
    digest = hashlib.sha256()
    for name,value in sorted(net.state_dict().items()):
        if name not in trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def forward_retained(rt,net,search):
    if len(net.backbone.blocks) != 12 or not hasattr(net.backbone,"MLP") or search.shape[0] != 1:
        raise ValueError("Native retained forward requires one crop and all original candidates")
    bb = net.backbone
    z = bb.patch_embed(rt.tracker.z_dict1.tensors) + bb.pos_embed_z
    x = bb.patch_embed(search) + bb.pos_embed_x
    tokens = bb.pos_drop(rt.s1.combine_tokens(z,x,mode=bb.cat_mode))
    for block in bb.blocks[:6]:
        tokens = block(tokens)
    # The original selector's weights and hard decision rule remain frozen.
    # Prefix gradients pass through the executed candidate, never through argmax.
    with rt.torch.no_grad():
        layer = rt.selected_layer(bb.MLP(tokens[:,:,0].detach().clone()))
    if layer not in range(7,13):
        raise ValueError("Native selector returned an invalid candidate")
    tokens = bb.blocks[layer - 1](tokens.clone())
    tokens = rt.s1.recover_tokens(tokens,rt.lens_z,rt.lens_x,mode=bb.cat_mode)
    return net.forward_head(bb.norm(tokens),None),layer


def shared_prefix_identity(rt, retained, six):
    a,b = trainable_state(retained),trainable_state(six)
    if set(a) != set(b) or not all(rt.torch.equal(a[k],b[k]) for k in a):
        raise ValueError("Structures do not share exact initial trainable prefix")
    return dict(trainable_parameters=sum(p.numel() for p in parameters(retained).values()),
        retained_stored_parameters=sum(p.numel() for p in retained.parameters()),
        six_stored_parameters=sum(p.numel() for p in six.parameters()),
        prefix_sha256=prefix_digest(a))


def prefix_digest(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
