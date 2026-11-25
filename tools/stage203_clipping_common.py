"""Loss-by-clipping factorial helpers; frozen Stage-202 losses and sampling."""
from __future__ import annotations

import numpy as np
import stage202_regret_common as previous

protocol = previous.protocol
ARMS = ('point_clip', 'point_no_clip', 'regret_clip', 'regret_no_clip')
pair_draws, flatten = previous.pair_draws, previous.flatten
objectives, decision_terms = previous.objectives, previous.decision_terms


def flags(arm):
    choices = dict(point_clip=('point', True), point_no_clip=('point', False),
                   regret_clip=('regret_margin', True), regret_no_clip=('regret_margin', False))
    if arm not in choices:
        raise ValueError('Unknown loss/clipping arm')
    return choices[arm]


def tensor_norm(torch, tensors):
    tensors = list(tensors)
    if not tensors:
        raise ValueError('Cannot measure an empty tensor list')
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.detach(), 2) for t in tensors]), 2)


def gradient_rule(torch, parameters, clipping):
    """Preserve legacy clipping exactly; the off branch never writes gradients."""
    parameters = list(parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    measured = tensor_norm(torch, gradients)
    if not bool(torch.isfinite(measured)):
        raise FloatingPointError('Nonfinite gradient norm')
    if clipping:
        norm = torch.nn.utils.clip_grad_norm_(parameters, protocol.DESIGN['gradient_clip'], error_if_nonfinite=True)
        scale = float((norm.new_tensor(protocol.DESIGN['gradient_clip']) / (norm + 1e-6)).clamp(max=1.))
    else:
        norm, scale = measured, 1.
    after = tensor_norm(torch, gradients)
    if not bool(torch.isfinite(after)):
        raise FloatingPointError('Nonfinite post-rule gradient norm')
    return dict(gradient_norm=float(norm), gradient_norm_after=float(after), gradient_scale=scale,
                gradient_was_clipped=scale < 1., inherited_threshold_exceeded=float(norm) > protocol.DESIGN['gradient_clip'])


def aggregate(rows, category, names, fields):
    result = []
    for arm in ARMS:
        for pool in ('fit', 'held'):
            for name in names:
                cell = [r for r in rows if r['arm'] == arm and r['pool'] == pool and r[category] == name]
                groups = sorted({r['group'] for r in cell})
                if len(groups) != 38 or len(cell) != 76:
                    raise ValueError('Incomplete two-seed source grid')
                row = dict(arm=arm, pool=pool, **{category: name}, videos=38)
                for field in fields:
                    values = []
                    for group in groups:
                        entries = [r[field] for r in cell if r['group'] == group]
                        if all(v is not None for v in entries):
                            values.append(float(np.mean(entries)))
                    row[field] = float(np.mean(values)) if values else None
                    row[field + '_supported_videos'] = len(values)
                result.append(row)
    return result


def paired_reports(policies, errors):
    output = []
    groups = sorted({r['group'] for r in policies})
    if len(groups) != 38:
        raise ValueError('Need all 38 development source groups')
    for pool in ('fit', 'held'):
        def values(table, arm, name, field):
            result = []
            for group in groups:
                cell = [r[field] for r in table if r['arm'] == arm and r['pool'] == pool and r['group'] == group
                        and r.get('policy', r.get('predictor')) == name]
                if len(cell) != 2:
                    raise ValueError('Incomplete paired source/seed grid')
                result.append(float(np.mean(cell)) if all(v is not None for v in cell) else None)
            return result
        def add(arm, kind, name, field, delta, direction='positive'):
            delta = [float(v) for v in delta if v is not None]
            if len(delta) < 2:
                stats = dict(videos=len(delta), mean_delta=delta[0] if delta else None, median_delta=delta[0] if delta else None,
                             ci_low=None, ci_high=None, positive=sum(v > 1e-9 for v in delta), negative=sum(v < -1e-9 for v in delta), worst_leave_one_video_out=None)
            else:
                stats = protocol.contrast(delta)
            output.append(dict(arm=arm, pool=pool, kind=kind, contrast=name, metric=field, favorable_direction=direction, **stats))
        def compare(a, pa, b, pb, field, table, kind, name, direction='positive'):
            x, y = values(table,a,pa,field), values(table,b,pb,field)
            add(a,kind,name,field,[v-w if v is not None and w is not None else None for v,w in zip(x,y)],direction)
        for arm in ARMS:
            for control in ('prior_only','fit_prediction_mean','native_L9'):
                for field in ('mean_iou','state_alignment_gain'):
                    compare(arm,'trained',arm,control,field,policies,'within','trained_minus_'+control)
            add(arm,'alignment','trained_minus_independent_expectation','mean_iou',values(policies,arm,'trained','state_alignment_gain'))
        for objective in ('point','regret'):
            for field in ('mean_iou','state_alignment_gain'):
                compare(objective+'_no_clip','trained',objective+'_clip','trained',field,policies,'clipping',objective+'_no_clip_minus_clip')
            for field,direction in (('interaction_mse','negative'),('within_video_mse','negative'),
                                    ('regret_margin_loss','negative'),('gap_weighted_pair_accuracy','positive')):
                compare(objective+'_no_clip','trained',objective+'_clip','trained',field,errors,'prediction',objective+'_no_clip_minus_clip',direction)
        for condition in ('clip','no_clip'):
            for field in ('mean_iou','state_alignment_gain'):
                compare('regret_'+condition,'trained','point_'+condition,'trained',field,policies,'objective','regret_minus_point_'+condition)
        for field in ('mean_iou','state_alignment_gain'):
            v = {a: values(policies,a,'trained',field) for a in ARMS}
            delta = [v['regret_no_clip'][i]-v['regret_clip'][i]-v['point_no_clip'][i]+v['point_clip'][i] for i in range(len(groups))]
            add('factorial','interaction','loss_by_clipping_interaction',field,delta)
    return output
