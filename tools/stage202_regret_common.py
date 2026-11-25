"""Matched MSE versus IoU-regret margin objectives and source-level summaries."""
from __future__ import annotations

import numpy as np
import stage194_module_common as protocol
import stage195_routing_common as metrics
import stage200_pair_common as pair_source

ARMS = ('point', 'regret_margin')
pair_draws = pair_source.pair_draws
flatten = pair_source.flatten
decomposition = pair_source.decomposition


def objectives(torch, score, label):
    if score.shape != label.shape or score.ndim != 2 or score.shape[1] != 6 or len(score) == 0:
        raise ValueError('Need nonempty matched [states,6] scores and IoU labels')
    error = score - label
    point = (error - error.mean(1, keepdim=True)).square().mean()
    # Training targets use the exact true maximum; all true maximizers have zero regret.
    best = label.argmax(1, keepdim=True)
    gaps = label.gather(1, best) - label
    # Center at the target score before max: the target entry is exactly zero.
    margins = score - score.gather(1, best) + gaps
    regret = margins.max(1).values.mean()
    return dict(point=point, regret_margin=regret)


def decision_terms(label, score):
    """Float64 evaluation bound for the inherited near-argmax tie rule."""
    y, s = metrics.validate_scores(label, score)
    best = y.argmax(1)
    index = np.arange(len(y))
    bound = np.max(s - s[index, best, None] + y[index, best, None] - y, axis=1)
    action = metrics.choose(s)
    regret = y.max(1) - y[index, action]
    violation = np.maximum(regret - bound - metrics.EPS, 0.)
    if (bound < -1e-12).any() or violation.max() > 1e-12:
        raise ValueError('IoU regret upper bound failed')
    return dict(bound=bound, regret=regret, violation=violation)


def aggregate(rows, category, names, fields):
    output = []
    for arm in ARMS:
        for pool in ('fit', 'held'):
            for name in names:
                selected = [r for r in rows if r['arm'] == arm and r['pool'] == pool and r[category] == name]
                groups = sorted({r['group'] for r in selected})
                if len(groups) != 38 or len(selected) != 76:
                    raise ValueError('Incomplete two-seed source summary')
                row = dict(arm=arm, pool=pool, **{category: name}, videos=38)
                for field in fields:
                    values = []
                    for group in groups:
                        cells = [r[field] for r in selected if r['group'] == group]
                        if all(v is not None for v in cells):
                            values.append(float(np.mean(cells)))
                    row[field] = float(np.mean(values)) if values else None
                    row[field + '_supported_videos'] = len(values)
                output.append(row)
    return output


def paired_reports(policies, errors):
    output = []
    for pool in ('fit', 'held'):
        def compare(a, pa, b, pb, field, table, kind, name, direction='positive'):
            delta = []
            for g in sorted({r['group'] for r in table}):
                get = lambda arm, p: [r[field] for r in table if r['arm'] == arm and r['pool'] == pool
                                      and r['group'] == g and r.get('policy', r.get('predictor')) == p]
                x, y = get(a, pa), get(b, pb)
                if len(x) != 2 or len(y) != 2:
                    raise ValueError('Incomplete paired source/seed cell')
                if all(v is not None for v in x + y):
                    delta.append(float(np.mean(x) - np.mean(y)))
            if len(delta) > 1:
                stats = protocol.contrast(delta)
            else:
                stats = dict(videos=len(delta),mean_delta=delta[0] if delta else None,median_delta=delta[0] if delta else None,
                             ci_low=None,ci_high=None,positive=sum(v>1e-9 for v in delta),negative=sum(v< -1e-9 for v in delta),worst_leave_one_video_out=None)
            output.append(dict(arm=a,pool=pool,kind=kind,contrast=name,metric=field,favorable_direction=direction,**stats))
        for arm in ARMS:
            for control in ('prior_only','fit_prediction_mean','native_L9'):
                for field in ('mean_iou','state_alignment_gain'):
                    compare(arm,'trained',arm,control,field,policies,'within','trained_minus_'+control)
            selected=[r for r in policies if r['arm']==arm and r['pool']==pool and r['policy']=='trained']
            delta=[float(np.mean([r['state_alignment_gain'] for r in selected if r['group']==g])) for g in sorted({r['group'] for r in selected})]
            output.append(dict(arm=arm,pool=pool,kind='alignment',contrast='trained_minus_independent_expectation',metric='mean_iou',
                               favorable_direction='positive',**protocol.contrast(delta)))
        for field in ('mean_iou','state_alignment_gain'):
            compare(ARMS[1],'trained',ARMS[0],'trained',field,policies,'between','regret_minus_point')
        for field,direction in (('interaction_mse','negative'),('within_video_mse','negative'),
                                ('gap_weighted_pair_accuracy','positive'),('regret_margin_loss','negative')):
            compare(ARMS[1],'trained',ARMS[0],'trained',field,errors,'prediction','regret_minus_point',direction)
    return output
