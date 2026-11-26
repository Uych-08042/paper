"""Source-cross-fitted residual trust; no held labels in coefficient fitting."""
from __future__ import annotations

import numpy as np
import stage204_target_relation as previous

ARMS = previous.ARMS
POLICIES = ('shrunk', 'raw', 'prior_only', 'native_L9', 'oracle')
PREDICTORS = ('shrunk', 'raw', 'prior_only')
FIELDS = ('mean_iou', 'mean_regret', 'state_alignment_gain', 'independent_mean_iou',
          'good_fraction', 'mean_selected_layer', 'nonprior_fraction')


def inner_fold(outer, dropped):
    fit = list(outer['fit'])
    if len(fit) != 19 or len(set(fit)) != 19 or dropped not in fit or set(fit) & set(outer['held']):
        raise ValueError('Need an outer 19/19 source split and a fit-only dropped source')
    return dict(fold=outer['fold'], fit=[g for g in fit if g != dropped], held=[dropped])


def centered(value):
    value = np.asarray(value, dtype=np.float64)
    return value - value.mean(-1, keepdims=True)


def estimate_alpha(videos, expected_groups):
    """One source = one vote, then states within source; not state-count weighted."""
    if set(videos) != set(expected_groups) or len(expected_groups) != len(set(expected_groups)) or not videos:
        raise ValueError('Missing or duplicated inner out-of-source predictions')
    moments = []
    for group in expected_groups:
        video = videos[group]
        y, score = previous.metrics.validate_scores(video['labels'], video['prediction'])
        prior = np.asarray(video['prior'], dtype=np.float64)
        if prior.shape != (6,) or not np.isfinite(prior).all():
            raise ValueError('Invalid inner prior')
        h, z = centered(score-prior), centered(y-prior)
        moments.append((float(np.mean(np.sum(h*z, axis=1))), float(np.mean(np.sum(h*h, axis=1))),
                        float(np.mean(z*z)), float(np.mean((h-z)**2))))
    numerator, denominator, risk0, risk1 = np.mean(moments, axis=0)
    unrestricted = float(numerator/denominator) if denominator > 0 else None
    alpha = float(np.clip(unrestricted, 0., 1.)) if unrestricted is not None else 0.
    risk = float(risk0 + (alpha*alpha*denominator - 2*alpha*numerator)/6)
    if not np.isfinite([numerator, denominator, risk0, risk1, risk, alpha]).all() or risk > min(risk0, risk1)+1e-12:
        raise ValueError('Nonfinite or nonminimal bounded quadratic calibration risk')
    return dict(alpha=alpha, unrestricted_alpha=unrestricted, numerator=float(numerator), denominator=float(denominator),
                calibration_mse_prior=float(risk0), calibration_mse_raw=float(risk1), calibration_mse_shrunk=risk,
                source_groups=len(videos), states=sum(len(v['labels']) for v in videos.values()),
                zero_denominator=bool(denominator == 0), boundary=bool(alpha in (0., 1.)))


def shrink(score, prior, alpha):
    score, prior = np.asarray(score, dtype=np.float64), np.asarray(prior, dtype=np.float64)
    if score.ndim != 2 or score.shape[1] != 6 or prior.shape != (6,) or not 0 <= alpha <= 1 or not np.isfinite(score).all() or not np.isfinite(prior).all():
        raise ValueError('Invalid residual shrinkage inputs')
    if alpha == 1:  # Preserve exact Stage-204 exported float32 scores, including near ties.
        return score.copy()
    if alpha == 0:
        return np.broadcast_to(prior, score.shape).copy()
    return prior + float(alpha)*centered(score-prior)


def summarize(rows, category, names, fields, seed=None):
    output = []
    for arm in ARMS:
        for name in names:
            cell = [r for r in rows if r['arm'] == arm and r[category] == name and (seed is None or r['seed'] == seed)]
            groups = sorted({r['group'] for r in cell})
            if len(groups) != 38 or len(cell) != (76 if seed is None else 38):
                raise ValueError('Incomplete held source/seed grid')
            values = dict(arm=arm, **{category:name}, videos=38)
            if seed is not None: values['seed'] = seed
            for field in fields:
                means = [float(np.mean([r[field] for r in cell if r['group'] == g])) for g in groups
                         if all(r[field] is not None for r in cell if r['group'] == g)]
                values[field] = float(np.mean(means)) if means else None
                values[field+'_supported_videos'] = len(means)
            output.append(values)
    return output


def paired(rows):
    output = []
    groups = sorted({r['group'] for r in rows})
    def values(arm, policy, field):
        result = []
        for group in groups:
            cell = [r[field] for r in rows if r['arm'] == arm and r['policy'] == policy and r['group'] == group]
            if len(cell) != 2: raise ValueError('Incomplete paired source seeds')
            result.append(float(np.mean(cell)))
        return np.asarray(result)
    def add(arm, name, metric, delta, primary=False):
        output.append(dict(arm=arm, contrast=name, metric=metric, primary=primary,
                           favorable_direction='positive', **previous.protocol.contrast(delta)))
    for arm in ARMS:
        for control in ('raw', 'prior_only'):
            for field in ('mean_iou', 'state_alignment_gain'):
                add(arm, 'shrunk_minus_'+control, field, values(arm,'shrunk',field)-values(arm,control,field),
                    arm == 'matched_relation' and field == 'mean_iou')
        for policy in ('shrunk', 'raw'):
            add(arm, policy+'_minus_independent_expectation', 'mean_iou', values(arm,policy,'state_alignment_gain'),
                arm == 'matched_relation' and policy == 'shrunk')
    for arm in ('concat','mismatched_relation'):
        for field in ('mean_iou','state_alignment_gain'):
            add('matched_relation', 'matched_shrunk_minus_'+arm+'_shrunk', field,
                values('matched_relation','shrunk',field)-values(arm,'shrunk',field), field == 'mean_iou')
    return output
