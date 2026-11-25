"""Cache-only video score baselines: future reference versus strictly past mean."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO / 'tools', REPO / 'tracking'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical, seal as sealed
from stage4_common import file_hash, fingerprint, write_csv
import stage194_module_common as protocol
import stage195_routing_common as metrics
import stage198_routing_calibration as correspondence
import stage200_pair_common as common

SCHEMA = 'stage201_score_baseline_identification_v1'
SOURCE = 'b9b4aafb6a648a38d54347e76dd1e2138f8ad9b188cfbea51c962bb59b30f2dc'
SOURCE_PLAN = '8792d5489f5d1711191ea202e4c2d550bbc310fa17f72615a308fd68f437522f'
BUNDLE_SHA256 = '00a487b9552785b6a4eb9f2a64a7477591e4ddbed32b5d348df362026a8d8838'
DEFAULT_BUNDLE = REPO / 'docs/experiments/06_sgla_uav_tracking/results/stage200_results_20261125T140231Z_1329509.txt'
PROTOCOL_FILE = REPO / 'docs/experiments/06_sgla_uav_tracking/STAGE201_SCORE_BASELINE_IDENTIFICATION.md'
ARMS = common.ARMS
SCORES = ('raw', 'offline_video_mean', 'causal_past_mean', 'prior_only')
POLICIES = (*SCORES, 'native_L9', 'oracle')
SOURCE_TABLES = ('model_scope', 'fit_summary', 'source_bridge', 'pair_schedule', 'pair_support', 'error_summary',
                 'error_video', 'policy_summary', 'policy_video', 'paired', 'frame_predictions', 'training_history')
TABLES = ('source_audit', 'calibration_video', 'policy_video', 'policy_summary', 'seed_summary', 'paired',
          'error_video', 'error_summary', 'frame_predictions')
POLICY_FIELDS = ('mean_iou', 'mean_regret', 'good_fraction', 'modal_fraction', 'independent_mean_iou',
                 'state_alignment_gain', 'layer_mix_gain_vs_prior', 'total_gain_vs_prior')
ERROR_FIELDS = ('between_video_mse', 'within_video_mse', 'within_video_skill_vs_zero', 'gap_weighted_pair_accuracy')


def load_bundle(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise ValueError('Need the exact reviewed Stage-200 bundle')
    matches = list(re.finditer(rb'(?:^|\r?\n)===== ([\w.]+) =====\r?\n', raw))
    parts = {}
    for i, m in enumerate(matches):
        name = m[1].decode()
        if name in parts:
            raise ValueError('Duplicate source section')
        parts[name] = raw[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(raw)]
    expected = {'stage200_decision.json', 'manifest.json', *(f'stage200_{n}.csv' for n in SOURCE_TABLES)}
    if set(parts) != expected:
        raise ValueError('Source inventory changed')
    d, p = json.loads(parts['stage200_decision.json']), json.loads(parts['manifest.json'])
    for value in (d, p):
        if value.get('fingerprint') != fingerprint({k: v for k, v in value.items() if k != 'fingerprint'}):
            raise ValueError('Source fingerprint changed')
    if (d['fingerprint'] != SOURCE or p['fingerprint'] != SOURCE_PLAN or d['plan_fingerprint'] != SOURCE_PLAN
            or d['training_runs'] != 8 or d['predicted_states'] != 14592
            or any(d[k] != 0 for k in ('new_image_reads', 'new_encoder_calls', 'new_full_trajectories', 'confirmation_groups_opened'))
            or set(d['table_sha256']) != {f'stage200_{n}.csv' for n in SOURCE_TABLES}):
        raise ValueError('Unexpected Stage-200 source identity or scope')
    tables = {}
    for name, sha in d['table_sha256'].items():
        b = parts[name]
        candidates = [b, b[:-1] if b.endswith(b'\n') else b, b[:-2] if b.endswith(b'\r\n') else b]
        value = next((v for v in candidates if hashlib.sha256(v).hexdigest() == sha), None)
        if value is None:
            raise ValueError('Source table checksum mismatch: ' + name)
        tables[name.removeprefix('stage200_').removesuffix('.csv')] = list(csv.DictReader(io.StringIO(value.decode())))
    for name, sha in p['code_sha256'].items():
        if file_hash(REPO / 'tools' / name) != sha:
            raise ValueError('Frozen Stage-200 code changed: ' + name)
    if file_hash(REPO / 'docs/experiments/06_sgla_uav_tracking/STAGE200_WITHIN_VIDEO_DIFFERENCE.md') != p['protocol_sha256']:
        raise ValueError('Frozen Stage-200 protocol changed')
    return d, p, tables


def baseline_scores(prediction, prior):
    """No labels. Current p is added only AFTER its past-only baseline is read."""
    p, mu = np.asarray(prediction, float), np.asarray(prior, float)
    if p.ndim != 2 or p.shape[1] != 6 or len(p) == 0 or mu.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(mu).all():
        raise ValueError('Expected finite nonempty six-layer scores and fit prior')
    offline = mu + p - p.mean(0)
    causal = np.empty_like(p)
    causal[0] = mu
    past_sum = p[0].copy()
    for t in range(1, len(p)):
        causal[t] = mu + p[t] - past_sum / t
        past_sum += p[t]
    return dict(raw=p.copy(), offline_video_mean=offline, causal_past_mean=causal,
                prior_only=np.broadcast_to(mu, p.shape).copy())


def models(d, plan, tables):
    scopes = tables['model_scope']
    expected = {(f, s, a) for f in (0, 1) for s in protocol.DESIGN['seeds'] for a in ARMS}
    if len(scopes) != 8 or {(int(r['fold']), int(r['seed']), r['arm']) for r in scopes} != expected:
        raise ValueError('Model grid changed')
    grouped = defaultdict(lambda: defaultdict(list))
    for row in tables['frame_predictions']:
        key = int(row['fold']), int(row['seed']), row['arm']
        if key not in expected:
            raise ValueError('Unknown model')
        grouped[key][row['group']].append(row)
    groups = sorted(next(iter(grouped.values())))
    if len(groups) != 38 or len(tables['frame_predictions']) != d['predicted_states']:
        raise ValueError('Incomplete state grid')
    folds = protocol.split_groups(groups)
    if folds != plan['folds']:
        raise ValueError('Fold split changed')
    reference = {}
    for scope in scopes:
        f, seed, arm = int(scope['fold']), int(scope['seed']), scope['arm']
        fold = folds[f]
        if scope['fit_groups'].split('|') != fold['fit'] or scope['held_groups'].split('|') != fold['held']:
            raise ValueError('Source fit/held scope changed')
        bank = grouped[f, seed, arm]
        if set(bank) != set(groups):
            raise ValueError('Source groups changed')
        videos = {}
        for g in groups:
            records = bank[g]
            ids = np.array([int(r['frame']) for r in records])
            y = np.array([[float(r[f'label_L{k}']) for k in range(7, 13)] for r in records])
            pred = np.array([[float(r[f'prediction_L{k}']) for k in range(7, 13)] for r in records], np.float32)
            choices = np.array([int(r['selected_layer']) - 7 for r in records])
            metrics.validate_scores(y, pred)
            if len(ids) != 48 or ids[0] <= 0 or (np.diff(ids) <= 0).any() or not np.array_equal(choices, metrics.choose(pred)):
                raise ValueError('State order or original action changed')
            for i, r in enumerate(records):
                if r['pool'] != ('fit' if g in fold['fit'] else 'held') or r['sequence'] != records[0]['sequence']:
                    raise ValueError('Pool or sequence changed')
                if abs(float(r['selected_iou']) - y[i, choices[i]]) > 1e-12:
                    raise ValueError('Source selected IoU changed')
            identity = (tuple(ids), tuple(map(tuple, y)))
            if g in reference and reference[g] != identity:
                raise ValueError('Source labels or frame IDs differ across models')
            reference[g] = identity
            videos[g] = dict(sequence=records[0]['sequence'], frame_id=ids, labels=y, prediction=pred, choice=choices)
        prior = np.stack([videos[g]['labels'].mean(0) for g in fold['fit']]).mean(0)
        fit_rows = [r for r in tables['fit_summary'] if int(r['fold']) == f]
        if len(fit_rows) != 1 or not np.allclose(prior, [float(fit_rows[0][f'prior_L{k}']) for k in range(7, 13)], atol=1e-12, rtol=0):
            raise ValueError('Fit prior changed')
        yield scope, fold, seed, arm, prior, videos


def paired(rows):
    result = []
    groups = sorted({r['group'] for r in rows})
    def values(arm, pool, policy, metric):
        data = []
        for g in groups:
            cell = [r[metric] for r in rows if r['arm'] == arm and r['pool'] == pool and r['policy'] == policy and r['group'] == g]
            if len(cell) != 2:
                raise ValueError('Incomplete paired source/seed cell')
            data.append(float(np.mean(cell)))
        return np.array(data)
    def add(arm, pool, name, delta):
        result.append(dict(arm=arm, pool=pool, contrast=name, metric='mean_iou', favorable_direction='positive', **protocol.contrast(delta)))
    for pool in ('fit', 'held'):
        for arm in ARMS:
            for policy in ('offline_video_mean', 'causal_past_mean'):
                for control in ('raw', 'prior_only'):
                    add(arm, pool, policy + '_minus_' + control, values(arm, pool, policy, 'mean_iou') - values(arm, pool, control, 'mean_iou'))
            add(arm, pool, 'causal_past_mean_minus_offline_video_mean', values(arm, pool, 'causal_past_mean', 'mean_iou') - values(arm, pool, 'offline_video_mean', 'mean_iou'))
            for policy in ('raw', 'offline_video_mean', 'causal_past_mean'):
                add(arm, pool, policy + '_minus_independent_expectation', values(arm, pool, policy, 'state_alignment_gain'))
        for policy in ('raw', 'offline_video_mean', 'causal_past_mean'):
            add('difference_vs_point', pool, policy + '_difference_minus_point', values(ARMS[1], pool, policy, 'mean_iou') - values(ARMS[0], pool, policy, 'mean_iou'))
    return result


def analyze(d, plan, source_tables, progress):
    rows = {k: [] for k in TABLES}
    for scope, fold, seed, arm, prior, videos in models(d, plan, source_tables):
        audit = dict(fold=fold['fold'], seed=seed, arm=arm, source_model_sha256=scope['final_sha256'],
                     source_pair_draws_sha256=scope['pair_draws_sha256'], source_states=sum(len(v['frame_id']) for v in videos.values()),
                     maximum_raw_policy_error=0., maximum_offline_mean_error=0., maximum_past_reconstruction_error=0., causal_baseline_violations=0)
        for g, v in videos.items():
            p, y, ids = v['prediction'].astype(float), v['labels'], v['frame_id']
            score = baseline_scores(p, prior)
            actions = {n: metrics.choose(s) for n, s in score.items()}
            actions.update(native_L9=np.full(len(p), 2), oracle=metrics.choose(y))
            if not np.array_equal(actions['raw'], v['choice']):
                raise ValueError('Raw action reproduction failed')
            base = dict(fold=fold['fold'], seed=seed, arm=arm, group=g, sequence=v['sequence'], pool='fit' if g in fold['fit'] else 'held')
            prior_iou = float(y[np.arange(len(y)), actions['prior_only']].mean())
            offline_error = float(np.abs(score['offline_video_mean'].mean(0) - prior).max())
            if offline_error > 1e-12:
                raise ValueError('Offline mean reconstruction failed')
            audit['maximum_offline_mean_error'] = max(audit['maximum_offline_mean_error'], offline_error)
            raw_reports = [r for r in source_tables['policy_video'] if int(r['fold']) == fold['fold'] and int(r['seed']) == seed
                           and r['arm'] == arm and r['group'] == g and r['policy'] == 'trained']
            if len(raw_reports) != 1:
                raise ValueError('Missing raw source policy')
            for policy in POLICIES:
                report = correspondence.correspondence(y, actions[policy])
                if policy == 'raw':
                    for field in ('mean_iou', 'state_alignment_gain', 'independent_mean_iou'):
                        error = abs(report[field] - float(raw_reports[0][field]))
                        audit['maximum_raw_policy_error'] = max(audit['maximum_raw_policy_error'], error)
                        if error > 1e-12:
                            raise ValueError('Source policy metric changed')
                rows['policy_video'].append(dict(base, policy=policy, states=len(y), **report,
                    layer_mix_gain_vs_prior=report['independent_mean_iou'] - prior_iou, total_gain_vs_prior=report['mean_iou'] - prior_iou,
                    future_prediction_reference=policy == 'offline_video_mean', ground_truth_reference=policy == 'oracle'))
            for predictor in SCORES:
                rows['error_video'].append(dict(base, predictor=predictor, **common.decomposition(y, score[predictor], prior), **metrics.ranking(y, score[predictor])))
            past_means = np.cumsum(p, axis=0)[:-1] / np.arange(1, len(p))[:, None]
            past_error = float(np.abs(score['causal_past_mean'][1:] - (prior + p[1:] - past_means)).max())
            if past_error > 1e-12 or not np.array_equal(score['causal_past_mean'][0], prior):
                raise ValueError('Past-only baseline reconstruction failed')
            violations = int(np.count_nonzero(ids[:-1] >= ids[1:]))
            audit['causal_baseline_violations'] += violations
            audit['maximum_past_reconstruction_error'] = max(audit['maximum_past_reconstruction_error'], past_error)
            centered_offset = p.mean(0) - prior
            centered_offset -= centered_offset.mean()
            baseline_delta = past_means - p.mean(0)
            baseline_delta -= baseline_delta.mean(1, keepdims=True)
            rows['calibration_video'].append(dict(base, states=len(p), first_frame=int(ids[0]), last_frame=int(ids[-1]),
                first_causal_layer=int(actions['causal_past_mean'][0])+7,
                raw_mean_layer_offset_rms=float(np.sqrt(np.mean(centered_offset**2))),
                past_baseline_centered_mse_to_full_mean=float(np.mean(baseline_delta**2)),
                maximum_offline_mean_error=offline_error, maximum_past_reconstruction_error=past_error, causal_baseline_violations=violations,
                **{f'prior_L{k}': float(prior[k-7]) for k in range(7,13)},
                **{f'prediction_mean_L{k}': float(p[:,k-7].mean()) for k in range(7,13)}))
            for i, frame in enumerate(ids):
                rows['frame_predictions'].append(dict(base, frame=int(frame), sample_index=i, causal_history_count=i,
                    causal_baseline_last_frame=int(ids[i-1]) if i else None, offline_baseline_last_frame=int(ids[-1]),
                    **{n+'_layer': int(a[i])+7 for n,a in actions.items()},
                    **{n+'_iou': float(y[i,a[i]]) for n,a in actions.items()},
                    **{f'{n}_score_L{k}': float(score[n][i,k-7]) for n in SCORES[:3] for k in range(7,13)},
                    **{f'label_L{k}': float(y[i,k-7]) for k in range(7,13)}))
            progress.advance()
        rows['source_audit'].append(audit)
    rows['policy_summary'] = common.aggregate(rows['policy_video'], 'policy', POLICIES, POLICY_FIELDS)
    rows['error_summary'] = common.aggregate(rows['error_video'], 'predictor', SCORES, ERROR_FIELDS)
    for arm in ARMS:
        for pool in ('fit', 'held'):
            for seed in protocol.DESIGN['seeds']:
                for policy in POLICIES:
                    cell = [r for r in rows['policy_video'] if r['arm']==arm and r['pool']==pool and r['seed']==seed and r['policy']==policy]
                    if len(cell) != 38:
                        raise ValueError('Incomplete seed summary')
                    rows['seed_summary'].append(dict(arm=arm, pool=pool, seed=seed, policy=policy, videos=38,
                        **{m: float(np.mean([r[m] for r in cell])) for m in POLICY_FIELDS}))
    rows['paired'] = paired(rows['policy_video'])
    return rows


def verify_results(root):
    d = correspondence.read_sealed(root / 'stage201_decision.json')
    p = correspondence.read_sealed(root / 'manifest.json')
    if (d['schema'] != SCHEMA or d['status'] != 'COMPLETE_SCORE_BASELINE_IDENTIFICATION_DIAGNOSTIC'
            or d['plan_fingerprint'] != p['fingerprint'] or d['source_stage200'] != SOURCE
            or d['models_audited'] != 8 or d['evaluated_states'] != 14592 or d['policy_video_rows'] != 1824
            or set(d['table_sha256']) != {f'stage201_{n}.csv' for n in TABLES}
            or any(d[k] != 0 for k in ('new_training_runs','new_training_updates','new_model_forwards','new_encoder_calls','new_image_reads','new_full_trajectories','confirmation_groups_opened'))):
        raise ValueError('Incomplete Stage-201 diagnostic')
    for name, sha in d['table_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError('Changed Stage-201 table: '+name)
    for name, sha in p['code_sha256'].items():
        if file_hash(REPO / 'tools' / name) != sha:
            raise ValueError('Changed Stage-201 code: '+name)
    if file_hash(PROTOCOL_FILE) != p['protocol_sha256']:
        raise ValueError('Changed Stage-201 protocol')
    return d


def export_bundle(root, d):
    path = root / ('stage201_results_' + d['fingerprint'][:12] + '.txt')
    data = bytearray()
    for name in ('stage201_decision.json','manifest.json',*(f'stage201_{n}.csv' for n in TABLES)):
        data.extend(('\n===== '+name+' =====\n').encode()); data.extend((root/name).read_bytes())
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Committed export changed')
    if not path.exists(): path.write_bytes(data)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check-results', action='store_true')
    args = parser.parse_args()
    root, bundle = args.output_dir.expanduser().resolve(), args.source_bundle.expanduser().resolve()
    if args.check_results:
        d = verify_results(root); print('Verified result bundle: '+str(export_bundle(root,d)),flush=True); return
    protocol.safe_output(root,[bundle.parent])
    with writer_lock(root):
        if (root/'stage201_decision.json').exists():
            d = verify_results(root); print('Reused result bundle: '+str(export_bundle(root,d)),flush=True); return
        d200, p200, tables = load_bundle(bundle)
        plan = sealed(dict(schema=SCHEMA, source_stage200=SOURCE, source_plan200=SOURCE_PLAN,
            source_bundle_sha256=BUNDLE_SHA256, source_table_sha256=d200['table_sha256'], arms=list(ARMS), policies=list(POLICIES),
            transforms='offline mu+p-mean_all(p); causal first mu, then mu+p-mean_strict_past(p); no windows/scales/clipping',
            precision='restore float32 source predictions, then float64 transforms; source labels float64',
            fit_prior='original fold fit labels, source groups equally weighted; no held-label calibration',
            state_order='all 48 cached points increasing original frame IDs; own sequence reset per model/video',
            bootstrap=protocol.DESIGN, folds=p200['folds'], seeds=p200['seeds'],
            primary='difference held causal-minus-prior, causal-minus-raw and causal alignment; offline reference is not online',
            source_scope='historically exposed 38 development sources; cached native-parent states, not own-history trajectories',
            numpy=str(np.__version__),python=sys.version,protocol_sha256=file_hash(PROTOCOL_FILE),
            code_sha256={Path(f).name:file_hash(f) for f in (__file__,common.__file__,protocol.__file__,metrics.__file__,correspondence.__file__)}))
        publish_identical(root/'manifest.json',plan)
        progress = OverallProgress('Stage-201',306,unit='work');progress.report('Verified source; baseline reconstruction');progress.advance()
        rows = analyze(d200,p200,tables,progress)
        hashes = {}
        for name in TABLES:
            path = root/f'stage201_{name}.csv';write_csv(path,rows[name]);hashes[path.name]=file_hash(path)
        if progress.done != progress.total-1: raise ValueError('Progress accounting changed')
        primary = [r for r in rows['paired'] if r['arm']==ARMS[1] and r['pool']=='held' and r['contrast'] in
                   ('causal_past_mean_minus_prior_only','causal_past_mean_minus_raw','causal_past_mean_minus_independent_expectation')]
        d = sealed(dict(schema=SCHEMA,status='COMPLETE_SCORE_BASELINE_IDENTIFICATION_DIAGNOSTIC',plan_fingerprint=plan['fingerprint'],
            source_stage200=SOURCE,source_bundle_sha256=BUNDLE_SHA256,source_groups=38,models_audited=8,
            evaluated_states=len(rows['frame_predictions']),policy_video_rows=len(rows['policy_video']),new_training_runs=0,new_training_updates=0,
            new_model_forwards=0,new_encoder_calls=0,new_image_reads=0,new_full_trajectories=0,confirmation_groups_opened=0,
            maximum_raw_policy_error=max(r['maximum_raw_policy_error'] for r in rows['source_audit']),
            maximum_offline_mean_error=max(r['maximum_offline_mean_error'] for r in rows['source_audit']),
            maximum_past_reconstruction_error=max(r['maximum_past_reconstruction_error'] for r in rows['source_audit']),
            causal_baseline_violations=sum(r['causal_baseline_violations'] for r in rows['source_audit']),
            primary_contrasts=primary,table_sha256=hashes,source_table_sha256=d200['table_sha256'],
            scope='text scores only; PT/NPZ not reverified; offline mean uses future predictions; past-only means are over native-parent cached samples',
            next_step='Review offline versus past-only outcomes and two-seed/source consistency; no automatic method adoption'))
        publish_identical(root/'stage201_decision.json',d)
        progress.phase='Complete';progress.advance();progress.report()
        verify_results(root);print('Result bundle: '+str(export_bundle(root,d)),flush=True)


if __name__ == '__main__':
    main()
