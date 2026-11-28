"""Frozen held-state localization evidence census; CPU arrays/GT only."""
import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'tools'), str(REPO / 'tracking')]
import stage221_target_components as previous
import stage217_predictor_audit as source
from stage207_state_common import covered
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, write_csv
from experiment_progress import OverallProgress

old = source.old
SCHEMA = 'stage222_cached_localization_evidence_v1'
SOURCE221 = 'a6980275270a08484d6dc4c6ee098d74d2cf126217491c021d38cdd9980f9dc5'
PROTOCOL = REPO / 'docs/experiments/06_sgla_uav_tracking/STAGE222_LOCALIZATION_EVIDENCE.md'
ORIGINS = source.prior.ORIGINS
CLASSES = ('adapted_good', 'not_fully_covered', 'covered_no_good_grid', 'covered_readout_miss')
TABLES = ('selection', 'cells', 'group_metrics', 'policy_summary', 'primary', 'source_replay', 'execution')


def category(raw_iou, adapted_iou, full, grid):
    if raw_iou >= .2:
        return 'raw_not_severe'
    if adapted_iou >= .5:
        return CLASSES[0]
    if not full:
        return CLASSES[1]
    return CLASSES[2] if grid < .5 else CLASSES[3]


def observations(trace, frames, truth, search_size):
    n = len(trace['frame_id'])
    np.testing.assert_array_equal(trace['frame_id'], np.arange(1, n + 1))
    for name in ('parent', 'reference_boxes', 'boxes'):
        value = np.asarray(trace[name])
        if value.shape != (n, 4) or not np.isfinite(value).all() or (value[:, 2:] <= 0).any():
            raise ValueError('Bad current geometry: ' + name)
    for name in ('fully_covered', 'dense_best_iou', 'factor', 'gt_side_network_px'):
        if np.asarray(trace[name]).shape != (n,):
            raise ValueError('Missing/bad adapted-branch evidence: ' + name)
    if np.asarray(trace['image_hw']).shape != (n, 2) or trace['fully_covered'].dtype != np.bool_:
        raise ValueError('Invalid coverage/HW schema')
    grid = np.asarray(trace['dense_best_iou'])
    if not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError('Invalid dense IoU')
    if len(set(frames)) != len(frames) or not frames or min(frames) < 1 or max(frames) >= len(truth) or max(frames) > n:
        raise ValueError('Invalid selected frames')
    result = []
    for frame in frames:
        i = frame - 1
        gt = np.asarray(truth[frame], float)
        valid = bool(np.isfinite(gt).all() and (gt[2:] > 0).all())
        row = dict(frame=frame, valid_current=valid, raw_iou=None, adapted_iou=None,
                   dense_best_iou=None, fully_covered=None, raw_error_center_l2=None,
                   raw_error_log_side=None, readout_gap=None, gt_side_network_px=None,
                   category='invalid_current_gt')
        if valid:
            full = covered(gt, trace['parent'][i], trace['factor'][i], trace['image_hw'][i], search_size)
            if full != bool(trace['fully_covered'][i]):
                raise ValueError('Executed integer crop coverage replay differs')
            raw_iou = float(old.tracking_iou(trace['reference_boxes'][i:i+1], gt)[0])
            adapted = float(old.tracking_iou(trace['boxes'][i:i+1], gt)[0])
            error = source.label(trace['reference_boxes'][i:i+1], gt[None])[0]
            nominal = float(np.sqrt(np.prod(gt[2:])) * trace['factor'][i])
            np.testing.assert_allclose(nominal, trace['gt_side_network_px'][i], rtol=1e-12, atol=1e-12)
            row.update(raw_iou=raw_iou, adapted_iou=adapted, dense_best_iou=float(grid[i]),
                       fully_covered=full, raw_error_center_l2=float(np.linalg.norm(error[:2])),
                       raw_error_log_side=float(error[2]), readout_gap=float(grid[i])-adapted,
                       gt_side_network_px=nominal, category=category(raw_iou, adapted, full, grid[i]))
        result.append(row)
    return result


def reduce_cells(rows):
    valid = [r for r in rows if r['valid_current']]
    bad = [r for r in valid if r['raw_iou'] < .2]
    metrics = dict(samples=len(rows), valid_current=len(valid), invalid_current=len(rows)-len(valid), raw_severe=len(bad))
    metrics.update({k+'_count': sum(r['category'] == k for r in bad) for k in CLASSES})
    if sum(metrics[k+'_count'] for k in CLASSES) != len(bad):
        raise ValueError('Non-exhaustive or overlapping evidence classes')
    for stratum, rr in (('all', valid), ('raw_severe', bad), ('raw_good', [r for r in valid if r['raw_iou'] >= .5])):
        metrics[stratum+'_samples'] = len(rr)
        for field in ('raw_iou', 'adapted_iou', 'dense_best_iou', 'readout_gap', 'raw_error_center_l2', 'gt_side_network_px'):
            metrics[stratum+'_'+field] = float(np.mean([r[field] for r in rr])) if rr else None
    return metrics


def summarize(cells):
    groups = sorted({r['group'] for r in cells})
    policies, primary = [], []
    for context, origins in [(o, (o,)) for o in ORIGINS] + [('provider_style', ORIGINS[:2])]:
        ratios = {k: [] for k in CLASSES}
        supported = []
        for stratum in ('all', 'raw_severe', 'raw_good'):
            per_group = []
            for group in groups:
                rr = [r for r in cells if r['group'] == group and r['origin'] in origins and r['valid_current']
                      and (stratum == 'all' or (r['raw_iou'] < .2 if stratum == 'raw_severe' else r['raw_iou'] >= .5))]
                if rr:
                    per_group.append((group, rr))
            row = dict(context=context, stratum=stratum, supported_groups=len(per_group),
                       unsupported_groups='|'.join(g for g in groups if g not in {g for g, _ in per_group}),
                       records=sum(len(rr) for _, rr in per_group))
            for name in ('raw_iou', 'adapted_iou', 'dense_best_iou', 'readout_gap', 'raw_error_center_l2', 'gt_side_network_px'):
                row[name] = float(np.mean([np.mean([r[name] for r in rr]) for _, rr in per_group])) if per_group else None
            for k in CLASSES:
                values = [float(np.mean([r['category'] == k for r in rr])) for _, rr in per_group]
                row[k+'_fraction'] = float(np.mean(values)) if values else None
                if context == 'provider_style' and stratum == 'raw_severe':
                    ratios[k] = values
            if context == 'provider_style' and stratum == 'raw_severe':
                supported = [g for g, _ in per_group]
            policies.append(row)
        if context == 'provider_style':
            for k in CLASSES:
                values = ratios[k]
                stats = previous.contrast(values)
                primary.append(dict(category=k, primary=True, supported_groups=len(supported),
                                    unsupported_groups='|'.join(g for g in groups if g not in supported),
                                    mean_fraction=stats['mean_delta'], median_fraction=stats['median_delta'],
                                    ci_low=stats['ci_low'], ci_high=stats['ci_high'],
                                    interpretation='descriptive source bootstrap; dependent class proportions, no adoption test'))
    return policies, primary


def verify(root):
    root = Path(root).resolve()
    p = old.read_sealed(root/'manifest.json'); d = old.read_sealed(root/'stage222_decision.json')
    if p['schema'] != SCHEMA or d['schema'] != SCHEMA or d['plan_fingerprint'] != p['fingerprint'] or d['source_stage221'] != SOURCE221 or d['status'] != 'REVIEW_LOCALIZATION_EVIDENCE':
        raise ValueError('Wrong evidence result identity')
    if set(d['table_sha256']) != {f'stage222_{n}.csv' for n in TABLES} or len(d['primary']) != 4 or any(d[k] != 0 for k in ('new_training_updates', 'new_image_reads', 'new_gpu_inference_calls', 'new_full_trajectories', 'confirmation_groups_opened')):
        raise ValueError('Incomplete evidence result')
    for n, h in (d['table_sha256'] | d['artifact_sha256']).items():
        pp = (root/n).resolve()
        if not pp.is_relative_to(root) or file_hash(pp) != h:
            raise ValueError('Changed output artifact')
    for n, h in p['code_sha256'].items():
        if file_hash(REPO/n) != h:
            raise ValueError('Changed dependency: '+n)
    if file_hash(PROTOCOL) != p['protocol_sha256']:
        raise ValueError('Changed protocol')
    print('Stage-222 verified; descriptive cached evidence, no new AUC', flush=True)


def experiment(root, plan, config, entries, p215, d215, p216, d216, ref221):
    start = perf_counter(); verified = {'215': {}, '216': {}}
    tables = {n: [] for n in TABLES}
    tables['selection'] = p216['selection']
    progress = OverallProgress('Stage-222', len(plan['entries'])*2*len(ORIGINS)+1, unit='checks')
    progress.report('Cached held localization evidence')
    for spec in plan['entries']:
        seq, g = spec['sequence'], spec['group']
        fold = next(f for f in plan['folds'] if g in f['held'])
        gt = entries[seq][1]
        frames = [r['frame'] for r in p216['selection'] if r['sequence'] == seq]
        if frames != list(source.prior.source.c.protocol.uniform_frames(len(gt)-2)):
            raise ValueError('Frozen frame selection changed')
        for seed in (2132, 2133):
            for origin in ORIGINS:
                path = source.prior.source.previous.path_for(Path(config['stage215_root']), 'full', seq, seed, origin)
                meta = old.read_sealed(path.with_suffix('.receipt.json'))['metadata']
                if any(meta[k] != v for k, v in dict(plan_fingerprint=p215['fingerprint'],sequence=seq,fold=fold['fold'],seed=seed,arm=origin,phase='full').items()):
                    raise ValueError('Wrong held trace identity')
                trace = source.bound(Path(config['stage215_root']), d215, path.relative_to(config['stage215_root']).as_posix(), meta, verified['215'])
                rel = f'full/{seq}__s{seed}__{origin}.npz'
                meta216 = old.read_sealed((Path(config['stage216_root'])/rel).with_suffix('.receipt.json'))['metadata']
                if any(meta216[k] != v for k, v in dict(plan_fingerprint=p216['fingerprint'],sequence=seq,origin=origin,seed=seed,frames=frames).items()):
                    raise ValueError('Wrong current replay identity')
                cache = source.bound(Path(config['stage216_root']), d216, rel, meta216, verified['216'])
                idx = np.asarray(frames)-1
                np.testing.assert_array_equal(cache['current_reference'], trace['reference_boxes'][idx])
                np.testing.assert_array_equal(cache['current_box'], trace['boxes'][idx])
                common = dict(fold=fold['fold'],group=g,sequence=seq,seed=seed,origin=origin)
                rows = observations(trace, frames, gt, plan['search_size'])
                for row in rows:
                    ref = ref221[g,seed,origin,row['frame']]
                    if (ref['valid_current'] == 'True') != row['valid_current']:
                        raise ValueError('GT support differs from Stage-221')
                    if row['valid_current']:
                        error = source.label(trace['reference_boxes'][row['frame']-1:row['frame']], gt[row['frame']:row['frame']+1])[0]
                        np.testing.assert_allclose(error,[float(ref['error_'+a]) for a in ('dx','dy','log_side')],rtol=1e-12,atol=1e-12)
                    tables['cells'].append(dict(common,**row))
                tables['group_metrics'].append(dict(common,**reduce_cells(rows)))
                tables['source_replay'].append(dict(common,samples=len(rows),current_boxes_exact=True,integer_coverage_exact=True,gt_scale_exact=True,error221_exact=True,dense_branch='adapted_only'))
                progress.advance()
    expected_keys={(r['group'],r['seed'],r['origin'],r['frame']) for r in tables['cells']}
    if expected_keys != set(ref221):
        raise ValueError('Incomplete Stage-221 held census')
    tables['policy_summary'], tables['primary'] = summarize(tables['cells'])
    tables['execution'] = [dict(device='cpu',seconds=perf_counter()-start,image_reads=0,cuda_calls=0,optimizer_updates=0)]
    publish_identical(root/'source_arrays.json', old.sealed(dict(plan_fingerprint=plan['fingerprint'],files=verified)))
    hashes = {}
    for name, rows in tables.items():
        path = root/f'stage222_{name}.csv'; write_csv(path,rows); hashes[path.name]=file_hash(path)
    decision = old.sealed(dict(schema=SCHEMA,status='REVIEW_LOCALIZATION_EVIDENCE',plan_fingerprint=plan['fingerprint'],source_stage221=SOURCE221,
                              new_training_updates=0,new_image_reads=0,new_gpu_inference_calls=0,new_full_trajectories=0,confirmation_groups_opened=0,
                              primary=tables['primary'],table_sha256=hashes,artifact_sha256={'source_arrays.json':file_hash(root/'source_arrays.json')}))
    publish_identical(root/'stage222_decision.json',decision)
    if progress.done != progress.total-1:
        raise ValueError('Incomplete progress accounting')
    progress.advance(); progress.report('Complete'); verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in (221,215,216,76,89,93,121,170,188,191):
        parser.add_argument(f'--stage{k}-root',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--check-results',action='store_true')
    args=parser.parse_args(); root=args.output_dir.resolve()
    if args.check_results:
        verify(root); return
    keys=(221,215,216,76,89,93,121,170,188,191)
    if any(getattr(args,f'stage{k}_root') is None for k in keys):
        parser.error('Need frozen cache roots; no checkpoint/GPU required')
    source.c.protocol.safe_output(root,[getattr(args,f'stage{k}_root') for k in keys]+[source.RESULTS])
    with writer_lock(root):
        if (root/'stage222_decision.json').exists():
            verify(root); return
        r221=args.stage221_root.resolve(); previous.verify(r221)
        p221=old.read_sealed(r221/'manifest.json');d221=old.read_sealed(r221/'stage221_decision.json')
        if d221['fingerprint'] != SOURCE221:
            raise ValueError('Need reviewed Stage-221 physical cache')
        d215,p215=source.prior.bundle(source.prior.DEFAULT_BUNDLE)
        d216,p216=source.source216(source.BUNDLE216)
        for k,d,p in ((215,d215,p215),(216,d216,p216)):
            src=getattr(args,f'stage{k}_root').resolve()
            if old.read_sealed(src/'manifest.json') != p or old.read_sealed(src/f'stage{k}_decision.json') != d:
                raise ValueError('Physical source differs')
            if any(p[n] != p221[n] for n in ('entries','folds','search_factor')):
                raise ValueError('Changed source roster/split')
        if p216['source_stage215'] != d215['fingerprint'] or p221['source_stage220'] != '82832d743f4497540486d699337eee7ab8cb1c959c08c43aed93fa064559370c':
            raise ValueError('Changed lineage')
        with (r221/'stage221_cells.csv').open(encoding='utf-8',newline='') as handle:
            rows=[r for r in csv.DictReader(handle) if r['split']=='held']
        ref={(r['group'],int(r['seed']),r['origin'],int(r['frame'])):r for r in rows}
        if len(ref) != len(rows) or len(rows) != 14592 or len(p221['entries']) != 38:
            raise ValueError('Incomplete reviewed held support')
        code={n:file_hash(REPO/n) for n in p221['code_sha256']}
        for n in ('tools/stage222_localization_evidence.py','tools/run_stage222_localization_evidence.sh','tools/stage207_state_common.py'):
            code[n]=file_hash(REPO/n)
        search_size=p215['runtime']['tracker']['config']['TEST']['SEARCH_SIZE']
        if search_size != 256 or p215['runtime']['tracker']['config']['TEST']['SEARCH_FACTOR'] != p221['search_factor']:
            raise ValueError('Changed executed search geometry')
        plan=old.sealed(dict(schema=SCHEMA,source_stage221=SOURCE221,source_stage215=d215['fingerprint'],source_stage216=d216['fingerprint'],
                             entries=p221['entries'],folds=p221['folds'],search_factor=p221['search_factor'],search_size=search_size,
                             code_sha256=code,protocol_sha256=file_hash(PROTOCOL),classes=list(CLASSES),policy='held fixed points, raw severe reference; adapted dense evidence only'))
        config={f'stage{k}_root':str(getattr(args,f'stage{k}_root').resolve()) for k in (215,216)}
        config['source_paths']={str(k):str(getattr(args,f'stage{k}_root').resolve()) for k in (76,89,93,121,170,188,191)}
        entries,dataset=source.runtime.load_entries(config,plan)
        source.c.protocol.safe_output(root,[dataset]);publish_identical(root/'manifest.json',plan)
        experiment(root,plan,config,entries,p215,d215,p216,d216,ref)


if __name__=='__main__':
    main()
