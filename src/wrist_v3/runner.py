"""One technical pilot and one complete V3 run per predefined checkpoint."""

import argparse
from collections import Counter
import json
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

from wrist_eval.adaptive_study import ViewRunner
from wrist_eval.common import image_path, verify_validation_files, identity, sha
from wrist_eval.metrics import distribution
from wrist_compare.common import checked_checkpoint, install_access_guard
from wrist_compare.infer import parameter_digest
from wrist_train.protocol import environment_check, load_protocol
from .core import predict
from .protocol import ROOT, LOCAL, PUBLIC, read, write_new, frozen, metadata, references, preserve, require


def initialize():
    cfg, _, lock = frozen()
    LOCAL.mkdir(parents=True, exist_ok=False)
    rows, shapes = metadata()
    files = verify_validation_files(rows)
    environment = environment_check(load_protocol()[0])
    checkpoints = {}
    for condition, name in (('A', 'standard'), ('D', 'rarity')):
        payload, receipt = checked_checkpoint(condition)
        require(receipt['sha256'] == cfg['checkpoints'][name]['sha256'], 'Wrong checkpoint')
        checkpoints[name] = receipt
        del payload
        references(name)
    pilot = read(ROOT/'outputs/error_analysis_sahi/pilot/standard_serial/selection.json')
    require(len(pilot) == 24 and len(set(pilot)) == 24 and set(pilot).issubset({r['filestem'] for r in rows}), 'Pilot differs')
    write_new(LOCAL/'manifest.json', {'freeze_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'freeze_sha256': sha(PUBLIC/'freeze.json'), 'lock': lock, 'environment': environment,
        'checkpoints': checkpoints, 'prepared_validation_bytes': files, 'membership': identity([r['filestem'] for r in rows]),
        'pilot': pilot, 'shape_profile_sha256': identity(shapes), 'configuration': cfg})
    print('V3 preflight passes for both fixed checkpoints and frozen validation membership', flush=True)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def execute(stage, condition):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    cfg, legacy, _ = frozen()
    manifest = read(LOCAL/'manifest.json')
    require(manifest['freeze_sha256'] == sha(PUBLIC/'freeze.json'), 'Execution freeze differs')
    rows, shapes = metadata(); prior = references(condition)
    if stage == 'pilot':
        by_stem = {r['filestem']: r for r in rows}; rows = [by_stem[stem] for stem in manifest['pilot']]
    else:
        # Both checkpoints are planned and technically gated before either full performance run.
        for name in cfg['checkpoints']:
            gate = read(LOCAL/f'pilot_{name}/summary.json')
            require(gate['passed'] and gate['freeze_sha256'] == sha(PUBLIC/'freeze.json'), 'Both current technical pilots must pass first')
    destination = LOCAL/f'{stage}_{condition}'
    destination.mkdir(exist_ok=False)
    write_new(destination/'selection.json', [r['filestem'] for r in rows])
    init_seeds(42, deterministic=True)
    began = time.perf_counter()
    payload, receipt = checked_checkpoint(cfg['checkpoints'][condition]['baseline'])
    require(receipt['sha256'] == cfg['checkpoints'][condition]['sha256'], 'Checkpoint identity differs')
    model = payload['ema'].float().eval().to('cuda:0').requires_grad_(False)
    del payload
    before = parameter_digest(model)
    with patch('wrist_eval.common.LOCAL', destination):
        view_runner = ViewRunner(model)
    forward_count = [0]
    def count_forward(module, inputs):
        forward_count[0] += 1
    hook = model.register_forward_pre_hook(count_forward)
    load_seconds = time.perf_counter()-began
    warmup_start = time.perf_counter()
    for _ in range(3):
        view_runner(np.full((1024, 576, 3), 114, np.uint8), None)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-warmup_start
    require(forward_count[0] == 3, 'Warmup forward count differs')

    def infer(row):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        call_before = forward_count[0]; began = time.perf_counter()
        image = cv2.imdecode(np.fromfile(image_path(row), np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.shape[:2] == (row['height'], row['width']), 'Prepared image dimensions differ')
        decoded = time.perf_counter(); global_elapsed = [None]
        def view(crop, shape):
            value = view_runner(crop, shape)
            if shape is not None:
                global_elapsed[0] = time.perf_counter()-began
                require(np.array_equal(value['predictions'], prior[row['filestem']]), 'Global output differs from frozen A/D')
            return value
        result = predict(image, view, shapes[row['filestem']], legacy, cfg)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        require(forward_count[0]-call_before == result['passes'] <= 3, 'Actual physical call count differs')
        timing = {'e2e_seconds': elapsed, 'paired_global_e2e_seconds': global_elapsed[0], 'decode_seconds': decoded-began,
            'forward_passes': result['passes'], 'planning_seconds': result['planning_seconds'], 'fusion_seconds': result['fusion_seconds'],
            'gpu_forward_ms': sum(v['counters']['gpu_forward_ms'] for v in [result['global_view'], *result['local_views']]),
            'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'reserved_peak_bytes': torch.cuda.max_memory_reserved()}
        return result, timing

    timings, saved, usage, roi_hist = [], {}, Counter(), Counter()
    loop_start = time.perf_counter()
    with (destination/'predictions.jsonl').open('x', encoding='utf-8') as pred_stream, (destination/'bundles.jsonl').open('x', encoding='utf-8') as bundle_stream:
        for index, row in enumerate(rows):
            result, timing = infer(row)
            pred = result['predictions']; saved[row['filestem']] = pred
            timings.append(timing); roi_hist[len(result['plan']['rois'])] += 1
            usage.update({'calls': result['passes'], 'local_calls': len(result['local_views']),
                'score_updates': len(result['diagnostics']['score_changes']), 'reserved_evidence': len(result['diagnostics']['reserved_evidence']),
                'accepted_matches': len(result['diagnostics']['matches'])})
            pred_stream.write(json.dumps({'stem': row['filestem'], 'pred': pred.tolist(), 'timing': timing,
                'global_ids': result['diagnostics']['output_global_ids']}, allow_nan=False)+'\n'); pred_stream.flush()
            bundle_stream.write(json.dumps(serializable({'stem': row['filestem'], **result, 'timing': timing}), allow_nan=False)+'\n'); bundle_stream.flush()
            if index % 100 == 0:
                print(f'V3 {condition}/{stage}: {index+1}/{len(rows)}', flush=True)
    loop_seconds = time.perf_counter()-loop_start
    replay_calls = 0
    if stage == 'pilot':
        for row in rows[:3]:
            result, _ = infer(row); replay_calls += result['passes']
            require(np.array_equal(result['predictions'], saved[row['filestem']]), 'Pilot replay differs')
    else:
        pilot = [json.loads(line) for line in (LOCAL/f'pilot_{condition}/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        require(all(np.array_equal(saved[v['stem']], np.asarray(v['pred'], np.float32).reshape(-1, 6)) for v in pilot), 'Pilot/full outputs differ')
    require(parameter_digest(model) == before, 'Model state changed')
    hook.remove()
    require(forward_count[0] == usage['calls']+replay_calls+3, 'Complete physical-call accounting differs')
    summary = {key: distribution([t[key] for t in timings]) for key in timings[0]}
    require(summary['allocated_peak_bytes']['max']/2**30 < cfg['evaluation']['technical_peak_allocated_gib_max'], 'Memory feasibility gate failed')
    require(summary['e2e_seconds']['mean']*3088 < cfg['evaluation']['technical_projected_seconds_max'], 'Runtime feasibility gate failed')
    receipt = {'passed': True, 'condition': condition, 'stage': stage, 'images': len(rows), 'timing': summary,
        'usage': dict(usage), 'roi_count_images': {str(k): roi_hist[k] for k in range(3)}, 'physical_model_calls_including_warmups_replays': forward_count[0],
        'warmup_calls': 3, 'replay_calls': replay_calls, 'pilot_AP_calculated': False,
        'checkpoint_sha256': cfg['checkpoints'][condition]['sha256'], 'freeze_sha256': sha(PUBLIC/'freeze.json'),
        'prediction_sha256': sha(destination/'predictions.jsonl'), 'bundles_sha256': sha(destination/'bundles.jsonl'),
        'global_equals_A_or_D': True, 'in_memory_model_unchanged': True,
        'load_seconds': load_seconds, 'warmup_seconds': warmup_seconds, 'loop_seconds_including_export': loop_seconds,
        'mean_paired_latency_ratio': summary['e2e_seconds']['mean']/summary['paired_global_e2e_seconds']['mean']}
    write_new(destination/'summary.json', receipt)
    print(f'V3 {condition}/{stage} passes: {len(rows)} images, {usage["calls"]} population calls', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('init', 'pilot', 'full'))
    parser.add_argument('condition', nargs='?', choices=('standard', 'rarity'))
    args = parser.parse_args()
    if args.stage != 'init' and args.condition is None:
        parser.error('Select a checkpoint condition')
    install_access_guard()
    destination = LOCAL if args.stage == 'init' else LOCAL/f'{args.stage}_{args.condition}'
    require(not destination.exists(), 'Existing V3 evidence is preserved; do not rerun a successful stage')
    try:
        initialize() if args.stage == 'init' else execute(args.stage, args.condition)
    except BaseException as error:
        if destination.exists():
            write_new(destination/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
