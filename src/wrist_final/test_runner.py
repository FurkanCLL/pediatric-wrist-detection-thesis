"""Exactly one full held-out pass per frozen condition, in A–F order."""

import argparse
from collections import Counter
import json
import sys
import time
import traceback

import numpy as np

from .execution import RUN, RESULTS, locked, require, access_guard
from .protocol import ROOT, read, sha, write_new
from .test_data import metadata, PREPARED, initialize
from .inference import create_predictor
from wrist_eval.metrics import distribution
from wrist_compare.infer import parameter_digest
from wrist_v3.runner import serializable


def run(condition):
    cfg, _ = locked(); sys.addaudithook(access_guard)
    order = cfg['execution']['condition_order']; position = order.index(condition)
    for earlier in order[:position]:
        require(read(RUN/earlier/'summary.json')['passed'], 'Earlier condition has not completed')
    destination = RUN/condition
    require(not destination.exists(), 'Preserve the existing condition; no automatic repeat is allowed')
    destination.mkdir()
    try:
        execute(condition, destination, cfg)
    except BaseException as error:
        write_new(destination/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise


def execute(condition, destination, cfg):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42, deterministic=True)
    definition = next(v for v in cfg['conditions'] if v['id'] == condition)
    method = definition['inference']; rows = metadata()
    require(len(rows) == 3178, 'Test membership differs')
    before_load = time.perf_counter()
    predictor = create_predictor(condition, destination, phase5d_authorized=True)
    model = predictor.view.model; original_state = parameter_digest(model)
    load_seconds = time.perf_counter()-before_load
    warmup = time.perf_counter()
    for _ in range(3):
        predictor.view(np.full((1024, 576, 3), 114, np.uint8), None)
    torch.cuda.synchronize()
    require(predictor.physical_calls == 3, 'Synthetic warmup call accounting differs')
    warmup_seconds = time.perf_counter()-warmup
    global_reference = None
    if method == 'cg_alr_v3':
        base = 'final_A' if condition == 'final_C' else 'final_D'
        reference = RUN/base/'predictions.jsonl'
        require(sha(reference) == read(RUN/base/'summary.json')['prediction_sha256'], 'Standard reference changed')
        global_reference = {v['stem']: np.asarray(v['pred'], np.float32).reshape(-1, 6)
                            for v in (json.loads(line) for line in reference.read_text(encoding='utf-8').splitlines())}
    timings, usage = [], Counter(); loop_start = time.perf_counter()
    with (destination/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        bundle_stream = (destination/'bundles.jsonl').open('x', encoding='utf-8') if method == 'cg_alr_v3' else None
        try:
            for index, row in enumerate(rows):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                image = cv2.imdecode(np.fromfile(PREPARED/'images/test'/(row['stem']+'.png'), np.uint8), cv2.IMREAD_COLOR)
                require(image is not None and image.shape == (row['height'], row['width'], 3), 'Test image dimensions differ')
                decoded = time.perf_counter()
                result = predictor(image, method, row['net_shape'], started_at=began)
                if global_reference is not None:
                    require(np.array_equal(result['diagnostics']['global'], global_reference[row['stem']]), 'V3 global differs from final_A/D')
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-began
                calls = result['actual_model_calls']
                require(calls == (5 if method == 'sahi_context' else 1) if method != 'cg_alr_v3' else 1 <= calls <= 3, 'Unexpected actual calls')
                timing = {'e2e_seconds': elapsed, 'decode_seconds': decoded-began, 'forward_passes': calls,
                    'paired_global_e2e_seconds': result['paired_global_e2e_seconds'],
                    'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'reserved_peak_bytes': torch.cuda.max_memory_reserved()}
                timings.append(timing); usage['calls'] += calls
                usage['slices' if method == 'sahi_context' else 'local_calls'] += 4 if method == 'sahi_context' else calls-1
                stream.write(json.dumps({'stem': row['stem'], 'pred': result['predictions'].tolist(), 'timing': timing}, allow_nan=False)+'\n'); stream.flush()
                if bundle_stream:
                    bundle_stream.write(json.dumps(serializable({'stem': row['stem'], **result['diagnostics']}), allow_nan=False)+'\n'); bundle_stream.flush()
                if index % 100 == 0:
                    print(f'{condition}: {index+1}/{len(rows)}', flush=True)
        finally:
            if bundle_stream:
                bundle_stream.close()
    require(parameter_digest(model) == original_state, 'Model state changed')
    require(predictor.physical_calls == usage['calls']+3, 'Full physical-call accounting differs')
    predictor.close()
    timing = {k: distribution([v[k] for v in timings if v[k] is not None]) for k in timings[0]}
    receipt = {'passed': True, **definition, 'split': 'test', 'images': len(rows), 'patients': 913,
        'timing': timing, 'usage': dict(usage), 'physical_calls_including_warmups': predictor.physical_calls,
        'warmup_calls': 3, 'load_seconds': load_seconds, 'warmup_seconds': warmup_seconds,
        'loop_seconds_including_export': time.perf_counter()-loop_start, 'model_state_sha256_before_after': original_state,
        'checkpoint_sha256': cfg['checkpoints'][definition['training']]['sha256'],
        'prediction_sha256': sha(destination/'predictions.jsonl'), 'global_reference_equal': global_reference is not None,
        'execution_freeze_sha256': sha(RESULTS/'execution_freeze.json')}
    if method == 'cg_alr_v3':
        receipt['bundles_sha256'] = sha(destination/'bundles.jsonl')
        receipt['mean_paired_latency_ratio'] = timing['e2e_seconds']['mean']/timing['paired_global_e2e_seconds']['mean']
    write_new(destination/'summary.json', receipt)
    print(f'{condition} complete: {len(rows)} images, {usage["calls"]} calls', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition', choices=['init', *['final_'+v for v in 'ABCDEF']])
    args = parser.parse_args()
    initialize() if args.condition == 'init' else run(args.condition)
