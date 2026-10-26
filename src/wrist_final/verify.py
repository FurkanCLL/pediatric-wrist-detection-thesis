"""Prepare Phase 5C receipts using synthetic tests and saved validation outputs only."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import numpy as np

from .protocol import (ROOT, PUBLIC, LOCAL, MATRIX, read, sha, identity, write_new, source_identities,
                       dependency_identity, verify_frozen, install_phase5c_guard, rectangular_shapes, agents_local_only)


def checkpoints():
    prior = read(ROOT/'docs/evidence/V3/execution_manifest.json')['checkpoints']
    return {name: {'path': f'outputs/training/{directory}/weights/best.pt', **prior[condition]}
            for name, directory, condition in (('standard', 'standard_baseline', 'standard'),
                                               ('rarity_only', 'rarity_only', 'rarity'))}


def snapshot():
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    return {'tracked': {p: sha(ROOT/p) for p in names if not Path(p).name.startswith('.env')},
            'checkpoints': {name: sha(ROOT/value['path']) for name, value in checkpoints().items()}}


def build_matrix():
    from wrist_v3.protocol import frozen
    frozen(require_pushed=False)
    artifacts = ['configs/inference_feasibility.json', 'configs/cg_alr_v3.json',
        'configs/adaptive_refinement_proposed.json', 'configs/training_official.json', 'configs/rarity_training.json',
        'docs/evidence/dataset_preparation/frozen_manifest_hashes.json',
        'docs/evidence/dataset_preparation/aggregate_statistics.json', 'docs/evidence/dataset_preparation/validation.json',
        'docs/evidence/error_analysis_sahi/compatibility_verified.json', 'docs/evidence/V3/freeze.json',
        'docs/evidence/V3/publication_checks.json', 'docs/evidence/V3/verification.json',
        'docs/evidence/rarity_inference_comparison/publication_checks.json',
        'docs/evidence/final_evaluation/FINAL_EVALUATION_PROTOCOL.md']
    weights = {
        'training_standard': {'final_D': 1, 'final_A': -1},
        'training_sahi': {'final_E': 1, 'final_B': -1},
        'training_v3': {'final_F': 1, 'final_C': -1},
        'sahi_standard_training': {'final_B': 1, 'final_A': -1},
        'sahi_rarity_training': {'final_E': 1, 'final_D': -1},
        'v3_standard_training': {'final_C': 1, 'final_A': -1},
        'v3_rarity_training': {'final_F': 1, 'final_D': -1},
        'v3_vs_sahi_standard_training': {'final_C': 1, 'final_B': -1},
        'v3_vs_sahi_rarity_training': {'final_F': 1, 'final_E': -1},
        'training_sahi_interaction': {'final_E': 1, 'final_D': -1, 'final_B': -1, 'final_A': 1},
        'training_v3_interaction': {'final_F': 1, 'final_D': -1, 'final_C': -1, 'final_A': 1}}
    return {'schema_version': 1, 'phase': '5C', 'status': 'frozen preparation; test execution requires separate authorization',
        'historical_base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'primary_proposed_configuration': 'final_C', 'primary_comparison': 'v3_standard_training',
        'conditions': [{'id': 'final_'+letter, 'display_id': letter, 'training': training, 'inference': inference}
            for letter, training, inference in [('A', 'standard', 'standard'), ('B', 'standard', 'sahi_context'),
                ('C', 'standard', 'cg_alr_v3'), ('D', 'rarity_only', 'standard'),
                ('E', 'rarity_only', 'sahi_context'), ('F', 'rarity_only', 'cg_alr_v3')]],
        'historical_ids_unchanged': {'C': 'CG-ALR V2', 'F': 'CG-ALR V2'},
        'checkpoints': checkpoints(),
        'population': {'split': 'test', 'images': 3178, 'patients': 913, 'annotations': 7361, 'classes': 8,
            'source': 'docs/evidence/dataset_preparation/aggregate_statistics.json',
            'split_manifest_sha256': read(ROOT/'docs/evidence/dataset_preparation/frozen_manifest_hashes.json')['split_manifest_sha256']},
        'primary_metric': 'eight-class mAP50-95',
        'secondary_metrics': ['mAP50', 'mAP75', 'all eight per-class AP50/AP75/AP50-95', 'fracture AP',
                              'TP', 'FP', 'FN', 'precision', 'recall'],
        'operating_point': {'confidence': .25, 'iou': .50},
        'AP_order': {'prediction_dtype': 'float32', 'canonical_matching_rows': ['descending unmodified confidence',
            'class ID', 'x1', 'y1', 'x2', 'y2', 'identical-row occurrence'],
            'AP_sort': ['descending unmodified confidence', 'frozen lexical image index',
                        'canonical row rank', 'bootstrap image occurrence'],
            'native_matcher_and_integration': 'unchanged pinned native_stats and compute_ap; ten IoUs .50:.05:.95',
            'scores_rounded_or_perturbed': False, 'same_policy_for_SAHI_new_rows': True},
        'bootstrap': {'replicates': 500, 'seed': 314159, 'unit': 'patient', 'confidence_level': .95},
        'bootstrap_absent_classes': 'undefined; omit affected paired class sample and eight-class overall, report effective counts',
        'comparisons': weights, 'secondary_intervals': 'nominal percentile, no multiplicity adjustment; no causal claim',
        'numerical_configuration': {'common_path': 'configs/inference_feasibility.json', 'v3_path': 'configs/cg_alr_v3.json',
            'v3_proposal_path': 'configs/adaptive_refinement_proposed.json', 'input_size': 768, 'reference_shape_batch': 12,
            'precision': 'FP32 unfused EMA; FP16 normalized input and CUDA FP16 autocast',
            'retained_confidence_floor': .001, 'per_view_nms_iou': .7, 'max_detections': 300,
            'multi_label': True, 'class_agnostic_nms': False, 'seed': 42,
            'sahi_variant': 'sahi_context', 'sahi_slice_fraction': .667, 'sahi_overlap': .2,
            'sahi_merge': 'class-aware NMS / IoU .70 with full-image prediction',
            'V3': 'identical CG-ALR V3.0 on both checkpoints; no numerical changes'},
        'execution': {'test_authorized_in_phase5c': False, 'test_loader_available_in_phase5c': False,
            'one_full_pass_per_condition_after_authorization': True, 'condition_order': ['final_A', 'final_B', 'final_C', 'final_D', 'final_E', 'final_F'],
            'serial_batch': 1, 'synthetic_warmups_per_condition': 3, 'no_test_pilot_or_replay': True,
            'maximum_calls': {'standard': 1, 'sahi_context': 5, 'cg_alr_v3': 3},
            'physical_calls_verified_by_model_hook': True, 'model_state_digest_before_after': True,
            'outputs': 'outputs/final_evaluation/phase5d/<new-run>/final_A through final_F',
            'cost': ['mean/median/p95 decode-to-final-output latency', 'actual forwards', 'slice/ROI count',
                     'peak CUDA allocated/reserved memory', 'paired global prefix for V3; no extra forward'],
            'export_and_offline_metrics_excluded_from_per_image_latency': True,
            'failure': 'stop affected condition, preserve partial evidence; no policy repair from test outcomes'},
        'engineering_gates': {'comparison': ['final_C-final_A', 'final_F-final_D'], 'overall_delta_positive': True,
            'overall_lower95_positive': True, 'map50_nondecreasing': True, 'fracture_AP_points_nondecreasing': True,
            'fracture_lower95_tolerance': -.005, 'FP_ratio_max': 1.10, 'fracture_FP_ratio_max': 1.10,
            'V3_mean_paired_latency_ratio_max': 2.0, 'maximum_V3_calls': 3,
            'meaning': 'historical engineering criteria; not clinical safety or noninferiority margins'},
        'identities': {'normalized_project_source_sha256': source_identities(),
            'artifact_sha256': {p: sha(ROOT/p) for p in artifacts}, 'dependencies': dependency_identity()}}


def validation_checks():
    from .metrics import canonical_predictions
    selection = set(read(ROOT/'outputs/error_analysis_sahi/pilot/standard_serial/selection.json'))
    linkage = read(ROOT/'outputs/rarity_comparison/study_01/validation_linkage.json')
    shapes = rectangular_shapes(linkage)
    if len(linkage) != 3088 or any(shapes[r['stem']] != r['net_shape'] for r in linkage):
        raise ValueError('Rectangular shapes differ from the saved validation reference')
    by_stem = {r['stem']: r for r in linkage}
    old = read(ROOT/'docs/evidence/rarity_inference_comparison/results.json')['conditions']
    paths = {
        'final_A': ('outputs/error_analysis_sahi/full/standard_serial/predictions.jsonl', old['A']['source']['prediction_sha256']),
        'final_B': ('outputs/error_analysis_sahi/full/sahi_context/predictions.jsonl', old['B']['source']['prediction_sha256']),
        'final_D': ('outputs/rarity_comparison/study_01/full/predictions.jsonl', old['D']['source']['prediction_sha256']),
        'final_E': ('outputs/rarity_inference_comparison/study_01/full_E/predictions.jsonl', old['E']['source']['prediction_sha256'])}
    for condition, name in (('final_C', 'standard'), ('final_F', 'rarity')):
        receipt = read(ROOT/f'docs/evidence/V3/full_{name}.json')
        paths[condition] = (f'outputs/V3/study_01/full_{name}/predictions.jsonl', receipt['prediction_sha256'])
    result = {}
    for method, (name, expected) in paths.items():
        path = ROOT/name
        if sha(path) != expected:
            raise ValueError('Saved validation predictions changed')
        checked = 0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item['stem'] not in selection:
                    continue
                row = by_stem[item['stem']]
                pred = np.asarray(item['pred'], np.float32).reshape(-1, 6)
                a = canonical_predictions(pred, row['width'], row['height'])
                b = canonical_predictions(pred[::-1], row['width'], row['height'])
                if not np.array_equal(a, b):
                    raise ValueError('Canonical content identities depend on exported row order')
                checked += 1
        if checked != 24:
            raise ValueError('Fixed validation verification membership differs')
        result[method] = {'saved_source_sha256': expected, 'images_checked': checked, 'row_permutation_invariant': True}
    return {'saved_validation_predictions': result, 'all_3088_reference_shapes_reproduced': True,
            'validation_images_or_labels_opened': 0, 'validation_performance_calculated': False}


def run_tests():
    path = LOCAL/'tests.json'
    current = source_identities()
    if path.exists():
        value = read(path)
        if not value['passed'] or value['source_identity'] != current:
            raise ValueError('Preserve the earlier test receipt and rerun changed sources in a new attempt')
        return value
    temporary = LOCAL/'tmp'; temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary.resolve())
    began = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (LOCAL/'tests.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    value = {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'failures': len(result.failures),
             'errors': len(result.errors), 'seconds': time.perf_counter()-began, 'source_identity': current,
             'private_log_sha256': sha(LOCAL/'tests.log'), 'dataset_access_guard_active': True}
    write_new(path, value)
    if not value['passed']:
        raise ValueError('Tests failed; inspect the preserved private test log')
    return value


def preflight():
    before = read(LOCAL/'preservation_before.json')
    agents = agents_local_only()
    tests = run_tests()
    if not MATRIX.exists():
        write_new(MATRIX, build_matrix())
    cfg = verify_frozen(require_preflight=False)
    aggregate = read(ROOT/'docs/evidence/dataset_preparation/aggregate_statistics.json')['splits']['test']
    if [aggregate[k] for k in ('images', 'patients', 'objects')] != [3178, 913, 7361]:
        raise ValueError('Saved aggregate test population differs from the approved counts')
    checks = validation_checks()
    after = snapshot()
    if before != after:
        raise ValueError('Historical tracked files or checkpoint bytes changed')
    for name, value in cfg['checkpoints'].items():
        if after['checkpoints'][name] != value['sha256']:
            raise ValueError('Final checkpoint hash differs')
    # The receipts refer to pre-existing verified aggregates; no mixed record container is read here.
    for directory in ('V3', 'rarity_inference_comparison'):
        receipt = read(ROOT/f'docs/evidence/{directory}/publication_checks.json')
        for name, expected in receipt['artifacts_sha256'].items():
            if sha(ROOT/f'docs/evidence/{directory}'/name) != expected:
                raise ValueError('Historical publication receipt no longer matches')
    write_new(PUBLIC/'preflight_verification.json', {'phase': '5C', 'status': 'READY',
        'ready_for': 'separately authorized Phase 5D; no test execution authorized by this receipt',
        'matrix_sha256': sha(MATRIX), 'matrix_identity': identity(cfg), 'protocol_sha256': sha(PUBLIC/'FINAL_EVALUATION_PROTOCOL.md'),
        'tests': {k: v for k, v in tests.items() if k != 'source_identity'}, 'validation_only_checks': checks,
        'agents': agents, 'historical_preservation': {'passed': True, 'tracked_files': len(before['tracked']),
            'tracked_inventory_sha256': identity(before['tracked']), 'best_checkpoint_sha256': after['checkpoints'],
            'prior_publication_receipts_verified': True, 'historical_output_directories_written': False},
        'test_boundaries': {'test_images_opened': 0, 'test_annotations_opened': 0, 'mixed_annotation_containers_opened': 0,
            'test_inference_calls': 0, 'test_metrics_calculated': False, 'new_training': False,
            'actual_test_membership_annotation_and_byte_checks': 'deferred until Phase 5D authorization; only frozen preparation aggregates used'},
        'execution_command': '.venv/Scripts/python.exe -m wrist_final.verify preflight',
        'no_final_performance_claim': True})
    print(f'Phase 5C READY: {tests["tests"]} tests; six saved validation sources; no test-content access', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baseline', 'preflight'))
    args = parser.parse_args()
    install_phase5c_guard()
    if args.stage == 'baseline':
        agents_local_only()
        write_new(LOCAL/'preservation_before.json', snapshot())
        print('Historical tracked-file and checkpoint baseline saved without dataset access', flush=True)
    else:
        preflight()


if __name__ == '__main__':
    main()
