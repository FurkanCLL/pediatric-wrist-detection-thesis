"""Immutable V3 settings, source identities and preservation boundaries."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from wrist_eval.common import ROOT, sha, identity, source_snapshot, verify_validation_files
from wrist_eval.adaptive_config import approved_settings
from wrist_analysis.evidence import preservation as earlier_inventory

PUBLIC = ROOT/'docs/evidence/V3'
LOCAL = ROOT/'outputs/V3/study_01'
CONFIG = ROOT/'configs/cg_alr_v3.json'
LOCK = PUBLIC/'freeze.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def settings():
    cfg = read(CONFIG); legacy, _ = approved_settings()
    require(identity(legacy) == cfg['historical_proposal_identity'], 'Historical numerical settings changed')
    require(cfg['maximum_local_views'] == 2 and cfg['maximum_model_calls'] == 3
            and cfg['evaluation']['bootstrap_replicates'] == 500 and cfg['evaluation']['bootstrap_seed'] == 314159,
            'V3 protocol limits differ')
    return cfg, legacy


def source_identity():
    files = sorted((ROOT/'src/wrist_v3').glob('*.py'))+sorted((ROOT/'tests').glob('test_v3_*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in files}


def freeze():
    cfg, _ = settings()
    design = PUBLIC/'V3_DESIGN.md'; mirror = ROOT/'docs/evidence/phase_5b_v3/V3_DESIGN.md'
    require(design.read_bytes() == mirror.read_bytes(), 'Requested design copies differ')
    tests = read(PUBLIC/'pre_gpu_tests.json')
    require(tests['passed'] and tests['source_identity'] == source_identity(), 'Current V3 code has not passed the test gate')
    write_new(LOCK, {'algorithm': cfg['algorithm'], 'source_identity': source_identity(), 'configuration_sha256': sha(CONFIG),
        'configuration_identity': identity(cfg), 'design_sha256': sha(design), 'design_mirror_sha256': sha(mirror),
        'test_gate_sha256': sha(PUBLIC/'pre_gpu_tests.json'), 'performance_calculated_before_freeze': False,
        'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})


def frozen(require_pushed=True):
    cfg, legacy = settings(); lock = read(LOCK)
    require(lock['source_identity'] == source_identity() and lock['configuration_sha256'] == sha(CONFIG)
        and lock['configuration_identity'] == identity(cfg) and lock['design_sha256'] == sha(PUBLIC/'V3_DESIGN.md')
        and lock['test_gate_sha256'] == sha(PUBLIC/'pre_gpu_tests.json'), 'Frozen V3 implementation/configuration changed')
    require(sha(ROOT/'docs/evidence/phase_5b_v3/V3_DESIGN.md') == lock['design_mirror_sha256'], 'Frozen design mirror changed')
    if require_pushed:
        changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', 'src/wrist_v3', 'configs/cg_alr_v3.json',
                                           'tests/test_v3_*.py', str(PUBLIC/'V3_DESIGN.md'), str(LOCK)], cwd=ROOT, text=True)
        require(not changed.strip(), 'Frozen implementation must be committed before execution')
        require(not subprocess.check_output(['git', 'rev-list', 'origin/main..HEAD'], cwd=ROOT, text=True).strip(),
                'Push the frozen implementation before execution')
        require(subprocess.check_output(['git', 'ls-files', 'src/wrist_v3/core.py'], cwd=ROOT, text=True).strip(), 'V3 is not committed')
    return cfg, legacy, lock


def phase_inventory():
    current = earlier_inventory()
    for name in ('phase_5a_analysis', 'rarity_training_completion', 'rarity_training_readiness',
                 'error_guided_training_feasibility', 'class_aware_design'):
        for p in sorted((ROOT/'outputs'/name).rglob('*')):
            if p.is_file() and not p.name.startswith('.env'):
                stat = p.stat()
                current['private'][p.relative_to(ROOT).as_posix()] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    **({} if p.suffix == '.pt' else {'sha256': sha(p)})}
    current['data_stat_identity'] = identity(source_snapshot())
    current['best_checkpoints_sha256'] = {name: sha(ROOT/f'outputs/training/{name}/weights/best.pt')
                                        for name in ('standard_baseline', 'rarity_only')}
    return current


def preserve(stage):
    path = ROOT/'outputs/V3/preservation_before.json'
    current = phase_inventory()
    if stage == 'before':
        write_new(path, current)
    else:
        before = read(path)
        require(all(current['tracked'].get(name) == digest for name, digest in before['tracked'].items()), 'Historical tracked file changed')
        for key in ('private', 'data_stat_identity', 'best_checkpoints_sha256'):
            require(current[key] == before[key], 'Historical preservation changed: '+key)
    return {'passed': True, 'stage': stage, 'historical_tracked_files': len(read(path)['tracked']),
            'historical_private_files': len(current['private']), 'data_stat_identity': current['data_stat_identity'],
            'best_checkpoint_bytes_unchanged': current['best_checkpoints_sha256'], 'other_checkpoint_protection': 'size and modification time'}


def metadata():
    # The GPU runner receives dimensions/identities only, never the annotation records.
    linkage = read(ROOT/'outputs/rarity_comparison/study_01/validation_linkage.json')
    manifest = read(ROOT/'outputs/rarity_comparison/study_01/manifest.json')
    require(identity([v['stem'] for v in linkage]) == manifest['validation_membership_sha256'], 'Validation membership differs')
    rows = [{'filestem': v['stem'], 'patient_id': v['patient_id'], 'width': v['width'], 'height': v['height'], 'split': 'val'} for v in linkage]
    require(len(rows) == 3088 and len({v['patient_id'] for v in rows}) == 914, 'Validation population differs')
    return rows, {v['stem']: v['net_shape'] for v in linkage}


def references(condition):
    result = read(ROOT/'docs/evidence/rarity_inference_comparison/results.json')
    receipt = read(ROOT/'docs/evidence/rarity_inference_comparison/publication_checks.json')
    require(sha(ROOT/'docs/evidence/rarity_inference_comparison/results.json') == receipt['artifacts_sha256']['results.json'], 'Official results differ')
    base = 'A' if condition == 'standard' else 'D'
    path = ROOT/('outputs/error_analysis_sahi/full/standard_serial/predictions.jsonl' if base == 'A'
                else 'outputs/rarity_comparison/study_01/full/predictions.jsonl')
    require(sha(path) == result['conditions'][base]['source']['prediction_sha256'], 'Global reference changed')
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows, _ = metadata()
    require([v['stem'] for v in values] == [r['filestem'] for r in rows], 'Global reference order differs')
    return {v['stem']: np.asarray(v['pred'], np.float32).reshape(-1, 6) for v in values}
