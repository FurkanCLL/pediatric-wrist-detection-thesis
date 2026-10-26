"""Final protocol identities and the Phase 5C data-access boundary."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT/'docs/evidence/final_evaluation'
LOCAL = ROOT/'outputs/final_evaluation/phase5c'
MATRIX = PUBLIC/'final_experiment_matrix.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def settings():
    cfg = read(MATRIX)
    expected = [(name, training, inference) for name, training, inference in (
        ('final_A', 'standard', 'standard'), ('final_B', 'standard', 'sahi_context'),
        ('final_C', 'standard', 'cg_alr_v3'), ('final_D', 'rarity_only', 'standard'),
        ('final_E', 'rarity_only', 'sahi_context'), ('final_F', 'rarity_only', 'cg_alr_v3'))]
    actual = [(v['id'], v['training'], v['inference']) for v in cfg['conditions']]
    if actual != expected or cfg['primary_proposed_configuration'] != 'final_C':
        raise ValueError('Final six-condition mapping differs from the approved protocol')
    if cfg['bootstrap'] != {'replicates': 500, 'seed': 314159, 'unit': 'patient', 'confidence_level': 0.95}:
        raise ValueError('Final paired-bootstrap policy changed')
    return cfg


def source_identities():
    # Bind the new adapters and existing project components without rewriting historical files.
    paths = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'tests').glob('test_final_*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in paths}


def verify_frozen(require_preflight=True):
    cfg = settings()
    if require_preflight:
        receipt = read(PUBLIC/'preflight_verification.json')
        if receipt['status'] != 'READY' or receipt['matrix_sha256'] != sha(MATRIX) or receipt['matrix_identity'] != identity(cfg):
            raise ValueError('Final matrix differs from its completed preflight receipt')
    for name, expected in cfg['identities']['normalized_project_source_sha256'].items():
        actual = hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen project source differs: '+name)
    for name, expected in cfg['identities']['artifact_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen configuration/evidence differs: '+name)
    if dependency_identity() != cfg['identities']['dependencies']:
        raise ValueError('Frozen package versions or evaluator/SAHI sources differ')
    return cfg


def dependency_identity():
    packages = ('numpy', 'Pillow', 'opencv-python', 'torch', 'torchvision', 'ultralytics', 'sahi', 'PyYAML')
    sources = {}
    for package, names in {
        'sahi': ('models/base.py', 'models/ultralytics.py', 'postprocess/combine.py', 'predict.py', 'slicing.py'),
        'ultralytics': ('engine/validator.py', 'models/yolo/detect/val.py', 'utils/metrics.py',
                        'utils/ops.py', 'utils/nms.py', 'data/augment.py')}.items():
        base = Path(importlib.util.find_spec(package).origin).parent
        sources[package] = {name: sha(base/name) for name in names}
    return {'packages': {p: importlib.metadata.version(p) for p in packages}, 'source_sha256': sources}


def phase5c_forbidden(path):
    path = Path(path).resolve()
    # Mixed preparation containers also contain test annotations, even if a caller later filters them.
    blocked = [ROOT/'data', ROOT/'datasets', ROOT/'downloads', ROOT/'outputs/dataset_preparation',
               ROOT/'outputs/dataset_audit']
    return path.name.startswith('.env') or any(path.is_relative_to(p.resolve()) for p in blocked)


def phase5c_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, Path)) and phase5c_forbidden(args[0]):
        raise PermissionError('Phase 5C forbids dataset contents and mixed annotation containers')


def install_phase5c_guard():
    sys.addaudithook(phase5c_guard)


def rectangular_shapes(rows):
    # Reproduce the frozen native batch-12 rectangular shape schedule from dimensions alone.
    if not rows or any(r['height'] <= 0 or r['width'] <= 0 for r in rows):
        raise ValueError('Positive image dimensions are required')
    ratios = np.array([r['height']/r['width'] for r in rows])
    order, result = ratios.argsort(), {}
    for start in range(0, len(rows), 12):
        indices = order[start:start+12]
        minimum, maximum = ratios[indices].min(), ratios[indices].max()
        shape = [maximum, 1] if maximum < 1 else [1, 1/minimum] if minimum > 1 else [1, 1]
        net = (np.ceil(np.array(shape)*768/32+.5).astype(int)*32).tolist()
        for i in indices:
            result[rows[i]['stem']] = net
    if len(result) != len(rows):
        raise ValueError('Duplicate image identities')
    return result


def agents_local_only():
    tracked = subprocess.check_output(['git', 'ls-files', '--', 'AGENTS.md'], cwd=ROOT, text=True).strip()
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
    ignored = subprocess.run(['git', 'check-ignore', '-q', 'AGENTS.md'], cwd=ROOT).returncode == 0
    if tracked or any(Path(p).name == 'AGENTS.md' for p in staged) or not ignored:
        raise ValueError('AGENTS.md must remain ignored, untracked and unstaged')
    return {'exists_locally': (ROOT/'AGENTS.md').is_file(), 'ignored': ignored, 'tracked': False, 'staged': False}
