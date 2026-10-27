"""Read-only sources and provenance for the final Chapter 5 exports."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/figures/final_results'
WORK = ROOT/'outputs/chapter5_figures/build_01'
EVIDENCE = ROOT/'docs/evidence/final_evaluation/test_results'
STUDY = ROOT/'outputs/final_evaluation/phase5d/study_01'
CLASSES = ('fracture', 'metal', 'text', 'periostealreaction', 'pronatorsign', 'bonelesion', 'softtissue', 'boneanomaly')
LABELS = ('Fracture', 'Metal', 'Text', 'Periosteal reaction', 'Pronator sign', 'Bone lesion', 'Soft tissue', 'Bone anomaly')
METHODS = ['final_'+c for c in 'ABCDEF']
SOURCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(not Path(path).name.startswith('.env'), 'Secrets are outside the figure workflow')
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def source(path, expected=None):
    path = ROOT/path
    digest = sha(path)
    require(expected is None or digest == expected, 'Saved source hash differs: '+str(path.relative_to(ROOT)))
    SOURCES[path.relative_to(ROOT).as_posix()] = digest
    return path


def training_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    require([r['epoch'] for r in rows] == list(range(1, 81)), 'Training history must contain the 80 recorded epochs once')
    require(all(np.isfinite(v) for r in rows for v in r.values()), 'Training history contains missing/nonfinite values')
    return rows


def load():
    receipt = read(source('docs/evidence/final_evaluation/test_results/publication_checks.json'))
    require(receipt['passed'], 'Final publication verification did not pass')
    for name, expected in receipt['artifacts_sha256'].items():
        source('docs/evidence/final_evaluation/test_results/'+name, expected)
    result = read(EVIDENCE/'results.json')
    matrix = read(source('docs/evidence/final_evaluation/final_experiment_matrix.json'))
    require([(v['id'], v['inference']) for v in matrix['conditions']] ==
            list(zip(METHODS, ['standard', 'sahi_context', 'cg_alr_v3']*2)), 'Final C/F must use V3')
    require((result['test']['images'], result['test']['patients']) == (3178, 913)
            and (result['validation']['images'], result['validation']['patients']) == (3088, 914), 'Final cohorts differ')
    inventory = read(source('outputs/V3/preservation_before.json'))['private']
    histories = {}
    for name, directory, epoch in (('Standard', 'standard_baseline', 22), ('Rarity-only', 'rarity_only', 26)):
        relative = f'outputs/training/{directory}/results.csv'
        rows = training_csv(source(relative, inventory[relative]['sha256']))
        require(rows[epoch-1]['metrics/mAP50-95(B)'] == max(r['metrics/mAP50-95(B)'] for r in rows), 'Logged checkpoint maximum differs')
        histories[name] = {'best_epoch': epoch, 'rows': rows}
    protocol = read(source('configs/training_official.json'))
    completed = read(source('outputs/training/standard_baseline/completed.json'))
    rarity = read(source('docs/evidence/rarity_training_completion/verification_summary.json'))
    license_record = read(source('docs/evidence/dataset_audit/source_release.json'))
    return {'results': result, 'matrix': matrix, 'histories': histories, 'training_protocol': protocol,
            'standard_completed': completed, 'rarity_completed': rarity, 'license': license_record['license']}


def snapshot():
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    return {name: sha(ROOT/name) for name in names if not Path(name).name.startswith('.env')}


def install_guard():
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, Path)):
            return
        import os
        path = Path(args[0]).resolve(); mode = args[1]; flags = args[2]
        if path.name.startswith('.env') or path.suffix in ('.pt', '.pth', '.ckpt'):
            raise PermissionError('Secrets and model weights are outside the figure workflow')
        writing = (isinstance(mode, str) and any(v in mode for v in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and (path.is_relative_to(ROOT/'data') or path.is_relative_to(ROOT/'outputs') and not path.is_relative_to(WORK)
                        or path.is_relative_to(ROOT/'docs/figures') and not path.is_relative_to(OUT)):
            raise PermissionError('Historical data, predictions and figures must remain unchanged')
    sys.addaudithook(guard)
