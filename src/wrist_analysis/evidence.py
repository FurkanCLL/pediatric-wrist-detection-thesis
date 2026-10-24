"""Read-only provenance checks and private/public analysis boundaries."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT/'docs/evidence/phase_5a_analysis'
LOCAL = ROOT/'outputs/phase_5a_analysis/analysis_01'
CLASSES = ('fracture', 'metal', 'text', 'periostealreaction', 'pronatorsign', 'bonelesion', 'softtissue', 'boneanomaly')
SOURCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(not path.name.startswith('.env'), 'Secrets are outside the analysis scope')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source(name, expected=None):
    path = ROOT/name
    digest = sha(path)
    require(expected is None or digest == expected, 'Source identity differs: '+name)
    SOURCES[name] = digest
    return path


def read(name, expected=None):
    return json.loads(source(name, expected).read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def install_guard():
    # No radiograph, raw source, held-out label, checkpoint or secret is needed.
    blocked = [ROOT/p for p in ('data/raw', 'data/prepared/grazpedwri_detection/images',
                                'data/prepared/grazpedwri_detection/labels/test')]
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, Path)):
            path = Path(args[0]).resolve()
            if path.name.startswith('.env') or path.suffix == '.pt' or any(path.is_relative_to(p) for p in blocked):
                raise PermissionError('Forbidden analysis input')
    sys.addaudithook(guard)


def preservation():
    # Checkpoint bytes are never loaded; other scientific records use SHA-256.
    names = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    tracked = {name: sha(ROOT/name) for name in names if not Path(name).name.startswith('.env')}
    private = {}
    roots = ('training/standard_baseline', 'training/rarity_only', 'error_analysis_sahi', 'adaptive_refinement',
             'rarity_comparison', 'rarity_inference_comparison', 'dataset_preparation')
    for name in roots:
        for path in sorted((ROOT/'outputs'/name).rglob('*')):
            if path.is_file() and not path.name.startswith('.env'):
                stat = path.stat()
                private[path.relative_to(ROOT).as_posix()] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    **({} if path.suffix == '.pt' else {'sha256': sha(path)})}
    return {'tracked': tracked, 'private': private}


def verify_publication(directory, receipt, key):
    value = read(directory+'/'+receipt)
    require(value['passed'], 'Historical publication checks did not pass')
    for name, expected in value[key].items():
        source(directory+'/'+name, expected)
    return value


def load_inputs():
    verify_publication('docs/evidence/rarity_inference_comparison', 'publication_checks.json', 'artifacts_sha256')
    checked = verify_publication('docs/evidence/adaptive_refinement_validation', 'publication_verification.json', 'public_evidence_sha256')
    for name, expected in checked['normalized_source_sha256'].items():
        path = source(name)
        require(hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() == expected, 'Frozen source changed')
    original = read('docs/evidence/error_analysis_sahi/implementation_provenance.json')
    for name, expected in original['normalized_source_sha256'].items():
        path = source(name)
        require(hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() == expected, 'Frozen evaluator changed')
    proposal = read('configs/adaptive_refinement_proposed.json')
    execution = read('configs/adaptive_refinement_execution.json')
    rarity_execution = read('configs/rarity_adaptive_execution.json')
    historical = read('docs/evidence/adaptive_refinement_validation/full_summary.json')
    require(identity(proposal) == historical['proposal_sha256'] == rarity_execution['historical_proposal_sha256'], 'Proposal differs')
    require(identity(execution) == historical['execution_manifest_sha256'] == rarity_execution['historical_execution_sha256'], 'Execution differs')
    read('docs/evidence/rarity_training_completion/verification_summary.json')
    read('docs/evidence/rarity_inference_comparison/ap_tie_diagnostic.json')
    read('docs/evidence/adaptive_refinement_validation/confidence_tie_diagnostics.json')
    result = read('docs/evidence/rarity_inference_comparison/results.json')
    f_summary = read('docs/evidence/rarity_inference_comparison/full_F.json')
    for name, expected in f_summary['source_identity'].items():
        path = source(name)
        require(hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() == expected, 'Frozen F implementation differs')
    d_receipt = read('docs/evidence/rarity_comparison/verification.json')
    require(d_receipt['passed'], 'Historical D verification missing')
    for name, expected in d_receipt['private_artifact_sha256'].items():
        source('outputs/rarity_comparison/study_01/'+name, expected)
    linkage = read('outputs/rarity_comparison/study_01/validation_linkage.json')
    manifest = read('outputs/rarity_comparison/study_01/manifest.json')
    require(identity([row['stem'] for row in linkage]) == manifest['validation_membership_sha256'], 'Validation membership changed')
    require(len(linkage) == 3088 and len({row['patient_id'] for row in linkage}) == 914, 'Validation population differs')
    require(all(Path(row['stem']).name == row['stem'] for row in linkage), 'Unsafe validation identity')
    rows = {}
    journal = source('outputs/dataset_preparation/completed.sqlite')
    label_hashes = []
    with sqlite3.connect(journal.as_uri()+'?mode=ro', uri=True) as connection:
        for row in linkage:
            path = ROOT/'data/prepared/grazpedwri_detection/labels/val'/(row['stem']+'.txt')
            expected = connection.execute('SELECT label_sha FROM completed WHERE stem=?', (row['stem'],)).fetchone()
            require(expected and sha(path) == expected[0], 'Prepared validation label changed')
            label_hashes.append(expected[0])
            boxes = []
            for line in path.read_text(encoding='ascii').splitlines():
                c, x, y, w, h = map(float, line.split())
                boxes.append([(x-w/2)*row['width'], (y-h/2)*row['height'],
                              (x+w/2)*row['width'], (y+h/2)*row['height'], c])
            rows[row['stem']] = {**row, 'gt': np.asarray(boxes, np.float32).reshape(-1, 5)}
    require(sum(len(r['gt']) for r in rows.values()) == 7204, 'Annotation count differs')
    predictions, bundles = {}, {}
    paths = {'A': 'outputs/error_analysis_sahi/full/standard_serial/predictions.jsonl',
        'B': 'outputs/error_analysis_sahi/full/sahi_context/predictions.jsonl',
        'D': 'outputs/rarity_comparison/study_01/full/predictions.jsonl',
        'E': 'outputs/rarity_inference_comparison/study_01/full_E/predictions.jsonl',
        'F': 'outputs/rarity_inference_comparison/study_01/full_F/predictions.jsonl'}
    for condition, name in paths.items():
        path = source(name, result['conditions'][condition]['source']['prediction_sha256'])
        items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require([item['stem'] for item in items] == list(rows), 'Prediction membership/order differs')
        predictions[condition] = {item['stem']: np.asarray(item['pred'], np.float32).reshape(-1, 6) for item in items}
    for condition, base, name in (
        ('C', 'A', 'outputs/adaptive_refinement/full/full_01/bundles.jsonl'),
        ('F', 'D', 'outputs/rarity_inference_comparison/study_01/full_F/bundles.jsonl')):
        expected = result['conditions']['C']['source']['prediction_sha256'] if condition == 'C' else read('docs/evidence/rarity_inference_comparison/full_F.json')['bundles_sha256']
        path = source(name, expected)
        items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require([item['stem'] for item in items] == list(rows), 'Bundle membership/order differs')
        bundles[condition] = items
        actual = {}
        for item in items:
            stem = item['stem']
            require(np.array_equal(np.asarray(item['global'], np.float32).reshape(-1, 6), predictions[base][stem]), 'Saved global predictions differ')
            actual[stem] = np.asarray(item['variants']['V2']['pred'], np.float32).reshape(-1, 6)
            if condition == 'F':
                require(np.array_equal(actual[stem], predictions['F'][stem]), 'F bundle differs from evaluated output')
        predictions[condition] = actual
    return result, rows, predictions, bundles, {'validation_labels': len(label_hashes), 'ordered_label_hashes_sha256': identity(label_hashes)}
