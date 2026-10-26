"""Hash-checked test membership and original-coordinate annotations after the freeze."""

import json
import sqlite3
import subprocess
import sys

import numpy as np

from .execution import RUN, RESULTS, locked, require, access_guard
from .protocol import ROOT, read, sha, identity, write_new, rectangular_shapes
from wrist_eval.common import CLASSES

PREPARED = ROOT/'data/prepared/grazpedwri_detection'


def population(records):
    groups = {split: {r['patient_id'] for r in records if r['split'] == split} for split in ('train', 'val', 'test')}
    require(all(not groups[a] & groups[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))), 'Patient leakage')
    rows = sorted([r for r in records if r['split'] == 'test'], key=lambda r: r['filestem'])
    require((len(rows), len(groups['test']), sum(len(r['boxes']) for r in rows)) == (3178, 913, 7361), 'Test population differs')
    require(len({r['filestem'] for r in records}) == len(records), 'Duplicate source image identity')
    return rows


def initialize():
    cfg, lock = locked(); sys.addaudithook(access_guard)
    require(not RUN.exists(), 'Existing final-test evidence must be preserved')
    RUN.mkdir()
    from wrist_prepare.records import load_records
    from wrist_prepare.labels import yolo_bytes
    records = load_records(ROOT/'outputs/dataset_preparation')
    freeze = read(ROOT/'docs/evidence/dataset_preparation/frozen_manifest_hashes.json')
    require(sha(ROOT/'outputs/dataset_preparation/source_records.jsonl') == freeze['records_sha256']
        and sha(ROOT/'outputs/dataset_preparation/split_manifest.csv') == freeze['split_manifest_sha256'], 'Frozen metadata identity differs')
    rows = population(records)
    shapes = rectangular_shapes([{'stem': r['filestem'], 'width': r['width'], 'height': r['height']} for r in rows])
    journal = ROOT/'outputs/dataset_preparation/completed.sqlite'
    old = read(ROOT/'outputs/V3/preservation_before.json')['private']['outputs/dataset_preparation/completed.sqlite']['sha256']
    require(sha(journal) == old, 'Conversion journal changed')
    with sqlite3.connect(journal.as_uri()+'?mode=ro', uri=True) as connection:
        hashes = {s: (im, lab) for s, im, lab in connection.execute('select stem,image_sha,label_sha from completed')}
    prepared_hashes = {}
    for index, row in enumerate(rows):
        stem = row['filestem']
        require(stem == __import__('pathlib').Path(stem).name, 'Invalid image identity')
        im = PREPARED/'images/test'/(stem+'.png'); lab = PREPARED/'labels/test'/(stem+'.txt')
        require((sha(im), sha(lab)) == hashes[stem] and lab.read_bytes() == yolo_bytes(row), 'Test image/label integrity differs')
        prepared_hashes[stem] = hashes[stem]
        if index % 250 == 0:
            print(f'Test integrity {index+1}/{len(rows)}', flush=True)
    # Preserve original annotation order; inference receives a separate metadata-only projection.
    write_new(RUN/'test_records.json', rows)
    write_new(RUN/'validation_records.json', sorted([r for r in records if r['split'] == 'val'], key=lambda r: r['filestem']))
    metadata = [{'stem': r['filestem'], 'patient_id': r['patient_id'], 'width': r['width'], 'height': r['height'],
                 'net_shape': shapes[r['filestem']]} for r in rows]
    write_new(RUN/'test_metadata.json', metadata)
    receipt = {'passed': True, 'images': 3178, 'patients': 913, 'objects': 7361, 'patient_overlap': 0,
        'all_prepared_test_images_and_labels_sha256_verified': True, 'all_labels_match_frozen_record_bytes': True,
        'records_sha256': sha(RUN/'test_records.json'), 'validation_records_sha256': sha(RUN/'validation_records.json'),
        'metadata_sha256': sha(RUN/'test_metadata.json'), 'prepared_hash_identity': identity(prepared_hashes),
        'prepared_files': prepared_hashes, 'membership_sha256': identity([r['stem'] for r in metadata]),
        'shape_identity': identity(shapes), 'execution_freeze_sha256': sha(RESULTS/'execution_freeze.json'),
        'execution_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_new(RUN/'data_receipt.json', receipt)
    print('Test membership, patient separation and image/annotation integrity pass', flush=True)


def metadata():
    receipt = read(RUN/'data_receipt.json')
    require(receipt['passed'] and sha(RUN/'test_metadata.json') == receipt['metadata_sha256'], 'Test metadata changed')
    return read(RUN/'test_metadata.json')


def records(split):
    receipt = read(RUN/'data_receipt.json')
    filename = 'test_records.json' if split == 'test' else 'validation_records.json'
    key = 'records_sha256' if split == 'test' else 'validation_records_sha256'
    require(sha(RUN/filename) == receipt[key], 'Verified annotation snapshot changed')
    return read(RUN/filename)


def truth(row):
    return np.asarray([b['xyxy']+[b['class_id']] for b in row['boxes']], np.float32).reshape(-1, 5)
