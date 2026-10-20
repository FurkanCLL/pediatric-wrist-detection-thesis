"""Read verified aggregates, training polygons and saved validation predictions."""

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from wrist_prepare.common import CLASSES
from wrist_prepare.records import load_records
from wrist_prepare.labels import target_boxes

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'outputs/thesis_figures/build_01'
FIGURES = ROOT/'docs/figures'
PREPARED = ROOT/'data/prepared/grazpedwri_detection'
RAW = ROOT/'data/raw/grazpedwri-dx'
EVIDENCE = ROOT/'docs/evidence/rarity_inference_comparison'
LABELS = ('Fracture', 'Metal', 'Text', 'Periosteal reaction', 'Pronator sign', 'Bone lesion', 'Soft tissue', 'Bone anomaly')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(not Path(path).name.startswith('.env'), 'Secret files are outside this task')
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregates():
    result = read(EVIDENCE/'results.json')
    receipt = read(EVIDENCE/'publication_checks.json')
    require(receipt['passed'] and sha(EVIDENCE/'results.json') == receipt['artifacts_sha256']['results.json'],
            'Verified A-F aggregate changed')
    matrix = csv_rows(EVIDENCE/'six_condition_matrix.csv')
    require([v['experiment'] for v in matrix] == list('ABCDEF'), 'A-F ordering differs')
    for row in matrix:
        value = result['conditions'][row['experiment']]
        require(float(row['map50_95']) == value['AP']['overall']['map50_95'], 'Matrix AP differs')
        require(float(row['map50']) == value['AP']['overall']['map50'], 'Matrix mAP50 differs')
        require(int(row['fp']) == value['operating_points']['conf0.25_iou0.5']['total_fp'], 'Matrix FP differs')
    counts = {name: 0 for name in CLASSES}
    for row in csv_rows(ROOT/'docs/evidence/dataset_preparation/class_distribution.csv'):
        counts[row['class']] += int(row['objects'])
    audit = {row['source_class']: int(row['instances']) for row in csv_rows(ROOT/'docs/evidence/dataset_audit/class_statistics.csv')}
    require(sum(counts.values()) == 47436 and all(counts[c] == audit[c] for c in CLASSES), 'Dataset census differs')
    return result, counts


def records():
    # Frozen metadata containers are checked before filtering; no test image/label is opened.
    return [row for row in load_records(ROOT/'outputs/dataset_preparation') if row['split'] in ('train', 'val')]


def prepared_paths(row):
    require(row['split'] in ('train', 'val') and Path(row['filestem']).name == row['filestem'], 'Only train/validation figures are allowed')
    return tuple(PREPARED/folder/row['split']/(row['filestem']+suffix) for folder, suffix in (('images', '.png'), ('labels', '.txt')))


def decode_yolo(text, width, height):
    values = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 5, 'Malformed prepared label')
        c = int(fields[0]); x, y, w, h = map(float, fields[1:])
        require(0 <= c < 8 and np.isfinite([x, y, w, h]).all() and w > 0 and h > 0, 'Invalid prepared label')
        values.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height, c])
    return np.asarray(values, np.float64).reshape(-1, 5)


def load_image(row):
    image_path, label_path = prepared_paths(row)
    journal = ROOT/'outputs/dataset_preparation/completed.sqlite'
    with sqlite3.connect(journal.as_uri()+'?mode=ro', uri=True) as connection:
        expected = connection.execute('SELECT image_sha,label_sha FROM completed WHERE stem=?', (row['filestem'],)).fetchone()
    require(expected and (sha(image_path), sha(label_path)) == expected, 'Prepared bytes differ from the frozen journal')
    labels = decode_yolo(label_path.read_text(encoding='ascii'), row['width'], row['height'])
    reference = np.asarray([list(b['xyxy'])+[b['class_id']] for b in row['boxes']], np.float64).reshape(-1, 5)
    require(labels.shape == reference.shape and np.allclose(labels, reference, rtol=0, atol=1e-8), 'Converted labels differ from frozen boxes')
    with Image.open(image_path) as image:
        require(image.mode == 'L' and image.size == (row['width'], row['height']), 'Prepared image type/dimensions differ')
        pixels = np.asarray(image).copy()
    return pixels, labels, {'image_path': image_path.relative_to(ROOT).as_posix(),
        'label_path': label_path.relative_to(ROOT).as_posix(), 'prepared_image_sha256': expected[0],
        'prepared_label_sha256': expected[1], 'label_roundtrip_max_delta_pixels': float(np.max(np.abs(labels-reference))) if labels.size else 0.}


def polygons(row):
    require(row['split'] == 'train', 'Conversion illustrations use training annotations only')
    path = RAW/'supervisely/wrist/ann'/(row['filestem']+'.json')
    require(sha(path) == row['source_annotation_sha256'], 'Original annotation hash differs')
    annotation = read(path)
    converted = target_boxes(annotation)
    require(len(converted) == len(row['boxes']) and all(a['class_id'] == b['class_id'] and
        a['source_object_index'] == b['source_object_index'] and list(a['xyxy']) == b['xyxy']
        for a, b in zip(converted, row['boxes'])), 'Original annotation conversion differs')
    # Only these two categories are polygons; source rectangles must not be relabelled as polygons.
    return [(i, annotation['objects'][box['source_object_index']]) for i, box in enumerate(row['boxes']) if box['class_id'] in (3, 5)], path


def saved_predictions(rows, result):
    rows = [row for row in rows if row['split'] == 'val']
    paths = {'D': ROOT/'outputs/rarity_comparison/study_01/full/predictions.jsonl',
        **{c: ROOT/f'outputs/rarity_inference_comparison/study_01/full_{c}/predictions.jsonl' for c in 'EF'}}
    predictions = {}
    for c, path in paths.items():
        require(sha(path) == result['conditions'][c]['source']['prediction_sha256'], 'Saved prediction hash differs')
        items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require([v['stem'] for v in items] == [row['filestem'] for row in rows], 'Saved prediction membership/order differs')
        predictions[c] = {v['stem']: np.asarray(v['pred'], np.float32).reshape(-1, 6) for v in items}
    details = {'D': read(ROOT/'outputs/rarity_comparison/study_01/analysis/D_details.json'),
        **{c: read(ROOT/f'outputs/rarity_inference_comparison/study_01/analysis/{c}_details.json') for c in 'EF'}}
    require(all(set(value) == {row['filestem'] for row in rows} for value in details.values()), 'Saved matching membership differs')
    return predictions, details, paths


def mask_text(pixels, boxes):
    # Redaction affects exported copies only, including any text inside a selected crop.
    output = pixels.copy()
    margin = max(4, round(max(pixels.shape)*.004))
    regions = []
    for box in boxes:
        if box['class_id'] != 2:
            continue
        x1, y1, x2, y2 = box['xyxy']
        region = [max(0, int(np.floor(x1))-margin), max(0, int(np.floor(y1))-margin),
            min(pixels.shape[1], int(np.ceil(x2))+margin), min(pixels.shape[0], int(np.ceil(y2))+margin)]
        a, b, c, d = region; output[b:d, a:c] = 0; regions.append(region)
    return output, regions


def crop_bounds(boxes, width, height, minimum_fraction=.24, padding=.5):
    rows = np.asarray(boxes).reshape(-1, 4)
    require(len(rows) > 0, 'A display crop needs a real target')
    left, top = rows[:, :2].min(axis=0); right, bottom = rows[:, 2:].max(axis=0)
    cx, cy = (left+right)/2, (top+bottom)/2
    w, h = max((right-left)*(1+2*padding), width*minimum_fraction), max((bottom-top)*(1+2*padding), height*minimum_fraction)
    # Keep elongated polygon details readable without stretching the actual pixels.
    w, h = max(w, h*.65), max(h, w*.65)
    return [max(0, int(np.floor(cx-w/2))), max(0, int(np.floor(cy-h/2))),
        min(width, int(np.ceil(cx+w/2))), min(height, int(np.ceil(cy+h/2)))]
