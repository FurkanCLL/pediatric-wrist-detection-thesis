"""Two deterministic, local-only radiograph comparisons from saved final predictions."""

import hashlib
import json

import numpy as np
from PIL import Image

from .data import ROOT, WORK, STUDY, EVIDENCE, read, source, sha, require, write_json


def mask_text(image, boxes, padding=10):
    output = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        output[max(0, int(y1)-padding):min(len(image), int(np.ceil(y2))+padding),
               max(0, int(x1)-padding):min(image.shape[1], int(np.ceil(x2))+padding)] = 0
    return output


def crop_box(target, width, height):
    # This square crop is a display operation after inference, identical in all panels.
    x1, y1, x2, y2 = target; side = min(max(width, height), max((x2-x1)*4, (y2-y1)*4, min(width, height)*.35))
    crop_w, crop_h = min(width, side), min(height, side)
    left = max(0, min(width-crop_w, (x1+x2-crop_w)/2)); top = max(0, min(height-crop_h, (y1+y2-crop_h)/2))
    return [int(left), int(top), int(np.ceil(left+crop_w)), int(np.ceil(top+crop_h))]


def clear_of_text(target, row):
    for box in row['boxes']:
        if box['class_id'] == 2:
            x1, y1, x2, y2 = box['xyxy']
            if min(target[2], x2+10) > max(target[0], x1-10) and min(target[3], y2+10) > max(target[1], y1-10):
                return False
    return True


def build():
    from wrist_final.metrics import canonical_predictions
    from wrist_eval.metrics import match, ious
    from .plots import plt, STANDARD
    from matplotlib.patches import Rectangle
    integrity = read(EVIDENCE/'data_integrity.json')
    records = read(source('outputs/final_evaluation/phase5d/study_01/test_records.json', integrity['records_sha256']))
    result = read(EVIDENCE/'results.json')['test']; predictions = {}
    for letter in 'ABC':
        method = 'final_'+letter
        path = source(f'outputs/final_evaluation/phase5d/study_01/{method}/predictions.jsonl', result['runtime'][method]['prediction_sha256'])
        predictions[letter] = {v['stem']: np.asarray(v['pred'], np.float32).reshape(-1, 6)
            for v in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())}
    candidates = {'v3_fracture_recovery': [], 'sahi_fracture_FP': []}
    for row in sorted(records, key=lambda r: hashlib.sha256(('chapter5:'+r['filestem']).encode()).hexdigest()):
        stem = row['filestem']; gt = np.asarray([b['xyxy']+[b['class_id']] for b in row['boxes']], np.float32).reshape(-1, 5)
        pred = {c: canonical_predictions(predictions[c][stem], row['width'], row['height']) for c in 'ABC'}
        ga, _, _ = match(gt, pred['A']); gc, _, _ = match(gt, pred['C']); _, pb, _ = match(gt, pred['B'])
        recovered = np.flatnonzero((gt[:, 4] == 0) & (ga < 0) & (gc >= 0))
        if len(recovered) and not candidates['v3_fracture_recovery']:
            for g in recovered:
                if clear_of_text(gt[g, :4], row):
                    candidates['v3_fracture_recovery'] = [row, gt[g, :4].tolist(), 'An annotated fracture missed by A and matched by C at .25/.50.']
                    break
        if not candidates['sahi_fracture_FP']:
            for i, box in enumerate(pred['B']):
                if box[5] != 0 or box[4] < .25 or pb[i] >= 0:
                    continue
                if not clear_of_text(box[:4], row):
                    continue
                overlap_gt = ious(gt[gt[:, 4] == 0, :4], box[None, :4])
                if overlap_gt.size and overlap_gt.max() >= .1:
                    continue
                others = np.concatenate([pred[c][(pred[c][:, 5] == 0) & (pred[c][:, 4] >= .25), :4] for c in 'AC'])
                if len(others) and ious(others, box[None, :4]).max() >= .5:
                    continue
                candidates['sahi_fracture_FP'] = [row, box[:4].tolist(), 'An unmatched SAHI fracture prediction with same-class annotated IoU below .10 and no overlapping operating A/C row at IoU .50.']
                break
        if all(candidates.values()):
            break
    folder = WORK/'qualitative'; folder.mkdir(parents=True, exist_ok=True); entries = []
    prepared_hashes = read(STUDY/'data_receipt.json')['prepared_files']
    for name, selected in candidates.items():
        require(selected, 'No saved qualitative example satisfies the frozen operating definitions')
        row, target, reason = selected; stem = row['filestem']
        path = ROOT/'data/prepared/grazpedwri_detection/images/test'/(stem+'.png')
        require(sha(path) == prepared_hashes[stem][0], 'Selected image differs from the verified test image')
        with Image.open(path) as source_image:
            array = np.asarray(source_image.convert('L'))
        text_boxes = [b['xyxy'] for b in row['boxes'] if b['class_id'] == 2]
        array = mask_text(array, text_boxes)
        crop = crop_box(target, row['width'], row['height']); x1, y1, x2, y2 = crop
        array = array[y1:y2, x1:x2]
        fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.55), layout='constrained')
        for ax, letter, title in zip(axes, (None, 'A', 'B', 'C'), ('Ground truth', 'Standard · A', 'SAHI Context · B', 'CG-ALR V3 · C')):
            ax.imshow(array, cmap='gray', vmin=0, vmax=255, extent=(x1, x2, y2, y1))
            for box in row['boxes']:
                if box['class_id'] == 0:
                    a, b, c, d = box['xyxy']; ax.add_patch(Rectangle((a, b), c-a, d-b, fill=False, ec='#b16e25', lw=.8, ls='--'))
            if letter:
                for box in predictions[letter][stem]:
                    a, b, c, d, confidence, class_id = box
                    if class_id == 0 and confidence >= .25:
                        ax.add_patch(Rectangle((a, b), c-a, d-b, fill=False, ec=STANDARD, lw=.85))
                        if x1 <= a <= x2 and y1 <= b <= y2:
                            ax.text(a, b, f'{confidence:.2f}', fontsize=6.5, color='white', va='bottom', bbox={'facecolor': '#263746', 'edgecolor': 'none', 'pad': .7})
            ax.set(xlim=(x1, x2), ylim=(y2, y1)); ax.set_title(title, fontsize=8); ax.axis('off')
        for extension in ('.png', '.svg'):
            fig.savefig(folder/(name+extension), dpi=300, metadata={'Date': None} if extension == '.svg' else {'Software': 'Matplotlib scientific figure renderer'})
        plt.close(fig)
        entries.append({'name': name, 'image_stem': stem, 'image_sha256': sha(path), 'target': target, 'display_crop': crop,
            'reason': reason, 'annotated_text_regions_masked': len(text_boxes), 'confidence': .25, 'matching_iou': .5,
            'displayed_class': 'fracture only', 'original_box_coordinates_unchanged': True,
            'files': [str(folder/(name+s)) for s in ('.png', '.svg')], 'publication': 'local only; individual privacy clearance not established'})
    write_json(WORK/'qualitative_sources.json', entries)
    (folder/'PRIVATE_INDEX.md').write_text('# Local qualitative examples\n\nBoth figures use the same presentation crop in Ground truth/A/B/C panels, fracture boxes only, confidence .25 and matching IoU .50. Dashed amber boxes are annotations; solid blue boxes are predictions. Annotated text is blacked out and original image metadata is not copied. Coordinates are not changed. Selection uses the first eligible hash-ranked saved case, not a representative sample.\n\nThese redacted figures remain local because complete privacy clearance is not established. The cached release records CC BY 4.0. Source attribution: Eszter Nagy, Michael Janisch, Franko Hrzic, Erich Sorantin and Sebastian Tschauner, GRAZPEDWRI-DX v2, DOI 10.6084/m9.figshare.14825193.v2. Changes: grayscale display crop, text masking and overlays. Do not upload these images or the patient-linked provenance without a separate privacy review.\n', encoding='utf-8')
    return {'figures': len(entries), 'section': '5.4', 'public': False, 'selection': 'first eligible case in SHA256(chapter5:image identity) order',
            'operating_point': {'confidence': .25, 'iou': .5}, 'display': 'fracture boxes only; identical original-coordinate display crop; annotated text masked',
            'local_directory': 'outputs/chapter5_figures/build_01/qualitative', 'privacy': 'not cleared for publication; kept local', 'cached_license': 'CC BY 4.0'}
