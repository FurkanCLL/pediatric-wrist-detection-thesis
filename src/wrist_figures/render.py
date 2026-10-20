"""Restrained static charts and redacted, original-coordinate image overlays."""

import os

from .data import ROOT, WORK, FIGURES, LABELS, CLASSES, load_image, polygons, mask_text

# Keep font caches out of historical experiment directories.
os.environ['MPLCONFIGDIR'] = str(WORK/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Bbox
import numpy as np

BAR = '#657787'
POLYGON = '#b36b2c'
PREDICTION = '#2872a3'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'svg.fonttype': 'none', 'svg.hashsalt': 'wrist-thesis-figures-v1', 'pdf.fonttype': 42})


def save(fig, relative, vector=False):
    path = FIGURES/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.12, metadata={'Software': 'Matplotlib thesis figure renderer'})
    files = [path]
    if vector:
        svg = path.with_suffix('.svg')
        fig.savefig(svg, bbox_inches='tight', pad_inches=.12, metadata={'Date': None, 'Creator': 'Matplotlib thesis figure renderer'})
        # Keep generated SVG paths free of trailing whitespace for clean Git diffs.
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        files.append(svg)
    plt.close(fig)
    return [p.relative_to(FIGURES.parent.parent).as_posix() for p in files]


def bars(ax, labels, values, ylabel, title, upper=None, value_format=None):
    ax.bar(range(len(values)), values, width=.64, color=BAR)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel); ax.set_title(title, pad=10)
    ax.set_ylim(0, upper if upper is not None else max(values)*1.15)
    ax.tick_params(axis='x', length=0)
    if value_format:
        offset = ax.get_ylim()[1]*.018
        for i, value in enumerate(values):
            ax.text(i, value+offset, value_format(value), ha='center', va='bottom', fontsize=8)


def overall_map50(result):
    # Match the existing mAP50-95 layout, using the distinct saved IoU-0.50 metric.
    ids = list('ABCDEF')
    values = [result['conditions'][c]['AP']['overall']['map50'] for c in ids]
    fig, ax = plt.subplots(figsize=(6.8, 4.), layout='constrained')
    bars(ax, ids, values, 'mAP50', 'Validation mAP50', .7, lambda x: f'{x:.4f}')
    ax.set_xlabel('Experiment')
    return {'name': 'overall_map50', 'description': 'Validation mAP50',
        'files': save(fig, 'results/overall_performance_map50.png', True),
        'plot_values': dict(zip(ids, values)),
        'sources': ['docs/evidence/rarity_inference_comparison/results.json'], 'experiments': ids}


def classwise_ap50(result):
    # Show class-specific baseline performance, with no inference-method comparison.
    values = {name: result['conditions']['A']['AP'][name]['ap50'] for name in CLASSES}
    labels = ['Fracture', 'Metal', 'Text', 'Periosteal\nreaction', 'Pronator\nsign', 'Bone\nlesion', 'Soft\ntissue', 'Bone\nanomaly']
    fig, ax = plt.subplots(figsize=(9.2, 4.3), layout='constrained')
    bars(ax, labels, list(values.values()), 'AP50', 'Class-wise AP50 · Experiment A', 1.1, lambda x: f'{x:.3f}')
    ax.set_yticks([0, .25, .5, .75, 1])
    return {'name': 'classwise_ap50', 'description': 'Class-specific AP50 for Experiment A only: standard training and standard inference',
        'files': save(fig, 'results/classwise_ap50_all_classes.png', True),
        'plot_values': values,
        'sources': ['docs/evidence/rarity_inference_comparison/results.json'], 'experiments': ['A']}


def charts(result, counts, ap_only=False):
    # A targeted export preserves every unrelated existing figure byte-for-byte.
    if ap_only:
        return [classwise_ap50(result), overall_map50(result)]
    entries = []
    fig, ax = plt.subplots(figsize=(9.2, 4.3), layout='constrained')
    wrapped = ['Fracture', 'Metal', 'Text', 'Periosteal\nreaction', 'Pronator\nsign', 'Bone\nlesion', 'Soft\ntissue', 'Bone\nanomaly']
    values = [counts[c] for c in CLASSES]
    bars(ax, wrapped, values, 'Annotation count', 'Class distribution', value_format=lambda x: f'{x:,}')
    entries.append({'name': 'class_distribution', 'description': 'Dataset-wide counts of the eight project classes (47,436 annotations)',
        'files': save(fig, 'thesis/class_distribution_chart.png', True), 'plot_values': dict(zip(CLASSES, values)),
        'sources': ['docs/evidence/dataset_preparation/class_distribution.csv', 'docs/evidence/dataset_audit/class_statistics.csv'], 'experiments': []})
    methods = result['conditions']; ids = list('ABCDEF')
    for name, title, getter, upper, relative in (
        ('overall_performance', 'Validation mAP50–95', lambda c: methods[c]['AP']['overall']['map50_95'], .4, 'overall_performance_map50_95.png'),
        ('fracture_ap50', 'Fracture AP50', lambda c: methods[c]['AP']['fracture']['ap50'], 1., 'fracture_ap50_comparison.png'),
        ('false_positives', 'False positives · confidence 0.25 / IoU 0.50', lambda c: methods[c]['operating_points']['conf0.25_iou0.5']['total_fp'], None, 'false_positives_comparison.png')):
        values = [getter(c) for c in ids]
        fig, ax = plt.subplots(figsize=(6.8, 4.), layout='constrained')
        count = name == 'false_positives'
        bars(ax, ids, values, 'False positives (count)' if count else ('mAP50–95' if name == 'overall_performance' else 'AP50'),
            title, upper, (lambda x: f'{x:,}') if count else (lambda x: f'{x:.4f}'))
        ax.set_xlabel('Experiment')
        entries.append({'name': name, 'description': title, 'files': save(fig, 'results/'+relative, True),
            'plot_values': dict(zip(ids, values)), 'sources': ['docs/evidence/rarity_inference_comparison/results.json'], 'experiments': ids})
    entries.extend([overall_map50(result), classwise_ap50(result)])
    ids = list('DEF'); labels = ['Standard', 'SAHI\nContext', 'CG-ALR V2']
    latency = [methods[c]['source']['timing']['e2e_seconds']['mean']*1000 for c in ids]
    calls = [methods[c]['source']['mean_passes'] for c in ids]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), layout='constrained')
    bars(axes[0], labels, latency, 'Mean end-to-end latency (ms)', 'Recorded latency', value_format=lambda x: f'{x:.1f}')
    bars(axes[1], labels, calls, 'Model calls per image', 'Model calls', value_format=lambda x: f'{x:.3f}'.rstrip('0').rstrip('.'))
    fig.suptitle('Rarity-only checkpoint · separate timing sessions', fontsize=10)
    entries.append({'name': 'inference_cost', 'description': 'Method-level recorded latency and actual mean model calls, D/E/F only; sessions are not controlled',
        'files': save(fig, 'results/inference_cost_comparison.png', True),
        'plot_values': {'latency_ms': dict(zip(ids, latency)), 'model_calls': dict(zip(ids, calls))},
        'sources': ['docs/evidence/rarity_inference_comparison/results.json', 'docs/evidence/rarity_comparison/full_D.json',
            'docs/evidence/rarity_inference_comparison/full_E.json', 'docs/evidence/rarity_inference_comparison/full_F.json'], 'experiments': ids})
    return entries


def image_axes(ax, pixels, crop, title):
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_title(title, fontsize=11, pad=8); ax.set_axis_off()


def rectangle(ax, box, color, dashed=False):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color,
        linewidth=1.35, linestyle=(0, (4, 2)) if dashed else '-', clip_on=True))


def conversion(row, case, index):
    pixels, labels, provenance = load_image(row)
    originals, annotation_path = polygons(row)
    pixels, regions = mask_text(pixels, row['boxes'])
    fig, axes = plt.subplots(1, 2, figsize=(8., 4.7), layout='constrained')
    crop = case['crop_xyxy']
    for ax, title in zip(axes, ('Original polygons', 'Converted bounding boxes')):
        image_axes(ax, pixels, crop, title)
    for i, obj in originals:
        axes[0].add_patch(Polygon(obj['points']['exterior'], closed=True, fill=False, edgecolor=POLYGON, linewidth=1.5))
        for ring in obj['points'].get('interior', []):
            if np.asarray(ring).ndim == 2:
                axes[0].add_patch(Polygon(ring, closed=True, fill=False, edgecolor=POLYGON, linewidth=1.5))
        rectangle(axes[1], labels[i, :4], PREDICTION)
    names = sorted({obj['classTitle'] for _, obj in originals})
    fig.suptitle(' / '.join(LABELS[CLASSES.index(c)] for c in names), fontsize=10)
    files = save(fig, f'thesis/annotation_conversion_candidates/example_{index:02d}.png')
    return {'name': f'annotation_conversion_{index:02d}', 'description': 'Original source polygons and actual converted YOLO boxes on the same training-image crop',
        'files': files, 'image_id': 'sha256:'+row['source_image_sha256'], 'classes': names,
        'annotations_shown': len(originals), 'source_target_annotations': len(row['boxes']), 'experiments': [],
        'sources': ['outputs/dataset_preparation/source_records.jsonl', 'outputs/dataset_preparation/completed.sqlite'],
        'source_annotation_sha256': row['source_annotation_sha256'],
        'prepared_image_sha256': provenance['prepared_image_sha256'], 'prepared_label_sha256': provenance['prepared_label_sha256'],
        'redacted_text_regions': len(regions)}, {**case, **provenance,
            'source_annotation_path': annotation_path.relative_to(ROOT).as_posix(),
            'source_annotation_sha256': row['source_annotation_sha256'], 'redaction_rectangles': regions}


def intersects(box, crop):
    return min(box[2], crop[2]) > max(box[0], crop[0]) and min(box[3], crop[3]) > max(box[1], crop[1])


def prediction_labels(ax, boxes, crop):
    # Move labels into nearby free space; prediction coordinates and scores stay unchanged.
    short = ('fracture', 'metal', 'text', 'periosteal', 'pronator', 'lesion', 'soft tissue', 'anomaly')
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    frame = ax.get_window_extent(renderer)
    for box in boxes:
        anchor = ax.transAxes.inverted().transform(ax.transData.transform((box[0], box[1])))
        text = ax.text(0, 0, f'{short[int(box[5])]} {box[4]:.2f}', transform=ax.transAxes, fontsize=7,
            color=PREDICTION, va='bottom', clip_on=True,
            bbox={'facecolor': 'white', 'alpha': .94, 'edgecolor': 'none', 'pad': 1.})
        extent = text.get_window_extent(renderer)
        width, height = (extent.width+8)/frame.width, (extent.height+8)/frame.height
        x = float(np.clip(anchor[0], .015, max(.015, .985-width)))
        candidates = [anchor[1]+.01+step*height for step in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5)]
        for value in candidates:
            y = float(np.clip(value, .015, max(.015, .985-height)))
            text.set_position((x, y))
            bounds = text.get_window_extent(renderer)
            padded = Bbox.from_extents(bounds.x0-3, bounds.y0-3, bounds.x1+3, bounds.y1+3)
            if not any(padded.overlaps(previous) for previous in occupied):
                occupied.append(padded)
                break
        else:
            raise ValueError('No readable label placement; revise this display layout')
        if abs(y-anchor[1]) > height*1.1:
            ax.plot([anchor[0], x], [anchor[1], y], transform=ax.transAxes, color=PREDICTION, linewidth=.5, clip_on=True)


def qualitative(row, case, predictions, index):
    pixels, labels, provenance = load_image(row)
    pixels, regions = mask_text(pixels, row['boxes'])
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.3), layout='constrained')
    crop, shown, labels_by_axis = case['crop_xyxy'], {}, []
    for ax, c, title in zip(axes, 'DEF', ('Standard', 'SAHI Context', 'CG-ALR V2')):
        image_axes(ax, pixels, crop, title)
        for box in labels:
            if box[4] != 2 and intersects(box[:4], crop):
                rectangle(ax, box[:4], POLYGON, True)
        visible = [box for box in predictions[c][row['filestem']] if box[4] >= .25 and box[5] != 2 and intersects(box[:4], crop)]
        shown[c] = len(visible)
        for box in visible:
            rectangle(ax, box[:4], PREDICTION)
        labels_by_axis.append(visible)
    fig.supxlabel('Dashed: annotation   ·   Solid: prediction (confidence ≥ 0.25)', fontsize=9)
    fig.canvas.draw()
    for ax, visible in zip(axes, labels_by_axis):
        prediction_labels(ax, visible, crop)
    files = save(fig, f'results/qualitative_example_{index:02d}.png')
    return {'name': f'qualitative_{index:02d}', 'description': case['reason'], 'files': files,
        'image_id': 'sha256:'+row['source_image_sha256'], 'experiments': list('DEF'), 'training': 'rarity-only',
        'display_confidence': .25, 'matching_iou_for_selection': .5, 'text_class_omitted_from_display': True,
        'displayed_predictions': shown, 'redacted_text_regions': len(regions),
        'prepared_image_sha256': provenance['prepared_image_sha256'], 'prepared_label_sha256': provenance['prepared_label_sha256'],
        'sources': ['outputs/dataset_preparation/source_records.jsonl',
            'outputs/rarity_comparison/study_01/full/predictions.jsonl',
            'outputs/rarity_inference_comparison/study_01/full_E/predictions.jsonl',
            'outputs/rarity_inference_comparison/study_01/full_F/predictions.jsonl']}, {**case, **provenance, 'redaction_rectangles': regions}
