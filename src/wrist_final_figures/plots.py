"""Minimal Matplotlib figures with explicit values, units and fixed A4 width."""

import os
from .data import OUT, WORK, METHODS, CLASSES, LABELS

os.environ['MPLCONFIGDIR'] = str(WORK/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

STANDARD, RARITY, NEUTRAL = '#526d80', '#87978d', '#c7cdd1'
COLORS = [STANDARD]*3+[RARITY]*3
X = np.array([0, 1, 2, 3.6, 4.6, 5.6])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'svg.fonttype': 'none', 'svg.hashsalt': 'wrist-chapter5-final-v1'})


def outside_canvas_text(fig):
    # Matplotlib keeps unused tick artists outside the view; those labels are not drawn.
    fig.canvas.draw()
    unused = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not low <= tick.get_loc() <= high:
                    unused.update((tick.label1, tick.label2))
    renderer = fig.canvas.get_renderer(); clipping = []
    for text in fig.findobj(Text):
        if text not in unused and text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
                clipping.append(text.get_text())
    return clipping


def save(fig, relative, values, caption, section):
    # A fixed physical width avoids tiny type after insertion into a thesis page.
    clipping = outside_canvas_text(fig)
    target = OUT/relative; target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix('.png'), dpi=300, metadata={'Software': 'Matplotlib scientific figure renderer'})
    fig.savefig(target.with_suffix('.svg'), metadata={'Date': None, 'Creator': 'Matplotlib scientific figure renderer'})
    svg = target.with_suffix('.svg')
    svg.write_text('\n'.join(v.rstrip() for v in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')
    plt.close(fig)
    return {'name': relative, 'section': section, 'caption': caption, 'plotted_values': values,
            'files': [target.with_suffix(s).relative_to(OUT).as_posix() for s in ('.png', '.svg')],
            'layout_text_outside_canvas': clipping, 'width_inches': fig.get_figwidth(), 'height_inches': fig.get_figheight(),
            'axes_limits': [{'x': list(ax.get_xlim()), 'y': list(ax.get_ylim()), 'ylabel': ax.get_ylabel()} for ax in fig.axes]}


def axes_bars(ax, values, ylabel, upper):
    ax.bar(X, values, width=.68, color=COLORS, edgecolor='#42505b', linewidth=.45)
    ax.set_xticks(X, list('ABCDEF')); ax.set_ylabel(ylabel); ax.set_ylim(0, upper)
    ax.tick_params(axis='x', length=0)


def training_legend(fig):
    fig.legend(handles=[Patch(facecolor=STANDARD, label='Standard training'), Patch(facecolor=RARITY, label='Rarity-only training')],
               loc='outside upper center', ncol=2, frameon=False)


def figures(data):
    entries = []; results = data['results']; test = results['test']; val = results['validation']
    for prefix, title in (('train', 'Training'), ('val', 'Validation')):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.65), layout='constrained'); plotted = {}
        for ax, name, label in zip(axes, ('box_loss', 'cls_loss', 'dfl_loss'), ('Box loss', 'Classification loss', 'DFL loss')):
            key = prefix+'/'+name; plotted[key] = {}
            for (strategy, history), color, style in zip(data['histories'].items(), (STANDARD, RARITY), ('-', '--')):
                values = [row[key] for row in history['rows']]; plotted[key][strategy] = values
                ax.plot(range(1, 81), values, color=color, ls=style, lw=1.1, label=strategy)
            ax.set(xlabel='Epoch', ylabel='Loss', xlim=(1, 80), ylim=(0, None)); ax.set_xticks([1, 20, 40, 60, 80]); ax.set_title(label)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False)
        entries.append(save(fig, '5_1_training/'+prefix+'_losses', {'epochs': list(range(1, 81)), **plotted},
            title+' losses for both 80-epoch runs. Raw recorded epochs, without smoothing or interpolation. Losses are native training-framework values.', '5.1'))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.5), layout='constrained'); plotted = {}
    keys = ('precision', 'recall', 'mAP50', 'mAP50-95')
    for ax, key, label in zip(axes.flat, keys, ('Precision', 'Recall', 'mAP50', 'mAP50–95')):
        plotted[key] = {}
        for (strategy, history), color, style in zip(data['histories'].items(), (STANDARD, RARITY), ('-', '--')):
            values = [r['metrics/'+key+'(B)']*100 for r in history['rows']]; plotted[key][strategy] = values
            ax.plot(range(1, 81), values, color=color, ls=style, lw=1.1, label=strategy)
            if key == 'mAP50-95':
                epoch = history['best_epoch']; ax.axvline(epoch, color=color, ls=':', lw=.9)
                ax.plot(epoch, values[epoch-1], marker='o', ms=4, color=color)
        ax.set(xlabel='Epoch', ylabel=label+' (%)', xlim=(1, 80), ylim=(0, 40 if key == 'mAP50-95' else 100)); ax.set_xticks([1, 20, 40, 60, 80])
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False)
    entries.append(save(fig, '5_1_training/native_validation_metrics', {'epochs': list(range(1, 81)), **plotted, 'selected_epochs': [22, 26]},
        'Native training-time validation metrics. Markers identify selected epochs 22 and 26 on mAP50–95. Native precision/recall use framework reporting, not the final fixed .25 operating point; these are not the final common original-coordinate metrics.', '5.1'))
    fig, ax = plt.subplots(figsize=(6.6, 3.15), layout='constrained'); training_legend(fig)
    values = [test['AP'][m]['overall']['map50_95']*100 for m in METHODS]
    axes_bars(ax, values, 'Test mAP50–95 (%)', 40)
    entries.append(save(fig, '5_2_overall/test_map50_95', dict(zip(METHODS, values)),
        'Final test eight-class mAP50–95. A–C use standard training; D–F use rarity-only training. In each group the inference order is Standard, SAHI Context, V3. C is the predefined primary pipeline. Bars begin at zero; tiny V3 differences should not be read as established superiority.', '5.2'))
    fig, ax = plt.subplots(figsize=(6.6, 3.15), layout='constrained'); plotted = {}
    for shift, split, color, hatch in ((-.18, 'validation', NEUTRAL, '//'), (.18, 'test', STANDARD, None)):
        values = [results[split]['AP'][m]['overall']['map50_95']*100 for m in METHODS]; plotted[split] = values
        ax.bar(X+shift, values, .34, color=color, hatch=hatch, edgecolor='#42505b', lw=.45, label=split.capitalize())
    ax.set_xticks(X, list('ABCDEF')); ax.set(ylabel='mAP50–95 (%)', ylim=(0, 40)); ax.tick_params(axis='x', length=0)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False)
    entries.append(save(fig, '5_2_overall/validation_test_map50_95', plotted,
        'Comparable validation (3,088 images/914 patients) and test (3,178 images/913 patients) results under the same final deterministic evaluator. C/F use V3 in both partitions. Separate cohorts; this is not a paired validation-minus-test inference.', '5.2'))
    fig, ax = plt.subplots(figsize=(6.6, 3.9), layout='constrained')
    values = np.array([[test['AP'][m][c]['ap50_95']*100 for m in METHODS] for c in CLASSES])
    image = ax.pcolormesh(np.arange(7)-.5, np.arange(9)-.5, values, vmin=0, vmax=100, cmap='Blues', shading='flat', rasterized=False)
    ax.set_ylim(7.5, -.5)
    ax.set_xticks(range(6), list('ABCDEF')); ax.set_yticks(range(8), LABELS); ax.tick_params(length=0)
    ax.axvline(2.5, color='white', lw=2)
    for i in range(8):
        for j in range(6):
            rgb = image.cmap(values[i, j]/100)[:3]
            lum = sum(w*(c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4) for w, c in zip((.2126, .7152, .0722), rgb))
            color = 'black' if (lum+.05)/.05 >= 1.05/(lum+.05) else 'white'
            ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', color=color, fontsize=8.5)
    colorbar = fig.colorbar(image, ax=ax, label='Test AP50–95 (%)', ticks=[0, 25, 50, 75, 100])
    colorbar.solids.set_rasterized(False)
    entries.append(save(fig, '5_3_classes/class_ap50_95_heatmap', {'classes': list(CLASSES), 'conditions': METHODS, 'AP_percent': values.tolist()},
        'Test AP50–95 for all eight classes; displayed cell labels are rounded to one decimal percent. A–C: standard training; D–F: rarity-only. The full 0–100% scale is shared. Table 3 gives support, including bone lesion (5 objects/4 patients) and bone anomaly (35/14).', '5.3'))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), layout='constrained'); plotted = {}; training_legend(fig)
    for ax, key, label in zip(axes, ('ap50', 'ap75', 'ap50_95'), ('AP50', 'AP75', 'AP50–95')):
        values = [test['AP'][m]['fracture'][key]*100 for m in METHODS]; plotted[key] = values
        axes_bars(ax, values, label+' (%)', 100)
    entries.append(save(fig, '5_3_classes/fracture_ap', plotted,
        'Fracture AP under the six final conditions. Every panel begins at zero. V3 preserves all three fracture point metrics relative to its corresponding standard inference; rarity-only training does not improve every target. Paired uncertainty is reported separately.', '5.3'))
    fig, ax = plt.subplots(figsize=(6.6, 3.75), layout='constrained'); plotted = []
    for i, c in enumerate(CLASSES):
        delta = test['AP_contrasts']['training_standard'][c]['ap50_95']*100
        ci = test['bootstrap']['contrasts']['training_standard'][c]['ap50_95']; lo, hi = ci['lower']*100, ci['upper']*100
        ax.hlines(i, lo, hi, color=STANDARD, lw=1.3); ax.plot(delta, i, 'o', color=STANDARD, ms=4)
        plotted.append({'class': c, 'delta_pp': delta, 'lower_pp': lo, 'upper_pp': hi, 'n': ci['replicates_present']})
    ax.axvline(0, color='#444444', lw=.8); ax.set_yticks(range(8), LABELS); ax.invert_yaxis(); ax.set_xlabel('AP50–95 difference, D − A (percentage points)')
    ax.grid(axis='x', color='#eeeeee', lw=.5); ax.set_axisbelow(True)
    entries.append(save(fig, '5_3_classes/rarity_class_differences', plotted,
        'Rarity-only minus standard training with standard inference (D−A), by class. Points and nominal patient-paired percentile 95% intervals use 500 draws/seed314159; valid counts vary with class presence. All bounds are retained. The sparse-class intervals limit interpretation; this compares two fixed training runs, not a general sampler effect.', '5.3'))
    fig, ax = plt.subplots(figsize=(6.6, 3.15), layout='constrained'); training_legend(fig)
    values = [test['operating'][m]['overall']['fp'] for m in METHODS]
    axes_bars(ax, values, 'Unmatched predictions (FP)', 4500)
    entries.append(save(fig, '5_4_errors/false_positives', dict(zip(METHODS, values)),
        'Annotation-relative unmatched predictions at confidence .25 and IoU .50 on the test set. These are not clinically confirmed false diagnoses. All six final conditions use the same operating-point matcher.', '5.4'))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15), layout='constrained'); pairs = [('final_B', 'final_A'), ('final_C', 'final_A'), ('final_E', 'final_D'), ('final_F', 'final_D')]
    recovered, lost, fp = [], [], []
    for other, base in pairs:
        r = test['recovery'][other+'-'+base]; recovered.append(sum(v['recovered'] for v in r.values())); lost.append(sum(v['lost'] for v in r.values()))
        fp.append(test['operating'][other]['overall']['fp']-test['operating'][base]['overall']['fp'])
    x = np.array([0, 1, 2.6, 3.6]); labels = ['B−A', 'C−A', 'E−D', 'F−D']
    axes[0].bar(x-.16, recovered, .30, color=STANDARD, label='Recovered'); axes[0].bar(x+.16, lost, .30, color=NEUTRAL, edgecolor='#555555', lw=.45, label='Lost')
    axes[0].set(ylabel='Annotated objects', ylim=(0, 210)); axes[0].legend(frameon=False)
    axes[1].bar(x, fp, .65, color=[STANDARD, STANDARD, RARITY, RARITY], edgecolor='#42505b', lw=.45); axes[1].set(ylabel='Additional FP', ylim=(0, 3100))
    for ax in axes:
        ax.set_xticks(x, labels); ax.tick_params(axis='x', length=0)
    entries.append(save(fig, '5_4_errors/recovery_fp_tradeoff', {'comparisons': labels, 'recovered': recovered, 'lost': lost, 'additional_FP': fp},
        'Recovered/lost annotated objects and added unmatched predictions relative to the corresponding standard inference. The panels have separate count scales. Recovery is determined by annotation correspondence, not by the number of changed confidence scores.', '5.4'))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.85), layout='constrained'); plotted = {}; training_legend(fig)
    for ax, key, label, upper in zip(axes, ('mean', 'p95', 'calls'), ('Mean latency (ms)', 'p95 latency (ms)', 'Mean model calls'), (300, 300, 5.5)):
        values = [test['runtime'][m]['timing']['forward_passes']['mean'] if key == 'calls' else test['runtime'][m]['timing']['e2e_seconds'][key]*1000 for m in METHODS]
        plotted[key] = values; axes_bars(ax, values, label, upper)
    entries.append(save(fig, '5_5_efficiency/latency_and_calls', plotted,
        'Measured complete-path mean/p95 latency and actual mean physical calls per image. Three synthetic warmups, loading, export and offline metrics are excluded. Separate sessions and fixed A–F order limit speed comparisons. V3 paired global-prefix ratios are 1.274×/1.243×; p95 is a latency percentile, not a confidence interval.', '5.5'))
    names = ['v3_standard_training', 'v3_rarity_training', 'training_standard', 'training_v3', 'training_sahi', 'sahi_standard_training', 'sahi_rarity_training']
    labels = ['C−A: V3 (primary)', 'F−D: V3', 'D−A: training', 'F−C: training', 'E−B: training', 'B−A: SAHI', 'E−D: SAHI']
    fig, ax = plt.subplots(figsize=(6.6, 3.7), layout='constrained'); plotted = []
    for i, name in enumerate(names):
        delta = test['AP_contrasts'][name]['overall']['map50_95']*100; ci = test['bootstrap']['contrasts'][name]['overall']['map50_95']; lo, hi = ci['lower']*100, ci['upper']*100
        ax.hlines(i, lo, hi, color=STANDARD, lw=1.4); ax.plot(delta, i, marker='D' if i == 0 else 'o', color=STANDARD, ms=4)
        plotted.append({'contrast': name, 'delta_pp': delta, 'lower_pp': lo, 'upper_pp': hi, 'n': ci['replicates_present']})
    ax.axvline(0, color='#333333', lw=.8); ax.set_yticks(range(7), labels); ax.invert_yaxis(); ax.set(xlim=(-6.1, 3.5), xlabel='mAP50–95 difference (percentage points)')
    ax.grid(axis='x', color='#eeeeee', lw=.5); ax.set_axisbelow(True)
    entries.append(save(fig, '5_6_uncertainty/main_paired_differences', plotted,
        'Main predefined test contrasts: nominal patient-paired percentile 95% intervals from 500 draws/seed314159 (492 valid eight-class replicates). The common scale and zero line retain SAHI/training effects without magnifying tiny V3 changes. C−A is primary; its interval includes zero. Secondary intervals are unadjusted and do not establish clinical validation. Table 4 contains all eleven contrasts.', '5.6'))
    return entries
