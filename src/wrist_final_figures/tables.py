"""Four compact tables for copying into Word, with CSV companions."""

import csv
from .data import OUT, METHODS, CLASSES, LABELS


def write_table(name, headers, rows, caption, section):
    folder = OUT/'tables'; folder.mkdir(parents=True, exist_ok=True)
    with (folder/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n'); writer.writerow(headers); writer.writerows(rows)
    title = name.replace('_', ' ').replace('table ', 'Table ', 1)
    text = '\n'.join(['# '+title, '', caption, '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])+'\n'
    (folder/(name+'.md')).write_text(text, encoding='utf-8', newline='\n')
    return {'name': name, 'section': section, 'caption': caption, 'headers': headers, 'rows': rows,
            'files': ['tables/'+name+'.csv', 'tables/'+name+'.md']}


def tables(data):
    result = []; test = data['results']['test']; cfg = data['training_protocol']['training']
    standard, rarity = data['histories']['Standard'], data['histories']['Rarity-only']
    sr, rr = standard['rows'][21], rarity['rows'][25]
    rows = [['Model', 'YOLOv8s', 'YOLOv8s'], ['Training epochs', 80, 80], ['Input size', cfg['imgsz'], cfg['imgsz']],
            ['Physical batch / steady accumulation', f"{cfg['batch']} / {cfg['nbs']//cfg['batch']}", f"{cfg['batch']} / {cfg['nbs']//cfg['batch']}"], ['Nominal effective batch', cfg['nbs'], cfg['nbs']],
            ['Optimizer', cfg['optimizer'], cfg['optimizer']], ['Initial learning rate', cfg['lr0'], cfg['lr0']],
            ['Weight decay', cfg['weight_decay'], cfg['weight_decay']], ['Training seed', cfg['seed'], cfg['seed']],
            ['Selected epoch', 22, 26], ['Native mAP50 at selected epoch (%)', f"{sr['metrics/mAP50(B)']*100:.3f}", f"{rr['metrics/mAP50(B)']*100:.3f}"],
            ['Native mAP50–95 at selected epoch (%)', f"{sr['metrics/mAP50-95(B)']*100:.3f}", f"{rr['metrics/mAP50-95(B)']*100:.3f}"],
            ['Successful optimizer updates', data['standard_completed']['successful_updates'], data['rarity_completed']['accounting']['successful']],
            ['Skipped optimizer updates', data['standard_completed']['skipped_updates'], data['rarity_completed']['accounting']['skipped']]]
    result.append(write_table('table_1_training_checkpoints', ['Setting', 'Standard', 'Rarity-only'], rows,
        'Training and selected checkpoints. Native metrics are the logged values at epochs selected by native eight-class validation mAP50–95; they are distinct from the final evaluator. Steady accumulation excludes the native warmup ramp. Full checkpoint identities are in the source manifest.', '5.1'))
    rows = []
    for method in METHODS:
        ap = test['AP'][method]['overall']; op = test['operating'][method]['overall']
        rows.append([method[-1], *[f'{ap[k]*100:.4f}' for k in ('map50_95', 'map50', 'map75')],
                     op['tp'], op['fp'], op['fn'], f"{op['precision']*100:.2f}", f"{op['recall']*100:.2f}"])
    result.append(write_table('table_2_final_test_performance', ['ID', 'mAP50–95 (%)', 'mAP50 (%)', 'mAP75 (%)', 'TP', 'FP', 'FN', 'Precision (%)', 'Recall (%)'], rows,
        'Final test performance at confidence .25 / IoU .50 for operating counts. A/B/C: standard training; D/E/F: rarity-only. Inference order within each group: Standard, SAHI Context, CG-ALR V3. C is primary. AP uses the common final deterministic evaluator. Displayed values are rounded; exact values remain in the verified Phase 5D results.json. The source manifest records displayed cells and source hashes.', '5.2'))
    rows = [[label, test['support'][c]['objects'], test['support'][c]['patients'],
             *[f"{test['AP'][m][c]['ap50_95']*100:.2f}" for m in METHODS]] for c, label in zip(CLASSES, LABELS)]
    result.append(write_table('table_3_class_AP_support', ['Class', 'Objects', 'Patients', *[c+' AP (%)' for c in 'ABCDEF']], rows,
        'Test class-specific AP50–95 and support. AP columns are percentages rounded to two decimals; tiny V3 changes can disappear at this precision. All 7,361 annotations are included. Bone-lesion and bone-anomaly support is sparse. AP50/AP75 remain available in the verified Phase 5D AP.csv.', '5.3'))
    labels = {'training_standard': 'D−A: training', 'training_sahi': 'E−B: training', 'training_v3': 'F−C: training',
        'sahi_standard_training': 'B−A: SAHI', 'sahi_rarity_training': 'E−D: SAHI', 'v3_standard_training': 'C−A: V3, primary',
        'v3_rarity_training': 'F−D: V3', 'v3_vs_sahi_standard_training': 'C−B: V3−SAHI', 'v3_vs_sahi_rarity_training': 'F−E: V3−SAHI',
        'training_sahi_interaction': '(E−D)−(B−A)', 'training_v3_interaction': '(F−D)−(C−A)'}
    rows = []
    for name, values in test['AP_contrasts'].items():
        ci = test['bootstrap']['contrasts'][name]['overall']['map50_95']
        rows.append([labels[name], f"{values['overall']['map50_95']*100:+.4f}",
                     f"[{ci['lower']*100:+.4f}, {ci['upper']*100:+.4f}]", ci['replicates_present']])
    result.append(write_table('table_4_paired_comparisons', ['Comparison', 'ΔmAP50–95 (pp)', 'Nominal 95% CI (pp)', 'Valid replicates'], rows,
        'All eleven predefined paired test comparisons. Patient bootstrap: 500 draws, seed 314159. Eight-class AP is undefined in samples missing a class; 492 samples contribute here. Secondary intervals are unadjusted for multiplicity. These are not clinical-validation or general training-seed uncertainty intervals.', '5.6'))
    return result
