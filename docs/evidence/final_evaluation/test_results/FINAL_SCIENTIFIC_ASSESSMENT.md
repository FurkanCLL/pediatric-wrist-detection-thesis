# Final scientific assessment

The primary proposed pipeline fails the positive overall mAP50–95 lower-interval criterion. The other eight checks pass. final_C remains the predefined primary configuration, and no new method selection or optimization follows.

## Observed effects and uncertainty

- D−A (training): overall +1.3177 pp; [+0.2208, +3.0145] (n=492); positive nominal interval.
- E−B (training): overall -0.1535 pp; [-1.2205, +1.7359] (n=492); interval includes zero.
- F−C (training): overall +1.3198 pp; [+0.2209, +3.0192] (n=492); positive nominal interval.
- B−A (SAHI): overall -3.1961 pp; [-4.3818, -2.0918] (n=492); negative nominal interval.
- E−D (SAHI): overall -4.6673 pp; [-5.6950, -3.3678] (n=492); negative nominal interval.
- C−A (V3, primary): overall +0.0014 pp; [-0.0046, +0.0076] (n=492); interval includes zero.
- F−D (V3): overall +0.0035 pp; [-0.0043, +0.0124] (n=492); interval includes zero.

Rarity-only effects compare two fixed single-seed training runs. They do not establish an average sampler effect across training seeds. SAHI and V3 effects use the same corresponding checkpoint. Secondary intervals and interaction contrasts are descriptive and unadjusted for multiplicity.

final_C versus final_A: TP +15, FP +33; recovered 15, lost 0; paired cost 1.274×. Strict success: False.
final_F versus final_D: TP +26, FP +29; recovered 26, lost 0; paired cost 1.243×. Strict success: False.

The full fracture AP50/AP75/AP50–95 and uncertainty are reported separately. An overall change does not by itself establish better fracture localization. FP changes are relative to incomplete dataset annotations and are not clinical error rates.

Rarity-only standard inference raises eight-class mAP50–95 by 1.3177 pp, but fracture AP50–95 falls by 0.5537 pp (95% [−1.0834,+0.0283] pp), with 17 fewer fracture TPs. Large class-wise point gains include bone anomaly (+3.6357 pp; 35 objects/14 patients) and bone lesion (+3.2291 pp; five objects/four patients). These small supports limit interpretation. The overall gain under V3 is similar, while the training effect under SAHI is uncertain. This is a mixed result across targets and inference methods, not evidence that sampling improves fracture detection generally.

SAHI adds 170/159 net TPs but also 2,525/2,769 FPs versus the corresponding standard inference. Its overall and fracture AP decline under both checkpoints. V3 exceeds SAHI by 3.1975/4.6708 overall AP points, with positive nominal paired intervals, and uses much less computation. However, V3 has lower operating recall than SAHI. Its advantage over SAHI does not establish superiority over the standard-inference baseline.

V3's overall changes versus standard are only +0.0014/+0.0035 pp, with intervals spanning zero. Its 15/26 recovered objects and 33/29 additional FPs describe a small recall–precision trade-off at bounded cost. All three fracture AP point metrics and both FP limits are preserved; the failed overall interval requirement still prevents strict success. The six non-target classes have exactly unchanged AP under the common deterministic policy.

## Contribution and limits

V3 provides a deterministic, bounded local-inference method with protected evidence reservation, unique ownership, moderated score transfer and structural ROI filtering. Its implementation preserves original global boxes/classes and adds no unanchored detections. Measured AP, recovery, FP and cost determine whether that engineering contribution improves this dataset; the design alone is not evidence of better performance.

Validation references use the same final evaluator, preserving historical metrics separately. Test evaluation was performed once per condition after the code/protocol freeze. This is evidence on unseen patients from the same dataset, not external clinical validation. One training seed, rare-class support, annotation uncertainty, prior validation-based development, and fixed-order laptop GPU sessions remain limitations. Nominal bootstrap intervals do not remove those limits.

The study demonstrates a bounded implementation and a small operating-point recovery–FP trade-off. It does not establish overall AP superiority over standard inference. Stop here: no further training, tuning, method selection, additional test experiments or Chapter 5 figures were performed.
