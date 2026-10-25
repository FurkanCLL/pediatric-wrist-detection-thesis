# CG-ALR V3 — scientific assessment

**PARTIAL IMPROVEMENT** under the predeclared interpretation. This is an exploratory follow-up, not final inference selection. [Detailed results](V3_EXPERIMENTAL_RESULTS.md), [design](V3_DESIGN.md), [aggregate evidence](results.json) and [verification](verification.json) separate measurements from mechanisms and hypotheses.

## What changed

V3 reserves all local evidence compatible with original protected detections, accepts only isolated one-to-one weak/local graph edges, and uses capped equal-midpoint score transfer. Sequential overlap arbitration protects current operating rows from later overlapping updates. The ROI planner retains V2 crop/rank/diversity/coalescing rules but excludes candidates that the same V3 protected-overlap rule already makes ineligible. Geometry is frozen to original global boxes. No new rows, learned calibration, extra detector, fallback proposals, training or threshold optimization are introduced.

This directly addresses V2's ambiguous evidence ownership and correlated positive confirmation. It does not make the model scores calibrated probabilities or establish that a reserved/overlapping row is clinically correct. Unique correspondence is an observable constraint, not proof of correctness. Original-class/box/row-count preservation and exact global reproduction pass for every validation image.

### Standard checkpoint

V3 mAP50–95 versus standard changes by +0.00507 pp historically and +0.00081 pp under deterministic ties. Stable fracture AP50–95 improves (+0.00596 pp); all fracture AP50/AP75/AP50–95 and uncertainty gates are reported separately. At .25/.50 it recovers 11 objects and loses 0. FP is 917 versus standard 884 and V2 1055; V3−V2 is -138. Strict success on both metric tracks: **False**. No checkpoint-specific rule was changed after these observations.
### Rarity checkpoint

V3 mAP50–95 versus standard changes by +0.01406 pp historically and +0.00243 pp under deterministic ties. Stable fracture AP50–95 improves (+0.01993 pp); all fracture AP50/AP75/AP50–95 and uncertainty gates are reported separately. At .25/.50 it recovers 20 objects and loses 0. FP is 1037 versus standard 1016 and V2 1152; V3−V2 is -115. Strict success on both metric tracks: **False**. No checkpoint-specific rule was changed after these observations.

## Recovery, harmful updates and efficiency

Promotion, reassignment and inherited TP/FP transitions are separately counted in the result report; an accepted match or changed score is not automatically useful. Compare true recovered/lost annotations as well as FP counts. Fewer false promotions can sacrifice recoveries or lower AP, so the strict AP/fracture/FP gates remain binding. The PARTIAL classification, when present, denotes only the frozen engineering trade-off definition; it is not a claim that every scientific criterion passed.

Compared with [frozen V2 diagnostics](../phase_5a_analysis/CG_ALR_V2_DIAGNOSTICS.md), standard-checkpoint promoted FPs fall from 171 to 33, while recovered objects fall from 38 to 11. Rarity promoted FPs fall from 135 to 21, while recovered objects fall from 34 to 20; V2 additionally had one TP reassignment and one inherited TP displaced to FP, neither observed in V3. Thus V3 loses 27/14 V2 recoveries, including seven/four fractures, and recovers no object that V2 missed at .25/.50. Fracture AP50–95 is slightly below V2 on both tracks/checkpoints, although all three fracture AP point metrics remain above the corresponding standard inference. Duplicate FPs fall from 221 to 164 (standard) and 174 to 124 (rarity), close to the baseline 163/124. Localization FPs also fall from 362 to 329 and 344 to 318; unchanged boxes mean this is altered operating membership, not better localization geometry.

ROI utilization records actual calls, no-update calls, recovery-producing calls, FP-producing calls and their overlap. Any reduction in total calls is structural pruning, not evidence that reordering alone found previously uncovered fractures. The four predeclared replay cells isolate view availability, arbitration and midpoint effects with geometry held constant; they explain mechanisms without selecting a superior alternative. Official frozen V2 includes its original geometry and remains unchanged.

Local calls fall 23.66% (1,285→981) and 27.33% (1,215→883), but the remaining ROIs are less productive by these outcome definitions: no-update fractions rise 69.81%→87.97% and 69.38%→85.05%; recovery-producing fractions fall 2.88%→1.12% and 2.80%→2.27%. Therefore call reduction is established, while improved useful information per remaining ROI is not. Harmful-promotion ROIs remain 33/981 and 21/883. The complete-path ratios 1.2424/1.2216 satisfy the cost gate; historical-session V2 latency is not a controlled speed comparison.

The fixed attribution replays explain the trade-off. With global geometry and the legacy V1 fusion rule held fixed, substituting V3 views preserves TP counts while reducing FPs by 26/24. Adding V3 arbitration then removes 25/12 TPs and 104/80 FPs. Midpoint moderation removes another two TPs and eight/11 FPs. These are conditional, sequential contrasts on the specified saved views, not order-independent causal effects or alternatives selected for deployment. They support structural filtering and stricter ownership as the main FP-control mechanisms, with a substantial recovery cost.

## AP ranking and measurement artifacts

V3 changes confidence rankings of existing global rows, not localization or detection coverage below the retained .001 baseline floor. Historical pinned AP may also vary because the global final order affects unstable equal-confidence sorting. The deterministic sensitivity track uses immutable image/global IDs and original-ID matching input. Six non-target classes are verified invariant across standard/V2/V3 under that track. Historical-versus-stable differences are measurement sensitivity, not extra detector gains, and cannot turn a failed strict gate into success.

The stable overall V3−standard changes are only +0.00081 pp (95% [−0.00282,+0.00575]) and +0.00243 pp ([−0.00339,+0.01192]); historical changes are +0.00507/+0.01406 pp with intervals also spanning zero. The smaller stable differences show material ordering sensitivity relative to these tiny effects. Stable V3−V2 overall intervals exclude zero only for the standard checkpoint, whereas fracture V3−V2 intervals include zero on both. Rarity fracture V3−D has a positive nominal interval under both tracks; this favorable component does not satisfy the failed overall criterion, account for repeated validation use, or establish an overall method benefit.

## Limits and stop

The algorithm cannot create unanchored missed findings or repair confident global errors; same-model correlated false patterns can still survive its guards. Protected overlap and strict graph uniqueness can reject legitimate nearby findings. Original coordinates deliberately leave localization defects unresolved. Bone lesion still has five objects/four patients, with undefined absent-class bootstrap samples. One training seed per checkpoint, validation-selected epochs, extensive prior validation use, fixed-order GPU sessions and annotation limitations constrain inference. Paired intervals remain conditional descriptive uncertainty; excluding zero does not make this independent confirmation or clinical validation.

The original success gates are reported individually for both checkpoints and both tracks, including the contemporaneous ≤2× cost and actual three-call bound. None was relaxed. Stop after this bounded study: no V4, extra optimization cycle, new training, final method selection or held-out evaluation is authorized.

Both checkpoints fail the required positive overall mAP50–95 lower interval bound under both tracks; all other listed strict gates pass. The retained historical `secondary_tradeoff_success` field is true for rarity, but it is not the frozen `strict_success` criterion and does not change this conclusion. The PARTIAL label follows the predeclared lower-FP-with-positive-recovery definition on both checkpoints, not demonstrated overall AP superiority or preservation of V2 recovery.

**PARTIAL IMPROVEMENT.**
