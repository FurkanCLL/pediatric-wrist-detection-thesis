# Confidence-Guided Adaptive Local Refinement: proposed technical specification

Phase 3C.1, 2026-10-17. **Design only; unapproved and not frozen.** CG-ALR is an engineering hypothesis for selective reinspection with the same YOLOv8s checkpoint. This document specifies a complete recommended interpretation for review; it reports no adaptive inference, GPU experiment, model training or held-out evaluation. No production algorithm is implemented. The student must authorize Phase 3C.2 separately.

The [proposed configuration](../../../configs/adaptive_refinement_proposed.json) records the numerical choices, meanings, alternatives and approval state. Its presence and hash are not authorization to execute it. This is development documentation, not student-authored final thesis prose or a claim of clinical validity.

## 1. Motivation and inspected evidence

Source review starts from repository revision `a709747438977f071ede9995fe3e9efbb418f9b3`. The [source audit](../dataset_audit/DATASET_AUDIT_REPORT.md), [preparation report](../dataset_preparation/DATASET_PREPARATION_REPORT.md), [training protocol](../training_protocol/TRAINING_PROTOCOL_REPORT.md), [baseline preservation](../class_aware_design/BASELINE_PRESERVATION_REPORT.md), [class-aware proposal](../class_aware_design/CLASS_AWARE_DESIGN_REPORT.md) and [Phase 3B report](../error_analysis_sahi/ERROR_ANALYSIS_SAHI_REPORT.md) were reviewed alongside `AGENTS.md`, `docs/THESIS.md`, `docs/ROADMAP.md` and `docs/DECISIONS.md`. Historical reports describe their own completion dates; their earlier “not trained” statements do not supersede the completed baseline.

The prepared data retain source spatial geometry and use the frozen full-range 16-bit-to-8-bit conversion, without enhancement. Polygon targets are axis-aligned envelopes. The eight IDs are 0 fracture, 1 metal, 2 text, 3 periostealreaction, 4 pronatorsign, 5 bonelesion, 6 softtissue and 7 boneanomaly. Excluded foreign-body labels are background, not ignore regions. Empty labels do not establish healthy radiographs. Patient grouping and annotation semantics remain unchanged.

The standard run completed 80 epochs; the selected checkpoint is epoch 22, SHA-256 `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`. Its reproduced native validation mAP50/mAP50-95 is 0.59908223/0.33269755 and fracture AP50/AP50-95 is 0.92116190/0.51973089. The [evaluation bridge](../error_analysis_sahi/evaluation_bridge.json) attributes the material difference below to original-image clipping, principally text boxes. It does not identify a better checkpoint.

The following values were checked against [aggregate errors](../error_analysis_sahi/error_analysis.json) and the preserved `full_*.json` summaries, rather than copied solely from the task description. All are validation/development evidence on 3,088 images, 914 patients and 7,204 objects.

| Common original-coordinate method | mAP50 | mAP50-95 | Fracture AP50 | Fracture AP50-95 | TP / FP / FN at .25, IoU .50 | Mean ms |
|---|---:|---:|---:|---:|---|---:|
| Standard serial | .59900535 | .33386896 | .92117499 | .51973720 | 6467 / 884 / 737 | 62.33 |
| SAHI context | .52576800 | .29314538 | .86714508 | .49159838 | 6599 / 3209 / 605 | 238.13 |
| SAHI detail | .48217326 | .26255895 | .82825523 | .46864979 | 6556 / 6328 / 648 | 419.23 |

Context slicing recovers 148 targets and loses 16, including 81/4 fractures and 44/6 periosteal targets. Of the 81 recovered fractures, 77 are baseline low-confidence residual misses, 68 have a same-class nearby prediction at confidence at least .05, and 54 have a nearby prediction shorter than 64 effective pixels. The corresponding periosteal figures are 44 low-confidence misses and 42 with a same-class nearby prediction at .05. These are separate, retrospective GT-conditioned counts. They do **not** establish their intersection, eligibility in [.05,.25), crop coverage, or a global/local IoU-.50 confirmation rate. The periosteal small-candidate diagnostic in Phase 3B also used 64 pixels, not the new 32-pixel priority reference.

Thirteen of the recovered fractures have no prediction of any class near the GT at .05. An anchor-only method cannot promise those recoveries. The negative SAHI AP differences and increased partial/duplicate predictions motivate strict admission control. Existing patient-bootstrap intervals support regressions for the two studied grids; they neither prove CG-ALR will help nor justify a new threshold search.

## 2. Objective, scope and invariants

CG-ALR runs one global view, plans zero to two local views, and uses their agreement to update existing eligible global rows. V1 changes scores only. V2 uses the same confirmation matches and scores, with a small guarded coordinate blend. Neither accepts independent new local detections. Only fracture and periosteal predictions drive selection or receive updates. There is no image-only fallback, extra disagreement pass, second network or checkpoint switching within an image.

Invariants are: one checkpoint per run; unchanged eight-class IDs; original-coordinate finite outputs; at most three model calls per image; output count exactly equal to the global count, at most 300; no global deletion; no modification of noneligible rows; and no GT argument in planning, inference or fusion. All confidence/geometry decisions use original global predictions, never successively boosted predictions. The algorithm and selected variant must eventually be identical for the standard and class-aware checkpoints. Their naturally different predictions may produce different ROI plans.

## 3. Global inference and baseline compatibility

Reuse the verified numerical path in `src/wrist_eval/sliced.py`: `make_adapter(model)`, `resize_and_pad(image, full_shape)`, pinned `DetectionValidator.postprocess`, `ops.scale_boxes`, padding-only removal and `validate_predictions`. Preserve FP32 saved EMA weights, unfused Conv/BN, eval mode, disabled gradients, CUDA float16 autocast and half inputs divided by 255. Decode existing prepared pixels into three equal channels. Use 768, sequential batch one, multi-label class-aware NMS at confidence .001 / IoU .7, max_det 300, no TTA or training transforms. No score recalibration or grayscale enhancement is introduced.

The retained pool G includes valid post-NMS predictions down to .001; it is not the ROI-candidate pool. Set global ID i to the original row position in this verified, stably confidence-sorted pool. Keep a byte-identical immutable copy. “Complete pool” means retained post-NMS/cap output, not raw anchors or predictions discarded by native NMS.

**Shape compatibility is an explicit integration requirement.** Phase 3B passes the saved native rectangular-batch `net_shape` to serial inference. Calling `resize_and_pad` without it can choose a different padded shape. Phase 3C.2 must extract a private, checkpoint-independent dimension/shape profile from the preserved reference metadata, bind its membership and hash, and pass the same shape for each paired global view. The adaptive core receives only image dimensions and a validated shape, never annotations or another checkpoint's prediction values. Both checkpoints use the same profile. A future arbitrary-image service needs a declared geometry-only global shape policy and a paired standard control; it must not silently claim equivalence to this saved validation profile. No new shape policy is frozen here.

Current `common.load_model()` hardcodes the standard best path/hash and `feasibility.run()` mixes inference with GT evaluation. Neither is a generic adaptive entry point. Future loading must verify a run-supplied checkpoint identity and its corresponding approved training manifest before extracting EMA. Leave the frozen trainer, protocol lock and existing preservation receipts untouched. Load one model once, not once per ROI.

## 4. Candidate eligibility and coordinates

Use continuous xyxy image-edge coordinates: 0<=x1<x2<=W and 0<=y1<y2<=H. Areas use width times height, without a +1 pixel convention. Let D=max(W,H), r=768/D, and s_i=r min(x2-x1,y2-y1). An anchor candidate must have class in {0,3}, .05<=p_i<.25, valid geometry and s_i>=1. The one-pixel minimum is an explicit engineering safeguard against near-degenerate high-size-priority artifacts; it is not a dataset-derived optimum. Scores below .05, scores at/above .25 and the other six classes remain in G unchanged.

The 64/32 references affect priority, not maximum eligibility. Large boxes may still be candidates, but planning can reject a crop with no enlargement or invalid containment. Boundary-adjacent predictions remain eligible. No extra class-agnostic NMS deletes global rows. Existing native same-class suppression remains; overlapping same/different-class candidates are resolved at the ROI planning stage. Multiple surviving candidates can correspond to one finding; this cannot be known from predictions alone.

If there are no geometrically feasible candidates, return G with original values and order and execute zero local calls. An empty G follows the same rule. A zero candidate score is allowed when eligibility passes; there is no hidden score floor.

## 5. Scoring equations and interpretation

For candidate class c, use t_0=64 and t_3=32 effective pixels:

```text
clip01(a) = min(1,max(0,a))
Q_i = clip01((p_i - 0.05) / 0.20)
Z_i = clip01(1 - s_i / t_c)
S_i = 0.6 Q_i + 0.4 Z_i
```

Q increases with credibility within the selected weak-score interval. Using decreasing confidence as “uncertainty” would reward the weakest admitted artifacts; correcting that interpretation requires no coefficient change. Z is zero at and above the class reference and rises toward one for smaller boxes. All components are bounded in [0,1]; p<.25 means Q<1 in exact arithmetic. Different class references make sizes relative to the declared class scale, not calibrated class-wise probabilities.

For example, fracture p=.20 and s=32 gives Q=.75, Z=.50, S=.65; periosteal p=.20 and s=16 gives the same score. A tiny fracture p=.05, s=1 scores .39375, below a reference-sized p=.20 candidate scoring .45. Smallness can still outrank modestly stronger evidence: these weights encode an engineering preference, not expected AP gain. There is no sum over neighboring candidates and no confidence-entropy/GT term.

Rank by (-S,-p,s,i), with exact comparisons after conversion of original float32 inputs to CPU float64. Global ID is the final tie-break, preserving the pinned native order rather than adding an undocumented preference for a target class. Exact ties may therefore inherit native ordering; no random seed search or score rounding resolves them.

## 6. Deterministic ROI selection and budget

First build a feasible crop for each candidate using section 7. Remove infeasible/no-zoom plans, retaining reasons privately. For each represented target class take its highest-ranked feasible candidate. Process these one-per-class heads in the common score order. Then process the remaining candidates in pooled score order. This gives class diversity without a fixed fracture-first or periosteal-first quota. It can deliberately select a lower-scoring class head before a second same-class candidate; record that opportunity cost.

When considering a candidate crop R, compare with the already selected ROIs. Coalesce its seed into an existing ROI only if crop IoU>=.70 **and** that existing ROI fully contains the candidate's expanded integer anchor envelope. Pick greatest IoU, then earliest ROI ID. Keep the existing crop unchanged: do not enlarge a union, average centers, sum scores or run another view. Otherwise append R if a slot remains. Continue coalescing/refilling until two distinct crops are selected or candidates are exhausted. Candidates left after the budget fills receive `budget_exhausted`; fusion may still confirm one if it is fully covered by a selected crop.

If two class heads coalesce, both classes have had an initial opportunity and the remaining slot may come from either class. If only one class is feasible, both slots may serve it. Crop overlap alone does not prove redundancy; the containment condition prevents dropping a seed whose context would be cut off in the earlier crop. Distinct overlapping ROIs may remain when containment fails. No dynamic extra pass repairs that coverage limit.

## 7. Context geometry and coordinate restoration

Fixed 75% dimensions give approximately 56.25% image area and at most 4/3 ideal linear enlargement, not a 25% enlargement and not a guaranteed clinically adequate field. This is broader than the negative SAHI crops but provides less detail. A fixed centered 75% rectangle can cut off an unusually large anchor. Recommend one minimal enlargement derived from the existing prediction, without a crop-size search.

Let m=2D/768 original pixels, using the same proposed two-effective-pixel safeguard scale. Construct an integer envelope around the prediction:

```text
E_left   = floor(max(0,x1-m)); E_top    = floor(max(0,y1-m))
E_right  = ceil(min(W,x2+m)); E_bottom = ceil(min(H,y2+m))
f  = max(0.75, (E_right-E_left)/W, (E_bottom-E_top)/H)
cw = min(W,ceil(f W)); ch = min(H,ceil(f H))
```

This retains aspect ratio to outward pixel rounding; it never stretches anatomy. For the x origin, L=max(0,E_right-cw), U=min(E_left,W-cw). Choose x0=clamp(floor((x1+x2-cw)/2+0.5),L,U), and use the analogous y formula. These integer intervals guarantee that the expanded envelope fits even when nearest-center placement would fail. Reject empty intervals, invalid dimensions or max(cw,ch)>=D (no ideal long-side enlargement). Do not consume a model-call slot for rejection. Tiny/extreme-aspect images may consequently use only their global view. Dimensions and pixel coverage are defined even when width/height are odd.

Use the immutable prepared subarray image[y0:y0+ch,x0:x0+cw]. Original image borders may prevent symmetric context; clipping the expanded envelope makes that limitation explicit. Other anatomical objects may be cut despite seed containment. Unlike training simulations, no GT visibility/fracture-retention test is allowed at inference.

For each local view reuse the existing resize and half-stride padding policy, without supplying the global batch shape. If actual resized dimensions are rw,rh, use g_x=rw/cw and g_y=rh/ch and the actual integer pads. The pinned `ops.scale_boxes` uses these separate gains. For a network x coordinate, local u=(x_net-pad_x)/g_x; similarly v=(y_net-pad_y)/g_y. Clip to crop bounds through the existing routine, count/remove padding-only collapsed boxes, then add (x0,y0) to both corners **once**. Validate in original W,H bounds. Do not invert an ideal single gain or apply the offset twice.

Before local evidence can enter fusion, reject a restored crop-coordinate box when its distance to an artificial edge, in resized content pixels, is <=2: u1*g_x on the left, (cw-u2)*g_x on the right, v1*g_y on top, or (ch-v2)*g_y on bottom. Test a side only if its crop edge is inside the original image, e.g. left when x0>0. Physical image edges are exempt. This prediction-only guard catches some crop fragments but cannot prove an object is complete; interior fragments and truncated source-image objects remain possible. Preserve rejection counts rather than treating the guard as verified protection against every FP.

## 8. V1: confirmation-only fusion

Fusion anchors are **all** eligible candidates from section 4 whose expanded E fits at least one selected ROI, not only the one/two ROI seeds. This uses already-paid views without accepting unanchored detections, but can update more than two global scores per image. That broader update scope requires approval and explicit counts.

Keep local predictions only as evidence: class 0/3, local score l>=.25, valid geometry, and passing the source ROI's artificial-boundary guard. Remove redundant local evidence across ROIs using deterministic greedy same-class NMS at IoU>.7, descending local score then ROI ID/per-view local ID. Keep the winning view and its provenance; do not average confidence or fuse its coordinates. This local-only stage never suppresses a global prediction.

Create an eligible global/local pair only for the same class, IoU>=.50 in original coordinates, and full E containment in that local detection's **own source ROI**. Sort pairs by descending IoU, descending local score, descending original global score, then ascending global ID, ROI ID and local ID. Greedily accept a pair only when neither ID is used. One global and one retained local can be consumed once across both ROIs. Use these same matches for V1 and V2, based on original G. No rematching follows a geometry veto.

For a matched anchor, update:

```text
p_new = min(p_original + 0.10, p_local, 0.35)
box_new = box_global_original
```

Because p_original<.25 and p_local>=.25, this never decreases its score and never raises it above the local evidence or the declared .35 ceiling. Apply once, without accumulation across views. An unmatched anchor, wrong-class local box, rejected boundary box or empty local result leaves its global row unchanged. Local non-target classes can be logged but cannot confirm a different target or add output rows. A conflicting class prediction does not cancel or relabel the global detection; that is an acknowledged limitation.

**Do not rerun global NMS.** All V1 boxes and row membership are unchanged; a new score order followed by NMS could delete existing global rows and break protection. Existing global duplicates may remain, including boxes made more overlapping by original-image clipping. The contract is no new rows, not proof of one output per biological finding. Use stable descending updated score / original-ID ties, and the original order if no field changes. Assert the count remains exactly |G|<=300; there is no later cap that can displace unrelated classes.

Boosting can promote a TP or an FP and change AP assignment/ranking. In exact arithmetic p<.15 cannot cross the .25 operating point with a +.10 cap; p>=.15 may cross only with sufficient local agreement. Scores cannot reach .50, so the declared confidence-.50 output is exactly unchanged for both variants. No GT-free formula establishes that the boosted anchors have better localization. Preserved global boxes alone do not guarantee preserved AP or per-GT identities under one-to-one matching.

## 9. V2: guarded localization refinement

The provisional IoU-.35–.50 replacement case is unsafe as an initial default: lower overlap weakens evidence that the two predictions describe the same finding, while local confidence .50 is not an IoU-quality estimate. Whole-box replacement could move an anchor onto a nearby fragment. **Recommended correction for approval:** use the same IoU>=.50 pairs as V1 and a 25% local blend, disabling lower-overlap replacement. This keeps two initial variants and changes one intended factor between them.

V2 inherits V1 scores exactly. For a matched anchor whose local score is at least .50, propose B*=.75 B_global+.25 B_local. Otherwise retain its global coordinates. Blend each xyxy coordinate, preserving class identity; do not blend centers with differently defined widths. Clip/quantize the proposed box to float32, validate it, and process geometry proposals in ascending original global ID. Comparing the values actually committed avoids a float64-pass/float32-collision discrepancy.

Before accepting B*, compare it with every other current same-class output box C. Reject if IoU(B*,C)>max(.70,IoU(B_current,C)). Thus an update cannot create a new above-.70 pair or increase an inherited above-.70 overlap. This is an incremental coordinate veto, not NMS: keep both original global rows. On invalid geometry or collision veto, retain the original global coordinates and the V1 score. Do not try another local pair. Protected boxes win by never being moved/deleted. Earlier accepted low-score moves have deterministic precedence; report veto counts and order effects.

Convex blending of valid boxes preserves positive dimensions before quantization and bounds each coordinate displacement to one quarter of the local/global difference. It does **not** guarantee higher GT IoU, correct anatomical correspondence or nondecreasing recall. Two nearby findings, an interior local fragment or correlated model error can still cause harm. The .25 blend and .50 local-score gate are conservative engineering hypotheses, not calibrated quality tests. No independent low-overlap variant or replacement sweep is included.

## 10. Pseudocode, numerical rules and edge cases

The following is normative pseudocode, not an implementation or executable launch interface. IoU uses continuous area and denominator union+1e-7; planning/matching uses CPU float64 from original float32 rows. Inclusive thresholds use >=, except candidate upper score <.25 and NMS/collision comparisons >. Final predictions are float32 N x 6 [x1,y1,x2,y2,confidence,class_id].

```text
CG_ALR(image, verified_runner, global_shape, proposed_settings, variant):
    G := verified_runner.global_view(image, global_shape)  # same numerical path
    validate G; keep original IDs and immutable G0
    A := eligible targeted anchors from G0; compute Q,Z,S and integer E
    P := feasible crop plans, keyed by the declared score order
    H := best plan of each represented class, sorted by that same key
    R := empty list
    for plan in H followed by remaining plans in score order:
        if a selected ROI has IoU>=.70 and contains plan.E:
            attach seed to the best matching ROI; continue
        if |R|<2: append plan with the next ROI ID
        else: record budget exhaustion
    if R is empty: return unchanged G0
    L := empty evidence list
    for roi in R:                           # one call each, no retries/new views
        local := verified_runner.local_view(original prepared crop)
        restore to crop coordinates; count padding-only omissions
        reject artificial-edge evidence; shift once to original coordinates
        retain valid targeted evidence with local confidence>=.25 in L
    L := deterministic local-only class-aware NMS(L, .70)
    pairs := eligible same-class IoU>=.50 edges, sorted as section 8
    M := greedy one-to-one assignment over pairs using original G0
    Y := copy of G0
    for (g,l) in M: Y[g].score := min(G0[g].score+.10,l.score,.35)
    if variant is V2:
        for matched global ID g in ascending order:
            if its matched local score>=.50:
                proposed := float32(clip(.75*G0[g].box+.25*L[g].box))
                if valid and no collision increase: Y[g].box := proposed
                else: record veto; retain original coordinates
    assert |Y|=|G0|<=300, unchanged IDs, finite valid original bounds
    assert every noneligible row unchanged; model_calls=1+|R|<=3
    return Y in stable updated-score/original-ID order, or G0 if unchanged
```

Here `L[g]` denotes the local box assigned to g in M, not another list lookup or match. The caller must not silently run both variants as six model calls under a three-call claim. They can share the same image-level view bundle because selection/inference do not depend on the fusion variant; separate timing semantics are specified in section 16.

| Situation | Required behavior |
|---|---|
| Empty pool; no target class; all scores outside interval | Exact G0 return, one global call |
| Tiny anchor below one effective pixel | Ineligible and unchanged; no artificial large priority |
| Large anchor needs almost full image | Minimal containment expansion; skip if no long-side enlargement |
| Boundary/narrow/odd-sized image | Integer containment placement; physical-edge exemption; no stretch |
| Two targets/classes propose same useful crop | Coalesce under both overlap and containment tests |
| More than two distinct plans | Record budget exclusion, never add a third local view |
| No usable local evidence, class conflict or low IoU | No change to that global row; no retry or relabel |
| Several evidence boxes/ROIs compete | Local-only suppression, then one global/local assignment |
| V2 collision/quantization invalidity | Retain original coordinates with V1 score |
| Native padding-only box | Count and omit via the existing original-coordinate export policy |
| Non-finite raw/global/local values, wrong hash/class mapping, CUDA/OOM | Fail attempt, preserve logs; no silent drop, FP32 fallback, crop change or automatic rerun |
| No net changes after useful local calls | Return original rows/order but retain the actual extra cost |

## 11. Realistic benefits and fundamental limits

Well-localized low-score anchors can move up the ranking or cross .25 under V1; inaccurate anchors cannot gain geometric IoU from V1. V2 may improve or worsen a matched low-score box. Completely absent detections, anchors removed by global NMS/cap, p<.05, p>=.25, subpixel anchors and non-target classes cannot be directly refined. A target can incidentally lie in a selected crop, but its unanchored local box is still rejected. Two broad views may miss other candidate regions.

Both variants preserve the complete original prediction count, not the number of predictions above an operating threshold. FP at .25 can increase when a weak existing FP is promoted. AP can improve or decline even with unchanged V1 boxes. At .50, all rows/boxes admitted by that confidence are unchanged, which is a useful falsifiable invariant. At .05, V1 retains the admitted set but can change its matching order; V2 can also change low-score coordinates. Recovered/lost GT counts must therefore be measured, not inferred from unchanged membership.

The six nontarget classes retain their exact prediction fields, and their within-class stable order/fixed-threshold counts should remain unchanged. Audit all eight classes anyway: the pinned `ap_per_class` uses `np.argsort(-conf)` without a stable-sort request, so altered cross-class interleaving and tied scores can produce numerical ordering effects. Its mean-F1 operating confidence is also shared across classes and may change auxiliary precision/recall despite unchanged nontarget rows. Preserve the evaluator; investigate/report tie effects rather than altering the metric or quietly accepting a broad tolerance. Class-agnostic confusion can change when targeted predictions change.

## 12. Cost and estimated pass budget

Let n=|G|<=300, a<=n eligible anchors, k<=2 ROIs and l<=300k local rows before filtering. Candidate construction/sorting costs O(a log a); selection compares each plan with at most two ROIs. Local evidence suppression is at worst O(l^2); building/matching edges costs O(al+e log e), e<=al, and V2 collision checks O(n times matched anchors). These small CPU bounds are planning estimates; measure their actual cost on Windows. No data-sized image-pair matrix or learned proposal network is needed.

Calls per image are exactly 1+k. If fractions of images with zero/one/two local views are f0,f1,f2, mean passes=1+f1+2f2, between one and three. Distribution is unknown until approved inference. Broad crops enlarge by 1 to approximately 1.333, less than the SAHI context 1.5x nominal view scale. Linear scaling from Phase 3B is not a timing prediction: decoding, shapes, local NMS, synchronization, laptop thermals and fusion differ. Standard 62.33 ms and context 238.13 ms are measured references, not promised CG-ALR bounds. New VRAM peaks and candidate coverage are unmeasured.

## 13. Accuracy and false-positive risks

The same model sees correlated global/local evidence, so confirmation is not an independent vote. Up to .10 score adjustment can outrank original .25–.35 detections, changing AP matching even though high-score fields are protected. Wrong anchors can be confirmed consistently. The confidence interval excludes p>=.25 errors, weak-score anchors below .15 cannot reach .25 under V1, broad crops may supply too little zoom, and the boundary guard rejects some correct near-edge findings.

V2 adds localization risk despite its tether and collision veto. Full-row protection is stronger than post-merge global NMS protection, but cannot certify fracture AP preservation. Existing duplicate global predictions can both be boosted by distinct residual evidence. Coalescing and diversity can suppress a more useful second crop; matching by highest IoU can select a .30 local confirmation over a lower-IoU .80 box, preventing V2 geometry on that anchor. This is intentional shared matching, not an undocumented rematch favoring V2. All these risks are measurable failure modes, not reasons to tune repeatedly until a gain appears.

## 14. Planned architecture and interfaces

No files in `src/`, tests or dependencies change in Phase 3C.1. The following are future interfaces, not available code:

| Planned location / interface | Responsibility and boundary |
|---|---|
| `src/wrist_eval/adaptive.py` | Pure candidate scoring, geometry, plan selection, boundary checks, evidence matching and V1/V2 fusion; core accepts arrays/dimensions, no filesystem, GT or checkpoint-condition flag |
| `plan_rois(global_predictions, image_hw, settings) -> RoiPlan` | Stable IDs, eligibility/rejection reasons, scores, crop rectangles and coalesced seed IDs; zero to two views |
| `fuse_predictions(global_predictions, local_views, plan, settings, variant) -> FusionResult` | Exact match pairs, updated rows and veto reasons; no model calls |
| `predict_adaptive(image, infer_view, global_shape, settings, variant) -> PredictionResult` | Orchestrates the same verified global/local view callback, enforces call count and aggregates timings |
| `src/wrist_eval/adaptive_study.py` | Explicit approved-run gate, generic verified EMA loader, shape-profile export/use, existing-adapter wrapper, protected output directories, timing/provenance and separate evaluation driver |
| Existing `metrics`, `analyze`, `bootstrap` utilities | Same AP, descriptive matching, recovery and paired patient resampling; minimally accept supplied named prediction bundles instead of fixed SAHI keys, preserving their original defaults/outputs |

Reuse `make_adapter(model)` and SAHI `get_prediction` for a single view rather than introducing another model-forward/preprocess implementation. A thin wrapper must set global dimensions/shape correctly, use the local default for genuine smaller crops, snapshot counters per view and expose valid Nx6 arrays. The adapter's current mutable global shape/counters mean concurrent calls on one instance are forbidden. Use sequential batch one and one CUDA context. Avoid reaching into private adapter fields throughout the core; contain any compatibility access in the wrapper and verify it against the existing standard path.

Future data records: Candidate(global_id,class_id,original_box,original_score,s,Q,Z,S,integer_E); Roi(id,xyxy_integer,seed_ids,score_key,actual_zoom); LocalView(roi_id,original_coordinate_rows,source_local_ids,actual_gains,pads,boundary_flags,counters); Match(global_id,roi_id,local_id,IoU,local_score); FusionResult(rows,matches,score_changes,coordinate_changes,veto_counts); PredictionResult(global_rows,rois,local_views,variants,passes,timings). Immutable originals and IDs remain accessible even after sorting.

Run manifests separate algorithm settings from checkpoint path/hash, corresponding approved training identity, shape-profile hash, split/preparation identities, environment/source revision and variant. Do not bake the standard best SHA into ROI logic or alter parameters for the class-aware model. The same loader contract must validate each real checkpoint's own provenance; a future class-aware hash/path cannot be fabricated now. Same-run global/local weights must be identical and never changed by `.train()` or `.fuse()`.

Private per-image logs should record candidate counts/rejection reasons, selected/coalesced/dropped seeds, ROI rectangles, actual passes, local evidence filters, pair decisions, score/box deltas, collision vetoes and synchronized timing/memory. Keep prediction arrays, identities, crops and overlays ignored under a new attempt-specific `outputs/adaptive_refinement/` root. Public outputs contain aggregates/hashes only. Fail on existing attempt directories, wrong status/approval, unexpected schema, invalid inputs or identity drift. Windows uses protected paths and sequential inference, not multiprocessing/persistent GPU workers. Existing pins suffice; no dependency addition is proposed.

## 15. Planned technical verification

Phase 3C.2 first needs pure, synthetic tests, separately from GPU integration. Cover exact interval endpoints .001/.05/.15/.25/.50; class IDs; Q/Z boundedness and monotonicity; size references as priorities; stable ties and reversed input with stable IDs; zero candidates; class-head diversity/refill; overlap without containment; full-envelope coalescing; at most two plans; large/one-pixel/narrow/odd/border crop geometry; source-pixel immutability; per-axis resize/offset round trips; artificial versus physical boundaries; wrong-class and one-to-many matches; one global/local use across ROIs; one boost only; all confidence caps; unmatched/veto fallback; inherited versus new collisions; protected rows; exact .50-output invariance; final count/order and no final global NMS. Include an adversarial example where confidence confirmation or blending worsens GT IoU/AP. Such tests prove rules, not method benefit.

The GPU pilot, only after authorization, uses the unchanged Phase 3B 24-image selection and synthetic edge fixtures. Annotation-based pilot coverage selection stays outside the inference API; no GT enters a crop decision. Verify checkpoint/EMA/precision/class identity, global output equality against preserved serial predictions at identical shapes, finite local outputs, all counters, geometry overlays and repeat determinism. Exact equality is expected in the pinned environment; >1e-6 score or coordinate differences trigger investigation rather than a widened acceptance tolerance. Check zero/one/two views when present; if real cases do not exercise a branch, use stubbed synthetic integration rather than choosing favorable patients. Do not claim a branch tested when no case reaches it.

Proposed resource gates inherit the Phase 3B 4 GiB peak tensor allocation and three-hour projected per-variant budget, with all calls<=3 and no concurrent model workload. Log peak allocated/reserved and available total-device memory separately; desktop/driver use is not tensor allocation. OOM/non-finite/integrity failures stop, preserve evidence and require diagnosis; there is no automatic smaller-crop or FP32 fallback. Source/interface repairs may be tested on bounded fixtures but cannot silently change approved scientific rules. All GPU and full-validation tests in this section are **planned, not performed**.

## 16. Controlled Phase 3C.2 validation study

After approval and the pilot gate, compare preserved standard serial and SAHI context with V1 and V2 on the same 3,088 validation images, frozen original-coordinate annotations and pinned metrics. Reuse completed standard/SAHI prediction receipts and timing summaries; do not rerun them just to improve a comparison. Verify their hashes. Each new adaptive full run includes its actual global pass; it cannot claim end-to-end speed from reading cached global predictions. Global outputs must reconcile per image with the saved serial reference before attributing differences to refinement.

V1/V2 have identical plans and model views. Their accuracy outputs may share one protected view bundle and branch into two CPU fusion operations, avoiding duplicate neural work. Report actual physical calls and the logical 1+k calls for each method. For full-set timings record synchronized common-stage cost and each fusion cost separately; `common + own_fusion` is an explicitly **composed per-variant latency**, excluding the other fusion branch. Do not present the whole dual-output wall time or half of shared GPU cost as single-method latency. Before any AP is inspected, use the fixed pilot to measure each standalone complete path and confirm timing boundaries/overhead; if stage sums do not faithfully describe standalone execution, retain them as diagnostics and measure separate complete adaptive paths. Every standalone invocation still has at most three calls. Preserve all attempts and avoid choosing a faster reported boundary after seeing outcomes.

Warm up three fixed synthetic images/calls as in Phase 3B, record cold load/warmup separately, synchronize CUDA around each stage and whole-image measurement, and use CUDA events for forward intervals. End-to-end starts before decoding and ends after final fusion/sort/cap assertion, excluding GT matching, AP and export. Record decode, planning/crop/preprocess, forward, per-view NMS/restoration, local-evidence suppression/matching and fusion; mean/median/p95/p99, throughput, peak allocation/reservation, zero/one/two-view image counts, mean passes and actual total calls. Loop wall time includes logging/export separately. No predicted latency replaces a measurement, and historical desktop conditions limit cross-session timing claims.

Report all eight classes: mAP50, mAP75, mAP50-95; per-class AP50/AP75/AP50-95; fracture scores at all three; fixed confidence .05/.25/.50 crossed with IoU .50/.75; TP/FP/FN, FP/image, precision/recall; per-GT both/recovered/lost/neither; matched-IoU changes on common detections and localization strata. Separately report candidate/proposal burden, eligible anchors inside the budget, confirmation/boost/geometry acceptance rates and boundary/collision losses. GT-conditioned coverage diagnosis is computed only after predictions are sealed, never used to alter a plan. Candidate coverage is not detection recovery.

Use 500 paired patient-cluster resamples, seed 314159, all views/examinations from each of the 914 sampled patients, with identical sampled indices across all four methods. Preserve original bootstrap receipts; new multi-method calculations get a separate destination. Record missing-class replicates and omit undefined eight-class mAP rather than substituting zero. Add paired intervals for AP differences, fixed-threshold recall/FP-per-image differences and V2-minus-V1. Intervals are conditional on this checkpoint and do not correct method selection or repeated validation reuse.

## 17. Success, failure, comparison and stopping policy

These proposed engineering gates need approval before execution; none is a clinical safety threshold or a result obtained here.

| Proposed gate | Exact interpretation |
|---|---|
| Primary AP benefit | Overall mAP50-95 point difference>0 and paired 95% lower difference bound>0 against standard; overall mAP50 point difference>=0 |
| Fracture preservation | AP50, AP75 and AP50-95 point differences>=0; each paired 95% lower bound>=-.005 absolute (0.5 percentage points), a proposed tolerance rather than proven noninferiority |
| FP control | At .25 / IoU .50, all-class and fracture FP counts each <=1.10 times their standard count; for the present baseline integer limits are 972 and 380; report FP/image difference/ratio uncertainty |
| Useful cost | Mean complete-path or explicitly validated composed latency<=2 times paired standard mean; historical planning value approximately 124.67 ms; report tails and differences in measurement session |
| Technical validity | Pilot/integrity/pass-count gates pass, no unreported failure/fallback and no protected-row mutation |

A positive point estimate with an interval spanning zero is inconclusive, not confirmed improvement. Report whether the FP ratio's uncertainty exceeds 10%; passing a point tolerance alone does not prove the increase is bounded. A predeclared secondary trade-off is considered only if overall mAP50-95 has nonnegative point change and lower interval>=-.005, fracture preservation/FP/cost gates pass, and fracture or periosteal AP50-95 has a positive paired lower difference bound. Label that outcome target-specific development evidence, not overall AP superiority. Otherwise retain the negative/inconclusive result and prefer the standard baseline for an accuracy/cost default.

If both variants pass, default to simpler V1 unless V2 also shows a positive paired lower bound for overall AP50-95 versus V1 with no added preservation/FP/cost failure. Report both regardless of selection. No per-image oracle choice between variants, per-checkpoint variant change or best-class-only reporting is allowed. These are development alternatives; one student-approved adaptive rule must later serve both checkpoints in thesis C/F. The original two-training by three-inference matrix is unchanged.

Stop on technical invalidity or after one complete predefined two-variant comparison. Do not grid-search confidence bands, crop fractions, overlap, score weights, blend coefficients or boost amounts. A bug repair must preserve the intended algorithm, disclose invalidated evidence and rerun only affected checks. A methodological revision after results needs a new written rationale, exact changes and student authorization, even if the initial results are negative. No automatic transition from this document to Phase 3C.2 is authorized.

## 18. Reproducibility, evidence and leakage safeguards

Hash canonical sorted algorithm JSON, the full proposal file, source revision/normalized modules, dependency/source pins, checkpoint bytes/sidecar and approved training identity, fixed split/preparation journals, shape profile and input membership. Run identity binds all these plus variant and attempt; checkpoint/shape manifests are separate from the common algorithm parameter hash. V1/V2 share all parameters except the declared geometry rule. Preserve proposal version and explicit unapproved status; a future approved execution manifest must reference it without silently editing frozen history.

Inference signatures accept image pixels, model outputs and metadata geometry only. Evaluation code receives GT after output publication. Keep image/patient associations, detailed traces and raw timings private. Never read test image/label contents during development; do not change class-aware A/B/C from these validation errors. The baseline checkpoint and 80-epoch protocol stay immutable. Formal supervisor acceptance of revised scope remains pending. Final held-out A–F evaluation waits for both real checkpoints, one common adaptive rule and all inference protocols to be approved/frozen.

This phase's verification is restricted to source/configuration/document inspection, JSON/links/hashes, mathematical examples and publication scope. The existing 110-test and GPU receipts are historical Phase 3B evidence; no new regression suite, forward pass or GPU probe is claimed. The accompanying [design verification receipt](design_verification.json) records the lightweight checks and source/configuration identities. It does not certify an implementation that does not yet exist.

## 19. Decision table and student approval gate

| Category | Recommended interpretation / alternative | Status and reason |
|---|---|---|
| Agreed principles | Same checkpoint, same baseline numerical path, only classes 0/3, two ROIs/three calls, no GT/fallback/unanchored additions, protect strong global rows | Explicit scope of the student's design request; execution still prohibited |
| Requested numeric defaults | [.05,.25), .6/.4 score weights, 64/32 priority references, .75 crop, V1 IoU .50/local .25/+ .10 | Engineering hypotheses with qualitative evidence motivation; no optimality claim |
| Confidence interpretation | Q rises linearly with confidence; Z falls linearly to zero at its size reference | Recommended precise meaning; reject uncertainty inversion that rewards weak artifacts |
| Geometry edge cases | Outward integer bounds, 2-global-pixel envelope margin, minimally enlarge .75 crop, no-zoom rejection, 2-local-pixel artificial-edge veto | New simple safeguards; numerical margins need approval and future tests |
| ROI competition | One feasible class head each before pooled refill; coalesce only at IoU .70 plus expanded-envelope containment | Diversity/efficiency hypotheses, no guaranteed anatomical coverage |
| Fusion scope and ordering | All covered eligible anchors, local-only NMS .70, one shared IoU-priority one-to-one matching, one boost, no final global NMS | Requires approval; preserves global membership and isolates V2's geometry factor |
| V2 correction | Recommend IoU>=.50, local>=.50 and .25 blend with collision veto | **Approve or revise before implementation**; .35–.50 replacement is disabled, not a third variant |
| Alternatives not selected | Full local union, global NMS after boosting, weighted score voting, independent local additions, image-only fallback, a third ROI | Conflict with protection/scope or add unsupported complexity/FP risk |
| Global shape/loader interface | Preserve dimension-based reference profile; inject a verified checkpoint; isolate evaluation from adaptive core | Necessary implementation work, not a new preprocessing policy or permission to edit trainer |
| Decision thresholds | 10% FP point tolerance, .005 uncertainty tolerance, 2x mean latency and the stated AP/variant selection gates | Proposed research decision rules; must be agreed before outcomes, not clinically validated |
| Separate training study | Candidate A/B/C remain exactly proposed; class-aware training is not authorized here | No validation-derived change to training weights/crops/caps |

The recommended specification is internally complete for independent implementation if accepted. The open decisions are approval of its new safeguards/fusion scope, the V2 correction, and the planned success/timing/selection rules; they are not invitations for an implementer to improvise. If the student chooses different settings, revise/version the proposal **before** Phase 3C.2 inference. Current recommendation: review this conservative anchored design, acknowledge its limited recovery ceiling, and authorize only a bounded implementation/pilot after these choices are settled. No adaptive AP, latency benefit or localization improvement is claimed.
