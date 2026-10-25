# CG-ALR V3.0 — frozen design and scientific protocol

25 October 2026. The direct Phase 5B request authorizes one algorithm, both existing checkpoints, one fixed technical pilot and one full validation run per checkpoint. This document is written before the final runner and before any V3 validation performance calculation. Numerical configuration: `configs/cg_alr_v3.json`. The freeze receipt will bind this design, implementation, tests and configuration before GPU execution. The requested `phase_5b_v3/V3_DESIGN.md` is an identical copy checked by the freeze command.

## Evidence and research question

[Phase 5A](../phase_5a_analysis/CG_ALR_V2_DIAGNOSTICS.md) found C/F recoveries of 38/34 with 171/136 net extra unmatched predictions, approximately 70% of local calls without an output change, protected matches consuming 76–77% of accepted local evidence, and no operating recovery attributable to geometry. Local score and IoU alone showed weak, inconsistent separation. Eligible adequate anchors were already covered under that retrospective definition. Therefore ROI reordering is not claimed to repair missing anchors.

Hypotheses: (H1) protected ownership and explicit ambiguity rejection reduce harmful promotions relative to V2; (H2) moderated confirmation can improve the AP/FP trade-off while preserving fracture AP; (H3) skipping regions with no V3-eligible candidate reduces calls without changing the policy's available updates; (H4) any supported gain persists under stable source-ID tie handling. All may fail. No numerical improvement is predicted.

## 1. Definitions and inherited numerical policy

An original global row is `g_i=(box_i,p_i,class_i)` with immutable ID `i` in the verified standard-inference output. N≤300. Reuse 768 input, class mapping, prepared grayscale replication, resize/padding/restoration, FP32 unfused EMA with CUDA float16 input/autocast, per-view confidence .001 and NMS .70. Only fracture (0) and periosteal reaction (3) are mutable. Original confidence≥.25 rows are protected; all other noncandidate rows are also unchanged.

Reuse V2 candidate bounds .05≤p<.25 and minimum effective short side 1 at ideal 768 scale. Q/Z references (64/32), .6Q+.4Z priority, class-head diversity, outward anchor envelope, .75 dimension crop, 2-local-view limit, containment and .70 ROI coalescing remain unchanged. No image-only proposal, training, learned calibrator or additional model is introduced.

## 2. Candidate evaluation and utility-aware ROI planning

Before local inference, mark a weak candidate `protected_conflict` if an ORIGINAL same-class protected global row overlaps it at IoU≥.50. Such a candidate cannot be updated by V3, so it must not request a local view. No outcome, ground truth, patient linkage or saved correctness label enters this step.

Run the original V2 ranking/diversity/coalescing procedure on the remaining candidate records, retaining their original IDs and original crop geometry. Rank by descending S, descending p, ascending short side, ascending original ID. Class heads are processed before pooled remaining candidates. Coalescing requires ROI IoU≥.70 AND expanded-envelope containment. Budget exhaustion never triggers an extra view. If no candidate remains, use the global result alone. Do not remove low-score candidates merely because they cannot cross .25: their AP ranking can still matter.

This is a structural utility filter, not a learned estimate of useful inference. Protected overlap does not prove a clinical duplicate; close legitimate findings can be excluded. Fewer calls and unchanged quality are hypotheses to measure, not guarantees relative to V2.

## 3. Local evidence processing and ownership graph

Reuse unchanged V2 local evidence preparation: target classes 0/3, confidence≥.25, two resized-content-pixel artificial-edge rejection, original-coordinate shift exactly once, and same-class union NMS at IoU>.70. Preserve `(ROI ID, per-view local row ID)` for every retained row, and record rejection reasons.

For each retained local row e, find all same-class protected global rows with IoU≥.50. If any exist, reserve e: it is unavailable to every mutable candidate. Reservation is not an assignment validating every protected row, and never changes protected output. This prevents a weak row with slightly higher IoU from stealing that evidence. The evidence may appropriately confirm a reliable existing object; that is not wasted correctness.

Build a bipartite graph between remaining mutable candidates and unreserved retained local evidence. Add an edge iff class agrees, IoU≥.50 and the candidate's expanded envelope fits the evidence's source ROI. Accept an edge ONLY when both endpoint degrees equal one in this complete eligible graph. Thus accepted edges form a unique one-to-one matching without choosing among ambiguous alternatives. Log zero evidence, reserved evidence, missing containment and ambiguous global/local degree explicitly. A global with two plausible local rows is rejected rather than using score to decide ownership.

A maximum-cardinality/Hungarian assignment is deliberately not used: maximizing assignments or a confidence/IoU cost does not resolve semantic ambiguity, and the study has no validated cost or margin. The isolated-edge constraint has a directly auditable uniqueness meaning and avoids additional fitted weights. It can reject useful corroboration in crowded/overlapping findings; this is a major declared failure scenario.

## 4. Selective score updating and protected output

Process accepted pairs by descending ORIGINAL global confidence, then original ID. Scores are rankings, not calibrated probabilities. For a candidate with local score q:

`p_proposed = min(p + .10, (p + q)/2, .35)`.

The equal two-view midpoint is a symmetric arithmetic moderation of two correlated scores, not a posterior or independence calculation. Half is fixed because there are two views; it is not fitted to validation outcomes. .10/.35 are inherited V2 upper bounds. Store the greatest float32 not exceeding the mathematical cap. q≥.25>p guarantees no score reduction.

Before updating, reject if the candidate's ORIGINAL box overlaps any CURRENT operating-confidence (≥.25) same-class row at IoU≥.50. This includes newly promoted earlier rows and prevents a later weak candidate from activating an overlapping operating row. Original protected conflicts were already excluded before ROI selection. This safeguard controls observable overlap, not annotation matching; it does not prove that all prior TPs retain their matcher ownership or that remaining promoted rows are correct.

All coordinates stay exactly equal to original global coordinates. Phase 5A showed mixed rare coordinate changes and no .25/.50 recovery effect; freezing geometry avoids attributing the main innovation to another blend heuristic. Class IDs and row count never change; no new rows, deletions, final NMS or truncation are allowed. No unmatched local row becomes a detection. Final order is descending updated confidence then original global ID, with exact original rows/order when no score changes. Output at confidence .50 is exactly unchanged.

## 5. Constants and rationale

| Constant | Value | Reason |
| --- | --- | --- |
| Candidate/local/ROI policies | Original V2 | Keep established view geometry and numerical path |
| Protected and operating confidence | .25 | Existing reporting/protection boundary |
| Ownership/duplicate overlap | IoU≥.50 | Same inherited matching and operating-overlap convention; new ownership role explicitly frozen |
| Midpoint local weight | .50 | Equal moderation of exactly two scores, not calibrated evidence or optimized weight |
| Maximum increment/final score | .10/.35 | Original V2 bounds |
| Geometry | Original global boxes | Conservative choice supported by mixed low-footprint V2 geometry |
| Calls | 1 global, at most 2 local | Original physical limit |

These are engineering conventions, not clinically validated thresholds. The midpoint/ownership combination is a hypothesis and may sacrifice recall/AP. No search or checkpoint-specific constants are permitted.

## 6. Determinism, complexity and implementation boundary

Pure core functions accept global predictions, local view records, image dimensions and fixed settings; no dataset rows, annotations, patient IDs or metrics. Reuse frozen checked_rows, IoU/containment, crop planning records, local evidence/coordinate restoration and numerical checks. Diagnostics bind decisions to immutable IDs.

For N globals and L≤600 pre-union local rows: protected/candidate comparisons O(N²), evidence filtering/union worst O(L²), graph construction O(NL), degree checks O(N+L+edges), deterministic sorting O(N log N), and unchanged two-ROI bounded planning. Rejecting ambiguity avoids cubic assignment. No GPU tensor is allocated by planning/fusion.

## 7. Frozen execution sequence and timing

Commit/push design, config, implementation, synthetic tests and freeze receipt BEFORE any V3 performance computation. Both standard epoch-22 and rarity epoch-26 are planned now; the second run is independent of the first result. Verify checkpoint hashes and all source/configuration/data identities. Historical A–F, Phase 5A, checkpoints and prepared files are immutable.

For each checkpoint: existing fixed 24-image pilot, no pilot AP; three predetermined first-image replays for bitwise determinism, recorded separately. Three synthetic warmups are excluded from population timing but counted. After both technical gates pass, one full 3,088-image run per checkpoint. Every required global array must reproduce saved A/D bit-for-bit before local calls. Pilot/full predictions must agree for all pilot images. Stop and preserve failure evidence for a technical defect; repairs may not use performance results.

Complete-path timing starts before decode and ends after planning, all views, fusion, numerical checks and CUDA synchronization; export, model loading/warmups, GT/AP and offline replays are excluded. The paired standard reference is decode plus the already-required global call through restored/validated rows, measured before local planning; it requires no additional forward. Actual calls are summed from verified per-view counters. Record mean/median/p95, paired cost, peak allocated/reserved, warmups/replays and total loop boundaries. Pilot memory ceiling is the historical 4 GiB allocated and projected full path ≤10,800 s. Do not rerun latency merely to improve a value.

## 8. Two metric tracks and patient uncertainty

Historical track: unchanged pinned native TP matcher and AP evaluator on each method's original final order, directly comparable to published A/C/D/F. Preserve all old scores/results; reuse checked old bootstrap replicates.

Deterministic sensitivity track: restore every method to original global-row-ID order BEFORE the unchanged per-image native TP matcher. For ranked AP, lexicographically order by descending unmodified score, frozen sorted validation image index, original global row ID and bootstrap image occurrence. Use the same pinned `compute_ap` integration (101-point interpolation) on cumulative TP/FP curves at ten IoUs. This avoids modifying scores and avoids the pinned unstable confidence-sort. Zero-prediction present classes receive zero AP; absent target classes are undefined. The bootstrap occurrence only separates replicated identical source identities; identical rows/TP flags tie consistently.

Apply both tracks consistently to A/C/V3-standard and D/F/V3-rarity. Native matching input differences and global confidence ranking are both explicit in the sensitivity convention. It is not a replacement for historical A–F. One class's AP cannot change solely because another class's final rows were sorted differently in this track. Preserve immutable source-ID arrays in private outputs.

Use the existing exact 500 paired patient samples (seed 314159), all views/examinations with multiplicity, common across all methods/tracks. Reuse historical baseline/V2 replicates after hash/membership checks; calculate new V3 replicates and the required new deterministic-track replicates. Require all eight supported classes for overall mAP: expect 492 valid overall/bone-lesion replicates and 500 for others. Report conditional paired percentile 95% intervals. Do not replace missing-class AP with zero.

## 9. Success, attribution and stopping

For V3 versus corresponding standard: positive overall mAP50–95 point and paired lower95%>0, nondecreasing mAP50, fracture AP50/AP75/AP50–95 point preservation with historical lower-interval tolerance −.005, overall/fracture FP ratios≤1.10 at .25/.50, mean paired full-path latency≤2×, maximum three physical calls. Apply historical gates unchanged and report them individually. To call the detector improvement supported, also require deterministic-track AP/fracture corroboration under the same rules; this additional predeclared safeguard prevents a tie-only success. Report V3 versus V2 separately, without choosing the best variant/checkpoint or relaxing any gate.

Predefined CPU attribution, after both full runs, uses four fixed cells with global coordinates unchanged: (1) original V1 score/greedy rule on historical V2 views; (2) that same frozen rule on saved V3 views; (3) V3 arbitration on saved V3 views with original V2 score cap instead of midpoint; (4) actual V3. Cells 1–2 attribute view availability, 2–3 arbitration, 3–4 midpoint moderation. The source global candidate list is the original full V2 list for cells 1–2; actual V3 eligibility applies for 3–4. These are explanatory replays, no extra GPU inference, not competing deployment variants. Official frozen V2 geometry remains a separate comparator. Do not select or alter a rule from these results.

Primary checkpoint results are reported separately from rarity robustness regardless of their sign. The global SUCCESS label requires strict success on both checkpoints. PARTIAL IMPROVEMENT means at least one strict checkpoint success, or lower FP than V2 with positive recovery over standard on both checkpoints while full gates fail; this describes an engineering trade-off, not a qualified overall AP improvement. Otherwise report NO DEMONSTRATED IMPROVEMENT. A claim of improvement across checkpoints requires both. Remaining limits include unanchored misses, confident correlated errors, legitimate nearby findings rejected as conflicts, sparse rare classes, one training seed, repeated validation exposure, contextual-crop limitations and unmatched annotations without clinical adjudication. This remains exploratory even if intervals exclude zero.

## 10. Relevant primary literature and contribution boundary

- [Solovyev et al., Weighted Boxes Fusion](https://arxiv.org/html/1910.13302v3): aggregates overlapping boxes and scores. It motivates reviewing aggregation, not transferring dataset gains. V3 keeps original boxes and uses a guarded two-score midpoint, not WBF box fusion.
- [Carion et al., DETR](https://arxiv.org/html/2005.12872v3): uses one-to-one bipartite matching between predictions and GT during training. V3 uses an inference-only graph between two prediction sets and rejects ambiguity; it neither trains DETR nor uses GT assignment in inference.
- [Küppers et al., Multivariate Confidence Calibration](https://arxiv.org/abs/2004.13546): examines detector confidence with location/scale information. V3 does not treat scores as calibrated probabilities or fit a calibrator.
- [Najibi et al., AutoFocus](https://arxiv.org/abs/1812.01600): selective coarse-to-fine inference uses learned focus regions. V3's small structural prefilter adds no learned proposal mechanism and makes no claim of AutoFocus-like performance.
- [Akyon et al., SAHI](https://arxiv.org/html/2202.06934v5): sliced inference and merging are established context. The project preserves SAHI as its historical comparator rather than introducing exhaustive slicing into V3.

The project-specific contribution is this frozen combination of protected reservation, isolated-edge ownership, moderated bounded score transfer, sequential operating-conflict protection and structural ROI eligibility, with source-ID metric sensitivity and measured cost. Matching, averaging and adaptive inference are established ideas. No novelty or performance claim is inferred from unrelated datasets. Literature accessed 25 October 2026.
