# CG-ALR V3 — implementation report

Implementation and synthetic verification completed on 26 October 2026, before any V3 GPU pilot or validation performance calculation. This is an additive exploratory follow-up; historical A–F implementations and results remain unchanged. The [design](V3_DESIGN.md) and `configs/cg_alr_v3.json` define one checkpoint-independent algorithm.

## Implementation

`src/wrist_v3/planning.py` reuses the frozen V2 candidate, crop, ranking, diversity and coalescing logic after excluding weak candidates already covered by original protected same-class detections. It preserves the complete original candidate list for the predefined component attribution.

`fusion.py` reserves local evidence compatible with protected detections, accepts only isolated one-to-one candidate/evidence graph edges, and moderates scores with the frozen midpoint, increment and absolute caps. Sequential overlap arbitration protects current operating detections. Every original box, class and row is retained; the output records immutable global IDs and explicit reservation, ambiguity, match and update decisions. `core.py` composes these stages without annotation inputs and enforces one global plus at most two local calls.

`runner.py` reuses verified checkpoint loading, preprocessing, local-view NMS and coordinate restoration. It checks every global output against saved A/D predictions before local inference. A model-level hook independently counts physical forwards. Fixed 24-image pilots include three predetermined replay images and no AP calculation. Both pilots must pass before either of the two full validation runs. New destinations are exclusive: an existing stage cannot be silently overwritten or repeated. Complete predictions, view bundles and timing logs remain private under `outputs/V3/`.

`metrics.py` preserves the historical pinned evaluator and separately implements the frozen deterministic source-ID sensitivity track. That track restores global-ID order before the unchanged native per-image matcher and orders AP ties by image/global identity and bootstrap occurrence, without modifying confidence. `analysis.py` checks historical metric reproduction, non-target invariance under deterministic ties, the same 500 patient-paired draws, absent-class handling, recovery and operating-point errors. Four predefined CPU replay cells attribute view availability, arbitration and midpoint effects without additional GPU calls or model selection.

`protocol.py` binds configuration, both design copies, code and tests in `freeze.json`. Execution requires a committed and pushed freeze. `publication.py` independently checks operating totals, saved prediction/bundle correspondence, all fusion replays, geometry, protected fields, call counts, paired intervals and historical preservation. Public exports contain aggregate evidence only. `reports.py` generates the measured results and scientific assessment after verification; it does not tune the algorithm.

## Pre-GPU verification

- All **268 tests pass** (41.547 seconds), including **29 new V3 tests**: 18 core, six metric and five protocol tests.
- Synthetic cases cover protected evidence ownership, competing weak candidates, ambiguous correspondence, sequential promotions, overlapping ROIs and restoration, empty/no-candidate/no-evidence paths, rare classes, numerical boundaries, output identity, deterministic ordering, exact three-call execution and hidden-call rejection.
- Metric tests cover pinned-evaluator agreement without ties, immutable-ID tie handling, repeated bootstrap occurrences, present classes without predictions and undefined absent classes.
- Protocol tests cover common checkpoint settings, strict verdict rules, missing-class contrasts and refusal to overwrite existing evidence.
- The private test log and current normalized source hashes are bound in [pre_gpu_tests.json](pre_gpu_tests.json). The source/configuration/design freeze is recorded in [freeze.json](freeze.json).

Before/after preservation compares the prior 386 tracked files, 784 private historical files, dataset file metadata identity and both best-checkpoint byte hashes. Historical non-best weights are protected by size and modification time. The pre-study inventory was captured before V3 execution; the completed receipt will report the after comparison. No training, new dependencies or detector changes are involved.

## Measurement boundaries

GPU timing uses sequential inference with three synthetic warmups. Complete image-path time includes decode, global and selected local views, planning and fusion; JSON serialization is outside the per-image interval and included separately in total loop time. The contemporaneous paired global prefix includes decode and the same global preprocessing/forward/NMS/restoration. Its timestamp precedes the global-reference assertion and V3 planning. Mean complete-path ratio is assessed against this prefix; session order and laptop GPU conditions remain limitations. Peak allocated and reserved CUDA memory and actual forwards are reported independently.

No real V3 AP, TP/FP, recovery or bootstrap result was available when this report and the protocol were frozen. Numerical rules cannot be revised in response to later outcomes. A technical failure must retain its evidence and be documented transparently. Final method selection, further optimization and held-out evaluation remain outside this phase.
