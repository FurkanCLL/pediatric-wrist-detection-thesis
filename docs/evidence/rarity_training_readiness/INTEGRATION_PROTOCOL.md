# Approved rarity-only integration and pre-GPU verification

The student's Phase 4B request approves the Phase 4A full-image patient-aware **rarity-only** condition. It retains two trained checkpoints, each compared with Standard / SAHI Context / CG-ALR V2. The earlier negative SAHI and adaptive findings remain evidence; retaining a comparator does not imply that it improves accuracy. Formal supervisor scope alignment remains pending.

Official 80-epoch training is **manual only**. This phase permits an isolated two-epoch real-data diagnostic after automated gates pass. No held-out test access or D/E/F evaluation is authorized. The diagnostic's validation is part of infrastructure verification and is not a thesis performance result.

## Reuse the approved selection exactly

`wrist_rarity.policy.RarityPlans` invokes the unchanged `MatchedPopulation.paired` and returns its first, rarity-only plan. The required difficulty array is a zero placeholder; neither mined features, predictions, source uncertainty nor a trained teacher is loaded. The unused paired result cannot influence the selected plan. The exact Candidate-B rarity formula, protected population, coverage restoration, weighted ticket allocation, random streams and final permutation remain unchanged.

The planned verification compares all 80 epochs at seeds 42, 1042 and 2042 with the hash-verified saved Phase 4A rarity plans. Runtime uses seed 42 and zero-based epochs 0–79. Every generated plan must retain 14,061 draws, exactly 625 replacements, original 9,307/4,754 fracture-status draw totals, all 4,264 patients, >=95% distinct images separately in each status, at most two draws/image, patient extras min(3,ceil(.5*N_p)), and all 66 views of the 16 lesion patients exactly once. No crop, label change, new weight or scientific coefficient is added. No-fracture-box does not mean clinically normal.

## Finite training loader and actual consumption

`RarityTrainer` inherits the frozen `OfficialTrainer` optimization, precision, validation and recovery hooks. It replaces only the training loader and adds consumption/provenance checks. Validation delegates directly to the original loader implementation. The standard trainer, common JSON/lock, previous source and evidence stay untouched.

Each epoch starts with a checked source-index plan before the native loop iterates. A finite PyTorch DataLoader starts two fresh workers, uses the native `seed_worker`, prefetches four batches per worker, and consumes exactly 2,344 batches (2,343 x six plus three). Worker seeds are deterministic functions of seed and epoch. Concurrent/unfinished iterators cannot be replaced by the next plan.

The pinned Ultralytics loop constructs `enumerate(train_loader)` twice when preparing its progress bar. A normal eager finite DataLoader would start workers for the discarded iterator too. The new loader's generator defers worker creation until actual consumption. The unused iterator therefore consumes no RNG and prefetches nothing. No native training-loop copy or monkey patch is introduced.

The wrapper retains the native YOLODataset, cached label order, original transforms and collation. It verifies every native cached label against its prepared-file signature and every source path against the frozen index order. Each request carries epoch, draw position and source index. Each delivered batch is checked against the intended plan before CUDA preprocessing/loss use and recorded privately. A stale epoch, wrong label/path/order/worker seed, incomplete epoch or excessive repetition fails rather than being silently repaired.

Both transformed occurrences of every repeated image are hashed without consuming RNG. This checks actual augmented tensors, separately from plan equality. Exact replay is tested with generated images, native YOLO transforms and two real workers across two epochs and two launches. A tensor collision would not by itself prove shared augmentation randomness; counts are reported rather than treated as a clinical or scientific threshold.

## Common learning conditions and additive identity

Official settings remain YOLOv8s/eight classes, verified original COCO `yolov8s.pt`, 768, 80 epochs, physical batch six, nbs24/steady accumulation four, AdamW, original cosine LR/warmup, AMP initial scale128 with the same growth/backoff/failure rules, identical augmentation, seed42, validation each epoch and latest-tie best eight-class mAP50-95. The original common protocol hash is preserved. The official run must never initialize from the trained baseline teacher.

`configs/rarity_training.json` and `condition_lock.json` bind the additive configuration, reused allocation sources and new implementation. Checkpoints carry the original common identity plus the rarity condition/source identity, diagnostic flag and completed consumption receipt. Source/config changes invalidate the readiness gates. Official output is `outputs/training/rarity_only`; the diagnostic has a separate named directory under `outputs/rarity_training_readiness/diagnostics/`. New runs require an absent directory, atomic checkpoints have verified predecessors, and a process lock excludes a second writer.

## Diagnostic and stopping gate

After the full tests, preservation checks and 240-plan comparison pass, the authorized diagnostic is exactly two complete training/validation epochs from COCO initialization with the real integrated loader, two workers, ordinary forward/backward/AdamW/AMP, checkpoint saving and reload checks. It has its own two-epoch cosine trajectory; it verifies infrastructure, not the 80-epoch LR trajectory or final convergence. Log actual consumption, augmentations, losses, attempts/successes/skips, scaler, peak memory, validation, checkpoints and clean completion.

The launcher has no automatic transition from smoke to official training. The readiness report will give separate preflight, manually confirmed start, status/monitor, recovery and post-training commands. Official configuration remains 80 even though the isolated diagnostic uses two.

## Reproducibility limits to retain

Fresh finite workers differ from the baseline's persistent infinite-worker lifecycle. Their epoch seeding is explicit and replayable, but individual augmented tensors and minibatch pairings do not match the baseline. The augmentation distribution/policy remains unchanged. Bitwise CUDA determinism and exact resumed optimization are not claimed.

Recovery is at a completed-epoch boundary into a new segment. Source plans and worker seeds replay by epoch, while the existing recovery implementation discards pending gradients and does not restore the native `last_opt_step` cadence. An interrupted epoch's partial work is not an official completed epoch. Keep every segment and report any interruption, rollback and actual update count. Best inference snapshots must not be used as resumable live optimization state. These limitations cannot be hidden by matching precomputed index lists.

The final decision will be READY, BLOCKED, or READY WITH DOCUMENTED LIMITATIONS based on completed evidence. Stop before the official model run.
