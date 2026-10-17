# Minimal later training integration

Read-only inspection of the frozen project and installed Ultralytics 8.4.177. This is an engineering proposal, not production integration or a training authorization. The official trainer and framework remain unchanged.

## Preserve the dataset and replace only its training index sequence

Use the existing training-only `YOLODataset` with its original `im_files`, aligned cached labels and full-image transforms. An isolated epoch sampler should yield source indices, with repeats represented by repeated indices. Do not rewrite `im_files` or labels to manufacture a duplicated dataset: that risks incorrect image/label/cache alignment. Keep the dataset length and sampler length exactly 14,061, physical batch six, drop-last false and 2,344 batches per epoch. Full-image training with the same augmentation policy preserves every source box before native augmentation; no targeted crop is proposed.

The plan must be created by a deterministic function of the fixed training population, policy identity, mining checkpoint/prediction/feature identities, seed and epoch. Verify its image and patient caps, 95% distinct-image coverage in each fracture stratum, complete patient/lesion-cohort coverage and exact budget before the first batch. Log the plan's hash and aggregate exposure; keep source IDs/indices private. The loader must map each planned index to its existing label record and log actual consumed indices/counts so prefetched but unconsumed batches cannot masquerade as delivered exposure.

## The persistent infinite loader is an important boundary

`DetectionTrainer.get_dataloader` in `ultralytics/models/yolo/detect/train.py` constructs the dataset and calls `build_dataloader`. `ultralytics/data/build.py` returns `InfiniteDataLoader`, wrapping the batch sampler in `_RepeatSampler`. It keeps one iterator across epochs and prefetches four batches per worker for shuffled training. Merely calling `sampler.set_epoch` at the next callback can therefore leave already-prefetched next-cycle batches from an old plan. In the single-GPU trainer loop, native `sampler.set_epoch(epoch)` is called only when `RANK != -1`, so a new single-GPU sampler needs an explicit epoch hook.

A minimal safe later implementation is a separate training-condition subclass/launcher that overrides only the train loader construction, using a **finite** PyTorch DataLoader and an explicit pre-iterator epoch plan. Leave validation construction and the original standard launcher unchanged. Alternatively prove an explicit infinite-loader reset/drain scheme; do not assume it works because one epoch has the correct length. Finite iteration is simpler, but changes worker/iterator lifecycle from the original loader and must be disclosed and checked before approval. This phase does not implement either path against real training data.

## Augmentation and RNG

Keep seed 42, two workers, the pinned `seed_worker` Python/NumPy seeding, a documented generator seed/epoch scheme and unchanged native transforms. Mosaic, MixUp, CopyPaste and other multi-image augmentation are already disabled. The normal rotation, translation, scale and value augmentation still applies to each draw; two source-index repeats must receive separately sampled views. Do not convert a repeat into a cached transformed image. Nominal source-box exposure simulations do not describe final post-affine/filtering box counts.

Plan reproducibility and bitwise augmented-tensor reproduction are different claims. Worker scheduling, iterator lifecycle, prefetch and RNG consumption change when source order changes. A finite loader may restart workers each epoch unless persistent workers are used; either policy needs a logged, tested RNG contract. The original baseline does not promise bitwise worker-RNG restoration on resume. A new condition should restore the epoch number/plan and explicitly record the same or stricter limitation, not claim exact resumed augmentation replay without evidence.

Before production approval, test synthetic index/label pairing through at least two worker epochs, exact consumed plan length/order, changed/repeated indices, source-pixel immutability, augmentation variation and deterministic repeated launches. Then run a separately authorized bounded training smoke with actual loss/gradient/scaler/update checks. Neither such a smoke nor official fitting is authorized in Phase 4A.

## Compute and provenance

With unchanged length/batch/schedule, the existing `wrist_train.budget.optimizer_budget` gives the same nominal 47,433 optimizer attempts over 80 epochs, including warmup accumulation ramp. Both candidates have 1,124,880 nominal image draws and steady four-step accumulation/effective batch 24. Actual successful/skipped AMP updates may differ with data and must be logged; equal nominal attempts are not a promise of equal finite updates or wall time.

A later condition needs its own output directory, run manifest and checked checkpoint provenance containing the original common protocol hash, new subclass/loader source hashes, training-index ordering, policy hash, mining checkpoint/epoch, prediction/feature identities, seed/epoch plans and actual exposure. Start from the same verified COCO initialization, never the trained mining checkpoint. Preserve the original protocol lock and baseline artifacts; the new condition's implementation manifest is additive. Sampler scoring is computed once from the preserved baseline, not refreshed from validation or during training.

The report will estimate run cost from the measured completed 80-epoch baseline rather than predicting AP or claiming inference timing equals training time. Any final comparison redesign, rarity-only ablation, model fitting or held-out evaluation remains a separate student decision.
