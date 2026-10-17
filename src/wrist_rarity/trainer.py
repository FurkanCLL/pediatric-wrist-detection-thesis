"""Keep frozen optimization/checkpoint hooks and replace only training data iteration."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from wrist_train.official import OfficialTrainer
from wrist_train.launch import write_json
from wrist_train.progress import append_json
from .data import IndexedDataset, FiniteEpochLoader
from .policy import RarityPlans, plan_hash
from .protocol import source_files_and_labels


class EpochAudit:
    def __init__(self, epoch, plan, files, signatures, directory, worker_base_seed):
        self.epoch, self.plan = epoch, np.asarray(plan)
        self.files, self.signatures = files, signatures
        self.cursor, self.batches = 0, 0
        self.workers, self.augmented = set(), {}
        self.worker_base_seed = worker_base_seed
        self.path = directory/f"epoch{epoch+1:03}_consumed.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    def consume(self, receipts, files):
        if len(receipts) != len(files):
            raise ValueError("Consumed batch receipt count differs")
        for receipt, file in zip(receipts, files):
            epoch, position, index, worker, seed, signature, augmented = receipt
            if epoch != self.epoch or position != self.cursor or self.cursor >= len(self.plan) or index != self.plan[self.cursor]:
                raise ValueError("Consumed source index/epoch/order differs from intended plan")
            if str(Path(file).resolve()) != self.files[index] or signature != self.signatures[index]:
                raise ValueError("Consumed image/annotation alignment failed")
            if worker not in (0, 1) or seed != self.worker_base_seed+worker:
                raise ValueError("Unexpected worker identity or RNG initialization")
            self.workers.add(worker)
            if augmented is not None:
                self.augmented.setdefault(index, []).append(augmented)
            self.cursor += 1
        self.batches += 1
        append_json(self.path, {"batch": self.batches, "receipts": receipts})

    def finish(self):
        repeats = np.flatnonzero(np.bincount(self.plan, minlength=len(self.files)) > 1)
        if self.cursor != len(self.plan) or self.workers != {0, 1} or set(self.augmented) != set(repeats):
            raise ValueError("Incomplete consumed epoch, worker coverage or repeat audit")
        if any(len(self.augmented[i]) != 2 for i in repeats):
            raise ValueError("Repeated-image audit does not contain both actual draws")
        different = sum(self.augmented[i][0] != self.augmented[i][1] for i in repeats)
        if not different:
            raise ValueError("No repeated source received a different augmented tensor")
        return {"epoch": self.epoch+1, "draws": self.cursor, "batches": self.batches,
                "plan_sha256": plan_hash(self.plan), "consumed_indices_exact_match": True,
                "source_labels_checked_each_draw": True, "workers": sorted(self.workers),
                "worker_seeds": [self.worker_base_seed, self.worker_base_seed+1],
                "repeated_sources": len(repeats), "different_augmented_pairs": different,
                "identical_augmented_pairs": len(repeats)-different,
                "consumption_log_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}


class RarityTrainer(OfficialTrainer):
    def __init__(self, protocol, identity, directory, overrides, checkpoint=None):
        self.rarity_plans = RarityPlans()
        self.source_files, self.source_labels = source_files_and_labels(self.rarity_plans.rows)
        super().__init__(protocol, identity, directory, overrides, checkpoint)
        if checkpoint:
            self.identity["rarity_best_epoch_evidence"] = checkpoint["official"]["rarity_best_epoch_evidence"]

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            # Native validation dataset, sampler, padding, workers and iterator are unchanged.
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1 or batch_size != 6 or self.args.workers != 2:
            raise ValueError("Rarity training supports only the approved single-GPU batch/worker setup")
        native = self.build_dataset(dataset_path, mode, batch_size)
        dataset = IndexedDataset(native, self.source_files, self.source_labels)
        return FiniteEpochLoader(dataset, batch_size=6, workers=2, pin_memory=True, seed=self.args.seed)

    def _start_training(self, trainer):
        super()._start_training(trainer)
        if not isinstance(self.train_loader, FiniteEpochLoader) or self.train_loader.num_workers != 2:
            raise ValueError("Finite two-worker training loader is missing")
        self._event({"event": "rarity_loader_ready", "finite": True, "workers": 2,
                     "validation_loader": type(self.test_loader).__name__, "initial_scale": self.scaler.get_scale()})

    def _begin_epoch(self, trainer):
        super()._begin_epoch(trainer)
        draws = self.rarity_plans.plan(self.args.seed, self.epoch)
        self.train_loader.select_epoch(self.epoch, draws)
        audit_directory = self.directory/"sampling"
        audit_directory.mkdir(exist_ok=True)
        np.savez_compressed(audit_directory/f"epoch{self.epoch+1:03}_plan.npz", indices=draws)
        self.epoch_audit = EpochAudit(self.epoch, draws, self.source_files, self.source_labels,
                                    audit_directory, self.train_loader.worker_base_seed)
        self.epoch_accounting_start = self.monitor.state_dict()
        self.epoch_scale_start = self.scaler.get_scale()
        self.accumulation_seen = set()
        self._event({"event": "rarity_epoch_plan", **self.rarity_plans.validate(draws)})

    def preprocess_batch(self, batch):
        # Reconcile actual delivered batches before moving data to CUDA or using it in a loss.
        receipts = batch.pop("rarity_receipt")
        self.epoch_audit.consume(receipts, batch["im_file"])
        self.accumulation_seen.add(int(self.accumulate))
        return super().preprocess_batch(batch)

    def _end_batch(self, trainer):
        super()._end_batch(trainer)
        if self.train_batches % 100 == 0:
            print(f"Rarity epoch {self.epoch+1}/{self.epochs}: {self.epoch_audit.cursor}/14061 draws; "
                  f"updates {self.monitor.successful}; skips {self.monitor.skipped}", flush=True)

    def _end_epoch(self, trainer):
        super()._end_epoch(trainer)
        receipt = self.epoch_audit.finish()
        receipt.update(self.rarity_plans.validate(self.epoch_audit.plan))
        receipt.update({"optimizer_epoch": {k: getattr(self.monitor, k)-self.epoch_accounting_start[k]
                                           for k in ("attempts", "successful", "skipped")},
                        "scale_start": self.epoch_scale_start, "scale_end": self.scaler.get_scale(),
                        "accumulation_values": sorted(self.accumulation_seen), "training_seconds": self.training_seconds,
                        "training_memory": self.training_peak, "finite_live_and_ema": True})
        self.identity["rarity_completed_epoch"] = receipt
        write_json(self.directory/"sampling"/f"epoch{self.epoch+1:03}_verified.json", receipt)

    def train(self):
        # Also shut down workers on a numerical or receipt failure, without an automatic retry.
        try:
            return super().train()
        finally:
            loader = getattr(self, "train_loader", None)
            if isinstance(loader, FiniteEpochLoader):
                loader.close()

    def save_model(self):
        # Recovery embeds the best epoch's receipt as well as the current completed plan.
        if self.fitness >= self._best_seen_score:
            self.identity["rarity_best_epoch_evidence"] = self.identity["rarity_completed_epoch"]
        return super().save_model()
