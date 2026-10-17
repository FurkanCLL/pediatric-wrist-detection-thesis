"""Finite, lazy training iteration with explicit epoch/position/source receipts."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from ultralytics.data.build import seed_worker


def label_signature(values):
    values = np.asarray(values, dtype="<f4").reshape(-1, 5)
    return hashlib.sha256(values.tobytes()).hexdigest()


def native_label_signature(label):
    return label_signature(np.concatenate([label["cls"], label["bboxes"]], axis=1))


class IndexedDataset(Dataset):
    def __init__(self, dataset, files, label_signatures):
        self.dataset = dataset
        self.files = list(map(str, files))
        self.label_signatures = list(label_signatures)
        if [str(Path(f).resolve()) for f in dataset.im_files] != self.files:
            raise ValueError("Native image order differs from frozen sampler order")
        if len(dataset.labels) != len(files) or len(label_signatures) != len(files):
            raise ValueError("Image/annotation population is inconsistent")
        for file, label, signature in zip(self.files, dataset.labels, self.label_signatures):
            if str(Path(label["im_file"]).resolve()) != file or native_label_signature(label) != signature:
                raise ValueError("Native cached annotations differ from prepared labels")

    def __len__(self):
        return len(self.files)

    def __getattr__(self, name):
        # The guard also prevents recursive lookup while Windows unpickles this wrapper.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, request):
        epoch, position, index, repeated = request
        if not 0 <= index < len(self):
            raise ValueError("Source request escaped the training population")
        signature = native_label_signature(self.dataset.labels[index])
        if signature != self.label_signatures[index]:
            raise ValueError("A worker's cached source annotations changed")
        sample = self.dataset[index]
        if str(Path(sample["im_file"]).resolve()) != self.files[index]:
            raise ValueError("Native transform returned a different source image")
        # Hash repeated transformed tensors without consuming any augmentation RNG.
        augmented = hashlib.sha256(sample["img"].contiguous().numpy().tobytes()).hexdigest() if repeated else None
        worker = get_worker_info()
        sample["rarity_receipt"] = (epoch, position, index, worker.id if worker else -1,
                                     torch.initial_seed(), signature, augmented)
        return sample

    @staticmethod
    def collate_fn(samples):
        from ultralytics.data.dataset import YOLODataset
        receipts = [sample.pop("rarity_receipt") for sample in samples]
        result = YOLODataset.collate_fn(samples)
        result["rarity_receipt"] = receipts
        return result


class EpochSampler(Sampler):
    def __init__(self, size):
        self.size = size
        self.epoch = None
        self.draws = None

    def set_plan(self, epoch, draws):
        draws = np.asarray(draws)
        if not isinstance(epoch, int) or not 0 <= epoch < 80 or draws.shape != (self.size,) or draws.dtype.kind not in "iu" or np.any(draws < 0) or np.any(draws >= self.size):
            raise ValueError("Invalid epoch source-index plan")
        self.epoch, self.draws = epoch, draws.copy()

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.epoch is None:
            raise RuntimeError("Select an epoch plan before consuming training data")
        repeated = np.bincount(self.draws, minlength=self.size) > 1
        for position, index in enumerate(self.draws):
            yield self.epoch, position, int(index), bool(repeated[index])


class FiniteEpochLoader(DataLoader):
    def __init__(self, dataset, batch_size=6, workers=2, pin_memory=True, seed=42):
        self.epoch_sampler = EpochSampler(len(dataset))
        self.active_iterator = None
        self.epoch_seed = seed
        generator = torch.Generator()
        super().__init__(dataset, batch_size=batch_size, sampler=self.epoch_sampler, num_workers=workers,
                         collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, generator=generator,
                         pin_memory=pin_memory, prefetch_factor=4 if workers else None,
                         persistent_workers=False, drop_last=False)

    def select_epoch(self, epoch, draws):
        if self.active_iterator is not None:
            raise RuntimeError("Previous training iterator is still active")
        self.epoch_sampler.set_plan(epoch, draws)
        # Epoch zero uses the native single-GPU generator seed; later workers restart deterministically.
        self.generator.manual_seed((6148914691236517205+self.epoch_seed-1+epoch) % (1 << 64))
        probe = torch.Generator()
        probe.set_state(self.generator.get_state())
        self.worker_base_seed = int(torch.empty((), dtype=torch.int64).random_(generator=probe).item())

    def __iter__(self):
        # Ultralytics creates a discarded enumerate(loader) before the visible progress iterator.
        # Deferring DataLoader.__iter__ prevents that object from spawning workers or advancing RNG.
        if self.active_iterator is not None:
            raise RuntimeError("Concurrent training iterators are unsupported")
        if self.epoch_sampler.epoch is None:
            raise RuntimeError("Select an epoch plan before consuming training data")
        iterator = super().__iter__()
        self.active_iterator = iterator
        try:
            yield from iterator
        finally:
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            self.active_iterator = None

    def close(self):
        if self.active_iterator is not None and hasattr(self.active_iterator, "_shutdown_workers"):
            self.active_iterator._shutdown_workers()
