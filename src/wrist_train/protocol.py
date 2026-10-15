"""Official configuration identity and read-only readiness checks."""

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile
import time

from .common import ROOT, OUTPUT, verify_frozen_data
from wrist_prepare.common import CLASSES, digest

CONFIG = ROOT / "configs/training_official.json"
LOCK = ROOT / "docs/evidence/training_protocol/protocol_lock.json"
RUN_ROOT = ROOT / "outputs/training/standard_baseline"
STATE = ROOT / "outputs/training_protocol"
SOURCE_FILES = ["src/wrist_train/" + name for name in (
    "amp.py", "checkpoints.py", "progress.py", "protocol.py", "official.py", "launch.py",
    "common.py", "diagnostics.py", "budget.py")] + ["src/wrist_prepare/common.py", "src/wrist_prepare/records.py"]


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def source_hashes():
    # Normalize line endings so Windows Git checkout settings do not change code identity.
    return {name: digest((ROOT / name).read_text(encoding="utf-8").encode()) for name in SOURCE_FILES}


def framework_hashes():
    import ultralytics
    base = Path(ultralytics.__file__).parent
    files = ("cfg/default.yaml", "data/augment.py", "data/build.py", "engine/trainer.py",
             "engine/validator.py", "models/yolo/detect/train.py", "utils/metrics.py", "utils/torch_utils.py")
    return {name: digest((base / name).read_bytes()) for name in files}


def validate_protocol(protocol):
    args = protocol["training"]
    expected = {"epochs": 80, "imgsz": 768, "batch": 6, "nbs": 24, "seed": 42,
                "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
                "warmup_epochs": 1.0, "amp": True, "cos_lr": True, "patience": 0,
                "save_period": 10, "val": True, "split": "val", "cls_pw": 0.0,
                "degrees": 10.0, "translate": 0.05, "scale": 0.1, "hsv_v": 0.1}
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("Official training values differ from the student-approved protocol")
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud", "shear", "perspective", "hsv_h", "hsv_s", "bgr"):
        if args[key] != 0:
            raise ValueError("Unexpected augmentation: " + key)
    if args["augmentations"] != [] or args["cache"] is not False or args["workers"] != 2:
        raise ValueError("Unexpected loader or optional augmentation policy")
    if protocol["dataset"]["classes"] != list(CLASSES) or protocol["runtime"]["amp_initial_scale"] != 128:
        raise ValueError("Incorrect class mapping or initial AMP scale")
    policy = protocol["runtime"]
    if not (1 < policy["overflow_max_consecutive"] <= policy["overflow_max_in_window"] <= policy["overflow_window"]):
        raise ValueError("Invalid persistent-overflow thresholds")


def load_protocol():
    protocol = json.loads(CONFIG.read_text())
    validate_protocol(protocol)
    lock = json.loads(LOCK.read_text())
    if canonical_hash(protocol) != lock["protocol_sha256"] or source_hashes() != lock["implementation_source_sha256"]:
        raise ValueError("Frozen protocol or implementation changed; review is required")
    if lock["training_epochs"] != protocol["training"]["epochs"]:
        raise ValueError("Frozen identity and configured epoch budget disagree")
    if framework_hashes() != lock["framework_source_sha256"]:
        raise ValueError("Installed framework source changed; resume compatibility requires review")
    return protocol, lock


def verify_inputs(protocol, journal_sha):
    from ultralytics.data.utils import IMG_FORMATS
    records, frozen = verify_frozen_data()
    for key in ("split_manifest_sha256", "records_sha256", "policy_sha256"):
        if frozen[key] != protocol["dataset"][key]:
            raise ValueError("Dataset split/provenance identity changed")
    groups, stems = {}, set()
    for row in records:
        if row["filestem"] in stems or groups.setdefault(row["patient_id"], row["split"]) != row["split"]:
            raise ValueError("Duplicate source identity or patient split overlap")
        stems.add(row["filestem"])
    journal = ROOT / "outputs/dataset_preparation/completed.sqlite"
    if digest(journal.read_bytes()) != journal_sha:
        raise ValueError("Frozen preparation completion journal changed")
    with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as connection:
        hashes = {stem: (image, label) for stem, image, label in connection.execute("SELECT * FROM completed")}
    prepared = ROOT / "data/prepared/grazpedwri_detection"
    counts = Counter()
    for split in ("train", "val"):
        selected = [row for row in records if row["split"] == split]
        expected = {row["filestem"] for row in selected}
        files = list((prepared / "images" / split).iterdir())
        if any(path.is_dir() for path in files):
            raise ValueError("Unexpected nested directory in the frozen image split")
        actual = {path.name for path in files if path.suffix[1:].lower() in IMG_FORMATS}
        labels = {path.stem for path in (prepared / "labels" / split).iterdir() if path.suffix == ".txt"}
        if actual != {stem + ".png" for stem in expected} or labels != expected:
            raise ValueError("Prepared train/validation membership differs from the frozen manifest")
        # Recheck prepared bytes against existing conversion hashes; no raw decoding or test-file access.
        for row in selected:
            stem = row["filestem"]
            image = (prepared / "images" / split / (stem + ".png")).read_bytes()
            label = (prepared / "labels" / split / (stem + ".txt")).read_bytes()
            if (digest(image), digest(label)) != hashes[stem]:
                raise ValueError("Prepared image/label hash mismatch")
            counts[split] += 1
    if (counts["train"], counts["val"], len(groups)) != (14061, 3088, 6091):
        raise ValueError("Unexpected dataset counts")
    weights = OUTPUT / "weights/yolov8s.pt"
    if digest(weights.read_bytes()) != protocol["initialization"]["sha256"]:
        raise ValueError("The verified COCO initialization is missing or changed")
    return dict(counts)


def environment_check(protocol):
    import torch
    from .common import require_cuda
    device = require_cuda()
    expected = protocol["environment"]
    actual = {key: importlib.metadata.version(key) for key in expected["packages"]}
    if actual != expected["packages"] or platform.python_version() != expected["python"]:
        raise ValueError("Package/Python versions differ from the frozen environment")
    if torch.version.cuda != expected["cuda_build"] or torch.cuda.get_device_name(device) != expected["gpu_name"]:
        raise ValueError("Unexpected CUDA build or GPU")
    probe = torch.ones(1, device=device)
    if probe.sum().item() != 1:
        raise RuntimeError("CUDA tensor operation failed")
    del probe
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < protocol["preflight"]["minimum_free_vram_gib"]:
        raise RuntimeError("Insufficient free VRAM for the frozen batch-6 protocol")
    return {"packages": actual, "python": platform.python_version(), "gpu": expected["gpu_name"],
            "cuda_build": torch.version.cuda, "free_vram_gib": free / 2**30, "total_vram_gib": total / 2**30,
            "allocated_after_probe_bytes": torch.cuda.memory_allocated()}


def output_check(root, resume=False):
    root = Path(root)
    if root.exists() != resume:
        raise FileExistsError("New runs require an absent output directory; resume requires an existing official run")
    root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root.parent).free / 2**30
    # Probe permissions without reserving or populating the official output directory.
    with tempfile.TemporaryDirectory(prefix=".preflight-", dir=root.parent) as temporary:
        path = Path(temporary) / "write-check"
        path.write_bytes(b"preflight")
        if path.read_bytes() != b"preflight":
            raise OSError("Output write/read probe failed")
    return free


def preflight(resume=False):
    started = time.perf_counter()
    protocol, identity = load_protocol()
    environment = environment_check(protocol)
    disk = output_check(RUN_ROOT, resume)
    if disk < protocol["preflight"]["minimum_free_disk_gib"]:
        raise RuntimeError("Insufficient free disk space for full-precision recovery checkpoints")
    counts = verify_inputs(protocol, identity["prepared_journal_sha256"])
    warnings = ["Epoch-boundary resume does not guarantee identical worker RNG, partial accumulation or update cadence.",
                f"{protocol['training']['epochs']}-epoch stability is unmeasured; no new full smoke epoch was run."]
    if environment["free_vram_gib"] < protocol["preflight"]["warn_below_free_vram_gib"]:
        warnings.append("Free VRAM is below 4 GiB; close other GPU applications before launch.")
    result = {"status": "passed", "mode": "resume" if resume else "new run", "training_epochs": protocol["training"]["epochs"],
              "protocol_sha256": identity["protocol_sha256"],
              "split_manifest_sha256": identity["split_manifest_sha256"], "pretrained_sha256": identity["pretrained_sha256"],
              "environment": environment, "free_disk_gib": disk, "verified_prepared_files": 2 * sum(counts.values()),
              "images": counts, "patient_overlap": 0, "test_files_opened": 0,
              "official_training_started": False, "seconds": time.perf_counter() - started, "warnings": warnings}
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "latest_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return protocol, identity, result
