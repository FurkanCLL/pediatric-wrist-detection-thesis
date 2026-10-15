"""Local settings, frozen-data checks and diagnostic input selection."""

import json
import os
from pathlib import Path
import shutil

from wrist_prepare.common import CLASSES, digest, json_bytes, protected_write
from wrist_prepare.records import load_records

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/gpu_feasibility"


def configure_runtime():
    # Set these before importing torch/Ultralytics, including in Windows workers.
    settings = OUTPUT / "framework_settings"
    for path in (settings / "Ultralytics", OUTPUT / "matplotlib_settings", OUTPUT / "weights"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib_settings")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    font = Path(os.environ.get("WINDIR", "")) / "Fonts/arial.ttf"
    destination = settings / "Ultralytics/Arial.ttf"
    if font.is_file() and not destination.exists():
        shutil.copyfile(font, destination)


def candidate():
    return json.loads((ROOT / "configs/training_candidate.json").read_text())


def select_training(records, count):
    # Hash ordering fixes the subset without selecting for model outcomes.
    training = [r for r in records if r["split"] == "train"]
    return sorted(training, key=lambda r: digest(("42:" + r["filestem"]).encode()))[:count]


def verify_frozen_data():
    state = ROOT / "outputs/dataset_preparation"
    records = load_records(state)
    frozen = json.loads((state / "freeze.json").read_text())
    published = json.loads((ROOT / "docs/evidence/dataset_preparation/frozen_manifest_hashes.json").read_text())
    assert frozen == published, "Frozen dataset receipt changed"
    validation = json.loads((state / "validation.json").read_text())
    assert validation["pixel_mapping_mismatches"] == 0 and validation["patient_overlap"] == 0
    import yaml
    data = yaml.safe_load((ROOT / "data/prepared/grazpedwri_detection/dataset.yaml").read_text())
    assert data["nc"] == 8 and data["channels"] == 3
    assert [data["names"][i] for i in range(8)] == list(CLASSES)
    return records, frozen


def write_data_yaml(directory, subset=False):
    records, frozen = verify_frozen_data()
    prepared = ROOT / "data/prepared/grazpedwri_detection"
    if subset:
        chosen = select_training(records, candidate()["benchmark"]["subset_images"])
        listing = directory / "benchmark_train.txt"
        protected_write(listing, ("\n".join((prepared / "images/train" / (r["filestem"] + ".png")).as_posix() for r in chosen) + "\n").encode())
        train = listing.resolve().as_posix()
    else:
        train = (prepared / "images/train").as_posix()
    # A separate diagnostic YAML omits the test path altogether.
    data = {"path": prepared.as_posix(), "train": train,
            "val": (prepared / "images/val").as_posix(), "nc": 8,
            "channels": 3, "names": dict(enumerate(CLASSES))}
    import yaml
    path = directory / "data.yaml"
    protected_write(path, yaml.safe_dump(data, sort_keys=False).encode())
    return path, frozen


def save_json(path, value):
    protected_write(path, json_bytes(value))


def require_cuda():
    import torch
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    return torch.device("cuda:0")
