"""Atomic local recovery files with checked, full-precision training state."""

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid

import torch

from .amp import state_is_finite
from wrist_prepare.common import CLASSES


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def model_snapshot(model):
    if not state_is_finite(model.state_dict()):
        raise FloatingPointError("Refusing to save non-finite model or EMA")
    result = deepcopy(model).float().cpu()
    result._forward_hooks.clear()
    if hasattr(result, "criterion"):
        result.criterion = None
    return result


def sidecar(path):
    return Path(str(path) + ".sha256.json")


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_bytes(path):
    path = Path(path)
    manifest = json.loads(sidecar(path).read_text())
    content = path.read_bytes()
    if len(content) != manifest["bytes"] or hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint hash/length mismatch; use a verified previous or periodic checkpoint")
    return content


def publish_checkpoint(path, payload, backup=False):
    path = Path(path)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    manifest = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                "completed_epoch": payload["epoch"] + 1,
                "protocol_sha256": payload["official"]["protocol_sha256"]}
    if path.exists():
        # Preserve a valid rolling predecessor before replacing last/best.
        if not backup:
            raise FileExistsError("Refusing to replace a recovery checkpoint")
        old = checked_bytes(path)
        previous = path.with_name(path.stem + ".previous.pt")
        atomic_bytes(previous, old)
        atomic_bytes(sidecar(previous), sidecar(path).read_bytes())
    atomic_bytes(path, content)
    # A power loss between these two renames is detected by the hash; the predecessor remains available.
    atomic_bytes(sidecar(path), (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())


def check_payload(payload, identity, require_resume=True):
    official = payload.get("official", {})
    for key in ("protocol_sha256", "split_manifest_sha256", "pretrained_sha256", "implementation_source_sha256", "training_epochs"):
        if official.get(key) != identity[key]:
            raise ValueError("Checkpoint does not match the frozen official identity: " + key)
    if official.get("format_version") != 1:
        raise ValueError("Unsupported official checkpoint format")
    for key in ("model", "ema", "best_ema"):
        if payload.get(key) is None or not state_is_finite(payload[key].state_dict()):
            raise ValueError("Missing or non-finite checkpoint state: " + key)
        if getattr(payload[key], "names", None) != dict(enumerate(CLASSES)):
            raise ValueError("Checkpoint class mapping is incompatible")
    if require_resume:
        required = ("optimizer", "scaler", "scheduler", "accounting", "train_results", "train_args", "updates")
        if not official.get("resumable") or any(payload.get(key) is None for key in required):
            raise ValueError("Checkpoint lacks resumable training state")
        # The saved epoch is zero-based; one more completed epoch must still fit the frozen budget.
        budget = identity["training_epochs"]
        if payload["train_args"].get("epochs") != budget:
            raise ValueError("Checkpoint arguments disagree with the frozen epoch budget")
        if not 0 <= payload.get("epoch", -1) < budget - 1:
            raise ValueError(f"Checkpoint has no unfinished epoch within the frozen {budget}-epoch budget")
        if payload["updates"] != payload["accounting"]["successful"]:
            raise ValueError("EMA and successful optimizer counts disagree")
        if payload["scheduler"]["last_epoch"] != payload["epoch"]:
            raise ValueError("Checkpoint schedule position disagrees with epoch")
        results = payload["train_results"]
        epochs = list(range(1, payload["epoch"] + 2))
        if results.get("epoch") != epochs or any(len(values) != len(epochs) for values in results.values()):
            raise ValueError("Checkpoint metrics history disagrees with completed epochs")
        if not nested_finite({key: payload[key] for key in ("optimizer", "scaler", "scheduler", "best_fitness")}):
            raise ValueError("Non-finite optimizer/scaler/scheduler state")
        if payload["scaler"].get("scale", 0) <= 0:
            raise ValueError("Missing or invalid saved AMP scale")


def nested_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(nested_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_recovery(path, identity):
    # Pickle loading is restricted by the CLI to locally generated, hash-verified official files.
    payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
    check_payload(payload, identity)
    return payload
