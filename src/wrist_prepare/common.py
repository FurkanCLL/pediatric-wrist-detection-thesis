"""Small file helpers shared by the preparation commands."""

import hashlib
import json
import os
from pathlib import Path


CLASSES = ("fracture", "metal", "text", "periostealreaction", "pronatorsign",
           "bonelesion", "softtissue", "boneanomaly")
SPLITS = ("train", "val", "test")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def protected_write(path, data):
    """Publish complete files atomically, refusing to replace different existing data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Existing output differs; use a separate destination")
        return False
    partial = path.with_name(path.name + ".partial")
    # Only this command's known incomplete file is reused after an interruption.
    with partial.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError("Output appeared while writing")
    os.replace(partial, path)
    return True


def raw_snapshot(root):
    # scandir-backed traversal avoids decoding or rehashing the complete source audit.
    result = {}
    for parent, _, files in os.walk(root):
        for name in files:
            path = Path(parent) / name
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return result
