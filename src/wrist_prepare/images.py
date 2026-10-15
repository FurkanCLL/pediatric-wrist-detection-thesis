"""Incremental PNG conversion and resumable, protected output writes."""

from concurrent.futures import ThreadPoolExecutor
import io
import sqlite3

import numpy as np
from PIL import Image

from .common import digest, json_bytes, protected_write
from .labels import yolo_bytes


def to_uint8(pixels):
    if pixels.dtype != np.uint16 or pixels.ndim != 2:
        raise ValueError("Expected original unsigned 16-bit grayscale pixels")
    # uint32 avoids overflow. The odd divisor makes exact half-way ties impossible.
    return ((pixels.astype(np.uint32) + 128) // 257).astype(np.uint8)


def convert_one(record, raw, destination, previous=None):
    stem, split = record["filestem"], record["split"]
    source = raw / "images" / (stem + ".png")
    stat = source.stat()
    if [stat.st_size, stat.st_mtime_ns] != record["source_image_stat"]:
        raise ValueError("Source image metadata changed after the split freeze")
    image_path = destination / "images" / split / (stem + ".png")
    label_path = destination / "labels" / split / (stem + ".txt")
    labels = yolo_bytes(record)
    if previous:
        if (digest(image_path.read_bytes()) != previous[0] or digest(label_path.read_bytes()) != previous[1]
                or previous[1] != digest(labels)):
            raise ValueError("A completed output changed; refusing to overwrite it")
        return stem, previous[0], previous[1], True
    source_data = source.read_bytes()
    if digest(source_data) != record["source_image_sha256"]:
        raise ValueError("Source image content differs from the audited original")
    with Image.open(io.BytesIO(source_data)) as image:
        pixels = np.asarray(image)
        if pixels.shape != (record["height"], record["width"]):
            raise ValueError("Original dimensions disagree with source annotations")
        converted = to_uint8(pixels)
    buffer = io.BytesIO()
    Image.fromarray(converted).save(buffer, format="PNG", compress_level=6, optimize=False)
    data = buffer.getvalue()
    protected_write(image_path, data)
    protected_write(label_path, labels)
    return stem, digest(data), digest(labels), False


def convert_records(records, raw, destination, database, freeze, workers=4):
    destination.mkdir(parents=True, exist_ok=True)
    protected_write(destination / "preparation.json", json_bytes(freeze))
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE IF NOT EXISTS completed (stem TEXT PRIMARY KEY, image_sha TEXT, label_sha TEXT)")
    previous = {r[0]: (r[1], r[2]) for r in connection.execute("SELECT * FROM completed")}
    skipped = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda r: convert_one(r, raw, destination, previous.get(r["filestem"])), records)
            for index, (stem, image_hash, label_hash, reused) in enumerate(results, 1):
                connection.execute("INSERT OR REPLACE INTO completed VALUES (?, ?, ?)", (stem, image_hash, label_hash))
                skipped += reused
                if index % 128 == 0:
                    connection.commit()
                if index % 1000 == 0:
                    print(f"Prepared PNG/label pairs: {index}/{len(records)}", flush=True)
        connection.commit()
    finally:
        connection.close()
    return {"images": len(records), "reused_completed_outputs": skipped, "newly_completed": len(records) - skipped}
