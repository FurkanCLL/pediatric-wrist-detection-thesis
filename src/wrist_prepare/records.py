"""Read verified source records and freeze patient associations locally."""

from collections import Counter, defaultdict
import csv
import io
import json

from wrist_audit.images import distribution
from wrist_audit.metadata import read_metadata
from .common import CLASSES, SPLITS, digest, json_bytes, protected_write, raw_snapshot
from .labels import target_boxes
from .splits import allocate_patients


def summarize(records):
    result = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        counts, images, patients = Counter(), defaultdict(set), defaultdict(set)
        for record in subset:
            for obj in record["boxes"]:
                c = CLASSES[obj["class_id"]]
                counts[c] += 1
                images[c].add(record["filestem"])
                patients[c].add(record["patient_id"])
        result[split] = {
            "patients": len({r["patient_id"] for r in subset}), "images": len(subset),
            "objects": sum(counts.values()), "empty_labels": sum(not r["boxes"] for r in subset),
            "fracture_images": sum(any(o["class_id"] == 0 for o in r["boxes"]) for r in subset),
            "nonfracture_images": sum(all(o["class_id"] != 0 for o in r["boxes"]) for r in subset),
            "foreignbody_images_retained": sum(r["foreignbody_boxes"] > 0 for r in subset),
            "classes": {c: {"objects": counts[c], "images": len(images[c]), "patients": len(patients[c])} for c in CLASSES},
        }
    return result


def training_statistics(records):
    subset = [r for r in records if r["split"] == "train"]
    sizes = defaultdict(list)
    for record in subset:
        for obj in record["boxes"]:
            left, top, right, bottom = obj["xyxy"]
            sizes[CLASSES[obj["class_id"]]].append(((right - left) * (bottom - top) / (record["width"] * record["height"]),
                                                   min(right - left, bottom - top)))
    return {"scope": "training split only; no sampling weights or crop parameters selected",
            "support": summarize(subset)["train"],
            "geometry": {c: {"relative_area": distribution([v[0] for v in sizes[c]]),
                             "short_side_pixels": distribution([v[1] for v in sizes[c]])} for c in CLASSES}}


def initialize(raw, audit, state, policy):
    state.mkdir(parents=True, exist_ok=True)
    policy_data = json_bytes(policy)
    protected_write(state / "policy.json", policy_data)
    if (state / "freeze.json").exists():
        return load_records(state)
    print("Checking audited source records and building the seed-42 allocation", flush=True)
    protected_write(state / "raw_snapshot_before.json", json_bytes(raw_snapshot(raw)))
    with (audit / "source_manifest.csv").open(encoding="utf-8", newline="") as stream:
        hashes = {r["path"]: r["sha256"] for r in csv.DictReader(stream)}
    if digest((raw / "dataset.csv").read_bytes()) != hashes["dataset.csv"]:
        raise ValueError("Metadata changed since the source audit")
    metadata, _, problems = read_metadata(raw / "dataset.csv")
    if problems:
        raise ValueError("Unresolved patient metadata problems")
    actual_images = {p.stem for p in (raw / "images").glob("*.png")}
    if actual_images != set(metadata):
        raise ValueError("Source image census does not match metadata")
    records = []
    for index, (stem, row) in enumerate(sorted(metadata.items()), 1):
        path = raw / "supervisely/wrist/ann" / (stem + ".json")
        annotation_data = path.read_bytes()
        if digest(annotation_data) != hashes[path.relative_to(raw).as_posix()]:
            raise ValueError("A Supervisely annotation changed since the audit")
        annotation = json.loads(annotation_data)
        image = raw / "images" / (stem + ".png")
        stat = image.stat()
        records.append({"filestem": stem, "patient_id": row["patient_id"], "study_number": row["study_number"],
                        "width": annotation["size"]["width"], "height": annotation["size"]["height"],
                        "source_image_sha256": hashes[image.relative_to(raw).as_posix()],
                        "source_annotation_sha256": digest(annotation_data),
                        "source_image_stat": [stat.st_size, stat.st_mtime_ns],
                        "boxes": target_boxes(annotation),
                        "foreignbody_boxes": sum(o["classTitle"] == "foreignbody" for o in annotation["objects"])})
        if index % 5000 == 0:
            print(f"Source annotation checks: {index}", flush=True)
    if (len(records) != policy["expected_images"] or len({r["patient_id"] for r in records}) != policy["expected_patients"]
            or sum(len(r["boxes"]) for r in records) != policy["expected_target_objects"]):
        raise ValueError("Source census differs from the approved audit references")
    allocation = allocate_patients(records, policy["seed"], policy["ratios"])
    for record in records:
        record["split"] = allocation[record["patient_id"]]
    # A repeated calculation checks determinism, using the same procedure and seed.
    if allocation != allocate_patients(list(reversed(records)), policy["seed"], policy["ratios"]):
        raise ValueError("Allocation depends on input ordering")
    image_counts = Counter(r["split"] for r in records)
    if any(abs(image_counts[s] / len(records) - ratio) > policy["maximum_image_proportion_deviation"] for s, ratio in zip(SPLITS, policy["ratios"])):
        raise ValueError("Construction fails the three-percentage-point image-balance safeguard; review before freezing")
    record_bytes = b"".join((json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode() for r in records)
    protected_write(state / "source_records.jsonl", record_bytes)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["filestem", "patient_id", "study_number", "split", "image", "label"])
    for record in records:
        stem, split = record["filestem"], record["split"]
        writer.writerow([stem, record["patient_id"], record["study_number"], split,
                         f"images/{split}/{stem}.png", f"labels/{split}/{stem}.txt"])
    manifest_bytes = buffer.getvalue().encode()
    protected_write(state / "split_manifest.csv", manifest_bytes)
    freeze = {"seed": policy["seed"], "ratios": policy["ratios"], "policy_sha256": digest(policy_data),
              "records_sha256": digest(record_bytes), "split_manifest_sha256": digest(manifest_bytes),
              "images": len(records), "patients": len(allocation), "objects": sum(len(r["boxes"]) for r in records)}
    protected_write(state / "construction_statistics.json", json_bytes(summarize(records)))
    protected_write(state / "training_statistics.json", json_bytes(training_statistics(records)))
    # Publishing this marker freezes the manifest before any pilot or full conversion.
    protected_write(state / "freeze.json", json_bytes(freeze))
    print(json.dumps({"frozen_splits": {k: {n: v[n] for n in ("patients", "images", "objects", "empty_labels")} for k, v in summarize(records).items()},
                      "manifest_sha256": freeze["split_manifest_sha256"]}), flush=True)
    return records


def load_records(state):
    freeze = json.loads((state / "freeze.json").read_text())
    data = (state / "source_records.jsonl").read_bytes()
    if (digest(data) != freeze["records_sha256"] or digest((state / "split_manifest.csv").read_bytes()) != freeze["split_manifest_sha256"]
            or digest((state / "policy.json").read_bytes()) != freeze["policy_sha256"]):
        raise ValueError("Frozen records, manifest or policy changed")
    return [json.loads(line) for line in data.splitlines()]
