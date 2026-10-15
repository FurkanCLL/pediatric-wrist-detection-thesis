"""One deterministic patient-level multilabel allocation, with no seed search."""

from collections import defaultdict
import math
import random

import numpy as np

from .common import CLASSES, SPLITS


def patient_capacities(count, ratios):
    raw = [count * r for r in ratios]
    capacities = [math.floor(v) for v in raw]
    for i in sorted(range(3), key=lambda j: (-(raw[j] - capacities[j]), j))[:count - sum(capacities)]:
        capacities[i] += 1
    return capacities


def allocate_patients(records, seed=42, ratios=(.7, .15, .15)):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("Three positive split proportions must sum to one")
    groups = defaultdict(list)
    for record in records:
        if not record["patient_id"]:
            raise ValueError("Missing patient identity")
        groups[record["patient_id"]].append(record)
    patients = sorted(groups)
    if len(patients) < 3:
        raise ValueError("At least three independent patients are needed")
    features, sizes = {}, {}
    for patient, images in groups.items():
        classes = {o["class_id"] for r in images for o in r["boxes"]}
        features[patient] = classes | ({8} if any(not r["boxes"] for r in images) else set())
        if any(all(o["class_id"] != 0 for o in r["boxes"]) for r in images):
            features[patient].add(9)
        sizes[patient] = len(images)
    support = np.array([sum(f in features[p] for p in patients) for f in range(10)])
    desired = np.zeros((3, 10), dtype=float)
    for f, count in enumerate(support):
        # Symmetric rounded holdout targets give a 24-patient class four per holdout.
        holdouts = [int(math.floor(count * r + .5)) for r in ratios[1:]]
        if count >= 3:
            holdouts = [max(1, n) for n in holdouts]
        desired[:, f] = [count - sum(holdouts), *holdouts]
    capacity = np.array(patient_capacities(len(patients), ratios))
    patient_counts, image_counts = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    target_images = np.array(ratios) * len(records)
    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    rank = {p: i for i, p in enumerate(shuffled)}
    split_ties = {p: rng.sample(range(3), 3) for p in patients}
    remaining = {f: {p for p in patients if f in features[p]} for f in range(10)}
    unassigned, allocation = set(patients), {}
    while unassigned:
        available = [f for f, members in remaining.items() if members]
        rare = min(available, key=lambda f: (len(remaining[f]), f)) if available else None
        candidates = remaining[rare] if rare is not None else unassigned
        # Seeded ordering avoids sending the largest examination groups to train first.
        ordered = sorted(candidates, key=lambda p: rank[p])
        for patient in ordered:
            eligible = [s for s in range(3) if patient_counts[s] < capacity[s]]
            def score(s):
                rare_need = desired[s, rare] if rare is not None else 0
                joint_need = sum(desired[s, f] / max(1, support[f]) for f in features[patient])
                return (rare_need, joint_need,
                        (target_images[s] - image_counts[s]) / target_images[s],
                        (capacity[s] - patient_counts[s]) / capacity[s],
                        -split_ties[patient].index(s))
            selected = max(eligible, key=score)
            allocation[patient] = SPLITS[selected]
            patient_counts[selected] += 1
            image_counts[selected] += sizes[patient]
            for f in features[patient]:
                desired[selected, f] -= 1
                remaining[f].discard(patient)
            unassigned.remove(patient)
    return allocation
