"""Independent checks on saved histories, sample counts and AMP events."""

from collections import deque
import math

import numpy as np

from wrist_prepare.common import CLASSES
from wrist_train.progress import METRIC_KEYS, LOSS_KEYS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(records, count, label):
    require([r["epoch"] for r in records] == list(range(1, count+1)), label+" epoch order/count differs")


def csv_history(rows, count=80):
    required = {"epoch", "time", *METRIC_KEYS, *("train/"+k for k in LOSS_KEYS),
                *("val/"+k for k in LOSS_KEYS), *(f"lr/pg{i}" for i in range(3))}
    require(len(rows) == count, "CSV must contain exactly the completed epoch budget")
    parsed = []
    for row in rows:
        require(required <= row.keys(), "CSV required column missing")
        values = {k: float(v) for k, v in row.items()}
        require(all(math.isfinite(v) for v in values.values()), "CSV contains non-finite values")
        require(all(0 <= values[k] <= 1 for k in METRIC_KEYS), "CSV metric out of range")
        parsed.append(values)
    ordered(parsed, count, "CSV")
    require(all(r["time"] > 0 for r in parsed) and
            all(b["time"] > a["time"] for a, b in zip(parsed, parsed[1:])), "CSV time is not increasing")
    return parsed


def class_history(records, rows):
    ordered(records, len(rows), "Per-class history")
    for record, row in zip(records, rows):
        classes = record["classes"]
        require(len(classes) == 8 and [c["class_id"] for c in classes] == list(range(8)) and
                [c["class"] for c in classes] == list(CLASSES), "Incomplete or changed class mapping")
        for key, metric in zip(("precision", "recall", "ap50", "ap50_95"), METRIC_KEYS):
            values = [float(c[key]) for c in classes]
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid per-class metric")
            # Native trainer validation rounds macro metrics to five decimal places before selection.
            require(round(sum(values)/8, 5) == row[metric], "Per-class macro disagrees with native CSV")
    # Select on the actual rounded fitness used by the frozen trainer, retaining the latest exact tie.
    return max(range(len(rows)), key=lambda i: (rows[i][METRIC_KEYS[3]], i))+1


def optimizer_coverage(model, optimizer):
    import torch
    norms = tuple(v for k, v in vars(torch.nn).items() if "Norm" in k)
    groups = [[], [], []]
    # Match the pinned AdamW grouping order without constructing or stepping an optimizer.
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            full = f"{module_name}.{name}" if module_name else name
            index = 2 if "bias" in full else 1 if isinstance(module, norms) or "logit_scale" in full else 0
            groups[index].append((full, parameter))
    require(len(optimizer["param_groups"]) == 3, "Optimizer parameter group count differs")
    mapping = {}
    for actual, expected in zip(optimizer["param_groups"], groups):
        require(len(actual["params"]) == len(expected), "Optimizer parameter group coverage differs")
        for index, pair in zip(actual["params"], expected):
            require(index not in mapping, "Duplicate optimizer parameter ID")
            mapping[index] = pair
    expected_state = {i for i, (_, p) in mapping.items() if p.requires_grad}
    require(set(optimizer["state"]) == expected_state, "Trainable optimizer state coverage differs")
    for index, state in optimizer["state"].items():
        require(all(state[k].shape == mapping[index][1].shape for k in ("exp_avg", "exp_avg_sq")),
                "AdamW moment shape differs from parameter")
    return [name for name, p in mapping.values() if not p.requires_grad]


def sampling_counts(indices, patient, positive, lesion_patients):
    indices = np.asarray(indices)
    require(indices.shape == (14061,) and indices.dtype.kind in "iu" and
            np.all((indices >= 0) & (indices < len(patient))), "Invalid consumed indices")
    counts = np.bincount(indices, minlength=len(patient))
    seen = counts > 0
    sizes = np.bincount(patient)
    extras = np.bincount(patient, weights=np.maximum(counts-1, 0), minlength=len(sizes))
    cap = np.minimum(3, np.ceil(.5*sizes))
    lesion = np.isin(patient, lesion_patients)
    unique_strata = [int((seen & (positive == status)).sum()) for status in (True, False)]
    draws = [int((positive[indices] == status).sum()) for status in (True, False)]
    require(seen.sum() == 13436 and np.maximum(counts-1, 0).sum() == 625 and counts.max() <= 2,
            "Image repetition/coverage differs")
    require(len(np.unique(patient[seen])) == 4264 and np.all(extras <= cap), "Patient coverage/cap differs")
    require(draws == [9307, 4754] and all(n >= math.ceil(.95*d) for n, d in zip(unique_strata, draws)),
            "Fracture-stratum exposure/coverage differs")
    require(len(lesion_patients) == 16 and lesion.sum() == 66 and np.all(counts[lesion] == 1),
            "Lesion cohort protection differs")
    return {"draws": len(indices), "unique_images": int(seen.sum()), "replacements": 625,
            "patients": len(np.unique(patient[seen])), "fracture_draws": draws[0],
            "no_fracture_box_draws": draws[1], "unique_fracture_images": unique_strata[0],
            "unique_no_fracture_box_images": unique_strata[1], "maximum_image_draws": int(counts.max()),
            "maximum_patient_extra": int(extras.max()), "lesion_cohort_views_once": int(lesion.sum())}


def amp_history(events, receipts, policy):
    amps = [e for e in events if e["event"] == "amp"]
    require(all(a["attempt"] < b["attempt"] for a, b in zip(amps, amps[1:])), "AMP event ordering differs")
    skips = {e["attempt"] for e in amps if not e["updated"]}
    require(len(skips) == sum(not e["updated"] for e in amps), "Duplicate overflow event")
    scale, tracker = policy["amp_initial_scale"], 0
    attempt = successful = skipped = consecutive = maximum_run = maximum_window = 0
    recent = deque(maxlen=policy["overflow_window"])
    expected_events, snapshots = [], []
    for receipt in receipts:
        require(receipt["scale_start"] == scale, "Epoch starting AMP scale differs")
        for _ in range(receipt["optimizer_epoch"]["attempts"]):
            attempt += 1
            updated, before = attempt not in skips, scale
            recovered = updated and consecutive > 0
            successful += int(updated)
            skipped += int(not updated)
            consecutive = 0 if updated else consecutive+1
            recent.append(int(not updated))
            maximum_run, maximum_window = max(maximum_run, consecutive), max(maximum_window, sum(recent))
            if updated:
                tracker += 1
                if tracker == policy["amp_growth_interval"]:
                    scale *= policy["amp_growth_factor"]
                    tracker = 0
            else:
                scale *= policy["amp_backoff_factor"]
                tracker = 0
            persistent = consecutive >= policy["overflow_max_consecutive"] or sum(recent) >= policy["overflow_max_in_window"]
            require(not persistent, "Persistent AMP instability in recorded history")
            if not updated or before != scale or recovered:
                expected_events.append({"epoch": receipt["epoch"], "event": "amp", "attempt": attempt,
                    "successful_updates": successful, "skipped_updates": skipped, "updated": updated,
                    "scale_before": before, "scale_after": scale, "recovered": recovered, "persistent": persistent})
        require(receipt["scale_end"] == scale, "Epoch ending AMP scale differs")
        snapshots.append({"accounting": {"attempts": attempt, "successful": successful, "skipped": skipped,
            "consecutive": consecutive, "recent": list(recent)}, "scale": scale, "tracker": tracker})
    require(amps == expected_events, "AMP growth, overflow, recovery or counters do not reconcile")
    return snapshots, {"events": len(amps), "overflow_skips": skipped,
        "growth_events": sum(e["scale_after"] > e["scale_before"] for e in amps),
        "recovery_events": sum(e["recovered"] for e in amps),
        "scale_min": min([policy["amp_initial_scale"], *(e["scale_after"] for e in amps)]),
        "scale_max": max([policy["amp_initial_scale"], *(e["scale_after"] for e in amps)]),
        "final_scale": scale, "final_growth_tracker": tracker,
        "maximum_consecutive_skips": maximum_run, "maximum_skips_in_window": maximum_window}
