"""One concise summary from each existing epoch validation result."""

import json
import math
from pathlib import Path


METRIC_KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
LOSS_KEYS = ("box_loss", "cls_loss", "dfl_loss")


def per_class_record(metrics, names):
    rows = []
    for index, class_id in enumerate(metrics.ap_class_index):
        values = list(map(float, metrics.class_result(index)))
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Missing or non-finite per-class validation metrics")
        rows.append({"class_id": int(class_id), "class": names[int(class_id)],
                     **dict(zip(("precision", "recall", "ap50", "ap50_95"), values))})
    if {row["class_id"] for row in rows} != set(range(8)):
        raise ValueError("Validation did not produce metrics for all eight supported classes")
    return rows


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def epoch_record(epoch, epochs, training, metrics, train_s, validation_s, best, accounting):
    required = (*METRIC_KEYS, *("val/" + key for key in LOSS_KEYS))
    if any(key not in metrics for key in required) or any(key not in training for key in LOSS_KEYS):
        raise ValueError("Missing required epoch/validation metrics; no summary or checkpoint is accepted")
    values = {key: float(metrics[key]) for key in required}
    train = {key: float(training[key]) for key in LOSS_KEYS}
    if not all(math.isfinite(value) for value in [*values.values(), *train.values(), train_s, validation_s, best]):
        raise FloatingPointError("Non-finite epoch/validation metrics")
    return {"epoch": epoch, "epochs": epochs, "train_losses": train,
            "validation_losses": {key: values["val/" + key] for key in LOSS_KEYS},
            "precision": values[METRIC_KEYS[0]], "recall": values[METRIC_KEYS[1]],
            "map50": values[METRIC_KEYS[2]], "map50_95": values[METRIC_KEYS[3]],
            "training_seconds": train_s, "validation_seconds": validation_s,
            "epoch_seconds": train_s + validation_s, "best_map50_95": best,
            "optimizer": accounting}


def display_epoch(directory, record):
    train = "/".join(f"{record['train_losses'][key]:.4f}" for key in LOSS_KEYS)
    val = "/".join(f"{record['validation_losses'][key]:.4f}" for key in LOSS_KEYS)
    text = (f"Epoch {record['epoch']}/{record['epochs']} | train box/cls/dfl {train} | val {val}\n"
            f"P {record['precision']:.4f} R {record['recall']:.4f} mAP50 {record['map50']:.4f} "
            f"mAP50-95 {record['map50_95']:.4f} best {record['best_map50_95']:.4f} | "
            f"train {record['training_seconds']:.1f}s val {record['validation_seconds']:.1f}s "
            f"epoch {record['epoch_seconds']:.1f}s | updates {record['optimizer']['successful']} "
            f"skipped {record['optimizer']['skipped']}")
    print(text, flush=True)
    with (Path(directory) / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
    append_json(Path(directory) / "epochs.jsonl", record)
