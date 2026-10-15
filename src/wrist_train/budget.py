"""Compute candidate update budgets using the installed trainer's accumulation cadence."""

import math

import numpy as np


def optimizer_budget(images, batch, nominal_batch, epochs, warmup_epochs):
    if min(images, batch, nominal_batch, epochs) <= 0 or warmup_epochs < 0:
        raise ValueError("Budget inputs must be positive, with nonnegative warmup")
    batches = math.ceil(images / batch)
    warmup = round(min(warmup_epochs, max(epochs - 1, 0)) * batches)
    accumulate = max(round(nominal_batch / batch), 1)
    last_step = -1
    updates = []
    # Warmup changes accumulation too, so images/effective_batch alone is not exact.
    for epoch in range(epochs):
        count = 0
        for index in range(batches):
            iteration = index + epoch * batches
            current = max(1, int(np.interp(iteration, [0, warmup], [1, nominal_batch / batch]).round())) if iteration < warmup else accumulate
            if iteration - last_step >= current:
                count += 1
                last_step = iteration
        updates.append(count)
    return {"draws_per_epoch": images, "physical_batch": batch, "batches_per_epoch": batches,
            "nominal_batch": nominal_batch, "steady_accumulation": accumulate,
            "steady_effective_batch": batch * accumulate, "warmup_iterations": warmup,
            "epochs": epochs, "updates_per_epoch": updates, "total_optimizer_updates": sum(updates)}
