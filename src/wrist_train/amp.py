"""Controlled AMP updates and persistent numerical-failure accounting."""

from collections import deque
import math

import torch


def state_is_finite(state):
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    return not tensors or bool(torch.stack([torch.isfinite(value).all() for value in tensors]).all())


def guard_training_loss(module, inputs, output):
    # Model(dict) returns the loss before the native loop starts backward.
    if module.training and inputs and isinstance(inputs[0], dict):
        loss, components = output
        if not state_is_finite({"loss": loss, **components}):
            raise FloatingPointError("Non-finite training loss; backward was not started")


class UpdateMonitor:
    def __init__(self, policy, state=None):
        self.policy = policy
        self.attempts = self.successful = self.skipped = self.consecutive = 0
        self.recent = deque(maxlen=policy["overflow_window"])
        if state:
            self.attempts, self.successful, self.skipped, self.consecutive = (
                state[key] for key in ("attempts", "successful", "skipped", "consecutive"))
            self.recent.extend(state["recent"])
            if self.attempts != self.successful + self.skipped:
                raise ValueError("Inconsistent saved optimizer accounting")

    def state_dict(self):
        return {key: getattr(self, key) for key in ("attempts", "successful", "skipped", "consecutive")} | {"recent": list(self.recent)}

    def record(self, updated, before, after):
        if not all(math.isfinite(value) and value > 0 for value in (before, after)):
            raise FloatingPointError("Invalid AMP loss scale")
        recovered = updated and self.consecutive > 0
        self.attempts += 1
        self.successful += int(updated)
        self.skipped += int(not updated)
        self.consecutive = 0 if updated else self.consecutive + 1
        self.recent.append(int(not updated))
        event = {"attempt": self.attempts, "successful_updates": self.successful,
                 "skipped_updates": self.skipped, "updated": updated, "scale_before": before,
                 "scale_after": after, "recovered": recovered,
                 "persistent": self.consecutive >= self.policy["overflow_max_consecutive"] or sum(self.recent) >= self.policy["overflow_max_in_window"]}
        return event


def controlled_step(model, optimizer, scaler, ema, monitor, emit):
    before = scaler.get_scale()
    scaler.unscale_(optimizer)
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    finite = state_is_finite(gradients)
    if finite:
        # A non-finite norm with finite elements is also unsafe; do not silently zero it.
        torch.nn.utils.clip_grad_norm_(model.parameters(), monitor.policy["gradient_clip_norm"], error_if_nonfinite=True)
    elif not scaler.is_enabled():
        raise FloatingPointError("Non-finite gradients with AMP disabled")
    parameter = next((p for p in model.parameters() if p.grad is not None), None)
    if parameter is None:
        raise RuntimeError("An optimizer attempt has no gradients")
    def step_number():
        value = optimizer.state.get(parameter, {}).get("step", 0)
        return float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    previous_step = step_number()
    # Fused AdamW calls step even on overflow, but its kernel leaves this counter unchanged.
    scaler.step(optimizer)
    scaler.update()
    difference = step_number() - previous_step
    if difference not in (0, 1):
        raise RuntimeError("Unexpected AdamW step-counter change")
    updated = difference == 1
    event = monitor.record(updated, before, scaler.get_scale())
    if not finite and updated:
        emit({**event, "corruption": "optimizer stepped on non-finite gradients"})
        raise FloatingPointError("Optimizer updated despite non-finite gradients")
    if updated:
        if not state_is_finite(model.state_dict()):
            emit({**event, "corruption": "non-finite model state"})
            raise FloatingPointError("Non-finite model parameter or buffer after optimizer update")
        if ema is not None:
            ema.update(model)
            if not state_is_finite(ema.ema.state_dict()):
                emit({**event, "corruption": "non-finite EMA state"})
                raise FloatingPointError("Non-finite EMA state")
    optimizer.zero_grad()
    if not updated or event["scale_before"] != event["scale_after"] or event["recovered"]:
        emit(event)
    if event["persistent"]:
        raise FloatingPointError("Persistent AMP instability: consecutive or rolling-window overflow limit reached")
    return event
