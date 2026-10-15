"""Compare FP32 and AMP scales on one identical training batch, without updates."""

import json
from types import SimpleNamespace

from .common import OUTPUT, configure_runtime, require_cuda, save_json

configure_runtime()
import torch
from .diagnostics import finite_tensors
from .feasibility import close_loaders, make_trainer


def main():
    require_cuda()
    directory = OUTPUT / "precision_probe_01"
    directory.mkdir(exist_ok=False)
    args = SimpleNamespace(action="benchmark", batch=6, no_amp=False, workers=2)
    trainer = make_trainer(args, directory)
    trainer._setup_train()
    trainer._model_train()
    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    state = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
    assert finite_tensors(batch)
    results = []
    # Restore weights and running statistics so precision is the only changing factor.
    for amp, scale in [(False, 1), (True, 65536), (True, 2048), (True, 512), (True, 128), (True, 1)]:
        trainer.model.load_state_dict(state)
        trainer.optimizer.zero_grad()
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=scale)
        with torch.autocast("cuda", enabled=amp):
            loss, items = trainer.model(batch)
            loss = loss.sum()
        assert finite_tensors(loss) and finite_tensors(items)
        scaler.scale(loss).backward()
        scaler.unscale_(trainer.optimizer)
        invalid = [name for name, parameter in trainer.model.named_parameters()
                   if parameter.grad is not None and not finite_tensors(parameter.grad)]
        result = {"amp": amp, "scale": scale, "loss": float(loss.detach()),
                  "components": {key: float(value) for key, value in items.items()},
                  "all_gradients_finite": not invalid, "nonfinite_gradient_layers": invalid,
                  "optimizer_updates": 0}
        results.append(result)
        print(json.dumps(result), flush=True)
    save_json(directory / "result.json", {"same_batch_and_parameters": True, "results": results,
              "interpretation": "precision-only comparison; no optimization or model performance evaluation"})
    close_loaders(trainer)


if __name__ == "__main__":
    main()
