"""Image-array routing to frozen standard, SAHI Context and V3 components."""

import numpy as np
import time

from wrist_v3.core import predict as predict_v3
from wrist_eval.metrics import validate_predictions


class FrozenPredictor:
    def __init__(self, model, view, sliced, v3_settings, legacy_settings, inference=None):
        self.view, self.sliced = view, sliced
        self.v3_settings, self.legacy_settings = v3_settings, legacy_settings
        self.inference = inference
        self.physical_calls = 0
        self.hook = model.register_forward_pre_hook(self._count_forward)

    def _count_forward(self, module, inputs):
        self.physical_calls += 1

    def close(self):
        self.hook.remove()

    def __call__(self, image, inference, net_shape, *, started_at):
        if self.inference is not None and inference != self.inference:
            raise ValueError('Inference method differs from the final condition identity')
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected a prepared three-channel uint8 image')
        before = self.physical_calls
        global_prefix = [None]
        def view(crop, shape):
            value = self.view(crop, shape)
            if shape is not None:
                global_prefix[0] = time.perf_counter()-started_at
            return value
        if inference == 'standard':
            value = view(image, net_shape)
            pred, calls, limit = value['predictions'], value['counters']['forward_passes'], 1
        elif inference == 'sahi_context':
            pred, value = self.sliced(image, net_shape)
            calls, limit = value['forward_passes'], 5
            if calls != value['slices']+1:
                raise ValueError('SAHI calls do not equal slices plus the global view')
        elif inference == 'cg_alr_v3':
            value = predict_v3(image, view, net_shape, self.legacy_settings, self.v3_settings)
            pred, calls, limit = value['predictions'], value['passes'], 3
        else:
            raise ValueError('Unknown final inference method')
        if self.physical_calls-before != calls or not 1 <= calls <= limit:
            raise ValueError('Independent physical model-call accounting failed')
        # Validate without replacing the inference output's original order or stable V3 IDs.
        validate_predictions(pred, image.shape[1], image.shape[0])
        if len(pred) > 300:
            raise ValueError('Frozen output cap exceeded')
        return {'predictions': pred, 'actual_model_calls': calls, 'diagnostics': value,
                'e2e_seconds': time.perf_counter()-started_at, 'paired_global_e2e_seconds': global_prefix[0]}


def create_predictor(condition, destination, *, phase5d_authorized=False):
    # Preparation must never load a model or open a test image through this factory.
    if not phase5d_authorized:
        raise PermissionError('Separate student authorization for Phase 5D is required')
    from pathlib import Path
    from unittest.mock import patch
    from wrist_compare.common import checked_checkpoint
    from wrist_eval.adaptive_study import ViewRunner
    from wrist_eval.sliced import predict
    from .protocol import ROOT, read, verify_frozen
    cfg = verify_frozen()
    conditions = {v['id']: v for v in cfg['conditions']}
    if condition not in conditions:
        raise ValueError('Use a distinct final_A through final_F identity')
    destination = Path(destination).resolve()
    if not destination.is_relative_to((ROOT/'outputs/final_evaluation').resolve()):
        raise ValueError('Final outputs must be separate from historical experiments')
    training = conditions[condition]['training']
    payload, receipt = checked_checkpoint('A' if training == 'standard' else 'D')
    if receipt['sha256'] != cfg['checkpoints'][training]['sha256']:
        raise ValueError('Final checkpoint identity differs')
    model = payload['ema'].float().eval().to('cuda:0').requires_grad_(False)
    with patch('wrist_eval.common.LOCAL', destination):
        view = ViewRunner(model)
    return FrozenPredictor(model, view, lambda image, shape: predict(image, view.adapter, 'sahi_context', shape),
                           read(ROOT/'configs/cg_alr_v3.json'), read(ROOT/'configs/adaptive_refinement_proposed.json'),
                           inference=conditions[condition]['inference'])
