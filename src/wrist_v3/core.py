"""Single-checkpoint V3 invocation with explicit physical model-call accounting."""

import time
import numpy as np

from .planning import plan_rois
from .fusion import fuse_predictions


def predict(image, infer_view, global_shape, legacy, settings):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected prepared three-channel uint8 image')
    began = time.perf_counter()
    global_view = infer_view(image, global_shape)
    planning_start = time.perf_counter()
    plan = plan_rois(global_view['predictions'], image.shape[:2], legacy, settings)
    planning_seconds = time.perf_counter()-planning_start
    views = []
    for roi in plan['rois']:
        x1, y1, x2, y2 = roi['box']
        views.append({**infer_view(image[y1:y2, x1:x2], None), 'roi_id': roi['id']})
    common_seconds = time.perf_counter()-began
    fusion_start = time.perf_counter()
    fused = fuse_predictions(global_view['predictions'], views, plan, legacy, settings)
    fusion_seconds = time.perf_counter()-fusion_start
    counters = [global_view['counters'], *[v['counters'] for v in views]]
    calls = sum(v['forward_passes'] for v in counters)
    if any(v['forward_passes'] != 1 for v in counters) or calls != 1+len(views) or calls > settings['maximum_model_calls']:
        raise AssertionError('V3 physical model-call limit violated')
    return {'global': global_view['predictions'], 'global_view': global_view, 'plan': plan,
            'local_views': views, **fused, 'passes': calls, 'common_seconds': common_seconds,
            'planning_seconds': planning_seconds, 'fusion_seconds': fusion_seconds}
