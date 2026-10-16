"""SAHI slicing/coordinate restoration with the native validator's numerical policy."""

import math
import time
import numpy as np

from .common import CLASSES, config
from .metrics import validate_predictions, remove_padding_only


def tile_dimensions(height, width, fraction):
    if not 0 < fraction <= 1 or min(height, width) <= 0:
        raise ValueError("Invalid slice geometry")
    return min(height, math.ceil(height * fraction)), min(width, math.ceil(width * fraction))


def resize_and_pad(image, full_shape=None):
    import cv2
    h, w = image.shape[:2]
    r = 768 / max(h, w)
    rh, rw = min(math.ceil(h * r), 768), min(math.ceil(w * r), 768)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rh, rw) != (h, w) else image.copy()
    # Match validation's half-stride padding and integer rounding, without stretching the radiograph.
    out_h, out_w = full_shape or (int(math.ceil(rh / 32 + .5) * 32), int(math.ceil(rw / 32 + .5) * 32))
    if out_h < rh or out_w < rw:
        raise ValueError("Requested letterbox does not contain resized image")
    left, top = round((out_w-rw)/2-.1), round((out_h-rh)/2-.1)
    padded = cv2.copyMakeBorder(resized, top, out_h-rh-top, left, out_w-rw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, ((rh/h, rw/w), (left, top))


def make_adapter(model):
    import torch
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops
    from .common import LOCAL

    class ValidationAdapter(UltralyticsDetectionModel):
        # Reuse SAHI's public model interface and box conversion. Only the numerical inference path differs:
        # native unfused EMA/autocast and multi-label validation NMS replace generic YOLO.predict defaults.
        def __init__(self):
            super().__init__(model=model, confidence_threshold=.001, device="cuda:0", image_size=768,
                             category_mapping={str(i): name for i, name in enumerate(CLASSES)}, fuse=False)
            self.validator = DetectionValidator(save_dir=LOCAL / "adapter_settings", args={"imgsz":768, "conf":.001,
                "iou":.7,"max_det":300,"plots":False,"augment":False,"agnostic_nms":False})
            self.validator.end2end = False
            self.validator.nc = 8
            self.global_hw, self.global_net_shape = None, None
            self.reset_counters()

        def reset_counters(self):
            self.counters = {"forward_passes":0, "pre_merge_predictions":0,"gpu_forward_ms":0.,
                             "preprocess_seconds":0., "model_wall_seconds":0., "nms_restore_seconds":0.,
                             "padding_only_predictions":0}

        @torch.inference_mode()
        def perform_batch_inference(self, images):
            if len(images) != 1:
                raise ValueError("This feasibility adapter explicitly uses sequential batch one")
            image = images[0]
            started = time.perf_counter()
            full_shape = self.global_net_shape if image.shape[:2] == self.global_hw else None
            padded, ratio_pad = resize_and_pad(image, full_shape)
            tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2,0,1))).to("cuda:0").half()[None] / 255
            torch.cuda.synchronize()
            self.counters["preprocess_seconds"] += time.perf_counter()-started
            started = time.perf_counter()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.autocast("cuda", dtype=torch.float16):
                raw = self.model(tensor, augment=False)
            end.record()
            end.synchronize()
            self.counters["gpu_forward_ms"] += begin.elapsed_time(end)
            self.counters["model_wall_seconds"] += time.perf_counter()-started
            if not torch.isfinite(raw[0]).all():
                raise FloatingPointError("Non-finite inference output")
            started = time.perf_counter()
            pred = self.validator.postprocess(raw)[0]
            restored = ops.scale_boxes(tensor.shape[2:], pred["bboxes"].clone(), image.shape[:2], ratio_pad)
            values = torch.cat((restored, pred["conf"][:,None],pred["cls"][:,None]),1).float().cpu().numpy()
            values,removed=remove_padding_only(values)
            self.counters["padding_only_predictions"]+=removed
            values = validate_predictions(values, image.shape[1], image.shape[0])
            self._original_predictions = [torch.tensor(values)]
            self._original_shapes = [image.shape]
            self.counters["nms_restore_seconds"] += time.perf_counter()-started
            self.counters["forward_passes"] += 1
            self.counters["pre_merge_predictions"] += len(values)

    return ValidationAdapter()


def predict(image, adapter, variant, net_shape):
    import torch
    from sahi.predict import get_prediction, get_sliced_prediction
    cfg = config()
    adapter.global_hw = image.shape[:2]
    adapter.global_net_shape = net_shape
    adapter.reset_counters()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fraction = cfg["variants"][variant]["slice_fraction"]
    if fraction is None:
        result = get_prediction(image, adapter, verbose=0)
        slices = 0
    else:
        h, w = tile_dimensions(*image.shape[:2], fraction)
        from sahi.slicing import get_slice_bboxes
        slices = len(get_slice_bboxes(*image.shape[:2], slice_height=h, slice_width=w,
                     overlap_height_ratio=.2, overlap_width_ratio=.2, auto_slice_resolution=False))
        if slices > cfg["full_run_limits"]["max_slices_per_image"]:
            raise RuntimeError("Slice budget exceeded")
        result = get_sliced_prediction(image, adapter, slice_height=h, slice_width=w,
            overlap_height_ratio=.2, overlap_width_ratio=.2, perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=.7,
            postprocess_class_agnostic=False, force_postprocess_type=True, merge_buffer_length=None,
            auto_slice_resolution=False, verbose=0, batch_size=1)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    pred = np.array([p.bbox.to_xyxy()+[p.score.value,p.category.id] for p in result.object_prediction_list],dtype=np.float32).reshape(-1,6)
    # SAHI does not expose a final max_det setting; apply the same image-level cap after its per-class NMS.
    pred = validate_predictions(pred, image.shape[1], image.shape[0])[:cfg["max_det"]]
    counters = {**adapter.counters, "seconds":seconds,"slices":slices,"post_merge_predictions":len(pred),
                "allocated_peak_bytes":torch.cuda.max_memory_allocated(),"reserved_peak_bytes":torch.cuda.max_memory_reserved(),
                "sahi_durations":result.durations_in_seconds}
    return pred, counters
