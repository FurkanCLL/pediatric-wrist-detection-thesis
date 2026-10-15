"""Exercise the installed Ultralytics detection data path without a model."""

from collections import Counter
import os
from pathlib import Path
import random

import cv2
import numpy as np

from .common import CLASSES, SPLITS, protected_write


def verify_loader(records, prepared, state):
    # Keep framework settings/font caches local and avoid optional network downloads.
    settings = state.resolve() / "framework_settings"
    user_config = settings / "Ultralytics"
    user_config.mkdir(parents=True, exist_ok=True)
    matplotlib_config = state.resolve() / "matplotlib_settings"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    font = Path(os.environ.get("WINDIR", "")) / "Fonts" / "arial.ttf"
    if font.is_file():
        protected_write(user_config / "Arial.ttf", font.read_bytes())
    import torch
    import torchvision
    import ultralytics
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.patches import imread

    random.seed(42)
    np.random.seed(42)
    data = check_det_dataset(prepared / "dataset.yaml", autodownload=False)
    if data["nc"] != 8 or [data["names"][i] for i in range(8)] != list(CLASSES) or data["channels"] != 3:
        raise ValueError("Ultralytics did not read the approved eight-class, three-channel configuration")
    receipt = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "opencv": cv2.__version__,
               "torch_cuda_build": torch.version.cuda, "model_created": False,
               "weights_loaded": False, "training_run": False, "augmentation": False,
               "technical_check_imgsz": 320, "splits": {}}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        expected = Counter(CLASSES[o["class_id"]] for r in subset for o in r["boxes"])
        dataset = YOLODataset(img_path=data[split], data=data, task="detect", imgsz=320,
                              augment=False, cache=False, rect=False, batch_size=2, stride=32)
        actual = Counter(CLASSES[int(c)] for label in dataset.labels for c in label["cls"].ravel())
        if len(dataset) != len(subset) or actual != expected:
            raise ValueError("The training loader dropped images or changed class/object counts")
        loaded_stems = {Path(path).stem for path in dataset.im_files}
        if loaded_stems != {r["filestem"] for r in subset}:
            raise ValueError("The training loader changed split membership")
        # IMREAD_COLOR must replicate grayscale values into three native-size channels.
        path = dataset.im_files[0]
        loaded = imread(path, flags=dataset.cv2_flag)
        plane = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if loaded.shape != (*plane.shape, 3) or not np.array_equal(loaded, np.repeat(plane[..., None], 3, axis=2)):
            raise ValueError("Unexpected grayscale-to-three-channel loading behavior")
        batch = dataset.collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
        tensor = batch["img"]
        if tensor.ndim != 4 or tensor.shape[1:] != (3, 320, 320) or tensor.dtype != torch.uint8:
            raise ValueError("Unexpected detection batch shape or dtype")
        if not torch.equal(tensor[:, 0], tensor[:, 1]) or not torch.equal(tensor[:, 0], tensor[:, 2]):
            raise ValueError("Prepared grayscale channels differ after loader transforms")
        receipt["splits"][split] = {"images_scanned": len(dataset), "objects_loaded": sum(actual.values()),
                                    "class_counts": {c: actual[c] for c in CLASSES},
                                    "cv2_read_flag": int(dataset.cv2_flag), "configured_channels": dataset.channels,
                                    "native_three_channel_replication_verified": True,
                                    "batch_shape": list(tensor.shape), "batch_dtype": str(tensor.dtype)}
    return receipt
