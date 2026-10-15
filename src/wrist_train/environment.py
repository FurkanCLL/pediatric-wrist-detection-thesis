"""Verify the CUDA build and obtain traceable official pretrained assets."""

import gc
import importlib.metadata
import json
import platform
import subprocess
import urllib.request

from .common import OUTPUT, configure_runtime, require_cuda, save_json
from wrist_prepare.common import digest, protected_write


def verify_environment():
    configure_runtime()
    import torch
    import torchvision
    device = require_cuda()
    torch.manual_seed(42)
    properties = torch.cuda.get_device_properties(device)
    baseline = torch.cuda.memory_allocated()
    # Exercise convolution, backward, an optimizer update and the torchvision CUDA extension.
    layer = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
    x = torch.randn(2, 3, 64, 64, device=device)
    before = layer.weight.detach().clone()
    loss = layer(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in layer.parameters())
    optimizer.step()
    assert not torch.equal(before, layer.weight)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    assert kept.tolist() == [0]
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del layer, optimizer, x, before, loss, boxes, kept
    gc.collect()
    torch.cuda.empty_cache()
    receipt = {"python": platform.python_version(), "platform": platform.system(),
               "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "Pillow", "PyYAML")},
               "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
               "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "cuda_available": True, "gpu": properties.name, "vram_bytes": properties.total_memory,
               "compute_capability": list(torch.cuda.get_device_capability()),
               "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
               "forward_backward_optimizer_nms": "passed", "peak_allocated_bytes": peak,
               "allocated_after_cleanup_bytes": torch.cuda.memory_allocated(), "baseline_allocated_bytes": baseline}
    assert receipt["allocated_after_cleanup_bytes"] == baseline
    save_json(OUTPUT / "environment/cuda_verification.json", receipt)
    from ultralytics import YOLO
    pretrained = YOLO(str(OUTPUT / "weights/yolov8s.pt")).model
    assert len(pretrained.names) == 80 and pretrained.names[0] == "person" and pretrained.names[79] == "toothbrush"
    save_json(OUTPUT / "environment/pretrained_verification.json", {
        "asset": "yolov8s.pt", "original_classes": 80, "task": "detect",
        "original_dataset": str(pretrained.args.get("data")).replace("\\", "/").rsplit("/", 1)[-1],
        "parameters": sum(p.numel() for p in pretrained.parameters()),
        "sha256": digest((OUTPUT / "weights/yolov8s.pt").read_bytes()),
        "grazpedwri_trained_initialization": False})
    print(json.dumps(receipt, indent=2))


def fetch_weights():
    configure_runtime()
    receipts = []
    # The auxiliary nano asset is used only by Ultralytics' native AMP compatibility check.
    for name in ("yolov8s.pt", "yolo26n.pt"):
        url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
        path = OUTPUT / "weights" / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            assert len(content) > 1000000
            protected_write(path, content)
        receipts.append({"asset": name, "url": url, "sha256": digest(path.read_bytes()),
                         "bytes": path.stat().st_size,
                         "role": "COCO pretrained training initialization" if name == "yolov8s.pt" else "framework AMP check only"})
    save_json(OUTPUT / "environment/pretrained_assets.json", receipts)
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify", "weights"])
    arguments = parser.parse_args()
    verify_environment() if arguments.action == "verify" else fetch_weights()
