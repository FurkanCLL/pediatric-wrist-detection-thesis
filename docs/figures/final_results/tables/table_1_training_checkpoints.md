# Table 1 training checkpoints

Training and selected checkpoints. Native metrics are the logged values at epochs selected by native eight-class validation mAP50–95; they are distinct from the final evaluator. Steady accumulation excludes the native warmup ramp. Full checkpoint identities are in the source manifest.

| Setting | Standard | Rarity-only |
| --- | --- | --- |
| Model | YOLOv8s | YOLOv8s |
| Training epochs | 80 | 80 |
| Input size | 768 | 768 |
| Physical batch / steady accumulation | 6 / 4 | 6 / 4 |
| Nominal effective batch | 24 | 24 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 0.001 | 0.001 |
| Weight decay | 0.0005 | 0.0005 |
| Training seed | 42 | 42 |
| Selected epoch | 22 | 26 |
| Native mAP50 at selected epoch (%) | 59.908 | 61.319 |
| Native mAP50–95 at selected epoch (%) | 33.270 | 33.669 |
| Successful optimizer updates | 47416 | 47415 |
| Skipped optimizer updates | 17 | 18 |
