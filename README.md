# detector-distillation-analysis

> **What actually matters in detection distillation?**  
> A systematic ablation of knowledge distillation loss components for object detection — from large transformer teacher to fast deployable student.

---

## Motivation

Large detection transformers (DINO, DETR) achieve strong accuracy but are too slow for edge deployment. Knowledge distillation compresses them into faster students — but the literature is noisy about which loss components drive the accuracy recovery.

Feature mimicking? Logit matching? Relation distillation? This project ablates each component in isolation on a fixed teacher/student pair, producing a clear picture of what to actually implement in production.

---

## Problem Setup

| Role | Model | Params | Latency (A100) |
|---|---|---|---|
| Teacher | DINO (ResNet-50) | ~47M | — |
| Student | FCOS (ResNet-18) | ~17M | — |
| Dataset | COCO 2017 + BDD100K | | |
| Export | ONNX (student only) | | |

*Latency numbers to be filled in during experiments.*

---

## Distillation Components Ablated

| Component | What it transfers | Loss type |
|---|---|---|
| Logit matching | Class probability distributions | KL divergence |
| Feature mimicking | Intermediate backbone features | L2 / cosine |
| Relation distillation | Pairwise feature correlations | L2 on Gram matrix |
| Soft NMS targets | Teacher's post-NMS box scores | BCE |
| Combined (full) | All of the above | Weighted sum |

Each component is ablated in isolation against the supervised student baseline, then combined incrementally to identify interactions.

---

## Repository Structure

```
detector-distillation-analysis/
├── data/
│   ├── coco/                         # symlink or download instructions
│   └── bdd100k/
├── src/
│   ├── models/
│   │   ├── teacher/                  # DINO teacher (frozen)
│   │   └── student/                  # FCOS student
│   ├── distillation/
│   │   ├── losses/
│   │   │   ├── logit_kd.py           # KL divergence on class logits
│   │   │   ├── feature_mimic.py      # L2/cosine feature matching
│   │   │   ├── relation_kd.py        # Gram matrix relation distillation
│   │   │   └── soft_nms_targets.py   # teacher box score matching
│   │   └── trainer.py                # distillation training loop
│   ├── export/
│   │   ├── export_onnx.py            # student → ONNX
│   │   └── benchmark_latency.py      # latency profiling
│   └── eval/
│       ├── benchmark.py              # mAP evaluation
│       └── tradeoff_curve.py         # latency vs accuracy plot
├── configs/
│   └── distill_fcos.yaml
├── notebooks/
│   └── tradeoff_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Results (preliminary)

### COCO mAP — ablation

| Configuration | mAP | mAP@50 | mAP@75 |
|---|---|---|---|
| Student (no distillation) | — | — | — |
| + Logit matching | — | — | — |
| + Feature mimicking | — | — | — |
| + Relation distillation | — | — | — |
| + Soft NMS targets | — | — | — |
| Full distillation | — | — | — |
| Teacher (upper bound) | — | — | — |

### Latency vs accuracy (student, ONNX)

| Backend | Batch size | Latency (ms) | mAP |
|---|---|---|---|
| PyTorch (fp32) | 1 | — | — |
| ONNX (fp32) | 1 | — | — |
| ONNX (fp16) | 1 | — | — |

*Results to be filled in as experiments complete.*

---

## Key Questions This Project Answers

1. Which distillation loss component contributes most to mAP recovery?
2. Do components interact positively, or does combining them add noise?
3. What is the actual latency/accuracy tradeoff after ONNX export?
4. Is feature mimicking worth the implementation complexity vs logit matching alone?

---

## Setup

```bash
git clone https://github.com/Nickvanhurck/detector-distillation-analysis
cd detector-distillation-analysis
pip install -r requirements.txt
```

Download COCO 2017 from [cocodataset.org](https://cocodataset.org/) and symlink to `data/coco/`.

```bash
# Train student baseline (no distillation)
python src/distillation/trainer.py --distill none

# Ablate single component
python src/distillation/trainer.py --distill logit_kd

# Full distillation
python src/distillation/trainer.py --distill full

# Export to ONNX and benchmark latency
python src/export/export_onnx.py --checkpoint runs/full/best.pth
python src/export/benchmark_latency.py --model exports/student.onnx
```

---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy
opencv-python
pycocotools
onnx
onnxruntime
```

---

## References

- [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605) — Zhang et al., ICLR 2023
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015
- [Mimicking Very Efficient Network for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf) — Li et al., CVPR 2017
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) — Tian et al., ICCV 2019

---

## Status

- [ ] Teacher setup (DINO, frozen weights)
- [ ] Student baseline training (no distillation)
- [ ] Logit matching loss
- [ ] Feature mimicking loss
- [ ] Relation distillation loss
- [ ] Soft NMS target loss
- [ ] Full ablation results
- [ ] ONNX export + latency benchmarking
- [ ] Tradeoff curve notebook

---

*Part of a 3-project portfolio on robust perception for autonomous driving.*  
*Related: [tent-ad-domain-shift](https://github.com/Nickvanhurck/tent-ad-domain-shift) · [ssl-detection-bdd](https://github.com/Nickvanhurck/ssl-detection-bdd)*
