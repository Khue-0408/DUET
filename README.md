# path: README.md
# DUET: Domain-adaptive Uncertainty-aware Evidential Two-stream Polyp Segmentation (PyTorch)

Repo này implement DUET end-to-end theo 3 stage như paper:
- Stage 1: Boundary-Aware Domain-Adaptive Encoder Pretraining (Sec. 3.1; Eq. (1)(2)(3); Fig.1; Training Details)
- Stage 2: Dual-Stream Feature Extraction (Sec. 3.2)
- Stage 3: Evidential Segmentation + Evidence Fusion (Sec. 3.3; Eq. (4)–(12); Training Details)

Paper-trace được ghi rõ trong docs/paper_trace.md.

## 0) Environment

Tested with Python 3.10+ and PyTorch 2.1+.

Install:
```bash
pip install -r requirements.txt



## 1) Repo Tree
duet/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── duet_default.yaml
│   ├── stage1_pretrain.yaml
│   └── ablations/
│       ├── no_ssl.yaml
│       ├── no_adv.yaml
│       ├── cnn_only.yaml
│       ├── vit_only.yaml
│       ├── early_fusion.yaml
│       ├── late_fusion.yaml
│       ├── no_edl.yaml
│       ├── no_boundary_weight.yaml
│       └── avg_prob_fusion.yaml
├── docs/
│   └── paper_trace.md
├── scripts/
│   ├── prepare_pseudomasks_placeholder.sh
│   ├── run_stage1.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_infer.sh
└── src/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── duet.py
    │   ├── encoders/
    │   │   ├── __init__.py
    │   │   ├── efficientnet_b4.py
    │   │   └── vit_mae.py
    │   ├── freq/
    │   │   ├── __init__.py
    │   │   └── fft_split.py
    │   ├── decoders/
    │   │   ├── __init__.py
    │   │   └── fpn_decoder.py
    │   ├── evidential/
    │   │   ├── __init__.py
    │   │   └── evidence.py
    │   ├── fusion/
    │   │   ├── __init__.py
    │   │   └── evidence_fusion.py
    │   ├── domain/
    │   │   ├── __init__.py
    │   │   ├── grl.py
    │   │   └── domain_discriminator.py
    │   └── losses/
    │       ├── __init__.py
    │       ├── contrastive_infonce.py
    │       ├── dice.py
    │       ├── edl.py
    │       └── region_weighting.py
    ├── data/
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── transforms.py
    │   └── samplers.py
    ├── train/
    │   ├── __init__.py
    │   ├── train_stage1.py
    │   ├── train_duet.py
    │   ├── eval.py
    │   └── infer.py
    └── utils/
        ├── __init__.py
        ├── config.py
        ├── seed.py
        ├── distributed.py
        ├── logger.py
        ├── checkpoint.py
        └── metrics.py
