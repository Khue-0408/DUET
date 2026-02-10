# path: src/models/fusion/__init__.py
from src.models.fusion.evidence_fusion import EvidenceFusion, EvidenceFusionConfig
from src.models.fusion.evidence_fusion_v2 import (
    EvidenceFusionV2,
    EvidenceFusionV2Config,
    EvidenceFusionV2Module,
    verify_stop_gradient,
)

__all__ = [
    'EvidenceFusion',
    'EvidenceFusionConfig',
    'EvidenceFusionV2',
    'EvidenceFusionV2Config',
    'EvidenceFusionV2Module',
    'verify_stop_gradient',
]
