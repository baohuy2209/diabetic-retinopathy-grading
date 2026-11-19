"""Multi-scale dual-branch feature extraction streams.

Includes the global context stream (ResNet-50 backbone + Non-Local + Quadrant Tokens)
and the local patch stream (EfficientNet-B0 backbone + CBAM + Dual Attention Pooling).
"""

from models.streams.global_context import GlobalBranch, GlobalContextStream
from models.streams.local_mil import GatedAttentionPool, LocalMILBranch

__all__ = [
    "GlobalBranch",
    "GlobalContextStream",
    "GatedAttentionPool",
    "LocalMILBranch",
]
