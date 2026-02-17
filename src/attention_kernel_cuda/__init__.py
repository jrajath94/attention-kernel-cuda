"""Attention Kernel CUDA: Flash Attention for non-standard head dimensions.

Custom CUDA-optimized attention kernels that efficiently handle arbitrary
head dimensions (48, 72, 96, 160, etc.) without the padding waste and
performance cliffs of standard Flash Attention implementations.
"""

from attention_kernel_cuda.core import (
    flash_attention,
    flash_attention_backward,
    FlashAttentionFunction,
)
from attention_kernel_cuda.models import AttentionConfig, AttentionOutput, TilingStrategy
from attention_kernel_cuda.tiling import compute_optimal_tiling

__version__ = "0.1.0"

__all__ = [
    "flash_attention",
    "flash_attention_backward",
    "FlashAttentionFunction",
    "AttentionConfig",
    "AttentionOutput",
    "TilingStrategy",
    "compute_optimal_tiling",
]
