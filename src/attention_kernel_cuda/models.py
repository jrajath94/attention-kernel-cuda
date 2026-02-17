"""Data models for attention kernel configuration and results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Shared memory limit for most GPUs (48KB default, 100KB+ on A100/H100)
DEFAULT_SHARED_MEMORY_BYTES = 49152
MAX_HEAD_DIM = 512
MIN_BLOCK_SIZE = 16
MAX_BLOCK_SIZE = 256

# Common non-standard head dimensions found in research models
NON_STANDARD_HEAD_DIMS = (48, 72, 80, 96, 112, 160, 192, 224, 256)

# Standard power-of-2 head dims that Flash Attention already handles well
STANDARD_HEAD_DIMS = (32, 64, 128, 256)


class TilingStrategy(str, Enum):
    """Strategy for tiling attention computation across shared memory.

    Each strategy trades off between shared memory usage, register pressure,
    and arithmetic intensity.
    """

    STANDARD = "standard"
    SPLIT_K = "split_k"
    VARIABLE_BLOCK = "variable_block"
    REGISTER_TILED = "register_tiled"


@dataclass(frozen=True)
class TilingConfig:
    """Concrete tiling parameters for a kernel launch.

    Attributes:
        block_m: Tile size along the query sequence dimension.
        block_n: Tile size along the key/value sequence dimension.
        block_k: Tile size along the head dimension (for split-K).
        num_warps: Number of warps per thread block.
        num_stages: Number of pipeline stages for async copies.
        shared_memory_bytes: Total shared memory required per block.
    """

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    shared_memory_bytes: int

    def validate(self, max_shared_memory: int = DEFAULT_SHARED_MEMORY_BYTES) -> bool:
        """Check if this tiling config fits hardware constraints.

        Args:
            max_shared_memory: Maximum shared memory in bytes.

        Returns:
            True if config is valid for the given hardware.
        """
        if self.shared_memory_bytes > max_shared_memory:
            logger.warning(
                "Tiling requires %d bytes shared memory, GPU has %d",
                self.shared_memory_bytes,
                max_shared_memory,
            )
            return False
        if self.block_m < MIN_BLOCK_SIZE or self.block_n < MIN_BLOCK_SIZE:
            return False
        return True


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for a flash attention kernel launch.

    Attributes:
        head_dim: Dimension of each attention head.
        num_heads: Number of attention heads.
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability (0.0 = no dropout).
        softmax_scale: Scaling factor for softmax. Defaults to 1/sqrt(head_dim).
        strategy: Tiling strategy to use.
        dtype: Tensor data type.
    """

    head_dim: int
    num_heads: int
    causal: bool = False
    dropout_p: float = 0.0
    softmax_scale: Optional[float] = None
    strategy: TilingStrategy = TilingStrategy.VARIABLE_BLOCK
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.head_dim < 1 or self.head_dim > MAX_HEAD_DIM:
            raise ValueError(
                f"head_dim must be in [1, {MAX_HEAD_DIM}], got {self.head_dim}"
            )
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")

    @property
    def scale(self) -> float:
        """Effective softmax scale factor."""
        if self.softmax_scale is not None:
            return self.softmax_scale
        return self.head_dim ** -0.5

    @property
    def is_standard_dim(self) -> bool:
        """Whether this head dim is a standard power-of-2 dim."""
        return self.head_dim in STANDARD_HEAD_DIMS

    @property
    def padded_head_dim(self) -> int:
        """Next power-of-2 head dim (what naive implementations pad to)."""
        power = 1
        while power < self.head_dim:
            power *= 2
        return power

    @property
    def padding_waste_ratio(self) -> float:
        """Fraction of compute wasted by naive power-of-2 padding."""
        if self.is_standard_dim:
            return 0.0
        return 1.0 - (self.head_dim / self.padded_head_dim)


@dataclass
class AttentionOutput:
    """Result of a flash attention computation.

    Attributes:
        output: The attention output tensor [batch, seq_len, num_heads, head_dim].
        logsumexp: Log-sum-exp values for backward pass [batch, num_heads, seq_len].
        tiling_config: The tiling configuration used.
        kernel_time_ms: Kernel execution time in milliseconds (if measured).
        flops: Floating point operations performed.
        memory_bytes: Peak memory usage in bytes.
    """

    output: torch.Tensor
    logsumexp: torch.Tensor
    tiling_config: TilingConfig
    kernel_time_ms: float = 0.0
    flops: int = 0
    memory_bytes: int = 0

    @property
    def tflops(self) -> float:
        """Effective TFLOPS achieved."""
        if self.kernel_time_ms <= 0:
            return 0.0
        return self.flops / (self.kernel_time_ms * 1e-3) / 1e12

    @property
    def bandwidth_gb_s(self) -> float:
        """Effective memory bandwidth in GB/s."""
        if self.kernel_time_ms <= 0:
            return 0.0
        return self.memory_bytes / (self.kernel_time_ms * 1e-3) / 1e9


@dataclass
class BenchmarkResult:
    """Result of benchmarking attention across configurations.

    Attributes:
        head_dim: Head dimension tested.
        seq_len: Sequence length tested.
        batch_size: Batch size tested.
        num_heads: Number of heads tested.
        strategy: Tiling strategy used.
        our_time_ms: Our kernel time in milliseconds.
        baseline_time_ms: Baseline (padded) time in milliseconds.
        our_tflops: Our kernel TFLOPS.
        baseline_tflops: Baseline TFLOPS.
        max_diff: Maximum absolute difference from reference.
        speedup: Speedup over baseline.
    """

    head_dim: int
    seq_len: int
    batch_size: int
    num_heads: int
    strategy: TilingStrategy
    our_time_ms: float
    baseline_time_ms: float
    our_tflops: float = 0.0
    baseline_tflops: float = 0.0
    max_diff: float = 0.0
    speedup: float = field(init=False)

    def __post_init__(self) -> None:
        if self.baseline_time_ms > 0:
            self.speedup = self.baseline_time_ms / self.our_time_ms
        else:
            self.speedup = 0.0
