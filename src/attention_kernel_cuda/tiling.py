"""Tiling strategy computation for non-standard head dimensions.

The core insight: standard Flash Attention tiles assume head_dim is a
power of 2, wasting register space and shared memory on padding. This
module computes optimal tile sizes that exactly fit arbitrary head dims,
eliminating waste and enabling better occupancy.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

from attention_kernel_cuda.exceptions import TilingConfigError
from attention_kernel_cuda.models import (
    DEFAULT_SHARED_MEMORY_BYTES,
    MAX_BLOCK_SIZE,
    MIN_BLOCK_SIZE,
    AttentionConfig,
    TilingConfig,
    TilingStrategy,
)

logger = logging.getLogger(__name__)

# Bytes per element for common dtypes
DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

# Warp size on NVIDIA GPUs
WARP_SIZE = 32


def _next_multiple(value: int, multiple: int) -> int:
    """Round up to the next multiple.

    Args:
        value: Value to round up.
        multiple: Multiple to round to.

    Returns:
        Smallest multiple of `multiple` >= `value`.
    """
    return ((value + multiple - 1) // multiple) * multiple


def _compute_shared_memory(
    block_m: int,
    block_n: int,
    head_dim: int,
    bytes_per_elem: int,
) -> int:
    """Calculate shared memory needed for Q, K, V tiles.

    Layout in shared memory:
      - Q tile: block_m x head_dim
      - K tile: block_n x head_dim
      - V tile: block_n x head_dim
      - Scratch for softmax: block_m x block_n (float32 always)

    Args:
        block_m: Query block size.
        block_n: Key/Value block size.
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element (2 for fp16, 4 for fp32).

    Returns:
        Total shared memory in bytes.
    """
    q_bytes = block_m * head_dim * bytes_per_elem
    k_bytes = block_n * head_dim * bytes_per_elem
    v_bytes = block_n * head_dim * bytes_per_elem
    # Softmax scratch always in fp32 for numerical stability
    softmax_bytes = block_m * block_n * 4
    return q_bytes + k_bytes + v_bytes + softmax_bytes


def _find_divisors_in_range(
    head_dim: int,
    min_block: int,
    max_block: int,
) -> List[int]:
    """Find values in [min_block, max_block] that evenly divide head_dim.

    For non-standard dims, we also include values where head_dim is a
    multiple of the block size in the K dimension, enabling clean tiling.

    Args:
        head_dim: The head dimension to tile.
        min_block: Minimum block size.
        max_block: Maximum block size.

    Returns:
        Sorted list of valid block sizes.
    """
    candidates = set()
    for block in range(min_block, max_block + 1):
        if head_dim % block == 0 or block % 16 == 0:
            candidates.add(block)
    return sorted(candidates)


def compute_optimal_tiling(
    config: AttentionConfig,
    max_shared_memory: int = DEFAULT_SHARED_MEMORY_BYTES,
) -> TilingConfig:
    """Compute the best tiling configuration for a given attention config.

    For non-standard head dimensions, this finds tile sizes that:
    1. Avoid padding waste (don't round to power-of-2)
    2. Maximize arithmetic intensity (large tiles)
    3. Fit within shared memory constraints
    4. Align to warp boundaries for coalesced access

    Args:
        config: Attention configuration with head dim, dtype, etc.
        max_shared_memory: GPU shared memory limit in bytes.

    Returns:
        Optimal tiling configuration.

    Raises:
        TilingConfigError: If no valid tiling exists.
    """
    strategy = config.strategy
    head_dim = config.head_dim
    bytes_per_elem = DTYPE_BYTES.get(str(config.dtype).split(".")[-1], 2)

    if strategy == TilingStrategy.STANDARD:
        return _compute_standard_tiling(head_dim, bytes_per_elem, max_shared_memory)
    elif strategy == TilingStrategy.SPLIT_K:
        return _compute_split_k_tiling(head_dim, bytes_per_elem, max_shared_memory)
    elif strategy == TilingStrategy.VARIABLE_BLOCK:
        return _compute_variable_block_tiling(
            head_dim, bytes_per_elem, max_shared_memory
        )
    elif strategy == TilingStrategy.REGISTER_TILED:
        return _compute_register_tiled(head_dim, bytes_per_elem, max_shared_memory)
    else:
        raise TilingConfigError(f"Unknown strategy: {strategy}")


def _compute_standard_tiling(
    head_dim: int,
    bytes_per_elem: int,
    max_shared_memory: int,
) -> TilingConfig:
    """Standard Flash Attention tiling (power-of-2 padded).

    Args:
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element.
        max_shared_memory: Shared memory limit.

    Returns:
        Tiling config with standard (padded) approach.
    """
    padded_dim = 1
    while padded_dim < head_dim:
        padded_dim *= 2

    # Start with block_m=block_n=64, scale down if needed
    for block_size in [128, 64, 32, 16]:
        shared_mem = _compute_shared_memory(
            block_size, block_size, padded_dim, bytes_per_elem
        )
        if shared_mem <= max_shared_memory:
            num_warps = max(4, block_size * block_size // (WARP_SIZE * 4))
            return TilingConfig(
                block_m=block_size,
                block_n=block_size,
                block_k=padded_dim,
                num_warps=min(num_warps, 8),
                num_stages=2,
                shared_memory_bytes=shared_mem,
            )

    raise TilingConfigError(
        f"Cannot fit standard tiling for head_dim={head_dim} "
        f"in {max_shared_memory} bytes"
    )


def _compute_variable_block_tiling(
    head_dim: int,
    bytes_per_elem: int,
    max_shared_memory: int,
) -> TilingConfig:
    """Variable-block tiling that adapts to non-standard head dims.

    Instead of padding to power-of-2, we use the actual head_dim and
    find the largest block_m x block_n that fits in shared memory.
    This is the key innovation: asymmetric blocks optimized per-dim.

    Args:
        head_dim: Actual head dimension (not padded).
        bytes_per_elem: Bytes per element.
        max_shared_memory: Shared memory limit.

    Returns:
        Optimized tiling config without padding.
    """
    best_config = None
    best_arithmetic_intensity = 0.0

    # Try asymmetric block sizes - not required to be equal
    for block_m in range(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE + 1, 16):
        for block_n in range(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE + 1, 16):
            shared_mem = _compute_shared_memory(
                block_m, block_n, head_dim, bytes_per_elem
            )
            if shared_mem > max_shared_memory:
                continue

            # Arithmetic intensity: compute / memory
            # More compute per byte loaded = better GPU utilization
            compute_ops = 2 * block_m * block_n * head_dim  # QK^T
            compute_ops += 2 * block_m * head_dim * block_n  # attn @ V
            memory_ops = shared_mem
            intensity = compute_ops / max(memory_ops, 1)

            if intensity > best_arithmetic_intensity:
                best_arithmetic_intensity = intensity
                num_warps = max(4, (block_m * block_n) // (WARP_SIZE * 4))
                best_config = TilingConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=head_dim,
                    num_warps=min(num_warps, 8),
                    num_stages=2 if shared_mem < max_shared_memory * 0.6 else 1,
                    shared_memory_bytes=shared_mem,
                )

    if best_config is None:
        raise TilingConfigError(
            f"No valid variable-block tiling for head_dim={head_dim}"
        )

    logger.debug(
        "Variable-block tiling: block_m=%d, block_n=%d, head_dim=%d, "
        "shared_mem=%d, intensity=%.2f",
        best_config.block_m,
        best_config.block_n,
        head_dim,
        best_config.shared_memory_bytes,
        best_arithmetic_intensity,
    )
    return best_config


def _compute_split_k_tiling(
    head_dim: int,
    bytes_per_elem: int,
    max_shared_memory: int,
) -> TilingConfig:
    """Split-K tiling for very large head dimensions.

    Splits the head dimension into chunks, computing partial attention
    in each chunk, then reducing. Useful when head_dim > 128.

    Args:
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element.
        max_shared_memory: Shared memory limit.

    Returns:
        Split-K tiling config.
    """
    # Find optimal split factor
    best_config = None
    best_efficiency = 0.0

    for num_splits in [2, 4, 8]:
        if head_dim % num_splits != 0:
            continue
        chunk_dim = head_dim // num_splits

        for block_size in [128, 96, 64, 48, 32]:
            shared_mem = _compute_shared_memory(
                block_size, block_size, chunk_dim, bytes_per_elem
            )
            # Need extra space for partial results reduction
            reduction_mem = block_size * chunk_dim * 4  # fp32
            total_mem = shared_mem + reduction_mem

            if total_mem <= max_shared_memory:
                occupancy = 1.0 - (total_mem / max_shared_memory)
                efficiency = occupancy * block_size * block_size
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    num_warps = max(4, block_size * block_size // (WARP_SIZE * 4))
                    best_config = TilingConfig(
                        block_m=block_size,
                        block_n=block_size,
                        block_k=chunk_dim,
                        num_warps=min(num_warps, 8),
                        num_stages=2,
                        shared_memory_bytes=total_mem,
                    )

    if best_config is None:
        # Fall back to variable block
        return _compute_variable_block_tiling(
            head_dim, bytes_per_elem, max_shared_memory
        )

    return best_config


def _compute_register_tiled(
    head_dim: int,
    bytes_per_elem: int,
    max_shared_memory: int,
) -> TilingConfig:
    """Register-tiled strategy maximizing register reuse.

    Uses smaller shared memory tiles but keeps more data in registers.
    Best for medium head dims (48-96) where register file is sufficient.

    Args:
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element.
        max_shared_memory: Shared memory limit.

    Returns:
        Register-optimized tiling config.
    """
    # Smaller blocks to leave room in shared memory for double-buffering
    target_registers_per_thread = 128  # Conservative for high occupancy
    elem_per_thread = target_registers_per_thread // 2  # 2 regs per fp16 pair

    for block_m in [64, 48, 32]:
        for block_n in [64, 48, 32]:
            shared_mem = _compute_shared_memory(
                block_m, block_n, head_dim, bytes_per_elem
            )
            # Double-buffered: need 2x shared memory for pipelining
            total_mem = shared_mem * 2

            if total_mem <= max_shared_memory:
                num_warps = max(4, (block_m * block_n) // (WARP_SIZE * 4))
                return TilingConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=head_dim,
                    num_warps=min(num_warps, 8),
                    num_stages=3,  # More stages for pipelining
                    shared_memory_bytes=total_mem,
                )

    raise TilingConfigError(
        f"No valid register-tiled config for head_dim={head_dim}"
    )


def analyze_tiling_efficiency(config: AttentionConfig) -> Dict[str, float]:
    """Compare tiling efficiency across strategies for a given config.

    Useful for benchmarking and selecting the best strategy.

    Args:
        config: Attention configuration to analyze.

    Returns:
        Dict mapping strategy name to efficiency metrics.
    """
    results: Dict[str, float] = {}
    bytes_per_elem = DTYPE_BYTES.get(str(config.dtype).split(".")[-1], 2)

    for strategy in TilingStrategy:
        try:
            test_config = AttentionConfig(
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                causal=config.causal,
                strategy=strategy,
                dtype=config.dtype,
            )
            tiling = compute_optimal_tiling(test_config)

            # Compute utilization ratio: useful compute / total compute
            useful_ops = (
                2 * tiling.block_m * tiling.block_n * config.head_dim
            )
            padded_dim = 1
            while padded_dim < config.head_dim:
                padded_dim *= 2
            padded_ops = 2 * tiling.block_m * tiling.block_n * padded_dim
            utilization = useful_ops / max(padded_ops, 1)

            # Shared memory efficiency
            mem_efficiency = 1.0 - (
                tiling.shared_memory_bytes / DEFAULT_SHARED_MEMORY_BYTES
            )

            results[strategy.value] = utilization * (1.0 + mem_efficiency)
        except TilingConfigError:
            results[strategy.value] = 0.0

    return results
