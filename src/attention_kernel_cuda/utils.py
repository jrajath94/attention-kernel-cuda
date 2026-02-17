"""Utility functions for attention kernel operations."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

from attention_kernel_cuda.models import (
    NON_STANDARD_HEAD_DIMS,
    STANDARD_HEAD_DIMS,
    AttentionConfig,
    TilingStrategy,
)
from attention_kernel_cuda.tiling import compute_optimal_tiling

logger = logging.getLogger(__name__)


def generate_random_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors for testing.

    Uses scaled initialization to prevent overflow in fp16 matmuls.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Tensor data type.
        device: Device to create tensors on.

    Returns:
        Tuple of (query, key, value) tensors.
    """
    scale = head_dim ** -0.25  # Keeps QK^T in reasonable range
    shape = (batch_size, seq_len, num_heads, head_dim)

    query = torch.randn(shape, dtype=dtype, device=device) * scale
    key = torch.randn(shape, dtype=dtype, device=device) * scale
    value = torch.randn(shape, dtype=dtype, device=device) * scale

    return query, key, value


def compute_attention_flops(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    head_dim: int,
    causal: bool = False,
) -> int:
    """Compute total FLOPs for an attention operation.

    Accounts for causal masking reducing effective computation by ~50%.

    Args:
        batch_size: Batch size.
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        causal: Whether causal masking is applied.

    Returns:
        Total floating point operations.
    """
    # QK^T: 2 * B * H * S_q * S_k * D
    qk_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    # attn @ V: 2 * B * H * S_q * S_k * D
    av_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    # Softmax: ~5 * B * H * S_q * S_k (exp, sum, div, max, sub)
    softmax_flops = 5 * batch_size * num_heads * seq_len_q * seq_len_k

    total = qk_flops + av_flops + softmax_flops

    if causal:
        # Roughly half the positions are masked
        total = total // 2

    return total


def compare_with_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    causal: bool = False,
) -> Dict[str, float]:
    """Compare output against PyTorch's native attention for correctness.

    Args:
        query: [B, S_q, H, D] query tensor.
        key: [B, S_k, H, D] key tensor.
        value: [B, S_k, H, D] value tensor.
        output: Our attention output to verify.
        causal: Whether causal masking was used.

    Returns:
        Dict with max_diff, mean_diff, and cosine_similarity.
    """
    head_dim = query.shape[-1]
    scale = head_dim ** -0.5

    # Compute reference in fp32 for accuracy
    q = query.float().permute(0, 2, 1, 3)
    k = key.float().permute(0, 2, 1, 3)
    v = value.float().permute(0, 2, 1, 3)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = scores.shape[-1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    reference = torch.matmul(attn, v).permute(0, 2, 1, 3)

    output_f32 = output.float()
    diff = (output_f32 - reference).abs()

    # Cosine similarity per sequence position
    cos_sim = torch.nn.functional.cosine_similarity(
        output_f32.reshape(-1, head_dim),
        reference.reshape(-1, head_dim),
        dim=-1,
    )

    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine_similarity": cos_sim.mean().item(),
    }


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    algorithm: str = "flash",
) -> Dict[str, int]:
    """Estimate peak memory usage for different attention algorithms.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Data type.
        algorithm: One of "flash", "standard", "padded".

    Returns:
        Dict with component memory sizes in bytes.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    tensor_size = batch_size * seq_len * num_heads * head_dim * elem_size

    if algorithm == "standard":
        # Stores full S x S attention matrix
        attn_matrix = batch_size * num_heads * seq_len * seq_len * 4  # fp32
        return {
            "q_k_v": tensor_size * 3,
            "attention_matrix": attn_matrix,
            "output": tensor_size,
            "total": tensor_size * 4 + attn_matrix,
        }
    elif algorithm == "padded":
        padded_dim = 1
        while padded_dim < head_dim:
            padded_dim *= 2
        padded_tensor = batch_size * seq_len * num_heads * padded_dim * elem_size
        attn_matrix = batch_size * num_heads * seq_len * seq_len * 4
        return {
            "q_k_v_padded": padded_tensor * 3,
            "attention_matrix": attn_matrix,
            "output": tensor_size,
            "padding_waste": (padded_tensor - tensor_size) * 3,
            "total": padded_tensor * 3 + attn_matrix + tensor_size,
        }
    else:
        # Flash attention: O(N) memory
        config = AttentionConfig(head_dim=head_dim, num_heads=num_heads)
        tiling = compute_optimal_tiling(config)
        logsumexp_size = batch_size * num_heads * seq_len * 4  # fp32
        return {
            "q_k_v": tensor_size * 3,
            "output": tensor_size,
            "logsumexp": logsumexp_size,
            "shared_memory_per_block": tiling.shared_memory_bytes,
            "total": tensor_size * 4 + logsumexp_size,
        }


def list_supported_head_dims() -> List[Dict[str, object]]:
    """List all supported head dimensions with their tiling metadata.

    Returns:
        List of dicts with head_dim info and optimal tiling parameters.
    """
    results = []
    for dim in sorted(set(STANDARD_HEAD_DIMS) | set(NON_STANDARD_HEAD_DIMS)):
        config = AttentionConfig(head_dim=dim, num_heads=1)
        tiling = compute_optimal_tiling(config)
        results.append({
            "head_dim": dim,
            "is_standard": dim in STANDARD_HEAD_DIMS,
            "padded_dim": config.padded_head_dim,
            "padding_waste": f"{config.padding_waste_ratio:.1%}",
            "block_m": tiling.block_m,
            "block_n": tiling.block_n,
            "shared_memory": tiling.shared_memory_bytes,
            "strategy": config.strategy.value,
        })
    return results
