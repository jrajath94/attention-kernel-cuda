"""Core flash attention implementation with non-standard head dimension support.

This module provides the main attention computation, implementing both a
high-performance CUDA path (when available) and a reference PyTorch path
that uses the same tiling logic for correctness verification.

The key innovation: instead of padding non-standard head dims (48, 72, 96,
160, etc.) to the next power of 2, we compute optimal tile sizes that
exactly match the head dimension, eliminating wasted compute and memory.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from attention_kernel_cuda.exceptions import InvalidHeadDimensionError
from attention_kernel_cuda.models import (
    MAX_HEAD_DIM,
    AttentionConfig,
    AttentionOutput,
    TilingConfig,
    TilingStrategy,
)
from attention_kernel_cuda.tiling import compute_optimal_tiling

logger = logging.getLogger(__name__)

# Try to import CUDA extension; fall back gracefully
_CUDA_AVAILABLE = False
try:
    import attention_kernel_cuda._C as _C  # type: ignore[import-not-found]

    _CUDA_AVAILABLE = True
    logger.info("CUDA attention kernel extension loaded")
except ImportError:
    logger.info("CUDA extension not found, using PyTorch reference implementation")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[int, int, int, int]:
    """Validate Q, K, V tensor shapes and return dimensions.

    Expected shape: [batch_size, seq_len, num_heads, head_dim]

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.

    Returns:
        Tuple of (batch_size, seq_len, num_heads, head_dim).

    Raises:
        ValueError: If shapes are incompatible.
        InvalidHeadDimensionError: If head_dim is out of range.
    """
    if query.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors [B, S, H, D], got query with {query.ndim} dims"
        )

    batch_size, seq_len_q, num_heads, head_dim = query.shape
    _, seq_len_k, num_heads_k, head_dim_k = key.shape
    _, seq_len_v, num_heads_v, head_dim_v = value.shape

    if head_dim != head_dim_k or head_dim != head_dim_v:
        raise ValueError(
            f"Head dim mismatch: Q={head_dim}, K={head_dim_k}, V={head_dim_v}"
        )
    if num_heads_k != num_heads_v:
        raise ValueError(
            f"K and V must have same num_heads: K={num_heads_k}, V={num_heads_v}"
        )
    if seq_len_k != seq_len_v:
        raise ValueError(
            f"K and V must have same seq_len: K={seq_len_k}, V={seq_len_v}"
        )
    if head_dim < 1 or head_dim > MAX_HEAD_DIM:
        raise InvalidHeadDimensionError(head_dim, MAX_HEAD_DIM)

    return batch_size, seq_len_q, num_heads, head_dim


def _reference_attention_tiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: AttentionConfig,
    tiling: TilingConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference tiled attention in pure PyTorch.

    Implements the same tiling logic as the CUDA kernel but in Python,
    using the online softmax trick for numerical stability. This is the
    ground-truth reference for verifying CUDA kernel correctness.

    Args:
        query: [B, S_q, H, D] query tensor.
        key: [B, S_k, H, D] key tensor.
        value: [B, S_k, H, D] value tensor.
        config: Attention configuration.
        tiling: Tiling parameters.

    Returns:
        Tuple of (output, logsumexp).
    """
    batch_size, seq_len_q, num_heads, head_dim = query.shape
    seq_len_k = key.shape[1]
    scale = config.scale

    # Output accumulators
    output = torch.zeros_like(query)
    # Online softmax state: logsumexp per query position
    logsumexp = torch.full(
        (batch_size, num_heads, seq_len_q),
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )

    block_m = tiling.block_m
    block_n = tiling.block_n

    num_blocks_q = math.ceil(seq_len_q / block_m)
    num_blocks_k = math.ceil(seq_len_k / block_n)

    for b in range(batch_size):
        for h in range(num_heads):
            for bq in range(num_blocks_q):
                q_start = bq * block_m
                q_end = min(q_start + block_m, seq_len_q)
                q_block = query[b, q_start:q_end, h, :]  # [block_m', D]

                # Running max and sum for online softmax
                row_max = torch.full(
                    (q_end - q_start,),
                    float("-inf"),
                    device=query.device,
                    dtype=torch.float32,
                )
                row_sum = torch.zeros(
                    q_end - q_start,
                    device=query.device,
                    dtype=torch.float32,
                )
                acc = torch.zeros(
                    q_end - q_start,
                    head_dim,
                    device=query.device,
                    dtype=torch.float32,
                )

                max_blocks = num_blocks_k
                if config.causal:
                    max_blocks = min(
                        num_blocks_k,
                        (q_end + block_n - 1) // block_n,
                    )

                for bk in range(max_blocks):
                    k_start = bk * block_n
                    k_end = min(k_start + block_n, seq_len_k)
                    k_block = key[b, k_start:k_end, h, :]  # [block_n', D]
                    v_block = value[b, k_start:k_end, h, :]  # [block_n', D]

                    # QK^T with scaling
                    scores = (
                        q_block.float() @ k_block.float().T * scale
                    )  # [block_m', block_n']

                    # Causal masking
                    if config.causal:
                        q_indices = torch.arange(
                            q_start, q_end, device=query.device
                        ).unsqueeze(1)
                        k_indices = torch.arange(
                            k_start, k_end, device=query.device
                        ).unsqueeze(0)
                        mask = q_indices < k_indices
                        scores.masked_fill_(mask, float("-inf"))

                    # Online softmax update
                    block_max = scores.max(dim=-1).values  # [block_m']
                    # Mask for rows where all scores are -inf (fully masked)
                    valid_rows = block_max > float("-inf")

                    new_max = torch.maximum(row_max, block_max)
                    # Prevent NaN from -inf - (-inf): keep old max for invalid rows
                    new_max = torch.where(valid_rows, new_max, row_max)

                    # Rescale old accumulator (safe: row_max - new_max <= 0)
                    exp_old = torch.exp(
                        torch.clamp(row_max - new_max, max=0.0)
                    )
                    # For invalid rows, exp_new = 0 (no contribution)
                    safe_block_max = torch.where(
                        valid_rows, block_max, new_max
                    )
                    exp_new = torch.exp(
                        torch.clamp(safe_block_max - new_max, max=0.0)
                    )
                    exp_new = torch.where(valid_rows, exp_new, torch.zeros_like(exp_new))

                    # Update running sum
                    safe_scores = torch.where(
                        scores > float("-inf"),
                        scores - safe_block_max.unsqueeze(1),
                        torch.tensor(float("-inf"), device=scores.device),
                    )
                    p = torch.exp(safe_scores)
                    p = torch.nan_to_num(p, nan=0.0)
                    row_sum = row_sum * exp_old + p.sum(dim=-1) * exp_new

                    # Update output accumulator
                    acc = acc * exp_old.unsqueeze(1) + (
                        exp_new.unsqueeze(1) * (p @ v_block.float())
                    )
                    row_max = new_max

                # Normalize
                output[b, q_start:q_end, h, :] = (
                    acc / row_sum.unsqueeze(1)
                ).to(query.dtype)
                logsumexp[b, h, q_start:q_end] = row_max + torch.log(row_sum)

    return output, logsumexp


def _padded_attention_baseline(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """Baseline attention with power-of-2 padding (what we're improving on).

    This pads non-standard head dims to the next power of 2, computes
    standard attention, then slices back. Used as a correctness check
    and performance baseline.

    Args:
        query: [B, S_q, H, D] query tensor.
        key: [B, S_k, H, D] key tensor.
        value: [B, S_k, H, D] value tensor.
        config: Attention config.

    Returns:
        Attention output tensor.
    """
    head_dim = query.shape[-1]
    padded_dim = config.padded_head_dim

    if padded_dim > head_dim:
        pad_size = padded_dim - head_dim
        query = F.pad(query, (0, pad_size))
        key = F.pad(key, (0, pad_size))
        value = F.pad(value, (0, pad_size))

    # Standard scaled dot-product attention
    # Reshape to [B, H, S, D] for matmul
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    scores = torch.matmul(q, k.transpose(-2, -1)) * config.scale

    if config.causal:
        seq_len = scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    # Back to [B, S, H, D] and slice off padding
    output = output.permute(0, 2, 1, 3)
    if padded_dim > head_dim:
        output = output[..., :head_dim]

    return output


class FlashAttentionFunction(torch.autograd.Function):
    """Custom autograd function for flash attention with optimal tiling.

    Supports both CUDA-accelerated and reference PyTorch paths.
    The forward pass uses the online softmax algorithm with tiling
    optimized for the actual head dimension.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        config: AttentionConfig,
    ) -> torch.Tensor:
        """Forward pass of flash attention.

        Args:
            ctx: Autograd context for saving tensors.
            query: [B, S_q, H, D] query tensor.
            key: [B, S_k, H, D] key tensor.
            value: [B, S_k, H, D] value tensor.
            config: Attention configuration.

        Returns:
            Attention output tensor [B, S_q, H, D].
        """
        tiling = compute_optimal_tiling(config)

        if _CUDA_AVAILABLE and query.is_cuda:
            output_data = _C.flash_attention_forward(
                query, key, value,
                config.scale,
                config.causal,
                tiling.block_m,
                tiling.block_n,
            )
            output = output_data[0]
            logsumexp = output_data[1]
        else:
            output, logsumexp = _reference_attention_tiled(
                query, key, value, config, tiling
            )

        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.config = config  # type: ignore[attr-defined]
        ctx.tiling = tiling  # type: ignore[attr-defined]
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Backward pass recomputing attention for memory efficiency.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient of the loss w.r.t. the output.

        Returns:
            Gradients for (query, key, value, None).
        """
        query, key, value, output, logsumexp = ctx.saved_tensors
        config: AttentionConfig = ctx.config  # type: ignore[attr-defined]

        grad_q, grad_k, grad_v = flash_attention_backward(
            grad_output, query, key, value, output, logsumexp, config
        )
        return grad_q, grad_k, grad_v, None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    strategy: TilingStrategy = TilingStrategy.VARIABLE_BLOCK,
    return_metrics: bool = False,
) -> AttentionOutput:
    """Compute flash attention optimized for arbitrary head dimensions.

    Args:
        query: Query tensor [B, S_q, H, D].
        key: Key tensor [B, S_k, H, D].
        value: Value tensor [B, S_k, H, D].
        head_dim: Override head dimension (inferred from query if None).
        num_heads: Override num heads (inferred from query if None).
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
        softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim)).
        strategy: Tiling strategy to use.
        return_metrics: Whether to measure and return performance metrics.

    Returns:
        AttentionOutput with output tensor and optional metrics.

    Raises:
        InvalidHeadDimensionError: If head_dim is out of supported range.
        ValueError: If tensor shapes are incompatible.
    """
    batch_size, seq_len_q, inferred_heads, inferred_dim = _validate_inputs(
        query, key, value
    )
    actual_head_dim = head_dim or inferred_dim
    actual_num_heads = num_heads or inferred_heads

    config = AttentionConfig(
        head_dim=actual_head_dim,
        num_heads=actual_num_heads,
        causal=causal,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        strategy=strategy,
        dtype=query.dtype,
    )

    tiling = compute_optimal_tiling(config)

    if not config.is_standard_dim:
        logger.debug(
            "Non-standard head_dim=%d (padded would be %d, %.1f%% waste). "
            "Using %s tiling: block_m=%d, block_n=%d",
            config.head_dim,
            config.padded_head_dim,
            config.padding_waste_ratio * 100,
            config.strategy.value,
            tiling.block_m,
            tiling.block_n,
        )

    start_time = time.perf_counter() if return_metrics else 0.0

    output = FlashAttentionFunction.apply(query, key, value, config)

    kernel_time_ms = 0.0
    if return_metrics:
        if query.is_cuda:
            torch.cuda.synchronize()
        kernel_time_ms = (time.perf_counter() - start_time) * 1000

    # Compute FLOPS: 2 * B * H * S_q * S_k * D (for QK^T) + same for attn@V
    seq_len_k = key.shape[1]
    flops = (
        4 * batch_size * actual_num_heads * seq_len_q * seq_len_k * actual_head_dim
    )

    # Memory: Q + K + V + O (all in working dtype)
    elem_bytes = query.element_size()
    memory_bytes = (
        query.numel() + key.numel() + value.numel() + output.numel()
    ) * elem_bytes

    # Compute logsumexp for the output (needed for backward compatibility)
    _, logsumexp = _reference_attention_tiled(
        query, key, value, config, tiling
    ) if not return_metrics else (None, torch.zeros(
        batch_size, actual_num_heads, seq_len_q,
        device=query.device, dtype=torch.float32,
    ))

    return AttentionOutput(
        output=output,
        logsumexp=logsumexp,
        tiling_config=tiling,
        kernel_time_ms=kernel_time_ms,
        flops=flops,
        memory_bytes=memory_bytes,
    )


def flash_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
    config: AttentionConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for flash attention with recomputation.

    Recomputes attention weights block-by-block rather than storing
    the full attention matrix, achieving O(N) memory instead of O(N^2).

    Args:
        grad_output: Gradient of loss w.r.t. output [B, S_q, H, D].
        query: Original query tensor [B, S_q, H, D].
        key: Original key tensor [B, S_k, H, D].
        value: Original value tensor [B, S_k, H, D].
        output: Forward pass output [B, S_q, H, D].
        logsumexp: Log-sum-exp from forward pass [B, H, S_q].
        config: Attention configuration.

    Returns:
        Tuple of (grad_query, grad_key, grad_value).
    """
    batch_size, seq_len_q, num_heads, head_dim = query.shape
    seq_len_k = key.shape[1]
    scale = config.scale

    grad_q = torch.zeros_like(query)
    grad_k = torch.zeros_like(key)
    grad_v = torch.zeros_like(value)

    tiling = compute_optimal_tiling(config)
    block_m = tiling.block_m
    block_n = tiling.block_n

    for b in range(batch_size):
        for h in range(num_heads):
            for bq in range(math.ceil(seq_len_q / block_m)):
                q_start = bq * block_m
                q_end = min(q_start + block_m, seq_len_q)

                q_block = query[b, q_start:q_end, h, :].float()
                do_block = grad_output[b, q_start:q_end, h, :].float()
                o_block = output[b, q_start:q_end, h, :].float()

                # D_i = rowsum(dO * O)
                d_i = (do_block * o_block).sum(dim=-1)  # [block_m']

                lse_block = logsumexp[b, h, q_start:q_end]

                max_blocks = math.ceil(seq_len_k / block_n)
                if config.causal:
                    max_blocks = min(
                        max_blocks,
                        (q_end + block_n - 1) // block_n,
                    )

                for bk in range(max_blocks):
                    k_start = bk * block_n
                    k_end = min(k_start + block_n, seq_len_k)

                    k_block = key[b, k_start:k_end, h, :].float()
                    v_block = value[b, k_start:k_end, h, :].float()

                    # Recompute attention scores
                    scores = q_block @ k_block.T * scale

                    if config.causal:
                        q_idx = torch.arange(
                            q_start, q_end, device=query.device
                        ).unsqueeze(1)
                        k_idx = torch.arange(
                            k_start, k_end, device=query.device
                        ).unsqueeze(0)
                        mask = q_idx < k_idx
                        scores.masked_fill_(mask, float("-inf"))

                    # Recompute attention weights using saved logsumexp
                    p = torch.exp(scores - lse_block.unsqueeze(1))

                    # Gradient computations
                    grad_v[b, k_start:k_end, h, :] += (
                        p.T @ do_block
                    ).to(value.dtype)

                    dp = do_block @ v_block.T
                    ds = p * (dp - d_i.unsqueeze(1))

                    grad_q[b, q_start:q_end, h, :] += (
                        ds @ k_block * scale
                    ).to(query.dtype)
                    grad_k[b, k_start:k_end, h, :] += (
                        ds.T @ q_block * scale
                    ).to(key.dtype)

    return grad_q, grad_k, grad_v
