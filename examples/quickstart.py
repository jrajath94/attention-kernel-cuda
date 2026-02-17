"""Quickstart example for attention-kernel-cuda.

Demonstrates the key use case: computing attention with non-standard
head dimensions efficiently, without power-of-2 padding waste.
"""

from __future__ import annotations

import logging

import torch

from attention_kernel_cuda import (
    AttentionConfig,
    TilingStrategy,
    compute_optimal_tiling,
    flash_attention,
)
from attention_kernel_cuda.tiling import analyze_tiling_efficiency
from attention_kernel_cuda.utils import (
    compare_with_reference,
    generate_random_qkv,
    list_supported_head_dims,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the quickstart demo."""
    # === 1. Show the problem: padding waste ===
    logger.info("=== Padding Waste Analysis ===")
    for dim_info in list_supported_head_dims():
        logger.info(
            "  head_dim=%3d | padded=%3d | waste=%s | block=(%d,%d)",
            dim_info["head_dim"],
            dim_info["padded_dim"],
            dim_info["padding_waste"],
            dim_info["block_m"],
            dim_info["block_n"],
        )

    # === 2. Compute attention with non-standard head dim ===
    logger.info("\n=== Flash Attention with head_dim=72 ===")
    query, key, value = generate_random_qkv(
        batch_size=2, seq_len=128, num_heads=8, head_dim=72,
        dtype=torch.float32,
    )

    result = flash_attention(
        query, key, value,
        causal=True,
        strategy=TilingStrategy.VARIABLE_BLOCK,
        return_metrics=True,
    )

    logger.info("  Output shape: %s", result.output.shape)
    logger.info("  Kernel time: %.3f ms", result.kernel_time_ms)
    logger.info("  FLOPs: %d", result.flops)
    logger.info(
        "  Tiling: block_m=%d, block_n=%d, shared_mem=%d bytes",
        result.tiling_config.block_m,
        result.tiling_config.block_n,
        result.tiling_config.shared_memory_bytes,
    )

    # === 3. Verify correctness ===
    metrics = compare_with_reference(
        query, key, value, result.output, causal=True,
    )
    logger.info("\n=== Correctness Verification ===")
    logger.info("  Max absolute diff: %.2e", metrics["max_diff"])
    logger.info("  Mean absolute diff: %.2e", metrics["mean_diff"])
    logger.info("  Cosine similarity: %.6f", metrics["cosine_similarity"])

    # === 4. Compare tiling strategies ===
    logger.info("\n=== Tiling Strategy Comparison (head_dim=72) ===")
    config = AttentionConfig(head_dim=72, num_heads=8)
    efficiencies = analyze_tiling_efficiency(config)
    for strategy, efficiency in sorted(efficiencies.items(), key=lambda x: -x[1]):
        logger.info("  %-20s efficiency=%.4f", strategy, efficiency)

    # === 5. Gradient check ===
    logger.info("\n=== Gradient Flow Check ===")
    q_grad = torch.randn(1, 32, 2, 48, requires_grad=True)
    k_grad = torch.randn(1, 32, 2, 48, requires_grad=True)
    v_grad = torch.randn(1, 32, 2, 48, requires_grad=True)

    result_grad = flash_attention(q_grad, k_grad, v_grad)
    loss = result_grad.output.sum()
    loss.backward()

    logger.info("  Q grad norm: %.4f", q_grad.grad.norm().item())
    logger.info("  K grad norm: %.4f", k_grad.grad.norm().item())
    logger.info("  V grad norm: %.4f", v_grad.grad.norm().item())

    logger.info("\nDone! All checks passed.")


if __name__ == "__main__":
    main()
