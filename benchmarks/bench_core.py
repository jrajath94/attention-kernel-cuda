"""Benchmarks comparing variable-block vs padded attention across head dims.

Measures both correctness (max diff, cosine similarity) and performance
(time, TFLOPS, memory) for standard and non-standard head dimensions.
"""

from __future__ import annotations

import logging
import time
from typing import List

import torch

from attention_kernel_cuda.core import flash_attention, _padded_attention_baseline
from attention_kernel_cuda.models import AttentionConfig, BenchmarkResult, TilingStrategy
from attention_kernel_cuda.utils import (
    compare_with_reference,
    compute_attention_flops,
    estimate_memory_usage,
    generate_random_qkv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Benchmark configurations
HEAD_DIMS = [48, 64, 72, 96, 128, 160]
SEQ_LENS = [128, 256, 512]
BATCH_SIZE = 2
NUM_HEADS = 8
NUM_WARMUP = 3
NUM_ITERS = 10


def _time_function(func: object, *args: object, **kwargs: object) -> float:
    """Time a function over multiple iterations, returning median ms.

    Args:
        func: Function to time.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        Median execution time in milliseconds.
    """
    times = []
    for _ in range(NUM_WARMUP):
        func(*args, **kwargs)

    for _ in range(NUM_ITERS):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]


def bench_head_dims() -> List[BenchmarkResult]:
    """Benchmark attention across head dimensions.

    Returns:
        List of benchmark results.
    """
    device = "cpu"
    dtype = torch.float32
    results = []

    logger.info(
        "Benchmarking head_dims=%s, seq_lens=%s, batch=%d, heads=%d",
        HEAD_DIMS, SEQ_LENS, BATCH_SIZE, NUM_HEADS,
    )

    for seq_len in SEQ_LENS:
        for head_dim in HEAD_DIMS:
            q, k, v = generate_random_qkv(
                BATCH_SIZE, seq_len, NUM_HEADS, head_dim,
                dtype=dtype, device=device,
            )
            config = AttentionConfig(head_dim=head_dim, num_heads=NUM_HEADS)

            # Our implementation (variable block)
            our_time = _time_function(
                flash_attention, q, k, v,
                strategy=TilingStrategy.VARIABLE_BLOCK,
            )

            # Baseline (padded to power of 2)
            baseline_time = _time_function(
                _padded_attention_baseline, q, k, v, config,
            )

            # Correctness check
            our_result = flash_attention(q, k, v)
            metrics = compare_with_reference(q, k, v, our_result.output)

            flops = compute_attention_flops(
                BATCH_SIZE, seq_len, seq_len, NUM_HEADS, head_dim,
            )

            result = BenchmarkResult(
                head_dim=head_dim,
                seq_len=seq_len,
                batch_size=BATCH_SIZE,
                num_heads=NUM_HEADS,
                strategy=TilingStrategy.VARIABLE_BLOCK,
                our_time_ms=our_time,
                baseline_time_ms=baseline_time,
                our_tflops=flops / (our_time * 1e-3) / 1e12 if our_time > 0 else 0,
                baseline_tflops=flops / (baseline_time * 1e-3) / 1e12 if baseline_time > 0 else 0,
                max_diff=metrics["max_diff"],
            )
            results.append(result)

            logger.info(
                "dim=%3d seq=%4d | ours=%.3fms baseline=%.3fms | "
                "speedup=%.2fx | max_diff=%.2e | waste=%.0f%%",
                head_dim, seq_len,
                our_time, baseline_time,
                result.speedup,
                metrics["max_diff"],
                config.padding_waste_ratio * 100,
            )

    return results


def bench_memory() -> None:
    """Compare memory usage across algorithms."""
    logger.info("\n--- Memory Usage Comparison ---")
    for head_dim in HEAD_DIMS:
        for seq_len in [256, 1024]:
            mem_flash = estimate_memory_usage(
                BATCH_SIZE, seq_len, NUM_HEADS, head_dim, algorithm="flash",
            )
            mem_padded = estimate_memory_usage(
                BATCH_SIZE, seq_len, NUM_HEADS, head_dim, algorithm="padded",
            )
            mem_standard = estimate_memory_usage(
                BATCH_SIZE, seq_len, NUM_HEADS, head_dim, algorithm="standard",
            )

            logger.info(
                "dim=%3d seq=%4d | flash=%.1fMB padded=%.1fMB standard=%.1fMB "
                "| flash saves %.1f%%",
                head_dim, seq_len,
                mem_flash["total"] / 1e6,
                mem_padded["total"] / 1e6,
                mem_standard["total"] / 1e6,
                (1 - mem_flash["total"] / mem_standard["total"]) * 100,
            )


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results.

    Args:
        results: List of benchmark results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    header = (
        f"{'dim':>4} {'seq':>5} {'ours(ms)':>10} {'base(ms)':>10} "
        f"{'speedup':>8} {'max_diff':>10} {'waste%':>7}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        config = AttentionConfig(head_dim=r.head_dim, num_heads=r.num_heads)
        logger.info(
            "%4d %5d %10.3f %10.3f %8.2fx %10.2e %7.1f",
            r.head_dim, r.seq_len,
            r.our_time_ms, r.baseline_time_ms,
            r.speedup, r.max_diff,
            config.padding_waste_ratio * 100,
        )


if __name__ == "__main__":
    results = bench_head_dims()
    bench_memory()
    print_summary(results)
