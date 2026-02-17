"""CLI interface for attention kernel benchmarking and analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

import torch

from attention_kernel_cuda.core import flash_attention
from attention_kernel_cuda.models import AttentionConfig, TilingStrategy
from attention_kernel_cuda.tiling import analyze_tiling_efficiency
from attention_kernel_cuda.utils import (
    compare_with_reference,
    generate_random_qkv,
    list_supported_head_dims,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_HEADS = 8


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity.

    Args:
        verbose: Whether to enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run attention benchmark across head dimensions.

    Args:
        args: Parsed CLI arguments.
    """
    head_dims = [int(d) for d in args.head_dims.split(",")]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info(
        "Benchmarking head_dims=%s on %s (dtype=%s)",
        head_dims, device, dtype,
    )

    for head_dim in head_dims:
        query, key, value = generate_random_qkv(
            args.batch_size, args.seq_len, args.num_heads, head_dim,
            dtype=dtype, device=device,
        )

        result = flash_attention(
            query, key, value,
            causal=args.causal,
            strategy=TilingStrategy(args.strategy),
            return_metrics=True,
        )

        metrics = compare_with_reference(
            query, key, value, result.output, causal=args.causal,
        )

        config = AttentionConfig(head_dim=head_dim, num_heads=args.num_heads)
        logger.info(
            "head_dim=%3d | padded=%3d | waste=%5.1f%% | "
            "time=%.3fms | max_diff=%.2e | cos_sim=%.6f",
            head_dim,
            config.padded_head_dim,
            config.padding_waste_ratio * 100,
            result.kernel_time_ms,
            metrics["max_diff"],
            metrics["cosine_similarity"],
        )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze tiling efficiency for a head dimension.

    Args:
        args: Parsed CLI arguments.
    """
    config = AttentionConfig(
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        dtype=torch.float16,
    )

    logger.info("Analyzing head_dim=%d", args.head_dim)
    logger.info(
        "  Standard dim: %s | Padded to: %d | Waste: %.1f%%",
        config.is_standard_dim,
        config.padded_head_dim,
        config.padding_waste_ratio * 100,
    )

    efficiencies = analyze_tiling_efficiency(config)
    logger.info("  Strategy efficiencies:")
    for strategy_name, efficiency in sorted(
        efficiencies.items(), key=lambda x: -x[1]
    ):
        logger.info("    %-20s %.4f", strategy_name, efficiency)


def cmd_list_dims(args: argparse.Namespace) -> None:
    """List all supported head dimensions.

    Args:
        args: Parsed CLI arguments.
    """
    dims_info = list_supported_head_dims()
    header = (
        f"{'dim':>4} {'std?':>5} {'padded':>7} {'waste':>7} "
        f"{'Bm':>4} {'Bn':>4} {'smem':>6} {'strategy':>15}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for info in dims_info:
        logger.info(
            "%4d %5s %7d %7s %4d %4d %6d %15s",
            info["head_dim"],
            "yes" if info["is_standard"] else "no",
            info["padded_dim"],
            info["padding_waste"],
            info["block_m"],
            info["block_n"],
            info["shared_memory"],
            info["strategy"],
        )


def main(argv: List[str] | None = None) -> None:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv).
    """
    parser = argparse.ArgumentParser(
        prog="attention-kernel",
        description="Flash Attention kernels for non-standard head dimensions",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "--head-dims", default="48,64,72,96,128,160",
        help="Comma-separated head dimensions to benchmark",
    )
    bench_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    bench_parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    bench_parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    bench_parser.add_argument("--causal", action="store_true")
    bench_parser.add_argument(
        "--strategy", default="variable_block",
        choices=[s.value for s in TilingStrategy],
    )
    bench_parser.add_argument("--cpu", action="store_true")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze tiling")
    analyze_parser.add_argument("--head-dim", type=int, required=True)
    analyze_parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)

    # List dims command
    subparsers.add_parser("list-dims", help="List supported head dims")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "list-dims":
        cmd_list_dims(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
