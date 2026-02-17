"""Tests for data models and configuration."""

from __future__ import annotations

import pytest
import torch

from attention_kernel_cuda.models import (
    AttentionConfig,
    AttentionOutput,
    BenchmarkResult,
    TilingConfig,
    TilingStrategy,
)


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""

    def test_create_valid_config(self) -> None:
        """Test creating a valid config with non-standard head dim."""
        config = AttentionConfig(head_dim=72, num_heads=8)
        assert config.head_dim == 72
        assert config.num_heads == 8
        assert config.causal is False
        assert config.dropout_p == 0.0

    def test_invalid_head_dim_zero(self) -> None:
        """Test that head_dim=0 raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be in"):
            AttentionConfig(head_dim=0, num_heads=8)

    def test_invalid_head_dim_too_large(self) -> None:
        """Test that oversized head_dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be in"):
            AttentionConfig(head_dim=1024, num_heads=8)

    def test_invalid_num_heads(self) -> None:
        """Test that num_heads=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be >= 1"):
            AttentionConfig(head_dim=64, num_heads=0)

    def test_scale_default(self) -> None:
        """Test default softmax scale is 1/sqrt(head_dim)."""
        config = AttentionConfig(head_dim=64, num_heads=8)
        assert abs(config.scale - 64 ** -0.5) < 1e-6

    def test_scale_custom(self) -> None:
        """Test custom softmax scale overrides default."""
        config = AttentionConfig(head_dim=64, num_heads=8, softmax_scale=0.5)
        assert config.scale == 0.5

    @pytest.mark.parametrize(
        "head_dim,expected",
        [(32, True), (64, True), (128, True), (256, True),
         (48, False), (72, False), (96, False), (160, False)],
    )
    def test_is_standard_dim(self, head_dim: int, expected: bool) -> None:
        """Test standard vs non-standard dim detection."""
        config = AttentionConfig(head_dim=head_dim, num_heads=1)
        assert config.is_standard_dim == expected

    @pytest.mark.parametrize(
        "head_dim,expected_padded",
        [(48, 64), (72, 128), (96, 128), (160, 256), (64, 64), (128, 128)],
    )
    def test_padded_head_dim(self, head_dim: int, expected_padded: int) -> None:
        """Test power-of-2 padding calculation."""
        config = AttentionConfig(head_dim=head_dim, num_heads=1)
        assert config.padded_head_dim == expected_padded

    @pytest.mark.parametrize(
        "head_dim,expected_waste",
        [(64, 0.0), (128, 0.0), (48, 0.25), (96, 0.25)],
    )
    def test_padding_waste_ratio(self, head_dim: int, expected_waste: float) -> None:
        """Test padding waste calculation."""
        config = AttentionConfig(head_dim=head_dim, num_heads=1)
        assert abs(config.padding_waste_ratio - expected_waste) < 0.01


class TestTilingConfig:
    """Tests for TilingConfig dataclass."""

    def test_valid_tiling(self) -> None:
        """Test creating a valid tiling config."""
        tiling = TilingConfig(
            block_m=64, block_n=64, block_k=48,
            num_warps=4, num_stages=2,
            shared_memory_bytes=32768,
        )
        assert tiling.validate(max_shared_memory=49152)

    def test_exceeds_shared_memory(self) -> None:
        """Test validation fails when shared memory is exceeded."""
        tiling = TilingConfig(
            block_m=64, block_n=64, block_k=256,
            num_warps=8, num_stages=2,
            shared_memory_bytes=65536,
        )
        assert not tiling.validate(max_shared_memory=49152)

    def test_block_too_small(self) -> None:
        """Test validation fails with undersized blocks."""
        tiling = TilingConfig(
            block_m=8, block_n=8, block_k=48,
            num_warps=4, num_stages=2,
            shared_memory_bytes=4096,
        )
        assert not tiling.validate()


class TestAttentionOutput:
    """Tests for AttentionOutput dataclass."""

    def test_tflops_calculation(self) -> None:
        """Test TFLOPS calculation from kernel time and FLOPs."""
        # 1e9 FLOPs in 1ms = 1e9 / 1e-3 / 1e12 = 1.0 TFLOPS
        output = AttentionOutput(
            output=torch.zeros(1),
            logsumexp=torch.zeros(1),
            tiling_config=TilingConfig(64, 64, 48, 4, 2, 32768),
            kernel_time_ms=1.0,
            flops=1_000_000_000,  # 1 GFLOP
        )
        assert abs(output.tflops - 1.0) < 1e-6

    def test_tflops_zero_time(self) -> None:
        """Test TFLOPS returns 0 when kernel_time_ms is 0."""
        output = AttentionOutput(
            output=torch.zeros(1),
            logsumexp=torch.zeros(1),
            tiling_config=TilingConfig(64, 64, 48, 4, 2, 32768),
            kernel_time_ms=0.0,
            flops=1_000_000_000,
        )
        assert output.tflops == 0.0

    def test_bandwidth_calculation(self) -> None:
        """Test bandwidth calculation."""
        output = AttentionOutput(
            output=torch.zeros(1),
            logsumexp=torch.zeros(1),
            tiling_config=TilingConfig(64, 64, 48, 4, 2, 32768),
            kernel_time_ms=1.0,
            memory_bytes=1_000_000_000,  # 1 GB
        )
        assert abs(output.bandwidth_gb_s - 1000.0) < 1e-3


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_speedup_calculation(self) -> None:
        """Test speedup is baseline / ours."""
        result = BenchmarkResult(
            head_dim=72, seq_len=256, batch_size=2, num_heads=8,
            strategy=TilingStrategy.VARIABLE_BLOCK,
            our_time_ms=5.0, baseline_time_ms=10.0,
        )
        assert abs(result.speedup - 2.0) < 1e-6

    def test_speedup_zero_baseline(self) -> None:
        """Test speedup is 0 when baseline time is 0."""
        result = BenchmarkResult(
            head_dim=72, seq_len=256, batch_size=2, num_heads=8,
            strategy=TilingStrategy.VARIABLE_BLOCK,
            our_time_ms=5.0, baseline_time_ms=0.0,
        )
        assert result.speedup == 0.0
