"""Tests for tiling strategy computation."""

from __future__ import annotations

import pytest

from attention_kernel_cuda.exceptions import TilingConfigError
from attention_kernel_cuda.models import AttentionConfig, TilingStrategy
from attention_kernel_cuda.tiling import (
    _compute_shared_memory,
    _next_multiple,
    analyze_tiling_efficiency,
    compute_optimal_tiling,
)


class TestNextMultiple:
    """Tests for _next_multiple utility."""

    @pytest.mark.parametrize(
        "value,multiple,expected",
        [(1, 16, 16), (16, 16, 16), (17, 16, 32), (48, 16, 48), (49, 16, 64)],
    )
    def test_rounds_up_correctly(
        self, value: int, multiple: int, expected: int
    ) -> None:
        """Test rounding up to next multiple."""
        assert _next_multiple(value, multiple) == expected


class TestComputeSharedMemory:
    """Tests for shared memory calculation."""

    def test_standard_dim_64(self) -> None:
        """Test shared memory for standard head_dim=64, block=64."""
        # Q: 64*64*2=8192, K: 64*64*2=8192, V: 64*64*2=8192
        # Softmax: 64*64*4=16384. Total = 40960
        mem = _compute_shared_memory(64, 64, 64, 2)
        assert mem == 40960

    def test_nonstandard_dim_48(self) -> None:
        """Test shared memory for non-standard head_dim=48, block=64."""
        # Q: 64*48*2=6144, K: 64*48*2=6144, V: 64*48*2=6144
        # Softmax: 64*64*4=16384. Total = 34816
        mem = _compute_shared_memory(64, 64, 48, 2)
        assert mem == 34816

    def test_smaller_blocks_use_less_memory(self) -> None:
        """Test that smaller block sizes reduce shared memory."""
        mem_large = _compute_shared_memory(64, 64, 96, 2)
        mem_small = _compute_shared_memory(32, 32, 96, 2)
        assert mem_small < mem_large


class TestComputeOptimalTiling:
    """Tests for the tiling optimizer."""

    @pytest.mark.parametrize("head_dim", [48, 72, 96, 128, 160])
    def test_produces_valid_config(self, head_dim: int) -> None:
        """Test optimal tiling is valid for various head dims."""
        config = AttentionConfig(
            head_dim=head_dim, num_heads=1,
            strategy=TilingStrategy.VARIABLE_BLOCK,
        )
        tiling = compute_optimal_tiling(config)
        assert tiling.validate()
        assert tiling.block_m >= 16
        assert tiling.block_n >= 16

    def test_variable_block_uses_actual_dim(self) -> None:
        """Test variable-block tiling uses exact head_dim, not padded."""
        config = AttentionConfig(
            head_dim=72, num_heads=1,
            strategy=TilingStrategy.VARIABLE_BLOCK,
        )
        tiling = compute_optimal_tiling(config)
        assert tiling.block_k == 72  # Not padded to 128

    def test_standard_tiling_pads(self) -> None:
        """Test standard tiling pads to power of 2."""
        config = AttentionConfig(
            head_dim=72, num_heads=1,
            strategy=TilingStrategy.STANDARD,
        )
        tiling = compute_optimal_tiling(config)
        assert tiling.block_k == 128  # Padded to next power of 2

    def test_variable_avoids_padding(self) -> None:
        """Test variable-block uses exact head_dim, avoiding padding waste."""
        for head_dim in [48, 72, 96, 160]:
            config_var = AttentionConfig(
                head_dim=head_dim, num_heads=1,
                strategy=TilingStrategy.VARIABLE_BLOCK,
            )
            config_std = AttentionConfig(
                head_dim=head_dim, num_heads=1,
                strategy=TilingStrategy.STANDARD,
            )
            tiling_var = compute_optimal_tiling(config_var)
            tiling_std = compute_optimal_tiling(config_std)

            # Variable-block uses exact head_dim (no padding)
            assert tiling_var.block_k == head_dim
            # Standard pads to power of 2
            assert tiling_std.block_k >= head_dim
            assert tiling_std.block_k & (tiling_std.block_k - 1) == 0  # power of 2

    @pytest.mark.parametrize(
        "strategy",
        [TilingStrategy.STANDARD, TilingStrategy.VARIABLE_BLOCK,
         TilingStrategy.SPLIT_K, TilingStrategy.REGISTER_TILED],
    )
    def test_all_strategies_produce_valid_output(
        self, strategy: TilingStrategy
    ) -> None:
        """Test all strategies produce valid configs for a common dim."""
        config = AttentionConfig(
            head_dim=96, num_heads=1, strategy=strategy,
        )
        tiling = compute_optimal_tiling(config)
        assert tiling.validate()


class TestAnalyzeTilingEfficiency:
    """Tests for the efficiency analysis function."""

    def test_returns_all_strategies(self) -> None:
        """Test analysis covers all tiling strategies."""
        config = AttentionConfig(head_dim=72, num_heads=4)
        results = analyze_tiling_efficiency(config)
        assert len(results) == len(TilingStrategy)
        for strategy in TilingStrategy:
            assert strategy.value in results

    def test_standard_dim_efficiency(self) -> None:
        """Test that standard dims have high utilization across strategies."""
        config = AttentionConfig(head_dim=64, num_heads=4)
        results = analyze_tiling_efficiency(config)
        # Standard dims should have utilization = 1.0 (no waste)
        assert results["standard"] > 0.5

    def test_nonstandard_dim_all_strategies_positive(self) -> None:
        """Test all strategies produce positive efficiency for non-standard dims."""
        config = AttentionConfig(head_dim=72, num_heads=4)
        results = analyze_tiling_efficiency(config)
        for strategy_name, efficiency in results.items():
            assert efficiency > 0, f"Strategy {strategy_name} has zero efficiency"
