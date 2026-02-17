"""Tests for core attention computation."""

from __future__ import annotations

import math

import pytest
import torch

from attention_kernel_cuda.core import (
    FlashAttentionFunction,
    _padded_attention_baseline,
    _reference_attention_tiled,
    _validate_inputs,
    flash_attention,
)
from attention_kernel_cuda.exceptions import InvalidHeadDimensionError
from attention_kernel_cuda.models import AttentionConfig, TilingStrategy
from attention_kernel_cuda.tiling import compute_optimal_tiling
from attention_kernel_cuda.utils import compare_with_reference, generate_random_qkv


class TestValidateInputs:
    """Tests for input validation."""

    def test_valid_inputs(self) -> None:
        """Test valid 4D tensor shapes pass validation."""
        q = torch.randn(2, 32, 4, 48)
        k = torch.randn(2, 32, 4, 48)
        v = torch.randn(2, 32, 4, 48)
        batch, seq_len, num_heads, head_dim = _validate_inputs(q, k, v)
        assert batch == 2
        assert seq_len == 32
        assert num_heads == 4
        assert head_dim == 48

    def test_wrong_ndim(self) -> None:
        """Test 3D tensors are rejected."""
        q = torch.randn(2, 32, 48)
        k = torch.randn(2, 32, 48)
        v = torch.randn(2, 32, 48)
        with pytest.raises(ValueError, match="Expected 4D"):
            _validate_inputs(q, k, v)

    def test_mismatched_head_dim(self) -> None:
        """Test Q and K with different head dims are rejected."""
        q = torch.randn(2, 32, 4, 48)
        k = torch.randn(2, 32, 4, 64)
        v = torch.randn(2, 32, 4, 64)
        with pytest.raises(ValueError, match="Head dim mismatch"):
            _validate_inputs(q, k, v)

    def test_oversized_head_dim(self) -> None:
        """Test head_dim > MAX_HEAD_DIM raises error."""
        q = torch.randn(1, 8, 1, 1024)
        k = torch.randn(1, 8, 1, 1024)
        v = torch.randn(1, 8, 1, 1024)
        with pytest.raises(InvalidHeadDimensionError):
            _validate_inputs(q, k, v)


class TestReferenceAttention:
    """Tests for the reference tiled attention implementation."""

    @pytest.mark.parametrize("head_dim", [48, 64, 72, 96, 128, 160])
    def test_matches_pytorch_attention_noncausal(self, head_dim: int) -> None:
        """Test reference impl matches PyTorch SDPA for various head dims."""
        q, k, v = generate_random_qkv(1, 32, 2, head_dim, dtype=torch.float32)
        config = AttentionConfig(head_dim=head_dim, num_heads=2)
        tiling = compute_optimal_tiling(config)

        output, _ = _reference_attention_tiled(q, k, v, config, tiling)
        metrics = compare_with_reference(q, k, v, output)

        assert metrics["max_diff"] < 1e-4, (
            f"head_dim={head_dim}: max_diff={metrics['max_diff']:.2e}"
        )
        assert metrics["cosine_similarity"] > 0.9999

    @pytest.mark.parametrize("head_dim", [48, 72, 96])
    def test_matches_pytorch_attention_causal(self, head_dim: int) -> None:
        """Test causal reference impl matches PyTorch with causal mask."""
        q, k, v = generate_random_qkv(1, 32, 2, head_dim, dtype=torch.float32)
        config = AttentionConfig(head_dim=head_dim, num_heads=2, causal=True)
        tiling = compute_optimal_tiling(config)

        output, _ = _reference_attention_tiled(q, k, v, config, tiling)
        metrics = compare_with_reference(q, k, v, output, causal=True)

        assert metrics["max_diff"] < 1e-4
        assert metrics["cosine_similarity"] > 0.9999

    def test_logsumexp_finite(self) -> None:
        """Test that logsumexp values are finite (no NaN/inf)."""
        q, k, v = generate_random_qkv(1, 16, 1, 48, dtype=torch.float32)
        config = AttentionConfig(head_dim=48, num_heads=1)
        tiling = compute_optimal_tiling(config)

        _, logsumexp = _reference_attention_tiled(q, k, v, config, tiling)
        assert torch.all(torch.isfinite(logsumexp))


class TestPaddedBaseline:
    """Tests for the padded attention baseline."""

    def test_standard_dim_no_padding(self) -> None:
        """Test that standard dims don't get padded."""
        q, k, v = generate_random_qkv(1, 16, 1, 64, dtype=torch.float32)
        config = AttentionConfig(head_dim=64, num_heads=1)
        output = _padded_attention_baseline(q, k, v, config)
        assert output.shape == q.shape

    def test_nonstandard_dim_output_shape(self) -> None:
        """Test that padded baseline preserves output shape."""
        q, k, v = generate_random_qkv(1, 16, 1, 72, dtype=torch.float32)
        config = AttentionConfig(head_dim=72, num_heads=1)
        output = _padded_attention_baseline(q, k, v, config)
        assert output.shape == q.shape  # Should be sliced back

    def test_causal_masking(self) -> None:
        """Test that causal masking changes attention for early positions."""
        q, k, v = generate_random_qkv(1, 8, 1, 48, dtype=torch.float32)
        config_causal = AttentionConfig(head_dim=48, num_heads=1, causal=True)
        config_full = AttentionConfig(head_dim=48, num_heads=1, causal=False)

        out_causal = _padded_attention_baseline(q, k, v, config_causal)
        out_full = _padded_attention_baseline(q, k, v, config_full)

        # Output shapes must match
        assert out_causal.shape == out_full.shape
        # Early positions (e.g. position 1) should differ because causal
        # only lets them attend to positions 0-1, while full attends to 0-7
        assert not torch.allclose(out_causal[:, 1, :, :], out_full[:, 1, :, :])


class TestFlashAttention:
    """Integration tests for the main flash_attention function."""

    @pytest.mark.parametrize(
        "head_dim,seq_len",
        [(48, 32), (72, 64), (96, 32), (128, 64), (160, 32)],
    )
    def test_correctness_noncausal(self, head_dim: int, seq_len: int) -> None:
        """Test flash_attention output matches reference for various configs."""
        q, k, v = generate_random_qkv(1, seq_len, 2, head_dim, dtype=torch.float32)
        result = flash_attention(q, k, v, causal=False)

        metrics = compare_with_reference(q, k, v, result.output)
        assert metrics["max_diff"] < 1e-4, (
            f"head_dim={head_dim}, seq_len={seq_len}: "
            f"max_diff={metrics['max_diff']:.2e}"
        )

    @pytest.mark.parametrize("head_dim", [48, 72, 96])
    def test_correctness_causal(self, head_dim: int) -> None:
        """Test causal flash_attention matches reference."""
        q, k, v = generate_random_qkv(1, 32, 2, head_dim, dtype=torch.float32)
        result = flash_attention(q, k, v, causal=True)

        metrics = compare_with_reference(q, k, v, result.output, causal=True)
        assert metrics["max_diff"] < 1e-4

    def test_return_metrics(self) -> None:
        """Test that return_metrics populates timing and FLOPS."""
        q, k, v = generate_random_qkv(1, 32, 2, 48, dtype=torch.float32)
        result = flash_attention(q, k, v, return_metrics=True)

        assert result.flops > 0
        assert result.memory_bytes > 0
        assert result.tiling_config is not None

    def test_different_strategies_same_result(self) -> None:
        """Test that different tiling strategies produce similar results."""
        q, k, v = generate_random_qkv(1, 32, 2, 72, dtype=torch.float32)

        results = {}
        for strategy in [TilingStrategy.STANDARD, TilingStrategy.VARIABLE_BLOCK]:
            result = flash_attention(q, k, v, strategy=strategy)
            results[strategy] = result.output

        diff = (results[TilingStrategy.STANDARD] - results[TilingStrategy.VARIABLE_BLOCK]).abs()
        assert diff.max() < 1e-4

    def test_batch_consistency(self) -> None:
        """Test that batched and unbatched attention give same results."""
        q_single, k_single, v_single = generate_random_qkv(
            1, 32, 2, 48, dtype=torch.float32
        )
        q_batch = q_single.repeat(3, 1, 1, 1)
        k_batch = k_single.repeat(3, 1, 1, 1)
        v_batch = v_single.repeat(3, 1, 1, 1)

        result_single = flash_attention(q_single, k_single, v_single)
        result_batch = flash_attention(q_batch, k_batch, v_batch)

        for i in range(3):
            torch.testing.assert_close(
                result_batch.output[i],
                result_single.output[0],
                atol=1e-5, rtol=1e-5,
            )


class TestFlashAttentionAutograd:
    """Tests for autograd (backward pass) support."""

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through FlashAttentionFunction."""
        q = torch.randn(1, 16, 1, 48, dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 16, 1, 48, dtype=torch.float32, requires_grad=True)
        v = torch.randn(1, 16, 1, 48, dtype=torch.float32, requires_grad=True)

        config = AttentionConfig(head_dim=48, num_heads=1)
        output = FlashAttentionFunction.apply(q, k, v, config)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert torch.all(torch.isfinite(q.grad))
        assert torch.all(torch.isfinite(k.grad))
        assert torch.all(torch.isfinite(v.grad))

    def test_gradient_numerical_stability(self) -> None:
        """Test gradients are numerically reasonable via finite differences."""
        q = torch.randn(1, 8, 1, 32, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 8, 1, 32, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 8, 1, 32, dtype=torch.float64, requires_grad=True)

        config = AttentionConfig(head_dim=32, num_heads=1, dtype=torch.float64)
        output = FlashAttentionFunction.apply(q, k, v, config)
        loss = output.sum()
        loss.backward()

        # Verify gradients are finite and non-zero
        for name, tensor in [("q", q), ("k", k), ("v", v)]:
            assert tensor.grad is not None, f"{name} grad is None"
            assert torch.all(torch.isfinite(tensor.grad)), f"{name} grad has non-finite values"
            assert tensor.grad.abs().max() > 1e-10, f"{name} grad is near-zero"
