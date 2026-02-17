"""Shared fixtures for attention kernel tests."""

from __future__ import annotations

import pytest
import torch

from attention_kernel_cuda.utils import generate_random_qkv


@pytest.fixture
def device() -> str:
    """Return the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype() -> torch.dtype:
    """Default dtype for testing."""
    return torch.float32


@pytest.fixture
def small_qkv(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small Q, K, V tensors for unit tests (head_dim=48)."""
    return generate_random_qkv(
        batch_size=1, seq_len=32, num_heads=2, head_dim=48,
        dtype=dtype, device="cpu",
    )


@pytest.fixture
def medium_qkv(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Medium Q, K, V tensors for integration tests (head_dim=96)."""
    return generate_random_qkv(
        batch_size=2, seq_len=128, num_heads=4, head_dim=96,
        dtype=dtype, device="cpu",
    )


@pytest.fixture
def standard_dim_qkv(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard head_dim=64 for baseline comparison."""
    return generate_random_qkv(
        batch_size=2, seq_len=64, num_heads=4, head_dim=64,
        dtype=dtype, device="cpu",
    )
