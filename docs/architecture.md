# Architecture: attention-kernel-cuda

## System Overview

```
                    ┌─────────────────┐
                    │   User API      │
                    │ flash_attention()│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  AttentionConfig │
                    │  (head_dim, etc) │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │    Tiling Strategy Selector   │
              │  compute_optimal_tiling()     │
              ├──────────┬──────────┬────────┤
              │ Standard │ Variable │ Split-K│
              │ (padded) │  Block   │        │
              └──────────┴────┬─────┴────────┘
                              │
              ┌───────────────▼───────────────┐
              │    FlashAttentionFunction      │
              │  (torch.autograd.Function)     │
              ├───────────────┬───────────────┤
              │  CUDA Kernel  │  PyTorch Ref  │
              │  (if GPU)     │  (fallback)   │
              └───────────────┴───────────────┘
```

## Component Responsibilities

### Tiling Engine (`tiling.py`)

The core innovation. Computes optimal tile sizes for arbitrary head dimensions:

- **Standard**: Pads to power-of-2 (baseline for comparison)
- **Variable Block**: Asymmetric tiles exactly matching head_dim
- **Split-K**: Chunks head_dim for very large dimensions (>128)
- **Register Tiled**: Maximizes register reuse for medium dims (48-96)

### Forward Pass (`core.py`)

Online softmax with tiled Q/K/V:

1. Load Q tile (block_m x head_dim) into shared memory
2. Iterate over K/V tiles (block_n x head_dim)
3. Compute QK^T scores, apply causal mask
4. Online softmax: track running max and sum
5. Accumulate attention-weighted values

### Backward Pass (`core.py`)

Recomputation-based (Flash Attention's key memory insight):

- Does NOT store the S x S attention matrix
- Recomputes attention weights block-by-block using saved logsumexp
- O(N) memory instead of O(N^2)

### CUDA Kernel (`kernels/flash_attention.cu`)

Template-parameterized for compile-time optimization:

- Block sizes as template params → compiler unrolling
- Warp-level reductions for softmax
- Shared memory layout: Q + K + V tiles + softmax scratch

## Data Flow

```
Input: Q, K, V [B, S, H, D]
  │
  ├─ Validate shapes
  ├─ Compute AttentionConfig (scale, padding waste, etc.)
  ├─ Select optimal TilingConfig
  │
  ├─ Forward: for each (batch, head, q_block):
  │    ├─ Load Q tile
  │    ├─ for each k_block:
  │    │    ├─ Load K, V tiles
  │    │    ├─ Compute S = Q @ K^T * scale
  │    │    ├─ Causal mask (if enabled)
  │    │    ├─ Online softmax update
  │    │    └─ Accumulate output
  │    └─ Normalize output
  │
  └─ Output: O [B, S, H, D], logsumexp [B, H, S]
```
