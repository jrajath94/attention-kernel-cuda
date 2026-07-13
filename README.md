# attention-kernel-cuda

> Custom CUDA Flash Attention kernel that eliminates 25-44% padding waste for non-standard head dimensions

[![CI](https://github.com/jrajath94/attention-kernel-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/jrajath94/attention-kernel-cuda/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## The Problem

Flash Attention is the foundation of modern LLM inference. The [original paper](https://arxiv.org/abs/2205.14135) shows 2-4x speedups over standard attention. But there's a hidden cost that nobody talks about.

Flash Attention is optimized for power-of-2 head dimensions: 64, 128, 256. Llama, Falcon, Mistral -- all use these. But when you hit a model that doesn't -- a custom MoE architecture with 72-dim heads, GPT-NeoX with 96-dim heads, StableLM with 48-dim heads -- the standard implementation pads to the next power of 2. A 96-dim head becomes 128. That's 25% wasted FLOPs and memory bandwidth on every single attention operation. On a trillion-token training run, that waste is measured in days of wall-clock time and hundreds of thousands of dollars in compute.

The waste compounds through every layer of the attention computation. Q @ K^T: each row of Q has 96 real elements and 32 padding zeros, but the GPU multiplies the zeros anyway. softmax(scores) @ V: same issue with 32 garbage columns discarded from the output. Memory bandwidth: loading padded Q, K, V from HBM to shared memory moves 25% more data than needed. On an A100 with 2 TB/s HBM bandwidth, that's 500 GB/s wasted on zeros.

I built a CUDA kernel that eliminates this waste entirely through variable-block tiling that matches the exact head dimension.

## What This Project Does

A drop-in Flash Attention replacement that handles any head dimension without padding, registered as a custom PyTorch autograd function with both forward and backward passes.

- **Variable-block tiling** -- asymmetric tile sizes computed at compile time to exactly fit the head dimension
- **Zero padding waste** -- 25-44% compute savings for non-standard dims (48, 72, 96, 160, etc.)
- **Online softmax** -- O(1) extra memory per row, numerically stable incremental computation
- **Recompute backward** -- O(N) memory instead of O(N^2) by recomputing attention weights block-by-block
- **Template-parameterized kernels** -- compile-time unrolling for each head dim eliminates runtime branching

## Architecture

```mermaid
graph TD
    A["Q, K, V Tensors<br/>[B, S, H, D]"] --> B["AttentionConfig<br/>head_dim, causal, strategy"]
    B --> C{"Tiling Engine"}
    C -->|"Standard"| D["Power-of-2 Padding<br/>(baseline)"]
    C -->|"Variable Block"| E["Asymmetric Tiles<br/>Exact head_dim"]
    C -->|"Split-K"| F["Chunked head_dim<br/>for large D"]
    E --> G["Flash Attention<br/>Online Softmax"]
    D --> G
    F --> G
    G --> H["Output [B, S, H, D]<br/>+ logsumexp"]
```

The kernel divides the head dimension into two parts at compile time: the aligned portion (largest multiple of 32 that fits within d) processed with vectorized half8 loads, and the remainder processed with masked scalar loads. For d=96, that's 64 elements as 8 vectorized loads plus 32 elements as 4 vectorized loads. Zero runtime branching -- the template parameter makes the split a compile-time constant that the compiler fully unrolls.

## Quick Start

```bash
git clone https://github.com/jrajath94/attention-kernel-cuda.git
cd attention-kernel-cuda
make install && make run
```

```python
import torch
from attention_kernel_cuda import flash_attention, TilingStrategy

# Non-standard head dim -- no padding waste
q = torch.randn(2, 512, 8, 72)  # head_dim=72
k = torch.randn(2, 512, 8, 72)
v = torch.randn(2, 512, 8, 72)

result = flash_attention(
    q, k, v,
    causal=True,
    strategy=TilingStrategy.VARIABLE_BLOCK,
)
print(result.output.shape)  # [2, 512, 8, 72] -- no padding
```

## Benchmarks

Performance benchmarks can be run locally with `make bench`. The implementation includes both a PyTorch reference implementation (runs on CPU/GPU) and an optional CUDA kernel extension for maximum performance on supported GPUs. Run benchmarks in your environment to measure speedups for your specific hardware and workloads.

## Design Decisions

| Decision                               | Rationale                                                        | Alternative Considered             | Tradeoff                                                        |
| -------------------------------------- | ---------------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------- |
| Variable-block tiling                  | Eliminates 25-44% padding waste with exact head_dim tiles        | Fixed power-of-2 blocks            | More complex tiling engine; compile-time specialization per dim |
| Template CUDA kernels                  | Compile-time unrolling for each head_dim, zero runtime branching | Runtime-variable kernel            | Longer compile time; one kernel binary per head dim             |
| Online softmax (incremental)           | O(1) extra memory per row; numerically stable                    | Two-pass softmax (needs full row)  | Slightly more complex per-block update logic                    |
| Recompute backward pass                | O(N) memory instead of O(N^2) by recomputing attention weights   | Store full attention matrix        | More compute in backward pass; provides memory efficiency       |
| Asymmetric blocks (block_m != block_n) | Better fit for non-square tiles when d doesn't divide 32 cleanly | Square blocks only                 | More edge cases in boundary handling                            |
| Shared memory layout without padding   | Reduces per-block memory usage, improving SM occupancy           | Padded layout (simpler addressing) | Requires conditional masking at block boundaries                |

## How It Works

The core optimization exploits how GPUs access memory. An NVIDIA warp contains 32 threads. Shared memory is organized in 32 banks of 4 bytes each. Power-of-2 dimensions align naturally -- with d=64, each thread processes 2 elements, and memory accesses are perfectly coalesced. With d=96, the access pattern doesn't divide evenly into warps. The standard workaround is padding to 128, wasting 32 elements per row.

This kernel takes a different approach: vectorized loads with conditional masking instead of padding. The head dimension is split into an aligned portion (largest multiple of 32 within d) and a remainder. The aligned portion uses standard vectorized loads (float4 or half8 for maximum throughput). The remainder uses masked loads. Because the split is a template parameter, the compiler unrolls both loops completely. For d=96: 64 elements processed as 8 vectorized loads, and 32 elements processed as 4 vectorized loads.

The shared memory layout also benefits. Standard Flash Attention allocates (BLOCK_SIZE x padded_d), which wastes memory for non-power-of-2 dimensions. This kernel allocates (BLOCK_SIZE x d) exactly, reducing per-block shared memory and allowing more blocks to run concurrently on each streaming multiprocessor.

Not all non-power-of-2 dimensions benefit equally. Dimensions that are multiples of 32 (like 96 and 160) get the full benefit with zero bank conflicts. Dimensions with factors of 7 (like 112 and 224) create shared memory bank conflicts that partially offset savings -- the kernel detects this at compile time and applies minimal padding only within the shared memory layout (not the computation) for these dimensions.

The backward pass uses the same non-padded approach. It recomputes attention weights block-by-block using the saved logsumexp from the forward pass rather than storing the full N x N attention matrix, maintaining O(N) memory.

## Testing

```bash
make test    # 85 tests
make bench   # Performance benchmarks
make lint    # Ruff + mypy
```

## Project Structure

```
attention-kernel-cuda/
├── src/attention_kernel_cuda/
│   ├── core.py          # Flash attention forward/backward with autograd
│   ├── tiling.py        # Optimal tiling computation (key innovation)
│   ├── models.py        # Config, output, benchmark dataclasses
│   ├── utils.py         # QKV generation, correctness checking
│   ├── cli.py           # CLI for benchmarking
│   └── exceptions.py    # Custom error types
├── kernels/
│   └── flash_attention.cu   # CUDA kernel (template-parameterized)
├── tests/               # 85 tests
├── benchmarks/          # Performance measurement
├── examples/            # Quickstart demo
└── docs/                # Architecture + interview prep
```

## What I'd Improve

- **Kernel fusion for QKV projection.** Attention doesn't start with Q, K, V in perfect shape -- they're projections from hidden states. Fusing projection + attention in one kernel would reduce memory traffic further.
- **Grouped query attention (GQA) support.** Increasingly common in modern architectures where K and V have fewer heads than Q. The padding math changes and there are additional opportunities for memory savings.
- **Hopper architecture (H100) support.** The H100 introduces hardware-accelerated asynchronous copies and a Tensor Memory Accelerator that could accelerate the kernel further.

## License

MIT -- Rajath John
