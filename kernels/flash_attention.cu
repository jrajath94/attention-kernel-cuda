/*
 * Custom Flash Attention CUDA kernel for non-standard head dimensions.
 *
 * Key insight: Instead of padding head_dim to the next power of 2 (wasting
 * 10-50% of compute), we tile at the exact head dimension using variable
 * block sizes. This eliminates padding waste and improves shared memory
 * utilization.
 *
 * Tiling strategy (variable-block):
 *   - Q tile: [BLOCK_M x HEAD_DIM] loaded into shared memory
 *   - K tile: [BLOCK_N x HEAD_DIM] loaded into shared memory
 *   - V tile: [BLOCK_N x HEAD_DIM] loaded into shared memory
 *   - S tile: [BLOCK_M x BLOCK_N] computed in registers
 *   - Online softmax with running max and sum
 *
 * Template parameters allow compile-time block sizes matching the
 * actual head dimension, enabling compiler optimizations that aren't
 * possible with runtime-variable padding.
 *
 * Copyright (c) 2026 Rajath John. MIT License.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace attention_kernel {

// Warp size for NVIDIA GPUs
constexpr int WARP_SIZE = 32;

// Maximum supported head dimension
constexpr int MAX_HEAD_DIM = 512;

/*
 * Warp-level reduction: max across a warp.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/*
 * Warp-level reduction: sum across a warp.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/*
 * Flash Attention forward kernel with variable block sizes.
 *
 * Template params:
 *   BLOCK_M: tile size for query sequence dim
 *   BLOCK_N: tile size for key/value sequence dim
 *   HEAD_DIM: exact head dimension (NOT padded)
 *   CAUSAL: whether to apply causal masking
 *
 * Grid: (num_heads, batch_size, ceil(seq_len_q / BLOCK_M))
 */
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM, bool CAUSAL>
__global__ void flash_attention_forward_kernel(
    const half* __restrict__ Q,     // [B, S_q, H, D]
    const half* __restrict__ K,     // [B, S_k, H, D]
    const half* __restrict__ V,     // [B, S_k, H, D]
    half* __restrict__ O,           // [B, S_q, H, D]
    float* __restrict__ L,          // [B, H, S_q] logsumexp
    const int seq_len_q,
    const int seq_len_k,
    const int num_heads,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int block_q_idx = blockIdx.z;

    const int q_start = block_q_idx * BLOCK_M;
    if (q_start >= seq_len_q) return;

    // Shared memory layout:
    //   Q tile: BLOCK_M x HEAD_DIM (half)
    //   K tile: BLOCK_N x HEAD_DIM (half)
    //   V tile: BLOCK_N x HEAD_DIM (half)
    extern __shared__ char smem[];
    half* s_Q = reinterpret_cast<half*>(smem);
    half* s_K = s_Q + BLOCK_M * HEAD_DIM;
    half* s_V = s_K + BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;

    // Strides for BHSD layout
    const int q_batch_stride = seq_len_q * num_heads * HEAD_DIM;
    const int k_batch_stride = seq_len_k * num_heads * HEAD_DIM;
    const int head_stride = HEAD_DIM;
    const int seq_stride = num_heads * HEAD_DIM;

    // Load Q tile into shared memory
    const half* q_ptr = Q + batch_idx * q_batch_stride
                          + head_idx * head_stride;
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_pos = q_start + row;
        if (q_pos < seq_len_q) {
            s_Q[i] = q_ptr[q_pos * seq_stride + col];
        } else {
            s_Q[i] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Per-thread accumulators for online softmax
    // Each thread handles a subset of rows in the Q block
    float row_max[BLOCK_M];
    float row_sum[BLOCK_M];
    float acc[BLOCK_M][HEAD_DIM];

    #pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[m][d] = 0.0f;
        }
    }

    // Determine how many K/V blocks to process
    int num_blocks_k = (seq_len_k + BLOCK_N - 1) / BLOCK_N;
    if (CAUSAL) {
        int max_k = q_start + BLOCK_M;  // don't attend beyond query position
        num_blocks_k = min(num_blocks_k, (max_k + BLOCK_N - 1) / BLOCK_N);
    }

    // Iterate over K/V blocks
    for (int bk = 0; bk < num_blocks_k; bk++) {
        int k_start = bk * BLOCK_N;

        // Load K tile
        const half* k_ptr = K + batch_idx * k_batch_stride
                              + head_idx * head_stride;
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int k_pos = k_start + row;
            if (k_pos < seq_len_k) {
                s_K[i] = k_ptr[k_pos * seq_stride + col];
            } else {
                s_K[i] = __float2half(0.0f);
            }
        }

        // Load V tile
        const half* v_ptr = V + batch_idx * k_batch_stride
                              + head_idx * head_stride;
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int v_pos = k_start + row;
            if (v_pos < seq_len_k) {
                s_V[i] = v_ptr[v_pos * seq_stride + col];
            } else {
                s_V[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T for this block pair
        // Each thread computes one or more rows
        for (int m = tid; m < BLOCK_M; m += blockDim.x) {
            float scores[BLOCK_N];
            float block_max = -INFINITY;

            #pragma unroll
            for (int n = 0; n < BLOCK_N; n++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += __half2float(s_Q[m * HEAD_DIM + d])
                         * __half2float(s_K[n * HEAD_DIM + d]);
                }
                dot *= scale;

                // Causal masking
                if (CAUSAL) {
                    int q_pos = q_start + m;
                    int k_pos = k_start + n;
                    if (k_pos > q_pos) {
                        dot = -INFINITY;
                    }
                }

                scores[n] = dot;
                block_max = fmaxf(block_max, dot);
            }

            // Online softmax update
            float new_max = fmaxf(row_max[m], block_max);
            float exp_old = expf(row_max[m] - new_max);
            float exp_new = expf(block_max - new_max);

            // Rescale old accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[m][d] *= exp_old;
            }
            row_sum[m] = row_sum[m] * exp_old;

            // Accumulate new attention-weighted values
            float p_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; n++) {
                float p = expf(scores[n] - block_max) * exp_new;
                p_sum += p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc[m][d] += p * __half2float(s_V[n * HEAD_DIM + d]);
                }
            }
            row_sum[m] += p_sum;
            row_max[m] = new_max;
        }
        __syncthreads();
    }

    // Write output: normalize by row_sum and store
    half* o_ptr = O + batch_idx * q_batch_stride + head_idx * head_stride;
    float* l_ptr = L + batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    for (int m = tid; m < BLOCK_M; m += blockDim.x) {
        int q_pos = q_start + m;
        if (q_pos < seq_len_q) {
            float inv_sum = 1.0f / fmaxf(row_sum[m], 1e-6f);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                o_ptr[q_pos * seq_stride + d] = __float2half(acc[m][d] * inv_sum);
            }
            l_ptr[q_pos] = row_max[m] + logf(fmaxf(row_sum[m], 1e-6f));
        }
    }
}

/*
 * Host-side launcher. Selects template instantiation based on
 * runtime block sizes and head dim.
 */
std::vector<torch::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal,
    int block_m,
    int block_n
) {
    const int batch_size = Q.size(0);
    const int seq_len_q = Q.size(1);
    const int num_heads = Q.size(2);
    const int head_dim = Q.size(3);
    const int seq_len_k = K.size(1);

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({batch_size, num_heads, seq_len_q},
                          Q.options().dtype(torch::kFloat32));

    // Shared memory: Q + K + V tiles
    int smem_bytes = (block_m * head_dim + 2 * block_n * head_dim) * sizeof(half);

    dim3 grid(num_heads, batch_size, (seq_len_q + block_m - 1) / block_m);
    int threads = min(block_m, 256);

    // Dispatch to template specialization
    // In production, use code generation to instantiate all needed combos
    #define LAUNCH_KERNEL(BM, BN, HD, CAUSAL)                                \
        flash_attention_forward_kernel<BM, BN, HD, CAUSAL>                   \
            <<<grid, threads, smem_bytes>>>(                                 \
                Q.data_ptr<at::Half>(),                                      \
                K.data_ptr<at::Half>(),                                      \
                V.data_ptr<at::Half>(),                                      \
                O.data_ptr<at::Half>(),                                      \
                L.data_ptr<float>(),                                         \
                seq_len_q, seq_len_k, num_heads, scale                       \
            );

    // Common non-standard head dims with optimal block sizes
    if (head_dim == 48 && block_m == 64 && block_n == 64) {
        if (causal) { LAUNCH_KERNEL(64, 64, 48, true); }
        else { LAUNCH_KERNEL(64, 64, 48, false); }
    } else if (head_dim == 72 && block_m == 64 && block_n == 64) {
        if (causal) { LAUNCH_KERNEL(64, 64, 72, true); }
        else { LAUNCH_KERNEL(64, 64, 72, false); }
    } else if (head_dim == 96 && block_m == 64 && block_n == 64) {
        if (causal) { LAUNCH_KERNEL(64, 64, 96, true); }
        else { LAUNCH_KERNEL(64, 64, 96, false); }
    } else if (head_dim == 160 && block_m == 32 && block_n == 32) {
        if (causal) { LAUNCH_KERNEL(32, 32, 160, true); }
        else { LAUNCH_KERNEL(32, 32, 160, false); }
    } else if (head_dim == 64 && block_m == 64 && block_n == 64) {
        if (causal) { LAUNCH_KERNEL(64, 64, 64, true); }
        else { LAUNCH_KERNEL(64, 64, 64, false); }
    } else if (head_dim == 128 && block_m == 64 && block_n == 64) {
        if (causal) { LAUNCH_KERNEL(64, 64, 128, true); }
        else { LAUNCH_KERNEL(64, 64, 128, false); }
    } else {
        AT_ERROR("Unsupported head_dim/block combination: head_dim=",
                 head_dim, " block_m=", block_m, " block_n=", block_n,
                 ". Add template instantiation or use reference impl.");
    }

    #undef LAUNCH_KERNEL

    return {O, L};
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward,
          "Flash Attention forward pass for non-standard head dims");
}

}  // namespace attention_kernel
