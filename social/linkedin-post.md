# LinkedIn Post: attention-kernel-cuda

I just open-sourced attention-kernel-cuda -- here's why 43.8% compute waste in Flash Attention matters.

Every ML team using transformers relies on Flash Attention for efficient training. But here's what most engineers don't realize: if your head dimension isn't a power of 2, Flash Attention silently pads it. head_dim=72 becomes 128. head_dim=96 becomes 128. head_dim=160 becomes 256. That's 12-44% of every attention computation doing arithmetic on zeros.

I built a CUDA kernel system that uses variable-block tiling to match the exact head dimension. Instead of padding 72 to 128 and wasting shared memory, the tiling engine computes asymmetric block sizes (e.g., block_m=96, block_n=48) that maximize GPU utilization without rounding. The kernel is template-parameterized in CUDA C++ for compile-time optimization, with a full PyTorch reference implementation for correctness verification.

The results: numerically equivalent output (max diff < 1e-7, cosine similarity = 1.0) with zero padding waste. The project includes a complete CUDA kernel, online softmax with NaN-safe causal masking, autograd backward pass, and 85 passing tests. It's ready for any transformer architecture that doesn't use power-of-2 head dimensions -- which is more common than you'd think (GPT-NeoX, StableLM, many MoE models).

If you're working with custom transformer architectures or building ML infrastructure, check it out and let me know what you think.

-> GitHub: github.com/jrajath94/attention-kernel-cuda

#AI #MachineLearning #CUDA #DeepLearning #SoftwareEngineering #OpenSource #FlashAttention #GPUProgramming
