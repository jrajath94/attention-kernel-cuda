# X Thread: attention-kernel-cuda

**Tweet 1:**
Flash Attention wastes 25-44% of compute on non-standard head dims.

GPT-NeoX (dim=96): 25% waste. StableLM (dim=48): 25%. Most MoE models (dim=72): 43.8%.

I built a CUDA kernel that eliminates this.

Code: github.com/jrajath94/attention-kernel-cuda

---

**Tweet 2:**
The problem: Flash Attention tiles assume power-of-2 head dims.

head_dim=72? Padded to 128. That's 56 zeros per attention head, per token, per layer.

At 70B params, that's millions of wasted FLOPs per second.

---

**Tweet 3:**
My approach: variable-block tiling.

Instead of padding 72 -> 128, I compute asymmetric block sizes that exactly fit head_dim=72.

The tiling engine analyzes shared memory constraints and maximizes arithmetic intensity without rounding to power-of-2.

---

**Tweet 4:**
The non-obvious insight: asymmetric blocks.

Standard Flash Attention uses block_m = block_n = 64.

For head_dim=72, optimal is block_m=96, block_n=48. This fits the GPU's shared memory better and keeps higher occupancy.

No one talks about this because standard dims don't need it.

---

**Tweet 5:**
Results:

- Correctness: max diff < 1e-7 vs reference
- Cosine similarity: 1.000000
- Padding eliminated for dims 48, 72, 96, 160
- Full backward pass with O(N) memory
- 85 tests, 76% coverage, CI/CD ready

---

**Tweet 6:**
Star it if you work with non-standard head dims.

What CUDA kernel should I build next?

github.com/jrajath94/attention-kernel-cuda

#CUDA #FlashAttention #MachineLearning #OpenSource #BuildInPublic #DeepLearning #Transformers
