# 06. Exercises

Use these to convert passive reading into actual CUDA intuition.

## Beginner

1. In `kernels/relu/`, change the block size and measure the effect.
2. In `kernels/elementwise/`, compare scalar and vectorized variants.
3. In `kernels/reduce/`, explain where shared memory is necessary and where warp shuffle is enough.

## Intermediate

1. In `kernels/softmax/`, explain how numerical stability is preserved.
2. In `kernels/mat-transpose/`, identify the naive access pattern and the optimized one.
3. In `kernels/layer-norm/` and `kernels/rms-norm/`, explain the cost of reduction plus normalization.

## Advanced

1. In `kernels/sgemm/`, identify which optimizations are for bandwidth and which are for compute throughput.
2. In `kernels/hgemm/`, compare WMMA and MMA paths.
3. In `kernels/flash-attn/`, explain why Split-Q can beat Split-KV.

## Review Template

For every kernel you study, write four lines:

- mapping: which data each thread or warp owns
- memory: where data is loaded from and reused
- sync: where barriers or warp communication happen
- bottleneck: what you think limits performance most

## Capstone

Build your own mini learning report:

1. choose one simple kernel
2. choose one reduction kernel
3. choose one GEMM kernel
4. choose one attention kernel
5. explain the optimization ladder from naive to optimized

If you can do that clearly, you are already thinking like an infra engineer
instead of only a framework user.
