# 04. GEMM and Attention

GEMM is the center of modern AI infra.
Attention kernels are built on the same optimization language: tiling, reuse,
register pressure management, and memory staging.

## Why Learn GEMM First

If you cannot reason about GEMM, you will struggle with attention.
The core ideas are the same:

- partition the problem into tiles
- stage data into shared memory
- keep tensor cores busy
- reduce redundant global memory traffic

## HGEMM Study Path

Start here:

```bash
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

Then study the progression inside `kernels/hgemm/`:

- `naive/`: correctness-first baseline
- `wmma/`: Tensor Core programming via WMMA
- `mma/`: lower-level MMA path with more control
- `cutlass/`: CuTe/CUTLASS-style tiling and swizzle ideas
- `cublas/`: external baseline for comparison

Then branch into the adjacent labs:

- `kernels/ws-hgemm/`: warp-specialization flavored HGEMM experiments
- `kernels/cutlass/cute/`: tiny CuTe-focused examples for layout intuition
- `HGEMM/`: standalone upstream extraction of the HGEMM track

Questions to ask:

- what tile sizes are chosen for block, warp, and instruction
- where shared memory buffering begins
- where register pressure rises
- how swizzle changes cache or bank-conflict behavior

## FlashAttention Study Path

Once HGEMM concepts are comfortable, move to:

```bash
python3 scripts/run_example.py kernels/flash-attn/flash_attn_mma.py --minimal-build --B 1 --H 8 --N 1024 --D 64 --iters 1 --warmup 0 --sdpa
```

Even if `flash-attn` Python package is not installed, the script now skips the
official FA comparison instead of crashing immediately. The `--minimal-build`
switch compiles only a teaching subset of kernels for faster smoke testing.

Focus on:

- Split-Q versus Split-KV
- shared KV or shared QKV
- SRAM complexity
- head dimension limits
- correctness checks against SDPA or FA

After that, continue with the larger-headdim follow-up:

- `ffpa-attn/`: the next-stage attention study for `D > 256`
- compare why FFPA uses finer-grained tiling than the FA-2 style kernels in `kernels/flash-attn/`

## Practical Reading Order

1. `kernels/sgemm/`
2. `kernels/hgemm/naive/`
3. `kernels/hgemm/wmma/`
4. `kernels/hgemm/mma/basic/`
5. `kernels/hgemm/mma/swizzle/`
6. `kernels/flash-attn/mma/basic/`
7. `kernels/flash-attn/mma/swizzle/`
8. `ffpa-attn/csrc/cuffpa/`

## What “Good” Looks Like

At this level, good code is not just fast.
It also has:

- explicit data movement
- predictable tiling
- benchmarkable outputs
- a clear correctness reference
- enough structure that you can ablate one optimization at a time
