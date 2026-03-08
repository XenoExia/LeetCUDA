# 05. Debugging and Profiling

CUDA learning is mostly a loop of:

1. write or read a kernel
2. verify correctness
3. profile
4. change one thing
5. measure again

## Correctness First

Always keep a trusted baseline:

- PyTorch op
- cuBLAS / cuDNN / SDPA
- a naive kernel

Do not start tuning until the numerical result is stable.

## Useful Repo Commands

```bash
python3 scripts/doctor.py
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
cd kernels/nvidia-nsight && nvcc -arch=sm_120 -o relu.bin --generate-line-info -g relu.cu
```

## What To Look For In Build Logs

- `ptxas info`: register count, shared memory usage, spill count
- `Killed` / exit `137`: host-memory pressure during compilation
- missing CUTLASS headers: submodule issue
- unsupported architecture: stale gencode flags

## Nsight Workflow

- Use Nsight Systems when you need the timeline view.
- Use Nsight Compute when you need kernel-level counters.

Good first metrics:

- achieved occupancy
- DRAM throughput
- shared-memory bank conflicts
- tensor core utilization
- warp stall reasons

## Repo Profiling Lab

If you want a smaller profiling sandbox before touching HGEMM or attention, use:

- `kernels/nvidia-nsight/relu.cu`
- `kernels/nvidia-nsight/elementwise.cu`
- `kernels/nvidia-nsight/bank_conflicts.md`

These files are useful because they are small enough to inspect line-by-line,
but still show the difference between poor and good memory behavior.

## Performance Interpretation

If the kernel is slower than expected, ask:

- Is memory access coalesced?
- Is shared memory actually reducing global traffic?
- Did register use get too high?
- Are you bound by bandwidth instead of compute?
- Is your benchmark dominated by launch overhead?

## Debugging Mindset

Change one variable at a time:

- block size
- tile size
- vector width
- number of pipeline stages
- swizzle on or off

This repository is strong precisely because many kernels are written as a
progression. Use that progression to isolate why each optimization exists.
