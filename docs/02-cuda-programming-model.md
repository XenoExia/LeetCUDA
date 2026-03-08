# 02. CUDA Programming Model

This chapter translates hardware ideas into code structure.

## Execution Model

Every CUDA kernel launch defines:

- grid size
- block size
- per-thread work

You should always know:

- which tensor indices a thread owns
- how blocks cover the full tensor
- whether memory access is contiguous for neighboring threads

## The First Questions To Ask

When reading or writing a kernel, ask these in order:

1. Is the mapping from threads to data correct?
2. Are global memory accesses coalesced?
3. Is there reuse worth staging in shared memory?
4. Is synchronization minimal and correct?
5. Is the kernel bandwidth-bound or compute-bound?

## Occupancy and Latency Hiding

The GPU hides latency by scheduling other warps while one warp waits on memory.
Occupancy is not the goal by itself; useful occupancy is.

Low occupancy is acceptable when:

- tensor cores are saturated
- register tiling gives high arithmetic intensity
- memory traffic is already minimized

High occupancy is useful when:

- memory latency dominates
- each warp does little work
- the kernel is simple and bandwidth-bound

## Memory Rules of Thumb

- Prefer contiguous global loads and stores.
- Avoid unnecessary host-device transfers.
- Use shared memory only when it reduces global traffic enough to justify the complexity.
- Watch for bank conflicts when multiple threads hit the same shared-memory bank.

## Synchronization Rules of Thumb

- Use block-wide synchronization only when data is actually exchanged.
- Prefer warp-level primitives when communication stays inside one warp.
- Avoid serial work inside one thread when it can be distributed across lanes.

## Match This Chapter With Code

Start in this order:

1. `kernels/elementwise/`
2. `kernels/relu/`
3. `kernels/dot-product/`
4. `kernels/reduce/`
5. `kernels/softmax/`

Run them with:

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/softmax/softmax.py
```

## What To Observe

- How the author computes row and column indices.
- When vectorized loads such as `float4` or packed `half` variants appear.
- How shared memory is introduced only after the naive version.
- How the benchmark output changes as arithmetic intensity grows.
