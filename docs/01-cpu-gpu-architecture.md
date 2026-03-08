# 01. CPU and GPU Architecture

Before writing CUDA, you need the right mental model of the hardware.

## CPU: Latency-Oriented

CPUs are optimized for low-latency execution of complex control flow:

- a few powerful cores
- large caches
- deep branch prediction and speculation
- strong single-thread performance

This is ideal for:

- operating systems
- request handling
- compilers
- orchestration and scheduling

In AI infra, CPUs usually handle:

- input pipelines
- kernel launch orchestration
- runtime scheduling
- memory allocation and bookkeeping

## GPU: Throughput-Oriented

GPUs are optimized to keep many arithmetic units busy at the same time:

- many SMs
- many warps in flight
- high memory bandwidth
- weaker control-flow handling than CPUs

This is ideal for workloads with:

- regular data access
- large tensors
- the same operation repeated many times

That is exactly what makes GPUs a natural fit for deep learning.

## GPU Building Blocks

- Thread: the smallest software execution unit you write in CUDA.
- Warp: a hardware scheduling group of 32 threads.
- Block: a group of threads that share `shared memory` and can synchronize.
- Grid: all blocks launched by one kernel.
- SM: streaming multiprocessor, the hardware unit that executes blocks.

When you write CUDA, you are really answering one question:

How should the tensor be partitioned so that many warps can do useful work with
coalesced memory access and minimal synchronization?

## Memory Hierarchy

- Registers: fastest, private to each thread, very limited.
- Shared memory: on-chip, shared by threads in a block, fast if bank conflicts are controlled.
- L2 cache: shared across SMs, useful for data reuse.
- Global memory: large and high bandwidth, but much slower than on-chip memory.

Most CUDA optimization is just moving data reuse upward in this hierarchy.

## Why AI Infra Cares

The core AI infra kernels are mostly combinations of:

- elementwise ops
- reductions
- normalization
- matrix multiply
- attention

Each one is constrained by a different bottleneck:

- bandwidth-bound
- latency-bound
- shared-memory-bound
- register-bound
- tensor-core-bound

The rest of this repo teaches you how to identify which bottleneck dominates.

## Match This Chapter With Code

- `kernels/elementwise/`
- `kernels/relu/`
- `kernels/reduce/`

These directories are the best first stop because they expose the memory and
thread layout questions without too much math or control complexity.
