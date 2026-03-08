# 08. Beyond Kernels

Once you can read CUDA kernels and reason about GEMM or attention, the next step
is learning how those ideas surface in higher-level infra stacks.

## Why This Chapter Exists

Real AI infra work is rarely only one handwritten CUDA file.
You also need to know:

- when a Triton kernel is good enough
- when a TensorRT path matters for deployment
- when distributed communication dominates compute

This repository already has examples for all three.

## Triton Track

Start in `kernels/openai-triton/`.

Recommended order:

1. `vector-add/`
2. `fused-softmax/`
3. `layer-norm/`
4. `matrix-multiplication/`
5. `merge-attn-states/`

Why Triton matters:

- it keeps the GPU programming model visible
- it is much faster to iterate on than raw CUDA
- it teaches tiling and memory movement in a more compact form

Suggested commands:

```bash
python3 kernels/openai-triton/vector-add/triton_vector_add.py
python3 kernels/openai-triton/fused-softmax/triton_fused_softmax.py
python3 kernels/openai-triton/layer-norm/triton_layer_norm.py
```

If you want a bridge from Triton back to CUDA, compare these with:

- `kernels/softmax/`
- `kernels/layer-norm/`
- `kernels/flash-attn/`

## Distributed PyTorch Track

Start in `others/pytorch/distributed/`.

What to study:

- collective communication patterns
- latency versus bandwidth tradeoffs
- how compute kernels and communication overlap

Start with:

- [distributed README](../others/pytorch/distributed/README.md)
- `test_all_reduce.py`
- `test_all_gather.py`
- `test_reduce_scatter.py`
- `test_all_to_all.py`

Question to keep in mind:

- if the kernel is fast, does communication now dominate step time?

## TensorRT Track

Start in `others/tensorrt/`.

The goal here is not to rewrite TensorRT internals.
The goal is to understand how handcrafted kernels connect to inference systems.

Look at:

- [TensorRT README](../others/tensorrt/README.md)
- `fmha/`
- `plugin/`

Focus on:

- graph export
- operator fusion
- plugin boundaries
- deployment-oriented kernel selection

## Companion Repos In This Workspace

This workspace also includes two upstream companion repos as git submodules:

- `HGEMM/`: a standalone extraction of the HGEMM path from LeetCUDA
- `ffpa-attn/`: a follow-up attention repo focused on large-headdim prefill kernels

Use them when:

- you want a smaller codebase than the full LeetCUDA repo
- you want to study one topic in isolation
- you want to compare how a topic evolves after leaving the teaching repo

## What Changes At This Stage

Earlier chapters ask:

- how do I make one kernel correct and fast?

This chapter asks:

- how do I choose the right implementation layer?
- how do I connect kernels to a system?
- how do I reason about end-to-end bottlenecks?

## Recommended Order After The Core Course

1. Finish `04 GEMM and Attention`
2. Do one pass through `05 Debugging and Profiling`
3. Read `07 Glossary and Cheatsheet` again
4. Study Triton examples
5. Study distributed communication examples
6. Study TensorRT deployment examples
