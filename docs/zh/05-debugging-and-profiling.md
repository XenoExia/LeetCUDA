# 05. 调试与剖析

学习 CUDA 的主循环通常是：

1. 写一个 kernel，或者读一个 kernel
2. 先验证正确性
3. 再做 profiling
4. 一次只改一个变量
5. 再测一次

## 本章术语对照

- 正确性基线（correctness baseline）
- 剖析（profiling）
- 时间线分析（timeline analysis）
- 内核级计数器（kernel-level counters）
- 占用率（occupancy）
- 寄存器溢出（register spill）
- bank conflict
- warp stall reason

## 先保正确，再谈性能

任何调优之前，都应该保留一个可信 reference：

- PyTorch 原生算子
- cuBLAS / cuDNN / SDPA
- naive kernel

数值结果不稳定时，不要开始“盲调”性能。

## 仓库里的实用命令

```bash
python3 scripts/doctor.py
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
cd kernels/nvidia-nsight && nvcc -arch=sm_120 -o relu.bin --generate-line-info -g relu.cu
```

## Build Log 里重点看什么

- `ptxas info`：寄存器数量、shared memory 用量、spill 情况
- `Killed` / exit `137`：通常表示主机内存压力过大
- missing CUTLASS headers：通常是 submodule 没初始化好
- unsupported architecture：通常是 gencode 或 `TORCH_CUDA_ARCH_LIST` 还停留在旧卡

## Nsight 的基本工作流

- 需要时间线时，用 Nsight Systems
- 需要 kernel 级指标时，用 Nsight Compute

第一轮建议先看这些指标：

- achieved occupancy
- DRAM throughput
- shared-memory bank conflicts
- tensor core utilization
- warp stall reasons

## 仓库里的 Profiling 实验区

如果你还不想直接碰 HGEMM 或 attention，可以先用这一组小实验：

- `kernels/nvidia-nsight/relu.cu`
- `kernels/nvidia-nsight/elementwise.cu`
- `kernels/nvidia-nsight/bank_conflicts.md`

这些文件的价值在于：

- 代码足够小，可以逐行对照
- 现象足够真实，能看到 memory behavior 的差异
- 很适合建立从源码到 profiler 指标的映射

## 如何解释性能结果

如果 kernel 比预期慢，先按这个顺序排查：

- 访存是否 coalesced
- shared memory 是否真的减少了 global traffic
- register use 是否过高
- 当前是 bandwidth-bound 还是 compute-bound
- benchmark 是否主要被 launch overhead 主导

## 调试时的基本纪律

一次只改一个变量，例如：

- block size
- tile size
- vector width
- pipeline stage 数量
- swizzle 开或关

这个仓库最有价值的一点，是许多 kernel 本身就按“逐级优化”的方式组织好了。
你应该利用这种层次结构去解释每一步优化为什么存在，而不是只盯着最终 TFLOPS。
