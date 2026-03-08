# 01. CPU 与 GPU 硬件结构

在写 CUDA 之前，先建立正确的硬件心智模型。

## 本章术语对照

- 流式多处理器（Streaming Multiprocessor, SM）
- 吞吐导向（throughput-oriented）
- 延迟导向（latency-oriented）
- 线程束（warp）
- 片上存储（on-chip memory）
- 全局内存（global memory）

## CPU：以低延迟为核心

CPU 擅长复杂控制流和低延迟任务：

- 少量但很强的核心
- 大缓存
- 深度分支预测与推测执行
- 很强的单线程性能

在 AI Infra 里，CPU 常负责：

- 输入流水线
- kernel launch 编排
- runtime 调度
- 内存分配与 bookkeeping

## GPU：以高吞吐为核心

GPU 的目标是同时让大量算术单元保持忙碌：

- 很多 SM
- 很多同时在飞的 warp
- 很高的内存带宽
- 控制流能力弱于 CPU，但数据并行能力极强

这非常适合：

- 规则的数据访问
- 大张量
- 同一种运算被重复很多次

## GPU 的基本构件

- `thread`：你在 CUDA 代码里直接写的最小执行单元
- `warp`：硬件调度的 32 个线程组
- `block`：可以共享 `shared memory` 并进行同步的线程组
- `grid`：一次 kernel launch 的所有 blocks
- `SM`：真正执行 block 的硬件单元

写 CUDA 时，你本质上一直在回答一个问题：

如何切分张量，才能让大量 warp 做有用的工作，同时保持合并访存并减少同步？

## 存储层次

- 寄存器（register）：最快，线程私有，但数量很少
- 共享内存（shared memory）：片上、block 内共享、速度快，但要注意 bank conflict
- L2 cache：跨 SM 共享
- 全局内存（global memory）：容量大、带宽高，但比片上存储慢得多

大部分 CUDA 优化，本质上都是在把“重用”往更高层的存储上推。

## 为什么 AI Infra 特别关心这件事

AI Infra 的核心 kernel，通常是这些模式的组合：

- elementwise
- reduction
- normalization
- GEMM
- attention

它们的主瓶颈可能完全不同：

- 带宽受限（bandwidth-bound）
- 延迟受限（latency-bound）
- 共享内存受限（shared-memory-bound）
- 寄存器受限（register-bound）
- Tensor Core 受限（tensor-core-bound）

后面的教程，就是教你如何识别“真正限制速度的是哪一个”。

## 对应代码入口

- `kernels/elementwise/`
- `kernels/relu/`
- `kernels/reduce/`

这些目录适合作为第一站，因为它们已经把“线程布局”和“内存布局”的核心问题暴露出来了，但数学复杂度还不高。
