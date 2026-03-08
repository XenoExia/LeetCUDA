# 06. 练习

这一章的目标，是把“看懂代码”变成“形成自己的 CUDA 直觉”。

## 本章术语对照

- 标量版本（scalar variant）
- 向量化版本（vectorized variant）
- reduction tree
- 数值稳定性（numerical stability）
- 归一化（normalization）
- 带宽瓶颈（bandwidth bottleneck）
- 计算吞吐（compute throughput）
- 性能复盘（performance review）

## 初级练习

1. 在 `kernels/relu/` 中修改 block size，并测量结果变化。
2. 在 `kernels/elementwise/` 中对比 scalar 与 vectorized 版本。
3. 在 `kernels/reduce/` 中解释 shared memory 必要的地方，以及 warp shuffle 足够的地方。

## 中级练习

1. 在 `kernels/softmax/` 中解释数值稳定性是如何维持的。
2. 在 `kernels/mat-transpose/` 中找出 naive 访存模式与优化后访存模式。
3. 在 `kernels/layer-norm/` 和 `kernels/rms-norm/` 中解释 reduction 加 normalization 的成本。

## 高级练习

1. 在 `kernels/sgemm/` 中区分哪些优化主要服务于 bandwidth，哪些主要服务于 compute throughput。
2. 在 `kernels/hgemm/` 中对比 WMMA 路径与 MMA 路径。
3. 在 `kernels/flash-attn/` 中解释为什么 Split-Q 有机会快过 Split-KV。

## 每个 Kernel 的复盘模板

每研究一个 kernel，至少写下这四行：

- mapping：一个 thread 或一个 warp 拿到了哪些数据
- memory：数据从哪里读入，又在哪里复用
- sync：屏障或 warp 内通信发生在哪里
- bottleneck：你认为当前最可能的性能瓶颈是什么

## 结课作业

做一个自己的 mini 学习报告：

1. 选一个简单一元算子
2. 选一个 reduction 算子
3. 选一个 GEMM kernel
4. 选一个 attention kernel
5. 解释它们从 naive 到优化版的“优化阶梯”

如果你能把这一份报告写清楚，说明你已经开始像 infra 工程师一样思考，而不只是框架用户。
