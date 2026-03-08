# 03. Kernel 学习路线图

这一章给出整个仓库的推荐学习顺序。

## 本章术语对照

- 向量化（vectorization）
- reduction
- 数据布局（data layout）
- 分块（tiling）
- Warp 专门化（warp specialization）
- 剖析实验（profiling lab）

## Stage 1：向量加与一元算子

目标：理解索引、连续访存、向量化 load/store。

- `kernels/elementwise/`
- `kernels/relu/`
- `kernels/sigmoid/`
- `kernels/swish/`
- `kernels/elu/`

建议命令：

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/relu/relu.py
```

## Stage 2：Reduction 与归一化

目标：理解 warp/block 协作与累加模式。

- `kernels/dot-product/`
- `kernels/reduce/`
- `kernels/layer-norm/`
- `kernels/rms-norm/`
- `kernels/softmax/`

重点看：

- warp shuffle 与 shared memory 的分工
- reduction tree 的形状
- 数值稳定性
- 累加精度

## Stage 3：布局与数据搬运

目标：理解为什么数据布局常常决定性能上限。

- `kernels/mat-transpose/`
- `kernels/embedding/`
- `kernels/rope/`
- `kernels/swizzle/`
- `kernels/histogram/`
- `kernels/nms/`

重点看：

- 行主序与列主序
- transpose 的真实代价
- shared memory bank conflict
- swizzle 是布局技巧，不是魔法

## Stage 4：GEMV 与 GEMM

目标：从 CUDA Core kernel 过渡到 Tensor Core kernel。

- `kernels/sgemv/`
- `kernels/hgemv/`
- `kernels/sgemm/`
- `kernels/hgemm/`
- `kernels/ws-hgemm/`
- `kernels/cutlass/`

这一阶段要吃透：

- tiling
- double buffering
- `cp.async`
- WMMA 与 MMA
- Tensor Core 友好的数据布局
- warp specialization
- CuTe layout algebra

## Stage 5：Attention 与 AI Infra Kernel

目标：把硬件模型与 transformer 核心算子连接起来。

- `kernels/flash-attn/`
- `kernels/transformer/`
- `kernels/openai-triton/`
- `others/tensorrt/`
- `others/pytorch/distributed/`

学到这里，你应该能回答：

- attention 的哪一段更像 bandwidth-bound
- SRAM 压力来自哪里
- 为什么 Q/K/V 的 tiled movement 重要
- 为什么不会 GEMM，就很难真正理解 attention

## Stage 6：剖析与检查实验

目标：不只会跑 benchmark，还会“看见” kernel 发生了什么。

- `kernels/nvidia-nsight/`
- `slides/cuda-slides/`

重点看：

- PTX 与 SASS
- 行号信息与源代码对应
- bank conflict
- timeline profiling 与 kernel counter profiling 的区别

## 进阶规则

在进入下一阶段之前，至少要能说清楚：

- 线程到数据的映射
- 主导的数据搬运路径
- 可能的瓶颈
- 优化版为什么比 naive 版快
