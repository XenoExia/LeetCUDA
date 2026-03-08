# 08. 进阶专题：从 Kernel 到系统

当你已经能读 CUDA kernel，也能分析 GEMM 和 attention，下一步就是理解这些思想如何出现在更高层的 infra 栈里。

## 本章术语对照

- 实现层级（implementation layer）
- Triton kernel
- 分布式通信（distributed communication）
- 集体通信（collective communication）
- 部署路径（deployment path）
- 插件边界（plugin boundary）
- 端到端瓶颈（end-to-end bottleneck）

## 为什么需要这一章

真实的 AI infra 工作，通常不只是手写一个 CUDA 文件。
你还需要知道：

- 什么时候 Triton 已经足够好
- 什么时候 TensorRT 路线会影响部署
- 什么时候通信开销开始压过计算开销

而这个仓库里，这三条线其实都已经具备了入门材料。

## Triton 路线

从 `kernels/openai-triton/` 开始。

推荐顺序：

1. `vector-add/`
2. `fused-softmax/`
3. `layer-norm/`
4. `matrix-multiplication/`
5. `merge-attn-states/`

Triton 值得学的原因：

- 它仍然保留了 GPU 编程模型
- 迭代速度通常比原始 CUDA 更快
- 可以用更紧凑的方式学习 tiling 与 memory movement

建议命令：

```bash
python3 kernels/openai-triton/vector-add/triton_vector_add.py
python3 kernels/openai-triton/fused-softmax/triton_fused_softmax.py
python3 kernels/openai-triton/layer-norm/triton_layer_norm.py
```

如果你想把 Triton 和 CUDA 对起来看，可以回头比较：

- `kernels/softmax/`
- `kernels/layer-norm/`
- `kernels/flash-attn/`

## PyTorch Distributed 路线

从 `others/pytorch/distributed/` 开始。

这里要学的是：

- collective communication pattern
- latency 与 bandwidth 的权衡
- compute kernel 与 communication 的重叠机会

建议先看：

- [distributed README](../../others/pytorch/distributed/README.md)
- `test_all_reduce.py`
- `test_all_gather.py`
- `test_reduce_scatter.py`
- `test_all_to_all.py`

要一直带着这个问题：

- 当 kernel 已经足够快之后，step time 会不会主要被通信主导

## TensorRT 路线

从 `others/tensorrt/` 开始。

这一段的目标不是重写 TensorRT 内部，而是理解手写 kernel 如何接到推理系统里。

重点看：

- [TensorRT README](../../others/tensorrt/README.md)
- `fmha/`
- `plugin/`

关注的问题：

- graph export
- operator fusion
- plugin boundary
- 面向部署的 kernel 选择

## 工作区里的配套子仓库

当前工作区还挂了两个上游 companion repo：

- `HGEMM/`：从 LeetCUDA 中拆出来的 HGEMM 专题仓库
- `ffpa-attn/`：面向大 headdim prefill attention 的后续仓库

它们适合在这些场景使用：

- 你想看比 LeetCUDA 更小、更聚焦的代码库
- 你想孤立地研究单一主题
- 你想比较某个主题离开“教学仓库”后是如何继续演化的

## 这一章改变了什么

前面的章节主要问：

- 如何把一个 kernel 写对、写快

这一章开始问：

- 应该选哪一层实现
- 如何把 kernel 接到系统里
- 端到端瓶颈到底在哪里

## 建议的后续顺序

1. 完成 `04 GEMM 与 Attention`
2. 再回看一遍 `05 调试与剖析`
3. 重新翻一次 `07 术语与速查`
4. 学 Triton
5. 学 distributed communication
6. 学 TensorRT 部署路径
