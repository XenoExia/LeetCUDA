# 09. 仓库全图

这一章的目标，是把整个仓库都纳入教程，而不是只停留在前几章已经点到的目录。

## 本章术语对照

- 主线实验（core lab）
- 侧向专题（side track）
- 系统级实验（system-level lab）
- 参考资料（reference material）
- 预留目录（reserved area）
- 子模块（submodule）

## 核心 Kernel 实验区

这些是 `kernels/` 下最主要的动手目录：

- `elementwise`、`relu`、`sigmoid`、`swish`、`elu`
  重点：索引、向量化、coalescing。
- `dot-product`、`reduce`、`softmax`、`layer-norm`、`rms-norm`
  重点：reduction 模式、数值稳定性、accumulation。
- `mat-transpose`、`embedding`、`rope`、`swizzle`、`histogram`、`nms`
  重点：数据布局、memory traffic、不规则访问。
- `sgemv`、`hgemv`、`sgemm`、`hgemm`、`ws-hgemm`
  重点：GEMV/GEMM、Tensor Core、流水化、warp specialization。
- `flash-attn`
  重点：transformer attention kernel 与 SRAM 权衡。

## 进阶 CUDA 侧线

这些目录不是第一站，但在你掌握主线后都应该被纳入课程：

- `kernels/cutlass/cute/`
  小型 CuTe 实验，例如 `vector_add.cu`、`mma_tile_tex.cc`。
- `kernels/openai-triton/`
  通过 Triton 进入更高层 GPU 编程。
- `kernels/nvidia-nsight/`
  Profiling、PTX/SASS 观察与 bank conflict 实验区。
- `kernels/ws-hgemm/`
  带 warp specialization 风格的 HGEMM 实验。

## 系统级实验

这些目录不在 `kernels/` 下，但对 infra 学习同样关键：

- `others/pytorch/distributed/`
  collective communication 和多 GPU 协调。
- `others/tensorrt/`
  推理部署、TensorRT 集成与插件实验。
- `others/pytorch/custom_ops/`
  当前还是占位区，未来可以扩成 custom-op 章节。

## 配套子仓库

这些内容通过 submodule 挂在同一工作区里：

- `HGEMM/`
  从本项目延伸出去的 HGEMM 专题仓库。
- `ffpa-attn/`
  面向大 headdim Flash Prefill Attention 的后续仓库。
- `third-party/cutlass/`
  多条学习路线都会用到的上游 CUTLASS 源码。

## 参考资料区

这些不是可执行实验，但属于教程的一部分：

- `slides/cuda-slides/`
  CUDA、CUTLASS、TensorRT、NCCL、PTX ISA、架构白皮书。
- `slides/vllm-slides/`
  vLLM、推理服务与系统设计资料。
- `others/pytorch/slides/`
  补充性的 PyTorch 幻灯片资料。

## 预留或暂时稀疏的区域

有些目录当前内容不多，但已经被纳入仓库地图：

- `kernels/transformer/`
  预留给后续 transformer kernel 或系统级实验。
- `kernels/openai-triton/fused-attention/`
  目前内容少于其他 Triton 子目录。
- `others/pytorch/custom_ops/`
  未来可以扩展成运算符扩展示例区。

## 完成主线后的建议游览顺序

1. 重新看 `kernels/nvidia-nsight/`
2. 学 `kernels/ws-hgemm/`
3. 学 `kernels/cutlass/cute/`
4. 学 `kernels/openai-triton/`
5. 学 `others/pytorch/distributed/`
6. 学 `others/tensorrt/`
7. 对照 `HGEMM/` 与 `ffpa-attn/`
