# 10. 阅读清单

代码通过“可执行”来教学，slides 和 PDF 通过“命名模式与建立心智模型”来教学。

## 本章术语对照

- 架构白皮书（architecture whitepaper）
- 调优指南（tuning guide）
- 指令集架构（instruction set architecture, ISA）
- 部署栈（deployment stack）
- 推理服务（inference serving）

## 先看架构

学习 `01 CPU 与 GPU 硬件结构` 时，优先读这些：

- `slides/cuda-slides/volta-architecture-whitepaper.pdf`
- `slides/cuda-slides/nvidia-ampere-architecture-whitepaper.pdf`
- `slides/cuda-slides/NVIDIA H100 Tensor Core GPU Architecture Overview.pdf`
- `slides/cuda-slides/gtc22-whitepaper-hopper.pdf`

## CUDA 与 PTX

学习 `02 CUDA 编程模型` 和 profiling 时，优先读这些：

- `slides/cuda-slides/CUDA_C_Programming_Guide_125.pdf`
- `slides/cuda-slides/Inline_PTX_Assembly.pdf`
- `slides/cuda-slides/ptx_isa_8.5.pdf`
- `slides/cuda-slides/Hopper_Tuning_Guide.pdf`

## CUTLASS 与 GEMM

学习 GEMM、Tensor Core、tiling 时，优先读这些：

- `slides/cuda-slides/CUTLASS/Graphene-CUTE-CUTLASS-2023.pdf`
- `slides/cuda-slides/CUTLASS/colfax-gemm-kernels-hopper.pdf`
- `slides/cuda-slides/CUTLASS/layout_algebra.pdf`
- `slides/cuda-slides/CUTLASS/How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance_ a Worklog.pdf`
- `slides/cuda-slides/CUTLASS/NVIDIA Tensor Core Programmability, Performance & Precision.pdf`

## Profiling

学习 `05 调试与剖析` 时，优先读这些：

- `slides/cuda-slides/Nsight Systems - DL Profiling Argonne National Labs 2022-06-30.pdf`
- `slides/cuda-slides/pytorch-nsys-profiling.pdf`
- `kernels/nvidia-nsight/README.md`
- `kernels/nvidia-nsight/bank_conflicts.md`

## 通信与多 GPU

学习 distributed systems 和 NCCL 时，优先读这些：

- `slides/cuda-slides/NCCL 2.0.pdf`
- `slides/cuda-slides/MULTI-GPU TRAINING WITHNCCL.pdf`
- `slides/cuda-slides/S31880 – NCCL- HIGH-SPEEDINTER-GPU COMMUNICATIONFOR LARGE-SCALE TRAINING.pdf`
- `slides/cuda-slides/S41784- FAST INTER-GPU COMMUNICATION WITH NCCL FORDEEP LEARNING TRAINING, AND MORE.pdf`
- `slides/cuda-slides/CWES52010- Connect with the ExpertsInter-GPU Communication Techniques and Libraries.pdf`

## TensorRT 与部署

学习 `others/tensorrt/` 时，优先读这些：

- `slides/cuda-slides/TensorRT-Developer-Guide 10.1.pdf`
- `slides/cuda-slides/TensorRT-API.pdf`
- `slides/cuda-slides/TensorRT-Operators.pdf`
- `slides/cuda-slides/NVIDIA-Torch-TensorRT.pdf`
- `slides/cuda-slides/ORT-Python-Docs.pdf`

## vLLM 与推理服务

如果你想把 kernel 进一步连接到推理系统，可以在主线之后阅读：

- `slides/vllm-slides/01 vLLM Q1 Update.pdf`
- `slides/vllm-slides/vLLM @ Ray Summit 2024 (Public).pptx`
- `slides/vllm-slides/vLLM @ Google Cloud (Public).pptx`
- `slides/vllm-slides/blogs/README.md`

## 这份清单怎么用

- 不要一开始就从头读完全部资料。
- 先选与你正在读的代码最匹配的那一组。
- 读到能够给代码里的模式命名就够了。
- 然后回到代码，用自己的话解释那一步优化。
