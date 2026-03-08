# 10. Reading List

The code teaches by execution. The slide decks and PDFs teach by giving names
and mental models to the patterns you see in the code.

## Architecture First

Read these when studying `01 CPU and GPU Architecture`:

- `slides/cuda-slides/volta-architecture-whitepaper.pdf`
- `slides/cuda-slides/nvidia-ampere-architecture-whitepaper.pdf`
- `slides/cuda-slides/NVIDIA H100 Tensor Core GPU Architecture Overview.pdf`
- `slides/cuda-slides/gtc22-whitepaper-hopper.pdf`

## CUDA and PTX

Read these when studying `02 CUDA Programming Model` and profiling:

- `slides/cuda-slides/CUDA_C_Programming_Guide_125.pdf`
- `slides/cuda-slides/Inline_PTX_Assembly.pdf`
- `slides/cuda-slides/ptx_isa_8.5.pdf`
- `slides/cuda-slides/Hopper_Tuning_Guide.pdf`

## CUTLASS and GEMM

Read these when studying GEMM, Tensor Cores, and tiling:

- `slides/cuda-slides/CUTLASS/Graphene-CUTE-CUTLASS-2023.pdf`
- `slides/cuda-slides/CUTLASS/colfax-gemm-kernels-hopper.pdf`
- `slides/cuda-slides/CUTLASS/layout_algebra.pdf`
- `slides/cuda-slides/CUTLASS/How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance_ a Worklog.pdf`
- `slides/cuda-slides/CUTLASS/NVIDIA Tensor Core Programmability, Performance & Precision.pdf`

## Profiling

Read these when studying `05 Debugging and Profiling`:

- `slides/cuda-slides/Nsight Systems - DL Profiling Argonne National Labs 2022-06-30.pdf`
- `slides/cuda-slides/pytorch-nsys-profiling.pdf`
- `kernels/nvidia-nsight/README.md`
- `kernels/nvidia-nsight/bank_conflicts.md`

## Communication and Multi-GPU

Read these when studying distributed systems and NCCL:

- `slides/cuda-slides/NCCL 2.0.pdf`
- `slides/cuda-slides/MULTI-GPU TRAINING WITHNCCL.pdf`
- `slides/cuda-slides/S31880 – NCCL- HIGH-SPEEDINTER-GPU COMMUNICATIONFOR LARGE-SCALE TRAINING.pdf`
- `slides/cuda-slides/S41784- FAST INTER-GPU COMMUNICATION WITH NCCL FORDEEP LEARNING TRAINING, AND MORE.pdf`
- `slides/cuda-slides/CWES52010- Connect with the ExpertsInter-GPU Communication Techniques and Libraries.pdf`

## TensorRT and Deployment

Read these when studying `others/tensorrt/`:

- `slides/cuda-slides/TensorRT-Developer-Guide 10.1.pdf`
- `slides/cuda-slides/TensorRT-API.pdf`
- `slides/cuda-slides/TensorRT-Operators.pdf`
- `slides/cuda-slides/NVIDIA-Torch-TensorRT.pdf`
- `slides/cuda-slides/ORT-Python-Docs.pdf`

## vLLM and Serving

Read these after the core kernel course if you want to connect kernels to
inference systems:

- `slides/vllm-slides/01 vLLM Q1 Update.pdf`
- `slides/vllm-slides/vLLM @ Ray Summit 2024 (Public).pptx`
- `slides/vllm-slides/vLLM @ Google Cloud (Public).pptx`
- `slides/vllm-slides/blogs/README.md`

## How To Use This Reading List

- Do not read everything first.
- Pick the section that matches the code you are studying.
- Read enough to name the pattern you already saw in code.
- Return to the code and explain the optimization in your own words.
