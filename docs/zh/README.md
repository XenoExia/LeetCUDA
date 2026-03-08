# LeetCUDA 中文教程

这是一套面向 AI Infra / CUDA 学习者的中文教程。英文主教程保留在 [docs/README.md](../README.md)。
中文章节会嵌入关键术语的中英对照，避免把硬件或编译相关概念误译成日常词汇。

## 语言切换

- 中文教程：当前页面
- English Tutorial: [../README.md](../README.md)

## 建议起步命令

```bash
make bootstrap
make doctor
make smoke
```

如果你想用带现代 GPU 默认参数的统一入口：

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

## 学习顺序

1. [00-getting-started.md](./00-getting-started.md)
2. [01-cpu-gpu-architecture.md](./01-cpu-gpu-architecture.md)
3. [02-cuda-programming-model.md](./02-cuda-programming-model.md)
4. [03-kernel-roadmap.md](./03-kernel-roadmap.md)
5. [04-gemm-and-attention.md](./04-gemm-and-attention.md)
6. [05-debugging-and-profiling.md](./05-debugging-and-profiling.md)
7. [06-exercises.md](./06-exercises.md)
8. [07-glossary-and-cheatsheet.md](./07-glossary-and-cheatsheet.md)
9. [08-beyond-kernels.md](./08-beyond-kernels.md)
10. [09-repo-atlas.md](./09-repo-atlas.md)
11. [10-reading-list.md](./10-reading-list.md)

## 你会学到什么

- CPU 与 GPU 的硬件组织方式，以及为什么 GPU 适合吞吐型任务。
- CUDA 的执行模型：grid、block、warp、lane、SM、occupancy、memory hierarchy。
- 如何从简单算子一路读到 GEMM、FlashAttention、Triton 和系统级实验。
- 如何验证正确性、测性能、看 PTX/SASS、定位瓶颈。

## 章节与实验入口

- `00 快速开始`：`make bootstrap`、`make doctor`、`make smoke`
- `01 硬件结构`：先读，再回到 `kernels/elementwise/` 和 `kernels/reduce/`
- `02 CUDA 编程模型`：重点跑 `elementwise`、`relu`、`softmax`
- `03 Kernel 路线图`：按目录逐级推进
- `04 GEMM 与 Attention`：`make hgemm-smoke`、`make flash-attn-smoke`
- `05 调试与剖析`：结合 `kernels/nvidia-nsight/`
- `06 练习`：把阅读结果写成自己的性能分析
- `07 术语与速查`：做随手翻阅的概念手册
- `08 进阶专题`：Triton、TensorRT、distributed、子仓库
- `09 仓库全图`：把剩余目录全部挂到教程中
- `10 阅读清单`：把 slides / PDF 资料与代码主线对齐
