# 04. GEMM 与 Attention

GEMM 是现代 AI infra 的中心算子。
Attention kernel 本质上也在复用同一套优化语言：分块、数据复用、寄存器压力控制、分阶段搬运。

## 本章术语对照

- 通用矩阵乘（GEMM, general matrix multiplication）
- 分块（tiling）
- 共享内存（shared memory）
- 寄存器压力（register pressure）
- 流水化/分阶段搬运（staging / pipelining）
- Tensor Core
- 指令级分块（instruction tile）
- Shared SRAM 复杂度（SRAM complexity）

## 为什么先学 GEMM

如果你对 GEMM 的数据流动没有直觉，就很难真正理解 attention。
两者的核心问题是一致的：

- 把大问题切成 block、warp、instruction 级别的小块
- 把数据阶段性搬进 shared memory
- 让 Tensor Core 尽可能持续工作
- 减少重复的 global memory 访问

## HGEMM 学习路径

先跑最小可执行例子：

```bash
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

然后按 `kernels/hgemm/` 里的层次往下读：

- `naive/`：先保证正确性
- `wmma/`：通过 WMMA API 理解 Tensor Core 编程
- `mma/`：更底层的 MMA 路径，控制力更强
- `cutlass/`：用 CuTe/CUTLASS 风格理解 layout 与 swizzle
- `cublas/`：把外部高性能库当成基准线

随后把相邻实验也接进主线：

- `kernels/ws-hgemm/`：带 warp specialization 风格的 HGEMM 实验
- `kernels/cutlass/cute/`：更小的 CuTe 布局实验，帮助建立 layout 直觉
- `HGEMM/`：从本仓库拆出的 HGEMM 上游子仓库

这一段重点要问自己：

- block、warp、instruction 三层 tile 是怎么选的
- shared memory buffering 从哪一版开始出现
- 哪一步开始引入明显的 register pressure
- swizzle 如何改变 bank conflict 或 cache 行为

## FlashAttention 学习路径

HGEMM 思路熟悉之后，再进入：

```bash
python3 scripts/run_example.py kernels/flash-attn/flash_attn_mma.py --minimal-build --B 1 --H 8 --N 1024 --D 64 --iters 1 --warmup 0 --sdpa
```

即使没有安装官方 `flash-attn` Python 包，当前脚本也会跳过官方对照，而不是导入时报错。
`--minimal-build` 只编译教学所需的最小子集，更适合 smoke test 和第一轮阅读。

阅读时重点看：

- Split-Q 与 Split-KV 的差异
- shared KV 或 fully shared QKV
- SRAM 复杂度
- head dimension 的上限来自哪里
- 与 SDPA 或 FA 的 correctness 对照如何建立

然后再接到更大 headdim 的后续路线：

- `ffpa-attn/`：面向 `D > 256` 的下一阶段 attention 学习仓库
- 对比它为什么比 `kernels/flash-attn/` 使用更细粒度的 tiling

## 实际阅读顺序

1. `kernels/sgemm/`
2. `kernels/hgemm/naive/`
3. `kernels/hgemm/wmma/`
4. `kernels/hgemm/mma/basic/`
5. `kernels/hgemm/mma/swizzle/`
6. `kernels/flash-attn/mma/basic/`
7. `kernels/flash-attn/mma/swizzle/`
8. `ffpa-attn/csrc/cuffpa/`

## 什么叫“学会了”

到了这一章，判断标准不只是“能跑快”。
还要能看清这些问题：

- 数据搬运路径是否明确
- tiling 是否可解释、可复现
- benchmark 是否有可靠基线
- correctness 是否有清楚的 reference
- 优化项能不能被逐个 ablation
