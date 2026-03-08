# 07. 术语与速查

这页是你在“知道结果不对或性能不对，但还不知道该用哪个概念解释”时的随身参考页。

## 硬件术语

- `SM`：streaming multiprocessor，NVIDIA GPU 的主要计算单元。
- `warp`：32 个线程组成的执行组，在 SIMT 模型下锁步执行。
- `lane`：warp 中的单个线程。
- `register`：最快的线程私有存储，但数量有限。
- `shared memory`：block 内线程共享的片上 scratchpad。
- `global memory`：片外 DRAM，容量大但延迟高。
- `Tensor Core`：面向 MMA 类矩阵计算的专用执行单元。

## 性能术语

- `occupancy`：相对硬件上限，一个 SM 上可驻留 warp 的比例。
- `coalescing`：相邻线程是否访问相邻的 global memory 地址。
- `bank conflict`：多个线程竞争同一 shared-memory bank。
- `register pressure`：寄存器使用过高，导致 occupancy 下降或 spill。
- `latency hiding`：让足够多的独立 warp 就绪，以掩盖 stall。
- `arithmetic intensity`：计算量与数据搬运量之比。
- `roofline`：kernel 最终受限于带宽还是算力的上界模型。

## Kernel 设计术语

- `tiling`：把大问题拆成 block、warp、instruction 级别的小块。
- `double buffering`：一边计算当前 tile，一边预取下一块数据。
- `cp.async`：把 global 到 shared 的搬运做成异步流水。
- `WMMA`：较高层的 Tensor Core API。
- `MMA`：更底层的 Tensor Core 指令接口，控制粒度更细。
- `CuTe/CUTLASS`：用于表达 tiled layout、copy、matmul 的模板库。
- `swizzle`：一种布局技巧，用于减少 bank conflict 或提升局部性。

## 阅读 Kernel 时先问什么

读任何一个 kernel，先按顺序回答这五个问题：

1. 一个 thread 拥有哪些数据
2. 一个 warp 拥有哪些数据
3. 哪些数据会在 shared memory 或 register 里复用
4. 同步发生在哪里
5. 这个 kernel 更像 bandwidth-bound 还是 compute-bound

## 常用命令速查

环境与验证：

```bash
make bootstrap
make doctor
make smoke
```

前期入门实验：

```bash
make elementwise
make relu
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/softmax/softmax.py
```

GEMM 与 attention：

```bash
make hgemm-smoke
make flash-attn-smoke
make hgemm-wheel
```

可选的多架构打包：

```bash
make hgemm-wheel-multi
```

## Profiling 命令速查

如果机器上已经安装 Nsight 工具：

```bash
nsys profile --trace=cuda,nvtx python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
ncu --set full python3 scripts/run_example.py kernels/relu/relu.py
```

重点看：

- register count
- shared-memory usage
- spill stores / spill loads
- DRAM throughput
- tensor core utilization
- warp stall reasons

## 学习里程碑

- 初级：能解释索引、coalescing、reduction 形状。
- 中级：能解释 shared-memory reuse、normalization、softmax stability。
- 高级：能解释 tiling、staging、Tensor Cores，以及为什么 attention 会继承 GEMM 的思路。
