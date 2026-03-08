# 00. 快速开始

这一章的目标不是“看懂全部代码”，而是先把环境跑通。

## 本章术语对照

- 快速验证（smoke test）
- 子模块（submodule）
- CUDA 架构列表（`TORCH_CUDA_ARCH_LIST`）
- 并行编译任务数（`MAX_JOBS`）
- 张量核心（Tensor Core）

## 一次性初始化

```bash
bash scripts/bootstrap_env.sh
python3 scripts/doctor.py
```

如果你不想在低内存机器上编译官方 `flash-attn` 包：

```bash
SKIP_FLASH_ATTN_INSTALL=1 bash scripts/bootstrap_env.sh
```

## 推荐入口

统一用这个包装器来跑 Python 示例：

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0
```

它会自动做两件事：

- 按当前 GPU 设置 `TORCH_CUDA_ARCH_LIST`
- 对大型 CUDA 构建使用更保守的 `MAX_JOBS`

## 推荐首次验证

```bash
make smoke
```

这条命令会串起来执行：

- `doctor`
- `elementwise`
- `relu`
- `hgemm-smoke`
- `flash-attn-smoke`

如果这一套能过，说明当前机器已经具备“可以边学边跑”的基础能力。

## 本机构建 wheel

如果你想验证 `HGEMM` 的本地打包路径：

```bash
make hgemm-wheel
```

它默认只为当前 GPU 架构打包。如果要做多架构分发：

```bash
make hgemm-wheel-multi
```

## 常见故障

- `cute/tensor.hpp: No such file or directory`
  原因：CUTLASS 子模块没初始化。
- `ModuleNotFoundError: No module named 'flash_attn'`
  原因：官方对照包没装上。
- `Killed` 或退出码 `137`
  原因：`nvcc` 并行度过高，主机内存不够。

## 学习建议

不要一开始就冲 `HGEMM` 或 `FlashAttention`。先把下面五件事建立起来：

1. 线程索引
2. 合并访存（coalesced memory access）
3. reduction
4. 归一化与 softmax
5. GEMM 分块（tiling）
