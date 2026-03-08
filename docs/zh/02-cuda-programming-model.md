# 02. CUDA 编程模型

这一章把硬件概念翻译成你在代码里真正会写的结构。

## 本章术语对照

- 执行模型（execution model）
- 占用率（occupancy）
- 隐藏延迟（latency hiding）
- 合并访存（coalesced memory access）
- 共享内存（shared memory）
- Bank 冲突（bank conflict）

## 执行模型

每一次 CUDA kernel launch 都会定义三件事：

- grid 大小
- block 大小
- 每个线程负责的工作

你必须始终知道：

- 一个线程负责哪些张量下标
- block 如何覆盖整个张量
- 相邻线程的内存访问是否连续

## 读 kernel 时先问什么

按这个顺序来：

1. 线程到数据的映射对不对？
2. 全局内存访问是不是合并的？
3. 有没有值得搬进 shared memory 的重用？
4. 同步是不是最少且正确？
5. 这个 kernel 更像 bandwidth-bound 还是 compute-bound？

## Occupancy 与隐藏延迟

GPU 通过调度别的 warp 来隐藏一个 warp 的等待时间。`occupancy` 不是目的本身，“有用的 occupancy” 才是目的。

低 occupancy 也可能完全没问题：

- Tensor Core 已经吃满
- register tiling 提高了算术强度
- 全局访存已经压到很低

高 occupancy 更有价值的场景：

- 内存延迟主导
- 每个 warp 做的工作很少
- kernel 很简单，本质上是带宽型

## 内存经验法则

- 优先保证全局内存 load/store 连续
- 避免不必要的 host-device 拷贝
- 只有当 shared memory 明显降低全局流量时，才引入它
- 多线程打到同一个 shared-memory bank 时，要警惕 bank conflict

## 同步经验法则

- 只有真的交换数据时，才做 block 级同步
- 如果通信只发生在 warp 内，优先用 warp-level primitive
- 避免把可以分给多个 lane 的工作串行塞给一个线程

## 对应代码顺序

建议按下面顺序跑：

1. `kernels/elementwise/`
2. `kernels/relu/`
3. `kernels/dot-product/`
4. `kernels/reduce/`
5. `kernels/softmax/`

```bash
python3 scripts/run_example.py kernels/elementwise/elementwise.py
python3 scripts/run_example.py kernels/reduce/block_all_reduce.py
python3 scripts/run_example.py kernels/softmax/softmax.py
```

## 观察重点

- 作者如何计算行列索引
- 什么时候开始出现 `float4`、pack `half` 之类的向量化 load
- 为什么 shared memory 总是在 naive 版本之后才出现
- 随着算术强度增长，benchmark 输出如何变化
