# CUDA Matrix Multiplication Optimizations

This project explores different optimization techniques for dense matrix multiplication using CUDA. Each implementation builds upon the previous one, with the goal of understanding how modern GPU optimizations improve performance.

## Implementations

- **Naive Matrix Multiplication**
  - One thread computes one output element.
  - Every multiplication reads directly from global memory.
  - No data reuse.

- **Shared Memory Tiling**
  - Matrix tiles are loaded into shared memory.
  - Reduces redundant global memory accesses.
  - Each thread still computes a single output element.

- **1 × 2 Register Tiling**
  - Shared memory tiling + register tiling.
  - Each thread computes two adjacent output elements.
  - Reduces shared memory accesses by reusing loaded values in registers.

- **2 × 2 Register Tiling**
  - Each thread computes a 2 × 2 output tile.
  - Further increases register reuse while reducing shared memory accesses.

---

# Benchmark Results

All benchmarks were performed on an NVIDIA RTX 4070.

## Matrix Size: 4096 × 4096

| Kernel | Time (ms) | Speedup |
|---------|----------:|---------:|
| Naive | 127.27 | 1.00× |
| Shared Memory Tiled | 88.99 | 1.43× |
| Register Tiled (1×2) | 47.47 | 2.68× |
| Register Tiled (2×2) | 35.62 | **3.57×** |

---

## Matrix Size: 1024 × 1024

| Kernel | Time (ms) | Speedup |
|---------|----------:|---------:|
| Naive | 2.41 | 1.00× |
| Shared Memory Tiled | 2.18 | 1.11× |
| Register Tiled (1×2) | 1.31 | 1.85× |
| Register Tiled (2×2) | 0.66 | **3.67×** |

---

## Correctness

All kernels were validated against a CPU implementation using floating-point tolerance checks.

---

# Observations

### Shared memory tiling

Moving data from global memory into shared memory provided a modest improvement over the naive implementation. The performance gain was smaller than expected because synchronization overhead and additional shared memory operations partially offset the reduction in global memory traffic.

---

### Register tiling

Register tiling provided the largest performance improvement.

Computing multiple output elements per thread allowed values loaded from shared memory to be reused several times before being discarded, significantly reducing shared memory traffic.

---

### Effect of Block Tile Size

Initially, all kernels used a **16 × 16** block tile.

Increasing the block tile size to **32 × 32** produced only a small improvement for the standard shared-memory implementation.

However, it dramatically improved the performance of the **2 × 2 register tiled kernel**.

The reason is that register tiling decouples the **output tile size** from the **number of threads**.

| Configuration | Threads per Block |
|--------------|------------------:|
| Shared Tile (32×32) | 1024 |
| Register Tile 2×2 (32×32 output tile) | 256 |

The larger output tile increased data reuse while maintaining a reasonable number of threads per block, resulting in substantially better performance.

---

# Key Takeaways

- Shared memory reduces global memory traffic but introduces synchronization overhead.
- Register tiling further reduces shared memory accesses by reusing values in registers.
- Larger output tiles do not necessarily require more threads when register tiling is used.
- GPU performance depends on balancing:
  - Global memory traffic
  - Shared memory reuse
  - Register usage
  - Threads per block
  - Warp occupancy

Simply increasing tile sizes or register usage does not always improve performance. The best performance was achieved by balancing these factors rather than maximizing any single one.

---

# Future Work

Planned optimizations include:

- Larger register tiles (2×4, 4×2, 4×4)
- Vectorized memory loads (`float2` / `float4`)
- Double buffering
- Software pipelining
- Warp-level primitives (`__shfl_sync`)
- Tensor Core implementation using WMMA
- Comparison against cuBLAS
