CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (2 threads) · GPU: Tesla T4 (sm_75) · Google Colab, CUDA 12.8

Median time for one dense flow field (both frames already in memory). EPE = median end-point error vs. the known shift, in pixels.

| method | 480p | 720p | 1080p | 4k | EPE |
|---|---|---|---|---|---|
| v1 (15-112) Python loop | ~32.9 s (est.) | ~110.5 s (est.) | ~225.1 s (est.) | ~899.3 s (est.) | — |
| NumPy reference (vectorized) | 709.2 ms | 1.5 s | 3.3 s | 15.1 s | 0.045 |
| C++ / OpenMP (2 threads) | 132.1 ms | 425.1 ms | 974.7 ms | 4.6 s | 0.045 |
| CUDA, shared-memory box filter | 3.1 ms | 7.8 ms | 16.6 ms | 51.3 ms | 0.045 |
| CUDA, naive box filter | 15.3 ms | 24.2 ms | 46.6 ms | 203.9 ms | 0.045 |
| OpenCV Farneback | 96.3 ms | 309.2 ms | 818.5 ms | 3.5 s | 0.042 |
| OpenCV DIS (medium) | 45.0 ms | 131.9 ms | 539.8 ms | 1.5 s | 0.044 |

CUDA rows are kernel time (CUDA events). Wall time including host↔device (PCIe) copies:

| method | 480p | 720p | 1080p | 4k |
|---|---|---|---|---|
| CUDA, shared-memory box filter | 5.7 ms | 17.0 ms | 33.4 ms | 133.8 ms |
| CUDA, naive box filter | 18.1 ms | 38.3 ms | 63.6 ms | 314.9 ms |

Native benchmark, no Python in the loop (`build/flipster_bench --size 1920x1080`, median of 10):

| backend | median wall | median compute |
|---|---|---|
| cuda / optimized | 29.30 ms | 12.58 ms |
| cuda / naive | 62.77 ms | 45.90 ms |
| cpu (OpenMP, 2 threads) | 1024.55 ms | — |

Correctness on the same machine: all 24 GoogleTest cases pass, including the 6 CUDA-vs-CPU parity tests
(flow, warp and synthesis, for both the optimized and naive kernels), and the full pytest suite passes.

Transcribed from a run of [notebooks/flipster_colab.ipynb](../../notebooks/flipster_colab.ipynb) on a free Colab T4.
