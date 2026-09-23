CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (2 threads) · GPU: none · Linux-6.18.44-fc-v37-x86_64-with-glibc2.39

Median time for one dense flow field (both frames already in memory). EPE = median end-point error vs. the known shift, in pixels.

| method | 480p | 720p | 1080p | EPE |
|---|---|---|---|---|
| v1 (15-112) Python loop | ~14.4 s (est.) | ~46.9 s (est.) | ~104.9 s (est.) | — |
| NumPy reference (vectorized) | 345.3 ms | 1.1 s | 2.7 s | 0.045 |
| C++ / OpenMP (2 threads) | 70.7 ms | 184.9 ms | 458.7 ms | 0.045 |
| OpenCV Farneback | 72.7 ms | 221.1 ms | 599.2 ms | 0.042 |
| OpenCV DIS (medium) | 26.1 ms | 69.6 ms | 178.4 ms | 0.045 |
