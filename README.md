# GEMM optimizations

Following https://siboehm.com/articles/22/CUDA-MMM.

4092 x 4092 x 4092
| | cycles | time (ms) | compute throughput | memory throughput |
| - | - | - | - | - |
| v1 | 673711019 | 606.53 | 5.98% | 98.64% |
| v2 | 1893253 | 1.71 | 66.62% | 72.06% |
| v2.1 | 58488647 | 52.75 | 73.98% | 68.62% |
| v3 | 36796355 | 33.20 | 75.16% | 89.38% |
| v4 | 18074104 | 16.28 | 55.70% | 82.63% |
