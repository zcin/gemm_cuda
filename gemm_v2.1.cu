#include <cstdio>
#include <fstream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <iomanip>

#define CEIL_DIV(x,y) (x+y-1)/(y)

__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int K, int N, float alpha, float beta) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M && c < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = alpha * acc + beta * C[r * N + c];
    }
}

int main() {
    int M = 4096, K = 4096, N = 4096;
    float alpha = 1.5, beta = 2.5;

    float *A = new float[M*K], *B = new float[K*N], *C = new float[M*N];
    for (int i = 0; i < 4096 * 4096; i++) {
        A[i] = static_cast<float>(i) / (M*K);
        B[i] = static_cast<float>(i) / (K*N);
        C[i] = static_cast<float>(i) / (M*N);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K * sizeof(float));
    cudaMalloc(&d_B, K*N * sizeof(float));
    cudaMalloc(&d_C, M*N * sizeof(float));
    cudaMemcpy(d_A, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
