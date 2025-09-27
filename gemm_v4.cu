#include <cstdio>
#include <fstream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <iomanip>
#include <cassert>

#define CEIL_DIV(x,y) (x+y-1)/(y)
#define BM 64
#define BN 64
#define BK 8
#define TM 8

// Assume M,K,N are multiplies of block size
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float A_sh[BM * BK];
    __shared__ float B_sh[BK * BN];
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // block dim = 512
    int tid = threadIdx.x;
    int r_a = tid / BK; // row within the A-tile(64,8)
    int c_a = tid % BK; // col within the A-tile(64,8)
    int r_b = tid / BN; // row within the B-tile(8,64)
    int c_b = tid % BN; // col within the B-tile(8,64)
    int r = tid / BN; // "row-tile" within thread block C(64,64). Ranges (0-7)
    int c = tid % BN; // "col-tile" within thread block C(64,64). Ranges (0-7)
    float threadResults[TM] = {0.0f};
    for (int kBlk = 0; kBlk < K; kBlk += BK) {
        // Load from GMEM to SMEM
        A_sh[r_a * BK + c_a] = A[r_a * K + c_a];
        B_sh[r_b * BN + c_b] = B[r_b * N + c_b];
        __syncthreads();

        // For "row-tile"=1, aka r=1, each thread (64-127, mapping to columns 0-63) would handle rows 8-15 of A_sh
        for (int k = 0; k < BK; k++) {
            float b = B_sh[k*BN + c];
            for (int i = 0; i < TM; i++) {
                threadResults[i] += b * A_sh[(r*TM + i)*BK + k];
            }
        }

        // Move A to the right by BK
        A += BK;
        // MOVE B down by BK
        B += BK * N;
        __syncthreads();
    }

    // For "row-tile"=0, each thread would write to rows 0-7 of the current C-tile(64,64)
    // For "row-tile"=1, each thread would write to rows 8-15 of the current C-tile(64,64)
    // For "row-tile"=2, each thread would write to rows 16-23 of the current C-tile(64,64)
    for (int i = 0; i < TM; i++) {
        C[(r*TM + i) * N + c] = alpha * threadResults[i] + beta * C[(r*TM + i) * N + c];
    }
}

int main() {
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Total global memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessor count: " << device_prop.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block: " << device_prop.sharedMemPerBlock << " KB" << std::endl;
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dim: " << device_prop.maxThreadsDim[0] << ", " << device_prop.maxThreadsDim[1] << ", " << device_prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid size: " << device_prop.maxGridSize[0] << ", " << device_prop.maxGridSize[1] << ", " << device_prop.maxGridSize[2] << std::endl;
    std::cout << "Warp size: " << device_prop.warpSize << std::endl;
    std::cout << "Max threads per multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Shared memory per multiprocessor: " << device_prop.sharedMemPerMultiprocessor / (1024) << " KB" << std::endl;
    std::cout << "Registers per multiprocessor: " << device_prop.regsPerMultiprocessor << std::endl;

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

    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
    dim3 blockDim(512);
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
    cudaMemcpy(C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost);
    assert(std::abs(C[100*N+100] - 75.543487) < 1e-6);

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
