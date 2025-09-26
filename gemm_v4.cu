#include <cstdio>
#include <fstream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <iomanip>

#define CEIL_DIV(x,y) (x+y-1)/(y)
#define TILE_SIZE 32

// Assume M,K,N are multiplies of block size
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int K, int N, float alpha, float beta) {
    __shared__ float A_sh[TILE_SIZE * TILE_SIZE];
    __shared__ float B_sh[TILE_SIZE * TILE_SIZE];
    A += blockIdx.y * TILE_SIZE * K;
    B += blockIdx.x * TILE_SIZE;
    C += blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE;

    float acc = 0.0f;
    for (int kBlk = 0; kBlk < K; kBlk += TILE_SIZE) {
        A_sh[threadIdx.y * TILE_SIZE + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        B_sh[threadIdx.y * TILE_SIZE + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_sh[threadIdx.y * TILE_SIZE + k] * B_sh[k * TILE_SIZE + threadIdx.x];
        }

        // Move A to the right by TILE_SIZE
        A += TILE_SIZE;
        // MOVE B down by TILE_SIZE
        B += TILE_SIZE * N;

        __syncthreads();
    }
    C[threadIdx.y * N + threadIdx.x] = alpha * acc + beta * C[threadIdx.y * N + threadIdx.x];
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
