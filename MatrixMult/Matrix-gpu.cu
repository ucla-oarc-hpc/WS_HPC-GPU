#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Kernel for multiplying matrices
__global__ void matrixMulCUDA(int *a, int *b, int *c, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int tempSum = 0;
    if (ROW < N && COL < N) {
        for (int i = 0; i < N; i++) {
            tempSum += a[ROW * N + i] * b[i * N + COL];
        }
        c[ROW * N + COL] = tempSum;
    }
}

int main() {
    int N = 1024;  // Size of the matrix (1024x1024)
    int SIZE = N * N;

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    size_t bytes = SIZE * sizeof(int);

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto start = std::chrono::high_resolution_clock::now();

    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CUDA Matrix multiplication time: " << elapsed.count() << " ms." << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}