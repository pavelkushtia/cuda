/**
 * CUDA Matrix Operations
 * 
 * This program demonstrates various matrix operations using CUDA:
 * - Matrix Addition
 * - Matrix Multiplication
 * - Matrix Transpose
 * - Matrix Scalar Operations
 * 
 * Features:
 * - Dynamic matrix size support
 * - Error checking and validation
 * - Performance timing
 * - CPU vs GPU comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>

// Matrix dimensions (can be modified)
#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

/**
 * CUDA kernel for matrix addition: C = A + B
 */
__global__ void matrixAddKernel(float* A, float* B, float* C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

/**
 * CUDA kernel for matrix multiplication: C = A * B
 */
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int widthA, int heightA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < heightA && col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; k++) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

/**
 * CUDA kernel for matrix transpose: B = A^T
 */
__global__ void matrixTransposeKernel(float* A, float* B, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        B[col * height + row] = A[row * width + col];
    }
}

/**
 * CUDA kernel for scalar multiplication: B = A * scalar
 */
__global__ void matrixScalarMultiplyKernel(float* A, float* B, float scalar, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int index = row * width + col;
        B[index] = A[index] * scalar;
    }
}

/**
 * Initialize matrix with random values
 */
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f; // Random values between 0-10
    }
}

/**
 * Print matrix (for small matrices)
 */
void printMatrix(float* matrix, int width, int height, const char* name) {
    printf("\n%s (%dx%d):\n", name, height, width);
    if (width > 10 || height > 10) {
        printf("Matrix too large to display. Showing first 5x5 elements:\n");
        for (int i = 0; i < min(5, height); i++) {
            for (int j = 0; j < min(5, width); j++) {
                printf("%6.2f ", matrix[i * width + j]);
            }
            printf("...\n");
        }
        printf("...\n");
    } else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%6.2f ", matrix[i * width + j]);
            }
            printf("\n");
        }
    }
}

/**
 * CPU matrix addition for verification
 */
void cpuMatrixAdd(float* A, float* B, float* C, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            C[index] = A[index] + B[index];
        }
    }
}

/**
 * CPU matrix multiplication for verification
 */
void cpuMatrixMultiply(float* A, float* B, float* C, int widthA, int heightA, int widthB) {
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < widthA; k++) {
                sum += A[i * widthA + k] * B[k * widthB + j];
            }
            C[i * widthB + j] = sum;
        }
    }
}

/**
 * Verify results between CPU and GPU
 */
bool verifyResults(float* cpuResult, float* gpuResult, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

/**
 * Matrix addition operation
 */
void matrixAdd(float* h_A, float* h_B, float* h_C, int width, int height) {
    printf("\n=== Matrix Addition ===\n");
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, width * height * sizeof(float));
    cudaMalloc((void**)&d_B, width * height * sizeof(float));
    cudaMalloc((void**)&d_C, width * height * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU verification
    float* h_C_cpu = (float*)malloc(width * height * sizeof(float));
    clock_t cpuStart = clock();
    cpuMatrixAdd(h_A, h_B, h_C_cpu, width, height);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;
    
    // Verify results
    bool correct = verifyResults(h_C_cpu, h_C, width * height);
    
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C_cpu);
}

/**
 * Matrix multiplication operation
 */
void matrixMultiply(float* h_A, float* h_B, float* h_C, int widthA, int heightA, int widthB) {
    printf("\n=== Matrix Multiplication ===\n");
    printf("Matrix A: %dx%d, Matrix B: %dx%d, Result: %dx%d\n", 
           heightA, widthA, widthA, widthB, heightA, widthB);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, widthA * heightA * sizeof(float));
    cudaMalloc((void**)&d_B, widthA * widthB * sizeof(float));
    cudaMalloc((void**)&d_C, heightA * widthB * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, widthA * heightA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((widthB + blockDim.x - 1) / blockDim.x, (heightA + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, widthA, heightA, widthB);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU verification
    float* h_C_cpu = (float*)malloc(heightA * widthB * sizeof(float));
    clock_t cpuStart = clock();
    cpuMatrixMultiply(h_A, h_B, h_C_cpu, widthA, heightA, widthB);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;
    
    // Verify results
    bool correct = verifyResults(h_C_cpu, h_C, heightA * widthB);
    
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_C_cpu);
}

/**
 * Matrix transpose operation
 */
void matrixTranspose(float* h_A, float* h_B, int width, int height) {
    printf("\n=== Matrix Transpose ===\n");
    printf("Original: %dx%d, Transposed: %dx%d\n", height, width, width, height);
    
    // Allocate device memory
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, width * height * sizeof(float));
    cudaMalloc((void**)&d_B, width * height * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixTransposeKernel<<<gridDim, blockDim>>>(d_A, d_B, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_B, d_B, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU Time: %.3f ms\n", gpuTime);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Matrix Operations\n");
    printf("=====================\n");
    printf("Matrix Size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Block Size: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // Initialize random seed
    srand(time(NULL));
    
    // Allocate host memory
    int size = MATRIX_SIZE * MATRIX_SIZE;
    float *h_A = (float*)malloc(size * sizeof(float));
    float *h_B = (float*)malloc(size * sizeof(float));
    float *h_C = (float*)malloc(size * sizeof(float));
    float *h_D = (float*)malloc(size * sizeof(float));
    
    // Initialize matrices
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);
    
    // Print sample of matrices
    printMatrix(h_A, MATRIX_SIZE, MATRIX_SIZE, "Matrix A");
    printMatrix(h_B, MATRIX_SIZE, MATRIX_SIZE, "Matrix B");
    
    // Perform matrix operations
    matrixAdd(h_A, h_B, h_C, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix(h_C, MATRIX_SIZE, MATRIX_SIZE, "Result (A + B)");
    
    matrixMultiply(h_A, h_B, h_C, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix(h_C, MATRIX_SIZE, MATRIX_SIZE, "Result (A * B)");
    
    matrixTranspose(h_A, h_D, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix(h_D, MATRIX_SIZE, MATRIX_SIZE, "Result (A^T)");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    
    printf("\nMatrix operations completed successfully!\n");
    return 0;
} 