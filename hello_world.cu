/**
 * Hello World CUDA Application
 * 
 * This program demonstrates basic CUDA functionality by:
 * 1. Allocating memory on the GPU
 * 2. Running a simple kernel that prints "Hello World from GPU!"
 * 3. Properly cleaning up GPU resources
 * 
 * Compile with: nvcc -o hello_world hello_world.cu
 * Run with: ./hello_world
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * CUDA kernel that prints "Hello World from GPU!"
 * This kernel runs on the GPU and each thread prints the message.
 * 
 * @param blockIdx.x - Block index in x dimension
 * @param threadIdx.x - Thread index within the block
 */
__global__ void helloFromGPU() {
    printf("Hello World from GPU! Block: %d, Thread: %d\n", 
           blockIdx.x, threadIdx.x);
}

/**
 * Check for CUDA errors and print error message if any
 * 
 * @param error - CUDA error code
 * @param msg - Error message to display
 */
void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

/**
 * Main function that demonstrates basic CUDA operations
 */
int main() {
    printf("Hello World from CPU!\n");
    
    // Check if CUDA is available
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "cudaGetDeviceCount");
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    // Get device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    checkCudaError(error, "cudaGetDeviceProperties");
    
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    
    // Launch kernel with 2 blocks, each containing 4 threads
    // This will result in 8 total threads (2 * 4)
    printf("\nLaunching kernel with 2 blocks, 4 threads per block...\n");
    helloFromGPU<<<2, 4>>>();
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    checkCudaError(error, "kernel launch");
    
    // Synchronize to ensure all GPU operations are complete
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize");
    
    printf("\nKernel execution completed successfully!\n");
    
    return EXIT_SUCCESS;
} 