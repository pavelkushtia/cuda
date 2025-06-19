/**
 * CUDA Device Information Utility
 * 
 * This program displays detailed information about available CUDA devices.
 * Useful for debugging CUDA installation and device compatibility.
 * 
 * Compile with: nvcc -o device_info device_info.cu
 * Run with: ./device_info
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Print detailed information about a CUDA device
 * 
 * @param deviceId - The device ID to query
 */
void printDeviceInfo(int deviceId) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    
    if (error != cudaSuccess) {
        printf("Error getting device properties for device %d: %s\n", 
               deviceId, cudaGetErrorString(error));
        return;
    }
    
    printf("\n=== Device %d: %s ===\n", deviceId, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Block Size: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("Unified Memory: %s\n", prop.unifiedAddressing ? "Yes" : "No");
}

int main() {
    printf("CUDA Device Information Utility\n");
    printf("===============================\n");
    
    // Get number of devices
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    
    // Print information for each device
    for (int i = 0; i < deviceCount; i++) {
        printDeviceInfo(i);
    }
    
    // Get current device
    int currentDevice;
    error = cudaGetDevice(&currentDevice);
    if (error == cudaSuccess) {
        printf("\nCurrent device: %d\n", currentDevice);
    }
    
    printf("\nCUDA Runtime Version: %s\n", CUDART_VERSION_STRING);
    
    return 0;
} 