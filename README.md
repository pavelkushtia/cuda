# CUDA Applications Suite

This repository contains a comprehensive collection of CUDA applications demonstrating various GPU programming concepts and optimization techniques using NVIDIA's CUDA platform.

## Overview

The application suite includes:
- **hello_world.cu**: Basic CUDA application that prints "Hello World" from both CPU and GPU
- **device_info.cu**: Utility to display detailed information about CUDA devices
- **matrix_ops.cu**: Basic matrix operations (addition, multiplication, transpose)
- **matrix_ops_advanced.cu**: Advanced matrix operations with optimizations
- **Makefile**: Automated build system with multiple targets
- **Build System**: Organized build directory with proper gitignore

## Applications

### 1. Hello World Application
- Demonstrates basic CUDA kernel execution
- Shows device information and capabilities
- Includes comprehensive error checking
- Proper resource management and cleanup

### 2. Device Information Utility
- Displays detailed GPU specifications
- Shows compute capability, memory, and performance metrics
- Helps diagnose CUDA installation issues
- Useful for hardware compatibility checking

### 3. Matrix Operations (Basic)
- **Matrix Addition**: C = A + B
- **Matrix Multiplication**: C = A × B
- **Matrix Transpose**: B = A^T
- **Performance Comparison**: CPU vs GPU timing
- **Error Verification**: Validates results against CPU computation

### 4. Matrix Operations (Advanced)
- **Optimized Matrix Addition**: Using coalesced memory access
- **Shared Memory Matrix Multiplication**: Enhanced performance with tile-based approach
- **Matrix Scalar Operations**: Multiply, Add, Subtract, Divide by scalar
- **Element-wise Operations**: Add, Subtract, Multiply, Divide matrices
- **Advanced Optimizations**: Shared memory, bank conflict avoidance

## Performance Highlights

Based on testing with 1024×1024 matrices on RTX 2060 SUPER:

- **Matrix Multiplication**: Up to 2223x speedup over CPU
- **Matrix Transpose**: ~0.049ms for 1024×1024 matrix
- **Scalar Operations**: ~0.032-0.038ms for element-wise operations
- **Memory Optimizations**: Shared memory, coalesced access, bank conflict avoidance

## Features

- Demonstrates basic and advanced CUDA kernel execution
- Includes comprehensive error checking and validation
- Shows device information and capabilities
- Proper resource management and cleanup
- Well-documented code with examples
- Performance benchmarking and comparison
- Optimized memory access patterns
- Shared memory utilization
- Bank conflict avoidance techniques

## Prerequisites

### CUDA Toolkit Installation

1. **Download CUDA Toolkit**: Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. **Install for your platform**:
   - **macOS**: Download and run the installer
   - **Linux**: Follow distribution-specific instructions
   - **Windows**: Download and run the installer

3. **Verify Installation**:
   ```bash
   nvcc --version
   ```

### Hardware Requirements

- NVIDIA GPU with CUDA support
- Compute capability 2.0 or higher (most modern GPUs)
- Sufficient GPU memory for matrix operations (recommended: 4GB+)

## Building and Running

### Quick Start

```bash
# Build all applications
make all

# Run specific applications
make run              # Hello World
make device-info      # Device information
make matrix-ops       # Basic matrix operations
make matrix-ops-adv   # Advanced matrix operations
```

### Manual Build

```bash
# Compile individual applications
nvcc -O2 -arch=sm_75 -o hello_world hello_world.cu
nvcc -o device_info device_info.cu
nvcc -o matrix_ops matrix_ops.cu
nvcc -o matrix_ops_advanced matrix_ops_advanced.cu

# Run applications
./hello_world
./device_info
./matrix_ops
./matrix_ops_advanced
```

### Makefile Targets

```bash
make all              # Build all CUDA applications
make run              # Build and run Hello World
make device-info      # Build and run device info utility
make matrix-ops       # Build and run basic matrix operations
make matrix-ops-adv   # Build and run advanced matrix operations
make clean            # Remove build artifacts
make check-cuda       # Check if CUDA is properly installed
make help             # Show all available targets
```

## Expected Output

### Hello World Application
```
Hello World from CPU!
Found 1 CUDA device(s)
Using device: NVIDIA GeForce RTX 2060 SUPER
Compute capability: 7.5
Number of multiprocessors: 34

Launching kernel with 2 blocks, 4 threads per block...
Hello World from GPU! Block: 0, Thread: 0
Hello World from GPU! Block: 0, Thread: 1
Hello World from GPU! Block: 0, Thread: 2
Hello World from GPU! Block: 0, Thread: 3
Hello World from GPU! Block: 1, Thread: 0
Hello World from GPU! Block: 1, Thread: 1
Hello World from GPU! Block: 1, Thread: 2
Hello World from GPU! Block: 1, Thread: 3

Kernel execution completed successfully!
```

### Matrix Operations
```
=== Matrix Multiplication (Shared Memory) ===
Matrix A: 1024x1024, Matrix B: 1024x1024, Result: 1024x1024
GPU Time: 2.737 ms
CPU Time: 6084.676 ms
Speedup: 2223.02x
Result: CORRECT
```

## Code Structure

### hello_world.cu
- **`helloFromGPU()`**: CUDA kernel that runs on the GPU
- **`checkCudaError()`**: Error checking utility function
- **`main()`**: Main function demonstrating CUDA operations

### device_info.cu
- **`printDeviceInfo()`**: Displays detailed GPU information
- **`main()`**: Queries and displays device properties

### matrix_ops.cu
- **`matrixAddKernel()`**: Matrix addition kernel
- **`matrixMultiplyKernel()`**: Matrix multiplication kernel
- **`matrixTransposeKernel()`**: Matrix transpose kernel
- **`verifyResults()`**: CPU vs GPU result validation

### matrix_ops_advanced.cu
- **`matrixAddOptimizedKernel()`**: Optimized addition with coalesced access
- **`matrixMultiplySharedKernel()`**: Shared memory multiplication
- **`matrixTransposeSharedKernel()`**: Optimized transpose with bank conflict avoidance
- **`matrixScalarOpsKernel()`**: Scalar operations kernel

### Key Concepts Demonstrated

1. **Kernel Launch**: Using `<<<blocks, threads>>>` syntax
2. **Error Handling**: Comprehensive CUDA error checking
3. **Device Information**: Querying GPU properties
4. **Synchronization**: Ensuring GPU operations complete
5. **Resource Management**: Proper cleanup of GPU resources
6. **Memory Optimization**: Shared memory, coalesced access
7. **Performance Tuning**: Block/grid dimension optimization
8. **Result Validation**: CPU vs GPU comparison

## Build System

### Directory Structure
```
cuda/
├── hello_world.cu          # Basic CUDA Hello World
├── device_info.cu          # GPU information utility
├── matrix_ops.cu           # Basic matrix operations
├── matrix_ops_advanced.cu  # Advanced optimized operations
├── Makefile                # Enhanced build system
├── .gitignore              # Excludes build artifacts
├── README.md               # This file
├── MATRIX_OPERATIONS_README.md  # Detailed matrix operations guide
└── build/                  # Compiled executables (git-ignored)
    ├── hello_world
    ├── device_info
    ├── matrix_ops
    └── matrix_ops_advanced
```

### Build Configuration
- **Build Directory**: All outputs go to `build/` directory
- **Git Integration**: Build artifacts are properly ignored
- **Clean Targets**: Easy cleanup with `make clean`
- **Modular Build**: Build individual or all applications

## Configuration

### Matrix Size
Modify the `MATRIX_SIZE` constant in matrix operation files:
```cuda
#define MATRIX_SIZE 1024  // Change to desired size
```

### Block and Tile Sizes
```cuda
#define BLOCK_SIZE 16     // Thread block size
#define TILE_SIZE 16      // Shared memory tile size
```

### Architecture Target
The Makefile uses `-arch=sm_75` for RTX 2000 series. Adjust based on your GPU:
- **GTX 900 series**: Use `-arch=sm_52`
- **GTX 1000 series**: Use `-arch=sm_61`
- **RTX 2000 series**: Use `-arch=sm_75`
- **RTX 3000 series**: Use `-arch=sm_86`
- **RTX 4000 series**: Use `-arch=sm_89`

## Troubleshooting

### Common Issues

1. **"nvcc not found"**:
   - Install CUDA Toolkit
   - Add CUDA to your PATH

2. **"No CUDA-capable devices found"**:
   - Ensure you have an NVIDIA GPU
   - Install proper NVIDIA drivers

3. **"CUDA driver version is insufficient"**:
   - Update NVIDIA drivers
   - Check CUDA version compatibility

4. **"Out of Memory"**:
   - Reduce matrix size
   - Check available GPU memory

### Debugging

Use the device info utility to diagnose issues:

```bash
make device-info
```

This will show detailed information about your CUDA devices and help identify compatibility issues.

## Performance Optimization

### Memory Access Patterns
- **Coalesced Access**: Optimize global memory access
- **Shared Memory**: Reduce global memory traffic
- **Bank Conflicts**: Avoid shared memory bank conflicts

### Kernel Configuration
- **Block Size**: Optimize for your GPU's compute capability
- **Grid Size**: Ensure sufficient parallelism
- **Memory Usage**: Balance shared memory usage

## Future Enhancements

### Planned Features
1. **Matrix Determinant**: For small matrices
2. **Matrix Inverse**: Using LU decomposition
3. **Matrix Rank**: Row reduction algorithm
4. **Eigenvalue Computation**: Power iteration method
5. **SVD Decomposition**: Singular Value Decomposition
6. **Multi-GPU Support**: Distribute work across multiple GPUs

### Advanced Optimizations
1. **cuBLAS Integration**: Use NVIDIA's optimized BLAS library
2. **Stream Processing**: Overlap computation and memory transfers
3. **Mixed Precision**: Use FP16 for better performance
4. **Tensor Cores**: Utilize RTX tensor cores for matrix operations

## Contributing

Feel free to submit issues and enhancement requests!

### Development Guidelines
1. Follow existing code style and structure
2. Include performance benchmarks for new features
3. Add comprehensive error checking
4. Update documentation for new functionality
5. Test on multiple GPU architectures

## Documentation

- **README.md**: This comprehensive guide
- **MATRIX_OPERATIONS_README.md**: Detailed matrix operations documentation
- **Code Comments**: Inline documentation in source files
- **Makefile**: Build system documentation

## License

This project is licensed under the terms specified in the LICENSE file.
