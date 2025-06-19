# CUDA Matrix Operations

This repository now includes comprehensive matrix operations implemented in CUDA, demonstrating various optimization techniques and mathematical operations.

## Programs Overview

### 1. Basic Matrix Operations (`matrix_ops.cu`)
- **Matrix Addition**: C = A + B
- **Matrix Multiplication**: C = A × B  
- **Matrix Transpose**: B = A^T
- **Performance Comparison**: CPU vs GPU timing

### 2. Advanced Matrix Operations (`matrix_ops_advanced.cu`)
- **Optimized Matrix Addition**: Using coalesced memory access
- **Shared Memory Matrix Multiplication**: Enhanced performance with tile-based approach
- **Matrix Scalar Operations**: Multiply, Add, Subtract, Divide by scalar
- **Element-wise Operations**: Add, Subtract, Multiply, Divide matrices
- **Advanced Optimizations**: Shared memory, bank conflict avoidance

## Performance Results

Based on testing with 1024×1024 matrices on RTX 2060 SUPER:

### Matrix Addition
- **Basic Version**: GPU slower than CPU (memory transfer overhead)
- **Optimized Version**: Better performance with coalesced access
- **Result**: CORRECT

### Matrix Multiplication
- **Basic Version**: ~945x speedup over CPU
- **Shared Memory Version**: ~2223x speedup over CPU
- **Result**: CORRECT (minor floating-point precision differences)

### Matrix Transpose
- **Performance**: ~0.049ms for 1024×1024 matrix
- **Optimization**: Shared memory with bank conflict avoidance

### Scalar Operations
- **Scalar Multiply**: ~0.038ms
- **Scalar Add**: ~0.032ms
- **Performance**: Excellent for element-wise operations

## Key Features

### 1. **Shared Memory Optimization**
```cuda
__shared__ float sA[TILE_SIZE][TILE_SIZE];
__shared__ float sB[TILE_SIZE][TILE_SIZE];
```
- Reduces global memory access
- Improves memory bandwidth utilization
- Significant performance gains for matrix multiplication

### 2. **Coalesced Memory Access**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < width * height; i += stride) {
    C[i] = A[i] + B[i];
}
```
- Optimizes memory access patterns
- Better for simple operations like addition

### 3. **Bank Conflict Avoidance**
```cuda
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
```
- Prevents memory bank conflicts in shared memory
- Improves transpose performance

### 4. **Comprehensive Error Checking**
- CPU vs GPU result verification
- Configurable tolerance for floating-point comparisons
- Detailed performance reporting

## Usage

### Building
```bash
# Build all programs
make all

# Build specific program
make matrix-ops
make matrix-ops-adv
```

### Running
```bash
# Run basic matrix operations
make matrix-ops

# Run advanced matrix operations
make matrix-ops-adv
```

### Available Commands
```bash
make all              # Build all programs
make matrix-ops       # Build and run basic matrix operations
make matrix-ops-adv   # Build and run advanced matrix operations
make clean            # Clean build artifacts
make help             # Show all available commands
```

## Configuration

### Matrix Size
Modify the `MATRIX_SIZE` constant in the source files:
```cuda
#define MATRIX_SIZE 1024  // Change to desired size
```

### Block and Tile Sizes
```cuda
#define BLOCK_SIZE 16     // Thread block size
#define TILE_SIZE 16      // Shared memory tile size
```

### Performance Tuning
- **For larger matrices**: Increase `TILE_SIZE` (must be ≤ `BLOCK_SIZE`)
- **For memory-bound operations**: Optimize memory access patterns
- **For compute-bound operations**: Increase thread occupancy

## Future Enhancements

### Planned Features
1. **Matrix Determinant**: For small matrices (2×2, 3×3)
2. **Matrix Inverse**: Using LU decomposition
3. **Matrix Rank**: Row reduction algorithm
4. **Eigenvalue Computation**: Power iteration method
5. **SVD Decomposition**: Singular Value Decomposition
6. **Cholesky Decomposition**: For symmetric positive definite matrices

### Advanced Optimizations
1. **cuBLAS Integration**: Use NVIDIA's optimized BLAS library
2. **Stream Processing**: Overlap computation and memory transfers
3. **Multi-GPU Support**: Distribute work across multiple GPUs
4. **Mixed Precision**: Use FP16 for better performance
5. **Tensor Cores**: Utilize RTX tensor cores for matrix operations

## Technical Details

### Memory Management
- **Host Memory**: Allocated with `malloc()`
- **Device Memory**: Allocated with `cudaMalloc()`
- **Memory Transfers**: Optimized with `cudaMemcpy()`
- **Cleanup**: Proper deallocation with `cudaFree()`

### Kernel Launch Configuration
```cuda
// For 2D operations
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
             (height + blockDim.y - 1) / blockDim.y);

// For 1D operations
int blockSize = 256;
int gridSize = (totalElements + blockSize - 1) / blockSize;
```

### Error Handling
```cuda
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
}
```

## Performance Analysis

### Factors Affecting Performance
1. **Memory Bandwidth**: Critical for matrix operations
2. **Compute Intensity**: Operations per memory access
3. **Thread Occupancy**: Number of active warps
4. **Memory Coalescing**: Access pattern efficiency
5. **Shared Memory Usage**: Reduces global memory access

### Optimization Guidelines
1. **Use shared memory** for frequently accessed data
2. **Ensure coalesced memory access** for global memory
3. **Avoid bank conflicts** in shared memory
4. **Maximize thread occupancy** within hardware limits
5. **Overlap computation and memory transfers** using streams

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (Compute Capability ≥ 3.0)
- **Memory**: Sufficient GPU memory for matrix storage
- **Driver**: Compatible NVIDIA driver
- **CUDA Toolkit**: Version 11.0 or later recommended

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce matrix size or use multiple GPU calls
2. **Kernel Launch Failures**: Check block/grid dimensions
3. **Incorrect Results**: Verify matrix dimensions and memory layout
4. **Poor Performance**: Profile with NVIDIA Visual Profiler

### Debugging Tips
1. Use `cuda-gdb` for kernel debugging
2. Enable error checking with `cudaGetLastError()`
3. Use `cuda-memcheck` for memory error detection
4. Profile with `nvprof` or Nsight Compute

## Contributing

Feel free to contribute additional matrix operations or optimizations:
1. Fork the repository
2. Add new operations or optimizations
3. Include performance benchmarks
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file. 