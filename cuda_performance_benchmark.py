#!/usr/bin/env python3
"""
CUDA Performance Benchmark in Python
====================================

This script demonstrates CUDA performance in Python using various libraries:
- NumPy (CPU baseline)
- CuPy (GPU acceleration)
- PyTorch (GPU acceleration)
- Numba (JIT compilation with CUDA)

The script performs matrix operations and measures performance differences.
"""

import time
import numpy as np
import sys
import os

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_result(operation, cpu_time, gpu_time, speedup):
    """Print formatted performance results."""
    print(f"{operation:20s} | CPU: {cpu_time:8.4f}s | GPU: {gpu_time:8.4f}s | Speedup: {speedup:6.2f}x")

def benchmark_numpy_vs_cupy():
    """Benchmark NumPy (CPU) vs CuPy (GPU) performance."""
    print_header("NumPy (CPU) vs CuPy (GPU) Performance Comparison")
    
    try:
        import cupy as cp
        
        # Test different matrix sizes
        sizes = [1000, 2000, 4000]
        
        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")
            print("-" * 50)
            
            # Create test matrices
            a_cpu = np.random.random((size, size)).astype(np.float32)
            b_cpu = np.random.random((size, size)).astype(np.float32)
            a_gpu = cp.asarray(a_cpu)
            b_gpu = cp.asarray(b_cpu)
            
            # Matrix multiplication
            # CPU
            start_time = time.time()
            c_cpu = np.dot(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            # GPU
            start_time = time.time()
            c_gpu = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Matrix Multiply", cpu_time, gpu_time, speedup)
            
            # Element-wise addition
            # CPU
            start_time = time.time()
            d_cpu = a_cpu + b_cpu
            cpu_time = time.time() - start_time
            
            # GPU
            start_time = time.time()
            d_gpu = a_gpu + b_gpu
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Element-wise Add", cpu_time, gpu_time, speedup)
            
            # Matrix transpose
            # CPU
            start_time = time.time()
            e_cpu = a_cpu.T
            cpu_time = time.time() - start_time
            
            # GPU
            start_time = time.time()
            e_gpu = a_gpu.T
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Matrix Transpose", cpu_time, gpu_time, speedup)
            
    except ImportError:
        print("CuPy not available. Skipping CuPy benchmarks.")
    except Exception as e:
        print(f"Error in CuPy benchmark: {e}")

def benchmark_pytorch():
    """Benchmark PyTorch CPU vs GPU performance."""
    print_header("PyTorch CPU vs GPU Performance Comparison")
    
    try:
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available in PyTorch. Skipping PyTorch GPU benchmarks.")
            return
        
        print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Test different matrix sizes
        sizes = [1000, 2000, 4000]
        
        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")
            print("-" * 50)
            
            # Create test tensors
            a_cpu = torch.randn(size, size, dtype=torch.float32)
            b_cpu = torch.randn(size, size, dtype=torch.float32)
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            
            # Matrix multiplication
            # CPU
            start_time = time.time()
            c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            # GPU
            torch.cuda.synchronize()  # Ensure GPU is ready
            start_time = time.time()
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()  # Ensure GPU computation is complete
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Matrix Multiply", cpu_time, gpu_time, speedup)
            
            # Element-wise addition
            # CPU
            start_time = time.time()
            d_cpu = a_cpu + b_cpu
            cpu_time = time.time() - start_time
            
            # GPU
            torch.cuda.synchronize()
            start_time = time.time()
            d_gpu = a_gpu + b_gpu
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Element-wise Add", cpu_time, gpu_time, speedup)
            
    except ImportError:
        print("PyTorch not available. Skipping PyTorch benchmarks.")
    except Exception as e:
        print(f"Error in PyTorch benchmark: {e}")

def benchmark_numba():
    """Benchmark Numba JIT compilation with CUDA."""
    print_header("Numba CUDA JIT Compilation Performance")
    
    try:
        from numba import cuda, jit
        import math
        
        # Check if CUDA is available
        if not cuda.is_available():
            print("CUDA not available in Numba. Skipping Numba CUDA benchmarks.")
            return
        
        print(f"Numba CUDA device: {cuda.get_current_device().name}")
        
        # Define CUDA kernel for vector addition
        @cuda.jit
        def vector_add_gpu(a, b, result):
            idx = cuda.grid(1)
            if idx < result.size:
                result[idx] = a[idx] + b[idx]
        
        # Define CPU function for comparison
        @jit(nopython=True)
        def vector_add_cpu(a, b, result):
            for i in range(len(result)):
                result[i] = a[i] + b[i]
        
        # Test different vector sizes
        sizes = [1000000, 5000000, 10000000]
        
        for size in sizes:
            print(f"\nVector size: {size:,}")
            print("-" * 50)
            
            # Create test data
            a = np.random.random(size).astype(np.float32)
            b = np.random.random(size).astype(np.float32)
            result_cpu = np.zeros_like(a)
            result_gpu = np.zeros_like(a)
            
            # CPU computation
            start_time = time.time()
            vector_add_cpu(a, b, result_cpu)
            cpu_time = time.time() - start_time
            
            # GPU computation
            # Configure the blocks
            threadsperblock = 256
            blockspergrid = (size + (threadsperblock - 1)) // threadsperblock
            
            # Copy data to GPU
            a_gpu = cuda.to_device(a)
            b_gpu = cuda.to_device(b)
            result_gpu_device = cuda.device_array(size, dtype=np.float64)
            
            # Launch kernel
            start_time = time.time()
            vector_add_gpu(blockspergrid, threadsperblock)(a_gpu, b_gpu, result_gpu_device)
            cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # Copy result back to CPU
            result_gpu = result_gpu_device.copy_to_host()
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print_result("Vector Addition", cpu_time, gpu_time, speedup)
            
            # Verify results are the same
            if np.allclose(result_cpu, result_gpu, rtol=1e-5):
                print("✓ Results match between CPU and GPU")
            else:
                print("✗ Results differ between CPU and GPU")
                
    except ImportError:
        print("Numba not available. Skipping Numba benchmarks.")
    except Exception as e:
        print(f"Error in Numba benchmark: {e}")

def benchmark_memory_transfer():
    """Benchmark memory transfer overhead."""
    print_header("Memory Transfer Overhead Analysis")
    
    try:
        import cupy as cp
        
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")
            print("-" * 50)
            
            # Create test matrix
            a_cpu = np.random.random((size, size)).astype(np.float32)
            
            # Measure CPU to GPU transfer
            start_time = time.time()
            a_gpu = cp.asarray(a_cpu)
            cp.cuda.Stream.null.synchronize()
            transfer_time = time.time() - start_time
            
            # Measure GPU to CPU transfer
            start_time = time.time()
            a_cpu_back = cp.asnumpy(a_gpu)
            transfer_back_time = time.time() - start_time
            
            print(f"CPU → GPU transfer: {transfer_time:8.4f}s")
            print(f"GPU → CPU transfer: {transfer_back_time:8.4f}s")
            
    except ImportError:
        print("CuPy not available. Skipping memory transfer benchmarks.")
    except Exception as e:
        print(f"Error in memory transfer benchmark: {e}")

def main():
    """Main function to run all benchmarks."""
    print("CUDA Performance Benchmark in Python")
    print("=====================================")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    # Run all benchmarks
    benchmark_numpy_vs_cupy()
    benchmark_pytorch()
    benchmark_numba()
    benchmark_memory_transfer()
    
    print_header("Summary")
    print("""
Performance Tips:
- GPU acceleration is most beneficial for large matrices/vectors
- Memory transfer overhead can negate benefits for small operations
- Use batch processing to amortize transfer costs
- Consider mixed precision (FP16) for additional speedup
- Profile your specific use case as performance varies by operation type
    """)

if __name__ == "__main__":
    main() 