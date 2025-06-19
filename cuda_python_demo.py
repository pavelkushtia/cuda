#!/usr/bin/env python3
"""
CUDA Python Libraries Demo
==========================

This script demonstrates practical usage of CUDA Python libraries
with best practices for performance optimization.
"""

import numpy as np
import time

def demo_cupy_usage():
    """Demonstrate effective CuPy usage."""
    print("=== CuPy Demo ===")
    
    try:
        import cupy as cp
        
        # Create large matrices
        size = 3000
        print(f"Creating {size}x{size} matrices...")
        
        # Generate data on CPU first
        a_cpu = np.random.random((size, size)).astype(np.float32)
        b_cpu = np.random.random((size, size)).astype(np.float32)
        
        # Transfer to GPU (this is the overhead)
        print("Transferring data to GPU...")
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)
        
        # Perform multiple operations on GPU (amortize transfer cost)
        print("Performing operations on GPU...")
        
        # Matrix multiplication
        start_time = time.time()
        c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        mm_time = time.time() - start_time
        
        # Element-wise operations
        start_time = time.time()
        d_gpu = a_gpu + b_gpu
        e_gpu = a_gpu * b_gpu
        f_gpu = cp.sin(a_gpu)
        cp.cuda.Stream.null.synchronize()
        elem_time = time.time() - start_time
        
        # Transfer results back to CPU
        print("Transferring results back to CPU...")
        c_cpu = cp.asnumpy(c_gpu)
        d_cpu = cp.asnumpy(d_gpu)
        
        print(f"Matrix multiplication: {mm_time:.4f}s")
        print(f"Element-wise operations: {elem_time:.4f}s")
        print(f"Total GPU operations completed successfully!")
        
    except ImportError:
        print("CuPy not available")

def demo_pytorch_usage():
    """Demonstrate effective PyTorch usage."""
    print("\n=== PyTorch Demo ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available in PyTorch")
            return
        
        # Create tensors directly on GPU (avoid transfer overhead)
        size = 3000
        print(f"Creating {size}x{size} tensors directly on GPU...")
        
        a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        # Perform operations
        print("Performing operations...")
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        mm_time = time.time() - start_time
        
        start_time = time.time()
        d_gpu = a_gpu + b_gpu
        e_gpu = torch.relu(a_gpu)  # Neural network activation
        torch.cuda.synchronize()
        elem_time = time.time() - start_time
        
        print(f"Matrix multiplication: {mm_time:.4f}s")
        print(f"Element-wise operations: {elem_time:.4f}s")
        print(f"PyTorch operations completed successfully!")
        
    except ImportError:
        print("PyTorch not available")

def demo_numba_usage():
    """Demonstrate effective Numba usage."""
    print("\n=== Numba Demo ===")
    
    try:
        from numba import cuda, jit
        
        if not cuda.is_available():
            print("CUDA not available in Numba")
            return
        
        # Define a custom CUDA kernel
        @cuda.jit
        def matrix_add_gpu(a, b, result):
            row, col = cuda.grid(2)
            if row < result.shape[0] and col < result.shape[1]:
                result[row, col] = a[row, col] + b[row, col]
        
        # Define CPU version for comparison
        @jit(nopython=True)
        def matrix_add_cpu(a, b, result):
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i, j] = a[i, j] + b[i, j]
        
        # Test with large matrices
        size = 2000
        print(f"Testing {size}x{size} matrix addition...")
        
        # Create test data
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
        result_cpu = np.zeros_like(a)
        result_gpu = np.zeros_like(a)
        
        # CPU computation
        start_time = time.time()
        matrix_add_cpu(a, b, result_cpu)
        cpu_time = time.time() - start_time
        
        # GPU computation
        threadsperblock = (16, 16)
        blockspergrid_x = (size + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (size + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Transfer to GPU
        a_gpu = cuda.to_device(a)
        b_gpu = cuda.to_device(b)
        result_gpu_device = cuda.device_array((size, size), dtype=np.float64)
        
        # Launch kernel
        start_time = time.time()
        matrix_add_gpu[blockspergrid, threadsperblock](a_gpu, b_gpu, result_gpu_device)
        cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Transfer back
        result_gpu = result_gpu_device.copy_to_host()
        
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify results
        if np.allclose(result_cpu, result_gpu, rtol=1e-5):
            print("✓ Results match!")
        else:
            print("✗ Results differ!")
            
    except ImportError:
        print("Numba not available")

def demo_best_practices():
    """Demonstrate best practices for CUDA Python."""
    print("\n=== Best Practices Demo ===")
    
    try:
        import cupy as cp
        
        print("1. Batch small operations:")
        
        # Bad: Multiple small transfers
        small_size = 100
        total_time_bad = 0
        for i in range(10):
            a = np.random.random((small_size, small_size))
            start = time.time()
            a_gpu = cp.asarray(a)
            result = cp.dot(a_gpu, a_gpu)
            cp.cuda.Stream.null.synchronize()
            total_time_bad += time.time() - start
        
        # Good: Single large operation
        large_size = 1000
        a = np.random.random((large_size, large_size))
        start = time.time()
        a_gpu = cp.asarray(a)
        result = cp.dot(a_gpu, a_gpu)
        cp.cuda.Stream.null.synchronize()
        total_time_good = time.time() - start
        
        print(f"   Bad (10 small ops): {total_time_bad:.4f}s")
        print(f"   Good (1 large op):  {total_time_good:.4f}s")
        print(f"   Efficiency gain: {total_time_bad/total_time_good:.2f}x")
        
        print("\n2. Use appropriate data types:")
        
        # Float32 vs Float64
        size = 2000
        a_f32 = np.random.random((size, size)).astype(np.float32)
        a_f64 = np.random.random((size, size)).astype(np.float64)
        
        a_gpu_f32 = cp.asarray(a_f32)
        a_gpu_f64 = cp.asarray(a_f64)
        
        start = time.time()
        result_f32 = cp.dot(a_gpu_f32, a_gpu_f32)
        cp.cuda.Stream.null.synchronize()
        time_f32 = time.time() - start
        
        start = time.time()
        result_f64 = cp.dot(a_gpu_f64, a_gpu_f64)
        cp.cuda.Stream.null.synchronize()
        time_f64 = time.time() - start
        
        print(f"   Float32: {time_f32:.4f}s")
        print(f"   Float64: {time_f64:.4f}s")
        print(f"   Float32 is {time_f64/time_f32:.2f}x faster")
        
    except ImportError:
        print("CuPy not available for best practices demo")

def main():
    """Run all demos."""
    print("CUDA Python Libraries Practical Demo")
    print("====================================")
    
    demo_cupy_usage()
    demo_pytorch_usage()
    demo_numba_usage()
    demo_best_practices()
    
    print("\n" + "="*50)
    print("Demo completed! Key takeaways:")
    print("- Use GPU for large operations (>2000x2000)")
    print("- Batch small operations to amortize transfer costs")
    print("- Choose appropriate data types (float32 often sufficient)")
    print("- Consider your specific use case when choosing libraries")

if __name__ == "__main__":
    main() 