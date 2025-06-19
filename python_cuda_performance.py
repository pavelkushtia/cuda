#!/usr/bin/env python3
"""
CUDA Performance Comparison in Python
=====================================

This script demonstrates different approaches to using CUDA in Python
and compares their performance with CPU implementations.
"""

import time
import numpy as np
import sys

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(operation, gpu_time, cpu_time, speedup, correct):
    print(f"{operation:25} | {gpu_time:8.3f}ms | {cpu_time:8.3f}ms | {speedup:8.2f}x | {correct}")

def cpu_matrix_add(a, b):
    """CPU matrix addition"""
    return a + b

def cpu_matrix_multiply(a, b):
    """CPU matrix multiplication"""
    return np.dot(a, b)

def benchmark_cpu_operations():
    """Benchmark CPU operations"""
    print_header("CPU Operations Benchmark")
    
    sizes = [512, 1024, 2048]
    
    for size in sizes:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 50)
        
        # Generate random matrices
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
        
        # Matrix Addition
        start_time = time.time()
        result_add = cpu_matrix_add(a, b)
        cpu_add_time = (time.time() - start_time) * 1000
        
        # Matrix Multiplication
        start_time = time.time()
        result_mult = cpu_matrix_multiply(a, b)
        cpu_mult_time = (time.time() - start_time) * 1000
        
        print(f"Addition:     {cpu_add_time:8.3f}ms")
        print(f"Multiplication: {cpu_mult_time:8.3f}ms")

def check_cuda_availability():
    """Check which CUDA libraries are available"""
    print_header("CUDA Library Availability")
    
    cuda_libs = {
        'Numba': False,
        'CuPy': False,
        'PyTorch': False,
        'PyCUDA': False,
        'TensorFlow': False
    }
    
    # Check Numba
    try:
        import numba
        from numba import cuda
        cuda_libs['Numba'] = True
        print("✅ Numba with CUDA: Available")
    except ImportError:
        print("❌ Numba with CUDA: Not available")
    
    # Check CuPy
    try:
        import cupy as cp
        cuda_libs['CuPy'] = True
        print("✅ CuPy: Available")
    except ImportError:
        print("❌ CuPy: Not available")
    
    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            cuda_libs['PyTorch'] = True
            print(f"✅ PyTorch with CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("❌ PyTorch with CUDA: Not available")
    except ImportError:
        print("❌ PyTorch: Not available")
    
    # Check PyCUDA
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        cuda_libs['PyCUDA'] = True
        print("✅ PyCUDA: Available")
    except ImportError:
        print("❌ PyCUDA: Not available")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            cuda_libs['TensorFlow'] = True
            print("✅ TensorFlow with CUDA: Available")
        else:
            print("❌ TensorFlow with CUDA: Not available")
    except ImportError:
        print("❌ TensorFlow: Not available")
    
    return cuda_libs

def benchmark_numba_cuda():
    """Benchmark Numba CUDA operations"""
    try:
        import numba
        from numba import cuda
        import numpy as np
        
        print_header("Numba CUDA Performance")
        
        # Initialize CUDA context
        cuda.select_device(0)
        
        @cuda.jit
        def matrix_add_gpu(a, b, result):
            row, col = cuda.grid(2)
            if row < result.shape[0] and col < result.shape[1]:
                result[row, col] = a[row, col] + b[row, col]
        
        @cuda.jit
        def matrix_multiply_gpu(a, b, result):
            row, col = cuda.grid(2)
            if row < result.shape[0] and col < result.shape[1]:
                sum_val = 0.0
                for k in range(a.shape[1]):
                    sum_val += a[row, k] * b[k, col]
                result[row, col] = sum_val
        
        size = 1024
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
        
        # GPU setup
        threadsperblock = (16, 16)
        blockspergrid_x = (size + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (size + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Matrix Addition
        result_gpu = np.zeros_like(a)
        start_time = time.time()
        
        # Transfer data to GPU
        a_device = cuda.to_device(a)
        b_device = cuda.to_device(b)
        result_device = cuda.device_array_like(a)
        
        # Launch kernel
        matrix_add_gpu[blockspergrid, threadsperblock](a_device, b_device, result_device)
        cuda.synchronize()
        
        # Copy result back
        result_gpu = result_device.copy_to_host()
        gpu_add_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = cpu_matrix_add(a, b)
        cpu_add_time = (time.time() - start_time) * 1000
        
        # Verify results
        add_correct = np.allclose(result_gpu, result_cpu, rtol=1e-5)
        add_speedup = cpu_add_time / gpu_add_time if gpu_add_time > 0 else 0
        
        print_result("Numba Addition", gpu_add_time, cpu_add_time, add_speedup, add_correct)
        
        # Matrix Multiplication
        result_gpu = np.zeros_like(a)
        start_time = time.time()
        
        # Launch kernel
        result_device = cuda.device_array_like(a)
        matrix_multiply_gpu[blockspergrid, threadsperblock](a_device, b_device, result_device)
        cuda.synchronize()
        
        # Copy result back
        result_gpu = result_device.copy_to_host()
        gpu_mult_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = cpu_matrix_multiply(a, b)
        cpu_mult_time = (time.time() - start_time) * 1000
        
        # Verify results
        mult_correct = np.allclose(result_gpu, result_cpu, rtol=1e-3)
        mult_speedup = cpu_mult_time / gpu_mult_time if gpu_mult_time > 0 else 0
        
        print_result("Numba Multiplication", gpu_mult_time, cpu_mult_time, mult_speedup, mult_correct)
        
        # Clean up
        cuda.close()
        
    except ImportError:
        print("❌ Numba CUDA not available")
    except Exception as e:
        print(f"Error in Numba benchmark: {e}")
        # Clean up CUDA context
        try:
            cuda.close()
        except:
            pass

def benchmark_cupy():
    """Benchmark CuPy operations"""
    try:
        import cupy as cp
        
        print_header("CuPy Performance")
        
        size = 1024
        a_cpu = np.random.random((size, size)).astype(np.float32)
        b_cpu = np.random.random((size, size)).astype(np.float32)
        
        # Transfer to GPU
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)
        
        # Matrix Addition
        start_time = time.time()
        result_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()
        gpu_add_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = cpu_matrix_add(a_cpu, b_cpu)
        cpu_add_time = (time.time() - start_time) * 1000
        
        # Verify results
        result_gpu_cpu = cp.asnumpy(result_gpu)
        add_correct = np.allclose(result_gpu_cpu, result_cpu, rtol=1e-5)
        add_speedup = cpu_add_time / gpu_add_time if gpu_add_time > 0 else 0
        
        print_result("CuPy Addition", gpu_add_time, cpu_add_time, add_speedup, add_correct)
        
        # Matrix Multiplication
        start_time = time.time()
        result_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_mult_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = cpu_matrix_multiply(a_cpu, b_cpu)
        cpu_mult_time = (time.time() - start_time) * 1000
        
        # Verify results
        result_gpu_cpu = cp.asnumpy(result_gpu)
        mult_correct = np.allclose(result_gpu_cpu, result_cpu, rtol=1e-3)
        mult_speedup = cpu_mult_time / gpu_mult_time if gpu_mult_time > 0 else 0
        
        print_result("CuPy Multiplication", gpu_mult_time, cpu_mult_time, mult_speedup, mult_correct)
        
    except ImportError:
        print("❌ CuPy not available")

def benchmark_pytorch():
    """Benchmark PyTorch CUDA operations"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ PyTorch CUDA not available")
            return
        
        print_header("PyTorch CUDA Performance")
        
        size = 1024
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)
        
        # Move to GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Matrix Addition
        torch.cuda.synchronize()
        start_time = time.time()
        result_gpu = a_gpu + b_gpu
        torch.cuda.synchronize()
        gpu_add_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = a_cpu + b_cpu
        cpu_add_time = (time.time() - start_time) * 1000
        
        # Verify results
        result_gpu_cpu = result_gpu.cpu()
        add_correct = torch.allclose(result_gpu_cpu, result_cpu, rtol=1e-5)
        add_speedup = cpu_add_time / gpu_add_time if gpu_add_time > 0 else 0
        
        print_result("PyTorch Addition", gpu_add_time, cpu_add_time, add_speedup, add_correct)
        
        # Matrix Multiplication
        torch.cuda.synchronize()
        start_time = time.time()
        result_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_mult_time = (time.time() - start_time) * 1000
        
        # CPU comparison
        start_time = time.time()
        result_cpu = torch.mm(a_cpu, b_cpu)
        cpu_mult_time = (time.time() - start_time) * 1000
        
        # Verify results
        result_gpu_cpu = result_gpu.cpu()
        mult_correct = torch.allclose(result_gpu_cpu, result_cpu, rtol=1e-3)
        mult_speedup = cpu_mult_time / gpu_mult_time if gpu_mult_time > 0 else 0
        
        print_result("PyTorch Multiplication", gpu_mult_time, cpu_mult_time, mult_speedup, mult_correct)
        
    except ImportError:
        print("❌ PyTorch not available")

def print_performance_summary():
    """Print performance summary and recommendations"""
    print_header("Performance Summary & Recommendations")
    
    print("\n📊 Performance Rankings (Typical):")
    print("1. PyCUDA (Direct CUDA C++)     - 98-99% of native C++")
    print("2. PyTorch CUDA                 - 95-99% of native C++")
    print("3. Numba CUDA                   - 95-98% of native C++")
    print("4. CuPy                         - 90-95% of native C++")
    print("5. TensorFlow GPU               - 90-95% of native C++")
    
    print("\n🎯 Use Cases:")
    print("• PyCUDA:     Maximum performance, direct CUDA control")
    print("• PyTorch:    Deep learning, research, production ML")
    print("• Numba:      Scientific computing, custom kernels")
    print("• CuPy:       NumPy replacement, scientific computing")
    print("• TensorFlow: Deep learning, production ML")
    
    print("\n⚡ Performance Tips:")
    print("• Minimize CPU-GPU memory transfers")
    print("• Use appropriate data types (float32 vs float64)")
    print("• Batch operations when possible")
    print("• Profile with NVIDIA tools (nvprof, Nsight)")
    print("• Consider mixed precision for better performance")

def main():
    """Main function"""
    print_header("CUDA Performance Analysis in Python")
    
    # Check availability
    cuda_libs = check_cuda_availability()
    
    # Print header for results
    print_header("Performance Results")
    print(f"{'Operation':25} | {'GPU Time':8} | {'CPU Time':8} | {'Speedup':8} | {'Correct'}")
    print("-" * 65)
    
    # Run benchmarks
    benchmark_cpu_operations()
    benchmark_numba_cuda()
    benchmark_cupy()
    benchmark_pytorch()
    
    # Print summary
    print_performance_summary()

if __name__ == "__main__":
    main() 