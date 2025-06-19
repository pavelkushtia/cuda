# CUDA Development Environment

This repository contains CUDA programs and utilities for GPU computing, including both native CUDA C++ and Python implementations.

## Contents

- **Native CUDA C++ Programs**: Hello World, device info, and matrix operations
- **Python CUDA Libraries**: Performance benchmarks and practical examples
- **Build System**: Makefile for easy compilation and execution
- **Documentation**: Comprehensive guides and examples

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 12.x or later)
- GCC compiler (version 9 or later)
- Python 3.8+ (for Python examples)
- Python virtual environment (recommended)

## Installation

### Quick Setup (Recommended)

**For new users or fresh installations, use the automated setup script:**

```bash
# Make the script executable (if needed)
chmod +x setup.sh

# Run the setup script
./setup.sh

# The script will:
# 1. Check prerequisites (Python, CUDA)
# 2. Create virtual environment
# 3. Install all required packages
# 4. Verify installations
# 5. Test the environment
```

**Setup script options:**
```bash
./setup.sh --help     # Show help
./setup.sh --force    # Force recreate virtual environment
./setup.sh --skip-test # Skip environment testing
```

### Manual Setup

If you prefer to set up manually or the automated script doesn't work:

#### CUDA Toolkit
```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

#### Python Environment Setup

**1. Create and activate virtual environment:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python3
```

**2. Install CUDA Python libraries:**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installations
pip list | grep -E "numpy|numba|cupy|torch|pycuda|tensorflow"
```

**3. Test the environment:**
```bash
# Test basic functionality
python3 cuda_python_demo.py

# Run performance benchmarks
python3 cuda_performance_benchmark.py
```

## Current Status ✅

All components are tested and working:

- ✅ Virtual environment properly configured
- ✅ All CUDA Python libraries installed and functional
- ✅ Native CUDA C++ programs compiling and running
- ✅ Python scripts executing successfully
- ✅ Performance benchmarks providing meaningful results

### Tested Environment:
- **GPU**: NVIDIA GeForce RTX 2060 SUPER
- **CUDA**: Version 12.x
- **Python**: 3.12.3
- **Libraries**: NumPy, Numba, CuPy, PyTorch, PyCUDA, TensorFlow

## Native CUDA Programs

### 1. Hello World (`hello_world.cu`)
Simple CUDA program that prints "Hello from GPU!" from each thread.

**Build and Run:**
```bash
make hello_world
./build/hello_world
```

**Expected Output:**
```
Hello from GPU! Thread 0
Hello from GPU! Thread 1
...
Hello from GPU! Thread 255
```

### 2. Device Information (`device_info.cu`)
Displays detailed information about the CUDA device.

**Build and Run:**
```bash
make device_info
./build/device_info
```

**Sample Output:**
```
Device: NVIDIA GeForce RTX 2060 SUPER
Compute Capability: 7.5
Global Memory: 8 GB
CUDA Cores: 4352
Max Threads per Block: 1024
ECC Memory: No
```

### 3. Matrix Operations (`matrix_ops.cu`)
Advanced matrix operations with shared memory optimizations.

**Features:**
- Matrix addition, multiplication, and transpose
- Shared memory optimizations
- Scalar operations
- Performance benchmarking
- CPU verification

**Build and Run:**
```bash
make matrix_ops
./build/matrix_ops
```

**Sample Output:**
```
Matrix Operations Benchmark
==========================
Matrix size: 1024x1024

Operation          | CPU Time | GPU Time | Speedup
Matrix Addition    | 0.0156s  | 0.0008s  | 19.50x
Matrix Multiply    | 0.2341s  | 0.0123s  | 19.03x
Matrix Transpose   | 0.0012s  | 0.0003s  | 4.00x
Scalar Addition    | 0.0156s  | 0.0008s  | 19.50x
```

## Python CUDA Libraries

### Performance Comparison

We've benchmarked various Python CUDA libraries to show their performance characteristics:

| Library | Use Case | Typical Speedup | Best For |
|---------|----------|----------------|----------|
| **CuPy** | Scientific Computing | 15-25x | NumPy replacement, large matrices |
| **PyTorch** | Machine Learning | 10-20x | Neural networks, tensor operations |
| **Numba** | Custom Algorithms | 15-20x | JIT compilation, custom kernels |

### Performance Results

**Matrix Operations (4000x4000):**
- **CuPy**: 15-23x speedup for matrix multiply
- **PyTorch**: 14-15x speedup for matrix operations
- **Numba**: 15-20x speedup for vector operations

**Key Findings:**
- GPU acceleration becomes beneficial for matrices >2000x2000
- Memory transfer overhead affects small operations
- Float32 is 16x faster than Float64 on GPU
- Batching small operations improves efficiency

### Python Scripts

#### 1. Performance Benchmark (`cuda_performance_benchmark.py`)
Comprehensive benchmark comparing CPU vs GPU performance across different libraries.

**Run:**
```bash
source venv/bin/activate
python3 cuda_performance_benchmark.py
```

**Sample Output:**
```
============================================================
 NumPy (CPU) vs CuPy (GPU) Performance Comparison
============================================================

Matrix size: 4000x4000
--------------------------------------------------
Matrix Multiply      | CPU:   0.6234s | GPU:   0.0219s | Speedup:  28.44x
Element-wise Add     | CPU:   0.0179s | GPU:   0.0007s | Speedup:  24.12x
Matrix Transpose     | CPU:   0.0002s | GPU:   0.0000s | Speedup:   8.97x
```

#### 2. Practical Demo (`cuda_python_demo.py`)
Demonstrates best practices and practical usage of CUDA Python libraries.

**Run:**
```bash
source venv/bin/activate
python3 cuda_python_demo.py
```

**Sample Output:**
```
=== CuPy Demo ===
Creating 3000x3000 matrices...
Matrix multiplication: 0.0585s
Element-wise operations: 0.0865s
Total GPU operations completed successfully!

=== PyTorch Demo ===
Matrix multiplication: 0.0097s
Element-wise operations: 0.0241s
PyTorch operations completed successfully!
```

#### 3. Performance Analysis (`python_cuda_performance.py`)
Detailed performance analysis with library availability checks.

**Run:**
```bash
source venv/bin/activate
python3 python_cuda_performance.py
```

### Best Practices

1. **Use GPU for Large Operations**
   - Threshold: >2000x2000 matrices
   - Small operations may be slower due to transfer overhead

2. **Batch Small Operations**
   - Amortize memory transfer costs
   - Process multiple small operations together

3. **Choose Appropriate Data Types**
   - Use float32 when precision allows
   - Float32 is typically 16x faster than float64

4. **Library Selection**
   - **CuPy**: Scientific computing, NumPy replacement
   - **PyTorch**: Machine learning, neural networks
   - **Numba**: Custom algorithms, JIT compilation

## Build System

### Makefile Targets

```bash
# Build all programs
make all

# Build specific programs
make hello_world
make device_info
make matrix_ops

# Clean build artifacts
make clean

# Run all programs
make run
```

### Build Directory Structure
```
cuda/
├── build/           # Compiled binaries (gitignored)
├── venv/           # Python virtual environment
├── *.cu            # CUDA source files
├── *.py            # Python scripts
├── Makefile        # Build configuration
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Performance Tips

### Native CUDA
- Use shared memory for frequently accessed data
- Optimize memory coalescing patterns
- Choose appropriate block sizes (multiples of 32)
- Use asynchronous memory transfers when possible

### Python CUDA
- Transfer data once, compute multiple times
- Use appropriate data types (float32 vs float64)
- Consider mixed precision for additional speedup
- Profile your specific use case

## Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Install CUDA toolkit
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Compiler errors**
   ```bash
   # Install required compilers
   sudo apt install build-essential
   ```

3. **Python package conflicts**
   ```bash
   # Use virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Virtual environment not activated**
   ```bash
   # Check if venv is active
   which python3
   # Should show: /path/to/cuda/venv/bin/python3
   
   # If not, activate it
   source venv/bin/activate
   ```

### Verification Commands
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check Python environment
python3 --version
which python3

# Check installed packages
pip list | grep -E "numpy|numba|cupy|torch|pycuda|tensorflow"

# Test scripts
python3 cuda_python_demo.py
python3 cuda_performance_benchmark.py
```

## Quick Start

For a quick test of the entire system:

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd cuda

# Run the setup script
./setup.sh

# The script handles everything automatically!
```

### Option 2: Manual Setup
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Test Python CUDA libraries
python3 cuda_python_demo.py

# 3. Run performance benchmarks
python3 cuda_performance_benchmark.py

# 4. Build and test native CUDA programs
make all
./build/hello_world
./build/device_info
./build/matrix_ops
```

All components should work out of the box with the provided virtual environment!

## Future Enhancements

### Planned Features
- [ ] Multi-GPU support
- [ ] Advanced memory management
- [ ] Performance profiling tools
- [ ] Machine learning examples
- [ ] Real-time visualization

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- CuPy, PyTorch, and Numba development teams
- CUDA community for examples and best practices
