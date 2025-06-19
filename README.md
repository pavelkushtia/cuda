# CUDA Hello World Application

This repository contains a simple CUDA Hello World application that demonstrates basic GPU programming concepts using NVIDIA's CUDA platform.

## Overview

The application includes:
- **hello_world.cu**: Main CUDA application that prints "Hello World" from both CPU and GPU
- **device_info.cu**: Utility to display detailed information about CUDA devices
- **Makefile**: Automated build system with multiple targets

## Features

- Demonstrates basic CUDA kernel execution
- Includes comprehensive error checking
- Shows device information and capabilities
- Proper resource management and cleanup
- Well-documented code with examples

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

## Building and Running

### Quick Start

```bash
# Build the application
make

# Run the application
make run

# Or build and run in one command
make run
```

### Manual Build

```bash
# Compile the main application
nvcc -O2 -arch=sm_52 -o hello_world hello_world.cu

# Compile the device info utility
nvcc -o device_info device_info.cu

# Run the applications
./hello_world
./device_info
```

### Makefile Targets

```bash
make all          # Build the CUDA Hello World application
make run          # Build and run the application
make clean        # Remove build artifacts
make check-cuda   # Check if CUDA is properly installed
make device-info  # Show CUDA device information
make help         # Show all available targets
```

## Expected Output

When you run the application, you should see output similar to:

```
Hello World from CPU!
Found 1 CUDA device(s)
Using device: NVIDIA GeForce RTX 3080
Compute capability: 8.6
Number of multiprocessors: 68

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

## Code Structure

### hello_world.cu

- **`helloFromGPU()`**: CUDA kernel that runs on the GPU
- **`checkCudaError()`**: Error checking utility function
- **`main()`**: Main function demonstrating CUDA operations

### Key Concepts Demonstrated

1. **Kernel Launch**: Using `<<<blocks, threads>>>` syntax
2. **Error Handling**: Comprehensive CUDA error checking
3. **Device Information**: Querying GPU properties
4. **Synchronization**: Ensuring GPU operations complete
5. **Resource Management**: Proper cleanup of GPU resources

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

### Debugging

Use the device info utility to diagnose issues:

```bash
make device-info
```

This will show detailed information about your CUDA devices and help identify compatibility issues.

## Architecture Notes

The Makefile uses `-arch=sm_52` which targets compute capability 5.2. You may need to adjust this based on your GPU:

- **GTX 900 series**: Use `-arch=sm_52`
- **GTX 1000 series**: Use `-arch=sm_61`
- **RTX 2000 series**: Use `-arch=sm_75`
- **RTX 3000 series**: Use `-arch=sm_86`
- **RTX 4000 series**: Use `-arch=sm_89`

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the terms specified in the LICENSE file.
