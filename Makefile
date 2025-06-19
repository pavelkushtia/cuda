# Makefile for CUDA Hello World Application
# 
# This Makefile provides targets to build and run the CUDA Hello World application.
# It includes proper compiler flags and error checking.

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_52
TARGET = hello_world
SOURCE = hello_world.cu

# Default target
all: $(TARGET)

# Build the CUDA application
$(TARGET): $(SOURCE)
	@echo "Compiling CUDA Hello World application..."
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)
	@echo "Build completed successfully!"

# Run the application
run: $(TARGET)
	@echo "Running CUDA Hello World application..."
	./$(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "Clean completed!"

# Check if CUDA is available
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version
	@echo "CUDA installation found!"

# Show device information
device-info:
	@echo "CUDA Device Information:"
	@nvcc -o device_info device_info.cu 2>/dev/null && ./device_info 2>/dev/null || echo "Cannot run device info (CUDA may not be available)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the CUDA Hello World application"
	@echo "  run          - Build and run the application"
	@echo "  clean        - Remove build artifacts"
	@echo "  check-cuda   - Check if CUDA is properly installed"
	@echo "  device-info  - Show CUDA device information"
	@echo "  help         - Show this help message"

.PHONY: all run clean check-cuda device-info help 