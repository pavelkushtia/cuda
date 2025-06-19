# Makefile for CUDA Hello World Application
# 
# This Makefile provides targets to build and run the CUDA Hello World application.
# It includes proper compiler flags and error checking.

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_75
BUILD_DIR = build
TARGET = $(BUILD_DIR)/hello_world
DEVICE_INFO = $(BUILD_DIR)/device_info
MATRIX_OPS = $(BUILD_DIR)/matrix_ops
MATRIX_OPS_ADV = $(BUILD_DIR)/matrix_ops_advanced
SOURCE = hello_world.cu
DEVICE_SOURCE = device_info.cu
MATRIX_SOURCE = matrix_ops.cu
MATRIX_ADV_SOURCE = matrix_ops_advanced.cu

# Default target
all: $(TARGET) $(DEVICE_INFO) $(MATRIX_OPS) $(MATRIX_OPS_ADV)

# Build the CUDA Hello World application
$(TARGET): $(SOURCE) | $(BUILD_DIR)
	@echo "Compiling CUDA Hello World application..."
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)
	@echo "Build completed successfully!"

# Build the device info utility
$(DEVICE_INFO): $(DEVICE_SOURCE) | $(BUILD_DIR)
	@echo "Compiling CUDA Device Info utility..."
	$(NVCC) -o $(DEVICE_INFO) $(DEVICE_SOURCE)
	@echo "Device info build completed successfully!"

# Build the matrix operations program
$(MATRIX_OPS): $(MATRIX_SOURCE) | $(BUILD_DIR)
	@echo "Compiling CUDA Matrix Operations program..."
	$(NVCC) -o $(MATRIX_OPS) $(MATRIX_SOURCE)
	@echo "Matrix operations build completed successfully!"

# Build the advanced matrix operations program
$(MATRIX_OPS_ADV): $(MATRIX_ADV_SOURCE) | $(BUILD_DIR)
	@echo "Compiling CUDA Advanced Matrix Operations program..."
	$(NVCC) -o $(MATRIX_OPS_ADV) $(MATRIX_ADV_SOURCE)
	@echo "Advanced matrix operations build completed successfully!"

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Run the application
run: $(TARGET)
	@echo "Running CUDA Hello World application..."
	./$(TARGET)

# Run device info
device-info: $(DEVICE_INFO)
	@echo "CUDA Device Information:"
	./$(DEVICE_INFO)

# Run matrix operations
matrix-ops: $(MATRIX_OPS)
	@echo "Running CUDA Matrix Operations..."
	./$(MATRIX_OPS)

# Run advanced matrix operations
matrix-ops-adv: $(MATRIX_OPS_ADV)
	@echo "Running CUDA Advanced Matrix Operations..."
	./$(MATRIX_OPS_ADV)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "Clean completed!"

# Check if CUDA is available
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version
	@echo "CUDA installation found!"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all CUDA applications (Hello World, Device Info, Matrix Ops, Advanced Matrix Ops)"
	@echo "  run          - Build and run the Hello World application"
	@echo "  device-info  - Build and run the device info utility"
	@echo "  matrix-ops   - Build and run the matrix operations program"
	@echo "  matrix-ops-adv - Build and run the advanced matrix operations program"
	@echo "  clean        - Remove build artifacts"
	@echo "  check-cuda   - Check if CUDA is properly installed"
	@echo "  help         - Show this help message"

.PHONY: all run clean check-cuda device-info matrix-ops matrix-ops-adv help 