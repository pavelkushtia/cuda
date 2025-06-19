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
SOURCE = hello_world.cu
DEVICE_SOURCE = device_info.cu

# Default target
all: $(TARGET) $(DEVICE_INFO)

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
	@echo "  all          - Build the CUDA Hello World application and device info"
	@echo "  run          - Build and run the Hello World application"
	@echo "  device-info  - Build and run the device info utility"
	@echo "  clean        - Remove build artifacts"
	@echo "  check-cuda   - Check if CUDA is properly installed"
	@echo "  help         - Show this help message"

.PHONY: all run clean check-cuda device-info help 