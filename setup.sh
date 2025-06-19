#!/bin/bash

# CUDA Development Environment Setup Script
# ========================================
# This script sets up the Python virtual environment and installs all required packages
# for the CUDA development environment.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    local major=$(echo $python_version | cut -d. -f1)
    local minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        print_error "Python 3.8+ is required. Found: $python_version"
        exit 1
    fi
    
    print_success "Python version: $python_version"
}

# Function to check CUDA installation
check_cuda_installation() {
    if ! command_exists nvcc; then
        print_warning "CUDA compiler (nvcc) not found. Some features may not work."
        print_warning "Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
    else
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_success "CUDA version: $cuda_version"
    fi
    
    if ! command_exists nvidia-smi; then
        print_warning "nvidia-smi not found. GPU information may not be available."
    else
        print_success "NVIDIA driver detected"
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment 'venv' already exists."
        read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf venv
        else
            print_status "Using existing virtual environment."
            return 0
        fi
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created successfully!"
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" == *"venv"* ]]; then
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded successfully!"
}

# Function to install requirements
install_requirements() {
    print_status "Installing Python packages from requirements.txt..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install packages with progress
    pip install -r requirements.txt
    
    print_success "All packages installed successfully!"
}

# Function to verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    local packages=("numpy" "numba" "cupy" "torch" "pycuda" "tensorflow")
    local missing_packages=()
    
    for package in "${packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            local version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            print_success "$package: $version"
        else
            print_warning "$package: NOT FOUND"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "All required packages are installed!"
    else
        print_warning "Some packages are missing: ${missing_packages[*]}"
        print_warning "You may need to install them manually or check your CUDA installation."
    fi
}

# Function to test the environment
test_environment() {
    print_status "Testing the environment..."
    
    # Test basic Python functionality
    if python3 -c "import numpy; print('NumPy works!')" 2>/dev/null; then
        print_success "Basic Python functionality: OK"
    else
        print_error "Basic Python functionality: FAILED"
        return 1
    fi
    
    # Test CUDA functionality if available
    if python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        print_success "PyTorch CUDA check: OK"
    else
        print_warning "PyTorch CUDA check: FAILED"
    fi
    
    # Test one of our scripts
    if python3 -c "import cuda_python_demo; print('Demo script import: OK')" 2>/dev/null; then
        print_success "Demo script import: OK"
    else
        print_warning "Demo script import: FAILED"
    fi
}

# Function to show next steps
show_next_steps() {
    echo
    echo "=========================================="
    echo "🎉 Setup completed successfully!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo
    echo "2. Test the environment:"
    echo "   python3 cuda_python_demo.py"
    echo
    echo "3. Run performance benchmarks:"
    echo "   python3 cuda_performance_benchmark.py"
    echo
    echo "4. Build native CUDA programs:"
    echo "   make all"
    echo
    echo "5. For more information, see README.md"
    echo
    echo "To deactivate the virtual environment:"
    echo "   deactivate"
    echo
}

# Main setup function
main() {
    echo "=========================================="
    echo "🚀 CUDA Development Environment Setup"
    echo "=========================================="
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_python_version
    check_cuda_installation
    echo
    
    # Create and setup virtual environment
    create_venv
    activate_venv
    upgrade_pip
    install_requirements
    echo
    
    # Verify and test
    verify_installations
    test_environment
    echo
    
    # Show next steps
    show_next_steps
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --force, -f    Force recreation of virtual environment"
        echo "  --skip-test    Skip environment testing"
        echo
        echo "This script sets up the CUDA development environment."
        exit 0
        ;;
    --force|-f)
        if [ -d "venv" ]; then
            print_status "Force removing existing virtual environment..."
            rm -rf venv
        fi
        ;;
    --skip-test)
        SKIP_TEST=true
        ;;
esac

# Run main setup
main

print_success "Setup completed! 🎉" 