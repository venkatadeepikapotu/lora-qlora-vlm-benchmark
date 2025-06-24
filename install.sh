#!/bin/bash

# One-click installation script for LoRA/QLoRA VLM Benchmark
# Usage: bash install.sh

set -e  # Exit on any error

echo " LoRA/QLoRA VLM Benchmark - One-Click Installation"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if conda is installed
print_step "Checking conda installation..."
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
else
    print_status "Conda found: $(conda --version)"
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    print_error "environment.yml not found in current directory"
    echo "Please run this script from the repository root directory"
    exit 1
fi

# Check if environment already exists
ENV_NAME="lora-qlora-benchmark"
print_step "Checking if environment '$ENV_NAME' already exists..."

if conda env list | grep -q "^$ENV_NAME\s"; then
    print_warning "Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        print_status "Existing environment removed"
    else
        print_status "Using existing environment"
        conda activate $ENV_NAME
        pip install -e .
        print_status "Package installed in existing environment"
        echo ""
        echo " Installation completed!"
        echo " To activate: conda activate $ENV_NAME"
        echo " To run benchmark: python scripts/run_benchmark.py"
        exit 0
    fi
fi

# Create conda environment
print_step "Creating conda environment from environment.yml..."
print_status "This may take several minutes..."

if conda env create -f environment.yml; then
    print_status "Conda environment '$ENV_NAME' created successfully"
else
    print_error "Failed to create conda environment"
    echo ""
    echo " Troubleshooting tips:"
    echo "1. Check your internet connection"
    echo "2. Try: conda clean --all"
    echo "3. Update conda: conda update conda"
    echo "4. Check CUDA version compatibility in environment.yml"
    exit 1
fi

# Activate environment
print_step "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    print_status "Environment activated successfully"
else
    print_error "Failed to activate environment"
    exit 1
fi

# Install package in development mode
print_step "Installing package in development mode..."
if pip install -e .; then
    print_status "Package installed successfully"
else
    print_error "Failed to install package"
    exit 1
fi

# Run environment check
print_step "Running environment verification..."
if python scripts/check_environment.py; then
    print_status "Environment verification passed"
else
    print_warning "Environment verification had issues (check output above)"
fi

# Success message
echo ""
echo " Installation completed successfully!"
echo " Summary:"
echo "    Conda environment '$ENV_NAME' created"
echo "    All dependencies installed"
echo "    Package installed in development mode"
echo "    Environment verified"
echo ""
echo " Next steps:"
echo "   1. Activate environment: conda activate $ENV_NAME"
echo "   2. Run benchmark: python scripts/run_benchmark.py"
echo "   3. Check results in: results/ directory"
echo ""
echo " Additional commands:"
echo "    Environment check: python scripts/check_environment.py"
echo "     Custom run: python scripts/run_benchmark.py --help"
echo "     Force CPU: python scripts/run_benchmark.py --force-cpu"
echo ""

# Check if we're in a terminal that supports conda activate
if [[ "$SHELL" == *"bash"* ]] || [[ "$SHELL" == *"zsh"* ]]; then
    echo " To activate environment in new terminal:"
    echo "   conda activate $ENV_NAME"
else
    echo " Note: You may need to restart your terminal or run:"
    echo "   source ~/.bashrc  # or ~/.zshrc"
    echo "   conda activate $ENV_NAME"
fi