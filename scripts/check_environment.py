#!/usr/bin/env python3
"""
Environment verification script for LoRA/QLoRA VLM Benchmark.

This script checks if all dependencies are installed correctly and
verifies GPU availability and configuration.

Usage:
    python scripts/check_environment.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


def check_python_version():
    """Check Python version compatibility."""
    print_header(" Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print(" Python version is compatible")
        return True
    else:
        print(" Python 3.8+ required")
        return False


def check_package_installation():
    """Check if required packages are installed."""
    print_header(" Package Installation")
    
    required_packages = [
        ('torch', '1.12.0'),
        ('torchvision', '0.13.0'),
        ('numpy', '1.21.0'),
        ('psutil', '5.8.0'),
        ('PIL', '8.0.0')  # Pillow
    ]
    
    all_installed = True
    
    for package, min_version in required_packages:
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
                package_name = 'Pillow'
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                package_name = package
            
            print(f" {package_name}: {version}")
            
        except ImportError:
            print(f" {package_name}: Not installed")
            all_installed = False
        except Exception as e:
            print(f"  {package_name}: Error checking version - {e}")
    
    return all_installed


def check_pytorch_installation():
    """Check PyTorch installation and CUDA support."""
    print_header(" PyTorch & CUDA")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
            
            gpu_count = torch.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            
            # List all GPUs
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test CUDA functionality
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.randn(10, 10).cuda()
                z = torch.matmul(x, y)
                print(" CUDA operations working correctly")
                return True
            except Exception as e:
                print(f" CUDA operations failed: {e}")
                return False
        else:
            print("  CUDA not available - will use CPU")
            return True
            
    except ImportError:
        print(" PyTorch not installed")
        return False
    except Exception as e:
        print(f" PyTorch check failed: {e}")
        return False


def check_system_resources():
    """Check system resources (CPU, memory)."""
    print_header(" System Resources")
    
    try:
        import psutil
        
        # CPU information
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        print(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"RAM: {memory_gb:.1f} GB total, {memory.percent}% used")
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"Disk space: {disk_gb:.1f} GB available")
        
        # Check if resources are sufficient
        if memory_gb >= 4 and disk_gb >= 1:
            print(" System resources sufficient")
            return True
        else:
            print("  Limited system resources")
            return True  # Still workable
            
    except Exception as e:
        print(f" System check failed: {e}")
        return False


def check_project_structure():
    """Check if the project structure is correct."""
    print_header("  Project Structure")
    
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/training',
        'src/utils',
        'configs',
        'scripts'
    ]
    
    required_files = [
        'requirements.txt',
        'environment.yml',
        'README.md'
    ]
    
    all_present = True
    
    print(f"Project root: {project_root}")
    
    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f" {dir_path}/")
        else:
            print(f" {dir_path}/ - Missing")
            all_present = False
    
    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f" {file_path}")
        else:
            print(f" {file_path} - Missing")
            all_present = False
    
    return all_present


def test_import_modules():
    """Test importing project modules."""
    print_header(" Module Import Test")
    
    # Add src to path
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / 'src'
    sys.path.insert(0, str(src_dir))
    
    modules_to_test = [
        'utils.device',
        'data.dataset',
        'models.base_vlm',
        'models.lora_vlm',
        'models.qlora_vlm',
        'training.trainer'
    ]
    
    all_imported = True
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f" {module_name}")
        except ImportError as e:
            print(f" {module_name} - Import failed: {e}")
            all_imported = False
        except Exception as e:
            print(f"  {module_name} - Warning: {e}")
    
    return all_imported


def run_quick_benchmark():
    """Run a quick benchmark to verify everything works."""
    print_header(" Quick Functionality Test")
    
    try:
        # Add src to path
        script_dir = Path(__file__).parent
        src_dir = script_dir.parent / 'src'
        sys.path.insert(0, str(src_dir))
        
        from utils.device import setup_device
        from data.dataset import SmallVLDataset
        from models.base_vlm import SimpleVLM
        
        # Test device setup
        device, gpu_count = setup_device()
        print(f" Device setup: {device}")
        
        # Test dataset creation
        dataset = SmallVLDataset()
        print(f" Dataset creation: {len(dataset)} samples")
        
        # Test model creation
        model = SimpleVLM(vocab_size=len(dataset.vocab))
        print(f" Model creation: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        import torch
        sample = dataset[0]
        images = sample['image'].unsqueeze(0)
        captions = sample['caption'].unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output = model(images, captions)
        
        print(f" Forward pass: output shape {output.shape}")
        print(" Quick test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f" Quick test failed: {e}")
        return False


def print_installation_help():
    """Print installation help if issues are found."""
    print_header("🛠️  Installation Help")
    
    print("If you encountered issues, try these steps:")
    print()
    print("1. Update pip and install dependencies:")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements.txt")
    print()
    print("2. For conda users:")
    print("   conda env create -f environment.yml")
    print("   conda activate lora-qlora-benchmark")
    print()
    print("3. For CUDA issues:")
    print("   - Check NVIDIA driver: nvidia-smi")
    print("   - Reinstall PyTorch with correct CUDA version")
    print("   - Visit: https://pytorch.org/get-started/locally/")
    print()
    print("4. For import issues:")
    print("   pip install -e .")
    print()
    print("5. Check the troubleshooting guide:")
    print("   docs/TROUBLESHOOTING.md")


def main():
    """Main environment check function."""
    print(" Environment Check for LoRA/QLoRA VLM Benchmark")
    print("This script will verify your installation and environment setup.")
    
    # Track all checks
    checks = []
    
    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Package Installation", check_package_installation()))
    checks.append(("PyTorch & CUDA", check_pytorch_installation()))
    checks.append(("System Resources", check_system_resources()))
    checks.append(("Project Structure", check_project_structure()))
    checks.append(("Module Imports", test_import_modules()))
    checks.append(("Quick Test", run_quick_benchmark()))
    
    # Summary
    print_header(" Summary")
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks:
        status = " PASS" if result else " FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the benchmark:")
        print("   python scripts/run_benchmark.py")
    else:
        print(f"\n {total - passed} checks failed. See installation help below.")
        print_installation_help()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nEnvironment check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nEnvironment check failed with error: {e}")
        raise