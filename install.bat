@echo off
REM One-click installation script for Windows
REM Usage: install.bat

echo  LoRA/QLoRA VLM Benchmark - Windows Installation
echo ====================================================

REM Check if conda is installed
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [INFO] Conda found

REM Check if environment.yml exists
if not exist "environment.yml" (
    echo [ERROR] environment.yml not found in current directory
    echo Please run this script from the repository root directory
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr "lora-qlora-benchmark" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Environment 'lora-qlora-benchmark' already exists
    set /p choice="Do you want to remove and recreate it? (y/N): "
    if /i "%choice%"=="y" (
        echo [STEP] Removing existing environment...
        conda env remove -n lora-qlora-benchmark -y
        echo [INFO] Existing environment removed
    ) else (
        echo [INFO] Using existing environment
        call conda activate lora-qlora-benchmark
        pip install -e .
        echo [INFO] Package installed in existing environment
        echo.
        echo  Installation completed!
        echo  To activate: conda activate lora-qlora-benchmark
        echo  To run benchmark: python scripts/run_benchmark.py
        pause
        exit /b 0
    )
)

REM Create conda environment
echo [STEP] Creating conda environment from environment.yml...
echo [INFO] This may take several minutes...

conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment
    echo.
    echo  Troubleshooting tips:
    echo 1. Check your internet connection
    echo 2. Try: conda clean --all
    echo 3. Update conda: conda update conda
    echo 4. Check CUDA version compatibility in environment.yml
    pause
    exit /b 1
)

echo [INFO] Conda environment created successfully

REM Activate environment
echo [STEP] Activating conda environment...
call conda activate lora-qlora-benchmark

REM Install package in development mode
echo [STEP] Installing package in development mode...
pip install -e .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install package
    pause
    exit /b 1
)

echo [INFO] Package installed successfully

REM Run environment check
echo [STEP] Running environment verification...
python scripts/check_environment.py

REM Success message
echo.
echo  Installation completed successfully!
echo  Summary:
echo     Conda environment 'lora-qlora-benchmark' created
echo     All dependencies installed
echo     Package installed in development mode
echo     Environment verified
echo.
echo  Next steps:
echo    1. Activate environment: conda activate lora-qlora-benchmark
echo    2. Run benchmark: python scripts/run_benchmark.py
echo    3. Check results in: results/ directory
echo.
echo  Additional commands:
echo     Environment check: python scripts/check_environment.py
echo     Custom run: python scripts/run_benchmark.py --help
echo     Force CPU: python scripts/run_benchmark.py --force-cpu
echo.
echo  Note: Open a new command prompt and run:
echo    conda activate lora-qlora-benchmark

pause