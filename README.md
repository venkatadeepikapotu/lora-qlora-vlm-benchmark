This benchmark provides a **complete, automated comparison** of parameter-efficient fine-tuning methods. With **one-click installation** and **automatic hardware detection**, you can quickly demonstrate the **95% parameter reduction** and **significant memory savings** achieved by LoRA and QLoRA while maintaining model performance.

##  Quick Start

### Option 1: One-Click Installation (Recommended)

**Linux/macOS:**
```bash
# 1. Clone the repository
git clone https://github.com/your-username/lora-qlora-vlm-benchmark.git
cd lora-qlora-vlm-benchmark

# 2. One-click install (creates environment, installs everything)
bash install.sh

# 3. Run benchmark
conda init
conda activate lora-qlora-benchmark
python scripts/run_benchmark.py
```

**Windows:**
```cmd
REM 1. Clone the repository
git clone https://github.com/your-username/lora-qlora-vlm-benchmark.git
cd lora-qlora-vlm-benchmark

REM 2. One-click install
install.bat

REM 3. Run benchmark
conda init
conda activate lora-qlora-benchmark
python scripts/run_benchmark.py
```

### Option 2: Manual Conda Installation

```bash
# 1. Clone the repository
git clone https://github.com/venkatadeepikapotu/lora-qlora-vlm-benchmark.git
cd lora-qlora-vlm-benchmark

# 2. Create conda environment
conda env create -f environment.yml
conda init
conda activate lora-qlora-benchmark

# 3. Install package in development mode
pip install -e .

# 4. Verify installation
python scripts/check_environment.py

# 5. Run benchmark
python scripts/run_benchmark.py
```

### Option 3: Using pip + venv

```bash
# 1. Clone the repository
git clone https://github.com/venkatadeepikapotu/lora-qlora-vlm-benchmark.git
cd lora-qlora-vlm-benchmark

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package
pip install -e .

# 5. Verify installation
python scripts/check_environment.py

# 6. Run benchmark
python scripts/run_benchmark.py
```

##  Configuration

### Basic Configuration

Edit `configs/default_config.py`:

```python
# Training parameters
EPOCHS = 3
BATCH_SIZE = 4  # Will be auto-scaled for multi-GPU
LEARNING_RATE = 1e-3

# LoRA parameters
LORA_RANK = 4
LORA_ALPHA = 1.0

# Model parameters
EMBED_DIM = 256
HIDDEN_DIM = 512

# Output settings
OUTPUT_DIR = "results"
SAVE_MODELS = False
VERBOSE = True
```

### Advanced Configuration

```bash
# Custom configuration
python scripts/run_benchmark.py --epochs 5 --batch-size 8 --lora-rank 8

# Save trained models
python scripts/run_benchmark.py --save-models --output-dir results/experiment_1

# Quiet mode
python scripts/run_benchmark.py --quiet
```

##  Interpreting Results

### Output Files

The benchmark creates several output files in the `results/` directory:

```
results/
├── benchmark_results.json      # Raw metrics
├── comparison_table.txt        # Human-readable comparison
├── efficiency_analysis.txt     # Detailed efficiency metrics
├── system_info.json          # Hardware information
└── training_logs.txt          # Detailed training logs
```

### Key Metrics

1. **Parameter Efficiency**: Trainable parameters / Total parameters
2. **Memory Efficiency**: Peak memory usage during training
3. **Time Efficiency**: Average iteration time and total training time
4. **Model Performance**: Final loss values and convergence

### Example Output

```
 COMPREHENSIVE MULTI-GPU BENCHMARK RESULTS
================================================================
Model        Total      Trainable    RAM      GPU      Iter Time    
Type         Params     Params       (MB)     (MB)     (sec)        
----------------------------------------------------------------
Regular      847,530    847,530      45.2     156.3    0.0847       
LoRA         847,530    42,368       28.7     98.7     0.0923       
QLoRA        847,530    42,368       22.1     76.2     0.1045       

 EFFICIENCY ANALYSIS
================================================================
LoRA:
    Parameter reduction: 95.0%
    RAM reduction: 36.5%
    GPU memory reduction: 36.9%
    Speed: 9.0% slower per iteration

QLoRA:
    Parameter reduction: 95.0%
    RAM reduction: 51.1%
    GPU memory reduction: 51.2%
    Speed: 23.4% slower per iteration
```

##  Environment Details

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 4GB
- GPU: Optional (will use CPU if unavailable)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Multiple GPUs: Automatically detected and utilized

### Software Dependencies

- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- psutil 5.8+


##  Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/run_benchmark.py --batch-size 2
   ```

2. **No GPU Detected**
   ```bash
   # Check CUDA installation
   python scripts/check_environment.py
   
   # Force CPU usage
   CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark.py
   ```

3. **Import Errors**
   ```bash
   # Reinstall in development mode
   pip install -e .
   ```

