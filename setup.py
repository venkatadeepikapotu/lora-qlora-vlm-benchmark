#!/usr/bin/env python3
"""
Setup script for LoRA/QLoRA VLM Benchmark package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="lora-qlora-vlm-benchmark",
    version="1.0.0",
    description="Comprehensive benchmark comparing LoRA, QLoRA, and regular fine-tuning for Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Venkata Deepika Potu",
    author_email="vpotu@ufl.edu",
    url="https://github.com/venkatadeepikapotu/lora-qlora-vlm-benchmark",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "logging": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ]
    },
    
    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "lora-benchmark=scripts.run_benchmark:main",
            "lora-check-env=scripts.check_environment:main",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="lora qlora fine-tuning vision-language-models pytorch benchmark parameter-efficient",
    
    # License
    license="MIT",
    
    # Zip safe
    zip_safe=False,
)