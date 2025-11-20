#!/bin/bash

# Detect if running on HPC system
# Check for module command availability or HPC-specific environment variables
IS_HPC=false
if command -v module &> /dev/null || [ -n "$SLURM_JOB_ID" ] || [ -n "$PBS_JOBID" ] || [ -n "$LSB_JOBID" ]; then
    IS_HPC=true
fi

# Only load modules if on HPC
if [ "$IS_HPC" = true ]; then
    module purge
    module load ver/2506  gcc/14.3.0
    module load python/3.11.13
fi

python -m venv .venv
source .venv/bin/activate

pip install uv
uv pip install catboost xgboost
uv pip install datasets
uv pip install rich
uv pip install tabpfn-client
uv pip install tabpfn-extensions[all]
uv pip install tabpfn
uv pip install numpy
uv pip install torch
uv pip install torchsummary
uv pip install tensorboard
uv pip install matplotlib
uv pip install tqdm
uv pip install scipy
uv pip install statsmodels
uv pip install scikit-learn
uv pip install pandas
uv pip install yfinance
uv pip install torchvision
uv pip install torchtext
uv pip install torchaudio
uv pip install torchdata
uv pip install torchmetrics
