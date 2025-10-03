#!/bin/bash
#SBATCH -J test_train_lstm_model
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --account=liu32_1378
#SBATCH --constraint=v100|a40

set -euo pipefail

# 1) Keep modules minimal; don't load nvhpc/gcc simultaneously.
#    If you're using PyTorch wheels/conda, you usually don't need cluster CUDA modules at all.
module purge
module load ver/2506

# 2) Initialize conda for non-interactive shells, then activate your env.
#    (No 'conda init' needed in your dotfiles.)
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
conda activate /home1/lkyamamo/.conda/envs/csci566-project

echo "=== Environment sanity checks ==="
echo "HOST: $(hostname)"
echo "Python: $(which python)"
python -c "import sys; print('Python', sys.version)"
python - <<'PY'
try:
    import torch, os
    print('torch', torch.__version__, 'CUDA available?', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('Device 0:', torch.cuda.get_device_name(0))
except Exception as e:
    print('Torch check failed:', e)
PY
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"

# 3) Run your code under srun so it inherits the GPU allocation/cgroup cleanly.
srun -u python -u main.py
