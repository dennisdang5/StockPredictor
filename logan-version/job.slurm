#!/bin/bash
#SBATCH -J multi_hybrid_lstm_train
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=2
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=liu32_1378
#SBATCH --constraint=v100
#SBATCH --exclusive

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

# Keep threads reasonable; let DataLoader workers do the I/O
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Network interface detection for NCCL
# Try to auto-detect the correct interface if bond0 doesn't exist
if ! ip link show bond0 >/dev/null 2>&1; then
    # bond0 doesn't exist, try common alternatives
    if ip link show ib0 >/dev/null 2>&1; then
        export NCCL_SOCKET_IFNAME=ib0
        echo "[NCCL] Using ib0 interface (InfiniBand)"
    elif ip link show eth0 >/dev/null 2>&1; then
        export NCCL_SOCKET_IFNAME=eth0
        echo "[NCCL] Using eth0 interface"
    else
        # Try to find any non-lo interface
        INTERFACE=$(ip link show | grep -E "^[0-9]+:" | grep -v lo | head -1 | awk '{print $2}' | sed 's/:$//')
        if [ -n "$INTERFACE" ]; then
            export NCCL_SOCKET_IFNAME=$INTERFACE
            echo "[NCCL] Auto-detected interface: $INTERFACE"
        else
            export NCCL_SOCKET_IFNAME=bond0  # Fallback (will show warning but may still work)
            echo "[NCCL] Warning: Could not detect interface, using bond0 (may fail)"
        fi
    fi
else
    export NCCL_SOCKET_IFNAME=bond0
    echo "[NCCL] Using bond0 interface"
fi

# If your cluster has no IB or IB is flaky, try:
# export NCCL_IB_DISABLE=1

# Master addr/port for rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((10000 + RANDOM % 50000))

# Processes per node = GPUs per node (fallback if env var missing)
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
fi

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT  NNODES=$SLURM_NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"

# 3) Run your code under srun so it inherits the GPU allocation/cgroup cleanly.
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main.py
