#!/bin/bash
# Bash execution script converted from job.slurm
# This script can be run locally without SLURM

set -euo pipefail

# Configuration - adjust these as needed
NNODES=${NNODES:-1}  # Number of nodes (default: 1 for local execution)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}  # Number of processes per node (default: 1)
MASTER_PORT=${MASTER_PORT:-29500}  # Master port for distributed training
MASTER_ADDR=${MASTER_ADDR:-localhost}  # Master address (default: localhost)

# 1) Keep modules minimal; don't load nvhpc/gcc simultaneously.
#    If you're using PyTorch wheels/conda, you usually don't need cluster CUDA modules at all.
# Uncomment if you need to load modules on your system
# module purge
# module load ver/2506

# 2) Initialize conda for non-interactive shells, then activate your env.
#    Update the conda path and environment name for your system
#    (No 'conda init' needed in your dotfiles.)
if [ -f "/apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh" ]; then
    source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
    conda activate /home1/lkyamamo/.conda/envs/csci566-project
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate csci566-project 2>/dev/null || conda activate base
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate csci566-project 2>/dev/null || conda activate base
elif command -v conda &> /dev/null; then
    # Try to activate if conda is already in PATH
    conda activate csci566-project 2>/dev/null || conda activate base 2>/dev/null || true
fi

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
# For local execution, use localhost; for multi-node, set MASTER_ADDR environment variable
if [ "$NNODES" -eq 1 ]; then
    MASTER_ADDR="localhost"
else
    # For multi-node, MASTER_ADDR should be set as environment variable
    if [ "$MASTER_ADDR" = "localhost" ]; then
        echo "Warning: Multi-node setup detected but MASTER_ADDR is localhost. Please set MASTER_ADDR to the master node's IP."
    fi
fi

# Auto-detect number of GPUs if NPROC_PER_NODE not set and we're on a single node
if [ "$NNODES" -eq 1 ] && [ "$NPROC_PER_NODE" -eq 1 ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            NPROC_PER_NODE=$GPU_COUNT
            echo "[Auto-detected] Found $GPU_COUNT GPU(s), setting NPROC_PER_NODE=$NPROC_PER_NODE"
        fi
    fi
fi

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"

# 3) Run your code with torchrun (no srun needed for local execution)
torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main.py

