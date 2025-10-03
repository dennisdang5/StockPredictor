#!/bin/bash
#SBATCH -J test_train_lstm_model
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --nodelist=e21-[03-16],e22-[01-16],e23-01 
#SBATCH --account=liu32_1378

module purge
module load ver/2506  gcc/14.3.0
module load cuda/12.9.1
module load cudnn/9.10.2.21-12-cuda-12.9.1
module load nvhpc/24.5

conda activate csci566-project

conda env list
which python

python main.py
