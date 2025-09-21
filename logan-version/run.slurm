#!/bin/bash
#BATCH -J lammps_test
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodelist=e23-02,b11-09 
#SBATCH --account=liu32_1378
#SBATCH --exclusive

module purge
module load usc
module load cuda
module load conda

source /home1/lkyamamo/.bashrc

conda activate csci566-project

srun python main.py
