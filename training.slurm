#!/bin/bash
#SBATCH --job-name=gaussian_diffusion_v2
#SBATCH --account=lp_emediaxr
#SBATCH --time=0:03:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
source "/data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh"
cd /data/leuven/366/vsc36675/txgs
conda activate /data/leuven/366/vsc36675/txgs/.conda

nvidia-smi
nvcc -V

python -u training.py
