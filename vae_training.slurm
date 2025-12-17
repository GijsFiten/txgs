#!/bin/bash
#SBATCH --job-name=gaussian_vae_v3
#SBATCH --account=lp_emediaxr
#SBATCH --time=24:00:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
source /data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh
cd /data/leuven/366/vsc36675/txgs
conda activate /data/leuven/366/vsc36675/txgs/.conda

nvidia-smi
nvcc -V

python -u vae_training.py
