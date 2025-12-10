#!/bin/bash
#SBATCH --job-name=gaussian_diffusion_v1
#SBATCH --account=lp_emediaxr
#SBATCH --time=0:05:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load conda correctly (adjust path if your conda is elsewhere)
source ~/.bashrc
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate /vsc-hard-mounts/leuven-data/366/vsc36675/txgs/.conda
cd /data/leuven/366/vsc36675/txgs/

python -u main2.py
