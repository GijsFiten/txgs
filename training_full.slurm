#!/bin/bash
#SBATCH --job-name=gaussian_diffusion_v1
#SBATCH --account=lp_emediaxr
#SBATCH --time=2:00:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /user/leuven/366/vsc36675/.bashrc
source "/data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh"
cd /data/leuven/366/vsc36675/txgs
conda activate /data/leuven/366/vsc36675/txgs/.conda

python -u training.py
