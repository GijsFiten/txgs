#!/bin/bash
#SBATCH --job-name=gaussian_vae_ffhq_ddu
#SBATCH --account=lp_emediaxr
#SBATCH --time=24:00:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
source /data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh
cd /data/leuven/366/vsc36675/txgs
conda activate /data/leuven/366/vsc36675/txgs/.conda

nvidia-smi
nvcc -V

srun torchrun --standalone --nproc_per_node=4 ./vae_training.py --config config/vae_training.yaml
