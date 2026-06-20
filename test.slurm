#!/bin/bash
#SBATCH --job-name=gaussian_vae_ffhq_ddu
#SBATCH --account=lp_emediaxr
#SBATCH --time=0:05:00
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
source /data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh
cd /data/leuven/366/vsc36675/code/txgs
conda activate gsplat

nvidia-smi
nvcc -V

# 4 GPUs on a single node: optimize threads
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node=1 ./vae_training.py --config config/vae_training.yaml
