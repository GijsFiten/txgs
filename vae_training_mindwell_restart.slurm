#!/bin/bash
#SBATCH --job-name=gaussian_vae_ffhq_restart
#SBATCH --account=lp_emediaxr
#SBATCH --time=24:00:00
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
source /data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh
cd /data/leuven/366/vsc36675/code/txgs

export LD_LIBRARY_PATH=/data/leuven/366/vsc36675/conda_envs/txgs/lib:$LD_LIBRARY_PATH
conda activate txgs

nvidia-smi
nvcc -V

# 50 epochs × ~28 min/epoch ≈ 23 h — 24 h gives a comfortable margin.
# To train more epochs: increase max_epochs in config/vae_training_mw_restart.yaml
# and bump --time proportionally.
export OMP_NUM_THREADS=16
torchrun --standalone --nproc_per_node=8 ./vae_training.py --config config/vae_training_mw_restart.yaml
