source ~/.bashrc
source "/data/leuven/366/vsc36675/miniconda3/etc/profile.d/conda.sh"
cd /data/leuven/366/vsc36675/txgs
conda activate /data/leuven/366/vsc36675/txgs/.conda

nvidia-smi
nvcc -V

python -u training.py