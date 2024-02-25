#!/bin/bash
#
#SBATCH --job-name=flex
##SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH -n 1   # number of cores
##SBATCH -w gpu-380-18
#SBATCH --output=flex.log
#SBATCH --error=flex.error
#SBATCH -p gpu
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1  # this is important to use GPUs
#SBATCH --array=0-44
#SBATCH --partition=medium

source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03
eval "$(conda shell.bash hook)"
conda activate metamod
nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

python flexnet_slurm.py --run-id $SLURM_ARRAY_TASK_ID --exp-name "neural_data_net_slurm"