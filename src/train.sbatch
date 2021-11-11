#!/bin/bash
#
#SBATCH --job-name=train
#
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1

ml python/3.6.1
module load py-pytorch/1.4.0_py36

python3 train.py --verbose 
