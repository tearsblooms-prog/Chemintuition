#!/bin/sh
#SBATCH -J predict_Condition
#SBATCH -o log/%j.out.txt
#SBATCH -e log/%j.err.txt
#SBATCH --partition=egroup
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
ulimit -n 65535

python predict.py