#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=%x_%j.out
#SBATCH -t 00:05:00
#SBATCH -N 1	#Number of nodes
#SBATCH --exclusive
#SBATCH --gres=gpu:T4:1

module load cuda/11.8.0 gnu12/12.2.1 python/3.12.9 openssl/1.1.1w
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_NTASKS=$SLURM_NTASKS
id && hostname
nvidia-smi
date

INPUT_SIZE=512
OUTPUT_SIZE=512

echo "============================= GPU =================================="
python3 benchmark_linear_layer.py --device gpu --input $INPUT_SIZE --output $OUTPUT_SIZE
echo "===================================================================="

date
