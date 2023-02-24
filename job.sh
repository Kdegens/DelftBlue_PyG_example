#!/bin/sh
#
#SBATCH --job-name="try"
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-as-msc-ce

module load 2022r2
module load python/3.8.12
module load cuda/11.6
module load py-pip

srun python -m main