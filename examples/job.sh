#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH --gpus=rtx_2080_ti:2
#SBATCH --gres=gpumem:11g
#SBATCH -A es_sachan
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=job_name
#SBATCH --output=/path/to/your/jobs/logs/file.out
#SBATCH --error=/path/to/your/jobs/logs/file.err

## load modules and activate environment
module load gcc/8.2.0;module load python/3.9.9;module load eth_proxy;module load cuda/11.8.0
source /path/to/your/venv/bin/activate

python /path/to/script.py
