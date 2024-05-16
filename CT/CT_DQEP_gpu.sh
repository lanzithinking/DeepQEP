#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH --mem=96G       # amount of RAM requested in GiB (2^40)
#SBATCH -t 0-02:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --gres=gpu:a100:1    # number of Request GPUs
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=FAIL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
# module load mamba/latest
# Using python, so source activate an appropriate environment
# source activate scicomp
# source activate grp_slan7
# source activate pytorch-gpu-2.0.1
# export PYTHONPATH=~/.conda/envs/grp_slan7/lib/python3.11/site-packages/:$PYTHONPATH
source ${HOME}/miniconda3/bin/activate pytorch

# go to working directory
cd ~/Projects/Deep-QEP/code/CT

# run python script

python -u run_Deep_QEP.py #> Deep_QEP.log &
# sbatch --job-name=DeepQEP --output=Deep_QEP.log CT_DQEP_gpu.sh