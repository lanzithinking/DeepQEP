#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH --mem=64G       # amount of RAM requested in GiB (2^40)
#SBATCH -t 0-08:00:00   # time in d-hh:mm:ss
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
cd ~/Projects/Deep-QEP/code/ImageClassification

# run python script
if [ $# -eq 0 ]; then
	dataset_name='mnist'
	likelihood='dirichlet'
elif [ $# -eq 1 ]; then
	dataset_name="$1"
	likelihood='dirichlet'
elif [ $# -eq 2 ]; then
	dataset_name="$1"
	likelihood="$2"
fi

python -u run_GP_${likelihood}.py ${dataset_name} #> GP.log &
# sbatch --job-name=GP --output=GP.log run_GP_gpu.sh