#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=128G                    # amount of RAM requested in GiB (2^40)
#SBATCH -p serial                      # Use gpu partition
#SBATCH -q normal              # Run job under wildfire QOS queue

#SBATCH -t 1-00:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# conda pytorch
source ${HOME}/miniconda3/bin/activate pytorch
# export PYTHONPATH=${HOME}/miniconda/lib/python3.8/site-packages:${PATHONPATH}

# go to working directory
cd ~/Projects/Deep-QEP/code/CT

# run python script

python -u run_Deep_QEP.py #> Deep_QEP.log &
# sbatch --job-name=DeepQEP --output=Deep_QEP.log CT_DQEP_cpu.sh