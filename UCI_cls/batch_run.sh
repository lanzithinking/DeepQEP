#!/bin/bash

# run python script
for dataset in 'haberman' 'dermatology' 'tic_tac_toe' 'car' 'nursery'
do
	sbatch --job-name=${dataset} --output=${dataset}.log run_on_gpu.sh ${dataset}
	echo "Job ${dataset} submitted."
done
