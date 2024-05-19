#!/bin/bash

# run python script
for dataset in 'parkinsons' 'elevators' 'protein' 'slice' '3droad' 'song'
do
	sbatch --job-name=${dataset} --output=${dataset}.log run_on_gpu.sh ${dataset}
	echo "Job ${dataset} submitted."
done
