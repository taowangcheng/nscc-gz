#!/bin/bash

module load anaconda3/2023.09
source activate pytorch2
# torchrun --nnodes=2 --nproc-per-node=8 --rdzv-id=1234 --rdzv-backend=c10d --rdzv-endpoint=$1:1234 slurm.py
run_cmd="torchrun --nnodes=2 --nproc-per-node=8 --rdzv-id=1234 --rdzv-backend=c10d --rdzv-endpoint=$1:1234 slurm.py"

echo $run_cmd
# sleep 100000
eval $run_cmd

set +x