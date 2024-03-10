#!/bin/bash

module load anaconda3/2023.09
source activate pytorch2
torchrun --nnodes=2 --nproc_per_node=8 --master_addr=$1 --master_port=1234 slurm.py
