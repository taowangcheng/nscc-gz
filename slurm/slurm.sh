#!/bin/bash

conda activate pytorch2
torchrun --nnodes=2 --nproc_per_node=8 --master_addr=$1 --master_port=1234 slurm.py
