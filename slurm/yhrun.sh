#!/bin/bash

HOSTNAME=$(hostname)
yhrun -N 2 -n 2 -p ai ./slurm.sh $HOSTNAME