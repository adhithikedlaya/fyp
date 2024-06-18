#!/bin/bash
#PBS -l select=1:ncpus=8:mem=100gb
#PBS -l walltime=24:00:00
#PBS -N p-dcm_loss_plotting


module load anaconda3/personal
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}

python3 loss_plotting.py
