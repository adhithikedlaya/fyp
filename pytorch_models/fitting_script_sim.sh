#!/bin/bash
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l walltime=12:00:00
#PBS -N p-dcm_fitting_all_vars_sim


module load anaconda3/personal
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}

python3 time_domain_model_simulated.py




