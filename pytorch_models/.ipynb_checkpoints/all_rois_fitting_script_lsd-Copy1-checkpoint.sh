#!/bin/bash
#PBS -J 100-400
#PBS -l select=1:ncpus=2:mem=64gb
#PBS -l walltime=6:00:00
#PBS -N p-dcm_fitting_all_rois_lsd
#PBS -o /rds/general/user/ak1920/home/fyp/fyp/pytorch_models/logs
#PBS -e /rds/general/user/ak1920/home/fyp/fyp/pytorch_models/logs

module load anaconda3/personal
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}

subj=$((${PBS_ARRAY_INDEX} / 100 + 1))
roi=$((${PBS_ARRAY_INDEX} % 100))

python3 time_domain_model.py $subj $roi "LSD"
