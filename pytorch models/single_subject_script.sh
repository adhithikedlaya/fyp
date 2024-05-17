#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=00:03:00
#PBS -N p-dcm_fitting
#PBS -J 0-115


module load anaconda3/personal
source activate pytorch_env
cd $PBS_O_WORKDIR
num_regions = 116
subj = $(($PBS_ARRAY_INDEX + 1/ num_regions)) 
roi = $(($PBS_ARRAY_INDEX % num_regions))
exp = "PLCB"
python3 time_domain_model $subj $roi $exp > /rds/general/user/ak1920/home/fyp/fyp/pytorch models/output_${exp}_${subj}_${roi}.txt


