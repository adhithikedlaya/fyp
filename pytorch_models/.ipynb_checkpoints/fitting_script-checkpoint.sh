#!/bin/bash
#PBS -l select=1:ncpus=24:mem=64gb
#PBS -l walltime=2:00:00
#PBS -N p-dcm_fitting_all_vars_sim
#PBS -o /rds/general/user/ak1920/home/fyp/fyp/pytorch_models/logs
#PBS -e /rds/general/user/ak1920/home/fyp/fyp/pytorch_models/logs

module load anaconda3/personal
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}


for ((subj=3; subj<4; subj+=1)); do
    for ((roi=0; roi<1; roi+=1)); do
        exp="PLCB"
        python3 time_domain_model_copy.py $subj $roi $exp 
    done
done



