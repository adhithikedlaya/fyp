#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -l walltime=10:00:00
#PBS -N p-dcm_fitting_all_vars


module load anaconda3/personal
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}


for ((subj=1; subj<2; subj+=1)); do
    for ((roi=3; roi<4; roi+=1)); do
        exp="PLCB"
        python3 time_domain_model_copy.py $subj $roi $exp 
    done
done



