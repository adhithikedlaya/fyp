#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=48:00:00
#PBS -N p-dcm_fitting


module load anaconda3/personal
source activate pytorch_env
cd $PBS_O_WORKDIR

for ((subj=1; subj<3; subj+=1)); do
    for ((roi=0; roi<116; roi+=step)); do
        exp = "PLCB"
        python3 time_domain_model $subj $roi $exp > /rds/general/user/ak1920/home/fyp/fyp/pytorch models/output_${exp}_${subj}_${roi}.txt
    done
done



