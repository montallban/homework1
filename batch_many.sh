#!/bin/bash
#

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=new/xor_exp%04a_stdout.txt
#SBATCH --error=new/xor_exp%04a_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=bool_test
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#################################################
source ~fagg/pythonenv/tensorflow/bin/activate
python nn_1.py --epochs 1000 --exp $SLURM_ARRAY_TASK_ID


