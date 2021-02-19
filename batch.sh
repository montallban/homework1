#!/bin/bash
#
# Reasonable partitions: debug_5min, debug_30min, normal
#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
#SBATCH --output=xor_%J_stdout.txt
#SBATCH --time=12:00:00
#SBATCH --error=xor_%J_stderr.txt
#SBATCH --mail-user=michael.montalbano@ou.edu	
#SBATCH --job-name=xor_test
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/new
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment
python nn_1.py

