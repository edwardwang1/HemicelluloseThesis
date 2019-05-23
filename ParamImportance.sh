#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-yankai
echo 'Script start'
python AssessingImportanceOfParameters.py 
echo 'Script End'