#!/bin/bash

#SBATCH --job-name=catalogRestrict    # Job name 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=logantenner@ufl.edu     # Where to send mail 
#SBATCH --ntasks=1                    # Run on a single CPU 
#SBATCH --mem=5gb                     # Job memory request 
#SBATCH --time=00:10:00               # Time limit hrs:min:sec 
#SBATCH --output=catalogRestrict_output_%j.log   # Standard output and error log 
pwd; hostname; date 

module purge 
module load conda
conda activate cosmosims

echo "Running plot script on a single CPU core" 

python /home/logantenner/wandering_bhs/py_files/wandering_bh_catalog.py --simName="SM5_LW10_DFD_3_TNG/"

date