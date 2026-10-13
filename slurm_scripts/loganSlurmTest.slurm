#!/bin/bash

#SBATCH --job-name=loganSlurmTest    # Job name 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=logantenner@ufl.edu     # Where to send mail 
#SBATCH --ntasks=1                    # Run on a single CPU 
#SBATCH --mem=1gb                     # Job memory request 
#SBATCH --time=00:05:00               # Time limit hrs:min:sec 
#SBATCH --output=loganSlurm_%j.log   # Standard output and error log 
pwd; hostname; date 

module purge 
module load conda
conda activate cosmosims

echo "Running plot script on a single CPU core" 

python /home/logantenner/wandering_bhs/py_files/slurmTest.py 

date