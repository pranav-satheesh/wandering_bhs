#!/bin/sh
#SBATCH --qos=lblecha
#SBATCH -J wbh_catalog
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/wandering_bhs/slurm_outputs/wandering_bh_catalog_%j.out
#SBATCH --error=/home/pranavsatheesh/wandering_bhs/slurm_outputs/wandering_bh_catalog_%j.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 01:00:00

ml conda
conda activate cosmosims

# h5py's file lock fails on /orange with "unable to lock file, errno = 11"
export HDF5_USE_FILE_LOCKING=FALSE

codePath=/home/pranavsatheesh/wandering_bhs/py_files
# pass a different run as the first argument, optionally followed by snapshot
# numbers (default: all): sbatch wandering_bh_catalog.slurm SM5_LW10_DFD_3_TNG/ 20 21 22
simName=${1:-SM5_LW10_LOWSPIN_RICH_DFD_3_TNG/}
shift
snaps="$@"
simPath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/
outPath=/orange/lblecha/pranavsatheesh/wandering_bhs/Data/

echo "=== $simName on $(hostname)  started $(date) ==="
python ${codePath}/wandering_bh_catalog.py \
  --simName $simName \
  --simPath $simPath \
  --outPath $outPath \
  --snaps $snaps
echo "=== finished $(date) ==="
