#!/bin/sh
#SBATCH --qos=lblecha
#SBATCH -J bh_2dplot
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/wandering_bhs/slurm_outputs/bh_2dplot.out
#SBATCH --error=/home/pranavsatheesh/wandering_bhs/slurm_outputs/bh_2dplot.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=64G
#SBATCH -t 04:00:00

ml conda
conda activate cosmosims

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DFD_3_TNG
codePath=/home/pranavsatheesh/wandering_bhs/py_files
outDir=/home/pranavsatheesh/wandering_bhs/figures/2d_plots

bg_ptype=0                 # 0=gas, 1=DM, 4=stars (background image)
view='xy'
radiative_efficiency=0.2
box_length=auto            # kpc, or a number e.g. 50
nbins=400

# "snap:subhalo" pairs to process. Edit as needed.
snap_subhalo_pairs=("32:9853" "32:6229" "25:1230" "25:3880")

for pair in "${snap_subhalo_pairs[@]}"; do
  snap=${pair%:*}
  subhaloID=${pair#*:}

  python ${codePath}/run_bh_2dplot.py \
    --basePath $basePath \
    --snap $snap \
    --subhalo $subhaloID \
    --outDir $outDir \
    --bg_ptype $bg_ptype \
    --view $view \
    --radiative_efficiency $radiative_efficiency \
    --box_length $box_length \
    --nbins $nbins
done
