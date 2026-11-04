#!/bin/sh
#SBATCH --qos=lblecha
#SBATCH -J bh_2dplot_seedcomp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/wandering_bhs/slurm_outputs/bh_2dplot_seedcomp.out
#SBATCH --error=/home/pranavsatheesh/wandering_bhs/slurm_outputs/bh_2dplot_seedcomp.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=64G
#SBATCH -t 04:00:00

ml conda
conda activate cosmosims

simPath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/
codePath=/home/pranavsatheesh/wandering_bhs/py_files
outDir=/home/pranavsatheesh/wandering_bhs/figures/2d_plots

# one panel per run; the galaxy is picked in the first and matched in the rest
runs="SM5_DFD_3_TNG SM5_LW10_DFD_3_TNG SM5_LW10_LOWSPIN_RICH_DFD_3_TNG"
redshift=0                     # nearest snapshot is used in every run
log_stellar_mass=10.75         # Milky Way: M* ~ 5-6e10 Msun
halo_type=any              # any / central / satellite
max_dlogm=0.15                 # fail if any galaxy is further than this (dex)

bg_ptype=4                 # 0=gas, 1=DM, 4=stars (background image)
view='xy'
radiative_efficiency=0.2
box_length=60              # kpc, shared by all panels
nbins=400
bg_cmap=inferno
bg_method=adaptive
adaptive_k=16
bh_cmap=viridis

python ${codePath}/run_bh_2dplot_seed_comparison.py \
  --simPath $simPath \
  --runs $runs \
  --redshift $redshift \
  --log_stellar_mass $log_stellar_mass \
  --halo_type $halo_type \
  --max_dlogm $max_dlogm \
  --outDir $outDir \
  --bg_ptype $bg_ptype \
  --view $view \
  --radiative_efficiency $radiative_efficiency \
  --box_length $box_length \
  --nbins $nbins \
  --bg_cmap $bg_cmap \
  --bg_method $bg_method \
  --adaptive_k $adaptive_k \
  --bh_cmap $bh_cmap
