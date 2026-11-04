#!/bin/sh
#SBATCH --qos=lblecha
#SBATCH -J test_wbh_cat
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/wandering_bhs/slurm_outputs/test_wandering_bh_catalog.out
#SBATCH --error=/home/pranavsatheesh/wandering_bhs/slurm_outputs/test_wandering_bh_catalog.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=16G
#SBATCH -t 00:30:00

ml conda
conda activate cosmosims

codePath=/home/pranavsatheesh/wandering_bhs/py_files
simName=SM5_DFD_3_TNG/
simPath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/

# the output filename is built from simName alone, so a test run pointed at the
# real Data/ directory would overwrite the production catalogue. keep it separate.
outPath=/orange/lblecha/pranavsatheesh/wandering_bhs/Data/test/
mkdir -p $outPath

# one snapshot from each regime: before seeding, first BHs, peak BH count,
# post-merger decline, and z=0
snaps="0 8 15 20 25 32"

echo "=== host $(hostname)  started $(date) ==="
/usr/bin/time -v python ${codePath}/wandering_bh_catalog.py \
  --simName $simName \
  --simPath $simPath \
  --outPath $outPath \
  --snaps $snaps
echo "=== catalogue finished $(date) ==="

# read the test catalogue back and check the invariants that the writer is
# supposed to guarantee: table lengths agree, the /bh <-> /subhalo index
# mapping round-trips, and no silent NaN/inf from the unit conversions
outFile=${outPath}/wandering_bh_${simName%/}.hdf5
python - "$outFile" <<'PYCHECK'
import sys
import numpy as np
import h5py

fails = []
with h5py.File(sys.argv[1], "r") as f:
    bh, sub, snapshot = f["bh"], f["subhalo"], f["snapshot"]
    n_bh, n_sub = len(bh["r_kpc"]), len(sub["index"])
    print(f"{n_bh} BHs, {n_sub} BH-hosting subhalos, {len(snapshot['snap'])} snapshots")
    print("attrs:", dict(f.attrs))

    for group, n in [(bh, n_bh), (sub, n_sub)]:
        for key in group:
            if len(group[key]) != n:
                fails.append(f"{group.name}/{key} has {len(group[key])} rows, expected {n}")

    # every BH's subhalo_row must point at the subhalo whose index it recorded,
    # and the two must agree on which snapshot they came from
    subhalo_row = bh["subhalo_row"][:]
    if n_bh and (subhalo_row.min() < 0 or subhalo_row.max() >= n_sub):
        fails.append("bh/subhalo_row out of range")
    else:
        if not np.array_equal(bh["subhalo_index"][:], sub["index"][:][subhalo_row]):
            fails.append("bh/subhalo_index disagrees with subhalo/index via subhalo_row")
        if not np.array_equal(bh["snap"][:], sub["snap"][:][subhalo_row]):
            fails.append("bh/snap disagrees with subhalo/snap via subhalo_row")

    # bh_start/bh_count must tile /bh exactly, in order, with no gaps
    bh_start, bh_count = sub["bh_start"][:], sub["bh_count"][:]
    if not np.array_equal(bh_start, np.concatenate([[0], np.cumsum(bh_count)[:-1]])):
        fails.append("subhalo/bh_start is not the running sum of bh_count")
    if bh_count.sum() != n_bh:
        fails.append(f"subhalo/bh_count sums to {bh_count.sum()}, but /bh has {n_bh} rows")
    if (bh_count < 1).any():
        fails.append("a subhalo was written with no BHs")

    # per-snapshot totals and offsets must match the flat tables
    if snapshot["n_bh"][:].sum() != n_bh:
        fails.append("snapshot/n_bh does not sum to the number of BH rows")
    if snapshot["n_subhalos_with_bh"][:].sum() != n_sub:
        fails.append("snapshot/n_subhalos_with_bh does not sum to the number of subhalo rows")
    for snap, start, count in zip(snapshot["snap"], snapshot["bh_start"], snapshot["n_bh"]):
        if count and not (bh["snap"][start:start + count] == snap).all():
            fails.append(f"snapshot {snap}: /bh rows at its bh_start are not all from that snap")

    # the unit conversions are where a float32 overflow or a divide-by-zero
    # would show up, so check they produced finite numbers
    for key in ["r_kpc", "r_over_stellar_halfrad", "mass_msun", "mdot_msun_per_yr", "lbol_erg_s"]:
        values = bh[key][:]
        if not np.isfinite(values).all():
            fails.append(f"bh/{key} has {np.count_nonzero(~np.isfinite(values))} non-finite values")
    if (bh["mass_msun"][:] <= 0).any():
        fails.append("bh/mass_msun has non-positive masses")
    if (sub["stellar_halfrad_kpc"][:] <= 0).any():
        fails.append("subhalo/stellar_halfrad_kpc has non-positive radii (starless host leaked through)")

    # a BH is wandering exactly when it sits outside the stellar half-mass
    # radius. the flag is decided in float64 but the ratio is stored as
    # float32, so a BH sitting essentially on the radius may round either way
    ratio = bh["r_over_stellar_halfrad"][:].astype(np.float64)
    disagrees = bh["is_wandering"][:] != (ratio > 1.0)
    on_boundary = np.isclose(ratio, 1.0, rtol=1e-6, atol=0.0)
    if (disagrees & ~on_boundary).any():
        fails.append(f"bh/is_wandering disagrees with r_over_stellar_halfrad > 1 for "
                     f"{np.count_nonzero(disagrees & ~on_boundary)} BHs away from the boundary")

    # exactly one closest and one most-massive BH per subhalo
    for key in ["is_closest_to_centre", "is_most_massive"]:
        flag = bh[key][:]
        per_subhalo = np.add.reduceat(flag.astype(np.int64), bh_start) if n_bh else np.zeros(0)
        if not (per_subhalo == 1).all():
            fails.append(f"bh/{key} is not set exactly once in every subhalo")

    # periodic wrapping means no separation can exceed half the box
    half_box = 0.5 * np.sqrt(3) * f.attrs["BoxSize_ckpc_over_h"]
    if (bh["r_ckpc_over_h"][:] > half_box).any():
        fails.append("bh/r_ckpc_over_h exceeds the periodic maximum; minimum image failed")

    n_wandering = int(bh["is_wandering"][:].sum())
    print(f"wandering: {n_wandering}/{n_bh} ({100 * n_wandering / max(n_bh, 1):.1f}%)")
    print("r_kpc percentiles [50, 90, 99, 100]:",
          np.percentile(bh["r_kpc"][:], [50, 90, 99, 100]).round(2) if n_bh else "n/a")
    for snap, z, n_sub_snap, n_bh_snap in zip(snapshot["snap"], snapshot["redshift"],
                                              snapshot["n_subhalos_with_bh"], snapshot["n_bh"]):
        print(f"  snap {snap:3d}  z={z:6.2f}  {n_sub_snap:6d} subhalos  {n_bh_snap:7d} BHs")

print()
if fails:
    print(f"FAILED ({len(fails)} checks):")
    for fail in fails:
        print("  -", fail)
    sys.exit(1)
print("all checks passed")
PYCHECK
