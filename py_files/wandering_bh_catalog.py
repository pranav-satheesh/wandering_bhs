import argparse
import os
import numpy as np
import h5py

from cosmo_sim_tools.arepo_tools import arepo_package as arepo
from cosmo_sim_tools import brahma
from cosmo_sim_tools.arepo_tools import mdot_to_Lbol


# catalogue
SUBHALO_REQUIRED = ["SubhaloLenType", "SubhaloPos", "SubhaloMass",
             "SubhaloMassType", "SubhaloHalfmassRadType", "SubhaloCM"]
SUBHALO_OPTIONAL = ["SubhaloGrNr"]


def load_subhalo_catalogue(sim_file, snap):
    """Postprocessed subhalo catalogue, tolerating absent optional fields."""
    return brahma.groupcat.loadSubhalos_postprocessed(
        sim_file, snap, fields=SUBHALO_REQUIRED + SUBHALO_OPTIONAL)

def central_mask(sim_file, snap, n_sub, grnr):
    """True for the primary (central) subhalo of each FoF group."""
    
    out = np.zeros(n_sub, dtype=bool)

    if grnr is None:
        # no FoF group membership available, can't tell centrals apart
        return out

    # GroupFirstSub[g] is the subhalo index of group g's central subhalo
    try:
        groups = brahma.groupcat.loadHalos_postprocessed(
            sim_file, snap, fields=["GroupFirstSub"])
        first_sub_of_group = np.asarray(groups["GroupFirstSub"]).ravel()
    except Exception:
        return out

    subhalo_idx = np.arange(n_sub)
    group_idx = np.asarray(grnr).ravel()[:n_sub]
    # guard against group ids that fall outside the loaded GroupFirstSub table
    in_range = (group_idx >= 0) & (group_idx < len(first_sub_of_group))
    # a subhalo is central if it is its own group's first subhalo
    out[in_range] = first_sub_of_group[group_idx[in_range]] == subhalo_idx[in_range]

    return out


def load_bh_field(sim_file, field, z, subhalo_index):
    """BH (p_type=5) particle array for one subhalo, or None if empty."""
    out = arepo.get_particle_property_within_postprocessed_groups(
        sim_file, particle_property=[field], p_type=5,
        desired_redshift=z, subhalo_index=subhalo_index,
        group_type='subhalo', store_all_offsets=0)
    arr = out[0]
    if arr is None or len(arr) == 0:
        return None
    return np.asarray(arr)


# ------------------------------------------------------------ one snapshot ----

def snapshot_rows(sim_file, snap, z, eps=0.2, verbose_every=200):
    """Build the per-BH and per-subhalo row dicts for a single snapshot."""

    subhalo_cat = load_subhalo_catalogue(sim_file, snap)
    lentype = np.asarray(subhalo_cat["SubhaloLenType"])
    n_sub_total = len(lentype) #total number of subhalos in this snapshot

    header = brahma.groupcat.loadHeader(sim_file, snap)
    h = float(header["HubbleParam"])
    boxsize = float(header["BoxSize"])
    a = float(header.get("Time", 1.0 / (1.0 + z))) #scale factor
    to_kpc = a / h                                 # to convert ckpc/h -> physical kpc

    subhalo_pos = np.asarray(subhalo_cat["SubhaloPos"], dtype=np.float64) #subhalo position within the box
    subhalo_cm = np.asarray(subhalo_cat["SubhaloCM"], dtype=np.float64) #subhalo center of mass
    
    #Is this subhalo a central or satellite?
    is_central_mask = central_mask(sim_file, snap, n_sub_total, subhalo_cat.get("SubhaloGrNr"))

    # Only subhalos that actually host a BH need a particle read
    bh_hosts = np.flatnonzero(lentype[:, 5] > 0)
    bh_data, subhalo_data = [], []

    for n_done, i in enumerate(bh_hosts):
        i = int(i) #ith subhalo that hosts BHs
        if verbose_every and n_done % verbose_every == 0:
            print(f"  snap {snap}: subhalo {n_done}/{len(bh_hosts)}", flush=True)

        #BH Data within subhalos that host them 
        BH_coords = load_bh_field(sim_file, "Coordinates", z, i)
        if BH_coords is None:
            continue                                    # catalogue/particle mismatch
        BH_coords = np.atleast_2d(np.asarray(BH_coords, dtype=np.float64))
        BH_mass = np.ravel(load_bh_field(sim_file, "BH_Mass", z, i))
        BH_mdot = np.ravel(load_bh_field(sim_file, "BH_Mdot", z, i))
        BHmass_msun = BH_mass * 1e10 / h #Msun
        BH_Mdot_msun_per_yr = BH_mdot * (1e10/(0.978*1e9)) #Msun/yr
        # arepo's own eps*Mdot*c^2 conversion factor is calibrated for the raw
        # code-unit BH_mdot, not the Msun/yr version above.
        BH_Lbol = BH_mdot * mdot_to_Lbol.get_conversion_factor_arepo(eps) #erg/s

        nBHs = len(BH_coords)
        if not (len(BH_mass) == len(BH_mdot) == nBHs):
            raise RuntimeError(
                f"snap {snap} subhalo {i}: BH field lengths disagree "
                f"(coords {nBHs}, mass {len(BH_mass)}, mdot {len(BH_mdot)})")

        #distance of BHs from subhalo PoS
        d = BH_coords - subhalo_pos[i]
        # the box wraps around (periodic boundary), so a BH near one edge and
        # the subhalo near the opposite edge are actually close together, not
        # far apart. this snaps any separation bigger than half the box back
        # to the short way around, giving the true shortest distance.
        d -= boxsize * np.round(d / boxsize)             # minimum image
        r_code = np.linalg.norm(d, axis=1)               # 3D distance per BH
        r_kpc = r_code * to_kpc                           # code units -> physical kpc

        subhalo_stellar_halfmass_rad = float(subhalo_cat["SubhaloHalfmassRadType"][i, 4]) * to_kpc

        BH_closest_to_center = np.zeros(nBHs, dtype=bool)
        BH_closest_to_center[np.argmin(r_code)] = True
        BH_heaviest = np.zeros(nBHs, dtype=bool)
        BH_heaviest[np.argmax(BH_mass)] = True

        # a BH sitting outside the stellar half-mass radius counts as
        # "wandering" and everything inside it is a central BH.
        BH_is_wandering = r_kpc > subhalo_stellar_halfmass_rad

        #subhalo_data dictionary
        subhalo_data.append(dict(
            snap=snap, 
            redshift=z, 
            index=i, 
            bh_count=nBHs,
            is_central=bool(is_central_mask[i]),
            pos=subhalo_pos[i], cm=subhalo_cm[i],
            subhalo_total_mass_msun=float(subhalo_cat["SubhaloMass"][i]) * 1e10 / h,
            stellar_mass_msun=float(subhalo_cat["SubhaloMassType"][i, 4]) * 1e10 / h,
            stellar_halfrad_kpc=subhalo_stellar_halfmass_rad,
        ))

        #bh_data dictionary
        bh_data.append(dict(
            n=nBHs, 
            index=i, 
            BH_r_kpc=r_kpc, 
            BH_r_code=r_code,
            BH_masses = BHmass_msun,
            BH_r_over_stellar_halfrad=r_kpc / subhalo_stellar_halfmass_rad,
            BH_Mdot = BH_Mdot_msun_per_yr,
            BH_Lbol = BH_Lbol,
            BH_closest=BH_closest_to_center,
            BH_heaviest=BH_heaviest,
            BH_is_wandering=BH_is_wandering,
            BH_coords=BH_coords,
        ))

    return bh_data, subhalo_data, n_sub_total, dict(HubbleParam=h, BoxSize=boxsize)


def write_catalogue(f, bh_data, subhalo_data, snapshot_summary):
    """Write the full /bh, /subhalo and /snapshot tables in one shot, after
    every snapshot has been collected in memory."""
    
    bh_group = f.require_group("bh")
    subhalo_group = f.require_group("subhalo")
    snapshot_group = f.require_group("snapshot")
    
    bh_count_per_subhalo = np.array([b["n"] for b in bh_data], dtype=np.int64)

    def concat_bh_column(key, dtype):
        """Stack one BH field across all subhalos into a single flat array."""
        if not bh_data:
            return np.zeros(0, dtype=dtype)
        return np.concatenate([np.asarray(b[key]).ravel() for b in bh_data]).astype(dtype)

    # which /subhalo row each BH belongs to, and where each subhalo's BHs
    # start within /bh
    bh_subhalo_row = np.repeat(np.arange(len(subhalo_data), dtype=np.int64), bh_count_per_subhalo)

    if len(bh_count_per_subhalo):
        bh_start_per_subhalo = np.concatenate([[0], np.cumsum(bh_count_per_subhalo)[:-1]])
    else:
        bh_start_per_subhalo = np.zeros(0, dtype=np.int64)

    # snap/redshift per BH comes from its host subhalo, repeated per BH count
    bh_group.create_dataset("snap", data=np.repeat(
        [s["snap"] for s in subhalo_data], bh_count_per_subhalo).astype(np.int16))
    bh_group.create_dataset("redshift", data=np.repeat(
        [s["redshift"] for s in subhalo_data], bh_count_per_subhalo).astype(np.float32))
    bh_group.create_dataset("subhalo_index", data=np.repeat(
        [b["index"] for b in bh_data], bh_count_per_subhalo).astype(np.int32)
        if bh_data else np.zeros(0, dtype=np.int32))
    bh_group.create_dataset("subhalo_row", data=bh_subhalo_row)
    bh_group.create_dataset("r_kpc", data=concat_bh_column("BH_r_kpc", np.float32))
    bh_group.create_dataset("r_ckpc_over_h", data=concat_bh_column("BH_r_code", np.float32))
    bh_group.create_dataset("r_over_stellar_halfrad",
                             data=concat_bh_column("BH_r_over_stellar_halfrad", np.float32))
    bh_group.create_dataset("mass_msun", data=concat_bh_column("BH_masses", np.float32))
    bh_group.create_dataset("mdot_msun_per_yr", data=concat_bh_column("BH_Mdot", np.float32))
    bh_group.create_dataset("lbol_erg_s", data=concat_bh_column("BH_Lbol", np.float64))
    bh_group.create_dataset("is_closest_to_centre", data=concat_bh_column("BH_closest", bool))
    bh_group.create_dataset("is_most_massive", data=concat_bh_column("BH_heaviest", bool))
    bh_group.create_dataset("is_wandering", data=concat_bh_column("BH_is_wandering", bool))
    bh_group.create_dataset("pos_ckpc_over_h", data=(
        np.concatenate([b["BH_coords"] for b in bh_data]).astype(np.float32)
        if bh_data else np.zeros((0, 3), dtype=np.float32)))

    subhalo_group.create_dataset("snap", data=np.array([s["snap"] for s in subhalo_data], dtype=np.int16))
    subhalo_group.create_dataset("redshift", data=np.array([s["redshift"] for s in subhalo_data], dtype=np.float32))
    subhalo_group.create_dataset("index", data=np.array([s["index"] for s in subhalo_data], dtype=np.int32))
    subhalo_group.create_dataset("bh_start", data=bh_start_per_subhalo)
    subhalo_group.create_dataset("bh_count", data=bh_count_per_subhalo.astype(np.int32))
    subhalo_group.create_dataset("is_central", data=np.array([s["is_central"] for s in subhalo_data], dtype=bool))
    subhalo_group.create_dataset("pos_ckpc_over_h", data=(
        np.array([s["pos"] for s in subhalo_data], dtype=np.float32)
        if subhalo_data else np.zeros((0, 3), dtype=np.float32)))
    subhalo_group.create_dataset("cm_ckpc_over_h", data=(
        np.array([s["cm"] for s in subhalo_data], dtype=np.float32)
        if subhalo_data else np.zeros((0, 3), dtype=np.float32)))
    for key, dtype in [("subhalo_total_mass_msun", np.float32),
                        ("stellar_mass_msun", np.float32),
                        ("stellar_halfrad_kpc", np.float32)]:
        subhalo_group.create_dataset(key, data=np.array([s[key] for s in subhalo_data], dtype=dtype))

    snapshot_group.create_dataset("snap", data=np.array([s["snap"] for s in snapshot_summary], dtype=np.int16))
    snapshot_group.create_dataset("redshift", data=np.array([s["redshift"] for s in snapshot_summary], dtype=np.float32))
    snapshot_group.create_dataset("n_subhalos", data=np.array([s["n_subhalos"] for s in snapshot_summary], dtype=np.int32))
    snapshot_group.create_dataset("n_subhalos_with_bh",
                                   data=np.array([s["n_subhalos_with_bh"] for s in snapshot_summary], dtype=np.int32))
    snapshot_group.create_dataset("n_bh", data=np.array([s["n_bh"] for s in snapshot_summary], dtype=np.int64))
    snapshot_group.create_dataset("bh_start", data=np.array([s["bh_start"] for s in snapshot_summary], dtype=np.int64))
    snapshot_group.create_dataset("subhalo_start",
                                   data=np.array([s["subhalo_start"] for s in snapshot_summary], dtype=np.int64))

    f.flush()
    return int(bh_count_per_subhalo.sum())


# main
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="BH-subhalo catalogue for wandering-BH statistics.")
    p.add_argument("--simName", default="SM5_DFD_3_TNG/")
    p.add_argument("--simPath", default="/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_"
                           "UNIFORM_RUNS/L12p5n512/AREPO/")
    p.add_argument("--outPath", default="/orange/lblecha/pranavsatheesh/wandering_bhs/Data/")
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="explicit snapshot numbers")
    args = p.parse_args()

    sim_file = os.path.join(args.simPath, args.simName)
    outfile_name = os.path.join(args.outPath, f"wandering_bh_{args.simName.strip('/').replace('/', '_')}.hdf5")

    snapshots, redshifts = arepo.get_snapshot_redshift_correspondence(sim_file)
    snapshots = np.asarray(snapshots)
    redshifts = np.asarray(redshifts, dtype=float)
    z_of = dict(zip(snapshots.tolist(), redshifts.tolist()))

    snaps = args.snaps if args.snaps else snapshots.tolist()
    snaps_todo = sorted(snaps, key=lambda s: -z_of[s])  # process highest redshift (earliest time) first

    with h5py.File(outfile_name, "w") as f:

        f.attrs["sim_name"] = args.simName
        f.attrs["sim_path"] = args.simPath
        f.attrs["centre_definition"] = "SubhaloPos (potential minimum)"
        f.attrs["units"] = ("mass: Msun | r_kpc: physical kpc | "
                            "r_ckpc_over_h: code (comoving kpc/h) | lbol: erg/s")

        all_bh_data, all_subhalo_data, snapshot_summary = [], [], []
        bh_row_offset, subhalo_row_offset = 0, 0

        for snap in snaps_todo:
            z = z_of[snap]

            bhs_in_each_subhalo, subhalo_information, n_sub_total, header = snapshot_rows(
                sim_file, snap, z, max_subhalos=None if args.numSubhalos < 0 else args.numSubhalos)

            n_bh_this_snap = sum(b["n"] for b in bhs_in_each_subhalo)
            snapshot_summary.append(dict(
                snap=snap, redshift=z, n_subhalos=n_sub_total,
                n_subhalos_with_bh=len(subhalo_information), 
                n_bh=n_bh_this_snap,
                bh_start=bh_row_offset, 
                subhalo_start=subhalo_row_offset,
            ))

            all_bh_data.extend(bhs_in_each_subhalo)
            all_subhalo_data.extend(subhalo_information)
            
            bh_row_offset += n_bh_this_snap
            subhalo_row_offset += len(subhalo_information)

            if "HubbleParam" not in f.attrs:
                f.attrs["HubbleParam"] = header["HubbleParam"]
                f.attrs["BoxSize_ckpc_over_h"] = header["BoxSize"]

            print(f"snap {snap} (z={z:.3f}): {len(subhalo_information)}/{n_sub_total} subhalos "
                  f" and has {n_bh_this_snap} BHs", flush=True)

        total_nBHs = write_catalogue(f, all_bh_data, all_subhalo_data, snapshot_summary)
        print(f"wrote {total_nBHs} BHs across {len(snapshot_summary)} snapshots")

    print(f"wrote {outfile_name}")


# Example hdf5 file structure:
# wandering_bh_SM5_DFD_3_TNG.hdf5
# │
# ├── attrs: sim_name, sim_path, centre_definition, units,
# │          HubbleParam, BoxSize_ckpc_over_h
# │
# ├── /bh/                              (one row per black hole, N_bh rows)
# │     snap                 int16
# │     redshift              float32
# │     subhalo_index         int32     # index into the ORIGINAL subhalo catalogue
# │     subhalo_row           int64     # row into /subhalo below
# │     r_kpc                 float32   # distance from host subhalo, physical kpc
# │     r_ckpc_over_h         float32   # same, in code units
# │     r_over_stellar_halfrad float32
# │     mass_msun             float32
# │     mdot_msun_per_yr      float32
# │     lbol_erg_s            float64
# │     is_closest_to_centre  bool
# │     is_most_massive       bool
# │     is_wandering          bool
# │     pos_ckpc_over_h       float32   shape (N_bh, 3)
# │
# ├── /subhalo/                         (one row per BH-hosting subhalo, N_sub rows)
# │     snap                      int16
# │     redshift                   float32
# │     index                      int32   # index into the original subhalo catalogue
# │     bh_start                   int64   # first row in /bh belonging to this subhalo
# │     bh_count                   int32
# │     is_central                 bool
# │     pos_ckpc_over_h            float32  shape (N_sub, 3)
# │     cm_ckpc_over_h             float32  shape (N_sub, 3)
# │     subhalo_total_mass_msun    float32
# │     stellar_mass_msun          float32
# │     stellar_halfrad_kpc        float32
# │
# └── /snapshot/                        (one row per snapshot processed, N_snap rows)
#       snap                 int16
#       redshift              float32
#       n_subhalos            int32   # total subhalos in that snapshot
#       n_subhalos_with_bh    int32
#       n_bh                  int64
#       bh_start              int64   # offset into /bh for this snapshot's rows
#       subhalo_start         int64   # offset into /subhalo for this snapshot's rows


#Example read
# import h5py

# with h5py.File("wandering_bh_SM5_DFD_3_TNG.hdf5") as f:
#     is_wandering = f["bh/is_wandering"][:]
#     r_kpc = f["bh/r_kpc"][is_wandering]          # distances of just the wandering BHs

#     # which subhalo hosts each of those BHs
#     host_rows = f["bh/subhalo_row"][is_wandering]
#     host_mass = f["subhalo/stellar_mass_msun"][:][host_rows]
