#!/usr/bin/env python
"""Side-by-side (1 x N) BH-in-subhalo maps of the same galaxy across several
seed models, meant to be launched from SLURM.

Galaxies are chosen from a target redshift and stellar mass. Every run uses the
snapshot nearest the target redshift. In the first (reference) run the galaxy
is the BH-hosting subhalo closest to the target stellar mass; in each other run
its counterpart is the nearest BH-hosting subhalo of the same type
(central/satellite) and similar total mass. The runs share initial conditions,
so this is the same galaxy, and the script prints the separation of each
centre from the reference one to confirm it.

All panels use the same box size, BH size scale and L_bol colour range, so
the BH populations can be compared directly.

Example:
    python run_bh_2dplot_seed_comparison.py \
        --simPath /orange/.../L12p5n512/AREPO/ \
        --runs SM5_DFD_3_TNG SM5_LW10_DFD_3_TNG SM5_LW10_LOWSPIN_RICH_DFD_3_TNG \
        --redshift 0 --log_stellar_mass 10.75 --outDir ./figures/2d_plots \
        --bg_ptype 0 --box_length 40 --views xy xz yz
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display on a compute node
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cosmo_sim_tools.arepo_tools import arepo_package as arepo
from cosmo_sim_tools.arepo_tools import mdot_to_Lbol
from cosmo_sim_tools import brahma
import bh_2dplot


SEED_MODEL_LABELS = {
    'SM5_DFD_3_TNG': 'SM5',
    'SM5_LW10_DFD_3_TNG': 'SM5_LW10',
    'SM5_LW10_LOWSPIN_DFD_3_TNG': 'SM5_LW10_LOWSPIN',
    'SM5_LW10_LOWSPIN_RICH_DFD_3_TNG': 'SM5_LW10_LOWSPIN_RICH'
}


def closest_snapshot(sim_file, redshift):
    """(snapshot, redshift) of the snapshot nearest the requested redshift."""
    snaps, zs = arepo.get_snapshot_redshift_correspondence(sim_file)
    position = int(np.argmin(np.abs(np.asarray(zs, dtype=float) - redshift)))
    return int(snaps[position]), float(zs[position])


def subhalo_properties(sim_file, snap):
    """Postprocessed-catalogue masses, BH counts, centres and central flags."""
    header = brahma.groupcat.loadHeader(sim_file, snap)
    h = header['HubbleParam']
    subhalos = brahma.groupcat.loadSubhalos_postprocessed(
        sim_file, snap,
        fields=['SubhaloLenType', 'SubhaloMassType', 'SubhaloMass',
                'SubhaloPos', 'SubhaloGrNr'])
    group_first_subhalo = brahma.groupcat.loadHalos_postprocessed(
        sim_file, snap, fields='GroupFirstSub').astype(np.int64)
    group_number = subhalos['SubhaloGrNr'].astype(np.int64)
    return {
        'stellar_mass': subhalos['SubhaloMassType'][:, 4] * 1e10 / h,   # Msun
        'total_mass': subhalos['SubhaloMass'] * 1e10 / h,               # Msun
        'n_bh': subhalos['SubhaloLenType'][:, 5],
        'position': np.asarray(subhalos['SubhaloPos'], dtype=float),    # ckpc/h
        'is_central': group_first_subhalo[group_number] == np.arange(group_number.size),
        'scale_factor': header['Time'],
        'h': h,
        'boxsize': header['BoxSize'],
    }


def pick_subhalo_by_stellar_mass(props, log_stellar_mass, halo_type='any'):
    """Index of the BH-hosting subhalo closest to the target log10 M_star.

    `halo_type` restricts the choice to 'central' or 'satellite' subhalos.
    """
    eligible = (props['n_bh'] > 0) & (props['stellar_mass'] > 0)
    if halo_type == 'central':
        eligible &= props['is_central']
    elif halo_type == 'satellite':
        eligible &= ~props['is_central']
    log_stellar = np.log10(np.where(eligible, props['stellar_mass'], 1.0))
    return int(np.argmin(np.where(eligible, np.abs(log_stellar - log_stellar_mass),
                                  np.inf)))


def match_subhalo(reference, reference_index, candidates, max_dlog_total_mass=0.5):
    """Index of the candidate subhalo that is the same galaxy as the reference one.

    The runs share initial conditions, so the same galaxy sits at nearly the
    same place in every run. Matching on mass alone can pick an unrelated
    galaxy of similar mass elsewhere in the box, so the match is the nearest
    BH-hosting subhalo of the same type (central/satellite) whose total mass is
    within `max_dlog_total_mass` dex of the reference.
    """
    log_total = np.log10(np.where(candidates['total_mass'] > 0, candidates['total_mass'], 1.0))
    eligible = ((candidates['n_bh'] > 0) & (candidates['stellar_mass'] > 0)
                & (candidates['is_central'] == reference['is_central'][reference_index])
                & (np.abs(log_total - np.log10(reference['total_mass'][reference_index]))
                   < max_dlog_total_mass))
    separation = candidates['position'] - reference['position'][reference_index]
    separation -= reference['boxsize'] * np.round(separation / reference['boxsize'])
    distance = np.linalg.norm(separation, axis=1)
    return int(np.argmin(np.where(eligible, distance, np.inf)))


def log_bolometric_luminosity(sim_file, redshift, subhalo_index,
                              radiative_efficiency, lum_floor):
    """log10 L_bol of every BH in the subhalo, as bh_subhalo_2Dplot computes it."""
    mdot = np.ravel(bh_2dplot._load_subhalo_field(sim_file, 'BH_Mdot', 5,
                                                  redshift, subhalo_index))
    lbol = mdot * mdot_to_Lbol.get_conversion_factor_arepo(radiative_efficiency)
    return np.log10(np.where(lbol > 0, lbol, lum_floor))


def main():
    p = argparse.ArgumentParser(
        description="Seed-model comparison of 2D BH-in-subhalo plots, one panel per run")
    p.add_argument('--simPath', required=True,
                   help='folder holding every seed-model run')
    p.add_argument('--runs', nargs='+', default=['SM5_DFD_3_TNG', 'SM5_LW10_DFD_3_TNG'],
                   help='run names, one panel each; the galaxy is picked in the first '
                        'and matched in the others')
    p.add_argument('--redshift', type=float, required=True,
                   help='target redshift; the nearest snapshot is used')
    p.add_argument('--log_stellar_mass', type=float, required=True,
                   help='target log10(M_star / Msun)')
    p.add_argument('--halo_type', default='any', choices=['any', 'central', 'satellite'],
                   help='restrict the reference pick (and so its matches) to centrals '
                        'or satellites')
    p.add_argument('--max_dlogm', type=float, default=0.1,
                   help='max |log M_star - target| (dex) allowed for any galaxy')
    p.add_argument('--outDir', required=True, help='output directory')
    p.add_argument('--bg_ptype', type=int, default=0,
                   help='background particle type (1=DM, 0=gas, 4=stars)')
    p.add_argument('--views', nargs='+', default=['xy'],
                   choices=sorted(bh_2dplot._VIEW_AXES),
                   help='projection planes, one figure each (e.g. xy xz yz)')
    p.add_argument('--radiative_efficiency', type=float, default=0.2,
                   help='eps in L_bol = eps * Mdot c^2')
    p.add_argument('--lum_floor', type=float, default=1e38,
                   help='L_bol assigned to non-accreting BHs')
    p.add_argument('--box_length', default='40',
                   help='full image side in kpc (shared by all panels)')
    p.add_argument('--nbins', type=int, default=400,
                   help='background histogram resolution')
    p.add_argument('--bg_cmap', default='inferno', help='background colormap')
    p.add_argument('--bg_method', default='adaptive', choices=['adaptive', 'hist'],
                   help='background smoothing: adaptive (kNN) or hist (plain)')
    p.add_argument('--adaptive_k', type=int, default=6,
                   help='adaptive neighbours (bigger=smoother, smaller=sharper)')
    p.add_argument('--bh_cmap', default='viridis', help='black hole colormap')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    sim_files = [os.path.join(args.simPath, run) for run in args.runs]

    snap, z = closest_snapshot(sim_files[0], args.redshift)
    for run, sim_file in zip(args.runs[1:], sim_files[1:]):
        snap_run, z_run = closest_snapshot(sim_file, args.redshift)
        if snap_run != snap or not np.isclose(z, z_run, atol=1e-3):
            raise ValueError(f"nearest snapshot to z={args.redshift} is snap {snap} "
                             f"(z={z:.4f}) in {args.runs[0]} but snap {snap_run} "
                             f"(z={z_run:.4f}) in {run}")

    props = [subhalo_properties(sim_file, snap) for sim_file in sim_files]
    reference, reference_index = props[0], pick_subhalo_by_stellar_mass(
        props[0], args.log_stellar_mass, args.halo_type)
    indices = [reference_index] + [match_subhalo(reference, reference_index, candidates)
                                   for candidates in props[1:]]

    print(f"target z = {args.redshift} -> snap {snap}, z = {z:.4f}; "
          f"target log M* = {args.log_stellar_mass:.3f}")
    for run, run_props, index in zip(args.runs, props, indices):
        # same ICs in every run -> a small separation means it is the same galaxy
        separation = run_props['position'][index] - reference['position'][reference_index]
        separation -= reference['boxsize'] * np.round(separation / reference['boxsize'])
        separation_kpc = np.linalg.norm(separation) * reference['scale_factor'] / reference['h']
        kind = 'central' if run_props['is_central'][index] else 'satellite'
        print(f"  {run:>32s} subhalo {index:5d}: "
              f"log M* = {np.log10(run_props['stellar_mass'][index]):.3f}, "
              f"log Mtot = {np.log10(run_props['total_mass'][index]):.3f}, "
              f"N_BH = {run_props['n_bh'][index]}, {kind}, "
              f"{separation_kpc:.1f} kpc from the reference centre")

    for run, run_props, index in zip(args.runs, props, indices):
        dlogm = np.log10(run_props['stellar_mass'][index]) - args.log_stellar_mass
        if abs(dlogm) > args.max_dlogm:
            raise ValueError(f"{run} subhalo {index} is {dlogm:+.2f} dex from the "
                             f"target stellar mass (limit {args.max_dlogm} dex)")

    # one L_bol colour range spanning every subhalo
    log_lbol = np.concatenate([
        log_bolometric_luminosity(sim_file, z, index,
                                  args.radiative_efficiency, args.lum_floor)
        for sim_file, index in zip(sim_files, indices)])
    vmin, vmax = log_lbol.min(), log_lbol.max()

    box_length = float(args.box_length)
    n_runs = len(args.runs)
    jobid = os.environ.get('SLURM_JOB_ID', '')
    suffix = f"_{jobid}" if jobid else ""
    for view in args.views:
        fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5.4), dpi=args.dpi,
                                 squeeze=False)
        axes = axes[0]
        for panel, (ax, sim_file, run_name, index) in enumerate(
                zip(axes, sim_files, args.runs, indices)):
            bh_2dplot.bh_subhalo_2Dplot(
                sim_file, snap, z, index,
                view=view, bg_ptype=args.bg_ptype, box_length=box_length,
                Nbins=args.nbins, radiative_efficiency=args.radiative_efficiency,
                cmap=args.bh_cmap, bg_cmap=args.bg_cmap, bg_method=args.bg_method,
                adaptive_k=args.adaptive_k, lum_floor=args.lum_floor,
                vmin=vmin, vmax=vmax, ax=ax, show_size_legend=(panel == n_runs - 1),
                colorbar=False)
            label = SEED_MODEL_LABELS.get(run_name, run_name)
            ax.set_title(f"{label}\n{ax.get_title()}")

        cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=args.bh_cmap),
                            ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label(r'$\log_{10}(L_{\mathrm{bol}}\;[\mathrm{erg\,s^{-1}}])$')

        save_name = os.path.join(
            args.outDir,
            f"bh_2dplot_seedcomp_snap{snap}_logMstar{args.log_stellar_mass:g}_"
            f"{'_vs_'.join(SEED_MODEL_LABELS.get(run, run) for run in args.runs)}_"
            f"sub{'_'.join(str(index) for index in indices)}_{view}_"
            f"bg{args.bg_ptype}_{args.box_length}_{args.bg_cmap}_{args.bg_method}{suffix}.pdf")
        fig.savefig(save_name, bbox_inches='tight', dpi=args.dpi)
        print(f"Saved {save_name}")
        plt.close(fig)


if __name__ == '__main__':
    main()
