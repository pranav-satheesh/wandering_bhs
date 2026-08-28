#!/usr/bin/env python
"""Driver for bh_2dplot.bh_subhalo_2Dplot, meant to be launched from SLURM.

Example:
    python run_bh_2dplot.py \
        --basePath /orange/.../SM5_DFD_3_TNG \
        --snap 32 --subhalo 0 --outDir ./figures/2d_plots \
        --bg_ptype 1 --view xy --radiative_efficiency 0.2 --box_length auto
"""
import os
import argparse

import matplotlib
matplotlib.use('Agg')   # no display on a compute node

from cosmo_sim_tools.arepo_tools import arepo_package as arepo
import bh_2dplot


def snap_to_redshift(basePath, snap):
    """Map a snapshot number to its redshift for this simulation."""
    snaps, zs = arepo.get_snapshot_redshift_correspondence(basePath)
    snaps = list(snaps)
    return zs[snaps.index(snap)]


def main():
    p = argparse.ArgumentParser(description="2D BH-in-subhalo plot driver")
    p.add_argument('--basePath', required=True,
                   help='simulation output folder (Brahma_sim_file)')
    p.add_argument('--snap', type=int, required=True, help='snapshot number')
    p.add_argument('--subhalo', type=int, required=True, help='subhalo index')
    p.add_argument('--outDir', required=True, help='output directory')
    p.add_argument('--bg_ptype', type=int, default=1,
                   help='background particle type (1=DM, 0=gas, 4=stars)')
    p.add_argument('--view', default='xy',
                   help='projection plane: xy/yx/xz/zx/yz/zy')
    p.add_argument('--radiative_efficiency', type=float, default=0.2,
                   help='eps in L_bol = eps * Mdot c^2')
    p.add_argument('--box_length', default='auto',
                   help='full image side in ckpc, or "auto"')
    p.add_argument('--nbins', type=int, default=300,
                   help='background histogram resolution')
    p.add_argument('--bg_cmap', default='afmhot',
                   help="background galaxy colormap (e.g. afmhot, gist_heat, hot, bone)")
    p.add_argument('--bg_method', default='adaptive', choices=['adaptive', 'hist'],
                   help="background smoothing: adaptive (kNN, smooth+sharp) or hist (plain)")
    p.add_argument('--adaptive_k', type=int, default=16,
                   help="adaptive neighbours (bigger=smoother, smaller=sharper)")
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--bh_cmap', default='plasma',
                   help="black hole colormap (e.g. plasma, viridis, inferno, magma)")
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    box_length = None if str(args.box_length).lower() == 'auto' \
        else float(args.box_length)

    z = snap_to_redshift(args.basePath, args.snap)
    print(f"snap {args.snap} -> z = {z:.4f}; subhalo {args.subhalo}; "
          f"bg_ptype={args.bg_ptype}; view={args.view}")

    # unique per-run suffix (SLURM job id) so each submission writes a NEW
    # file -> no stale PDF-viewer cache showing an old render of the same path
    jobid = os.environ.get('SLURM_JOB_ID', '')
    suffix = f"_{jobid}" if jobid else ""
    save_name = os.path.join(
        args.outDir,
        f"bh_2dplot_snap{args.snap}_sub{args.subhalo}_{args.view}_"
        f"bg{args.bg_ptype}_{args.box_length}_{args.bg_cmap}_{args.bg_method}{suffix}.pdf")

    ax = bh_2dplot.bh_subhalo_2Dplot(
        args.basePath, args.snap, z, args.subhalo,
        view=args.view, bg_ptype=args.bg_ptype, box_length=box_length,
        Nbins=args.nbins, radiative_efficiency=args.radiative_efficiency,cmap=args.bh_cmap,
        bg_cmap=args.bg_cmap, bg_method=args.bg_method,
        adaptive_k=args.adaptive_k,
        save_name=save_name, dpi=args.dpi)

    if ax is None:
        print("No figure produced (no black holes in this subhalo).")
    else:
        print(f"Saved {save_name}")


if __name__ == '__main__':
    main()
