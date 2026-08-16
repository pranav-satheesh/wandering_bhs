import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from cosmo_sim_tools.arepo_tools import arepo_package as arepo
from cosmo_sim_tools import brahma
from cosmo_sim_tools.arepo_tools import mdot_to_Lbol


# view -> which two centred axes form the (horizontal, vertical) plane
_VIEW_AXES = {
    'xy': (0, 1), 'yx': (1, 0),
    'xz': (0, 2), 'zx': (2, 0),
    'yz': (1, 2), 'zy': (2, 1),
}


def _load_subhalo_field(sim_file, field, p_type, z, subhalo_index):
    """Thin wrapper around the group-catalog particle loader.

    Returns just the subhalo particle array for `field`, or None if the
    subhalo has no particles of this type.

    Uses get_particle_property_within_groups (NOT ..._postprocessed): the
    postprocessed catalogue has a different subhalo ordering/count, so feeding
    it an index from brahma.groupcat.loadSubhalos returns the wrong (often
    empty) particle set for satellite subhaloes. This loader is consistent
    with the standard catalogue used here for the centre and n_bh.
    """
    # store_all_offsets=0 -> compute group offsets on the fly instead of
    # caching them to a .npy inside the (read-only) simulation folder.
    out = arepo.get_particle_property_within_groups(
        sim_file, particle_property=[field], p_type=p_type,
        desired_redshift=z, subhalo_index=subhalo_index, group_type='subhalo',
        store_all_offsets=0)
    arr = out[0]
    if arr is None or len(arr) == 0:
        return None
    return np.asarray(arr)


def bh_subhalo_2Dplot(sim_file, snap, desired_redshift, subhalo_index,
                      view='xy', bg_ptype=0, box_length=None, Nbins=400,
                      radiative_efficiency=0.2,
                      cmap='plasma', bg_cmap='bone', bg_smooth=1.2,
                      bg_dyn_range=3.5, facecolor='black',
                      s_min=4, s_max=120, logM_ref=(5.0, 9.0),
                      lum_floor=1e38, vmin=None, vmax=None,
                      ax=None, figsize=(6.5, 5), show_size_legend=True,
                      save_name=None, dpi=200):
    """
    2D map of the black holes in a subhalo, with the subhalo particles as a
    smoothed surface-density background.

      * marker SIZE  scales with log10(M_BH)
      * marker COLOR shows the BH bolometric luminosity (L_bol = eps * Mdot c^2)

    Distances are physical kpc (comoving coords * scale factor a / h).

    sim_file       : path to the simulation output folder (Brahma_sim_file)
    snap           : snapshot number (used for the subhalo catalogue)
    desired_redshift : redshift passed to the particle loaders
    subhalo_index  : subhalo index within the catalogue
    view           : projection plane, one of xy/yx/xz/zx/yz/zy
    bg_ptype       : particle type for the background image
                     (0 = gas, 1 = DM, 4 = stars)
    box_length     : full side length of the image in kpc; None -> auto
    Nbins          : background image resolution
    radiative_efficiency : eps in L_bol = eps * Mdot c^2 (codebase default 0.2)
    bg_cmap, bg_smooth, bg_dyn_range, facecolor : background styling
                     (gaussian smoothing sigma, dex shown below the peak)
    logM_ref       : (min, max) log10(M_BH/Msun) mapped onto (s_min, s_max)
    lum_floor      : L_bol value assigned to non-accreting (Mdot=0) BHs
    vmin, vmax     : colour limits for log10(L_bol); None -> from data
    ax             : existing axis to draw on; None -> make a new figure
    """
    header = brahma.groupcat.loadHeader(sim_file, snap)
    h = header['HubbleParam']
    boxsize = header['BoxSize']
    a = header.get('Time', 1.0)                  # scale factor (a = 1/(1+z))
    masstable = header.get('MassTable', None)

    subhalos = brahma.groupcat.loadSubhalos(
        sim_file, snap, fields=['SubhaloLenType', 'SubhaloPos'])
    center = np.asarray(subhalos['SubhaloPos'][subhalo_index], dtype=float)
    n_bh = int(subhalos['SubhaloLenType'][subhalo_index, 5])

    if n_bh == 0:
        print(f"subhalo {subhalo_index} (snap {snap}) has no black holes.")
        return None

    # ---- black holes -----------------------------------------------------
    bh_coords = _load_subhalo_field(sim_file, 'Coordinates', 5,
                                    desired_redshift, subhalo_index)
    bh_mass = _load_subhalo_field(sim_file, 'BH_Mass', 5,
                                  desired_redshift, subhalo_index)
    bh_mdot = _load_subhalo_field(sim_file, 'BH_Mdot', 5,
                                  desired_redshift, subhalo_index)

    bh_mass = np.ravel(bh_mass)
    bh_mdot = np.ravel(bh_mdot)
    MBH = bh_mass * 1e10 / h                              # Msun
    conv = mdot_to_Lbol.get_conversion_factor_arepo(radiative_efficiency)
    Lbol = bh_mdot * conv                                 # erg/s
    Lbol = np.where(Lbol > 0, Lbol, lum_floor)            # avoid log(0)

    # centre on the subhalo, correct for periodic box, convert to physical kpc
    def _recentre(coords):
        d = np.asarray(coords, dtype=float) - center
        d -= boxsize * np.round(d / boxsize)              # periodic wrap
        return d * a / h                                  # physical kpc

    bh_d = _recentre(bh_coords)
    ih, iv = _VIEW_AXES[view]
    bh_x, bh_y = bh_d[:, ih], bh_d[:, iv]

    # ---- auto box size ---------------------------------------------------
    if box_length is None:
        rmax = np.max(np.sqrt(bh_x**2 + bh_y**2)) if len(bh_x) else 1.0
        box_length = max(2.0 * 1.15 * rmax, 5.0)
    half = box_length / 2.0

    # ---- figure / axis ---------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    ax.set_facecolor(facecolor)

    # ---- background: smoothed subhalo surface density --------------------
    # (styling borrowed from twoDplot.galaxy2Dplots: mass-weighted projection,
    #  gaussian smoothing, log10, displayed with a clipped dynamic range)
    bg = _load_subhalo_field(sim_file, 'Coordinates', bg_ptype,
                             desired_redshift, subhalo_index)
    if bg is not None:
        bg_d = _recentre(bg)
        bx, by = bg_d[:, ih], bg_d[:, iv]

        # per-particle mass in code units: uniform from the MassTable when set
        # (e.g. DM), otherwise the per-particle Masses field (gas/stars).
        weights = None
        if masstable is not None and masstable[bg_ptype] > 0:
            weights = np.full(len(bx), masstable[bg_ptype])
        else:
            mfield = _load_subhalo_field(sim_file, 'Masses', bg_ptype,
                                         desired_redshift, subhalo_index)
            if mfield is not None:
                weights = np.ravel(mfield)

        edges = np.linspace(-half, half, Nbins + 1)
        H, xe, ye = np.histogram2d(bx, by, bins=[edges, edges],
                                   weights=weights)
        pix_kpc = box_length / Nbins
        if weights is not None:
            H = H * 1e10 / h / (pix_kpc ** 2)             # Msun / kpc^2
        H = gaussian_filter(H, sigma=bg_smooth)

        with np.errstate(divide='ignore'):
            logH = np.log10(H)
        finite = np.isfinite(logH)
        if finite.any():
            vmax_bg = np.percentile(logH[finite], 99.5)
            vmin_bg = vmax_bg - bg_dyn_range
            logH = np.ma.masked_where(~finite | (logH < vmin_bg), logH)
            ax.pcolormesh(xe, ye, logH.T, cmap=bg_cmap,
                          vmin=vmin_bg, vmax=vmax_bg, rasterized=True,
                          zorder=0)
    else:
        print(f"subhalo {subhalo_index} has no p_type={bg_ptype} particles "
              "for the background.")

    # ---- black hole scatter ---------------------------------------------
    logM = np.log10(MBH)
    lo, hi = logM_ref
    frac = np.clip((logM - lo) / (hi - lo), 0.0, 1.0)
    sizes = s_min + (s_max - s_min) * frac

    c = np.log10(Lbol)
    sc = ax.scatter(bh_x, bh_y, s=sizes, c=c, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor='white', linewidth=0.4,
                    zorder=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\log_{10}(L_{\mathrm{bol}}\;[\mathrm{erg\,s^{-1}}])$')

    # ---- size legend -----------------------------------------------------
    if show_size_legend:
        ref_logM = np.linspace(lo, hi, 4)
        handles = []
        for lm in ref_logM:
            f = np.clip((lm - lo) / (hi - lo), 0.0, 1.0)
            s = s_min + (s_max - s_min) * f
            handles.append(ax.scatter([], [], s=s, facecolor='lightgrey',
                                      edgecolor='k', linewidth=0.4,
                                      label=r'$10^{%.1f}$' % lm))
        ax.legend(handles=handles, title=r'$M_{\rm BH}\,[M_\odot]$',
                  loc='upper right', labelspacing=1.1, frameon=True,
                  fontsize=8, title_fontsize=8)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{view[0]} [kpc]')
    ax.set_ylabel(f'{view[1]} [kpc]')
    ax.set_title(f"z= {desired_redshift}: subhalo {subhalo_index}, {n_bh} BH(s)")

    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight', dpi=dpi)
    return ax
