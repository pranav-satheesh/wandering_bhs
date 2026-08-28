import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
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


def _nice_length(x):
    """A 'nice' round number <= x (1, 2 or 5 x 10^n) for the scale bar."""
    if x <= 0:
        return 1.0
    e = np.floor(np.log10(x))
    f = x / 10 ** e
    nice = 5.0 if f >= 5 else (2.0 if f >= 2 else 1.0)
    return nice * 10 ** e


def _fmt_mass(m):
    """Format a mass in Msun as a LaTeX 'a x 10^b' string."""
    if m <= 0:
        return r"0"
    e = int(np.floor(np.log10(m)))
    mant = m / 10 ** e
    return rf"{mant:.1f}\times10^{{{e}}}"


def _load_subhalo_field(sim_file, field, p_type, z, subhalo_index):
    """Thin wrapper around the postprocessed-group particle loader.

    Returns just the subhalo particle array for `field`, or None if the
    subhalo has no particles of this type.

    Uses get_particle_property_within_postprocessed_groups together with the
    POSTPROCESSED subhalo catalogue (loadSubhalos_postprocessed) for the index
    and centre. The two must match: the loader reads the group-ordered snapshot
    via the postprocessed offsets, so indexing it with the standard
    loadSubhalos catalogue (different ordering/count) returns the right *number*
    of particles but the wrong ones - scattered across the whole box.
    """
    # store_all_offsets=0 -> compute group offsets on the fly instead of
    # caching them to a .npy inside the (read-only) simulation folder.
    out = arepo.get_particle_property_within_postprocessed_groups(
        sim_file, particle_property=[field], p_type=p_type,
        desired_redshift=z, subhalo_index=subhalo_index, group_type='subhalo',
        store_all_offsets=0)
    arr = out[0]
    if arr is None or len(arr) == 0:
        return None
    return np.asarray(arr)


def _adaptive_density(bx, by, weights, edges, k=16):
    """Adaptive (SPH-like) surface density on the grid defined by `edges`.

    Each particle's smoothing length is its distance to the k-th nearest
    neighbour, so sparse outskirts are smoothed heavily while the dense
    disk/arms stay sharp -> smooth map without the global blur of a single
    fixed kernel. Implemented as a few geometric smoothing bands (fast,
    fully vectorised) instead of per-particle splatting.
    """
    from scipy.spatial import cKDTree

    nb = len(edges) - 1
    n = len(bx)
    if n < 3:
        H, _, _ = np.histogram2d(bx, by, bins=[edges, edges], weights=weights)
        return H

    pix = edges[1] - edges[0]
    pts = np.column_stack([bx, by])
    kk = min(k, n - 1)
    dist, _ = cKDTree(pts).query(pts, k=kk + 1)        # +1: self at dist 0
    hsml_px = np.clip(dist[:, -1] / pix, 0.6, nb / 3.0)

    sig_bands = np.geomspace(hsml_px.min(), hsml_px.max(), 6)
    band = np.argmin(np.abs(np.log(hsml_px)[:, None]
                            - np.log(sig_bands)[None, :]), axis=1)
    H = np.zeros((nb, nb))
    for b, sig in enumerate(sig_bands):
        sel = band == b
        if not sel.any():
            continue
        w = None if weights is None else weights[sel]
        Hb, _, _ = np.histogram2d(bx[sel], by[sel],
                                  bins=[edges, edges], weights=w)
        H += gaussian_filter(Hb, sigma=sig)
    return H


def bh_subhalo_2Dplot(sim_file, snap, desired_redshift, subhalo_index,
                      view='xy', bg_ptype=0, box_length=None, Nbins=400,
                      radiative_efficiency=0.2,
                      cmap='plasma', bg_cmap='bone', bg_smooth=0.8,
                      bg_method='adaptive', adaptive_k=16, facecolor='black',
                      s_min=4, s_max=120, logM_ref=(5.0, 9.0),
                      lum_floor=1e38, vmin=None, vmax=None, scalebar=True,
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
    bg_cmap, facecolor : background colormap and axis face colour
    bg_method      : 'adaptive' (kNN/SPH-like smoothing; smooth but sharp) or
                     'hist' (plain 2D histogram + fixed gaussian, can be grainy)
    adaptive_k     : neighbours for the adaptive smoothing length (bigger=smoother)
    bg_smooth      : gaussian smoothing of the background image, in pixels (hist)
    scalebar       : draw a physical scale bar (twoDplot style) and hide ticks
    logM_ref       : (min, max) log10(M_BH/Msun) mapped onto (s_min, s_max)
    lum_floor      : L_bol value assigned to non-accreting (Mdot=0) BHs
    vmin, vmax     : colour limits for log10(L_bol); None -> from data
    ax             : existing axis to draw on; None -> make a new figure
    """
    header = brahma.groupcat.loadHeader(sim_file, snap)
    h = header['HubbleParam']
    boxsize = header['BoxSize']
    a = header.get('Time', 1.0)                  # scale factor (a = 1/(1+z))

    # POSTPROCESSED catalogue, to match the postprocessed particle loader.
    subhalos = brahma.groupcat.loadSubhalos_postprocessed(
        sim_file, snap, fields=['SubhaloLenType', 'SubhaloPos', 'SubhaloMassType'])
    center = np.asarray(subhalos['SubhaloPos'][subhalo_index], dtype=float)
    n_bh = int(subhalos['SubhaloLenType'][subhalo_index, 5])
    Mstar = float(subhalos['SubhaloMassType'][subhalo_index, 4]) * 1e10 / h  # Msun

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

        # Gas/stars carry a per-particle Masses field -> mass-weighted density.
        # DM has no Masses field (all DM particles share one mass), so that
        # load raises; fall back to unweighted counts, which give the same
        # density *shape* for a uniform-mass type (we log + percentile-clip
        # below, so the absolute normalisation is irrelevant).
        weights = None
        try:
            mfield = _load_subhalo_field(sim_file, 'Masses', bg_ptype,
                                         desired_redshift, subhalo_index)
            if mfield is not None:
                weights = np.ravel(mfield)
        except Exception:
            weights = None

        edges = np.linspace(-half, half, Nbins + 1)
        pix_kpc = box_length / Nbins
        if bg_method == 'adaptive':
            # SPH-like: smooth each particle over its kNN distance (smooth in
            # sparse outskirts, sharp in the dense disk) -> no salt-and-pepper.
            H = _adaptive_density(bx, by, weights, edges, k=adaptive_k)
        else:  # 'hist': plain 2D histogram + single fixed-width gaussian
            H, _, _ = np.histogram2d(bx, by, bins=[edges, edges],
                                     weights=weights)
            if bg_smooth:
                H = gaussian_filter(H, sigma=bg_smooth)
        if weights is not None:
            H = H * 1e10 / h / (pix_kpc ** 2)             # Msun / kpc^2

        Hpos = H[H > 0]
        if bg_method == 'adaptive' and Hpos.size:
            # clip to 3 dex below the peak so the smooth faint tails don't wash
            vmax_n = Hpos.max()
            vmin_n = vmax_n * 1e-3
            Hm = np.ma.masked_where(H < vmin_n, H)
            norm = mcolors.LogNorm(vmin=vmin_n, vmax=vmax_n)
        else:
            Hm = np.ma.masked_where(H <= 0, H)
            norm = mcolors.LogNorm()
        ax.pcolormesh(edges, edges, Hm.T, cmap=bg_cmap,
                      norm=norm, rasterized=True, zorder=0)
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
                    vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.7,
                    zorder=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\log_{10}(L_{\mathrm{bol}}\;[\mathrm{erg\,s^{-1}}])$')

    # ---- size legend -----------------------------------------------------
    if show_size_legend:
        # integer powers of ten spanning logM_ref (no fractional exponents)
        ref_logM = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        handles = []
        for lm in ref_logM:
            f = np.clip((lm - lo) / (hi - lo), 0.0, 1.0)
            s = s_min + (s_max - s_min) * f
            handles.append(ax.scatter([], [], s=s, facecolor='lightgrey',
                                      edgecolor='k', linewidth=0.4,
                                      label=r'$10^{%d}$' % lm))
        ax.legend(handles=handles, title=r'$M_{\rm BH}\,[M_\odot]$',
                  loc='upper right', labelspacing=1.1, frameon=True,
                  fontsize=8, title_fontsize=8)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # twoDplot-style physical scale bar instead of axis ticks
    if scalebar:
        sb = _nice_length(box_length / 4.0)
        fontprops = fm.FontProperties(size=9)
        bar = AnchoredSizeBar(ax.transData, sb, f"{sb:g} kpc", 'lower left',
                              pad=0.5, color='white', frameon=False,
                              size_vertical=box_length * 0.004,
                              fontproperties=fontprops)
        ax.add_artist(bar)

    ax.set_title(rf"$z={desired_redshift:.2f}$,  "
                 rf"$M_\star={_fmt_mass(Mstar)}\,M_\odot$,  "
                 rf"$N_{{\rm BH}}={n_bh}$")

    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight', dpi=dpi)
    return ax
