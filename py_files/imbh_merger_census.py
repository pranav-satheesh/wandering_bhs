"""Census of IMBH-IMBH, IMRI and EMRI mergers in the SM5_DFD_3_TNG box, in the style of ASTRID.

Classes are set by the progenitor masses in blackhole_mergers.hdf5:
    IMBH-IMBH  both BHs below IMBH_MAX_MSUN
    IMRI       one IMBH, the other between IMBH_MAX_MSUN and SMBH_MIN_MSUN
    EMRI       one IMBH, the other above SMBH_MIN_MSUN
Dwarf-galaxy mergers are those whose host has a stellar mass below DWARF_MAX_MSTAR_MSUN.
The host is the subhalo holding the BHs at the last snapshot before the merger,
or the surviving BH's subhalo at the first snapshot after when the catalogue
does not hold the BHs before (seeded inside the snapshot gap).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append(str(Path(__file__).parent))
import plot_style as ps

SIM_NAME = 'SM5_DFD_3_TNG'
SIM_FILE = ('/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/'
            f'L12p5n512/AREPO/{SIM_NAME}/')
DATA_DIR = Path('/orange/lblecha/pranavsatheesh/wandering_bhs/Data')
FIGURE_DIR = Path(__file__).parent.parent / 'figures'

SEED_MASS_CODE = 1e-5
IMBH_MAX_MSUN = 1e6
SMBH_MIN_MSUN = 1e8
DWARF_MAX_MSTAR_MSUN = 1e9
EXCLUDE_SEED_SEED = False

REDSHIFT_EDGES = [0, 1, 2, 3, 5, 20]
LOG_Q_BINS = np.linspace(-4, 0, 21)
ID_STRIDE = 10**12

catalogue = h5py.File(DATA_DIR / f'wandering_bh_{SIM_NAME}.hdf5', 'r')
hubble_param = float(catalogue.attrs['HubbleParam'])
box_volume_cmpc3 = (float(catalogue.attrs['BoxSize_ckpc_over_h']) / 1e3 / hubble_param)**3
snapshots = catalogue['snapshot/snap'][:].astype(np.int64)
snapshot_scale_factor = 1 / (1 + catalogue['snapshot/redshift'][:].astype(np.float64))

bh_snap = catalogue['bh/snap'][:].astype(np.int64)
bh_subhalo_row = catalogue['bh/subhalo_row'][:]
subhalo_stellar_mass = catalogue['subhalo/stellar_mass_msun'][:]
subhalo_is_central = catalogue['subhalo/is_central'][:]
catalogue_key = bh_snap * ID_STRIDE + np.load(DATA_DIR / f'wandering_bh_ids_{SIM_NAME}.npy')
key_order = np.argsort(catalogue_key)
sorted_catalogue_key = catalogue_key[key_order]

with h5py.File(SIM_FILE + 'blackhole_mergers.hdf5', 'r') as merger_file:
    mergers = {name: merger_file[name][:]
               for name in ['ScaleFactor', 'BH_ID1', 'BH_ID2', 'BH_Mass1', 'BH_Mass2']}
_, unique_rows = np.unique(
    np.column_stack([mergers['ScaleFactor'], mergers['BH_ID1'], mergers['BH_ID2']]),
    axis=0, return_index=True)
unique_rows = unique_rows[np.argsort(mergers['ScaleFactor'][unique_rows], kind='stable')]
mergers = {name: value[unique_rows] for name, value in mergers.items()}

merger_redshift = 1 / mergers['ScaleFactor'] - 1
mass1_msun = mergers['BH_Mass1'] * 1e10 / hubble_param
mass2_msun = mergers['BH_Mass2'] * 1e10 / hubble_param
light_mass_msun = np.minimum(mass1_msun, mass2_msun)
heavy_mass_msun = np.maximum(mass1_msun, mass2_msun)
mass_ratio = light_mass_msun / heavy_mass_msun
is_seed_seed = (mergers['BH_Mass1'] < 1.01 * SEED_MASS_CODE) & (mergers['BH_Mass2'] < 1.01 * SEED_MASS_CODE)

gap = np.searchsorted(snapshot_scale_factor, mergers['ScaleFactor'], side='right') - 1


def catalogue_row(snapshot, bh_id):
    """Row in /bh of `bh_id` at `snapshot`, or -1 when the catalogue does not hold it."""
    query = snapshot.astype(np.int64) * ID_STRIDE + bh_id
    position = np.clip(np.searchsorted(sorted_catalogue_key, query), 0, len(key_order) - 1)
    return np.where(sorted_catalogue_key[position] == query, key_order[position], -1)


host_row = np.full(len(unique_rows), -1)
for snapshot_of_host, bh_id in [(snapshots[gap], mergers['BH_ID1']),
                                (snapshots[gap], mergers['BH_ID2']),
                                (snapshots[gap + 1], mergers['BH_ID1'])]:
    row = catalogue_row(snapshot_of_host, bh_id)
    host_row = np.where((host_row < 0) & (row >= 0), bh_subhalo_row[np.maximum(row, 0)], host_row)

has_host = host_row >= 0
host_mstar_msun = np.where(has_host, subhalo_stellar_mass[np.maximum(host_row, 0)], np.nan)
host_is_satellite = has_host & ~subhalo_is_central[np.maximum(host_row, 0)]
in_dwarf = has_host & (host_mstar_msun < DWARF_MAX_MSTAR_MSUN)

bh_is_wandering = catalogue['bh/is_wandering'][:]
row1 = catalogue_row(snapshots[gap], mergers['BH_ID1'])
row2 = catalogue_row(snapshots[gap], mergers['BH_ID2'])
wandering1 = (row1 >= 0) & bh_is_wandering[np.maximum(row1, 0)]
wandering2 = (row2 >= 0) & bh_is_wandering[np.maximum(row2, 0)]
one_or_both_wandering = wandering1 | wandering2
both_wandering = wandering1 & wandering2

light_is_imbh =light_mass_msun < IMBH_MAX_MSUN
imbh_imbh = light_is_imbh & (heavy_mass_msun < IMBH_MAX_MSUN)
imri = light_is_imbh & (heavy_mass_msun >= IMBH_MAX_MSUN) & (heavy_mass_msun < SMBH_MIN_MSUN)
emri = light_is_imbh & (heavy_mass_msun >= SMBH_MIN_MSUN)

kept = ~is_seed_seed if EXCLUDE_SEED_SEED else np.ones(len(unique_rows), dtype=bool)
seed_seed_note = 'seed-seed excluded' if EXCLUDE_SEED_SEED else 'seed-seed included'

print(f'{len(unique_rows):,} distinct mergers, {is_seed_seed.sum():,} seed-seed ({seed_seed_note}); '
      f'host found for {has_host.mean():.1%}; box {box_volume_cmpc3:.0f} cMpc^3')
print(f'IMBH < {IMBH_MAX_MSUN:.0e} Msun, SMBH > {SMBH_MIN_MSUN:.0e} Msun, '
      f'dwarf: M* < {DWARF_MAX_MSTAR_MSUN:.0e} Msun\n')

redshift_ranges = list(zip(REDSHIFT_EDGES[:-1], REDSHIFT_EDGES[1:]))
classes = {
    'all mergers': kept,
    'IMBH-IMBH': kept & imbh_imbh,
    'IMBH-IMBH, dwarf host': kept & imbh_imbh & in_dwarf,
    'IMBH-IMBH, dwarf satellite': kept & imbh_imbh & in_dwarf & host_is_satellite,
    'IMBH-IMBH, host unknown': kept & imbh_imbh & ~has_host,
    'IMRI': kept & imri,
    'IMRI, dwarf host': kept & imri & in_dwarf,
    'EMRI': kept & emri,
}
header = f'{"":28}' + ''.join(f'{f"{low}<z<{high}":>11}' for low, high in redshift_ranges) + f'{"all z":>9}'
print(header)
for name, selection in classes.items():
    counts = [np.sum(selection & (merger_redshift >= low) & (merger_redshift < high))
              for low, high in redshift_ranges]
    print(f'{name:28}' + ''.join(f'{count:11d}' for count in counts) + f'{selection.sum():9d}')

print('\nmergers with a wandering progenitor at snap_before (any / both):')
print(header)
for name in ['all mergers', 'IMBH-IMBH', 'IMBH-IMBH, dwarf host', 'IMRI', 'EMRI']:
    for label, wandering in [('any', one_or_both_wandering), ('both', both_wandering)]:
        selection = classes[name] & wandering
        counts = [np.sum(selection & (merger_redshift >= low) & (merger_redshift < high))
                  for low, high in redshift_ranges]
        print(f'{f"{name} [{label}]":28}' + ''.join(f'{count:11d}' for count in counts) + f'{selection.sum():9d}')

print('\nz = 2-3 per cMpc^3 (for comparison with ASTRID):')
at_cosmic_noon = (merger_redshift >= 2) & (merger_redshift < 3)
for name in ['IMBH-IMBH', 'IMBH-IMBH, dwarf host', 'IMRI', 'EMRI']:
    count = np.sum(classes[name] & at_cosmic_noon)
    print(f'  {name:24} {count:5d}  ->  {count / box_volume_cmpc3:.2e} cMpc^-3')
print(f'smallest q among IMRIs {mass_ratio[classes["IMRI"]].min():.1e}; '
      f'EMRIs {mass_ratio[classes["EMRI"]].min() if emri.any() else float("nan"):.1e}')

ps.set_style('paper', font_scale=1.15)
panels = [('All BH mergers', kept),
          (f'IMBH-IMBH in dwarfs ($M_\\star<10^{{{np.log10(DWARF_MAX_MSTAR_MSUN):.0f}}}\\,M_\\odot$)',
           classes['IMBH-IMBH, dwarf host']),
          ('IMRIs (IMBH + MBH)', classes['IMRI'])]
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharey=True)
colors = plt.cm.viridis_r(np.linspace(0.05, 0.9, len(redshift_ranges)))
for ax, (title, selection) in zip(axes, panels):
    for color, (low, high) in zip(colors, redshift_ranges):
        in_range = selection & (merger_redshift >= low) & (merger_redshift < high)
        ax.hist(np.log10(mass_ratio[in_range]), bins=LOG_Q_BINS, histtype='step', color=color,
                linewidth=2, label=f'{low} < z < {high} ({in_range.sum()})')
    ax.set(xlabel=r'$\log_{10}\, q = \log_{10}(M_2/M_1)$', yscale='log', title=title)
    ax.legend(frameon=False, fontsize='small', loc='upper left')
axes[0].set_ylabel('Number of mergers')
fig.tight_layout()
fig.savefig(FIGURE_DIR / f'imbh_merger_mass_ratio_{SIM_NAME}.pdf')
catalogue.close()
