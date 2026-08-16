import numpy as np
import matplotlib.pyplot as plt

def final_med_nbh_vs_stellar_at_snap(snapshot):
    stellar_mass = brahma.groupcat.loadSubhalos(Brahma_sim_file, snapshot, fields="SubhaloMassType")[:,4]
    nonzero_indices = np.where(stellar_mass>1e-4)
    with np.errstate(divide='ignore'):
        stellar_mass_log = np.log10(stellar_mass[nonzero_indices])+10
        nbh_nonzero_log = np.log10((brahma.groupcat.loadSubhalos(Brahma_sim_file, snapshot, fields="SubhaloLenType")[:, 5])[nonzero_indices])
    num_halos = stellar_mass_log.size
    num_bins = num_halos//5
    enough_per_bin = False
    while(not enough_per_bin):
        enough_per_bin = True
        bins = np.histogram_bin_edges(stellar_mass_log, num_bins)
        medians = np.empty(num_bins)
        fifths = np.empty(num_bins)
        ninety_fifths = np.empty(num_bins)
        for i in range(num_bins):
            binset = nbh_nonzero_log[np.where(np.logical_and(bins[i]<=stellar_mass_log, stellar_mass_log<=bins[i+1]))]
            if binset.size < 5: 
                enough_per_bin = False
                num_bins -= 1
                break
            else:
                medians[i] = np.median(binset)
                fifths[i] = np.percentile(binset, 5)
                ninety_fifths[i] = np.percentile(binset, 95)
    return (bins[1:]+bins[:-1])/2, medians, fifths, ninety_fifths

def main():
    # first, figure out sim file
    # from the getting started file:
    Brahma_sim_path = '/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/'
    Brahma_sim_name = 'SM5_DFD_3_TNG/'
    Brahma_sim_file = Brahma_sim_path+Brahma_sim_name

    snapshots, redshifts = arepo_package.get_snapshot_redshift_correspondence(Brahma_sim_file)

    snap_choices = np.arange(15, 32, 2)
    num_bins = 10
    for s in snap_choices: 
        snapchoicebins, snapchoicemeds, snapchoicefifths, snapchoiceninetyfifths = final_med_nbh_vs_stellar_at_snap(s)
        plt.plot(snapchoicebins, snapchoicemeds, label=round(redshifts[s], 2))
        plt.fill_between(snapchoicebins, snapchoicefifths, snapchoiceninetyfifths, alpha=0.3)
    plt.legend(loc="upper left", ncols=2, title="Redshift")
    plt.title(r"Median $N_{bh}$ (and $5^{th}$ to $95^{th}$ percentiles) vs Stellar Mass")
    plt.xlabel(r"$\log_{10}(M_{\rm \star} [M_{\odot}])$")
    plt.ylabel(r"$\log_{10}(N_{\rm BH})$")
