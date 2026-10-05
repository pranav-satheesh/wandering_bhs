import numpy as np
import matplotlib.pyplot as plt
import cosmo_sim_tools as cosmosim
from cosmo_sim_tools.arepo_tools import arepo_package
from cosmo_sim_tools import illustris as il
from cosmo_sim_tools import brahma
import argparse

def dist(c1, c2):
    return ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5

def lum_from_mdot(mdot): # mdot in units of 10^10Msun / 0.978Gyr
    eps = 0.2
    c = 300000000 # m/s
    lum = (mdot)*eps*(c**2)/(1-eps) # (10^10Msun/0.978Gyr)*(m/s)^2 = 6.44144*10^30 erg/sec
    return lum * 6.44144e20 # in 10^10 erg / sec

def make_dict(Brahma_sim_file, redshifts, snap, numsubs):
    z = redshifts[snap]
    this_snap_info = brahma.groupcat.loadSubhalos(Brahma_sim_file, snap, fields=["SubhaloCM","SubhaloParent","SubhaloMass", "SubhaloMassType", "SubhaloHalfmassRadType"])
    
    snapdict = {"redshift": z, "numSubhalos": len(this_snap_info["SubhaloCM"])}
    if numsubs == -1: numsubs = snapdict["numSubhalos"]
    for i in range(numsubs):
        if (i%25==0): print(i)
        snapdict[i] = {"id": i}
        
        snapdict[i]['BH_masses'] = arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['BH_Mass'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0]
        mdots = arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['BH_Mdot'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0]
        coords = arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['Coordinates'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0]
    
        snapdict[i]['CM']=this_snap_info["SubhaloCM"][i]
        snapdict[i]['central'] = (this_snap_info["SubhaloParent"][i] == 0)
        snapdict[i]['totalMass'] = this_snap_info["SubhaloMass"][i]
        snapdict[i]['stellarMass'] = this_snap_info["SubhaloMassType"][i][4]
        snapdict[i]['stellarHalfRad'] = this_snap_info["SubhaloHalfmassRadType"][i][4]
        numBH = len(coords)
        snapdict[i]['BH_rad'] = np.empty(numBH)
        snapdict[i]['BH_bolLum'] = np.empty(numBH)
        for j in range(len(coords)):
            snapdict[i]['BH_rad'][j]=dist(snapdict[i]['CM'], coords[j])
            snapdict[i]['BH_bolLum'][j]=lum_from_mdot(mdots[j])
    return snapdict

def add_snap_to_file(Brahma_sim_file, redshifts, snap, numsubs, hdf):
    z = redshifts[snap]
    this_snap_info = brahma.groupcat.loadSubhalos(Brahma_sim_file, snap, fields=["SubhaloCM", "SubhaloParent", "SubhaloMass", "SubhaloMassType", "SubhaloHalfmassRadType"])
    thisSnap = hdf.make_group(str(snap))
    z = redshifts[snap]
    red = thisSnap.make_database("redshift", data=[z])
    num = thisSnap.make_database("numSubhalos", data=[len(this_snap_info["SubhaloCM"]])
    if numsubs == -1: numsubs = thisSnap["numSubhalos"][0]
    for i in range(numsubs):
        if (i%25 == 0): print(i)
        thisSub = thisSnap.make_group(str(i))
        bhm = thisSub.make_dataset("BH_masses", data=arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['BH_Mass'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0])
        mdots = arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['BH_Mdot'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0]
        coords = arepo_package.get_particle_property_within_postprocessed_groups(output_path=Brahma_sim_file,particle_property=['Coordinates'],p_type=5,desired_redshift=z,subhalo_index=i,group_type='subhalo',store_all_offsets=0)[0]
        cm = this_snap_info["SubhaloCM"][i]
        thisSub.make_dataset("CM", data=cm)
        thisSub.make_dataset("central", data=[this_snap_info["SubhaloParent"]==0])
        thisSub.make_dataset("totalMass", data=[this_snap_info["SubhaloMass"][i]])
        thisSub.make_dataset("stellarMass", data=[this_snap_info["SubhaloMassType"][i][4]])
        thisSub.make_dataset("stellarHalfRad", data=[this_snap_info["SubhaloHalfmassRadType"][i][4]])
        numBH = len(coords)
        radarr = np.empty(numBH)
        lumarr = np.empty(numBH)
        for j in range(len(coords)):
            radarr[j] = dist(cm, coords[j])
            lumarr[j] = lum_from_mdot(mdots[j])
        thisSub.make_dataset("BH_rad", data=radarr)
        thisSub.make_dataset("BH_bollum", data=lumarr)

def main(): # loop through snaps, call make_dict, save the dictionaries somewhere
    # first, figure out sim file
    p = argparse.ArgumentParser(description="Dictionary of subhalo information for a snapshot. Gives number of subhalos and redshift for the snap, and a list of dictionaries which each hold the following information for a specific subhalo: black hole masses [10^10 solar mass/h], distances from center [ckpc/h], and bolometric luminosity [10^10 erg/sec]")
    p.add_argument("--simName", default='SM5_DFD_3_TNG/')
    p.add_argument("--numSnaps", default=-1)
    p.add_argument("--numSubhalos", default=-1)
    args = p.parse_args()
    
    Brahma_sim_path = '/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/'
    Brahma_sim_name = args.simName
    Brahma_sim_file = Brahma_sim_path+Brahma_sim_name
    snapshots, redshifts = arepo_package.get_snapshot_redshift_correspondence(Brahma_sim_file)

    if args.numSnaps == -1: args.numSnaps = len(snapshots)

    f = h5py.File("subhalos"+args.simName[:-1]+".h5", 'w')
    for i in range(args.numSnaps):
        add_snap_to_file(Brahma_sim_file, redshifts, i, args.numSubhalos, f)

if __name__ == '__main__':
    main()
