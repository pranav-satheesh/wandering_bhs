import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import h5py
import time
import illustris_python_mod as il
import sys
import os
sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo
from scipy.spatial import cKDTree
import BRAHMA_python as il_brahma
from tqdm import tqdm
import datetime


class MergerInfo:
    
    def __init__(self, *args, **kwargs): 
        """ Initialize class instance for extracting merger info from SubLink tree. """
        self.full_tree = args[0]
        self.basepath = args[1]

        self.minMassRatio = kwargs.get('minMassRatio',1.0e-10)
        tree_index = kwargs.get('index', 0)
        getFullTree = kwargs.get('getFullTree', True)
        self.minNdm = kwargs.get('minNdm', 0)
        self.minNgas = kwargs.get('minNgas', 0)
        self.minNstar = kwargs.get('minNstar', 0)
        self.minNbh = kwargs.get('minNbh', 0)
        verbose = kwargs.get('verbose', False)

        self.count = 0
        self.progMassRatio = np.array([])
        self.fpSnap = np.array([]).astype('int64')
        self.npSnap = np.array([]).astype('int64')
        self.fpMass = np.array([])
        self.npMass = np.array([])
        self.descMass = np.array([])
        self.descSnap = np.array([]).astype('int64')
        self.fpSubhaloID = np.array([]).astype('int64')
        self.fpSubfindID = np.array([]).astype('int64')
        #self.fpSubhaloMassType = np.array([]).reshape(0,6)
        self.fpNpart = np.array([]).astype('int64').reshape(0,6)
        self.npSubhaloID = np.array([]).astype('int64')
        self.npSubfindID = np.array([]).astype('int64')
        #self.npSubhaloMassType = np.array([]).reshape(0,6)
        self.npNpart = np.array([]).astype('int64').reshape(0,6)
        self.descSubhaloID = np.array([]).astype('int64')
        self.descSubfindID = np.array([]).astype('int64')
        #self.descMassTyp = np.array([]).reshape(0,6)
        self.descNpart = np.array([]).astype('int64').reshape(0,6)
        
        # New fields for Pop_Illustris compatibility
        self.fpBHMass = np.array([])
        self.npBHMass = np.array([])
        self.time = np.array([]).reshape(0, 3)
        self.fpHalfRadType = np.array([]).reshape(0, 6)
        self.npHalfRadType = np.array([]).reshape(0, 6)
        self.descHalfRadType = np.array([]).reshape(0, 6)
        self.fpMassInRadType = np.array([]).reshape(0, 6)
        self.npMassInRadType = np.array([]).reshape(0, 6)
        self.descMassInRadType = np.array([]).reshape(0, 6)
        self.fpVelDisp = np.array([])
        self.npVelDisp = np.array([])
        self.descVelDisp = np.array([])


        #self.countDupDesc = self.countMrgFromDupDesc(tree)
        self.getMergerInfoSubtree(tree_index, minMassRatio=self.minMassRatio, getFullTree=getFullTree, minNdm=self.minNdm, 
                                  minNgas=self.minNgas, minNstar=self.minNstar, minNbh=self.minNbh,verbose=verbose)
        
    def getMergerInfoSubtree(self,tree_index,minMassRatio=1e-10,getFullTree=True,minNdm=0,minNgas=0,minNstar=100,minNbh=0,verbose=False):

        
        invMinMassratio = 1/minMassRatio
        rootID = self.full_tree['SubhaloID'][tree_index]
        rootSubfindID = self.full_tree['SubfindID'][tree_index]
        rootSnap = self.full_tree['SnapNum'][tree_index]
        fpID = self.full_tree['FirstProgenitorID'][tree_index]
        fpSnap = rootSnap
        fdesID = self.full_tree['DescendantID'][tree_index]

        while fpID!=-1:
            fpIndex = tree_index+(fpID-rootID)
            fpMass = self.get_maxPastMass(fpIndex)
            fpNpart = self.get_subhalolentype(fpIndex)


            #explore breadth
            npID = self.full_tree['NextProgenitorID'][fpIndex]
            npSnap = self.full_tree['SnapNum'][fpIndex]
            ndesID = self.full_tree['DescendantID'][fpIndex]

            fdesIndex = tree_index+(fdesID-rootID)

            while npID!=-1:
                npIndex = tree_index+(npID-rootID)
                npMass = self.get_maxPastMass(npIndex)
                npNpart = self.get_subhalolentype(npIndex)

                ndesIndex = tree_index+(ndesID-rootID)
                
                if fpMass>0 and npMass>0:
                    massratio = npMass/fpMass
                    if (massratio>=minMassRatio and massratio<=invMinMassratio and
                        min(npNpart[0], fpNpart[0]) >= self.minNgas and 
                        min(npNpart[1], fpNpart[1]) >= self.minNdm and
                        min(npNpart[4], fpNpart[4]) >= self.minNstar and
                        min(npNpart[5], fpNpart[5]) >= self.minNbh):

                        #print("massratio: %2.3f"%(massratio))
                        self.count+=1

                        if fpSnap == (npSnap+2):
                            if verbose: print(f"NOTE: SubLink skipped snap {npSnap+1} in finding descendant.")
                        elif fpSnap != (npSnap+1) and fpSnap != (npSnap+2):
                            raise Exception(f'ERROR: snaps not contiguous b/t prog ({npSnap}) & desc ({fpSnap}).')
                        
                        self.fpSnap = np.append(self.fpSnap, self.full_tree['SnapNum'][fpIndex])
                        self.npSnap = np.append(self.npSnap, self.full_tree['SnapNum'][npIndex])
                        self.descSnap = np.append(self.descSnap, self.full_tree['SnapNum'][ndesIndex])
                        # first progenitor subfind ID and masses
                        self.fpSubhaloID = np.append(self.fpSubhaloID, self.full_tree['SubhaloID'][fpIndex])
                        self.fpSubfindID = np.append(self.fpSubfindID, self.full_tree['SubfindID'][fpIndex])
                        self.fpMass = np.append(self.fpMass, self.full_tree['Mass'][fpIndex])
                        self.fpNpart = np.vstack((self.fpNpart, fpNpart))
                        # next progenitor subfind ID and masses
                        self.npSubhaloID = np.append(self.npSubhaloID, self.full_tree['SubhaloID'][npIndex])
                        self.npSubfindID = np.append(self.npSubfindID, self.full_tree['SubfindID'][npIndex])
                        self.npMass = np.append(self.npMass, self.full_tree['Mass'][npIndex])
                        self.npNpart = np.vstack((self.npNpart, npNpart))
           
                        # mass ratio (defined to be nextProg / firstProg *at max past mass*)
                        self.progMassRatio = np.append(self.progMassRatio, massratio)
                        
                        # self.infallMassRatio = np.append(self.infallMassRatio, npinfallMass / fpinfallMass)
                        # self.fpMass_mod = np.append(self.fpMass_mod, self.full_tree['Mass'][fpIndex])
                        # self.npMass_mod = np.append(self.npMass_mod, self.full_tree['Mass'][npIndex])
            

                        # descendant subfind ID and masses
                        self.descSubhaloID = np.append(self.descSubhaloID, self.full_tree['SubhaloID'][ndesIndex])
                        self.descSubfindID = np.append(self.descSubfindID, self.full_tree['SubfindID'][ndesIndex])
                        self.descMass = np.append(self.descMass, self.full_tree['Mass'][ndesIndex])
                        descNpart = self.get_subhalolentype(ndesIndex)
                        self.descNpart = np.vstack((self.descNpart, descNpart))
                        
                        # Load additional properties for Pop_Illustris compatibility
                        fpBH, fpHrad, fpMrad, fpVd = self.get_subhalo_properties(fpIndex)
                        npBH, npHrad, npMrad, npVd = self.get_subhalo_properties(npIndex)
                        descBH, descHrad, descMrad, descVd = self.get_subhalo_properties(ndesIndex)
                        
                        self.fpBHMass = np.append(self.fpBHMass, fpBH)
                        self.npBHMass = np.append(self.npBHMass, npBH)
                        self.fpHalfRadType = np.vstack((self.fpHalfRadType, fpHrad))
                        self.npHalfRadType = np.vstack((self.npHalfRadType, npHrad))
                        self.descHalfRadType = np.vstack((self.descHalfRadType, descHrad))
                        self.fpMassInRadType = np.vstack((self.fpMassInRadType, fpMrad))
                        self.npMassInRadType = np.vstack((self.npMassInRadType, npMrad))
                        self.descMassInRadType = np.vstack((self.descMassInRadType, descMrad))
                        self.fpVelDisp = np.append(self.fpVelDisp, fpVd)
                        self.npVelDisp = np.append(self.npVelDisp, npVd)
                        self.descVelDisp = np.append(self.descVelDisp, descVd)

                #go to next progenitor
                npID = self.full_tree['NextProgenitorID'][npIndex]
                npSnap = self.full_tree['SnapNum'][npIndex]
                ndescID = self.full_tree['DescendantID'][npIndex]

                if getFullTree and self.full_tree['FirstProgenitorID'][npIndex]!=-1:
                    if verbose: print(f"tracing subtree...")
                    self.getMergerInfoSubtree(npIndex)


            fpID = self.full_tree['FirstProgenitorID'][fpIndex]
            fpSnap = self.full_tree['SnapNum'][fpIndex]
            fdesID = self.full_tree['DescendantID'][fpIndex]
        #print(fpID,fpIndex,fpSnap)

        return self     

    def get_maxPastMass(self,tree_index):
        
        mleaf_idx = tree_index + self.full_tree['MainLeafProgenitorID'][tree_index] - self.full_tree['SubhaloID'][tree_index]

        if mleaf_idx==tree_index:
            return self.full_tree['Mass'][tree_index]
        else:
            mass_history = self.full_tree['Mass'][tree_index:mleaf_idx]
            max_past_mass = np.max(mass_history)
            return max_past_mass
        
    def get_subhalolentype(self,index):
        
        snap_num = self.full_tree['SnapNum'][index]
        subhalo_index = self.full_tree['SubfindID'][index]
        subhalolentypes = il_brahma.groupcat.loadSubhalos_postprocessed(self.basepath,snap_num,fields=['SubhaloLenType'])

        return subhalolentypes[subhalo_index]
    
    def get_subhalo_properties(self, index):
        """Load BH mass, half-mass radii, mass in radius, and velocity dispersion for a subhalo."""
        snap_num = self.full_tree['SnapNum'][index]
        subhalo_index = self.full_tree['SubfindID'][index]
        
        fields = ['SubhaloBHMass', 'SubhaloHalfmassRadType', 'SubhaloMassInRadType', 'SubhaloVelDisp']
        subhalo_data = il_brahma.groupcat.loadSubhalos_postprocessed(self.basepath, snap_num, fields=fields)
        
        bh_mass = subhalo_data['SubhaloBHMass'][subhalo_index]
        half_rad_type = subhalo_data['SubhaloHalfmassRadType'][subhalo_index]
        mass_in_rad_type = subhalo_data['SubhaloMassInRadType'][subhalo_index]
        vel_disp = subhalo_data['SubhaloVelDisp'][subhalo_index]
        
        return bh_mass, half_rad_type, mass_in_rad_type, vel_disp


def WriteMergerFile(savePath,basePath, snapNum, minNdm=0, minNgas=0, minNstar=100, minNbh=0, verbose=False,minMassRatio=1e-10):
    
    simName = basePath.split('/')[-2]
    brahma_snapshots,brahma_redshifts = arepo.get_snapshot_redshift_correspondence(basePath)
    sub_lentype = il_brahma.groupcat.loadSubhalos_postprocessed(basePath,snapNum=snapNum,fields=['SubhaloLenType'])

    sub_hdr = il_brahma.groupcat.loadHeader(basePath,snapNum)
    nsubs = sub_hdr['Nsubgroups_Total']

    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    idx = np.where((Ngas >= minNgas) & (Ndm >= minNdm) & (Nstar >= minNstar) & (Nbh >= minNbh))[0]
    nselect = idx.size
    print(nselect)
    
    print(f"{nselect} subhalos meet criteria: {minNdm=}, {minNgas=},"
          +f"{minNstar=}, {minNbh=}")
    if nselect==0:
        return None
    ncheck = 10**np.int64(np.log10(nselect)-1)
    print(f"Total number of subhalos in snap {snapNum}: {nsubs}. {ncheck=}.")
    sys.stdout.flush()

    outfilename = f"galaxy-mergers_brahma_{simName}_gas-{minNgas:03d}_dm" \
                  f"-{minNdm:03d}_star-{minNstar:03d}_bh-{minNbh:03d}.hdf5"
    
    with h5py.File(f"{savePath}/{outfilename}",'w') as mf:
        now = datetime.datetime.now()
        mf.attrs['created'] = str(now)
        mf.attrs['hubbleParam'] = sub_hdr['HubbleParam']
        mf.attrs['Omega0'] = sub_hdr['Omega0']
        mf.attrs['OmegaLambda'] = sub_hdr['OmegaLambda']
        mf.attrs['box_volume_mpc'] = (sub_hdr['BoxSize']/1000.0 / sub_hdr['HubbleParam'])**3 
        mf.attrs['min_parts'] = np.array([minNgas, minNdm, minNstar, minNbh])
        mf.attrs['part_names'] = ['gas', 'dm', 'star', 'bh']
        mf.attrs['part_types'] = np.array([0,1,4,5])
        mf.attrs['snapshots'] = brahma_snapshots
        mf.attrs['redshifts'] = brahma_redshifts
        print(f"Finished writing initial header data to file.\n")
        sys.stdout.flush()

        # initialize arrays for output data
    
    allMrgSubhID = np.array([]).astype('int64').reshape(0,3)
    allMrgSubfID = np.array([]).astype('int64').reshape(0,3)
    allMrgSnaps = np.array([]).astype('int64').reshape(0,3)
    #allMrgTimes = np.array([]).reshape(0,3)
    allMrgProgMassRatio = np.array([])
    allMrgfpMass = np.array([])
    allMrgnpMass = np.array([])
    allMrgSubhLenType = np.array([]).astype('int64').reshape(0,6,3)

    #open the brahma tree file
    tree=h5py.File(basePath+'postprocessing/tree_extended.hdf5','r')
    full_tree = {}
    for key in tree.keys():
        full_tree[key] = tree.get(key)[:]

    total_mrg_count = 0
    # loop over all subhalos meeting length criteria

    for k,isub in enumerate(idx):
        if k%ncheck == 0 or verbose: 
            print(f"processing sub {isub} ({k} of {nselect} meeting criteria)...")
            sys.stdout.flush()
        
        if (sub_lentype[isub,0] < minNgas or sub_lentype[isub,1] < minNdm
            or sub_lentype[isub,4] < minNstar or sub_lentype[isub,5] < minNbh):
            err = f"Error! subhalo {isub} does not meet length criteria."
            raise ValueError(err)

        if verbose: 
            print(verbose)
            print(f"Ngas={sub_lentype[isub,0]}, Ndm={sub_lentype[isub,1]},"
                  f"Nstar={sub_lentype[isub,4]}, Nbh={sub_lentype[isub,5]}")
            
        tree_index = np.where((full_tree['SnapNum']==snapNum)&(full_tree['SubfindID']==isub))[0]

        if tree_index.size==0:
            continue
        else: 
            tree_index = tree_index[0]

        mrg = MergerInfo(full_tree, basePath, index=tree_index, verbose=verbose, minNgas=minNgas, 
                         minNdm=minNdm, minNstar=minNstar, minNbh=minNbh,minMassRatio=minMassRatio)

        if verbose==True: print("count:", mrg.count)
        if mrg.count >0:
            
            # if k%ncheck == 0 or verbose:
            #     print(f" # mergers from dup DescendantIDs: {mrg.countDupDesc}, from MergerInfo class: {mrg.count}")
            #     sys.stdout.flush()
            total_mrg_count += mrg.count
            allMrgSubhID = np.vstack( (allMrgSubhID, np.array([mrg.fpSubhaloID, mrg.npSubhaloID, 
                                                               mrg.descSubhaloID]).T) )
            allMrgSubfID = np.vstack( (allMrgSubfID, np.array([mrg.fpSubfindID, mrg.npSubfindID, 
                                                               mrg.descSubfindID]).T) )
            allMrgSnaps = np.vstack( (allMrgSnaps, np.array([mrg.fpSnap, mrg.npSnap, mrg.descSnap]).T) )
            allMrgProgMassRatio = np.append( allMrgProgMassRatio, mrg.progMassRatio )
            allMrgfpMass = np.append( allMrgfpMass, mrg.fpMass )
            allMrgnpMass = np.append( allMrgnpMass, mrg.npMass )
            
            #loop through mergers in the subhalo
            for j in range(mrg.count):
                allMrgSubhLenType = np.vstack( (allMrgSubhLenType, 
                                                np.vstack([mrg.fpNpart[j,:], 
                                                           mrg.npNpart[j,:], 
                                                           mrg.descNpart[j,:]]
                                                         ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE
                
    print(f"allMrgSubhID shape: {allMrgSubhID.shape}")
    sys.stdout.flush()

    with h5py.File(f"{savePath}/{outfilename}",'a') as mf:

        mf.attrs['num_mergers'] = total_mrg_count
        mf.attrs['merger_components_in_data_arrays'] = '(FirstProg, NextProg, Descendant)'

        mf.create_dataset('SubhaloLenType', data=allMrgSubhLenType)
        mf['SubhaloLenType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloLenType'].attrs['units'] = 'none'

        mf.create_dataset('shids_tree', data=allMrgSubhID)
        mf['shids_tree'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_tree'].attrs['units'] = 'none'

        mf.create_dataset('shids_subf', data=allMrgSubfID)
        mf['shids_subf'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_subf'].attrs['units'] = 'none'

        mf.create_dataset('snaps', data=allMrgSnaps)
        mf['snaps'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['snaps'].attrs['units'] = 'none'

        mf.create_dataset('ProgMassRatio', data=allMrgProgMassRatio)
        mf['ProgMassRatio'].attrs['dataShape'] = 'Nmrg'
        mf['ProgMassRatio'].attrs['units'] = 'none'

        mf.create_dataset('fpMass', data=allMrgfpMass * 1.0e10 / sub_hdr['HubbleParam']) #: [Msun]
        mf['fpMass'].attrs['dataShape'] = 'Nmrg'
        mf['fpMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('npMass', data=allMrgnpMass * 1.0e10 / sub_hdr['HubbleParam']) #: [Msun]
        mf['npMass'].attrs['dataShape'] = 'Nmrg'
        mf['npMass'].attrs['units'] = '[Msun]'

    print(f"Finished processing merger trees for {nsubs} subhalos in snap {snapNum}.")
    print(f"Found {total_mrg_count} mergers.")   


def WriteMergerFileforevolution(savePath,basePath, snapNum, minNdm=0, minNgas=0, minNstar=100, minNbh=0, verbose=False,minMassRatio=1e-10):
    
    simName = basePath.split('/')[-2]
    brahma_snapshots,brahma_redshifts = arepo.get_snapshot_redshift_correspondence(basePath)
    sub_lentype = il_brahma.groupcat.loadSubhalos_postprocessed(basePath,snapNum=snapNum,fields=['SubhaloLenType'])

    sub_hdr = il_brahma.groupcat.loadHeader(basePath,snapNum)
    nsubs = sub_hdr['Nsubgroups_Total']

    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    idx = np.where((Ngas >= minNgas) & (Ndm >= minNdm) & (Nstar >= minNstar) & (Nbh >= minNbh))[0]
    nselect = idx.size
    print(nselect)
    
    print(f"{nselect} subhalos meet criteria: {minNdm=}, {minNgas=},"
          +f"{minNstar=}, {minNbh=}")
    if nselect==0:
        return None
    ncheck = 10**np.int64(np.log10(nselect)-1)
    print(f"Total number of subhalos in snap {snapNum}: {nsubs}. {ncheck=}.")
    sys.stdout.flush()

    outfilename = f"galaxy-mergers_brahma_{simName}_gas-{minNgas:03d}_dm" \
                  f"-{minNdm:03d}_star-{minNstar:03d}_bh-{minNbh:03d}_evolution.hdf5"
    
    with h5py.File(f"{savePath}/{outfilename}",'w') as mf:
        now = datetime.datetime.now()
        mf.attrs['created'] = str(now)
        mf.attrs['hubbleParam'] = sub_hdr['HubbleParam']
        mf.attrs['Omega0'] = sub_hdr['Omega0']
        mf.attrs['OmegaLambda'] = sub_hdr['OmegaLambda']
        mf.attrs['box_volume_mpc'] = (sub_hdr['BoxSize']/1000.0 / sub_hdr['HubbleParam'])**3 
        mf.attrs['min_parts'] = np.array([minNgas, minNdm, minNstar, minNbh])
        mf.attrs['part_names'] = ['gas', 'dm', 'star', 'bh']
        mf.attrs['part_types'] = np.array([0,1,4,5])
        mf.attrs['snapshots'] = brahma_snapshots
        mf.attrs['redshifts'] = brahma_redshifts
        print(f"Finished writing initial header data to file.\n")
        sys.stdout.flush()

        # initialize arrays for output data
    
    allMrgSubhID = np.array([]).astype('int64').reshape(0,3)
    allMrgSubfID = np.array([]).astype('int64').reshape(0,3)
    allMrgSnaps = np.array([]).astype('int64').reshape(0,3)
    #allMrgTimes = np.array([]).reshape(0,3)
    allMrgProgMassRatio = np.array([])
    allMrgfpMass = np.array([])
    allMrgnpMass = np.array([])
    allMrgSubhLenType = np.array([]).astype('int64').reshape(0,6,3) ### CHANGED ARRAY SHAPE
    
    # New arrays for Pop_Illustris (evolution version)
    allMrgBHMass = np.array([]).reshape(0,2)
    allMrgHalfRadType = np.array([]).reshape(0,6,3)
    allMrgMassInRadType = np.array([]).reshape(0,6,3)
    allMrgVelDisp = np.array([]).reshape(0,3)

    #open the brahma tree file
    tree=h5py.File(basePath+'postprocessing/tree_extended.hdf5','r')
    full_tree = {}
    for key in tree.keys():
        full_tree[key] = tree.get(key)[:]

    total_mrg_count = 0
    # loop over all subhalos meeting length criteria

    for k,isub in enumerate(idx):
        if k%ncheck == 0 or verbose: 
            print(f"processing sub {isub} ({k} of {nselect} meeting criteria)...")
            sys.stdout.flush()
        
        if (sub_lentype[isub,0] < minNgas or sub_lentype[isub,1] < minNdm
            or sub_lentype[isub,4] < minNstar or sub_lentype[isub,5] < minNbh):
            err = f"Error! subhalo {isub} does not meet length criteria."
            raise ValueError(err)

        if verbose: 
            print(verbose)
            print(f"Ngas={sub_lentype[isub,0]}, Ndm={sub_lentype[isub,1]},"
                  f"Nstar={sub_lentype[isub,4]}, Nbh={sub_lentype[isub,5]}")
            
        tree_index = np.where((full_tree['SnapNum']==snapNum)&(full_tree['SubfindID']==isub))[0]

        if tree_index.size==0:
            continue
        else: 
            tree_index = tree_index[0]

        mrg = MergerInfo(full_tree, basePath, index=tree_index, verbose=verbose, minNgas=minNgas, 
                         minNdm=minNdm, minNstar=minNstar, minNbh=minNbh,minMassRatio=minMassRatio)

        if verbose==True: print("count:", mrg.count)
        if mrg.count >0:
            
            # if k%ncheck == 0 or verbose:
            #     print(f" # mergers from dup DescendantIDs: {mrg.countDupDesc}, from MergerInfo class: {mrg.count}")
            #     sys.stdout.flush()
            total_mrg_count += mrg.count
            allMrgSubhID = np.vstack( (allMrgSubhID, np.array([mrg.fpSubhaloID, mrg.npSubhaloID, 
                                                               mrg.descSubhaloID]).T) )
            allMrgSubfID = np.vstack( (allMrgSubfID, np.array([mrg.fpSubfindID, mrg.npSubfindID, 
                                                               mrg.descSubfindID]).T) )
            allMrgSnaps = np.vstack( (allMrgSnaps, np.array([mrg.fpSnap, mrg.npSnap, mrg.descSnap]).T) )
            allMrgProgMassRatio = np.append( allMrgProgMassRatio, mrg.progMassRatio )
            allMrgfpMass = np.append( allMrgfpMass, mrg.fpMass )
            allMrgnpMass = np.append( allMrgnpMass, mrg.npMass )
            
            # New fields for Pop_Illustris (evolution version)
            allMrgBHMass = np.vstack( (allMrgBHMass, np.array([mrg.fpBHMass, mrg.npBHMass]).T) )
            allMrgHalfRadType = np.vstack( (allMrgHalfRadType, np.array([mrg.fpHalfRadType, mrg.npHalfRadType, 
                                                                          mrg.descHalfRadType]).T.reshape(mrg.count, 6, 3)) )
            allMrgMassInRadType = np.vstack( (allMrgMassInRadType, np.array([mrg.fpMassInRadType, mrg.npMassInRadType, 
                                                                              mrg.descMassInRadType]).T.reshape(mrg.count, 6, 3)) )
            allMrgVelDisp = np.vstack( (allMrgVelDisp, np.array([mrg.fpVelDisp, mrg.npVelDisp, 
                                                                  mrg.descVelDisp]).T.reshape(mrg.count, 3)) )
            
            #loop through mergers in the subhalo
            for j in range(mrg.count):
                allMrgSubhLenType = np.vstack((allMrgSubhLenType, 
                                                np.vstack([mrg.fpNpart[j,:], 
                                                           mrg.npNpart[j,:], 
                                                           mrg.descNpart[j,:]]).T.reshape(1,6,3))) ### CHANGED ARRAY SHAPE
      
    print(f"allMrgSubhID shape: {allMrgSubhID.shape}")
    sys.stdout.flush()

    with h5py.File(f"{savePath}/{outfilename}",'a') as mf:

        mf.attrs['num_mergers'] = total_mrg_count
        mf.attrs['merger_components_in_data_arrays'] = '(FirstProg, NextProg, Descendant)'

        mf.create_dataset('SubhaloLenType', data=allMrgSubhLenType)
        mf['SubhaloLenType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloLenType'].attrs['units'] = 'none'

        mf.create_dataset('shids_tree', data=allMrgSubhID)
        mf['shids_tree'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_tree'].attrs['units'] = 'none'

        mf.create_dataset('shids_subf', data=allMrgSubfID)
        mf['shids_subf'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_subf'].attrs['units'] = 'none'

        mf.create_dataset('snaps', data=allMrgSnaps)
        mf['snaps'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['snaps'].attrs['units'] = 'none'

        mf.create_dataset('ProgMassRatio', data=allMrgProgMassRatio)
        mf['ProgMassRatio'].attrs['dataShape'] = 'Nmrg'
        mf['ProgMassRatio'].attrs['units'] = 'none'

        mf.create_dataset('fpMass', data=allMrgfpMass * 1.0e10 / sub_hdr['HubbleParam']) #: [Msun]
        mf['fpMass'].attrs['dataShape'] = 'Nmrg'
        mf['fpMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('npMass', data=allMrgnpMass * 1.0e10 / sub_hdr['HubbleParam']) #: [Msun]
        mf['npMass'].attrs['dataShape'] = 'Nmrg'
        mf['npMass'].attrs['units'] = '[Msun]'
        
        # New datasets for Pop_Illustris (evolution-specific)
        mf.create_dataset('SubhaloBHMass', data=allMrgBHMass * 1.0e10 / sub_hdr['HubbleParam'])  # [Msun]
        mf['SubhaloBHMass'].attrs['dataShape'] = '(Nmrg, 2)'
        mf['SubhaloBHMass'].attrs['units'] = '[Msun]'
        
        mf.create_dataset('SubhaloHalfmassRadType', data=allMrgHalfRadType * 1e3 / sub_hdr['HubbleParam'])  # [cm]
        mf['SubhaloHalfmassRadType'].attrs['dataShape'] = '(Nmrg, 6, 3)'
        mf['SubhaloHalfmassRadType'].attrs['units'] = '[cm]'
        
        mf.create_dataset('SubhaloMassInRadType', data=allMrgMassInRadType * 1.0e10 / sub_hdr['HubbleParam'])  # [Msun]
        mf['SubhaloMassInRadType'].attrs['dataShape'] = '(Nmrg, 6, 3)'
        mf['SubhaloMassInRadType'].attrs['units'] = '[Msun]'
        
        mf.create_dataset('SubhaloVelDisp', data=allMrgVelDisp)  # [km/s]
        mf['SubhaloVelDisp'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloVelDisp'].attrs['units'] = '[km/s]'

    print(f"Finished processing merger trees for {nsubs} subhalos in snap {snapNum}.")
    print(f"Found {total_mrg_count} mergers.")   


if __name__ == "__main__":

    if len(sys.argv)>6:
        basePath = sys.argv[1]
        snapNum = int(sys.argv[2])
        minNdm = int(sys.argv[3])
        minNgas = int(sys.argv[4])
        minNstar = int(sys.argv[5])
        minNbh = int(sys.argv[6])
        savePath = sys.argv[7]
        evol_key = sys.argv[8] if len(sys.argv)>8 else None
        if len(sys.argv)>9:
            print("Too many command line args ({sys.argv}).")
            sys.exit()

    else:
        print("expecting 6 command line args: basePath, snapNum, minNdm, minNgas, minNstar, minNbh.")
        sys.exit()
    if evol_key==1:
        WriteMergerFileforevolution(savePath,basePath, snapNum, minNdm=minNdm, minNgas=minNgas, 
                    minNstar=minNstar, minNbh=minNbh, verbose=True)
    else:
        WriteMergerFile(savePath,basePath, snapNum, minNdm=minNdm, minNgas=minNgas, 
                    minNstar=minNstar, minNbh=minNbh, verbose=True)
