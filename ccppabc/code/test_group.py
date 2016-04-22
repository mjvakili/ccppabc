'''

Tests for GMF

'''
#general python modules
import os
import pickle
import numpy as np
from multiprocessing import cpu_count
from numpy.linalg import solve

#haltools functions
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables import FoFGroups

#our ccppabc functions
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF

import matplotlib.pyplot as plt 



def test_subvol_gmf(Mr):
    '''
    '''
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(simname='multidark',
                        masking_function=datsubvol,
                        enforce_PBC=False)
    

    #compute group richness    
    # These are wrong because they assume periodicity
    #rich = richness(model.mock.compute_fof_group_ids())
    #gmf = GMF(rich)  #GMF
    #print gmf
    #print GMF(rich , counts = True) 

    galaxy_sample = model.mock.galaxy_table
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vz = galaxy_sample['vz']

    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z', velocity = vz , velocity_distortion_dimension="z")

    b_para, b_perp = 0.2, 0.2
    groups = FoFGroups(pos, b_perp, b_para, period = None, 
                      Lbox = 200 , num_threads='max')

    gids = groups.group_ids
    rich = richness(gids)
    gmf = GMF(rich)

    print gmf
    print GMF(rich , counts = True) 
    return None 


def test_GMFbinning(Mr): 
    ''' Tests for the GMF binning scheme in order to make it sensible. 
    '''
    gids_saved = 'gids_saved.p'
    if not os.path.isfile(gids_saved): 
        thr = -1. * np.float(Mr)
        model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                        halocat='multidark', redshift=0.)
        model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

        datsubvol = lambda x: util.mask_func(x, 0)
        model.populate_mock(simname='multidark',
                            masking_function=datsubvol,
                            enforce_PBC=False)
        
        galaxy_sample = model.mock.galaxy_table
        x = galaxy_sample['x']
        y = galaxy_sample['y']
        z = galaxy_sample['z']
        vz = galaxy_sample['vz']

        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z', velocity = vz , velocity_distortion_dimension="z")

        b_para, b_perp = 0.2, 0.2
        groups = FoFGroups(pos, b_perp, b_para, period = None, 
                          Lbox = 200 , num_threads='max')
        
        pickle.dump(groups, open(gids_saved, 'wb'))
    else: 
        groups = pickle.load(open(gids_saved, 'rb'))

    gids = groups.group_ids
    rich = richness(gids)
    #print "rich=" , rich
    rbins = np.logspace(np.log10(3.), np.log10(20), 10)
    rbins = np.array([1, 2.,3.,4.,5.,6.,7,9,11,14,17,20])
    gmf = GMF(rich, counts=False, bins=rbins)
    gmf_counts = GMF(rich, counts=True, bins=rbins)

    print rbins
    print gmf
    print gmf_counts
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    sub.plot(0.5*(rbins[:-1] + rbins[1:]), gmf)

    sub.set_xscale('log') 
    sub.set_yscale('log') 
    
    plt.show()
    #print gmf
    #print GMF(rich , counts = True) 
    return None 


if __name__ == '__main__':
     #test_subvol_gmf(21)
     test_GMFbinning(21)
