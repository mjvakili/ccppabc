'''
testing the tpcf calculation with precomputed RRs
'''
from __future__ import division
import numpy as np
from multiprocessing import cpu_count
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables.pair_counters import npairs
from halotools.sim_manager import CachedHaloCatalog

def test_precomputed_rr(Nr, Mr = 21):
    '''
    Mr = Luminositty threshold
    Nr = Number of randoms
    '''

    rbins = np.logspace(-1, 1.25, 15)
    rmax = rbins.max()
    rbin_centers = (rbins[1:] + rbins[0:-1])/2.    

    halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0)
    model = PrebuiltHodModelFactory("zheng07")
    model.populate_mock(halocat = halocat, enforce_PBC = False)
    data = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
    print data.shape
    L = halocat.Lbox

    xmin , ymin , zmin = 0., 0., 0.
    xmax , ymax , zmax = L, L, L
   
    
    num_randoms = Nr
    xran = np.random.uniform(xmin, xmax, num_randoms)
    yran = np.random.uniform(ymin, ymax, num_randoms)
    zran = np.random.uniform(zmin, zmax, num_randoms)
    randoms = np.vstack((xran, yran, zran)).T
    

    verbose = False
    num_threads = cpu_count()

    period = None
    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]

    normal_result = tpcf(
            data, rbins, data, 
            randoms=randoms, period = period, 
            max_sample_size=int(1e4), estimator='Landy-Szalay', 
            approx_cell1_size=approx_cell1_size, 
            approx_cellran_size=approx_cellran_size)


    #count data pairs
    DD = npairs(
            data, data, rbins, period,
            verbose, num_threads,
            approx_cell1_size, approx_cell2_size)
    DD = np.diff(DD)
    #count random pairs
    RR = npairs(
            randoms, randoms, rbins, period,
            verbose, num_threads,
            approx_cellran_size, approx_cellran_size)
    RR = np.diff(RR)
    #count data random pairs 
    DR = npairs(
            data, randoms, rbins, period,
            verbose, num_threads,
            approx_cell1_size, approx_cell2_size)
    DR = np.diff(DR)

    print "DD=", DD
    print "DR=", DR
    print "RR=", RR        
    
    ND = len(data)
    NR = len(randoms)

    factor1 = ND*ND/(NR*NR)
    factor2 = ND*NR/(NR*NR)

    mult = lambda x,y: x*y
    xi_LS = mult(1.0/factor1,DD/RR) - mult(1.0/factor2,2.0*DR/RR) + 1.0

    print "xi=" , xi_LS
    print "normal=" , normal_result

    result_with_RR_precomputed = tpcf(
            data, rbins, data, 
            randoms=randoms, period = period, 
            max_sample_size=int(1e5), estimator='Landy-Szalay', 
            approx_cell1_size=approx_cell1_size, 
            approx_cellran_size=approx_cellran_size, 
            RR_precomputed = RR, 
            NR_precomputed = NR)
    
    print "xi_pre=" , result_with_RR_precomputed
 


if __name__ == '__main__':

    test_precomputed_rr(1e5 , 21)
