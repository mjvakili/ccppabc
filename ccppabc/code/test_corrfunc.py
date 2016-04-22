from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import os.path as path
import time
from Corrfunc import _countpairs
from Corrfunc.utils import read_catalog

import numpy as np 
# --- Local ---
import util
from data_multislice import data_RR
from data_multislice import data_random
from data_multislice import hardcoded_xi_bins

# --- halotools ---
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables import FoFGroups
from halotools.mock_observables.pair_counters import npairs_3d
import matplotlib.pyplot as plt
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors
from Corrfunc.utils import read_catalog
import time

def main():
   

    ###############################Multi-Dark########################### 
    
    model = PrebuiltHodModelFactory('zheng07', threshold=-21)
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    t0 = time.time()
    model.populate_mock(halocat, enforce_PBC = False)
    print(time.time() - t0)
    t0 = time.time()
    model.populate_mock(halocat, enforce_PBC = False)
    print(time.time() - t0)
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
    nthreads = 1

    binfile = path.join(path.dirname(path.abspath(__file__)),
                        "/home/mj/Corrfunc/xi_theory/tests/", "bins")
    autocorr = 1
    pos = pos.astype(np.float32)
   
    x , y , z = pos[:,0] , pos[:,1] , pos[:,2]
    DD = _countpairs.countpairs(autocorr, nthreads, binfile, x, y, z,
                                x , y , z)
    ND = len(pos)
    NR = 50*800000
    DD = np.array(DD)[:,3]
    
    
    num_randoms = 50 * 800000
    xran = np.random.uniform(0, 1000, num_randoms)
    yran = np.random.uniform(0, 1000, num_randoms)
    zran = np.random.uniform(0, 1000, num_randoms)
    randoms = np.vstack((xran, yran, zran)).T 
    
    randoms = randoms.astype(np.float32)

    xran = randoms[:,0]
    yran = randoms[:,1]
    zran = randoms[:,2]

     
    results_RR = _countpairs.countpairs(autocorr, nthreads, binfile, xran, yran, zran,
                                        xran, yran, zran)
    
    RR = np.array(results_RR)[:,3]
    factor1 = 1.*ND*ND/(NR*NR)
    mult = lambda x,y: x*y
    xi_MD = mult(1.0/factor1 , DD/RR) - 1.

    print(xi_MD)
   

    ###############################Subvolume of Multi-Dark########################### 
     
    num_srandoms = 50 * 8000
    xran = np.random.uniform(0, 200, num_srandoms)
    yran = np.random.uniform(0, 200, num_srandoms)
    zran = np.random.uniform(0, 200, num_srandoms)
    randoms = np.vstack((xran, yran, zran)).T 
    
    randoms = randoms.astype(np.float32)

    xran = randoms[:,0]
    yran = randoms[:,1]
    zran = randoms[:,2]

    results_RR = _countpairs.countpairs(autocorr, nthreads, binfile, xran, yran, zran,
                                        xran, yran, zran)
    RR_sub = np.array(results_RR)[:,3] 
    
    
    import util    
    sub_model = PrebuiltHodModelFactory('zheng07' , threshold = -21)
    sub_model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
    sub_halocat = CachedHaloCatalog(simname='multidark', redshift=0, halo_finder='rockstar')     
    xi_subs = [] 
    for i in range(10):
      simsubvol = lambda x: util.mask_func(x, i)
      sub_model.populate_mock(sub_halocat, masking_function=simsubvol, enforce_PBC=False)
      sub_pos = three_dim_pos_bundle(sub_model.mock.galaxy_table, 'x', 'y', 'z')
      nthreads = 1
      binfile = path.join(path.dirname(path.abspath(__file__)),
                        "/home/mj/Corrfunc/xi_theory/tests/", "bins")
      autocorr = 1
      sub_pos = sub_pos.astype(np.float32)
      x , y , z = sub_pos[:,0] , sub_pos[:,1] , sub_pos[:,2]
      DD_sub = _countpairs.countpairs(autocorr, nthreads, binfile, x, y, z,
                                        x, y, z)
    
      ND_sub = len(sub_pos)
      NR_sub = 50 * 8000
      DD_sub = np.array(DD_sub)[:,3]
      factor1 = 1.*ND_sub*ND_sub/(NR_sub*NR_sub)
      mult = lambda x,y: x*y
      xi_n = mult(1.0/factor1 , DD_sub/RR_sub) - 1.
      xi_subs.append(xi_n)
    
    xi_subs = np.array(xi_subs)
    np.savetxt("xi_subs.dat" , xi_subs)
     
    import matplotlib.pyplot as plt
    from ChangTools.plotting import prettyplot
    from ChangTools.plotting import prettycolors 

        
    binfile = path.join(path.dirname(path.abspath(__file__)),
                        "/home/mj/Corrfunc/xi_theory/tests/", "bins")
    rbins =  np.loadtxt(binfile)
    rbins_centers = np.mean(rbins , axis = 1)
    
    xi_subs = np.loadtxt("xi_subs.dat")

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    for i in range(10):

        ax.semilogx(rbins_centers , xi_subs[i,:] / xi_MD , alpha = 0.2)
        plt.xlabel(r"$r$" , fontsize = 20)
        plt.ylabel(r"$\xi_{\rm subvolume}(r) / \xi_{\rm MD}(r)$" , fontsize = 20) 
        plt.savefig("xi_ratios.pdf")
        


if __name__ == "__main__":
    main()
