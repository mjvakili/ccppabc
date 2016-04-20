'''

Code to test for the 2pcf 

'''
import numpy as np 
# --- Local ---
import util
from data import data_RR
from data import data_random
from data import hardcoded_xi_bins

# --- halotools ---
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables import FoFGroups
from halotools.mock_observables.pair_counters import npairs

import matplotlib.pyplot as plt

from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors

def Subvolume_Analytic(N_sub, ratio=False): 
    ''' Test the 2PCF estimates from MultiDark subvolume versus the 
    analytic 2PCF for the entire MultiDark volume

    Parameters
    ----------
    N_sub : (int)
        Number of subvolumes to sample

    '''
    prettyplot()
    pretty_colors = prettycolors()
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    xi_bin = hardcoded_xi_bins() 

    # Entire MultiDark Volume (Analytic xi) 
    model = PrebuiltHodModelFactory('zheng07', threshold=-21)
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    
    model.populate_mock(halocat)
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
    
    # while the estimator claims to be Landy-Szalay, I highly suspect it
    # actually uses Landy-Szalay since DR pairs cannot be calculated from 
    # analytic randoms
    xi = tpcf(pos, xi_bin, period=model.mock.Lbox, max_sample_size=int(2e5), estimator='Landy-Szalay', num_threads=1)
    if not ratio:  
        sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), xi, lw=2, ls='-', c='k', label=r'Analytic $\xi$ Entire Volume') 
        
    # MultiDark SubVolume (precomputed RR pairs) 
    sub_model = PrebuiltHodModelFactory('zheng07', threshold=-21)
    sub_model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
    sub_halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    RR = data_RR()
    randoms = data_random()
    NR = len(randoms)
    
    for method in ['Landy-Szalay', 'Natural']:
        
        if method == 'Landy-Szalay': 
            iii = 3
        elif method == 'Natural': 
            iii = 5
        
        sub_xis = [] 
        for ii in range(N_sub): 
            # randomly sample one of the subvolumes
            rint = np.random.randint(1, 125)
            simsubvol = lambda x: util.mask_func(x, rint)
            sub_model.populate_mock(sub_halocat, masking_function=simsubvol, enforce_PBC=False)
               
            pos = three_dim_pos_bundle(sub_model.mock.galaxy_table, 'x', 'y', 'z')

            xi , yi , zi = util.random_shifter(rint)
            temp_randoms = randoms.copy()
            temp_randoms[:,0] += xi
            temp_randoms[:,1] += yi
            temp_randoms[:,2] += zi
            
            rmax = xi_bin.max()
            approx_cell1_size = [rmax , rmax , rmax]
            approx_cellran_size = [rmax , rmax , rmax]

            sub_xi = tpcf(
                    pos, xi_bin, pos, 
                    randoms=temp_randoms, 
                    period = None, 
                    max_sample_size=int(1e5), 
                    estimator=method, 
                    approx_cell1_size = approx_cell1_size, 
                    approx_cellran_size = approx_cellran_size,
                    RR_precomputed = RR,
                    NR_precomputed = NR)
            label = None 
            if ii == N_sub - 1: 
                label = 'Subvolumes'
            
            if not ratio: 
                sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi, lw=0.5, ls='--', c=pretty_colors[iii])
            sub_xis.append(sub_xi)

        sub_xi_avg = np.sum(sub_xis, axis=0)/np.float(N_sub)
        if not ratio: 
            sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg, 
                    lw=2, ls='--', c=pretty_colors[iii], label='Subvolume '+method)
        else: 
            sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg/xi, 
                    lw=2, ls='--', c=pretty_colors[iii], label='Subvolume '+method)
    
    sub.set_xlim([0.1, 50.])
    sub.set_xlabel('r', fontsize=30)
    sub.set_xscale('log')
    
    if not ratio: 
        sub.set_ylabel(r"$\xi \mathtt{(r)}$", fontsize=25)
        sub.set_yscale('log')
    else: 
        sub.set_ylabel(r"$\xi^\mathtt{subvolume}/\xi^\mathtt{volume}$", fontsize=25)

    sub.legend(loc='lower left')
    
    if ratio: 
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_analytic.ratio.png'])
    else:
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_analytic.png'])
    fig.savefig(fig_file, bbox_inches='tight', dpi=100)
    plt.close()
    return None 


if __name__=="__main__": 
    Subvolume_Analytic(5, ratio=False)
    Subvolume_Analytic(5, ratio=True)
