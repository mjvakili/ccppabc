'''

Code to test for the 2pcf 

'''
import os
import pickle
import numpy as np 
# --- Local ---
import util
from data import data_RR
from data import data_random
from data import xi_binedges 

# --- halotools ---
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle

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

    pickle_file = ''.join([
        '/export/bbq2/hahn/ccppabc/dump/', 
        'xi_subvolume_test', 
        '.Nsub', str(N_sub), 
        '.p'])
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    xi_bin = xi_binedges() 
    
    if os.path.isfile(pickle_file):
        data_dump = pickle.load(open(pickle_file, 'rb'))
        full_xi = data_dump['full_xi']
    else: 
        # Entire MultiDark Volume (Analytic xi) 
        model = PrebuiltHodModelFactory('zheng07', threshold=-21)
        halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
        
        model.populate_mock(halocat)
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
        
        # while the estimator claims to be Landy-Szalay, I highly suspect it
        # actually uses Landy-Szalay since DR pairs cannot be calculated from 
        # analytic randoms
        full_xi = tpcf(pos, xi_bin, period=model.mock.Lbox, max_sample_size=int(2e5), estimator='Landy-Szalay', num_threads=1)
        data_dump = {} 
        data_dump['full_xi'] = full_xi

    if not ratio:  
        sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), full_xi, lw=2, ls='-', c='k', label=r'Analytic $\xi$ Entire Volume') 
    
    if not os.path.isfile(pickle_file):
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
        
        if not os.path.isfile(pickle_file): 
            sub_xis_list = [] 
            sub_xis = np.zeros(len(full_xi)) 

            for ii in range(1,N_sub+1): 
                # randomly sample one of the subvolumes
                rint = ii #np.random.randint(1, 125)
                simsubvol = lambda x: util.mask_func(x, rint)
                sub_model.populate_mock(sub_halocat, masking_function=simsubvol, enforce_PBC=False)
                   
                pos = three_dim_pos_bundle(sub_model.mock.galaxy_table, 'x', 'y', 'z')

                xi, yi , zi = util.random_shifter(rint)
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
                        RR_precomputed=RR,
                        NR_precomputed=NR)
                label = None 
                if ii == N_sub - 1: 
                    label = 'Subvolumes'
                
                #if not ratio: 
                #    sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi, lw=0.5, ls='--', c=pretty_colors[iii])
                sub_xis += sub_xi
                sub_xis_list.append(sub_xi)

            sub_xi_avg = sub_xis/np.float(N_sub)

            data_dump[method] = {} 
            data_dump[method]['sub_xi_avg'] = sub_xi_avg
            data_dump[method]['sub_xis_list'] = sub_xis_list 
        else: 
            sub_xis_list = data_dump[method]['sub_xis_list']
            sub_xi_avg = data_dump[method]['sub_xi_avg'] 

        if not ratio: 
            sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg, 
                    lw=2, ls='--', c=pretty_colors[iii], label='Subvolume '+method)
        else: 
            sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg/full_xi, 
                    lw=2, ls='--', c=pretty_colors[iii], label='Subvolume '+method)
    
    if not os.path.isfile(pickle_file): 
        pickle.dump(data_dump, open(pickle_file, 'wb')) 

    sub.set_xlim([0.1, 50.])
    sub.set_xlabel('r', fontsize=30)
    sub.set_xscale('log')
    
    if not ratio: 
        sub.set_ylabel(r"$\xi \mathtt{(r)}$", fontsize=25)
        sub.set_yscale('log')
    else: 
        sub.set_ylabel(r"$\overline{\xi^\mathtt{sub}}/\xi^\mathtt{all}$", fontsize=25)

    sub.legend(loc='lower left')
    
    if ratio: 
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_analytic.Nsub', str(N_sub), '.ratio.png'])
    else:
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_analytic.Nsub', str(N_sub), '.png'])
    fig.savefig(fig_file, bbox_inches='tight', dpi=100)
    plt.close()
    return None 

def Subvolume_Fullvolume(N_sub, ratio=False): 
    ''' Test the 2PCF estimates from MultiDark subvolume versus the 
    2PCF for the entire MultiDark volume WITHOUT periodic boundary conditions
    and actual pair counts. 

    Parameters
    ----------
    N_sub : (int)
        Number of subvolumes to sample

    '''
    prettyplot()
    pretty_colors = prettycolors()

    pickle_file = ''.join([
        '/export/bbq2/hahn/ccppabc/dump/', 
        'xi_subvolume_fullvolume_test', 
        '.Nsub', str(N_sub), 
        '.p'])
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    xi_bin = xi_binedges() 
    
    if os.path.isfile(pickle_file):
        data_dump = pickle.load(open(pickle_file, 'rb'))
        full_xi = data_dump['full_xi']
    else: 
        # Entire MultiDark Volume (No Periodic Boundary Conditions) 
        model = PrebuiltHodModelFactory('zheng07', threshold=-21)
        halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')

        full_randoms = data_random(box='md_all')
        full_RR = data_RR(box='md_all')
        full_NR = len(full_randoms)

        rmax = xi_bin.max()
        full_approx_cell1_size = [rmax , rmax , rmax]
        full_approx_cellran_size = [rmax , rmax , rmax]

        model.populate_mock(halocat)
        full_pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
        
        full_xi = tpcf(
                full_pos, xi_bin, 
                randoms=full_randoms, period=None, 
                do_auto=True,
                do_cross=False, 
                num_threads=5, 
                max_sample_size=int(full_pos.shape[0]), estimator='Natural', 
                approx_cell1_size=full_approx_cell1_size, 
                approx_cellran_size=full_approx_cellran_size,
                RR_precomputed = full_RR,
                NR_precomputed = full_NR)

        data_dump = {} 
        data_dump['full_xi'] = full_xi

    if not ratio:  
        sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), full_xi, lw=2, ls='-', c='k', label=r'Full Volume') 
    
    if not os.path.isfile(pickle_file):
        # MultiDark SubVolume (precomputed RR pairs) 
        sub_model = PrebuiltHodModelFactory('zheng07', threshold=-21)
        sub_model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
        sub_halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
        sub_RR = data_RR(box='md_sub')
        sub_randoms = data_random(box='md_sub')
        sub_NR = len(sub_randoms)
    
        
        sub_xis_list = [] 
        sub_xis = np.zeros(len(full_xi)) 

        for ii in range(1,N_sub+1): 
            print 'Subvolume ', ii 
            # randomly sample one of the subvolumes
            rint = ii #np.random.randint(1, 125)
            simsubvol = lambda x: util.mask_func(x, rint)
            sub_model.populate_mock(sub_halocat, masking_function=simsubvol, enforce_PBC=False)
               
            pos = three_dim_pos_bundle(sub_model.mock.galaxy_table, 'x', 'y', 'z')

            xi, yi , zi = util.random_shifter(rint)
            temp_randoms = sub_randoms.copy()
            temp_randoms[:,0] += xi
            temp_randoms[:,1] += yi
            temp_randoms[:,2] += zi
            
            rmax = xi_bin.max()
            sub_approx_cell1_size = [rmax , rmax , rmax]
            sub_approx_cellran_size = [rmax , rmax , rmax]

            sub_xi = tpcf(
                    pos, xi_bin,  
                    randoms=temp_randoms, 
                    period = None, 
                    do_auto=True,
                    do_cross=False, 
                    num_threads=5, 
                    max_sample_size=int(pos.shape[0]), 
                    estimator='Natural', 
                    approx_cell1_size = sub_approx_cell1_size, 
                    approx_cellran_size = sub_approx_cellran_size,
                    RR_precomputed=sub_RR,
                    NR_precomputed=sub_NR)

            label = None 
            if ii == N_sub - 1: 
                label = 'Subvolumes'
            
            sub_xis += sub_xi
            sub_xis_list.append(sub_xi)

        sub_xi_avg = sub_xis/np.float(N_sub)

        data_dump['Natural'] = {} 
        data_dump['Natural']['sub_xi_avg'] = sub_xi_avg
        data_dump['Natural']['sub_xis_list'] = sub_xis_list 
    else: 
        sub_xis_list = data_dump['Natural']['sub_xis_list']
        sub_xi_avg = data_dump['Natural']['sub_xi_avg'] 

    if not os.path.isfile(pickle_file): 
        pickle.dump(data_dump, open(pickle_file, 'wb')) 

    if not ratio: 
        sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg, 
                lw=2, ls='--', c=pretty_colors[3], label='Subvolume')
    else: 
        sub.plot(0.5*(xi_bin[:-1]+xi_bin[1:]), sub_xi_avg/full_xi, 
                lw=2, ls='--', c=pretty_colors[3], label='Subvolume')

    sub.set_xlim([0.1, 50.])
    sub.set_xlabel('r', fontsize=30)
    sub.set_xscale('log')
    
    if not ratio: 
        sub.set_ylabel(r"$\xi \mathtt{(r)}$", fontsize=25)
        sub.set_yscale('log')
    else: 
        sub.set_ylabel(r"$\overline{\xi^\mathtt{sub}}/\xi^\mathtt{all}$", fontsize=25)

    sub.legend(loc='lower left')
    
    if ratio: 
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_fullvolume.Nsub', str(N_sub), '.ratio.png'])
    else:
        fig_file = ''.join([util.fig_dir(), 'test_xi_subvolume_fullvolume.Nsub', str(N_sub), '.png'])
    fig.savefig(fig_file, bbox_inches='tight', dpi=100)
    plt.close()
    return None 

if __name__=="__main__": 
    Subvolume_Fullvolume(10, ratio=False)
    Subvolume_Fullvolume(10, ratio=True)
    #Subvolume_Analytic(100, ratio=False)
    #Subvolume_Analytic(100, ratio=True)
