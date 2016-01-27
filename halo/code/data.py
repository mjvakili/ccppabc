'''

Modules for data and observables in ABC-PMC. 

'''
import os 
import numpy as np
from numpy.linalg import solve

import util
from group_richness import gmf_bins
from group_richness import richness 
from group_richness import gmf as GMF
from halotools.empirical_models import PrebuiltHodModelFactory

def data_hod_param(Mr=20):
    '''
    HOD parameters of 'observations'. Returns dictionary with hod parameters. 
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict

# --- GMF --- 
def data_gmf(Mr=20, Nmock=500): 
    ''' Observed GMF from 'data'
    '''
    gmf_dat_file = ''.join([util.dat_dir(), 'gmf.Mr', str(Mr), '.dat'])
    gmf_sig_file = ''.join([util.dat_dir(), 'gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    return [np.loadtxt(gmf_dat_file), np.loadtxt(gmf_sig_file)]

def data_gmf_bins(): 
    ''' Just for consistency  
    '''
    return gmf_bins()

def data_gmf_cov(Mr=20, Nmock=500): 
    '''
    GMF covariance matrix 
    '''
    cov_file = ''.join([util.dat_dir(), 'gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    return np.loadtxt(cov_file)

# --- nbar --- 
def data_nbar(Mr=20, Nmock=500): 
    '''
    Observed nbar from 'data'
    '''
    nbar = np.loadtxt(''.join([util.dat_dir(), 'nbar.Mr', str(Mr), '.dat']))
    nbar_cov = np.loadtxt(''.join([util.dat_dir(), 'nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat']))
    return [nbar, nbar_cov]

# --- 2PCF --- 
def data_xi(Mr=20, Nmock=500): 
    '''
    Observed xi (2PCF) from 'data' and the diagonal elements of the xi covariance matrix
    '''
    xi_dat_file = ''.join([util.dat_dir(), 'xir.Mr', str(Mr), '.dat'])
    xi = np.loadtxt(xi_dat_file, unpack=True)

    # load covariance of xi 
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)
    cii = np.diag(cov)   # diagonal elements

    return [xi, cii]

def data_xi_full_cov(Mr=20, Nmock=500): 
    '''
    Observed xi (2PCF) from 'data' and the diagonal elements of the xi covariance matrix
    '''
    xi_dat_file = ''.join([util.dat_dir(), 'xir.Mr', str(Mr), '.dat'])
    xi = np.loadtxt(xi_dat_file, unpack=True)

    # load covariance of xi 
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)
    #cii = np.diag(cov)   # diagonal elements

    return [xi, cov]

def data_xi_bins(Mr=20):
    ''' r bins for xi(r)
    '''
    rbin_file = ''.join([util.dat_dir(), 'xir_rbin.Mr', str(Mr), '.dat'])
    return np.loadtxt(rbin_file)

def data_xi_cov(Mr=20, Nmock=500): 
    '''
    Observed xi covariance. The entire covariance matrix
    '''
    # load covariance of xi 
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)

    return cov 

def data_xi_inv_cov(Mr=20, Nmock=500, unbias_str=True):
    '''
    Observed inverse covariance of xi with/without the unbias estimator 
    factor. Default multiplies by the unbias estimator factor. 
    '''
    if unbias_str: 
        unbias_str = '.unbias'
    else: 
        unbias_str = ''

    inv_cov_file = ''.join([util.dat_dir(), 
        'xi_inv_cov', unbias_str, '.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])

    inv_cov = np.loadtxt(inv_cov_file) 

    return inv_cov
    

# Build observables ---------------
def build_xi_nbar_gmf(Mr=20): 
    '''
    Build "data" xi, nbar, GMF values and write to file 
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    model.populate_mock() # population mock realization 

    # write xi 
    data_xir = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[1]
    output_file = ''.join([util.dat_dir(), 'xir.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, data_xir)

    # write nbar values 
    nbar = model.mock.number_density
    output_file = ''.join([util.dat_dir(), 'nbar.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, [nbar]) 
    
    # write GMF 
    rich = richness(model.mock.compute_fof_group_ids())
    gmf = GMF(rich)  # GMF
    output_file = ''.join([util.dat_dir(), 'gmf.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, gmf) 

    return None 

def build_xi_bins(Mr=20): 
    ''' hardcoded r bins for xi. 
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    model.populate_mock() # population mock realization 
    r_bin  = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[0]
    output_file = ''.join([util.dat_dir(), 'xir_rbin.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, r_bin)
    return None

def hardcoded_xi_bins(): 
    ''' hardcoded xi bin edges.They are spaced out unevenly due to sparseness
    at inner r bins. So the first bin ranges from 0.15 to 0.5
    '''
    r_bins = np.concatenate([np.array([0.15]), np.logspace(np.log10(0.5), np.log10(20.), 15)])
    return r_bins

def build_xi_nbar_gmf_cov(Mr=20, Nmock=500): 
    '''
    Build covariance matrix for xi, variance for nbar, and a bunch of stuff for gmf 
    ...  
    using Nmock realizations of halotool mocks  
    ''' 
    xir = [] 
    nbars = [] 
    gmfs = [] 
    gmf_counts = [] 
    for i in xrange(Nmock): 
        print 'mock#', i
        model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
        model.populate_mock() 
        
        # xi(r)
        xir.append(model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[1])
        # nbar
        nbars.append(model.mock.number_density)
        # gmf
        rich = richness(model.mock.compute_fof_group_ids())
        gmfs.append(GMF(rich))  # GMF
        gmf_counts.append(GMF(rich, counts=True))   # Group counts 

    # save xi covariance 
    xi_covar = np.cov(np.array(xir).T)
    output_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, xi_covar)

    # save nbar values 
    nbar_cov = np.var(nbars, axis=0) 
    output_file = ''.join([util.dat_dir(), 'nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, [nbar_cov]) 
    
    # write GMF covariance 
    gmf_cov = np.cov(np.array(gmfs).T)
    output_file = ''.join([util.dat_dir(), 'gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, gmf_cov) 
    # write GMF Poisson
    gmf_counts_mean = np.mean(gmf_counts, axis=0)
    poisson_gmf = np.sqrt(gmf_counts_mean) / 250.**3    # poisson errors
    output_file = ''.join([util.dat_dir(), 'gmf_sigma_poisson.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, poisson_gmf) 
    # write GMF standard dev 
    sigma_gmf = np.std(gmfs, axis=0)                    # sample variance 
    output_file = ''.join([util.dat_dir(), 'gmf_sigma_stddev.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, sigma_gmf) 
    # write GMF total noise 
    sigma_tot = (sigma_gmf**2 + poisson_gmf**2)**0.5    # total sigma
    output_file = ''.join([util.dat_dir(), 'gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, sigma_tot) 
    return None

def build_xi_inv_cov(Mr=20, Nmock=500, unbias=True): 
    '''
    Calculate the inverse covariance of xi multiplied by the unbiased
    estimator factor (Nmocks - 2 - Nbins)/(Nmocks - 1). 

    Mainly used in MCMC inference
    '''
    xi_cov = data_xi_cov(Mr=Mr, Nmock=Nmock)    # covariance matrix of xi
    N_bins = int(np.sqrt(xi_cov.size))          # cov matrix is N_bin x N_bin
    
    if unbias: 
        f_unbias = np.float(Nmock - 2. - N_bins)/np.float(Nmock - 1.)
        unbias_str = '.unbias'
    else: 
        f_unbias = 1.0 
        unbias_str = ''
    
    inv_c = solve(np.eye(N_bins) , xi_cov) * f_unbias 

    output_file = ''.join([util.dat_dir(), 
        'xi_inv_cov', unbias_str, '.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, inv_c) 

    return None

def build_observations(Mr=20, Nmock=500): 
    ''' Build all the fake observations
    '''
    # xi, nbar, gmf
    print 'Building xi(r), nbar, GMF ... ' 
    build_xi_bins(Mr=Mr)
    build_xi_nbar_gmf(Mr=Mr)
    
    # covariances
    print 'Building covariances ... ' 
    build_xi_nbar_gmf_cov(Mr=Mr, Nmock=Nmock)
    print 'Build inverse covariance for xi ...'
    build_xi_inv_cov(Mr=Mr, Nmock=Nmock, unbias=True)

if __name__=='__main__': 
    build_observations()
