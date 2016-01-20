'''

Modules for data and observables in ABC-PMC. 

'''
import os 
import numpy as np
from group_richness import gmf_bins
from group_richness import richness 
from group_richness import gmf as GMF
from halotools.empirical_models import PrebuiltHodModelFactory

def data_gmf(Mr=20, Nmock=500): 
    ''' Observed GMF from 'data'
    '''
    gmf_dat_file = ''.join(['../dat/gmf.Mr', str(Mr), '.dat'])
    gmf_sig_file = ''.join(['../dat/gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    return [np.loadtxt(gmf_dat_file), np.loadtxt(gmf_sig_file)]

def data_gmf_bins(): 
    ''' Just for consistency  
    '''
    return gmf_bins()

def data_nbar(Mr=20, Nmock=500): 
    '''
    Observed nbar from 'data'
    '''
    nbar = np.loadtxt(''.join(['../dat/nbar.Mr', str(Mr), '.dat']))
    nbar_cov = np.loadtxt(''.join(['../dat/nbar_cov.Mr', str(Mr), '.Nock', str(Nmock), '.dat']))
    return [nbar, nbar_cov]

def data_xi(Mr=20, Nmock=500): 
    '''
    Observed xi (2PCF) from 'data' and the diagonal elements of the xi covariance matrix
    '''
    xi_dat_file = ''.join(['../dat/xir.Mr', str(Mr), '.dat'])
    xi = np.loadtxt(xi_dat_file, unpack=True)

    # load covariance of xi 
    cov_dat_file = ''.join(['../dat/xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)
    cii = np.diag(covariance)   # diagonal elements

    return [xi, cii]

def data_xi_bin(Mr=20):
    ''' r bins for xi(r)
    '''
    rbin_file = ''.join(['../dat/xir_rbin.Mr', str(Mr), '.dat'])
    return np.loadtxt(rbin_file)

def data_xi_cov(Mr=20, Nmock=500): 
    '''
    Observed xi covariance. The entire covariance matrix
    '''
    # load covariance of xi 
    cov_dat_file = ''.join(['../dat/xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)

    return cov 
    

# Build observables ---------------
def build_xi_nbar_gmf(Mr=20): 
    '''
    Build "data" xi, nbar, GMF values and write to file 
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    model.populate_mock() # population mock realization 

    # write xi 
    data_xir  = model.mock.compute_galaxy_clustering()[1]
    output_file = ''.join(['../dat/xir.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, data_xir)

    # write nbar values 
    nbar = model.mock.number_density
    output_file = ''.join(['../dat/nbar.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, [nbar]) 
    
    # write GMF 
    rich = richness(model.mock.compute_fof_group_ids())
    gmf = GMF(rich)  # GMF
    output_file = ''.join(['../dat/gmf.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, gmf) 

    return None 

def build_xi_bin(Mr=20): 
    ''' Write out xi r bins
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    model.populate_mock() # population mock realization 

    # write xi 
    r_bin  = model.mock.compute_galaxy_clustering()[0]
    output_file = ''.join(['../dat/xir_rbin.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, r_bin)

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
        model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
        model.populate_mock() 
        
        # xi(r)
        xir.append(model.mock.compute_galaxy_clustering()[1])
        # nbar
        nbars.append(model.mock.number_density)
        # gmf
        rich = richness(model.mock.compute_fof_group_ids())
        gmfs.append(GMF(rich))  # GMF
        gmf_counts.append(GMF(rich, counts=True))   # Group counts 

    # save xi covariance 
    xi_covar = np.cov(np.array(xir).T)
    output_file = ''.join(['../dat/xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, xi_covar)

    # save nbar values 
    nbar_cov = np.var(nbars, axis=0) 
    output_file = ''.join(['../dat/nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, [nbar_cov]) 
    
    # write GMF covariance 
    gmf_cov = np.cov(np.array(gmfs).T)
    output_file = ''.join(['../dat/gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, gmf_cov) 
    # write GMF Poisson
    gmf_counts_mean = np.mean(gmf_counts, axis=0)
    poisson_gmf = np.sqrt(gmf_counts_mean) / 250.**3    # poisson errors
    output_file = ''.join(['../dat/gmf_sigma_poisson.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, poisson_gmf) 
    # write GMF standard dev 
    sigma_gmf = np.std(gmfs, axis=0)                    # sample variance 
    output_file = ''.join(['../dat/gmf_sigma_stddev.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, sigma_gmf) 
    # write GMF total noise 
    sigma_tot = (sigma_gmf**2 + poisson_gmf**2)**0.5    # total sigma
    output_file = ''.join(['../dat/gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, sigma_tot) 

    
    return None


def build_observations(Mr=20, Nmock=500): 
    ''' Build all the fake observations
    '''
    # xi, nbar, gmf
    print 'Building xi(r), nbar, GMF ... ' 
    build_xi_nbar_gmf(Mr=Mr)
    build_xi_bin(Mr=Mr)
    
    # covariances
    print 'Building covariances ... ' 
    build_xi_nbar_gmf_cov(Mr=Mr, Nmock=Nmock)

if __name__=='__main__': 
    build_observations(Nmock=2)

