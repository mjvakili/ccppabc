'''

Modules for data and observables in ABC-PMC. 

'''
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

def build_xi_cov(Mr=20, Nmock=500): 
    '''
    Build covariance matrix for xi, using Nmock simulations 
    ''' 
    xir = [] 
    for i in xrange(Nmock): 
        model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
        model.populate_mock() 

        xir.append(model.mock.compute_galaxy_clustering()[1])

    covar = np.cov(np.array(xir).T)
    output_file = ''.join(['../dat/xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])

    np.savetxt(output_file, covar)
    
    return None

def build_nbar_cov(Mr=20, Nmock=500): 
    ''' Build observed nbar value
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    nbars = [] 
    for i in xrange(Nmock): 
        model.populate_mock() 
        nbars.append(model.mock.number_density)
    
    nbar_cov = np.var(nbars, axis=0) 

    # save nbar values 
    output_file = ''.join(['../dat/nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, [nbar_cov]) 
    return None

    
def build_gmf_sigma(Mr=20, Nmock=500): 
    ''' Build 'observed' uncertainty in GMF from Nmock simulated realizations. 
    Uncertainty is the quadruture of poisson errors and stddev. 
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    
    gmfs = [] 
    gmf_counts = [] 
    for i in xrange(Nmock): 
        model.populate_mock() 
        rich = richness(model.mock.compute_fof_group_ids())
        gmfs.append(GMF(rich))  # GMF
        gmf_counts.append(GMF(rich, counts=True))   # Group counts 

    gmf_counts_mean = np.mean(gmf_counts, axis=0)
    poisson_gmf = np.sqrt(gmf_counts_mean) / 250.**3    # poisson errors
    sigma_gmf = np.std(gmfs, axis=0)                    # sample variance 

    sigma_tot = (sigma_gmf**2 + poisson_gmf**2)**0.5    # total sigma

    # save to file  
    output_file = ''.join(['../dat/gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, sigma_tot) 
    return None

def build_observations(Mr=20, Nmock=500): 
    ''' Build all the fake observations
    '''
    # xi, nbar, gmf
    print 'Building xi(r), nbar, GMF ... ' 
    build_xi_nbar_gmf(Mr=Mr)
    
    # covariances
    print 'Building xi covariance ... ' 
    build_xi_cov(Mr=Mr, Nmock=Nmock)
    # nbar
    print 'Building nbar covariance ... ' 
    build_nbar_cov(Mr=Mr, Nmock=Nmock)
    # gmf
    print 'Building gmf covariance ... ' 
    build_gmf_sigma(Mr=Mr, Nmock=Nmock)
    return None
