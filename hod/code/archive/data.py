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
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle

def data_hod_param(Mr=21):
    '''
    HOD parameters of 'observations'. Returns dictionary with hod parameters.
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict

# --- GMF ---
def data_gmf(Mr=21, Nmock=500):
    '''
    This loads the observed GMF from the mock 'data' and the covariance matrix (or sigma) corresponding to that.
    Note to self. Return the inverse covariance matrix motherfucker!
    '''
    gmf_dat_file = ''.join([util.dat_dir(), 'gmf.Mr', str(Mr), '.dat'])
    gmf_sig_file = ''.join([util.dat_dir(), 'gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    return [np.loadtxt(gmf_dat_file), np.loadtxt(gmf_sig_file)]

def data_gmf_bins():
    ''' Just for consistency, returns the bins
    '''
    return gmf_bins()

def data_gmf_cov(Mr=21, Nmock=500):
    '''
    Returns the GMF covariance matrix
    '''
    cov_file = ''.join([util.dat_dir(), 'gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    return np.loadtxt(cov_file)

# --- nbar ---
def data_nbar(Mr=21, Nmock=500):
    '''
    Observed nbar measured from the mock 'data' and the covariance matrix
    '''
    nbar = np.loadtxt(''.join([util.dat_dir(), 'nbar.Mr', str(Mr), '.dat']))
    nbar_cov = np.loadtxt(''.join([util.dat_dir(), 'nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat']))
    return [nbar, nbar_cov]

# --- 2PCF ---
def data_xi(Mr=21, Nmock=500):
    '''
    Observed xi (2PCF) from the mock 'data' and the diagonal elements of the xi covariance matrix
    '''
    xi_dat_file = ''.join([util.dat_dir(), 'xir.Mr', str(Mr), '.dat'])
    xi = np.loadtxt(xi_dat_file, unpack=True)

    # load covariance of xi
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)
    cii = np.diag(cov)   # diagonal elements

    return [xi, cii]

def data_xi_full_cov(Mr=21, Nmock=500):
    '''
    Observed xi (2PCF) from the 'data' and the diagonal elements of the xi covariance matrix
    '''
    xi_dat_file = ''.join([util.dat_dir(), 'xir.Mr', str(Mr), '.dat'])
    xi = np.loadtxt(xi_dat_file, unpack=True)

    # load covariance of xi
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)
    #cii = np.diag(cov)   # diagonal elements

    return [xi, cov]

def data_xi_bins(Mr=21):
    ''' 
    r bins for xi(r)
    '''
    rbin_file = ''.join([util.dat_dir(), 'xir_rbin.Mr', str(Mr), '.dat'])
    return np.loadtxt(rbin_file)

def data_xi_cov(Mr=21, Nmock=500):
    '''
    Observed xi covariance. The entire covariance matrix
    '''
    # load covariance of xi
    cov_dat_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    cov = np.loadtxt(cov_dat_file)

    return cov

def data_xi_inv_cov(Mr=21, Nmock=500, unbias_str=True):
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


def data_full_cov(Mr=21, Nmock=500):

    full_cov_fn = ''.join([util.dat_dir(),
                          'nbar_gmf_xir_cov.Mr', str(Mr),
                          '.Nmock', str(Nmock), '.dat'])

    fullcov = np.loadtxt(full_cov_fn)

    return fullcov


def data_nbar_gmf_cov(Mr=21, Nmock=500):

    nbgmf_cov_fn = ''.join([util.dat_dir(),
                          'nbar_gmf_cov.Mr', str(Mr),
                          '.Nmock', str(Nmock), '.dat'])

    nbgmf_cov = np.loadtxt(nbgmf_cov_fn)

    return nbgmf_cov


def data_nbar_xi_cov(Mr=21, Nmock=500):

    nbxi_cov_fn = ''.join([util.dat_dir(),
                          'nbar_xi_cov.Mr', str(Mr),
                          '.Nmock', str(Nmock), '.dat'])

    nbxi_cov = np.loadtxt(nbxi_cov_fn)

    return nbxi_cov


def data_gmf_xi_cov(Mr=21, Nmock=500):

    gmfxi_cov_fn = ''.join([util.dat_dir(),
                          'gmf_xi_cov.Mr', str(Mr),
                          '.Nmock', str(Nmock), '.dat'])

    gmfxi_cov = np.loadtxt(gmfxi_cov_fn)

    return gmfxi_cov


def data_nb_gmf_xi_fullicov(Mr=21, Nmock=500):

    inv_cov_fn = ''.join([util.dat_dir(),
                         'nbar_gmf_xir_inv_cov.Mr', str(Mr),
                         '.Nmock', str(Nmock), '.dat'])

    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov


def data_nbar_gmf_inv_cov(Mr=21, Nmock=500):

    inv_cov_fn = ''.join([util.dat_dir(),
                         'nbar_gmf_inv_cov.Mr', str(Mr),
                         '.Nmock', str(Nmock), '.dat'])

    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov


def data_nbar_xi_inv_cov(Mr=21, Nmock=500):

    inv_cov_fn = ''.join([util.dat_dir(),
                         'nbar_xi_inv_cov.Mr', str(Mr),
                         '.Nmock', str(Nmock), '.dat'])

    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov


def data_gmf_xi_inv_cov(Mr=21, Nmock=500):

    inv_cov_fn = ''.join([util.dat_dir(),
                         'gmf_xi_inv_cov.Mr', str(Mr),
                         '.Nmock', str(Nmock), '.dat'])

    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov

#function for loading the precomputed RRs

def data_RR(NR):
    '''
    loads precomputed RR
    NR = number of random points
    it should probably also accept the size of the box.
    note to self: put the size of the box in there later after 
    merging the sample variance with the rest of the thing
    '''

    rr = ''.join([util.dat_dir(),
		 'RR_NR' , str(NR) , '.dat'])

    rr = np.loadtxt(rr)

    return rr


def build_data_rr(nr):

from halotools.sim_manager import CachedHaloCatalog
halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0)
num_randoms = 1e5
xran = np.random.uniform(xmin, xmax, num_randoms)
yran = np.random.uniform(ymin, ymax, num_randoms)
zran = np.random.uniform(zmin, zmax, num_randoms)
randoms = np.vstack((xran, yran, zran)).T
Lbox = halocat.Lbox

# Build observables ---------------
def build_xi_nbar_gmf(Mr=21):
    '''
    Build "data" xi, nbar, GMF values and write to file
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    model.populate_mock(halocat = halocat , enforce_PBC = False) # population mock realization
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')

    # write xi
    rbins = hardcoded_xi_bins()
    rmax = rbins.max()
    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]
    period = np.array([Lbox , Lbox , Lbox]) 
    data_xir = tpcf(
            sample1, rbins, sample2 = sample2, 
            randoms=randoms, period = period, 
            max_sample_size=int(1e4), estimator='Landy-Szalay', 
            approx_cell1_size=approx_cell1_size, 
            approx_cellran_size=approx_cellran_size, 
            RR_precomputed = RR, 
            NR_precomputed = NR1) 
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

def build_xi_bins(Mr=21):
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

def build_xi_nbar_gmf_cov(Mr=21, Nmock=500):
    '''
    Build covariance matrix for xi, variance for nbar, and a bunch of stuff for gmf
    ...
    using Nmock realizations of halotool mocks
    '''
    xir = []
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    #nbars = []
    #gmfs = []
    #gmf_counts = []
    for i in xrange(Nmock):
        print 'mock#', i
        model.populate_mock()

        # xi(r)
        xir.append(model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[1])
        # nbar
        #nbars.append(model.mock.number_density)
        # gmf
        #rich = richness(model.mock.compute_fof_group_ids())
        #gmfs.append(GMF(rich))  # GMF
        #gmf_counts.append(GMF(rich, counts=True))   # Group counts

    # save xi covariance
    xi_covar = np.cov(np.array(xir).T)
    output_file = ''.join([util.dat_dir(), 'xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(output_file, xi_covar)

    # save nbar values
    #nbar_cov = np.var(nbars, axis=0)
    #output_file = ''.join([util.dat_dir(), 'nbar_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(output_file, [nbar_cov])

    # write GMF covariance
    #gmf_cov = np.cov(np.array(gmfs).T)
    #output_file = ''.join([util.dat_dir(), 'gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(output_file, gmf_cov)
    # write GMF Poisson
    #gmf_counts_mean = np.mean(gmf_counts, axis=0)
    #poisson_gmf = np.sqrt(gmf_counts_mean) / 250.**3    # poisson errors
    #output_file = ''.join([util.dat_dir(), 'gmf_sigma_poisson.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(output_file, poisson_gmf)
    # write GMF standard dev
    #sigma_gmf = np.std(gmfs, axis=0)                    # sample variance
    #output_file = ''.join([util.dat_dir(), 'gmf_sigma_stddev.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(output_file, sigma_gmf)
    # write GMF total noise
    #sigma_tot = (sigma_gmf**2 + poisson_gmf**2)**0.5    # total sigma
    #output_file = ''.join([util.dat_dir(), 'gmf_sigma.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(output_file, sigma_tot)

    # write full covariance matrix of various combinations of the data

    # covariance for all three
    #fulldatarr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
    #                       np.array(gmfs), np.array(xir))

    #fullcov = np.cov(fulldatarr.T)
    #outfn = ''.join([util.dat_dir(), 'nbar_gmf_xir_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(outfn, fullcov)

    # covariance for nbar and gmf
    ##nbgmf_arr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
    #                     np.array(gmfs))
    #nbgmf_cov = np.cov(nbgmf_arr.T)
    #outfn = ''.join([util.dat_dir(), 'nbar_gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(outfn, nbgmf_cov)

    # covariance for nbar and xi
    #nbxi_arr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
    #                     np.array(xir))
    #nbxi_cov = np.cov(nbxi_arr.T)
    #outfn = ''.join([util.dat_dir(), 'nbar_xi_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(outfn, nbxi_cov)

    # covariance for gmf and xi
    #gmfxi_arr = np.hstack(np.array(gmfs), np.array(xir))
    #gmfxi_cov = np.cov(gmfxi_arr.T)
    #outfn = ''.join([util.dat_dir(), 'gmf_xi_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    #np.savetxt(outfn, gmfxi_cov)

    return None


def build_xi_inv_cov(Mr=21, Nmock=500, unbias=True):
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


def build_full_inv_covars(Mr=21, Nmock=50):
    '''
    Calculate the inverse covariance of full data vectors
    for MCMC inference
    '''

    full_cov = data_full_cov(Mr=Mr, Nmock=Nmock)
    N_bins = int(np.sqrt(full_cov.size))
    f_unbias = np.float(Nmock - 2. - N_bins)/np.float(Nmock - 1.)
    inv_c = solve(np.eye(N_bins) , full_cov) * f_unbias

    outfn = ''.join([util.dat_dir(),
                     'nbar_gmf_xir_inv_cov.Mr', str(Mr),
                     '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, inv_c)

    nbgmf_cov = data_nbar_gmf_cov(Mr=Mr, Nmock=Nmock)
    N_bins = int(np.sqrt(nbgmf_cov.size))
    f_unbias = np.float(Nmock - 2. - N_bins)/np.float(Nmock - 1.)
    inv_c = solve(np.eye(N_bins) , nbgmf_cov) * f_unbias

    outfn = ''.join([util.dat_dir(),
                     'nbar_gmf_inv_cov.Mr', str(Mr),
                     '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, inv_c)

    nbxi_cov = data_nbar_xi_cov(Mr=Mr, Nmock=Nmock)
    N_bins = int(np.sqrt(nbxi_cov.size))
    f_unbias = np.float(Nmock - 2. - N_bins)/np.float(Nmock - 1.)
    inv_c = solve(np.eye(N_bins) , nbxi_cov) * f_unbias

    outfn = ''.join([util.dat_dir(),
                     'nbar_xi_inv_cov.Mr', str(Mr),
                     '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, inv_c)

    gmfxi_cov = data_gmf_xi_cov(Mr=Mr, Nmock=Nmock)
    N_bins = int(np.sqrt(gmfxi_cov.size))
    f_unbias = np.float(Nmock - 2. - N_bins)/np.float(Nmock - 1.)
    inv_c = solve(np.eye(N_bins) , gmfxi_cov) * f_unbias

    outfn = ''.join([util.dat_dir(),
                     'gmf_xi_inv_cov.Mr', str(Mr),
                     '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, inv_c)

    return None


def build_observations(Mr=21, Nmock=200):
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
    print 'Building the rest of the inverse covariances...'
    #build_full_inv_covars(Mr=Mr, Nmock=Nmock)


if __name__=='__main__':
    build_observations()
