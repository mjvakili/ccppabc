'''

module for building and loading all the possible data and covariance
combinations that can come up in the generalised inference

'''
#general python modules
import numpy as np
from multiprocessing import cpu_count
from numpy.linalg import solve
import pyfof

#haltools functions
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.mock_observables import FoFGroups
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables.pair_counters import npairs_3d

#our ccppabc functions
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF


def data_hod_param(Mr=21):
    ''' HOD parameters of 'observations'. Returns dictionary with hod parameters.
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict

## UTILITY FUNCTIONS FOR LOADING OBSERVABLES, COV and INVERSECOV MATRICES ##
def data_file(Mr=21, b_normal=0.25): 
    ''' Data vector file 
    
    Parameters
    ----------
    Mr : (int)
        Absolute magnitude M_r cutoff. Default is M_r = -21

    b_normal : (float) 
        FoF linking length parameter used for the GMF.  
    '''
    data_file = ''.join([util.obvs_dir(),
        'data_vector', 
        '.Mr', str(Mr),
        '.bnorm', str(round(b_normal, 2)), 
        '.dat'])
    return data_file 

def data_nbar(Mr=21, b_normal=0.25):
    ''' Load the observed nbar from the data vector file
    '''
    dat = np.loadtxt(data_file(Mr=Mr, b_normal=b_normal))
    nbar = dat[0]
    return nbar

def data_xi(Mr=21, b_normal=0.25):
    ''' Load the observed xi (2PCF) from data vector file
    '''
    dat = np.loadtxt(data_file(Mr=Mr, b_normal=b_normal))
    xi = dat[1:16]
    return xi

def data_xi_bin(Mr=21): 
    ''' Load the r bins of the xi measurements. These are _not_ the bin edges, but 
    the center of the bins. 
    '''
    xi_bin_file = ''.join([util.obvs_dir(),'xir_rbin.Mr', str(Mr),'.dat'])
    return np.loadtxt(xi_bin_file) 

def data_gmf(Mr=21, b_normal=0.25):
    ''' Load the observed GMF from data vector file 
    '''
    dat = np.loadtxt(data_file(Mr=Mr, b_normal=b_normal))
    gmf = dat[16:]
    return gmf

def data_cov(Mr=21, b_normal=0.25, inference='abc'):
    ''' Load the entire covariance matrix with sample variance,
        Note: this is for MCMC
    '''
    if inference == 'abc': 
        inf_str = 'ABC'
    elif inference == 'mcmc': 
        inf_str = 'MCMC'

    cov_fn = ''.join([util.obvs_dir(),
        inf_str, 
        '.nbar_xi_gmf_cov', 
        '.no_poisson', 
        '.Mr', str(Mr),
        '.bnorm', str(round(b_normal, 2)), 
        '.dat'])
    cov = np.loadtxt(cov_fn)
    return cov

def data_random(box='md_sub'):
    ''' Load pregenerated random points
    '''
    random_file = ''.join([util.obvs_dir(), 'randoms', '.', box, '.dat'])
    randoms = np.loadtxt(random_file)
    return randoms

def data_RR(box='md_sub'):
    ''' Load precomputed RR pairs
    '''
    RR_file = ''.join([util.obvs_dir(), 'RR', '.', box, '.dat'])
    RR = np.loadtxt(RR_file)
    return RR

def data_gmf_bins():
    ''' Just for consistency, returns the bins
    '''
    return gmf_bins()

def xi_binedges():
    ''' Load hardcoded xi r-bin edges. They are spaced out unevenly due to sparseness
    at inner r bins. So the first bin ranges from 0.15 to 0.5
    '''
    r_bins = np.concatenate([np.array([0.15]),
                             np.logspace(np.log10(0.5), np.log10(20.), 15)])
    #r_bins = np.logspace(-1. , np.log10(17) , 15)
    return r_bins

#Build centers of bins for 2PCF calculations
def build_xi_bins(Mr=21):
    ''' Build hardcoded r_bin centers for xi and then save to file.
    '''
    rbins = xi_binedges() 
    rbin = .5 * (rbins[1:] + rbins[:-1])
    output_file = ''.join([util.obvs_dir(),'xir_rbin.Mr', str(Mr),'.dat'])
    np.savetxt(output_file, rbin)
    return None

#Build Randoms and precomputed RRs

def build_randoms_RR(Nr=1e6, box='md_sub'):
    '''
    builds Nr number of random points using numpy.random.uniform 
    in a subvolume and precomputes the RR.

    The set of randoms and  precomputed RR will be used to compute tpcf
    with Landay-Szalay estimator in each subvolume. 

    Parameters
    ----------
    Nr : (int) 
        Number of random points

    box : (string) 
        'md_sub' specifies multidark subvolume box (L_box = 200). 
        'md_all' specifies multidark entire box (L_box = 1000). 
    '''
    if box == 'md_sub':  
        L = 200 #this is L_md / 5   
        if Nr < 5e5: 
            raise ValueError('Too few number of randoms') 
    elif box == 'md_all': 
        L = 1000
        if Nr < 2.5e6: 
            raise ValueError('Too few number of randoms') 
 
    xmin , ymin , zmin = 0., 0., 0.
    xmax , ymax , zmax = L, L, L
    
    num_randoms = Nr
    xran = np.random.uniform(xmin, xmax, num_randoms)
    yran = np.random.uniform(ymin, ymax, num_randoms)
    zran = np.random.uniform(zmin, zmax, num_randoms)
    randoms = np.vstack((xran, yran, zran)).T
    verbose = False
    period = None
    num_threads = cpu_count()
    rbins = xi_binedges()
    rmax = rbins.max()
    approx_cellran_size = [rmax, rmax, rmax]
    
    RR = npairs_3d(
            randoms, randoms, rbins, period,
            verbose, num_threads,
            approx_cellran_size, approx_cellran_size)
    RR = np.diff(RR)
  
    output_file_random = ''.join([util.obvs_dir(),
        'randoms', '.', box, '.dat'])
    np.savetxt(output_file_random, 
            randoms)
    
    output_file_RR = ''.join([util.obvs_dir(),
        'RR', '.', box, '.dat'])
    np.savetxt(output_file_RR, RR)
    return None


# Build observables ---------------
def build_nbar_xi_gmf(Mr=21, b_normal=0.25):
    ''' Build data vector [nbar, xi, gmf] and save to file 
    This data vector is built from the zeroth slice of the multidark
    The other slices will be used for building the covariance matrix.

    Parameters
    ----------
    Mr : (int) 
        Absolute magnitude cut off M_r. Default M_r = -21.

    b_normal : (float) 
        FoF Linking length
    '''
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr)
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    ####model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    ####datsubvol = lambda x: util.mask_func(x, 0)
    ####model.populate_mock(halocat, masking_function=datsubvol, enforce_PBC=False)
    model.populate_mock(halocat)
    
    #all the things necessary for tpcf calculation
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
    #masking the galaxies outside the subvolume 0
    pos = util.mask_galaxy_table(pos , 0)
    rbins = xi_binedges()
    rmax = rbins.max()
    approx_cell1_size = [rmax , rmax , rmax]
    approx_cellran_size = [rmax , rmax , rmax]

    #compute number density    
    nbar = len(pos) / 200**3.
    
    # load MD subvolume randoms and RRs
    randoms = data_random(box='md_sub')
    RR = data_RR(box='md_sub')
    NR = len(randoms)
 
    #compue tpcf with Natural estimator
    data_xir = tpcf(
                 pos, rbins, pos, 
                 randoms=randoms, period=None, 
                 max_sample_size=int(2e5), estimator='Natural', 
                 approx_cell1_size=approx_cell1_size, 
                 approx_cellran_size=approx_cellran_size, 
                 RR_precomputed=RR, 
                 NR_precomputed=NR)

    fullvec = np.append(nbar, data_xir)
    
    #compute gmf
    b = b_normal * (nbar)**(-1./3) 
    groups = pyfof.friends_of_friends(pos , b)
    w = np.array([len(x) for x in groups])
    gbins = gmf_bins()
    gmf = np.histogram(w , gbins)[0] / (200.**3.)
    fullvec = np.append(fullvec, gmf)

    output_file = data_file(Mr=Mr, b_normal=b_normal)
    np.savetxt(output_file, fullvec)
    return None

# --- function to build all the covariance matrices and their inverses ---
def build_MCMC_cov_nbar_xi_gmf(Mr=21, b_normal=0.25):
    ''' Build covariance matrix used in MCMC for the full nbar, xi, gmf data vector
    using realisations of galaxy mocks for "data" HOD parameters in the 
    halos from the other subvolumes (subvolume 1 to subvolume 125) of
    the simulation. Covariance matrices for different sets of observables
    can be extracted from the full covariance matrix by slicing through 
    the indices. 

    '''
    nbars = []
    xir = []
    gmfs = []
    gmf_counts = []

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr)
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    ###model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
    
    #some settings for tpcf calculations
    rbins = xi_binedges()
    rmax = rbins.max()
    approx_cell1_size = [rmax , rmax , rmax]
    approx_cellran_size = [rmax , rmax , rmax]
    
    #load randoms and RRs
    
    randoms = data_random(box='md_sub')
    RR = data_RR(box='md_sub')
    NR = len(randoms)

    for i in xrange(1,125):
        print 'mock#', i

        # populate the mock subvolume
        ###mocksubvol = lambda x: util.mask_func(x, i)
        ###model.populate_mock(halocat,
        ###                    masking_function=mocksubvol,
        ###                    enforce_PBC=False)
        model.populate_mock(halocat)
        # returning the positions of galaxies in the entire volume
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
        # masking out the galaxies outside the subvolume i
        pos = util.mask_galaxy_table(pos , i)
        # calculate nbar
        
        nbars.append(len(pos) / 200**3.)
        # translate the positions of randoms to the new subbox
        xi , yi , zi = util.random_shifter(i)
        temp_randoms = randoms.copy()
        temp_randoms[:,0] += xi
        temp_randoms[:,1] += yi
        temp_randoms[:,2] += zi
        #calculate xi(r)        
        xi=tpcf(
             pos, rbins, pos, 
             randoms=temp_randoms, period=None, 
             max_sample_size=int(3e5), estimator='Natural', 
             approx_cell1_size=approx_cell1_size, 
             approx_cellran_size=approx_cellran_size,
             RR_precomputed = RR,
	     NR_precomputed = NR)
        xir.append(xi)
        # calculate gmf

        nbar = len(pos) / 200**3.
        b = b_normal * (nbar)**(-1./3) 
        groups = pyfof.friends_of_friends(pos , b)
    	w = np.array([len(x) for x in groups])
    	gbins = gmf_bins()
    	gmf = np.histogram(w , gbins)[0] / 200.**3.
        gmfs.append(gmf)

    # save nbar variance
    nbar_var = np.var(nbars, axis=0, ddof=1)
    nbar_file = ''.join([util.obvs_dir(), 'nbar_var.Mr', str(Mr), '.dat'])
    np.savetxt(nbar_file, [nbar_var])

    # write full covariance matrix of various combinations of the data
    # and invert for the likelihood evaluations

    # --- covariance for all three ---
    fulldatarr = np.hstack((np.array(nbars).reshape(len(nbars), 1),
                            np.array(xir),
                            np.array(gmfs)))
    fullcov = np.cov(fulldatarr.T)
    fullcorr = np.corrcoef(fulldatarr.T)

    # and save the covariance matrix
    nopoisson_file = ''.join([util.obvs_dir(),
        'MCMC.nbar_xi_gmf_cov', '.no_poisson', '.Mr', str(Mr), '.bnorm', str(round(b_normal,2)), '.dat'])
    np.savetxt(nopoisson_file, fullcov)
    return None

def build_ABC_cov_nbar_xi_gmf(Mr=21, b_normal=0.25):
    ''' Build covariance matrix used in ABC for the full nbar, xi, gmf data vector
    using realisations of galaxy mocks for "data" HOD parameters in the 
    halos from the multidark simulation. Covariance matrices for different sets of observables
    can be extracted from the full covariance matrix by slicing through 
    the indices. 

    Notes 
    -----
    * This covariance matrix is the covariance matrix calculated from the *entire* multidark 
        box. So this does _not_ account for the sample variance, which the MCMC covariance does. 
    '''
    nbars, xir, gmfs = [], [], [] 

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr)
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    rbins = xi_binedges()  # some setting for tpcf calculations

    rmax = rbins.max()
    approx_cell1_size = [rmax , rmax , rmax]
    approx_cellran_size = [rmax , rmax , rmax]
    
    # load randoms and RRs for the ENTIRE MultiDark volume 
    randoms = data_random(box='md_all')
    RR = data_RR(box='md_all')
    NR = len(randoms)

    for i in xrange(1,125):
        print 'mock#', i
        # populate the mock subvolume
        model.populate_mock(halocat)
        # returning the positions of galaxies
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')

        # calculate nbar
        nbars.append(len(pos) / 1000**3.)

        # calculate xi(r) for the ENTIRE MultiDark volume 
        # using the natural estimator DD/RR - 1
        xi = tpcf(
                pos, rbins, pos, 
                randoms=randoms, period=None, 
                max_sample_size=int(3e5), estimator='Natural', 
                approx_cell1_size=approx_cell1_size, 
                approx_cellran_size=approx_cellran_size,
                RR_precomputed = RR,
                NR_precomputed = NR)
        xir.append(xi)

        # calculate gmf
        nbar = len(pos) / 1000**3.
        b = b_normal * (nbar)**(-1./3) 
        groups = pyfof.friends_of_friends(pos , b)
        w = np.array([len(x) for x in groups])
        gbins = gmf_bins()
        gmf = np.histogram(w , gbins)[0] / (1000.**3.)
        gmfs.append(gmf)  # GMF

    # save nbar variance
    nbar_var = np.var(nbars, axis=0, ddof=1)
    nbar_file = ''.join([util.obvs_dir(), 'abc_nbar_var.Mr', str(Mr), '.dat'])
    np.savetxt(nbar_file, [nbar_var])

    # write full covariance matrix of various combinations of the data
    # and invert for the likelihood evaluations

    # --- covariance for all three ---
    fulldatarr = np.hstack((np.array(nbars).reshape(len(nbars), 1),
                            np.array(xir),
                            np.array(gmfs)))
    fullcov = np.cov(fulldatarr.T)
    fullcorr = np.corrcoef(fulldatarr.T)
    # and save the covariance matrix
    nopoisson_file = ''.join([util.obvs_dir(),
        'ABC.nbar_xi_gmf_cov', '.no_poisson', '.Mr', str(Mr), '.bnorm', str(round(b_normal, 2)), '.dat'])
    np.savetxt(nopoisson_file, fullcov)
    return None

def build_observations(Mr=21, b_normal=0.25, make=['data', 'covariance']):
    ''' Wrapper to build all the required fake observations and their
    corresponding covariance matrices. 
    '''
    # download the Multidark halo catalog if necessary
    try:
        halocat = CachedHaloCatalog(simname='multidark',
                                    halo_finder='rockstar',
                                    redshift=0.0)
    except InvalidCacheLogEntry:
        from halotools.sim_manager import DownloadManager
        dman = DownloadManager()
        dman.download_processed_halo_table('multidark', 'rockstar', 0.0)
    
    if 'data' in make: 
        # xi, nbar, gmf
        print 'Building nbar, xi(r), GMF data vector... '
        build_xi_bins(Mr=Mr)
        print 'Building randoms and RRs for the subvolumes'
        build_randoms_RR(Nr=5e5, box='md_sub')
        
        build_nbar_xi_gmf(Mr=Mr, b_normal=b_normal)
    
    if 'covariance' in make:
        print 'Computing covariance matrix of data...'
        build_MCMC_cov_nbar_xi_gmf(Mr=Mr, b_normal=b_normal)
        build_ABC_cov_nbar_xi_gmf(Mr=Mr, b_normal=b_normal)

    return None


if __name__ == "__main__":
    build_observations(Mr = 21, b_normal=0.25)
