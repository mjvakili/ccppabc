'''


module for building and loading all the possible data and covariance
combinations that can come up in the generalised inference



'''
#general python modules
import numpy as np
from multiprocessing import cpu_count
from numpy.linalg import solve

#haltools functions
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.mock_observables import FoFGroups
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables.pair_counters import npairs

#our ccppabc functions
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF


def cov2corr(mat):
    ''' @MJ: What is this code for? 
    '''
    numat = np.empty(mat.shape)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            numat[i, j] = mat[i, j] / (mat[i, i] * mat[j, j])

    return numat

def data_hod_param(Mr=21):
    ''' HOD parameters of 'observations'. Returns dictionary with hod parameters.
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict

## UTILITY FUNCTIONS FOR LOADING OBSERVABLES, COV and INVERSECOV MATRICES ##

def data_nbar(Mr=21):
    ''' Load the observed nbar from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    nbar = dat[0]
    #nbar_var = np.loadtxt(''.join([util.multidat_dir(),
    #                              'nbar_var.Mr', str(Mr),
    #                              '.dat']))
    #reutrn [nbar, nbar_var]
    return nbar

def data_xi(Mr=21):
    ''' Load the observed xi (2PCF) from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(), 'data_vector.Mr', str(Mr), '.dat'])) 
    xi = dat[1:16]
    return xi

def data_gmf(Mr=21):
    ''' Load the observed GMF from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    gmf = dat[16:]
    return gmf

def data_inv_cov(datcombo, Mr=21):
    ''' Load the inverse covariance matrix for given 'datcombo'
    
    Parameters
    ----------
    datcombo : (string)
        One of the following strings to specify the observables: 
        'gmf', 'nbar_gmf', 'nbar_xi', 'nbar_xi_gmf', 'xi' 
    '''
    inv_cov_fn = ''.join([util.multidat_dir(),
                         '{0}_inv_cov.Mr'.format(datcombo), str(Mr),
                         '.dat'])
    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov

def data_cov(Mr=21):
    ''' Load the entire covariance matrix 
    '''
    cov_fn = ''.join([util.multidat_dir(),
                         'nbar_xi_gmf_cov.Mr', str(Mr),
                         '.dat'])
    cov = np.loadtxt(cov_fn)
    return cov

def data_random():
    ''' Load pregenerated random points
    '''
    random_file = ''.join([util.multidat_dir(),
                         'randoms',
                         '.dat'])
    randoms = np.loadtxt(random_file)

    return randoms


def data_RR():
    ''' Load precomputed RR pairs
    '''
    RR_file = ''.join([util.multidat_dir(), 'RR', '.dat'])
    RR = np.loadtxt(RR_file)
    return RR

def data_gmf_bins():
    ''' Just for consistency, returns the bins
    '''
    return gmf_bins()

def hardcoded_xi_bins():
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
    rbins = hardcoded_xi_bins()
    rbin = .5 * (rbins[1:] + rbins[:-1])
    output_file = ''.join([util.multidat_dir(),'xir_rbin.Mr', str(Mr),'.dat'])
    np.savetxt(output_file, rbin)
    return None

#Build Randoms and precomputed RRs

def build_randoms_RR(Nr=5e5):
    '''
    builds Nr number of random points using numpy.random.uniform 
    in a subvolume and precomputes the RR.

    The set of randoms and  precomputed RR will be used to compute tpcf
    with Landay-Szalay estimator in each subvolume. 
    '''

    L = 200 #this is L_md / 5   
 
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
    rbins = hardcoded_xi_bins()
    rmax = rbins.max()
    approx_cellran_size = [rmax, rmax, rmax]
    
   
    
    RR = npairs(
            randoms, randoms, rbins, period,
            verbose, num_threads,
            approx_cellran_size, approx_cellran_size)
    RR = np.diff(RR)
  

    output_file_random = ''.join([util.multidat_dir(),'randoms','.dat'])
    np.savetxt(output_file_random, randoms)
    

    output_file_RR = ''.join([util.multidat_dir(),'RR','.dat'])
    np.savetxt(output_file_RR, RR)

    return None

# Build observables ---------------
def build_nbar_xi_gmf(Mr=21):
    ''' Build data vector [nbar, xi, gmf] and save to file 
    This data vector is built from the zeroth slice of the multidark
    The other slices will be used for building the covariance matrix.
    
    Note to self : need to shift the positions of random points
    '''
    
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(simname='multidark',
                        masking_function=datsubvol,
                        enforce_PBC=False)
    
    #all the things necessary for tpcf calculation

 
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
    rbins = hardcoded_xi_bins()
    rmax = rbins.max()
    period = None
    approx_cell1_size = [rmax , rmax , rmax]
    approx_cellran_size = [rmax , rmax , rmax]

    #compute number density    

    nbar = len(pos) / 200**3.

    #load randoms and RRs
    
    randoms = data_random()
    RR = data_RR()
    NR = len(randoms)
    
    #compue tpcf with Landy-Szalay estimator

    data_xir = tpcf(
            pos, rbins, pos, 
            randoms=randoms, period = period, 
            max_sample_size=int(1e5), estimator='Landy-Szalay', 
            approx_cell1_size=approx_cell1_size, 
            approx_cellran_size=approx_cellran_size, 
            RR_precomputed = RR, 
            NR_precomputed = NR)


    fullvec = np.append(nbar, data_xir)

    #compute group richness    

    galaxy_sample = model.mock.galaxy_table
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vz = galaxy_sample['vz']
    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z'
                               , velocity = vz , velocity_distortion_dimension="z")
    b_para, b_perp = 0.7, 0.15
    groups = FoFGroups(pos, b_perp, b_para, period = None, 
                      Lbox = 200 , num_threads='max')
    gids = groups.group_ids
    rich = richness(gids)

    gmf = GMF(rich)  # GMF

    fullvec = np.append(fullvec, gmf)

    output_file = ''.join([util.multidat_dir(), 'data_vector.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, fullvec)

    return None


# --- function to build all the covariance matrices and their inverses ---
def build_nbar_xi_gmf_cov(Mr=21):
    ''' Build covariance matrix for the full nbar, xi, gmf data vector
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
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
    
    #some settings for tpcf calculations
    rbins = hardcoded_xi_bins()
    rmax = rbins.max()
    period = None
    approx_cell1_size = [rmax , rmax , rmax]
    approx_cellran_size = [rmax , rmax , rmax]
    
    #load randoms and RRs
    
    randoms = data_random()
    RR = data_RR()
    NR = len(randoms)

    for i in xrange(1,125):
        print 'mock#', i

        # populate the mock subvolume
        mocksubvol = lambda x: util.mask_func(x, i)
        model.populate_mock(simname='multidark',
                            masking_function=mocksubvol,
                            enforce_PBC=False)
        # returning the positions of galaxies
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
        # calculate nbar
        nbars.append(len(pos) / 200**3.)
        # translate the positions of randoms to the new subbox
        xi , yi , zi = util.random_shifter(i)
        temp_randoms = randoms.copy()
        temp_randoms[:,0] += xi
        temp_randoms[:,1] += yi
        temp_randoms[:,2] += zi
        #calculate xi(r)        
        xi = tpcf(
            pos, rbins, pos, 
            randoms=temp_randoms, period = period, 
            max_sample_size=int(1e5), estimator='Landy-Szalay', 
            approx_cell1_size=approx_cell1_size, 
            approx_cellran_size=approx_cellran_size,
            RR_precomputed = RR,
	    NR_precomputed = NR)
        xir.append(xi)
        # calculate gmf
        galaxy_sample = model.mock.galaxy_table
        x = galaxy_sample['x']
        y = galaxy_sample['y']
        z = galaxy_sample['z']
        vz = galaxy_sample['vz']
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z'
                               , velocity = vz , velocity_distortion_dimension="z")
        b_para, b_perp = 0.7, 0.15
        groups = FoFGroups(pos, b_perp, b_para, period = None, 
                      Lbox = 200 , num_threads='max')
        gids = groups.group_ids
        rich = richness(gids)
        gmfs.append(GMF(rich))  # GMF
        gmf_counts.append(GMF(rich, counts=True))   # Group counts
    # save nbar variance
    nbar_var = np.var(nbars, axis=0, ddof=1)
    nbar_file = ''.join([util.multidat_dir(), 'nbar_var.Mr', str(Mr), '.dat'])
    np.savetxt(nbar_file, [nbar_var])

    # determine extra poisson noise on gmf
    gmf_counts_mean = np.mean(gmf_counts, axis=0)

    # write full covariance matrix of various combinations of the data
    # and invert for the likelihood evaluations

    # --- covariance for all three ---
    fulldatarr = np.hstack((np.array(nbars).reshape(len(nbars), 1),
                            np.array(xir),
                            np.array(gmfs)))
    fullcov = np.cov(fulldatarr.T)
    fullcorr = np.corrcoef(fulldatarr.T)

    # and save the covariance matrix
    nopoisson_file = ''.join([util.multidat_dir(), 'nbar_xi_gmf_cov.no_poisson.Mr', str(Mr), '.dat'])
    np.savetxt(nopoisson_file, fullcov)

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):
        fullcov[1+len(xi) + i, 1+len(xi)+ i] += gmf_counts_mean[i] / 200**6.
    # save poisson noise for GMF
    gmf_poisson_file = ''.join([util.multidat_dir(),
                    'gmf_poisson.Mr', str(Mr), '.dat'])
    np.savetxt(gmf_poisson_file, gmf_counts_mean)

    # and save the covariance matrix
    full_cov_file = ''.join([util.multidat_dir(), 'nbar_xi_gmf_cov.Mr', str(Mr), '.dat'])
    np.savetxt(full_cov_file, fullcov)
    # and a correlation matrix
    full_corr_file = ''.join([util.multidat_dir(), 'nbar_xi_gmf_corr.Mr', str(Mr), '.dat'])
    np.savetxt(full_corr_file, fullcorr)
    return None


def build_observations(Mr=21):
    '''
    Build all the fake observations
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

    # xi, nbar, gmf
    print 'Building nbar, xi(r), GMF data vector... '
    build_xi_bins(Mr=Mr)
    build_nbar_xi_gmf(Mr=Mr)
    print 'Computing covariance matrix of data...'
    build_nbar_xi_gmf_cov(Mr=Mr)


if __name__ == "__main__":
    #build_randoms_RR(Nr=5e5)
    build_observations(Mr = 21)
