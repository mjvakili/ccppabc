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
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables.pair_counters import npairs

#our ccppabc functions
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF


def cov2corr(mat):

    numat = np.empty(mat.shape)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            numat[i, j] = mat[i, j] / (mat[i, i] * mat[j, j])

    return numat


def data_hod_param(Mr=21):
    '''
    HOD parameters of 'observations'. Returns dictionary with hod parameters.
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict


# ---load  nbar ---
def data_nbar(Mr=21):
    '''
    loads the observed nbar from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    nbar = dat[0]
    nbar_var = np.loadtxt(''.join([util.multidat_dir(),
                                  'nbar_var.Mr', str(Mr),
                                  '.dat']))
    return [nbar, nbar_var]


# --- load 2PCF ---
def data_xi(Mr=21):
    '''
    loads the observed xi (2PCF) from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    xi = dat[1:16]

    return xi


# --- load GMF ---
def data_gmf(Mr=21):
    '''
    loads the observed GMF from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    gmf = dat[16:]

    return gmf


# --- load inverse covariance matrices ---
def data_inv_cov(datcombo, Mr=21):

    '''
    loads the inverse covariance matrix 
    '''

    inv_cov_fn = ''.join([util.multidat_dir(),
                         '{0}.Mr'.format(datcombo), str(Mr),
                         '.dat'])
    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov

#--- load randoms ---
def data_random():

    '''
    loads pregenerated random points
    '''

    random_file = ''.join([util.multidat_dir(),
                         'randoms',
                         '.dat'])
    randoms = np.loadtxt(random_file)

    return randoms


#--- load RR ---
def data_RR():

    '''
    loads precomputed RR
    '''

    RR_file = ''.join([util.multidat_dir(),
                         'RR',
                         '.dat'])
    RR = np.loadtxt(RR_file)

    return RR

# Build bin edges for 2PCF calculations

def hardcoded_xi_bins():
    '''
    loads the hardcoded xi bin edges.They are spaced out unevenly due to sparseness
    at inner r bins. So the first bin ranges from 0.15 to 0.5
    '''
    r_bins = np.concatenate([np.array([0.15]),
                             np.logspace(np.log10(0.5), np.log10(20.), 15)])
    
    #r_bins = np.logspace(-1. , np.log10(17) , 15)
   
    return r_bins

#Build centers of bins for 2PCF calculations

def build_xi_bins(Mr=21):
    '''
    builds and saves the hardcoded r_bin centers for xi.
    '''
    rbins=hardcoded_xi_bins()
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
    '''
    Builds and saves "data" vector <nbar, xi, gmf> and write to file

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

    rich = richness(model.mock.compute_fof_group_ids())
    gmf = GMF(rich)  # GMF

    fullvec = np.append(fullvec, gmf)

    output_file = ''.join([util.multidat_dir(), 'data_vector.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, fullvec)

    return None


# --- function to build all the covariance matrices and their inverses ---

def build_nbar_xi_gmf_cov(Mr=21):
    '''
    Build covariance matrix for nbar, xi, gmf data vector
    using realisations of galaxy mocks for "data" HOD
    parameters in the halos from the other subvolumes
    (subvolume 1 to subvolume 125) of
    the simulation.

    Note to self: need to shift the positions of random points 
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
        rich = richness(model.mock.compute_fof_group_ids())
        gmfs.append(GMF(rich))  # GMF
        gmf_counts.append(GMF(rich, counts=True))   # Group counts

    # save nbar variance

    nbar_var = np.var(nbars, axis=0, ddof=1)
    output_file = ''.join([util.multidat_dir(),
                          'nbar_var.Mr', str(Mr),
                          '.dat'])
    np.savetxt(output_file, [nbar_var])

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

    print len(gmf_counts_mean)
    print fullcov.shape

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):

        fullcov[16 + i, 16 + i] += gmf_counts_mean[i] / 200**6.

    # and save the covariance matrix
    outfn = ''.join([util.multidat_dir(),
                    'nbar_xir_gmf_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, fullcov)
    # and a correlation matrix
    outfn = ''.join([util.multidat_dir(),
                    'nbar_xir_gmf_corr.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, fullcorr)

    # full covariance matrix inverse
    N_bins = len(fullcov[0])

    """
    math:C_{estimate}^-1 = ( (Nmocks - 2 - N_bins) / (Nmock - 1) ) C^-1
    """
    f_unbias = (124 - 2. - N_bins) / (124. - 1) 
    inv_c = solve(np.eye(N_bins) , fullcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_xir_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for nbar-xi data vector covariance
    datarr = np.hstack((np.array(nbars).reshape(len(nbars), 1),
                        np.array(xir)))

    nbxicov = np.cov(datarr.T)

    # and save the nbar-xi covariance matrix
    outfn = ''.join([util.multidat_dir(),
                    'nbar_xir_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, nbxicov)

    # and generate and save a correlation matrix for inspection
    nbxicor = np.corrcoef(datarr.T)

    outfn = ''.join([util.multidat_dir(),
                    'nbar_xi_corr.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, nbxicor)
    # and save the inverse covariance matrix for nbar , xi
    N_bins = int(np.sqrt(nbxicov.size))
    f_unbias = (124 - 2. - N_bins) / (124. - 1)
    inv_c = solve(np.eye(N_bins) , nbxicov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_xi_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for nbar-gmf data vector covariance
    datarr = np.hstack((np.array(nbars).reshape(len(nbars), 1),
                        np.array(gmfs)))

    nbgmfcov = np.cov(datarr.T)
    nbgmfcor = np.corrcoef(datarr.T)

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):
        nbgmfcov[1 + i, 1 + i] += gmf_counts_mean[i] / 200**6.

    outfn = ''.join([util.multidat_dir(),
                    'nbar_gmf_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, nbgmfcov)

    N_bins = int(np.sqrt(nbgmfcov.size))
    f_unbias = (124 - 2. - N_bins) / (124 - 1.)
    inv_c = solve(np.eye(N_bins) , nbgmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for xi-gmf data vector covariance
    datarr = np.hstack((np.array(xir),
                        np.array(gmfs)))

    xigmfcov = np.cov(datarr.T)
    xigmfcor = np.corrcoef(datarr.T)

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):
        xigmfcov[15 + i, 15 + i] += gmf_counts_mean[i] / 200**6.

    outfn = ''.join([util.multidat_dir(),
                    'xi_gmf_corr.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, xigmfcor)

    # save the covariance matrix for gmf and xi
    outfn = ''.join([util.multidat_dir(),
                    'xi_gmf_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, xigmfcov)
   
    # save the inverse covariance matrix fot xi and gmf 
    N_bins = int(np.sqrt(xigmfcov.size))
    f_unbias = (124 - 2. - N_bins) / (124 - 1.)
    inv_c = solve(np.eye(N_bins) , xigmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'xi_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for xi data vector covariance
    xicov = np.cov(np.array(xir).T)
    xicor = np.corrcoef(np.array(xir).T)

    outfn = ''.join([util.multidat_dir(),
                    'xi_corr.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, xicor)

    N_bins = int(np.sqrt(xicov.size))
    f_unbias = (124 - 2. - N_bins) / (124. - 1)
    inv_c = solve(np.eye(N_bins) , xicov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'xi_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for gmf data vector covariance
    gmfcov = np.cov(np.array(gmfs).T)
    gmfcor = np.corrcoef(np.array(gmfs).T)

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):
        gmfcov[i, i] += gmf_counts_mean[i] / 200**6.


    outfn = ''.join([util.multidat_dir(),
                    'gmf_corr.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, gmfcor)

    N_bins = int(np.sqrt(gmfcov.size))
    f_unbias = (124 - 2. - N_bins) / (124. - 1)
    inv_c = solve(np.eye(N_bins) , gmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

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
