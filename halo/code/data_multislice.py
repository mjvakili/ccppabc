'''
module for building and loading all the possible data and covariance
combinations that can come up in the generalised inference
'''
import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF


def data_hod_param(Mr=20):
    '''
    HOD parameters of 'observations'. Returns dictionary with hod parameters.
    '''
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))
    return model.param_dict


# --- nbar ---
def data_nbar(Mr=20):
    '''
    Observed nbar from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    nbar = dat[0]
    nbar_var = np.loadtxt(''.join([util.multidat_dir(),
                                  'nbar_var.Mr', str(Mr),
                                  '.dat'])
    return [nbar, nbar_var]


# --- 2PCF ---
def data_xi(Mr=20):
    '''
    Observed xi (2PCF) from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    xi = dat[1:17]

    return xi


# --- GMF ---
def data_gmf(Mr=20):
    '''
    Observed GMF from 'data'
    '''
    dat = np.loadtxt(''.join([util.multidat_dir(),
                             'data_vector.Mr', str(Mr),
                             '.dat']))
    gmf = dat[17:]

    return gmf


# --- inverse covariance matrices ---
def data_inv_cov(datcombo, Mr=20):

    inv_cov_fn = ''.join([util.multidat_dir(),
                         '{0}.Mr'.format(datcombo), str(Mr),
                         '.dat'])
    inv_cov = np.loadtxt(inv_cov_fn)

    return inv_cov


# --- functions to build initial fake data ---
def hardcoded_xi_bins():
    '''
    hardcoded xi bin edges.They are spaced out unevenly due to sparseness
    at inner r bins. So the first bin ranges from 0.15 to 0.5
    '''
    r_bins = np.concatenate([np.array([0.15]),
                             np.logspace(np.log10(0.5), np.log10(20.), 15)])
    return r_bins


def build_xi_bins(Mr=20):
    '''
    hardcoded r bins for xi.
    '''
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(masking_function=datsubvol, enforce_PBC=False)
    r_bin  = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[0]
    output_file = ''.join([util.multidat_dir(), 'xir_rbin.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, r_bin)
    return None


# Build observables ---------------
def build_nbar_xi_gmf(Mr=20):
    '''
    Build "data" vector <nbar, xi, gmf> and write to file
    '''
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(masking_function=datsubvol, enforce_PBC=False)

    nbar = model.mock.number_density

    data_xir = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[1]

    fullvec = np.append(nbar, data_xir)

    rich = richness(model.mock.compute_fof_group_ids())
    gmf = GMF(rich)  # GMF

    fullvec = np.append(fullvec, gmf)

    output_file = ''.join([util.multidat_dir(), 'data_vector.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, fullvec)

    return None


# --- function to build all the covariance matrices and their inverses ---
def build_nbar_xi_gmf_cov(Mr=20):
    '''
    Build covariance matrix for nbar, xi, gmf data vector
    using realisations of galaxy mocks for "data" HOD
    parameters in the halos from the other subvolumes of
    the simulation.
    '''
    nbars = []
    xir = []
    gmfs = []
    gmf_counts = []

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    for i in xrange(1,125):
        print 'mock#', i

        # populate the mock subvolume
        mocksubvol = lambda x: util.mask_func(x, i)
        model.populate_mock(masking_function=mocksubvol, enforce_PBC=False)

        # nbar
        nbars.append(model.mock.number_density)
        # xi(r)
        xir.append(model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[1])
        # gmf
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
    fulldatarr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
                           np.array(xir),
                           np.array(gmfs))

    fullcov = np.cov(fulldatarr.T)
    fullcorr = np.corrcoeff(fulldatarr.T)

    # add in poisson noise for gmf
    for i in range(len(gmf_counts_mean)):
        fullcov[17 + i, 17 + i] += gmf_counts_mean[i]

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
    N_bins = int(np.sqrt(fullcov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , fullcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_xir_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for nbar-xi data vector covariance
    nbxicov = fullcov[:17, :17]

    N_bins = int(np.sqrt(nbxicov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , nbxicov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_xi_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for nbar-gmf data vector covariance
    nbgmfcov = np.empty((fullcov.shape[0] - 16, fullcov.shape[0] - 16))
    nbgmfcov[1:,1:] = fullcov[17:, 17:]
    nbgmfcov[:, 0] = nbgmfcov[0, :] = np.append(fullcov[0, 0],
                                                fullcov[0, 17:])

    N_bins = int(np.sqrt(nbgmfcov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , nbgmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'nbar_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for xi-gmf data vector covariance
    xigmfcov = fullcov[1:, 1:]

    N_bins = int(np.sqrt(xigmfcov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , xigmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'xi_gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for xi data vector covariance
    xicov = fullcov[1:17, 1:17]

    N_bins = int(np.sqrt(xicov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , xicov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'xi_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    # inverse for gmf data vector covariance
    gmfcov = fullcov[17:, 17:]

    N_bins = int(np.sqrt(gmfcov.size))
    f_unbias = (124 - 2. - N_bins) / 124.
    inv_c = solve(np.eye(N_bins) , gmfcov) * f_unbias

    outfn = ''.join([util.multidat_dir(),
                    'gmf_inv_cov.Mr', str(Mr),
                    '.dat'])
    np.savetxt(outfn, inv_c)

    return None


def build_observations(Mr=20):
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
    build_nbar_xi_gmf_cov(Mr=Mr):


if __name__ == "__main__":

    build_observations()
