import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.mock_observables import FoFGroups
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF
from data_multislice import hardcoded_xi_bins

def build_nbar_xi_gmf_cov(Mr=21):
    ''' Build covariance matrix for the full nbar, xi, gmf data vector
    using realisations of galaxy mocks for "data" HOD parameters in the 
    halos from the multidark simulation. Covariance matrices for different sets of observables
    can be extracted from the full covariance matrix by slicing through 
    the indices. 

    '''
    nbars = []
    xir = []
    gmfs = []

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    #some settings for tpcf calculations
    rbins = hardcoded_xi_bins()

    for i in xrange(1,125):
        print 'mock#', i

        # populate the mock subvolume
        model.populate_mock(simname='multidark')
        # returning the positions of galaxies
        pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')
        # calculate nbar
        nbars.append(len(pos) / 1000**3.)
        # translate the positions of randoms to the new subbox
        #calculate xi(r)        
        xi = tpcf(pos, rbins, period = model.mock.Lbox, 
                  max_sample_size=int(2e5), estimator='Landy-Szalay')
        xir.append(xi)
        # calculate gmf
        galaxy_sample = model.mock.galaxy_table
        x = galaxy_sample['x']
        y = galaxy_sample['y']
        z = galaxy_sample['z']
        vz = galaxy_sample['vz']
        pos_rsd = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z'
                               , velocity = vz , velocity_distortion_dimension="z")
        b_para, b_perp = 0.5, 0.2
        groups = FoFGroups(pos_rsd, b_perp, b_para, Lbox = model.mock.Lbox, num_threads=1)
        gids = groups.group_ids
        rich = richness(gids)
        gmfs.append(GMF(rich)*(200**3.)/(1000.**3.))  # GMF
    # save nbar variance
    nbar_var = np.var(nbars, axis=0, ddof=1)
    nbar_file = ''.join([util.multidat_dir(), 'abc_nbar_var.Mr', str(Mr), '.dat'])
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
    nopoisson_file = ''.join([util.multidat_dir(), 'abc_nbar_xi_gmf_cov.no_poisson.Mr', str(Mr), '.dat'])
    np.savetxt(nopoisson_file, fullcov)

    # and a correlation matrix
    full_corr_file = ''.join([util.multidat_dir(), 'abc_nbar_xi_gmf_corr.Mr', str(Mr), '.dat'])
    np.savetxt(full_corr_file, fullcorr)

    return None



if __name__ == "__main__":

    build_nbar_xi_gmf_cov(Mr=21)
