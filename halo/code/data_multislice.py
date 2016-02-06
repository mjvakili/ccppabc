import numpy as np
from halotools.sim_manager import CachedHaloCatalog

import util


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
    model.new_haloprop_func_dict = {sim_subvol: mk_id_column}
    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(masking_function=datsubvol, enforce_PBC=False)
    r_bin  = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins())[0]
    output_file = ''.join([util.multidat_dir(), 'xir_rbin.Mr', str(Mr), '.dat'])
    np.savetxt(output_file, r_bin)
    return None


def build_xi_nbar_gmf_cov(Mr=20):
    '''
    Build covariance matrix for xi, variance for nbar, and a bunch of stuff for gmf
    ...
    using Nmock realizations of halotool mocks
    '''
    xir = []
    nbars = []
    gmfs = []
    gmf_counts = []

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {sim_subvol: mk_id_column}

    for i in xrange(1,125):
        print 'mock#', i

        # populate the mock subvolume
        mocksubvol = lambda x: util.mask_func(x, i)
        model.populate_mock(masking_function=mocksubvol, enforce_PBC=False)

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

    # write full covariance matrix of various combinations of the data

    # covariance for all three
    fulldatarr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
                           np.array(gmfs), np.array(xir))

    fullcov = np.cov(fulldatarr.T)
    outfn = ''.join([util.dat_dir(), 'nbar_gmf_xir_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, fullcov)

    # covariance for nbar and gmf
    nbgmf_arr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
                         np.array(gmfs))
    nbgmf_cov = np.cov(nbgmf_arr.T)
    outfn = ''.join([util.dat_dir(), 'nbar_gmf_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, nbgmf_cov)

    # covariance for nbar and xi
    nbxi_arr = np.hstack(np.array(nbars).reshape(nbars.shape[0], 1),
                         np.array(xir))
    nbxi_cov = np.cov(nbxi_arr.T)
    outfn = ''.join([util.dat_dir(), 'nbar_xi_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, nbxi_cov)

    # covariance for gmf and xi
    gmfxi_arr = np.hstack(np.array(gmfs), np.array(xir))
    gmfxi_cov = np.cov(gmfxi_arr.T)
    outfn = ''.join([util.dat_dir(), 'gmf_xi_cov.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
    np.savetxt(outfn, gmfxi_cov)

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

    # make the subvolume index column if it doesn't exist
    subvol_id_fn = util.multidat_dir() + 'subvol_ids.dat'
    try:
        subvol_ids = np.loadtxt(subvol_id_fn)
    except IOError:
        halocat = CachedHaloCatalog(simname='multidark',
                                    halo_finder='rockstar',
                                    redshift=0.0)
        edges = np.linspace(0, 800, 5)
        xs = halocat.halo_table["halo_x"]
        ys = halocat.halo_table["halo_y"]
        zs = halocat.halo_table["halo_z"]

        subvol_ids = np.empty(xs.shape)
        for i in xrange(len(xs)):
            xi = np.where(edges < xs[0][i])[-1]
            yi = np.where(edges < ys[0][i])[-1]
            zi = np.where(edges < zs[0][i])[-1]
            subvol_ids[i] = zi * 25 + yi * 5 + xi

        np.savetxt(subvol_id_fn, subvol_ids)


    # xi, nbar, gmf
    print 'Building xi(r), nbar, GMF ... '
    build_xi_bins(Mr=Mr)
    build_xi_nbar_gmf(Mr=Mr)


