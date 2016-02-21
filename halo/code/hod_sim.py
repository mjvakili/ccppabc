'''

HaloTools HOD Simulation

temporary turned off parameters other than alpha.
also turning off sub-box population and sample variance for now.
Will go back to sample variance and full hod exploration once mcmc
works. god bless us and save us.

#   code commented out
### comments 

'''
import numpy as np
from halotools.empirical_models import PrebuiltHodModelFactory

import util
from data import data_xi_bins
from data import data_gmf_bins
from data import hardcoded_xi_bins
from group_richness import gmf
from group_richness import richness


class HODsim(object):

    def __init__(self, Mr=21):
        '''
        Class object that describes our forward model used in AMC-PMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools.
        '''
        self.Mr = Mr
        self.model = PrebuiltHodModelFactory('zheng07', threshold=-1*Mr)

    def sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
        '''
        Given theta, sum_stat calculates the observables from our forward model

        Parameters
        ----------
        theta : (self explanatory)
        prior_range : If specified, checks to make sure that theta is within the prior range.
        '''
        self.model.param_dict['logM0'] = theta[0]
        self.model.param_dict['sigma_logM'] = np.exp(theta[1])
        self.model.param_dict['logMmin'] = theta[2]
        self.model.param_dict['alpha'] = theta[3]
        self.model.param_dict['logM1'] = theta[4]
        #self.model.param_dict['alpha'] = theta[0]

        if prior_range is None:
            self.model.populate_mock()                  # forward model HOD galaxy catalog

            obvs = []
            for obv in observables:
                if obv == 'nbar':
                    obvs.append(self.model.mock.number_density)       # nbar of the galaxy catalog
                elif obv == 'gmf':
                    group_id = self.model.mock.compute_fof_group_ids(num_threads=1)
                    group_richness = richness(group_id)         # group richness of the galaxies
                    obvs.append(gmf(group_richness))                 # calculate GMF
                elif obv == 'xi':
                    r, xi_r = self.model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins(), num_threads=1)
                    obvs.append(xi_r)
                else:
                    raise NotImplementedError('Only nbar and GMF implemented so far')

            return obvs

        else:
            if np.all((prior_range[:,0] < theta) & (theta < prior_range[:,1])):
                # if all theta_i is within prior range ...
                try:
                    self.model.populate_mock()

                    obvs = []
                    for obv in observables:
                        if obv == 'nbar':
                            obvs.append(self.model.mock.number_density)     # nbar
                        elif obv == 'gmf':
                            group_id = self. model.mock.compute_fof_group_ids(num_threads=1)
                            group_richness = richness(group_id)         # group richness of the galaxies
                            obvs.append(gmf(group_richness))                 # calculate GMF
                        elif obv == 'xi':
                            r, xi_r = self.model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins(), num_threads=1)
                            obvs.append(xi_r)
                        else:
                            raise NotImplementedError('Only nbar and GMF implemented so far')

                    return obvs

                except ValueError:

                    obvs = []
                    for obv in observables:
                        if obv == 'nbar':
                            obvs.append(10.)
                        elif obv == 'gmf':
                            bins = data_gmf_bins()
                            obvs.append(np.ones_like(bins)[:-1]*1000.)
                        elif obv == 'xi':
                            bins = data_xi_bins(Mr=self.Mr)
                            obvs.append(np.zeros(len(bins)))
                    return obvs
            else:
                obvs = []
                for obv in observables:
                    if obv == 'nbar':
                        obvs.append(10.)
                    elif obv == 'gmf':
                        bins = data_gmf_bins()
                        obvs.append(np.ones_like(bins)[:-1]*1000.)
                    elif obv == 'xi':
                        bins = data_xi_bins(Mr=self.Mr)
                        obvs.append(np.zeros(len(bins)))
                return obvs

def HODsimulator(theta, prior_range, observables=['nbar', 'gmf'], Mr=21):
    '''
    Given theta, sum_stat calculates the observables from our forward model

    Parameters
    ----------
    theta : (self explanatory)
    prior_range : If specified, checks to make sure that theta is within the prior range.
    '''
    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr)
    #model = PrebuiltHodModelFactory('zheng07', threshold=thr,
    #                                halocat='multidark', redshift=0.)
    #model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    #rint = np.random.randint(1, 125)
    #simsubvol = lambda x: util.mask_func(x, rint)

    ### feed theta into model param_dict

    model.param_dict['logM0'] = theta[0]
    model.param_dict['sigma_logM'] = np.exp(theta[1])
    model.param_dict['logMmin'] = theta[2]
    model.param_dict['alpha'] = theta[3]
    model.param_dict['logM1'] = theta[4]
    #model.param_dict['alpha'] = theta[0]

    if prior_range is None:

        ###forward model HOD galaxy catalog

        #model.populate_mock(masking_function=simsubvol, enforce_PBC=False)
        model.populate_mock()

        obvs = []
        for obv in observables:
            if obv == 'nbar':
                obvs.append(model.mock.number_density)       # nbar of the galaxy catalog
            elif obv == 'gmf':
                group_id = model.mock.compute_fof_group_ids(num_threads=1)
                group_richness = richness(group_id)         # group richness of the galaxies
                obvs.append(gmf(group_richness))                 # calculate GMF
            elif obv == 'xi':
                r, xi_r = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins(), num_threads=1)
                obvs.append(xi_r)
            else:
                raise NotImplementedError('Only nbar, gmf, xi are implemented so far')

        return obvs

    else:
        if np.all((prior_range[:,0] < theta) & (theta < prior_range[:,1])):
            ### if all theta_i is within prior range ...
            try:
                #model.populate_mock(masking_function=simsubvol,
                #                    enforce_PBC=False)
		model.populate_mock()

                obvs = []
                for obv in observables:
                    if obv == 'nbar':
                        obvs.append(model.mock.number_density)     # nbar
                    elif obv == 'gmf':
                        group_id = model.mock.compute_fof_group_ids(num_threads=1)
                        group_richness = richness(group_id)         # group richness of the galaxies
                        obvs.append(gmf(group_richness))                 # calculate GMF
                    elif obv == 'xi':
                        r, xi_r = model.mock.compute_galaxy_clustering(rbins=hardcoded_xi_bins(), num_threads=1)
                        obvs.append(xi_r)
                    else:
                        raise NotImplementedError('Only nbar, gmf, xi are implemented so far')

                return obvs

            except ValueError:

                obvs = []
                for obv in observables:
                    if obv == 'nbar':
                        obvs.append(10.)
                    elif obv == 'gmf':
                        bins = data_gmf_bins()
                        obvs.append(np.ones_like(bins)[:-1]*1000.)
                    elif obv == 'xi':
                        bins = data_xi_bins(Mr=self.Mr)
                        obvs.append(np.zeros(len(bins)))
                return obvs
