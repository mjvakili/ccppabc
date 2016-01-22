'''

HaloTools HOD Simulation

'''
import numpy as np
from halotools.empirical_models import PrebuiltHodModelFactory

from data import data_gmf_bins
from group_richness import gmf 
from group_richness import richness


class HODsim(object): 
    
    def __init__(self, Mr=20): 
        '''
        Class object that describes our forward model used in AMC-PMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools. 
        '''

        #self.model = zheng07()      # Zheng et al. (2007) model
        self.model = PrebuiltHodModelFactory('zheng07', threshold=-1*Mr)
    
    def sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
        '''
        Given theta, sum_stat calculates the observables from our forward model 

        Parameters
        ----------
        theta : (self explanatory)
        prior_range : If specified, checks to make sure that theta is within the prior range.
        '''
        # feed theta into model param_dict
        self.model.param_dict['logM0'] = theta[0]
        self.model.param_dict['sigma_logM'] = np.exp(theta[1])
        self.model.param_dict['logMmin'] = theta[2]     
        self.model.param_dict['alpha'] = theta[3]
        self.model.param_dict['logM1'] = theta[4]
    
        if prior_range is None:
            self.model.populate_mock()                  # forward model HOD galaxy catalog 
            
            obvs = []  
            for obv in observables: 
                if obv == 'nbar': 
                    obvs.append(self.model.mock.number_density)       # nbar of the galaxy catalog
                elif obv == 'gmf': 
                    group_id = self.model.mock.compute_fof_group_ids() 
                    group_richness = richness(group_id)         # group richness of the galaxies
                    obvs.append(gmf(group_richness))                 # calculate GMF
                elif obv == 'xi': 
                    r, xi_r = self.model.mock.compute_galaxy_clustering()
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
                            group_id = self. model.mock.compute_fof_group_ids()
                            group_richness = richness(group_id)         # group richness of the galaxies
                            obvs.append(gmf(group_richness))                 # calculate GMF
                        elif obv == 'xi': 
                            r, xi_r = self.model.mock.compute_galaxy_clustering()
                            obvs.append(xi_r)
                        else: 
                            raise NotImplementedError('Only nbar and GMF implemented so far')

                    return obvs 

                except ValueError:
                    bins = data_gmf_bins()
                    return [10. , np.ones_like(bins)[:-1]*1000.]
            else:
                bins = data_gmf_bins()
                return [10. , np.ones_like(bins)[:-1]*1000.]
