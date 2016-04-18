'''
HaloTools HOD Simulation
'''
import numpy as np
import pyfof
#haltools functions
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables import FoFGroups
from halotools.mock_observables.pair_counters import npairs
#ccppabc functions
import util
import data_multislice
from data_multislice import data_random
from data_multislice import data_RR
from data_multislice import data_gmf_bins
from data_multislice import hardcoded_xi_bins
from group_richness import gmf
from group_richness import richness


<<<<<<< HEAD:hod/code/hod_sim.py
class HODsim(object):
    def __init__(self, Mr=21, b_normal=0.25, inference='abc'):
        ''' Class object that describes our forward model used in AMC-PMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools.
        '''
        self.Mr = Mr
        self.b_normal = b_normal
        self.inference = inference
        thr = -1. * np.float(Mr)

        if self.inference == 'abc':
            self.model = PrebuiltHodModelFactory('zheng07', threshold=thr)
            self.halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
            self.RR = data_RR()
            self.randoms = data_random()
            self.NR = len(self.randoms)
        elif self.inference == 'mcmc': 
            self.model = PrebuiltHodModelFactory('zheng07', threshold=thr)
            self.halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
        else: 
            raise ValueError('Inference has to be either using abc or mcmc') 

    def __call__(self, theta, prior_range=None, observables=['nbar', 'gmf']):
        if self.inference == 'abc':  
            return self._ABC_sum_stat(theta, prior_range=prior_range, observables=observables)
        elif self.inference == 'mcmc': 
            return self._MCMC_sum_stat(theta, prior_range=prior_range, observables=observables)

    def _ABC_sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
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

        rbins = hardcoded_xi_bins()
        rmax = rbins.max()
        period = None
        approx_cell1_size = [rmax , rmax , rmax]
        approx_cellran_size = [rmax , rmax , rmax]

        if prior_range is None:
            rint = np.random.randint(1, 125)
            simsubvol = lambda x: util.mask_func(x, rint)
            self.model.populate_mock(self.halocat,
                            masking_function=simsubvol,
                            enforce_PBC=False)
           
            pos =three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z')

            xi , yi , zi = util.random_shifter(rint)
            temp_randoms = self.randoms.copy()
            temp_randoms[:,0] += xi
            temp_randoms[:,1] += yi
            temp_randoms[:,2] += zi

            obvs = []
            for obv in observables:
                if obv == 'nbar':
                    obvs.append(len(pos) / 200**3.)       # nbar of the galaxy catalog
                elif obv == 'gmf':
                    #compute group richness 
                    nbar = len(pos) / 200**3.
    		    b = self.b_normal * (nbar)**(-1./3) 
    		    groups = pyfof.friends_of_friends(pos , b)
    		    w = np.array([len(x) for x in groups])
    		    gbins = data_gmf_bins()
    		    gmf = np.histogram(w , gbins)[0] / (200.**3.)
                    obvs.append(gmf)   
                elif obv == 'xi':
                    xi = tpcf(
                        pos, rbins, pos, 
                        randoms=temp_randoms, period = period, 
                        max_sample_size=int(1e5), estimator='Landy-Szalay', 
                        approx_cell1_size=approx_cell1_size, 
                        approx_cellran_size=approx_cellran_size,
                        RR_precomputed = self.RR,
	                NR_precomputed = self.NR)

                    obvs.append(xi)
                else:
                    raise NotImplementedError('Only nbar 2pcf, gmf implemented so far')

            return obvs

        else:
            if np.all((prior_range[:,0] < theta) & (theta < prior_range[:,1])):
                # if all theta_i is within prior range ...
                try:
                    rint = np.random.randint(1, 125)
                    simsubvol = lambda x: util.mask_func(x, rint)
                    self.model.populate_mock(self.halocat,
                                masking_function=simsubvol,
                                enforce_PBC=False)
           
                    pos =three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z')

            	    xi , yi , zi = util.random_shifter(rint)
            	    temp_randoms = self.randoms.copy()
            	    temp_randoms[:,0] += xi
            	    temp_randoms[:,1] += yi
            	    temp_randoms[:,2] += zi
            	    obvs = []

            	    for obv in observables:
                        if obv == 'nbar':
                    	    obvs.append(len(pos) / 200**3.)       # nbar of the galaxy catalog
                        elif obv == 'gmf':
                            nbar = len(pos) / 200**3.
    		            b = self.b_normal * (nbar)**(-1./3) 
    		            groups = pyfof.friends_of_friends(pos , b)
    		            w = np.array([len(x) for x in groups])
    		    	    gbins =data_gmf_bins()
    		            gmf = np.histogram(w , gbins)[0] / (200.**3.)
                    	    obvs.append(gmf)   
                        elif obv == 'xi':
                    	    xi = tpcf(
                                     pos, rbins, pos, 
                        	     randoms=temp_randoms, period = period, 
                                     max_sample_size=int(2e5), estimator='Landy-Szalay', 
                                     approx_cell1_size=approx_cell1_size, 
                                     approx_cellran_size=approx_cellran_size,
                                     RR_precomputed = self.RR,
	                             NR_precomputed = self.NR)

                    	    obvs.append(xi)
                        else:
                            raise NotImplementedError('Only nbar, tpcf, and gmf are implemented so far')

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
                            obvs.append(np.zeros(len(hardcoded_xi_bins()[:-1])))
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
                        obvs.append(np.zeros(len(hardcoded_xi_bins()[:-1])))
                return obvs

    def _MCMC_sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
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

        rbins = hardcoded_xi_bins()
        rmax = rbins.max()

        if prior_range is None:
            
            self.model.populate_mock(self.halocat)
            pos =three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z')
            obvs = []

            for obv in observables:
                if obv == 'nbar':
                    obvs.append(len(pos) / 1000.**3.)       # nbar of the galaxy catalog
                elif obv == 'gmf':
                    nbar = len(pos) / 1000**3.
    		    b = self.b_normal * (nbar)**(-1./3) 
    		    groups = pyfof.friends_of_friends(pos , b)
    		    w = np.array([len(x) for x in groups])
    		    gbins =data_gmf_bins()
    		    gmf = np.histogram(w , gbins)[0] / (1000.**3.)
                    obvs.append(gmf)   
                elif obv == 'xi':
                    xi = tpcf(pos, rbins, period = self.model.mock.Lbox, max_sample_size=int(2e5), estimator='Landy-Szalay', num_threads = 1)
                    obvs.append(xi)
                else:
                    raise NotImplementedError('Only nbar 2pcf, gmf implemented so far')

            return obvs

        else:
            if np.all((prior_range[:,0] < theta) & (theta < prior_range[:,1])):
                # if all theta_i is within prior range ...
                try:


                    self.model.populate_mock(self.halocat) 
                    pos=three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z')
            	    obvs = []
            	    for obv in observables:
                        if obv == 'nbar':
                    	    obvs.append(len(pos) / 1000**3.)       # nbar of the galaxy catalog
                        elif obv == 'gmf':
                    	    nbar = len(pos) / 1000**3.
    		    	    b = self.b_normal * (nbar)**(-1./3) 
    		    	    groups = pyfof.friends_of_friends(pos , b)
    		    	    w = np.array([len(x) for x in groups])
    		    	    gbins =data_gmf_bins()
    		    	    gmf = np.histogram(w , gbins)[0] / (1000.**3.)
                    	    obvs.append(gmf)   
                        elif obv == 'xi':
                            xi = tpcf(pos, rbins, period = self.model.mock.Lbox, max_sample_size=int(1e5), estimator='Landy-Szalay', num_threads = 1)
                    	    obvs.append(xi)
                        else:
                            raise NotImplementedError('Only nbar, tpcf, and gmf are implemented so far')

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
                            obvs.append(np.zeros(len(hardcoded_xi_bins()[:-1])))
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
                        obvs.append(np.zeros(len(hardcoded_xi_bins()[:-1])))
                return obvs