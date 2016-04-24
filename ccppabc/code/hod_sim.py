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
#ccppabc functions
import util
from data import data_random
from data import data_RR
from data import data_gmf_bins
from data import xi_binedges 
from group_richness import gmf
from group_richness import richness

class MCMC_HODsim(object):
    def __init__(self, Mr=21, b_normal=0.25):
        ''' Class object that describes our forward model used in MCMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools.
        '''
        self.Mr = Mr
        self.b_normal = b_normal
        thr = -1. * np.float(Mr)

        self.model = PrebuiltHodModelFactory('zheng07', threshold=thr)
        self.halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')

        ##self.RR = data_RR(box='md_all')
        ##self.randoms = data_random('md_all')
        ##self.NR = len(self.randoms)
    
    def __call__(self, theta, prior_range=None, observables=['nbar', 'gmf']):
        return self._sum_stat(theta, prior_range=prior_range, observables=observables)

    def _sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
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

        rbins = xi_binedges()
        rmax = rbins.max()
        approx_cell1_size = [rmax , rmax , rmax]
        approx_cellran_size = [rmax , rmax , rmax]

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
                    greek_xi = tpcf(
                            pos, rbins,  
                            period=self.model.mock.Lbox, 
                            max_sample_size=int(3e5), estimator='Natural', 
                            approx_cell1_size=approx_cell1_size)
                    obvs.append(greek_xi)
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
                            greek_xi = tpcf(
                                    pos, rbins, period=self.model.mock.Lbox, 
                                    max_sample_size=int(3e5), estimator='Natural', 
                                    approx_cell1_size=approx_cell1_size)
                            obvs.append(greek_xi)
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
                            obvs.append(np.zeros(len(xi_binedges()[:-1])))
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
                        obvs.append(np.zeros(len(xi_binedges()[:-1])))
                return obvs


class ABC_HODsim(object):
    def __init__(self, Mr=21, b_normal=0.25):
        ''' Class object that describes our forward model used in AMC-PMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools.
        '''
        self.Mr = Mr
        self.b_normal = b_normal
        thr = -1. * np.float(Mr)

        self.model = PrebuiltHodModelFactory('zheng07', threshold=thr)
        ####self.model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
        self.halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
        self.RR = data_RR(box='md_sub')
        self.randoms = data_random(box='md_sub')
        self.NR = len(self.randoms)

    def __call__(self, theta, prior_range=None, observables=['nbar', 'gmf']):
        return self._sum_stat(theta, prior_range=prior_range, observables=observables)

    def _sum_stat(self, theta, prior_range=None, observables=['nbar', 'gmf']):
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

        rbins = xi_binedges()
        rmax = rbins.max()
        period = None
        approx_cell1_size = [rmax , rmax , rmax]
        approx_cellran_size = [rmax , rmax , rmax]

        if prior_range is None:
            rint = np.random.randint(1, 125)
            ####simsubvol = lambda x: util.mask_func(x, rint)
            ####self.model.populate_mock(self.halocat,
            ####                masking_function=simsubvol,
            ####                enforce_PBC=False)
            self.model.populate_mock(self.halocat)
                        
            pos =three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z')
            pos = util.mask_galaxy_table(pos , rint) 

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
                    greek_xi = tpcf(
                        pos, rbins, pos, 
                        randoms=temp_randoms, period = period, 
                        max_sample_size=int(1e5), estimator='Natural', 
                        approx_cell1_size=approx_cell1_size, 
                        approx_cellran_size=approx_cellran_size,
                        RR_precomputed = self.RR,
	                NR_precomputed = self.NR)

                    obvs.append(greek_xi)
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
                    #imposing mask on the galaxy table
                    pos = util.mask_galaxy_table(pos , rint) 
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
                    	    greek_xi = tpcf(
                                     pos, rbins, pos, 
                        	     randoms=temp_randoms, period = period, 
                                     max_sample_size=int(1e5), estimator='Natural', 
                                     approx_cell1_size=approx_cell1_size, 
                                     approx_cellran_size=approx_cellran_size,
                                     RR_precomputed = self.RR,
	                             NR_precomputed = self.NR)

                    	    obvs.append(greek_xi)
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
                            obvs.append(np.zeros(len(xi_binedges()[:-1])))
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
                        obvs.append(np.zeros(len(xi_binedges()[:-1])))
                return obvs

