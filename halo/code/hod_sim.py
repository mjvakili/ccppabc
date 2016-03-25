'''
HaloTools HOD Simulation
'''
import numpy as np

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


class HODsim(object):

    def __init__(self, Mr=21):
        '''
        Class object that describes our forward model used in AMC-PMC inference.
        Our model forward models the galaxy catalog using HOD parameters using HaloTools.
        '''
        self.Mr = Mr
        thr = -1. * np.float(Mr)
        self.model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                           halocat='multidark', redshift=0.)
        self.model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}
        self.RR = data_RR()
        self.randoms = data_random()
        self.NR = len(self.randoms)

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

        rbins = hardcoded_xi_bins()
        rmax = rbins.max()
        period = None
        approx_cell1_size = [rmax , rmax , rmax]
        approx_cellran_size = [rmax , rmax , rmax]

        if prior_range is None:
            
            rint = np.random.randint(1, 125)
            simsubvol = lambda x: util.mask_func(x, rint)
            self.model.populate_mock(simname='multidark',
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
    		    galaxy_sample = self.model.mock.galaxy_table
    	 	    x = galaxy_sample['x']
    	            y = galaxy_sample['y']
    	            z = galaxy_sample['z']
    	            vz = galaxy_sample['vz']
    	            pos = three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z', velocity = vz , velocity_distortion_dimension="z")
    		    b_para, b_perp = 0.5, 0.2
    	 	    groups = FoFGroups(pos, b_perp, b_para, period = None, Lbox = 200 , num_threads=1)
                    gids = groups.group_ids
                    group_richness = richness(gids)
                    obvs.append(gmf(group_richness))                 # calculate GMF
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
                    self.model.populate_mock(simname='multidark',
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
    		    	    galaxy_sample = self.model.mock.galaxy_table
    	 	            x = galaxy_sample['x']
    	                    y = galaxy_sample['y']
    	                    z = galaxy_sample['z']
    	                    vz = galaxy_sample['vz']
    	                    pos = three_dim_pos_bundle(self.model.mock.galaxy_table, 'x', 'y', 'z', velocity = vz , velocity_distortion_dimension="z")
    		            b_para, b_perp = 0.5, 0.2
    	 	            groups = FoFGroups(pos, b_perp, b_para, period = None, Lbox = 200 , num_threads=1)
                            gids = groups.group_ids
                            group_richness = richness(gids)
                            obvs.append(gmf(group_richness))                 # calculate GMF
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

