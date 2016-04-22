'''

tests for hod_sim.py

'''
import time
import numpy as np 


from halotools.empirical_models import PrebuiltHodModelFactory
import data as Data 
from hod_sim import ABC_HODsim
from prior import PriorRange

def prebuilt_hodmodel_time_profile(): 
    ''' Profile how long it takes to declare PrebuiltHodModelFactory.
    It takes roughly 0.000933730602264 seconds. So not very long. 
    '''
    ittook = [] 
    for i in xrange(20): 
        start_time = time.time() 

        model = PrebuiltHodModelFactory('zheng07', threshold=-20)

        ittook.append(time.time() - start_time)
    
    print np.mean(np.array(ittook)), ' seconds' 


def abc_hodsim(Mr=21, b_normal=0.25):

    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min
    prior_range[:,1] = prior_max

    abc_hod = ABC_HODsim() 

    data_hod_dict = Data.data_hod_param(Mr=21)
    data_hod = np.array([
        data_hod_dict['logM0'],                 # log M0 
        np.log(data_hod_dict['sigma_logM']),    # log(sigma)
        data_hod_dict['logMmin'],               # log Mmin
        data_hod_dict['alpha'],                 # alpha
        data_hod_dict['logM1']                  # log M1
        ])
    print Data.data_nbar(Mr=Mr, b_normal=b_normal), Data.data_gmf(Mr=Mr, b_normal=b_normal)
    print abc_hod(data_hod, prior_range=prior_range, observables=['nbar', 'gmf']) 


if __name__=='__main__': 
    abc_hodsim()

