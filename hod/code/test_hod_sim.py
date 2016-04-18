'''

tests for hod_sim.py

'''
import time
import numpy as np 
from halotools.empirical_models import PrebuiltHodModelFactory

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



if __name__=='__main__': 
    prebuilt_hodmodel_time_profile()
