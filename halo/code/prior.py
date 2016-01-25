'''

Module to deal with prior range

Author(s): CHH, MJV

Notes
-----
 - Prior range implemented Jan 22, 2016 (CHH)

'''
import numpy as np

def PriorRange(prior_name): 
    ''' Given prior dictionary name, return the prior range. 
    '''
    if prior_name is None: 
        prior_name = 'first_try'
    
    dict_table = prior_dict_table() 

    prior_dict = dict_table[prior_name]

    prior_min = prior_dict['prior_min']
    prior_max = prior_dict['prior_max']
    return [prior_min, prior_max]

def prior_dict_table(): 
    ''' dictionary table of priors 
    '''
    dict_table = { 
            'first_try': {
                'prior_min': [10., np.log(0.1), 11.02, 0.8, 13.],
                'prior_max': [13., np.log(0.7), 13.02, 1.3, 14.]
                }
            }
            
    return dict_table 
