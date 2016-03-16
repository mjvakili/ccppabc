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
                'prior_min': [10.5, np.log(0.2), 11.8, 0.85, 13.],
                'prior_max': [13.5, np.log(0.8), 13.8, 1.45, 15.5]
                }
            }
            
    return dict_table 
