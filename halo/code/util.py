'''

Utility modules 

'''
import os 

def observable_id_flag(observables): 
    '''
    Observable identification flag string given list of observables
    '''
    return '_'.join(observables) 

def fig_dir(): 
    '''
    figure directory
    '''
    fig_dir = os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'fig/'
    return fig_dir 

def dat_dir(): 
    '''
    Dat directory
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'dat/'
