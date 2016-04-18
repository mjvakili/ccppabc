'''

Tests for abc_pemcee module

'''
import pickle

from hod_sim import HODsim

def simz_crash_test(): 
    '''
    ########################## 
    ######## RESOLVED ######## 
    ########################## 
    Simz crashes and gives None as output. Why? 

    Because of an indentation error in the sum_stat 
    moduel in the HODsim class.
    '''
    theta = pickle.load(open('simz_crash_theta.p', 'rb'))
    kwargs = pickle.load(open('simz_crash_kwargs.p', 'rb'))

    ourmodel = HODsim()
    return ourmodel.sum_stat(theta, **kwargs)
