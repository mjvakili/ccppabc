'''

Modules for data and observables in ABC-PMC

'''
import numpy as np

def data_gmf(Mr=20): 
    '''
    Observed GMF from 'data'
    '''

    gmf_dat_file = ''.join([
        '../dat/', 
        'gmf_Mr', str(Mr), '.dat'
        ])
    
    gmf_sig_file = ''.join([
        '../dat/', 
        'gmf_noise_Mr20_2.dat'
        ])
    sigma = np.loadtxt(gmf_sig_file)
    
    return [np.loadtxt(gmf_dat_file), sigma]

def data_gmf_bins(Mr=20):
    '''
    Observed GMF bins
    '''
    bins = np.array([3.000000000000000000e+00, 4.000000000000000000e+00, 5.000000000000000000e+00, 6.000000000000000000e+00, 7.000000000000000000e+00, 8.000000000000000000e+00, 9.000000000000000000e+00, 1.000000000000000000e+01, 1.100000000000000000e+01, 1.200000000000000000e+01, 1.300000000000000000e+01, 1.400000000000000000e+01, 1.500000000000000000e+01, 1.600000000000000000e+01, 1.700000000000000000e+01, 1.800000000000000000e+01, 1.900000000000000000e+01, 2.000000000000000000e+01, 2.100000000000000000e+01, 2.200000000000000000e+01, 2.300000000000000000e+01, 2.500000000000000000e+01, 2.900000000000000000e+01, 3.100000000000000000e+01, 3.500000000000000000e+01, 4.300000000000000000e+01, 6.100000000000000000e+01]) # same hardcoded bins as data 

    return bins

def data_nbar(Mr=20): 
    '''
    Observed nbar from 'data'
    '''
    nbar_dat_file = ''.join([
        '../dat/',
        'nbar_Mr', str(Mr),'.dat'
        ])
    mock_nbar = np.loadtxt(nbar_dat_file)

    return [np.mean(mock_nbar), np.var(mock_nbar)]
