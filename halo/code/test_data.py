'''

Test data.py module 

'''

def xi_cov(Mr=20, Nmock=500):
    '''
    Plot xi covariance
    '''
    # covariance  
    cov_dat_file = ''.join(['../dat/xir_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.dat'])
