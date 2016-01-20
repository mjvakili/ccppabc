'''

Test data.py module 

'''
import numpy as np 
import matplotlib.pyplot as plt

import data 

def xi_cov(Mr=20, Nmock=500):
    '''Plot the reduced xi covariance
    '''
    # covariance  
    xi_cov = data.data_xi_cov(Mr=Mr, Nmock=Nmock)
    n_bin = int(np.sqrt(xi_cov.size))
    r_bin = data.data_xi_bin(Mr=Mr)
    
    # calculate the reduced covariance for plotting
    xi_cov_red = np.zeros([n_bin, n_bin])
    for ii in range(n_bin): 
        for jj in range(n_bin): 
            xi_cov_red[ii][jj] = xi_cov[ii][jj]/np.sqrt(xi_cov[ii][ii] * xi_cov[jj][jj])

    fig = plt.figure()
    sub = fig.add_subplot(111)
    cont = sub.pcolormesh(r_bin, r_bin, xi_cov_red, cmap=plt.cm.afmhot_r)
    plt.colorbar(cont)

    sub.set_xlim([r_bin[0], r_bin[-1]])
    sub.set_ylim([r_bin[0], r_bin[-1]])
    sub.set_xscale('log')
    sub.set_yscale('log')

    sub.set_xlabel(r'$r\;(\mathrm{Mpc}/h)$', fontsize=25)
    sub.set_ylabel(r'$r\;(\mathrm{Mpc}/h)$', fontsize=25)
    plt.show()

if __name__=='__main__': 
    xi_cov(Mr=20, Nmock=5)
