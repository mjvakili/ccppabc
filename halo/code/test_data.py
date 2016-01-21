'''

Test data.py module 

'''
import numpy as np 
import matplotlib.pyplot as plt

import data 
import util 

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
    fig_file = ''.join([util.fig_dir(),
        'xi_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.png'])
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close()


def gmf_cov(Mr=20, Nmock=500):
    '''Plot the reduced xi covariance
    '''
    # covariance  
    gmf_cov = data.data_gmf_cov(Mr=Mr, Nmock=Nmock)
    n_bin = int(np.sqrt(gmf_cov.size))
    x_bin = data.data_gmf_bins()
    
    # calculate the reduced covariance for plotting
    gmf_cov_red = np.zeros([n_bin, n_bin])
    for ii in range(n_bin): 
        for jj in range(n_bin): 
            gmf_cov_red[ii][jj] = gmf_cov[ii][jj]/np.sqrt(gmf_cov[ii][ii] * gmf_cov[jj][jj])

    fig = plt.figure()
    sub = fig.add_subplot(111)
    cont = sub.pcolormesh(x_bin, x_bin, gmf_cov_red, cmap=plt.cm.afmhot_r)
    plt.colorbar(cont)

    sub.set_xlim([x_bin[0], x_bin[-1]])
    sub.set_ylim([x_bin[0], x_bin[-1]])

    sub.set_xlabel(r'$\mathtt{Group\;\;Richness}$', fontsize=25)
    sub.set_ylabel(r'$\mathtt{Group\;\;Richness}$', fontsize=25)
    
    fig_file = ''.join([util.fig_dir(),
        'gmf_covariance.Mr', str(Mr), '.Nmock', str(Nmock), '.png'])
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close()

if __name__=='__main__': 
    xi_cov(Mr=20, Nmock=500)
    gmf_cov(Mr=20, Nmock=500)
