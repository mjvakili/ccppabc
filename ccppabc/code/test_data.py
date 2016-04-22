'''

Test the data.py module 

'''
import numpy as np 
import matplotlib.pyplot as plt

import util 
import data as Data
# --- Halotools ---
from halotools.empirical_models import PrebuiltHodModelFactory

from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def PlotCovariance(obvs, Mr=21, b_normal=0.25, inference='mcmc'):
    ''' Plot the covariance matrix for a specified obvs 
    '''
    # import the covariance matrix 
    covar = Data.data_cov(Mr=Mr, b_normal=b_normal, inference=inference)

    if obvs == 'xi': 
        obvs_cov = covar[1:16 , 1:16]
        r_bin = Data.xi_binedges()
    elif obvs == 'gmf': 
        obvs_cov = covar[17:, 17:]
        binedges = Data.data_gmf_bins()
        r_bin = 0.5 * (binedges[:-1] + binedges[1:]) 
    
    n_bin = int(np.sqrt(obvs_cov.size))
    
    # calculate the reduced covariance for plotting
    red_covar = np.zeros([n_bin, n_bin])
    for ii in range(n_bin): 
        for jj in range(n_bin): 
            red_covar[ii][jj] = obvs_cov[ii][jj]/np.sqrt(obvs_cov[ii][ii] * obvs_cov[jj][jj])
    
    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)
    cont = sub.pcolormesh(r_bin, r_bin, red_covar, cmap=plt.cm.afmhot_r)
    plt.colorbar(cont)

    sub.set_xlim([r_bin[0], r_bin[-1]])
    sub.set_ylim([r_bin[0], r_bin[-1]])
    sub.set_xscale('log')
    sub.set_yscale('log')

    sub.set_xlabel(r'$\mathtt{r}\;[\mathtt{Mpc/h}$]', fontsize=25)
    sub.set_ylabel(r'$\mathtt{r}\;[\mathtt{Mpc/h}$]', fontsize=25)
    fig_file = ''.join([util.fig_dir(),
        obvs.upper(), 'covariance',
        '.Mr', str(Mr), 
        '.bnorm', str(round(b_normal,2)), 
        '.', inference, '_inf.png'])
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close()
    return None 


# ---- Plotting ----
def xi(Mr=20, Nmock=500): 
    '''
    Plot xi(r) of the fake observations
    '''
    prettyplot() 
    pretty_colors = prettycolors() 

    xir, cii = Data.data_xi(Mr=Mr, Nmock=Nmock)
    rbin = Data.data_xi_bins(Mr=Mr) 
    
    fig = plt.figure(1) 
    sub = fig.add_subplot(111)
    sub.plot(rbin, rbin*xir, c='k', lw=1)
    sub.errorbar(rbin, rbin*xir, yerr = rbin*cii**0.5 , fmt="ok", ms=1, capsize=2, alpha=1.)

    sub.set_xlim([0.1, 15])
    sub.set_ylim([1, 10])
    sub.set_yscale("log")
    sub.set_xscale("log")

    sub.set_xlabel(r'$\mathtt{r}\; (\mathtt{Mpc})$', fontsize=25)
    sub.set_ylabel(r'$\mathtt{r} \xi_{\rm gg}$', fontsize=25)

    fig_file = ''.join([util.fig_dir(),
        'xi.Mr', str(Mr), '.Nmock', str(Nmock), '.png'])
    fig.savefig(fig_file, bbox_inches='tight')
    plt.close()
    return None

def gmf(Mr=20, Nmock=500): 
    '''
    Plot Group Multiplicty Function of fake observations 
    '''
    prettyplot() 
    pretty_colors = prettycolors() 
    
    # import fake obs GMF
    gmf, sig_gmf = Data.data_gmf(Mr=Mr, Nmock=Nmock)
    # group richness bins
    gmf_bin = Data.data_gmf_bins()

    fig = plt.figure(1) 
    sub = fig.add_subplot(111)

    sub.errorbar(
            0.5*(gmf_bin[:-1]+gmf_bin[1:]), gmf, yerr=sig_gmf,
            fmt="ok", capsize=1.0
            )
    sub.set_xlim([1, 60])
    sub.set_yscale('log')
    sub.set_ylabel(r"Group Multiplicity Function (h$^{3}$ Mpc$^{-3}$)", fontsize=20)
    sub.set_xlabel(r"$\mathtt{Group\;\;Richness}$", fontsize=20)
    # save to file 
    fig_file = ''.join([util.fig_dir(), 
        'gmf.Mr', str(Mr), '.Nmock', str(Nmock), '.png'])
    fig.savefig(fig_file, bbox_inches='tight')
    return None

# ---- tests -----
def xi_binning_tests(Mr=20):
    model = PrebuiltHodModelFactory('zheng07', threshold = -1.0*np.float(Mr))

    rbins = np.concatenate([np.array([0.1]), np.logspace(np.log10(0.5), np.log10(20.), 15)])
    print 'R bins = ', rbins
    for ii in xrange(10): 
        model.populate_mock() # population mock realization 
        
        #rbins = np.logspace(-1, np.log10(20.), 16)
        r_bin, xi_r = model.mock.compute_galaxy_clustering(rbins=rbins)
        print xi_r

def test_nbar(Mr=21, b_normal=0.25): 
    print Data.data_nbar(Mr=Mr, b_normal=b_normal)


if __name__=='__main__': 
    PlotCovariance('gmf', inference='mcmc')

    #test_nbar()
    #xi_cov(Mr=20, Nmock=500)
    #xi_binning_tests(Mr=20)
