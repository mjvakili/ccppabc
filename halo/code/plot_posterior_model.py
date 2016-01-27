'''

Module for plotting model posterior predictions 
for different observables
Author(s): Chang, MJ

'''
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy.linalg import solve
plt.switch_backend("Agg")
from halotools.empirical_models import PrebuiltHodModelFactory

# --- Local ---
import util
import data as Data
from hod_sim import HODsim
from hod_sim import HODsimulator 
from group_richness import richness
from prior import PriorRange
import corner
    
def plot_xi_model(theta, observables=['xi'], Mr=20):
    
    #load the data   
 
    fake_obs = []
    fake_obs_cov = [] 
    for obv in observables: 
        if obv == 'nbar': 
            data_nbar, data_nbar_var = Data.data_nbar(**data_dict)
            fake_obs.append(data_nbar)
            fake_obs_cov.append(data_nbar_var)
        if obv == 'gmf': 
            data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
            fake_obs.append(data_gmf)
            fake_obs_cov.append(data_gmf)
        if obv == 'xi': 
            # import xir and full covariance matrix of xir
            data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict)   
            fake_obs.append(data_xi)
            fake_obs_cov.append(data_xi_invcov)
    
    xi_gg = []

    for i in xrange(len(theta)):
        
        mod = PrebuiltHodModelFactory('zheng07', threshold=-1*Mr)
        mod.param_dict["logM0"] = theta[i][0]
        mod.param_dict["sigma_logM"] = np.exp(theta[i][1])
    	mod.param_dict["logMmin"] = theta[i][2]
    	mod.param_dict["alpha"] = theta[i][3]
    	mod.param_dict["logM1"] = theta[i][4]  
        
        mod.populate_mock()
        """ if we want to make a smooth plot, we do this:"""
        #rr , xi = mod.compute_average_compute_average_galaxy_clustering(num_iterations = 6 )
        """else"""
        rr , xi = mod.mock.compute_galaxy_clustering()
        xi_gg.append(xi)
 
    xi_gg = np.array(xi_gg)
    a, b, c, d, e = np.percentile(xi_gg, [2.5, 16, 50, 84, 97.5], axis=0)

    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)

    
    ax.fill_between(rr, a, e, color="k", alpha=0.1, edgecolor="none")
    ax.fill_between(rr, b, d, color="k", alpha=0.3, edgecolor="none")
    ax.plot(rr, c, "k", lw=1)
    ax.errorbar(rr, data_xi, yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)    
    xlabel = ax.set_xlabel(r'$r[\mathrm{Mpc}h^{-1}]$', fontsize=20)
    ylabel = ax.set_ylabel(r'$\xi_{\rm gg}$', fontsize=25)
    
    plt.xscale('log')
    plt.xlim(xmin = .1 , xmax = 15)
    fig1.savefig('../figs/xi_posterior_prediction'+str(Mr)+'.pdf', 
	        bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight') 



    fig2 = plt.figure()
    ax = fig2.add_subplot(111)

    ax.fill_between(rr, rr*a, rr*e, color="k", alpha=0.1, edgecolor="none")
    ax.fill_between(rr, rr*b, rr*d, color="k", alpha=0.3, edgecolor="none")
    #ax.plot(rr, c, "k", lw=1) we don't care about the best fit here
    ax.errorbar(rr, rr*data_xi, yerr = rr*np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)
    xlabel = ax.set_xlabel(r'$r[\mathrm{Mpc}h^{-1}]$', fontsize=20)
    ylabel = ax.set_ylabel(r'$r\xi_{\rm gg}$', fontsize=25)

    plt.xscale('log')
    plt.xlim(xmin = .1 , xmax = 15)
    fig2.savefig('../figs/xi_scaled_posterior_prediction'+str(Mr)+'.pdf',
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

    
    fig3 = plt.figure()
    ax = fig3.add_subplot(111)

    ax.fill_between(rr, a - data_xi, e-data_xi, color="k", alpha=0.1, edgecolor="none")
    ax.fill_between(rr, b - data_xi, d-data_xi, color="k", alpha=0.3, edgecolor="none")
    #ax.plot(rr, c, "k", lw=1) we don't care about the best fit here
    ax.errorbar(rr, data_xi - data_xi, yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)
    xlabel = ax.set_xlabel(r'$r[\mathrm{Mpc}h^{-1}]$', fontsize=20)
    ylabel = ax.set_ylabel(r'$\Delta \xi_{\rm gg}$', fontsize=25)

    plt.xscale('log')
    plt.xlim(xmin = .1 , xmax = 15)
    fig3.savefig('../figs/xi_residual_posterior_prediction'+str(Mr)+'.pdf',
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')


    fig4, axes = pl.subplots(2, 1, figsize=(10, 8) , sharex = True)
    fig4.subplots_adjust(wspace=0.0 hspace=0.4)

    ax1 = axes[0,0]
    
    ax1.fill_between(rr, a, e, color="k", alpha=0.1, edgecolor="none")
    ax1.fill_between(rr, b, d, color="k", alpha=0.3, edgecolor="none")
    #ax1.plot(rr, c, "k", lw=1)
    ax1.errorbar(rr, data_xi, yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)    
    ylabel = ax1.set_ylabel(r'$\xi_{\rm gg}$', fontsize=25)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1_set.xlim(xmin = .1 , xmax = 15)
    
    ax2 = axes[0,0]
    ax2.fill_between(rr, a - data_xi, e- data_xi, color="k", alpha=0.1, edgecolor="none")
    ax2.fill_between(rr, b- data_xi, d- data_xi, color="k", alpha=0.3, edgecolor="none")
    #ax2.plot(rr, c, "k", lw=1)
    ax2.errorbar(rr, data_xi- data_xi , yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)    
    ylabel = ax2.set_ylabel(r'$\Delta \xi_{\rm gg}$', fontsize=25)
    ax2.set_xscale("log")
    ax2_set.xlim(xmin = .1 , xmax = 15)

    fig4.savefig('../figs/xi&residual_posterior_prediction'+str(Mr)+'.pdf',
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')



    fig5, axes = pl.subplots(2, 1, figsize=(10, 8) , sharex = True)
    fig5.subplots_adjust(wspace=0.0 hspace=0.4)

    ax1 = axes[0,0]
    
    ax1.fill_between(rr,rr* a, rr* e, color="k", alpha=0.1, edgecolor="none")
    ax1.fill_between(rr, rr* b, rr* d, color="k", alpha=0.3, edgecolor="none")
    #ax1.plot(rr, c, "k", lw=1)
    ax1.errorbar(rr, rr* data_xi, rr* yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)    
    ylabel = ax1.set_ylabel(r'$r\xi_{\rm gg}$', fontsize=25)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1_set.xlim(xmin = .1 , xmax = 15)
    
    ax2 = axes[0,0]
    ax2.fill_between(rr, rr* a - rr* data_xi, rr* e- rr* data_xi, color="k", alpha=0.1, edgecolor="none")
    ax2.fill_between(rr, rr* b- rr* data_xi, rr* d- rr* data_xi, color="k", alpha=0.3, edgecolor="none")
    #ax2.plot(rr, c, "k", lw=1)
    ax2.errorbar(rr, rr*data_xi- rr* data_xi , yerr = rr* np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                capsize=0)    
    ylabel = ax2.set_ylabel(r'$\Delta(r\xi_{\rm gg})$', fontsize=25)
    ax2.set_xscale("log")
    ax2_set.xlim(xmin = .1 , xmax = 15)

    fig5.savefig('../figs/xi&residual_scaled_posterior_prediction'+str(Mr)+'.pdf',
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
