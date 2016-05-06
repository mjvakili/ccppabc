'''
Plotting modules
'''
import os
from astroML.plotting import plot_mcmc
#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=True)
import h5py
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors
prettycolors()
# --- local ---
import util
import data as Data
from prior import PriorRange
plt.switch_backend("Agg")
from matplotlib import lines as mlines

def overlay_pdfs_contours(abc_filename , mcmc_filename , nwalkers , nburns , Mr):
    import matplotlib as mpl
    label_size = 35
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    MP_LINEWIDTH = 2.4
    MP_TICKSIZE = 10.
    thick_line1 = mlines.Line2D([], [], ls = '-', c = 'blue', linewidth=4, 
                           label='ABC-PMC')
    thick_line2 = mlines.Line2D([], [], ls = '-', c = 'red', linewidth=4, 
                           label=r'Gaussian $\mathcal{L}$ +MCMC')
    mpl.rc('axes', linewidth=MP_LINEWIDTH)
    data_hod_dict = Data.data_hod_param(Mr=Mr)
    truths = np.array([
            data_hod_dict['logM0'],                 # log M0
            np.log(data_hod_dict['sigma_logM']),    # log(sigma)
            data_hod_dict['logMmin'],               # log Mmin
            data_hod_dict['alpha'],                 # alpha
            data_hod_dict['logM1']                  # log M1
            ])

    prior_min, prior_max = PriorRange(None)
    plot_range = np.zeros((len(prior_min),2))
    plot_range[:,0] = prior_min
    plot_range[:,1] = prior_max
    
    mcmc_sample = np.loadtxt(mcmc_filename)[nburns*nwalkers:,:]
    abc_sample = np.loadtxt(abc_filename)
    
    ##################### scatter plots ##########################
    fig, axes = plt.subplots(1, 3, figsize=(48, 13))
    fig.subplots_adjust(wspace=0.4, hspace=0.2)    
   
    ax = axes[0]
    ax_list = corner.hist2d(mcmc_sample[:,3],mcmc_sample[:,2],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                            fill_contours = True, alpha = 10. , color = 'r', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[2,:]])
    ax_list = corner.hist2d(abc_sample[:,3],abc_sample[:,2],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                            fill_contours = True,alpha = 0.1, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[2,:]])
    ax.plot(truths[3] , truths[2] , marker="*", markersize=25 , color = "yellow")
    ax.set_ylabel(r'$\log M_{\rm min}$', fontsize = 50)
    ax.set_xlabel(r'$\alpha$', fontsize = 50)
    ax.set_xlim([plot_range[3,0] , plot_range[3,1]])
    ax.set_ylim([plot_range[2,0] , plot_range[2,1]])
    

    ax = axes[1]
    ax_list = corner.hist2d(mcmc_sample[:,2],mcmc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                            fill_contours = True, color = 'r', smooth = 1. , linewidth=4, range = [plot_range[2,:],plot_range[4,:]])
    ax_list = corner.hist2d(abc_sample[:,2],abc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                           fill_contours = True, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[2,:],plot_range[4,:]])
    ax.plot(truths[2] , truths[4] , marker="*", markersize=25 , color = "yellow")
    ax.set_xlabel(r'$\log M_{\rm min}$', fontsize = 50)
    ax.set_ylabel(r'$\log M_{1}$', fontsize = 50)
    ax.set_xlim([plot_range[2,0] , plot_range[2,1]])
    ax.set_ylim([plot_range[4,0] , plot_range[4,1]])



    ax = axes[2]
    ax_list = corner.hist2d(mcmc_sample[:,3],mcmc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                            fill_contours = True, color = 'r', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[4,:]])
    ax_list = corner.hist2d(abc_sample[:,3], abc_sample[:,4], bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False , 
                            fill_contours = True, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[4,:]])
    ax.plot(truths[3] , truths[4] , marker="*", markersize=25 , color = "yellow")
    ax.set_ylabel(r'$\log M_{1}$', fontsize = 50)
    ax.set_xlabel(r'$\alpha$', fontsize = 50)
    ax.set_xlim([plot_range[3,0] , plot_range[3,1]])
    ax.set_ylim([plot_range[4,0] , plot_range[4,1]])


    plt.legend(handles=[thick_line2, thick_line1], frameon=False, loc='best', fontsize=50)

    plt.savefig("contours_nbarxi2.pdf") 
   
    ################## HISTOGRAMS#########################################
 
    fig, axes = plt.subplots(1, 3, figsize=(48, 13))
    fig.subplots_adjust(wspace=0.4, hspace=0.2)    
   
    ax = axes[0]
    q = ax.hist(mcmc_sample[:,2], bins =20, range = [plot_range[2,:].min(),plot_range[2,:].max()] , normed = True , alpha = 1. , color = 'r', linewidth=4 , histtype='step')
    qq = ax.hist(abc_sample[:,2], bins =20, range = [plot_range[2,:].min(),plot_range[2,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4 , histtype='step')
    ax.vlines(truths[2], 0, max(q[0].max(),qq[0].max()), color = "k", linewidth = 5)
    ax.set_xlabel(r'$\log M_{\rm min}$', fontsize = 50)
    ax.set_xlim([plot_range[2,0] , plot_range[2,1]])
    

    ax = axes[1]
    q = ax.hist(mcmc_sample[:,3], bins =20, range = [plot_range[3,:].min(),plot_range[3,:].max()] ,normed = True , alpha = 1. , color = 'r', linewidth=4, histtype='step')
    qq = ax.hist(abc_sample[:,3], bins =20, range = [plot_range[3,:].min(),plot_range[3,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4, histtype='step')
    ax.vlines(truths[3], 0, max(q[0].max(),qq[0].max()), color = "k" , linewidth = 5)
    ax.set_xlabel(r'$\alpha$', fontsize = 50)
    ax.set_xlim([plot_range[3,0] , plot_range[3,1]])


    ax = axes[2]
    q = ax.hist(mcmc_sample[:,4], bins =20, range = [plot_range[4,:].min(),plot_range[4,:].max()] ,normed = True , alpha = 1. , color = 'r', linewidth=4, histtype='step')
    qq = ax.hist(abc_sample[:,4], bins =20, range = [plot_range[4,:].min(),plot_range[4,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4 , histtype='step')
    ax.vlines(truths[4] , 0, max(q[0].max(),qq[0].max()), colors='k' , linewidth = 5)
    ax.set_xlabel(r'$\log M_{1}$', fontsize = 50)
    ax.set_xlim([plot_range[4,0] , plot_range[4,1]])


    plt.legend(handles=[thick_line2, thick_line1], frameon=False, loc='best', fontsize=30)

    plt.savefig("histograms_nbarxi2.pdf") 
    return None

if __name__ == "__main__":

    mcmc_filename = "results/nbar_xi.mcmc.mcmc_chain.dat"
    abc_filename  = "results/nbar_xi_theta_t8.abc.dat"
    #abc_filename = "results/nbar_gmf_theta_t8.ABCnbargmf.dat"
    overlay_pdfs_contours(abc_filename , mcmc_filename , 100 , 6000 , 21) 
