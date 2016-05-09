'''
Plotting modules
'''
import os
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
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
# from ChangTools.plotting import prettyplot
# from ChangTools.plotting import prettycolors
# prettycolors()
# --- local ---
import util
import data as Data
from prior import PriorRange
plt.switch_backend("Agg")
from matplotlib import lines as mlines
from matplotlib import gridspec

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

#     ##################### scatter plots ##########################
#     fig, axes = plt.subplots(2, 3, figsize=(48, 13))
#     fig.subplots_adjust(wspace=0.4, hspace=0.2)
#
#     ax = axes[0]
#     ax_list = corner.hist2d(mcmc_sample[:,3],mcmc_sample[:,2],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                             fill_contours = True, alpha = 10. , color = 'r', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[2,:]])
#     ax_list = corner.hist2d(abc_sample[:,3],abc_sample[:,2],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                             fill_contours = True,alpha = 0.1, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[2,:]])
#     ax.plot(truths[3] , truths[2] , marker="*", markersize=25 , color = "yellow")
#     ax.set_ylabel(r'$\log M_{\rm min}$', fontsize = 50)
#     ax.set_xlabel(r'$\alpha$', fontsize = 50)
#     ax.set_xlim([plot_range[3,0] , plot_range[3,1]])
#     ax.set_ylim([plot_range[2,0] , plot_range[2,1]])
#
#
#     ax = axes[1]
#     ax_list = corner.hist2d(mcmc_sample[:,2],mcmc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                             fill_contours = True, color = 'r', smooth = 1. , linewidth=4, range = [plot_range[2,:],plot_range[4,:]])
#     ax_list = corner.hist2d(abc_sample[:,2],abc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                            fill_contours = True, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[2,:],plot_range[4,:]])
#     ax.plot(truths[2] , truths[4] , marker="*", markersize=25 , color = "yellow")
#     ax.set_xlabel(r'$\log M_{\rm min}$', fontsize = 50)
#     ax.set_ylabel(r'$\log M_{1}$', fontsize = 50)
#     ax.set_xlim([plot_range[2,0] , plot_range[2,1]])
#     ax.set_ylim([plot_range[4,0] , plot_range[4,1]])
#
#
#
#     ax = axes[2]
#     ax_list = corner.hist2d(mcmc_sample[:,3],mcmc_sample[:,4],bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                             fill_contours = True, color = 'r', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[4,:]])
#     ax_list = corner.hist2d(abc_sample[:,3], abc_sample[:,4], bins =20,levels=[0.68,0.95], ax = ax, plot_datapoints = False ,
#                             fill_contours = True, color = 'b', smooth = 1. , linewidth=4, range = [plot_range[3,:],plot_range[4,:]])
#     ax.plot(truths[3] , truths[4] , marker="*", markersize=25 , color = "yellow")
#     ax.set_ylabel(r'$\log M_{1}$', fontsize = 50)
#     ax.set_xlabel(r'$\alpha$', fontsize = 50)
#     ax.set_xlim([plot_range[3,0] , plot_range[3,1]])
#     ax.set_ylim([plot_range[4,0] , plot_range[4,1]])
#
#
#     plt.legend(handles=[thick_line2, thick_line1], frameon=False, loc='best', fontsize=50)
#
#     plt.savefig("contours_nbarxi2.pdf")

    ################## HISTOGRAMS#########################################

#     fig, axes = plt.subplots(1, 3, figsize=(48, 13))
#     fig.subplots_adjust(wspace=0.4, hspace=0.2)
#
#     ax = axes[0]
#     q = ax.hist(mcmc_sample[:,2], bins =20, range = [plot_range[2,:].min(),plot_range[2,:].max()] , normed = True , alpha = 1. , color = 'r', linewidth=4 , histtype='step')
#     qq = ax.hist(abc_sample[:,2], bins =20, range = [plot_range[2,:].min(),plot_range[2,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4 , histtype='step')
#     ax.vlines(truths[2], 0, max(q[0].max(),qq[0].max()), color = "k", linewidth = 5)
#     ax.set_xlabel(r'$\log M_{\rm min}$', fontsize = 50)
#     ax.set_xlim([plot_range[2,0] , plot_range[2,1]])
#
#
#     ax = axes[1]
#     q = ax.hist(mcmc_sample[:,3], bins =20, range = [plot_range[3,:].min(),plot_range[3,:].max()] ,normed = True , alpha = 1. , color = 'r', linewidth=4, histtype='step')
#     qq = ax.hist(abc_sample[:,3], bins =20, range = [plot_range[3,:].min(),plot_range[3,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4, histtype='step')
#     ax.vlines(truths[3], 0, max(q[0].max(),qq[0].max()), color = "k" , linewidth = 5)
#     ax.set_xlabel(r'$\alpha$', fontsize = 50)
#     ax.set_xlim([plot_range[3,0] , plot_range[3,1]])
#
#
#     ax = axes[2]
#     q = ax.hist(mcmc_sample[:,4], bins =20, range = [plot_range[4,:].min(),plot_range[4,:].max()] ,normed = True , alpha = 1. , color = 'r', linewidth=4, histtype='step')
#     qq = ax.hist(abc_sample[:,4], bins =20, range = [plot_range[4,:].min(),plot_range[4,:].max()] ,normed = True , alpha = 1. , color = 'b', linewidth=4 , histtype='step')
#     ax.vlines(truths[4] , 0, max(q[0].max(),qq[0].max()), colors='k' , linewidth = 5)
#     ax.set_xlabel(r'$\log M_{1}$', fontsize = 50)
#     ax.set_xlim([plot_range[4,0] , plot_range[4,1]])
#
#
#     plt.legend(handles=[thick_line2, thick_line1], frameon=False, loc='best', fontsize=30)
#
#     plt.savefig("histograms_nbarxi2.pdf")
#     return None

    # get the 68% and 95% confidence interval indices for sorted chain
    normie = norm()
    sig1lo = normie.cdf(-1)
    sig1hi = normie.cdf(1)
    sig2lo = normie.cdf(-2)
    sig2hi = normie.cdf(2)

    nsamples_mcmc = len(mcmc_sample[:, 2])
    nsamples_abc = len(abc_sample[:, 2])

    sig1lo_mcmc = int(sig1lo * nsamples_mcmc)
    sig2lo_mcmc = int(sig2lo * nsamples_mcmc)
    sig1hi_mcmc = int(sig1hi * nsamples_mcmc)
    sig2hi_mcmc = int(sig2hi * nsamples_mcmc)

    sig1lo_abc = int(sig1lo * nsamples_abc)
    sig2lo_abc = int(sig2lo * nsamples_abc)
    sig1hi_abc = int(sig1hi * nsamples_abc)
    sig2hi_abc = int(sig2hi * nsamples_abc)

    # choose colours here
    abclr = 'darkcyan'
    mcmclr = 'darkorchid'

    fig = plt.figure(1, figsize=(50, 25))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

    ax = plt.subplot(gs[0])
    q = ax.hist(mcmc_sample[:,2], bins=20,
                range=[plot_range[2,:].min(),
                       plot_range[2,:].max()],
                normed=True, alpha=1., color=mcmclr,
                linewidth=4, histtype='step')
    qq = ax.hist(abc_sample[:,2], bins=20,
                 range=[plot_range[2,:].min(),
                        plot_range[2,:].max()],
                 normed=True, alpha=1., color=abclr,
                 linewidth=4, histtype='step')
#     ax.vlines(truths[2], 0, max(q[0].max(),qq[0].max()),
#               color = "k", linewidth = 5)
    ax.axvline(truths[2], color='k', linewidth=5)
    ax.set_xticklabels([])
    ax.set_xlim([plot_range[2,0] , plot_range[2,1]])

    ax = plt.subplot(gs[1])
    q = ax.hist(mcmc_sample[:,3], bins=20,
                range=[plot_range[3,:].min(),
                       plot_range[3,:].max()],
                normed=True, alpha=1., color=mcmclr,
                linewidth=4, histtype='step')
    qq = ax.hist(abc_sample[:,3], bins=20,
                 range=[plot_range[3,:].min(),
                        plot_range[3,:].max()],
                 normed=True, alpha=1., color=abclr,
                 linewidth=4, histtype='step')
#     ax.vlines(truths[3], 0, max(q[0].max(),qq[0].max()),
#               color="k", linewidth=5)
    ax.axvline(truths[3], color='k', linewidth=5)
    ax.set_xticklabels([])
    ax.set_xlim([plot_range[3,0], plot_range[3,1]])

    ax = plt.subplot(gs[2])
    q = ax.hist(mcmc_sample[:,4], bins=20,
                range=[plot_range[4,:].min(),
                       plot_range[4,:].max()],
                normed=True, alpha=1., color=mcmclr,
                linewidth=4, histtype='step')
    qq = ax.hist(abc_sample[:,4], bins=20,
                 range=[plot_range[4,:].min(),
                        plot_range[4,:].max()],
                 normed=True, alpha=1., color=abclr,
                 linewidth=4, histtype='step')
#     ax.vlines(truths[4], 0, max(q[0].max(),qq[0].max()),
#               colors='k', linewidth=5)
    ax.axvline(truths[4], color='k', linewidth=5)
    ax.set_xticklabels([])
    ax.set_xlim([plot_range[4,0], plot_range[4,1]])

    thick_line1 = mlines.Line2D([], [], ls='-', c=abclr, linewidth=4,
                                label='ABC-PMC')
    thick_line2 = mlines.Line2D([], [], ls='-', c=mcmclr, linewidth=4,
                                label=r'Gaussian $\mathcal{L}$ +MCMC')

    ax.legend(handles=[thick_line2, thick_line1],
              frameon=False, loc='best', fontsize=30)

    # and now the box plots below

    # general box properties
    boxprops = {'color': 'k'}
    medianprops = {'alpha': 0.}
    bplots1 = []
    ax = plt.subplot(gs[3])
#     bplots.append(ax.boxplot(mcmc_sample[:,2],
#                              vert=False, patch_artist=True))
#     bplots.append(ax.boxplot(abc_sample[:,2],
#                              vert=False, patch_artist=True))
    # stats dict for each box
    bplots1.append({'med': np.median(mcmc_sample[:, 2]),
                   'q1': np.sort(mcmc_sample[:, 2])[sig1lo_mcmc],
                   'q3': np.sort(mcmc_sample[:, 2])[sig1hi_mcmc],
                   'whislo': np.sort(mcmc_sample[:, 2])[sig2lo_mcmc],
                   'whishi': np.sort(mcmc_sample[:, 2])[sig2hi_mcmc],
                   'fliers': []})
    bplots1.append({'med': np.median(abc_sample[:, 2]),
                   'q1': np.sort(abc_sample[:, 2])[sig1lo_abc],
                   'q3': np.sort(abc_sample[:, 2])[sig1hi_abc],
                   'whislo': np.sort(abc_sample[:, 2])[sig2lo_abc],
                   'whishi': np.sort(abc_sample[:, 2])[sig2hi_abc],
                   'fliers': []})

    bxp1 = ax.bxp(bplots1, positions=[1,2], vert=False, patch_artist=True,
                  showfliers=False, boxprops=boxprops, medianprops=medianprops)
                  # boxprops=boxprops, medianprops=medianprops, sym='')

    for i, box in enumerate(bxp1['boxes']):
        if i == 0:
            box.set(facecolor=mcmclr, alpha=0.5)
        elif i == 1:
            box.set(facecolor=abclr, alpha=0.5)
#             for patch in bplot['boxes']:
#                 patch.set_facecolor(abclr)
#                 patch.set_alpha(0.5)

#     ax.vlines(truths[2], 0, ax.get_ylim(), # max(q[0].max(),qq[0].max()),
#               color = "k", linewidth = 5)
    ax.axvline(truths[2], color='k', linewidth=5)

    ax.set_xlim([plot_range[2,0] , plot_range[2,1]])
    ax.set_xlabel(r'$\log M_{\rm min}$', fontsize = 50)

    ax.set_yticks([1,2])
    ax.set_yticklabels(["MCMC", "ABC"])

    ax = plt.subplot(gs[4])
#     bplots.append(ax.boxplot(mcmc_sample[:,3],
#                              vert=False, patch_artist=True))
#     bplots.append(ax.boxplot(abc_sample[:,3],
#                              vert=False, patch_artist=True))
    bplots2 = []
    bplots2.append({'med': np.median(mcmc_sample[:, 3]),
                   'q1': np.sort(mcmc_sample[:, 3])[sig1lo_mcmc],
                   'q3': np.sort(mcmc_sample[:, 3])[sig1hi_mcmc],
                   'whislo': np.sort(mcmc_sample[:, 3])[sig2lo_mcmc],
                   'whishi': np.sort(mcmc_sample[:, 3])[sig2hi_mcmc],
                   'fliers': []})
    bplots2.append({'med': np.median(abc_sample[:, 3]),
                   'q1': np.sort(abc_sample[:, 3])[sig1lo_abc],
                   'q3': np.sort(abc_sample[:, 3])[sig1hi_abc],
                   'whislo': np.sort(abc_sample[:, 3])[sig2lo_abc],
                   'whishi': np.sort(abc_sample[:, 3])[sig2hi_abc],
                   'fliers': []})
#     for i, bplot in enumerate(bplots[2:]):
#         if i == 0:
#             for patch in bplot['boxes']:
#                 patch.set_facecolor(mcmclr)
#                 patch.set_alpha(0.5)
#         elif i == 1:
#             for patch in bplot['boxes']:
#                 patch.set_facecolor(abclr)
#                 patch.set_alpha(0.5)
    bxp2 = ax.bxp(bplots2, positions=[1,2], vert=False, patch_artist=True,
                  showfliers=False, boxprops=boxprops, medianprops=medianprops)

    for i, box in enumerate(bxp2['boxes']):
        if i == 0:
            box.set(facecolor=mcmclr, alpha=0.5)
        elif i == 1:
            box.set(facecolor=abclr, alpha=0.5)

#     ax.vlines(truths[3], 0, ax.get_ylim(), # max(q[0].max(),qq[0].max()),
#               color = "k" , linewidth = 5)
    ax.axvline(truths[3], color='k', linewidth=5)

    ax.set_xlim([plot_range[3,0], plot_range[3,1]])
    ax.set_xlabel(r'$\alpha$', fontsize = 50)
    ax.set_yticks([])

    ax = plt.subplot(gs[5])
#     bplots.append(ax.boxplot(mcmc_sample[:,4],
#                              vert=False, patch_artist=True))
#     bplots.append(ax.boxplot(abc_sample[:,4],
#                              vert=False, patch_artist=True))
#     for i, bplot in enumerate(bplots[4:]):
#         if i == 0:
#             for patch in bplot['boxes']:
#                 patch.set_facecolor(mcmclr)
#                 patch.set_alpha(0.5)
#         elif i == 1:
#             for patch in bplot['boxes']:
#                 patch.set_facecolor(abclr)
#                 patch.set_alpha(0.5)
    bplots3 = []
    bplots3.append({'med': np.median(mcmc_sample[:, 4]),
                   'q1': np.sort(mcmc_sample[:, 4])[sig1lo_mcmc],
                   'q3': np.sort(mcmc_sample[:, 4])[sig1hi_mcmc],
                   'whislo': np.sort(mcmc_sample[:, 4])[sig2lo_mcmc],
                   'whishi': np.sort(mcmc_sample[:, 4])[sig2hi_mcmc],
                   'fliers': []})
    bplots3.append({'med': np.median(abc_sample[:, 4]),
                   'q1': np.sort(abc_sample[:, 4])[sig1lo_abc],
                   'q3': np.sort(abc_sample[:, 4])[sig1hi_abc],
                   'whislo': np.sort(abc_sample[:, 4])[sig2lo_abc],
                   'whishi': np.sort(abc_sample[:, 4])[sig2hi_abc],
                   'fliers': []})

    bxp3 = ax.bxp(bplots3, positions=[1,2], vert=False, patch_artist=True,
                  showfliers=False, boxprops=boxprops, medianprops=medianprops)

    for i, box in enumerate(bxp3['boxes']):
        if i == 0:
            box.set(facecolor=mcmclr, alpha=0.5)
        elif i == 1:
            box.set(facecolor=abclr, alpha=0.5)

#     ax.vlines(truths[4] , 0, ax.get_ylim(), # max(q[0].max(),qq[0].max()),
#               colors='k' , linewidth = 5)
    ax.axvline(truths[4], color='k', linewidth=5)

    ax.set_xlim([plot_range[4,0], plot_range[4,1]])
    ax.set_xlabel(r'$\log M_{1}$', fontsize = 50)
    ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.0)

    plt.savefig("histograms_nbarxi2_boxplot_fix.pdf")
    return None



if __name__ == "__main__":

    mcmc_filename = "results/nbar_xi.mcmc.mcmc_chain.dat"
    abc_filename  = "results/nbar_xi_theta_t8.abc.dat"
    #abc_filename = "results/nbar_gmf_theta_t8.ABCnbargmf.dat"
    overlay_pdfs_contours(abc_filename , mcmc_filename , 100 , 6000 , 21)
