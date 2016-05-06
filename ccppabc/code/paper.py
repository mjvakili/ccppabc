'''

Plots and calculations directly for the paper 

Authors: Chang and MJ 
'''
import os
import corner
import pickle
import numpy as np 

import util as ut
import data as Data
from prior import PriorRange

from hod_sim import ABC_HODsim

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import colorConverter
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


## ABC figures 
''' Figures for the ABC Results section 
'''

def PoolEvolution(obvs): 
    ''' Demostrative plot for the evolution of the pool. Illustrate the pool evolution for
    log M_min versus log M_1, which has the starkest evolution from its prior. 

    '''
    if obvs == 'nbargmf':
        result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
        theta_file = lambda tt: ''.join([result_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
        t_list = [0, 1, 2, 3, 5, 8]
    elif obvs == 'nbarxi':
        result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/'])
        theta_file = lambda tt: ''.join([result_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat']) 
        t_list = [0, 1, 2, 3, 7, 9]
    else:
        raise ValueError
    
    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((2,2))
    prior_range[:,0] = np.array([prior_min[2], prior_min[4]])
    prior_range[:,1] = np.array([prior_max[2], prior_max[4]])

    # true HOD parameter
    true_dict = Data.data_hod_param(Mr=21)
    true_pair = [true_dict['logMmin'], true_dict['logM1']]

    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(figsize=(12,8))
    all_fig = fig.add_subplot(111) 

    for i_t, t in enumerate(t_list): 
        sub = fig.add_subplot(2, len(t_list)/2, i_t+1)

        theta_Mmin, theta_M1 = np.loadtxt(theta_file(t), unpack=True, usecols=[2, 4]) 
        corner.hist2d(theta_Mmin, theta_M1, bins=20, range=prior_range, 
                levels=[0.68, 0.95], color='c', fill_contours=True, smooth=1.0)
        
        t_label = r"$\mathtt{t = "+str(t)+"}$"
        sub.text(13.0, 15.0, t_label, fontsize=25) 

        if i_t == len(t_list) - 1: 
            true_label = r'$``\mathtt{true}"$'
        else: 
            true_label = None 

        plt.scatter(np.repeat(true_pair[0],2), np.repeat(true_pair[1],2), 
                s=75, marker='*', c='k', lw=0, label=true_label) 
        if i_t == len(t_list) - 1: 
            plt.legend(loc='lower left', scatterpoints=1, markerscale=2.5, 
                    handletextpad=-0.25, scatteryoffsets=[0.5])

        if i_t == 0: 
            sub.set_xticklabels([])
            sub.set_yticklabels([13., 13.5, 14., 14.5, 15., 15.5])
        elif (i_t > 0) and (i_t < len(t_list)/2): 
            sub.set_xticklabels([])
            sub.set_yticklabels([])
        elif i_t == len(t_list)/2: 
            sub.set_yticklabels([13., 13.5, 14., 14.5, 15.])
        elif i_t > len(t_list)/2: 
            sub.set_yticklabels([])

    all_fig.set_xticklabels([])
    all_fig.set_yticklabels([])
    all_fig.set_ylabel(
            r'$\mathtt{log}\;\mathcal{M}_\mathtt{1}$', 
            fontsize=30, labelpad=50)
    all_fig.set_xlabel(
            r'$\mathtt{log}\;\mathcal{M}_\mathtt{min}$', 
            fontsize=30, labelpad=25)

    fig.subplots_adjust(hspace=0.0)
    fig_name = ''.join([ut.fig_dir(), 
        'paper_ABC_poolevolution', 
        '.', obvs, 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    plt.close()


def PosteriorObservable(Mr=21, b_normal=0.25, clobber=False):
    ''' Plot 1\sigma and 2\sigma model predictions from ABC-PMC posterior likelihood
    '''
    prettyplot()
    pretty_colors=prettycolors()
    fig = plt.figure(1, figsize=(16,12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[1,1]) 

    for obvs in ['nbargmf', 'nbarxi']: 

        if obvs == 'nbargmf':
            result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
            theta_file = lambda tt: ''.join([result_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
            tf = 8 
            obvs_list = ['gmf']
        elif obvs == 'nbarxi':
            result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/'])
            theta_file = lambda tt: ''.join([result_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat']) 
            tf = 9
            obvs_list = ['xi']
        else:
            raise ValueError

        theta = np.loadtxt(theta_file(tf))  # import thetas
        #theta = theta[:10]
        
        obvs_file = ''.join(theta_file(tf).rsplit('.dat')[:-1] + ['.', obvs_list[0], '.p'])
        print obvs_file
        
        HODsimulator = ABC_HODsim(Mr=Mr, b_normal=b_normal)
        if not os.path.isfile(obvs_file) or clobber:
            model_obv = [] 
            for i in xrange(len(theta)):
                print i 
                obv_i = HODsimulator(
                        theta[i], 
                        prior_range=None,
                        observables=obvs_list)
                model_obv.append(obv_i[0])
            model_obv = np.array(model_obv)
            pickle.dump(model_obv, open(obvs_file, 'wb'))
        else:
            model_obv = pickle.load(open(obvs_file, 'rb'))

        if 'xi' in obvs:
            r_bin = Data.data_xi_bin(Mr=Mr)
        elif 'gmf' in obvs:
            r_binedge = Data.data_gmf_bins()
            r_bin = 0.5 * (r_binedge[:-1] + r_binedge[1:]) 

        a, b, c, d, e = np.percentile(model_obv, [2.5, 16, 50, 84, 97.5], axis=0)

        # plotting
        if obvs == 'nbarxi': 
            ax = plt.subplot(gs[0])
        elif obvs == 'nbargmf': 
            ax = plt.subplot(gs[1])

        if 'xi' in obvs:  # 2PCF
            xi_data = Data.data_xi(Mr=Mr, b_normal=b_normal)
            cov_data = Data.data_cov(Mr=Mr, b_normal=b_normal, inference='mcmc') 
            data_xi_cov = cov_data[1:16, 1:16]

            ax.fill_between(r_bin, a, e, color=pretty_colors[3], alpha=0.3, edgecolor="none")
            ax.fill_between(r_bin, b, d, color=pretty_colors[3], alpha=0.5, edgecolor="none")
            ax.errorbar(r_bin, xi_data, yerr = np.sqrt(np.diag(data_xi_cov)), fmt="o", color='k', 
                    markersize=0, lw=0, capsize=3, elinewidth=1.5)
            ax.scatter(r_bin, xi_data, c='k', s=10, lw=0)
            ax.set_ylabel(r'$\xi_\mathtt{gg}(\mathtt{r})$', fontsize=27)
            ax.set_yscale('log') 
            ax.set_xscale('log')
            ax.set_xticklabels([])
            ax.set_xlim([0.1, 20.])
            ax.set_ylim([0.09, 1000.])

            ax = plt.subplot(gs[2])
            ax.fill_between(r_bin, a/xi_data, e/xi_data, color=pretty_colors[3], alpha=0.3, edgecolor="none")
            ax.fill_between(r_bin, b/xi_data, d/xi_data, color=pretty_colors[3], alpha=0.5, edgecolor="none")
            ax.errorbar(r_bin, np.repeat(1., len(xi_data)), yerr=np.sqrt(np.diag(data_xi_cov))/xi_data, 
                    fmt="o", color='k', markersize=0, lw=0, capsize=3, elinewidth=1.5)
            ax.plot(np.arange(0.1, 20., 0.1), np.repeat(1., len(np.arange(0.1, 20, 0.1))), c='k', ls='--', lw=2)
            ax.set_xlim([0.1, 20.])
            ax.set_xscale('log') 
            ax.set_ylim([0.5, 1.5]) 
            ax.set_xlabel(r'$\mathtt{r}\;[\mathtt{Mpc}/h]$', fontsize=25)
            ax.set_ylabel(r'$\xi_\mathtt{gg}/\xi_\mathtt{gg}^\mathtt{obvs}$', fontsize=25)

        elif 'gmf' in obvs:   # GMF
            data_gmf = Data.data_gmf(Mr=Mr, b_normal=b_normal)
            cov_data = Data.data_cov(Mr=Mr, b_normal=b_normal, inference='mcmc') 
            data_gmf_cov = cov_data[16:, 16:]

            ax.fill_between(r_bin, a, e, color=pretty_colors[3], alpha=0.3, edgecolor="none")
            ax.fill_between(r_bin, b, d, color=pretty_colors[3], alpha=0.5, edgecolor="none", label='ABC Posterior')
            ax.errorbar(r_bin, data_gmf, yerr=np.sqrt(np.diag(data_gmf_cov)), fmt="o", color='k', 
                    markersize=0, lw=0, capsize=4, elinewidth=2)
            ax.scatter(r_bin, data_gmf, s=15, lw=0, c='k', label='Mock Observation')
            ax.legend(loc='upper right', scatterpoints=1, prop={'size': 25}, borderpad=1.0)

            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_label_position('right') 
            ax.set_ylabel(r'$\zeta$ $[(\mathrm{h}/\mathtt{Mpc})^{3}]$', fontsize=25)

            ax.set_yscale('log')
            ax.set_xlim([1., 20.])
            ax.set_xticklabels([])
            ax.set_ylim([10.**-7.2, 2*10**-4.])

            ax = plt.subplot(gs[3])
            ax.fill_between(r_bin, a/data_gmf, e/data_gmf, color=pretty_colors[3], alpha=0.3, edgecolor="none")
            ax.fill_between(r_bin, b/data_gmf, d/data_gmf, color=pretty_colors[3], alpha=0.5, edgecolor="none")

            ax.errorbar(r_bin, np.repeat(1., len(data_gmf)), yerr=np.sqrt(np.diag(data_gmf_cov))/data_gmf, 
                    fmt="o", color='k', markersize=0, lw=0, capsize=3, elinewidth=1.5)
            ax.plot(np.arange(1., 20., 1), np.repeat(1., len(np.arange(1., 20, 1))), c='k', ls='--', lw=1.75)

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right') 
            ax.set_ylim([-0.1, 2.1])
            ax.set_ylabel(r'$\zeta/\zeta^\mathtt{obvs}$', fontsize=25)
            ax.set_xlim([1., 20.])
            ax.set_xlabel(r'$\mathtt{N}$ [Group Richness]', fontsize=25)

    fig.subplots_adjust(wspace=0.05, hspace=0.0)
    fig_name = ''.join([ut.fig_dir(), 
        'paper', 
        '.ABCposterior', 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


if __name__=="__main__": 
    #PosteriorObservable(Mr=21, b_normal=0.25)
    PoolEvolution('nbargmf')
    PoolEvolution('nbarxi')
