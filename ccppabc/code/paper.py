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
from matplotlib import lines as mlines
from matplotlib import gridspec
from scipy.stats import norm
from matplotlib.colors import colorConverter
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def TrueObservables(Mr=21, b_normal=0.25): 
    ''' Plot xi and gmf for the data 
    '''
    # xi data 
    xi_data = Data.data_xi(Mr=Mr, b_normal=b_normal)
    cov_data = Data.data_cov(Mr=Mr, b_normal=b_normal, inference='mcmc') 
    data_xi_cov = cov_data[1:16, 1:16]
    xi_r_bin = Data.data_xi_bin(Mr=Mr)
    # gmf data 
    data_gmf = Data.data_gmf(Mr=Mr, b_normal=b_normal)
    cov_data = Data.data_cov(Mr=Mr, b_normal=b_normal, inference='mcmc') 
    data_gmf_cov = cov_data[16:, 16:]

    r_binedge = Data.data_gmf_bins()
    gmf_r_bin = 0.5 * (r_binedge[:-1] + r_binedge[1:]) 
   
    prettyplot()
    pretty_colors = prettycolors() 

    fig = plt.figure(figsize=(12,6))
    sub_xi = fig.add_subplot(121) 
    sub_gmf = fig.add_subplot(122)
   
    #sub_xi.errorbar(xi_r_bin, xi_data, yerr = np.sqrt(np.diag(data_xi_cov)), fmt="o", color='k', 
    #        markersize=0, lw=0, capsize=3, elinewidth=1.5)
    #sub_xi.scatter(xi_r_bin, xi_data, c='k', s=10, lw=0)
    sub_xi.fill_between(xi_r_bin, 
            xi_data-np.sqrt(np.diag(data_xi_cov)), xi_data+np.sqrt(np.diag(data_xi_cov)), 
            color=pretty_colors[1])
    sub_xi.set_yscale('log')
    sub_xi.set_xscale('log')
    sub_xi.set_xlim(0.1, 20)
    sub_xi.set_xlabel(r'$\mathtt{r}\; [\mathtt{Mpc}/h]$', fontsize=25)
    sub_xi.set_ylabel(r'$\xi(r)$', fontsize=25)

    #sub_gmf.errorbar(gmf_r_bin, data_gmf, yerr=np.sqrt(np.diag(data_gmf_cov)), 
    #        fmt="o", color='k', 
    #        markersize=0, lw=0, capsize=4, elinewidth=2)
    #sub_gmf.scatter(gmf_r_bin, data_gmf, s=15, lw=0, c='k', label='Mock Observation')
    sub_gmf.fill_between(gmf_r_bin, 
            data_gmf-np.sqrt(np.diag(data_gmf_cov)), data_gmf+np.sqrt(np.diag(data_gmf_cov)), 
            color=pretty_colors[1])
    sub_gmf.set_xlim(1, 20)
    sub_gmf.set_xlabel(r'$\mathtt{N}$ (Group Richness)', fontsize=25)

    sub_gmf.yaxis.tick_right()
    sub_gmf.yaxis.set_ticks_position('both')
    sub_gmf.yaxis.set_label_position('right') 
    sub_gmf.set_ylim([10**-7, 2.0*10**-4])
    sub_gmf.set_yscale('log')
    sub_gmf.set_ylabel(r'$\zeta(\mathtt{N})$', fontsize=25)

    fig.subplots_adjust(hspace=0.05)
    fig_name = ''.join([ut.fig_dir(), 
        'paper.data_observables', 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


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


def ABC_Coner(obvs, weighted=False): 
    ''' Pretty corner plot 
    '''
    if obvs == 'nbargmf':
        result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
        theta_file = lambda tt: ''.join([result_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
        w_file = lambda tt: ''.join([result_dir, 'nbar_gmf_w_t', str(tt), '.ABCnbargmf.dat'])
        tf = 8 
    elif obvs == 'nbarxi':
        result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/'])
        theta_file = lambda tt: ''.join([result_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat']) 
        w_file = lambda tt: ''.join([result_dir, 'nbar_xi_w_t', str(tt), '.abc.dat']) 
        tf = 9 
    else:
        raise ValueError

    theta = np.loadtxt(theta_file(tf)) 
    if weighted: 
        weights = np.loadtxt(w_file(tf))
    else: 
        weights = None

    true_dict = Data.data_hod_param(Mr=21)
    true_theta = [
            true_dict['logM0'], 
            np.log(true_dict['sigma_logM']), 
            true_dict['logMmin'], 
            true_dict['alpha'], 
            true_dict['logM1']
            ]

    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = np.array(prior_min)
    prior_range[:,1] = np.array(prior_max) 

    fig = corner.corner(
            theta,
            weights=weights,
            truths=true_theta,
            truth_color='k',
            labels=[
                r'$\log\;\mathcal{M}_{0}}$',
                r'$\log\;\sigma_\mathtt{log\;M}}$',
                r'$\log\;\mathcal{M}_\mathtt{min}}$',
                r'$\alpha$',
                r'$\log\;\mathcal{M}_{1}}$'
                ],
            label_kwargs={'fontsize': 25},
            range=prior_range,
            quantiles=[0.16,0.5,0.84],
            show_titles=True,
            title_args={"fontsize": 12},
            plot_datapoints=True,
            fill_contours=True,
            levels=[0.68, 0.95],
            color='#ee6a50',
            bins=20,
            smooth=1.0)
    
    fig_name = ''.join([ut.fig_dir(), 
        'paper.ABCcorner', 
        '.', obvs, 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    plt.close()
    return None 

    
def ABC_Convergence(weighted=False): 
    ''' Plot the error bars on the parameters as a function of time step
    '''
    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure(figsize=(20, 10)) 
    
    for i_obv, obvs in enumerate(['nbargmf', 'nbarxi']): 
        if obvs == 'nbargmf':
            result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
            theta_file = lambda tt: ''.join([result_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
            w_file = lambda tt: ''.join([result_dir, 'nbar_gmf_w_t', str(tt), '.ABCnbargmf.dat'])
            tf = 8 
        elif obvs == 'nbarxi':
            result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/'])
            theta_file = lambda tt: ''.join([result_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat']) 
            w_file = lambda tt: ''.join([result_dir, 'nbar_xi_w_t', str(tt), '.abc.dat']) 
            tf = 9 
        else:
            raise ValueError
    
        t_list = range(tf+1) 
        
        columns = [
                r"$\mathtt{log}\;\mathcal{M}_0$", 
                r"$\sigma_{\mathtt{log}\;\mathcal{M}}$",
                r"$\mathtt{log}\;\mathcal{M}_\mathtt{min}$", 
                r"$\alpha$", 
                r"$\mathtt{log}\;\mathcal{M}_1$"
                ]
        
        true_dict = Data.data_hod_param(Mr=21)
        true_theta = [
                true_dict['logM0'], 
                np.log(true_dict['sigma_logM']), 
                true_dict['logMmin'], 
                true_dict['alpha'], 
                true_dict['logM1']
                ]

        prior_min, prior_max = PriorRange('first_try')

        a_theta = np.zeros((len(t_list), 5)) 
        b_theta = np.zeros((len(t_list), 5)) 
        d_theta = np.zeros((len(t_list), 5)) 
        e_theta = np.zeros((len(t_list), 5)) 
        for i_t, tt in enumerate(t_list): 
            theta_i = np.loadtxt(theta_file(tt), unpack=True) 
            w_i = np.loadtxt(w_file(tt))
            for i_par in range(len(theta_i)): 
                if not weighted: 
                    a, b, d, e = np.percentile(theta_i[i_par], [2.5, 16, 84, 97.5], axis=0)
                else: 
                    a = ut.quantile_1D(theta_i[i_par], w_i, 0.025)
                    b = ut.quantile_1D(theta_i[i_par], w_i, 0.16)
                    d = ut.quantile_1D(theta_i[i_par], w_i, 0.84)
                    e = ut.quantile_1D(theta_i[i_par], w_i, 0.975)

                a_theta[i_t, i_par] = a
                b_theta[i_t, i_par] = b
                d_theta[i_t, i_par] = d 
                e_theta[i_t, i_par] = e 

        keep_index = [2,3,4]
        for ii, i in enumerate(keep_index):
            if i == keep_index[-1]: 
                true_label = r'$``\mathtt{true}"$'
                abc_1sig_label = r'ABC Pool'
            else: 
                true_label = None
                abc_1sig_label = None

            sub = fig.add_subplot(2, len(keep_index), i_obv * len(keep_index) + ii+1)
            sub.fill_between(t_list, a_theta[:, i], e_theta[:,i], 
                    color=pretty_colors[3], alpha=0.3, edgecolor="none")
            sub.fill_between(t_list, b_theta[:, i], d_theta[:,i], 
                    color=pretty_colors[3], alpha=0.5, edgecolor="none", 
                    label=abc_1sig_label)
            sub.plot(t_list, np.repeat(true_theta[i], len(t_list)), c='k', ls='--', lw=2, 
                    label=true_label)


            if ii == 0: 
                if obvs == 'nbargmf': 
                    sub.text(4.85, 13.4, r"$\bar{\mathtt{n}}$ and $\zeta(\mathtt{N})$", fontsize=25) 
                elif obvs == 'nbarxi': 
                    sub.text(4.85, 13.4, r"$\bar{\mathtt{n}}$ and $\xi(\mathtt{r})$", fontsize=25) 

            sub.set_ylabel(columns[i], fontsize=25)
            sub.set_ylim([prior_min[i], prior_max[i]]) 
    
            sub.set_xlim([-1, 10])
            if i_obv == 1:  
                sub.legend(loc='upper right', borderpad=1.) 
                sub.set_xlabel('iterations', fontsize=25)  
                if i == 4: 
                    sub.set_yticklabels([13.0, 13.5, 14.0, 14.5, 15.])
            else: 
                sub.set_xticklabels([])

    fig.subplots_adjust(wspace=0.3, hspace=0.0)
    if weighted: 
        weight_str = '.weighted'
    else: 
        weight_str = ''

    fig_name = ''.join([ut.fig_dir(), 
        'paper', 
        '.ABCconvergence', 
        weight_str, 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    plt.close() 
    return None



def ABCvsMCMC_histogram(obvs, nwalkers=100, nburns=9000):  
    ''' Plots that compare the ABC posteriors to the MCMC posteriors 
    '''
    if obvs == 'nbargmf':
        abc_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
        abc_theta_file = lambda tt: ''.join([abc_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
        tf = 8
        mcmc_dir = ''.join([ut.dat_dir(), 'paper/']) 
        mcmc_filename = ''.join([mcmc_dir, 'nbar_gmf.mcmc.mcmc_chain.p'])
    elif obvs == 'nbarxi': 
        abc_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/',])
        abc_theta_file = lambda tt: ''.join([abc_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat'])
        tf = 9
        mcmc_dir = ''.join([ut.dat_dir(), 'paper/']) 
        mcmc_filename = ''.join([mcmc_dir, 'nbar_xi.mcmc.mcmc_chain.p'])
    else: 
        raise ValueError

    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min 
    prior_range[:,1] = prior_max 

    # true HOD parameter
    true_dict = Data.data_hod_param(Mr=21)
    truths = [
            true_dict['logM0'],                 # log M0
            np.log(true_dict['sigma_logM']),    # log(sigma)
            true_dict['logMmin'],               # log Mmin
            true_dict['alpha'],                 # alpha
            true_dict['logM1']                  # log M1
            ]
    
    mcmc_sample = pickle.load(open(mcmc_filename, 'rb'))[nburns*nwalkers:,:]
    abc_sample = np.loadtxt(abc_theta_file(tf))

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

    par_labels = [
                r'$\mathtt{log}\;\mathcal{M}_{0}$',
                r'$\mathtt{log}\;\sigma_\mathtt{log\;M}$',
                r'$\mathtt{log}\;\mathcal{M}_\mathtt{min}$',
                r'$\alpha$',
                r'$\mathtt{log}\;\mathcal{M}_{1}$'
                ]

    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(1, figsize=(20,8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.75, 1])
    
    # first panel 
    for i in [0, 1, 2]:
        if i == 0: 
            i_col = 2
            prior_range[i_col,0] = 12.5
            prior_range[i_col,1] = 13.
            plot_range = prior_range[i_col, :]
        elif i == 1: 
            i_col = 3 
            plot_range = prior_range[i_col, :]
        elif i == 2:
            i_col = 4
            prior_range[i_col,0] = 13.5
            prior_range[i_col,1] = 14.25
            plot_range = np.array([13.5, 14.5])

        ax = plt.subplot(gs[i])
        q = ax.hist(mcmc_sample[:,i_col], bins=20,
                    range=[prior_range[i_col,0], prior_range[i_col,1]], 
                    normed=True, 
                    alpha=0.75, 
                    color=pretty_colors[1],
                    linewidth=2, 
                    histtype='stepfilled',
                    edgecolor=None
                    )
        qq = ax.hist(abc_sample[:,i_col], bins=20,
                     range=[prior_range[i_col,0], prior_range[i_col,1]],
                     normed=True, 
                     alpha=0.75, 
                     color=pretty_colors[3],
                     linewidth=2, 
                     histtype='stepfilled', 
                     edgecolor=None
                     )
        ax.axvline(truths[i_col], color='k', ls='--', linewidth=3)
        ax.set_xticklabels([])
        ax.set_xlim([plot_range[0] , plot_range[1]])
        
        if i == 2: 
            thick_line2 = mlines.Line2D([], [], ls='-', c=pretty_colors[1], linewidth=4,
                                        label='$\mathcal{L}^\mathtt{Gauss}$ \nMCMC')
            thick_line1 = mlines.Line2D([], [], ls='-', c=pretty_colors[3], linewidth=4,
                                        label='ABC-PMC')

            ax.legend(loc='upper right', handles=[thick_line2, thick_line1],
                      frameon=False, fontsize=25, handletextpad=-0.0)

        # general box properties
        boxprops = {'color': 'k'}
        medianprops = {'alpha': 0.}
        bplots1 = []
        ax = plt.subplot(gs[i+3])
        # stats dict for each box
        bplots1.append({'med': np.median(mcmc_sample[:, i_col]),
                       'q1': np.sort(mcmc_sample[:, i_col])[sig1lo_mcmc],
                       'q3': np.sort(mcmc_sample[:, i_col])[sig1hi_mcmc],
                       'whislo': np.sort(mcmc_sample[:, i_col])[sig2lo_mcmc],
                       'whishi': np.sort(mcmc_sample[:, i_col])[sig2hi_mcmc],
                       'fliers': []})
        bplots1.append({'med': np.median(abc_sample[:, i_col]),
                       'q1': np.sort(abc_sample[:, i_col])[sig1lo_abc],
                       'q3': np.sort(abc_sample[:, i_col])[sig1hi_abc],
                       'whislo': np.sort(abc_sample[:, i_col])[sig2lo_abc],
                       'whishi': np.sort(abc_sample[:, i_col])[sig2hi_abc],
                       'fliers': []})
        whiskprop = dict(linestyle='-', linewidth=2, color='k') 
        boxprops = dict(linestyle='-', linewidth=2, color='k')
        bxp1 = ax.bxp(bplots1, positions=[1,2], vert=False, patch_artist=True, 
                      showfliers=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskprop)

        for ibox, box in enumerate(bxp1['boxes']):
            if ibox == 0:
                box.set(facecolor=pretty_colors[1], alpha=0.75)
            elif ibox == 1:
                box.set(facecolor=pretty_colors[3], alpha=0.75)
        ax.axvline(truths[i_col], color='k', ls='--', linewidth=3)

        ax.set_xlim([plot_range[0] , plot_range[1]])
        ax.set_xlabel(par_labels[i_col], fontsize=25, labelpad=15)
        
        if i == 0: 
            ax.set_yticks([1,2])
            ax.set_yticklabels([r"$\mathtt{MCMC}$", r"$\mathtt{ABC}$"])
        else: 
            ax.set_yticks([])

    fig.subplots_adjust(wspace=0.2, hspace=0.0)
    fig_name = ''.join([ut.fig_dir(), 
        'paper.ABCvsMCMC', 
        '.', obvs, 
        '.pdf'])
    print fig_name 
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    return None 
    #plt.show()


def ABCvsMCMC_contour(obvs, nwalkers=100, nburns=9000):  
    ''' Plots that compare the ABC posteriors to the MCMC posteriors 
    '''
    if obvs == 'nbargmf':
        abc_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/run1/',])
        abc_theta_file = lambda tt: ''.join([abc_dir, 'nbar_gmf_theta_t', str(tt), '.ABCnbargmf.dat']) 
        tf = 8
        mcmc_dir = ''.join([ut.dat_dir(), 'paper/']) 
        mcmc_filename = ''.join([mcmc_dir, 'nbar_gmf.mcmc.mcmc_chain.p'])
    elif obvs == 'nbarxi': 
        abc_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/',])
        abc_theta_file = lambda tt: ''.join([abc_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat'])
        tf = 9
        mcmc_dir = ''.join([ut.dat_dir(), 'paper/']) 
        mcmc_filename = ''.join([mcmc_dir, 'nbar_xi.mcmc.mcmc_chain.p'])
    else: 
        raise ValueError

    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min 
    prior_range[:,1] = prior_max 

    # true HOD parameter
    true_dict = Data.data_hod_param(Mr=21)
    truths = [
            true_dict['logM0'],                 # log M0
            np.log(true_dict['sigma_logM']),    # log(sigma)
            true_dict['logMmin'],               # log Mmin
            true_dict['alpha'],                 # alpha
            true_dict['logM1']                  # log M1
            ]
    
    mcmc_sample = pickle.load(open(mcmc_filename, 'rb'))[nburns*nwalkers:,:]
    abc_sample = np.loadtxt(abc_theta_file(tf))

    par_labels = [
                r'$\mathtt{log}\;\mathcal{M}_{0}$',
                r'$\mathtt{log}\;\sigma_\mathtt{log\;M}$',
                r'$\mathtt{log}\;\mathcal{M}_\mathtt{min}$',
                r'$\alpha$',
                r'$\mathtt{log}\;\mathcal{M}_{1}$'
                ]

    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(1, figsize=(20,6))
    gs = gridspec.GridSpec(1, 3)
    
    # first panel 
    for i in [0, 1, 2]:
        plot_range = np.zeros((2,2))
        if i == 0: 
            col_pair = [2, 3] 
            plot_range[0, 0] = 12.5
            plot_range[0, 1] = 13.0
            plot_range[1, 0] = prior_range[3, 0] 
            plot_range[1, 1] = prior_range[3, 1] 
        elif i == 2: 
            col_pair = [4, 2] 
            plot_range[0, 0] = 13.6
            plot_range[0, 1] = 14.2
            plot_range[1, 0] = 12.5
            plot_range[1, 1] = 13.0
        elif i == 1:
            col_pair = [3, 4] 
            plot_range[0, 0] = prior_range[3, 0]
            plot_range[0, 1] = prior_range[3, 1] 
            plot_range[1, 0] = 13.6
            plot_range[1, 1] = 14.2

        if i == 2: 
            mcmc_label = r'$\mathcal{L}^\mathtt{Gauss}$ MCMC' 
            abc_label = 'ABC-PMC'
        else: 
            mcmc_label = None
            abc_label = None
        
        mcmc_par1 = mcmc_sample[:,col_pair[0]]
        mcmc_par2 = mcmc_sample[:,col_pair[1]]

        abc_par1 = abc_sample[:, col_pair[0]] 
        abc_par2 = abc_sample[:, col_pair[1]] 

        ax = plt.subplot(gs[i])

        corner.hist2d(mcmc_par1, mcmc_par2, bins=20, range=plot_range, ax = ax, plot_datapoints=False,
                levels=[0.68, 0.95], color='#1F77B4', fill_contours=True, smooth=1.0, label=mcmc_label)
        
        corner.hist2d(abc_par1, abc_par2, bins=20, range=plot_range, ax = ax, 
                levels=[0.68, 0.95], color='#FF7F0E', fill_contours=True, smooth=1.0, label=abc_label)

        ax.scatter(np.repeat(truths[col_pair[0]],2), np.repeat(truths[col_pair[1]],2), 
                s=100, marker='*', c='k', lw=0, label=None) 
        #ax.axvline(truths[i_col], color='k', ls='--', linewidth=3)
        #ax.set_xticklabels([])
        ax.set_xlim([plot_range[0, 0] , plot_range[0, 1]])
        ax.set_ylim([plot_range[1, 0] , plot_range[1, 1]])
        
        if i == 2: 
            thick_line1 = mlines.Line2D([], [], ls='-', c='#FF7F0E', linewidth=12, alpha=0.5,
                                        label='ABC-PMC')
            ax.legend(loc='upper right', handles=[thick_line1],
                      frameon=False, fontsize=25, handletextpad=0.1, scatteryoffsets=[0.5])
        elif i == 1: 
            thick_line2 = mlines.Line2D([], [], ls='-', c='#1F77B4', linewidth=12, alpha=0.5, 
                                        label='$\mathcal{L}^\mathtt{Gauss}$ \nMCMC')
            ax.legend(loc='upper right', handles=[thick_line2],
                      frameon=False, fontsize=25, handletextpad=0.1, scatteryoffsets=[0.5])

        ax.set_xlabel(par_labels[col_pair[0]], fontsize=25, labelpad=15)
        ax.set_ylabel(par_labels[col_pair[1]], fontsize=25)

    fig.subplots_adjust(wspace=0.3)
    fig_name = ''.join([ut.fig_dir(), 
        'paper.ABCvsMCMC.contour', 
        '.', obvs, 
        '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    return None 




if __name__=="__main__": 
    #TrueObservables()
    ABCvsMCMC_contour('nbargmf', nwalkers=100, nburns=9000)
    ABCvsMCMC_contour('nbarxi', nwalkers=100, nburns=9000)
    #ABC_Coner('nbargmf')
    #ABC_Coner('nbarxi')
    #ABC_Convergence(weighted=True)
    #ABC_Convergence(weighted=False)
    #PosteriorObservable(Mr=21, b_normal=0.25)
    #PoolEvolution('nbargmf')
    #PoolEvolution('nbarxi')
    #PosteriorObservable(Mr=21, b_normal=0.25)
    #PoolEvolution('nbargmf')
    #PoolEvolution('nbarxi')
