'''

Plots and calculations directly for the paper 

Authors: Chang and MJ 
'''
import corner
import numpy as np 

import util as ut
from prior import PriorRange

import matplotlib.pyplot as plt
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
        t_list = [0, 2, 3, 8]
    elif obvs == 'nbarxi':
        result_dir = ''.join([ut.dat_dir(), 'paper/ABC', obvs, '/'])
        theta_file = lambda tt: ''.join([result_dir, 'nbar_xi_theta_t', str(tt), '.abc.dat']) 
        t_list = [0, 2, 5, 9]
    else:
        raise ValueError
    
    prior_min, prior_max = PriorRange('first_try')
    prior_range = np.zeros((2,2))
    prior_range[:,0] = np.array([prior_min[2], prior_min[4]])
    prior_range[:,1] = np.array([prior_max[2], prior_max[4]])
    
    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(figsize=(8,8))
    for i_t, t in enumerate(t_list): 
        sub = fig.add_subplot(2,2,i_t+1)

        theta_Mmin, theta_M1 = np.loadtxt(theta_file(t), unpack=True, usecols=[2, 4]) 
        corner.hist2d(theta_Mmin, theta_M1, bins=20, range=prior_range, 
                levels=[0.68, 0.95], color='b', fill_contours=True, smooth=1.0)
        
        t_label = r"$\mathtt{t = "+str(t)+"}$"
        sub.text(13.0, 15.0, t_label, fontsize=25) 

        if i_t == 0: 
            sub.set_xticklabels([])
            sub.set_yticklabels([13., 13.5, 14., 14.5, 15., 15.5])
            plt.ylabel(r'$\mathtt{log}\;\mathtt{M_{1}}$', fontsize=25)
        elif i_t == 1: 
            sub.set_xticklabels([])
            sub.set_yticklabels([])
        elif i_t == 2: 
            sub.set_yticklabels([13., 13.5, 14., 14.5, 15.])
            plt.ylabel(r'$\mathtt{log}\;\mathtt{M_{1}}$', fontsize=25)
            plt.xlabel(r'$\mathtt{log}\;\mathtt{M_{min}}$', fontsize=25)
        elif i_t == 3: 
            sub.set_yticklabels([])
            plt.xlabel(r'$\mathtt{log}\;\mathtt{M_{min}}$', fontsize=25)

    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig_name = ''.join([ut.fig_dir(), 'paper_ABC_poolevolution.png'])
    fig.savefig(fig_name, bbox_inches='tight', dpi=150) 
    plt.close()


def PosteriorObservable(obvs):
    ''' Plot 1\sigma and 2\sigma model predictions from ABC-PMC posterior likelihood
    '''
    # load the particles
    if abc_theta_file is None:
        raise ValueError("Please specify the theta output file from ABC-PMC run")
    theta = np.loadtxt(abc_theta_file)

    if observable == 'scaledxi':
        obvs_str = 'xi'
    else:
        obvs_str = observable

    obvs_file = ''.join(abc_theta_file.rsplit('.dat')[:-1] + ['.', observable, '.dat'])
    print obvs_file
    if not os.path.isfile(obvs_file) or clobber:
        for i in xrange(len(theta)):
            obv_i  = HODsimulator(
                    theta[i], prior_range=None,
                    observables=[obvs_str], Mr=data_dict['Mr'])
            try:
                model_obv.append(obv_i[0])
            except UnboundLocalError:
                model_obv = [obv_i[0]]
        model_obv = np.array(model_obv)
        np.savetxt(obvs_file, model_obv)
    else:
        model_obv = np.loadtxt(obvs_file)

    if 'xi' in observable:
        r_bin = Data.data_xi_bins(Mr=data_dict['Mr'])
    elif observable == 'gmf':
        r_bin = Data.data_gmf_bins()

    a, b, c, d, e = np.percentile(model_obv, [2.5, 16, 50, 84, 97.5], axis=0)

    # plotting
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    if observable == 'xi':  # 2PCF
        data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict) # data

        ax.fill_between(r_bin, a, e, color="k", alpha=0.1, edgecolor="none")
        ax.fill_between(r_bin, b, d, color="k", alpha=0.3, edgecolor="none")
        #ax.plot(r_bin, c, "k", lw=1)
        ax.errorbar(r_bin, data_xi, yerr = np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                    capsize=0)
        ax.set_xlabel(r'$r\;[\mathrm{Mpc}/h]$', fontsize=20)
        ax.set_ylabel(r'$\xi_{\rm gg}$', fontsize=25)
        ax.set_xscale('log')
        ax.set_xlim([0.1, 20.])

    elif observable == 'scaledxi':  # Scaled 2PFC (r * xi)
        data_xi, data_xi_cov = Data.data_xi_full_cov(**data_dict) # data

        ax.fill_between(r_bin, r_bin*a, r_bin*e, color="k", alpha=0.1, edgecolor="none")
        ax.fill_between(r_bin, r_bin*b, r_bin*d, color="k", alpha=0.3, edgecolor="none")
        ax.errorbar(r_bin, r_bin*data_xi, yerr=r_bin*np.sqrt(np.diag(data_xi_cov)), fmt=".k",
                    capsize=0)
        ax.set_xlabel(r'$r\;[\mathrm{Mpc}/h]$', fontsize=20)
        ax.set_ylabel(r'$r \xi_{\rm gg}$', fontsize=25)
        ax.set_xscale('log')
        ax.set_xlim([0.1, 20.])

    elif observable == 'gmf':   # GMF
        data_gmf, data_gmf_sigma = Data.data_gmf(**data_dict)
        ax.fill_between(r_bin, a, e, color="k", alpha=0.1, edgecolor="none")
        ax.fill_between(r_bin, b, d, color="k", alpha=0.3, edgecolor="none")
        ax.errorbar(r_bin, data_gmf, yerr = data_gmf_sigma, fmt=".k",
                    capsize=0)
        ax.set_xlabel(r'Group Richness', fontsize=25)
        ax.set_ylabel(r'GMF $[\mathrm{h}^3\mathrm{Mpc}^{-3}]$', fontsize=25)

        ax.set_yscale('log')
        ax.set_xlim([1., 50.])

    fig.savefig(
            ''.join([util.fig_dir(),
                observable, '.posterior_prediction',
                '.Mr', str(data_dict['Mr']), '_Nmock', str(data_dict['Nmock']),
                '.pdf']),
            bbox_inches='tight')


if __name__=="__main__": 
    PoolEvolution('nbargmf')
