'''

Plotting modules 

'''
import os 
import h5py
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

# --- local ---
import util
import data as Data
from prior import PriorRange
from hod_sim import HODsimulator    
plt.switch_backend("Agg")

def plot_thetas(theta, w , t, Mr=20, truths=None, plot_range=None, observables=None, 
        filename=None, output_dir=None): 
    '''
    Corner plots of input theta values 
    '''
    if output_dir is None: 
        fig_dir = util.fig_dir()
    else: 
        fig_dir = output_dir

    # weighted theta
    fig = corner.corner(
            theta, 
            weights=w.flatten(), 
            truths=truths,
            truth_color='#ee6a50', 
            labels=[
                r'$\mathtt{\log\;M_{0}}$',
                r'$\mathtt{\log\;\sigma_{\logM}}$',
                r'$\mathtt{\log\;M_{min}}$',
                r'$\mathtt{\alpha}$',
                r'$\mathtt{\log\;M_{1}}$'
                ],
            label_kwargs={'fontsize': 25},
            range=plot_range,
            quantiles=[0.16,0.5,0.84], 
            show_titles=True, 
            title_args={"fontsize": 12},
            plot_datapoints=True, 
            fill_contours=True, 
            levels=[0.68, 0.95], 
            color='b', 
            bins=20,
            smooth=1.0)
     
    fig_file = ''.join([fig_dir, 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '_t', str(t), '.png'])
    plt.savefig(fig_file)
    plt.close()
    
    # not weighted
    fig = corner.corner(
            theta, 
            truths=truths,
            truth_color='#ee6a50', 
            labels=[
                r'$\logM_{0}$',r'$\log \sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'
                ],
            range=plot_range, 
            quantiles=[0.16,0.5,0.84],
            show_titles=True, 
            title_args={"fontsize": 12},
            plot_datapoints=True, 
            fill_contours=True, 
            levels=[0.68, 0.95], 
            color='b', 
            bins=16, 
            smooth=1.0)
    fig_file = ''.join([fig_dir, util.observable_id_flag(observables), 
        '_Mr', str(Mr), '_now_t', str(t), '.png'])
    plt.savefig(fig_file)
    plt.close()

def plot_mcmc(Nwalkers, Niter=10000, Nchains_burn=100, Mr=20, truths=None, 
        observables=['nbar', 'xi'], plot_range=None): 
    '''
    Plot MCMC chains
    '''
    if truths is None: 
        data_hod_dict = Data.data_hod_param(Mr=Mr)
        truths = np.array([
            data_hod_dict['logM0'],                 # log M0 
            np.log(data_hod_dict['sigma_logM']),    # log(sigma)
            data_hod_dict['logMmin'],               # log Mmin
            data_hod_dict['alpha'],                 # alpha
            data_hod_dict['logM1']                  # log M1
            ])
    if plot_range is None: 
        prior_min, prior_max = PriorRange(None)
        plot_range = np.zeros((len(prior_min),2))
        plot_range[:,0] = prior_min
        plot_range[:,1] = prior_max
    
    # chain files 
    chain_file = ''.join([util.dat_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '_theta.Niter', str(Niter), '.mcmc_chain.dat'])
    
    #f = h5py.File(chain_file, 'r')
    #sample = f['positions'][:]
    sample = np.loadtxt(chain_file)

    # Posterior Likelihood Corner Plot     
    fig = corner.corner(
            sample[Nchains_burn*Nwalkers:], 
            truths=truths,
            truth_color='#ee6a50', 
            labels=[
                r'$\mathtt{\log\;M_{0}}$',
                r'$\mathtt{\log\;\sigma_{\logM}}$',
                r'$\mathtt{\log\;M_{min}}$',
                r'$\mathtt{\alpha}$',
                r'$\mathtt{\log\;M_{1}}$'
                ],
            label_kwargs={'fontsize': 25},
            range=plot_range, 
            quantiles=[0.16,0.5,0.84],
            show_titles=True, 
            title_args={"fontsize": 12},
            plot_datapoints=True, 
            fill_contours=True, 
            levels=[0.68, 0.95], 
            color='b', 
            bins=16, 
            smooth=1.0)

    fig_file = ''.join([util.fig_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '.Niter', str(Niter),
        '.Nburn', str(Nchains_burn), '.mcmc_samples.test.png'])
    plt.savefig(fig_file)
    plt.close()
    
    # MCMC Chain plot
    Ndim = len(sample[0])
    Nchain = len(sample)/Nwalkers
    
    chain_ensemble = sample.reshape(Nchain, Nwalkers, Ndim)
    fig , axes = plt.subplots(5, 1 , sharex=True, figsize=(10, 12))

    labels=[
        r'$\mathtt{\log\;M_{0}}$',
        r'$\mathtt{\log\;\sigma_{\logM}}$',
        r'$\mathtt{\log\;M_{min}}$',
        r'$\mathtt{\alpha}$',
        r'$\mathtt{\log\;M_{1}}$'
        ]
  
    for i in xrange(5):
        axes[i].plot(chain_ensemble[:, :, i], color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].axhline(truths[i], color="#888888", lw=2)
        axes[i].vlines(Nchains_burn, plot_range[i,0], plot_range[i,1], colors='#ee6a50', linewidth=4, alpha=1)
        axes[i].set_ylim([plot_range[i,0], plot_range[i,1]])
        axes[i].set_ylabel(labels[i], fontsize=25)

    axes[4].set_xlabel("Step Number", fontsize=25)    
    fig.tight_layout(h_pad=0.0) 
    fig_file = ''.join([util.fig_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '.Niter', str(Niter),
        '.Nburn', str(Nchains_burn), '.mcmc_time.test.png'])
    plt.savefig(fig_file)
    plt.close()

def plot_posterior_model(observable, abc_theta_file=None, data_dict={'Mr':20, 'Nmock':500}, 
        clobber=False):
    '''
    Plot 1\sigma and 2\sigma model predictions from ABC-PMC posterior likelihood 

    Parameters
    ----------
    observable : string
        One of the following strings ['xi', 'scaledxi', 'gmf']
        
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
            #rr , xi = mod.compute_average_compute_average_galaxy_clustering(rbins = hardcoded_xi_bins , num_iterations = 6 )
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



if __name__=='__main__':
    plot_posterior_model('xi',  
            abc_theta_file="../dat/nbar_gmf_Mr20_theta_t18.dat", 
            data_dict={'Mr':20, 'Nmock':500}, clobber=True)
    #plot_mcmc(100, Niter=10000, Nchains_burn=500, Mr=20, observables=['nbar', 'xi'])
    #plot_mcmc_samples(100, Niter=10000, Nchains_burn=500, Mr=20, observables=['nbar', 'xi'])
