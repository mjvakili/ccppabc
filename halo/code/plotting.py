'''

Plotting modules 

'''
import h5py
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

# --- local ---
import util
import data as Data
from prior import PriorRange
plt.switch_backend("Agg")

def plot_thetas(theta, w , t, Mr=20, truths=None, plot_range=None, observables=None, filename=None): 
    '''
    Corner plots of input theta values 
    '''
    # weighted theta
    fig = corner.corner(
            theta, 
            weights=w.flatten(), 
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
            bins=20,
            smooth=1.0)
    
    fig_file = ''.join([util.fig_dir(), 
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
    fig_file = ''.join([util.fig_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '_now_t', str(t), '.png'])
    plt.savefig(fig_file)
    plt.close()


def plot_mcmc_chains(Nwalkers, Nchains_burn=100, Mr=20, truths=None, observables=['nbar', 'xi'], 
        plot_range=None): 
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
        '_Mr', str(data_dict["Mr"]), '_theta.Niter', str(Nchain), '.mcmc_chain.dat'])
    #f = h5py.File(chain_file, 'r')
    #sample = f['positions'][:]
    sample = np.loadtxt(chain_file)
    Ndim = len(sample[0])
    Nchain = len(sample) / Nwalkers 
    
    chain_ensemble = sample.reshape(Nchain , Nwalkers, Ndim)
    fig , axes = plt.subplots(5, 1 , sharex=True, figsize=(10, 12))

    labels=[
        r'$\logM_{0}$',r'$\log \sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'
           ]
  
    for i in xrange(5):
        
        axes[i].plot(chain_ensemble[:, :, i], color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].axhline(truths[i], color="#888888", lw=2)
        axes[i].set_ylabel(labels[i])
    axes[4].set_xlabel("step number")    
    fig.tight_layout(h_pad=0.0) 
    fig_file = ''.join([util.fig_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '.Nchain', str(Nchain),
        '.Nburn', str(Nchains_burn), '.mcmc_time.test.png'])
    plt.savefig(fig_file)
    plt.close()

def plot_mcmc_samples(Nwalkers, Nchains_burn=100, Mr=20, truths=None, observables=['nbar', 'xi'], 
        plot_range=None): 
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
        '_Mr', str(data_dict["Mr"]), '_theta.Niter', str(Nchain), '.mcmc_chain.dat'])
    
    #f = h5py.File(chain_file, 'r')
    #sample = f['positions'][:]
    sample = np.loadtxt(chain_file)
    Nchain = len(sample) / Nwalkers 
        
    fig = corner.corner(
            sample[Nchains_burn*Nwalkers:], 
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

    fig_file = ''.join([util.fig_dir(), 
        util.observable_id_flag(observables), 
        '_Mr', str(Mr), '.Nchain', str(Nchain),
        '.Nburn', str(Nchains_burn), '.mcmc_samples.test.png'])
    plt.savefig(fig_file)
    plt.close()


if __name__=='__main__':
    plot_mcmc_chains(10, Nchains_burn=1, Mr=20, observables=['nbar', 'xi'])
    #plot_mcmc_samples(10, Nchains_burn=1, Mr=20, observables=['nbar', 'xi'])
