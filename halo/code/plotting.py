'''

Plotting modules 

'''
import util
import corner
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

def plot_thetas(theta, w , t, Mr=20, truths=None, plot_range=None, observables=None): 
    '''
    Corner plots of input theta values 
    '''
    # weighted theta
    fig = corner.corner(
            theta, 
            weights=w.flatten(), 
            truths=truths,
            labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'],
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
            labels=[r'$\logM_{0}$',r'$\sigma_{\logM}$',r'$\logM_{min}$',r'$\alpha$',r'$\logM_{1}$'],
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
