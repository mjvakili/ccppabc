'''

Plotting modules 

'''
import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# --- local ---
import util
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

def prettyplot():
    '''
    Some settings to make pretty plots
    '''
    # Use Latex
    mpl.rcParams['text.usetex']=True
    mpl.rcParams['text.latex.preamble']=[r"\usepackage[T1]{fontenc}",
    r"\usepackage{cmbright}",]
    # Set Major tick size and width
    mpl.rcParams['xtick.major.size']=10
    mpl.rcParams['xtick.major.width']=2.5
    mpl.rcParams['ytick.major.size']=10
    mpl.rcParams['ytick.major.width']=2.5
    # Set minor tick size and wdith
    mpl.rcParams['ytick.minor.size']=3
    mpl.rcParams['ytick.minor.width']=1.5
    mpl.rcParams['xtick.minor.size']=3
    mpl.rcParams['xtick.minor.width']=1.5
    # Set space between axes and tick label
    mpl.rcParams['xtick.major.pad']=12
    mpl.rcParams['ytick.major.pad']=12
    mpl.rcParams['xtick.labelsize']='large'
    mpl.rcParams['ytick.labelsize']='large'
    mpl.rcParams['axes.linewidth']=2.5

    mpl.rcParams['font.family']='monospace'
    mpl.rcParams['font.monospace']='Courier'
    mpl.rcParams['font.weight']=800
    mpl.rcParams['font.size']=16
    # legend settings
    mpl.rcParams['legend.frameon']=False
    mpl.rcParams['legend.markerscale']=5.0
    mpl.rcParams['axes.xmargin']=1
    return None

def prettycolors(): 
    ''' preset pretty colors
    '''
    # Tableau20 colors 
    pretty_colors = [
            (89, 89, 89), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
            (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
            ]  

    # Scale the RGB values to the [0, 1] range
    for i in range(len(pretty_colors)):  
        r, g, b = pretty_colors[i] 
        pretty_colors[i] = (r / 255., g / 255., b / 255.)  

    return pretty_colors 

