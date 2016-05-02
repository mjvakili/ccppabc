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
        t_list = [0, 2, 5, 8]
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
    print prior_range
    
    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(figsize=(8,8))
    for i_t, t in enumerate(t_list): 
        sub = fig.add_subplot(2,2,i_t+1)

        theta_Mmin, theta_M1 = np.loadtxt(theta_file(t), unpack=True, usecols=[2, 4]) 
        col = pretty_colors[0]
        print 
        corner.hist2d(theta_Mmin, theta_M1, bins=20, range=prior_range, 
                levels=[0.68, 0.95], color='b', fill_contours=True, smooth=1.0)

        if i_t == 0: 
            sub.set_xticklabels([])
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
    plt.show()

if __name__=="__main__": 
    PoolEvolution('nbargmf')
