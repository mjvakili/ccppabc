import pyfits as fits
import numpy as py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import os 

def prettyplot():
    '''
    Some settings to make pretty plots
    '''
    mpl.rcParams['text.usetex']=True
    mpl.rcParams['text.latex.preamble']=[r"\usepackage[T1]{fontenc}",
    r"\usepackage{cmbright}",]
    mpl.rcParams['xtick.major.size']=10
    mpl.rcParams['xtick.major.width']=2.5
    mpl.rcParams['ytick.major.size']=10
    mpl.rcParams['ytick.major.width']=2.5
    mpl.rcParams['ytick.minor.size']=3
    mpl.rcParams['ytick.minor.width']=1.5
    mpl.rcParams['xtick.minor.size']=3
    mpl.rcParams['xtick.minor.width']=1.5
    mpl.rcParams['xtick.major.pad']=12
    mpl.rcParams['ytick.major.pad']=12
    mpl.rcParams['xtick.labelsize']='large'
    mpl.rcParams['ytick.labelsize']='large'
    mpl.rcParams['axes.linewidth']=2.5
    mpl.rcParams['font.family']='monospace'
    mpl.rcParams['font.monospace']='Courier'
    mpl.rcParams['font.weight']=800
    mpl.rcParams['font.size']=16
    mpl.rcParams['legend.frameon']=False
    mpl.rcParams['legend.markerscale']=5.0
    mpl.rcParams['axes.xmargin']=1

def prettycolors(): 
    # Tableau20 colors 
    pretty_colors = [(89, 89, 89), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
            (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
    # Scale the RGB values to the [0, 1] range
    for i in range(len(pretty_colors)):  
        r, g, b = pretty_colors[i]  
        pretty_colors[i] = (r / 255., g / 255., b / 255.)  
    return pretty_colors 
