'''

Utility modules

'''
import os
import numpy as np

def random_shifter(index):

    zi = (index / 25) * 200.
    i2 = index % 25
    yi = (i2 / 5) * 200.
    i3 = i2 % 5
    xi = (i3) * 200.

    return xi , yi , zi

def edge(index):
    '''
    returns the edge of the subvolume of a given index
    '''    
    zi = (index / 25) * 200.
    i2 = index % 25
    yi = (i2 / 5) * 200.
    i3 = i2 % 5
    xi = (i3) * 200.

    return xi , yi , zi


def mask_galaxy_table(pos , subvol_index):
    '''
    maskes the positions of the galaxies in the entire subvolume
    returns the galaxies that are in a subvolume with index subvol_index
    '''
    xi , yi , zi  = edge(subvol_index)
    mask = np.where()
        
    submask = np.where((xi <pos[:, 0]) * \
                       (pos[:, 0] < xi + 200) * \
                       (yi < pos[:, 1]) * \ 
                       (pos[:, 1] < yi + 200) * \
                       (zi <pos[:, 2]) * \ 
                       (pos[:, 2] < zi + 200))
    masked_positions = pos[submask]

    return masked_positions
      
    
def mk_id_column(table=None):

    # set up ids from 0 to 124 for the box split into 5 along each edge
    edges = np.linspace(0, 800, 5)

    xs = table["halo_x"]
    ys = table["halo_y"]
    zs = table["halo_z"]

    subvol_ids = np.empty(xs.shape)
    for i in xrange(len(xs)):
        xi = np.where(edges < xs[i])[0][-1]
        yi = np.where(edges < ys[i])[0][-1]
        zi = np.where(edges < zs[i])[0][-1]
        subvol_ids[i] = zi * 25 + yi * 5 + xi

    return subvol_ids


def mask_func(halo_table, subvol_index):
    '''
    Function passed to the `populate mock` method of a halotools model
    instance. Returns a mask for the subvolume of the halos that match the
    chosen `subvol_index` value.
    '''
    ids = halo_table["sim_subvol"]
    return np.where(ids == subvol_index)[0]


def observable_id_flag(observables):
    '''
    Observable identification flag string given list of observables
    '''
    return '_'.join(observables)

def fig_dir():
    '''
    figure directory
    '''
    fig_dir = os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'fig/'
    return fig_dir

def dat_dir():
    '''
    Dat directory
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'dat/'

def obvs_dir(): 
    return dat_dir()+'observations/'

def abc_dir():
    return dat_dir()+'pmc_abc/'

def mcmc_dir():
    return dat_dir()+'mcmc/'

def crash_dir():
    return dat_dir()+'crash/


