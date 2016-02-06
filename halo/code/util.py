'''

Utility modules

'''
import os


def mask_func(halo_table, subvol_index):
    '''
    Function passed to the `populate mock` method of a halotools model
    instance. Returns a mask for the subvolume of the halos that match the
    chosen `subvol_index` value.
    '''
    ids = halo_table["sim_subvol"]
    return np.where(ids == subvol_index)[0]


def mk_id_column():
    '''
    Function which adds a subvolume id column to the halo table
    '''
    subvol_id_fn = util.multidat_dir() + 'subvol_ids.dat'
    return np.loadtxt(subvol_id_fn)


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


def multidat_dir():
    '''
    Dat directory
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'dat/multidark/'
