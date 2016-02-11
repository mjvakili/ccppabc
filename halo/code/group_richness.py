'''


Modules for Galaxy Group Richness


'''
import numpy as np
from astropy.table import Table


def richness(group_id):
    '''
    Calculate the richness of a group given group_ids of galaxies. Uses astropy.table module
    '''
    gals = Table()
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)

    return grp_richness

def gmf_bins():
    '''
    Observed GMF bins. Hardcoded.
    '''
    bins = np.array([
        3.000000000000000000e+00,
        4.000000000000000000e+00,
        5.000000000000000000e+00,
        6.000000000000000000e+00,
        7.000000000000000000e+00,
        8.000000000000000000e+00,
        9.000000000000000000e+00,
        1.000000000000000000e+01,
        1.100000000000000000e+01,
        1.200000000000000000e+01,
        1.300000000000000000e+01,
        1.400000000000000000e+01,
        1.500000000000000000e+01,
        1.600000000000000000e+01,
        1.700000000000000000e+01,
        1.800000000000000000e+01,
        1.900000000000000000e+01,
        2.000000000000000000e+01,
        2.100000000000000000e+01,
        2.200000000000000000e+01,
        2.300000000000000000e+01,
        2.500000000000000000e+01,
        2.900000000000000000e+01,
        3.100000000000000000e+01,
        3.500000000000000000e+01,
        4.300000000000000000e+01]) # same hardcoded bins as data
    return bins

def gmf(group_richness, counts=False):
    '''
    Calculate Group Multiplicity Function (GMF) given group richness (see above module).
    '''
    bins = gmf_bins()

    if not counts:
        gmeff = np.histogram(group_richness , bins)[0] / 200.**3.   # calculate GMF
        return gmeff
    else:
        g_counts = np.histogram(group_richness , bins)[0]   # calculate group counts
        return g_counts
