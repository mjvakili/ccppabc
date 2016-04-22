from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables import FoFGroups
import numpy as np
from astropy.table import Table
import pyfof

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
model = PrebuiltHodModelFactory('zheng07', threshold=-21)
halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')

model.populate_mock(halocat)
pos =three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')

#groups = FoFGroups(pos, 0.75, 0.75, Lbox = model.mock.Lbox, num_threads=1)
#gids = groups.group_ids

#print richness(gids)
import time

a = time.time()

groups = pyfof.friends_of_friends(pos, 0.75*(len(pos)/1000**3.)**(-1./3))

w = np.array([len(x) for x in groups])

bins = np.array([2.,3.,4.,5.,6.,7,9,11,14,17,20])


gmeff = np.histogram(w , bins)[0] / 1000.**3.

print gmeff

print time.time() - a


