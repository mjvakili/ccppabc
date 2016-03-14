
#general python modules
import numpy as np
from multiprocessing import cpu_count
from numpy.linalg import solve

#haltools functions
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from halotools.mock_observables.pair_counters import npairs

#our ccppabc functions
import util
from group_richness import gmf_bins
from group_richness import richness
from group_richness import gmf as GMF



def test_subvol_gmf(Mr):

    thr = -1. * np.float(Mr)
    model = PrebuiltHodModelFactory('zheng07', threshold=thr,
                                    halocat='multidark', redshift=0.)
    model.new_haloprop_func_dict = {'sim_subvol': util.mk_id_column}

    datsubvol = lambda x: util.mask_func(x, 0)
    model.populate_mock(simname='multidark',
                        masking_function=datsubvol,
                        enforce_PBC=False)
    

    #compute group richness    

    rich = richness(model.mock.compute_fof_group_ids())
    gmf = GMF(rich)  #GMF
    print gmf
    print GMF(rich , counts = True) 

    galaxy_sample = model.mock.galaxy_table
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vz = galaxy_sample['vz']

    pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z'
                               , velocity = vz , velocity_distortion_dimension="z")

    from halotools.mock_observables import FoFGroups

    b_para, b_perp = 0.7, 0.15
    groups = FoFGroups(pos, b_perp, b_para, period = None, 
                      Lbox = 200 , num_threads='max')

    gids = groups.group_ids
    rich = richness(gids)
    gmf = GMF(rich)

    print gmf
    print GMF(rich , counts = True) 

if __name__ == '__main__':

     test_subvol_gmf(21)
