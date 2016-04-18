from halotools.empirical_models import Zheng07 , model_defaults
from halotools.mock_observables import wp
from halotools.mock_observables.clustering import tpcf
from halotools.empirical_models.mock_helpers import (three_dim_pos_bundle,
                                                     infer_mask_from_kwargs)
from halotools.mock_observables.clustering import wp
from halotools.sim_manager import supported_sims
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import time
import numpy as np
model = Zheng07()

xir = []
for i in range(500):
 model.populate_mock()
 xir.append(model.mock.compute_galaxy_clustering()[1])

covar = np.cov(np.array(xir).T)
np.savetxt("clustering_covariance_Mr20.dat" , covar)

"""
a = time.time()
model.mock.compute_galaxy_clustering()
print time.time()  - a
rbins = model_defaults.default_rbins
rbin_centers  = (rbins[1:] + rbins[:-1])/2.
cat = supported_sims.HaloCatalog()
l = cat.Lbox
print l
p_bins = np.linspace(0,l/2,200)
mask = infer_mask_from_kwargs(model.mock.galaxy_table)


pos = three_dim_pos_bundle(table=model.mock.galaxy_table,
                               key1='x', key2='y', key3='z', mask=mask,
                               return_complement=False)
figure = plt.figure(figsize=(10,10))

cl = wp(pos , rbins, p_bins , period = l , estimator = 'Landy-Szalay')

for n_pbins in np.array([2,8,16]):

  p_bins = np.linspace(0 , l/2 , n_pbins)
  a = time.time()
  clustering = wp(pos, rbins, p_bins , period = l , estimator = 'Landy-Szalay')
  print time.time() - a
  plt.plot(rbin_centers , (clustering)/cl , label = "$N\pi_{bin}$="+str(n_pbins) , lw = 2)
  plt.xscale("Log")
  plt.yscale("Log")
  plt.legend() 
plt.savefig("/home/mj/public_html/wpex.png")"""
