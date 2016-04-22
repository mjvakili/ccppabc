import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07
from astropy.table import Table
import numpy as np



model = Zheng07(threshold=-21)


nmocks = 500

nbins = 14


xir = np.empty((nmocks , nbins))

for i in xrange(nmocks):

    model.populate_mock()
    r , xr = model.mock.compute_galaxy_clustering()
    xir[i,:] = xr

np.savetxt("xir.dat" , xir)
