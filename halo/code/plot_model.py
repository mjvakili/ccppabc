import numpy as np


import numpy as np
import abcpmc
import matplotlib.pyplot as plt
from interruptible_pool import InterruptiblePool
import time
plt.switch_backend("Agg")
from halotools.empirical_models import Zheng07

from astropy.table import Table
model = Zheng07(threshold = -21.)
print 'Data HOD Parameters ', model.param_dict
import pandas as pd



def richness(group_id):
    gals = Table()
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)
    return grp_richness




n_mocks = 1000
n_bins = 12
hist_bins = np.rint(3 * np.logspace(0., 1.2, n_bins+1 ))
hist_bins[-1] = 10000
print hist_bins



model.populate_mock()

    # number density
#    avg_nz.append(model.mock.number_density)

#     # richness histogram
group_id = model.mock.compute_fof_group_ids()
group_richness = richness(group_id)
# 
#     #print 'Group Richness computation takes ', time.time() - hod_time, ' seconds'
hist, bin_edge = np.histogram(group_richness, bins=hist_bins)
bin_mid = 0.5 * (bin_edge[1:] + bin_edge[:-1])
#    
#     histograms[i,:] = hist

# np.savetxt("group_rich.dat", histograms)

#np.savetxt("nz.dat", avg_nz)

"""data and covariance """

nz = np.loadtxt("nz.dat")
histograms = np.loadtxt("group_rich.dat")

covar_nz = np.cov(nz)
covar_gr = np.cov(histograms)

snr_gr   = 1./np.diag(covar_gr)

yerr =  snr_gr**-1.
y =  np.mean(histograms , axis = 1)
x =  bin_mid


theta = np.loadtxt("weighted_l2_theta_hod3_flat_t9.dat")

model.param_dict['alpha'] , model.param_dict['sigma_logM'] , model.param_dict['logMmin']=  np.median(theta , axis = 0)

#print theta1

#'logM0': 11.92, 'sigma_logM': 0.39, 'logMmin': 12.79, 'alpha': 1.15, 'logM1': 13.94
his = np.zeros_like(hist)

for i in xrange(50): 
  model.populate_mock()

    # number density
#    avg_nz.append(model.mock.number_density)

#     # richness histogram
  group_id = model.mock.compute_fof_group_ids()
  group_richness = richness(group_id)
# 
#     #print 'Group Richness computation takes ', time.time() - hod_time, ' seconds'
  hi, bin_edge = np.histogram(group_richness, bins=hist_bins)
  his = his + hi


plt.errorbar(x , y , yerr , fmt=".k", ms=4,
            capsize=0, alpha=0.3)
plt.step(bin_edge[1:], his/50. , lw=2  , alpha = .4)
plt.xlim(5 , 100)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Group Richness")
plt.ylabel("Richness Abundance")
plt.savefig("/home/mj/public_html/richness.png")
