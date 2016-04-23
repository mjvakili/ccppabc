import numpy as np
import matplotlib.pyplot as plt
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import tpcf
from halotools.empirical_models.factories.mock_helpers import three_dim_pos_bundle
from util import mask_func


def mk_id_column(table=None):

    # set up ids from 0 to 63 for the box split into 4 along each edge
    edges = np.linspace(0, 750, 4)

    xs = table["halo_x"]
    ys = table["halo_y"]
    zs = table["halo_z"]

    subvol_ids = np.empty(xs.shape)
    for i in xrange(len(xs)):
        xi = np.where(edges < xs[i])[0][-1]
        yi = np.where(edges < ys[i])[0][-1]
        zi = np.where(edges < zs[i])[0][-1]
        subvol_ids[i] = zi * 16 + yi * 4 + xi

    return subvol_ids


def random_shifter(index):

    zi = (index / 16) * 250.
    i2 = index % 16
    yi = (i2 / 4) * 250.
    i3 = i2 % 4
    xi = (i3) * 250.

    return xi , yi , zi


def mk_id_column_2(table=None):

    edges = np.linspace(0, 500, 2)

    xs = table["halo_x"]
    ys = table["halo_y"]
    zs = table["halo_z"]

    subvol_ids = np.empty(xs.shape)
    for i in xrange(len(xs)):
        xi = np.where(edges < xs[i])[0][-1]
        yi = np.where(edges < ys[i])[0][-1]
        zi = np.where(edges < zs[i])[0][-1]
        subvol_ids[i] = zi * 4 + yi * 2 + xi

    return subvol_ids


def random_shifter_2(index):

    zi = (index / 4) * 500.
    i2 = index % 4
    yi = (i2 / 2) * 500.
    i3 = i2 % 2
    xi = (i3) * 500.

    return xi , yi , zi


xi_bin = np.concatenate([np.array([0.15]),
                         np.logspace(np.log10(0.5),
                                     np.log10(20.), 15)])
rmax = xi_bin.max()

# Get those randoms
num_randoms = 50 * 800000
xran = np.random.uniform(0, 1000, num_randoms)
yran = np.random.uniform(0, 1000, num_randoms)
zran = np.random.uniform(0, 1000, num_randoms)
full_randoms = np.vstack((xran, yran, zran)).T

# Get the full box mock
model = PrebuiltHodModelFactory('zheng07', threshold=-21)
halocat = CachedHaloCatalog(simname='multidark', redshift=0,
                            halo_finder='rockstar')
model.populate_mock(halocat, enforce_PBC=False)
pos = three_dim_pos_bundle(model.mock.galaxy_table, 'x', 'y', 'z')

# Get full tpcf
print "getting full vol tpcf..."
xi_full_pc = tpcf(pos, xi_bin,
                  randoms=full_randoms,
                  do_auto=True, do_cross=False,
                  max_sample_size=int(pos.shape[0]),
                  estimator='Natural',
                  approx_cell1_size=[rmax, rmax, rmax],
                  approx_cellran_size=[rmax, rmax, rmax])
print "done"

Nsub = 8

# Now set up for subvol boxes
num_randoms = 50 * 8000
xran = np.random.uniform(0, 250, num_randoms)
yran = np.random.uniform(0, 250, num_randoms)
zran = np.random.uniform(0, 250, num_randoms)
sub_randoms = np.vstack((xran, yran, zran)).T

sub_model = PrebuiltHodModelFactory('zheng07')
sub_model.new_haloprop_func_dict = {'sim_subvol': mk_id_column}
sub_halocat = CachedHaloCatalog(simname='multidark', redshift=0,
                                halo_finder='rockstar')

print "starting with 250 mpc subvols:"
sub_xi_list = []
for ii in range(1,Nsub+1):
    print ii
    simsubvol = lambda x: mask_func(x, ii)
    sub_model.populate_mock(sub_halocat,
    masking_function=simsubvol, enforce_PBC=False)

    sub_pos = three_dim_pos_bundle(sub_model.mock.galaxy_table,
                                   'x', 'y', 'z')

    xi, yi, zi = random_shifter(ii)
    temp_randoms = sub_randoms.copy()
    temp_randoms[:,0] += xi
    temp_randoms[:,1] += yi
    temp_randoms[:,2] += zi

    sub_xi = tpcf(sub_pos, xi_bin,
                  randoms=temp_randoms,
                  do_auto=True,
                  do_cross=False,
                  max_sample_size=int(sub_pos.shape[0]),
                  estimator='Natural',
                  approx_cell1_size=full_approx_cell1_size,
                  approx_cellran_size=full_approx_cellran_size)
    sub_xi_list.append(sub_xi)

# And again for the larger subvols
num_randoms = 50 * 100000
xran = np.random.uniform(0, 500, num_randoms)
yran = np.random.uniform(0, 500, num_randoms)
zran = np.random.uniform(0, 500, num_randoms)
sub_randoms = np.vstack((xran, yran, zran)).T

sub_model = PrebuiltHodModelFactory('zheng07')
sub_model.new_haloprop_func_dict = {'sim_subvol': mk_id_column_2}
sub_halocat = CachedHaloCatalog(simname='multidark', redshift=0,
                                halo_finder='rockstar')

print "starting with 500 mpc subvols"
sub_xi_list_2 = []
for ii in range(1,Nsub+1):
    print ii
    simsubvol = lambda x: mask_func(x, ii)
    sub_model.populate_mock(sub_halocat,
    masking_function=simsubvol, enforce_PBC=False)

    sub_pos = three_dim_pos_bundle(sub_model.mock.galaxy_table,
                                   'x', 'y', 'z')

    xi, yi, zi = random_shifter_2(ii)
    temp_randoms = sub_randoms.copy()
    temp_randoms[:,0] += xi
    temp_randoms[:,1] += yi
    temp_randoms[:,2] += zi

    sub_xi = tpcf(sub_pos, xi_bin,
                  randoms=temp_randoms,
                  do_auto=True,
                  do_cross=False,
                  max_sample_size=int(sub_pos.shape[0]),
                  estimator='Natural',
                  approx_cell1_size=full_approx_cell1_size,
                  approx_cellran_size=full_approx_cellran_size,
                  RR_precomputed=sub_RR,
                  NR_precomputed=sub_NR)
    sub_xi_list_2.append(sub_xi)

# And plot!
fig = plt.figure()

ax = fig.add_subplot(111)

print "plotting..."
lines = []
for ii in range(1,Nsub+1):

    # plot small vol ratio
    rat1 = sub_xi_list[ii] / xi_full_pc
    lines.append(ax.plot(0.5*(xi_bin[:-1] + xi_bin[1:]), rat1,
                 color='g', alpha=0.2))

    # plot large vol ratio
    rat2 = sub_xi_list[ii] / xi_full_pc
    lines.append(ax.plot(0.5*(xi_bin[:-1] + xi_bin[1:]), rat2,
                 color='r', alpha=0.2))

ax.set_xscale('log')
ax.set_xlabel('r')
ax.set_ylabel('xi ratio')

fig.savefig("subvol_compare.pdf")
