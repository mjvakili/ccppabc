from halotools.empirical_models import Zheng07
import numpy as np
import plotting
import abcpmc
import matplotlib.pyplot as plt
from astropy.table import Table
plt.switch_backend("Agg")
plotting.prettyplot()

def richness(group_id): 
    gals = Table() 
    gals['groupid'] = group_id
    gals['dummy'] = 1
    grouped_table = gals.group_by('groupid')
    grp_richness = grouped_table['dummy'].groups.aggregate(np.sum)
    return grp_richness



"""Clustering"""

#input HOD parameters for creating the mock data

true_hod = np.array([11.38 , 0.26 , 12.02 , 1.06 , 13.31])


#prior range chosen for galaxy clustering analyses

prior = abcpmc.TophatPrior([10.,.1,11.02,.8,13.],[13.,.5,13.02,1.3,14.])
prior_dict = {

    'logM0'  : {'shape': 'uniform', 'min': 10.  ,  'max': 13.},
    'sigma_logM': {'shape': 'uniform', 'min': .1 ,  'max': .5},
    'logMmin': {'shape': 'uniform', 'min': 11.02,  'max': 13.02},
    'alpha': {'shape': 'uniform', 'min': .8 ,  'max': 1.3},
    'logM1'  : {'shape': 'uniform', 'min': 13.  ,  'max': 14.},
}

#plot range same as prior range

plot_range = []
for key in ['logM0', 'sigma_logM', 'logMmin','alpha','logM1']:
        plot_range.append([prior_dict[key]['min'], prior_dict[key]['max']])
prior_range = np.array(plot_range)


theta_cl = []
for i in xrange(20):
    theta_cl.append(np.loadtxt("/home/mj/public_html/nbar_clustering_Mr20_theta_t"+str(i)+".dat"))
theta_cl = np.array(theta_cl)

w_cl = []
for i in xrange(20):
    w_cl.append(np.loadtxt("/home/mj/public_html/nbar_clustering_Mr20_w_t"+str(i)+".dat"))
w_cl = np.array(w_cl)



merrs_cl = []
for i in xrange(20):
    merrs_cl.append(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(theta_cl[i], [16, 50, 84],
                                                axis=0))))

iteration = np.arange(20)
merrs_cl = np.array(merrs_cl)

labels=['$\log \ M_{0}$',r'$\sigma_{\logM}$',r'$\log \ M_{min}$',r'$\alpha$',r'$\log \ M_{1}$']

for i in xrange(2,5):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabel = ax.set_xlabel('Iteration', fontsize=20)
    ylabel = ax.set_ylabel(labels[i], fontsize=25)
    #title = ax.set_title(r'$M_{\ast} > 10^{10.5} M_{\odot}$', fontsize=25)

    ax.errorbar(iteration, merrs_cl[:,i][:,0], yerr=[merrs_cl[:,i][:,1], merrs_cl[:,i][:,2]], 
                fmt='--o' , capsize = 2, color='blue', linewidth=2)


    ax.plot(np.linspace(-1, 20, 100), np.ones(100)*true_hod[i]  , color='gray')

    #plt.xscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin = -1, xmax=20)
    #plt.ylim(ymin = -0.5, ymax = 1)

    fig.savefig('/home/mj/public_html/abcpaper/clustering_convergence'+str(i)+'.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

"""
model = Zheng07()
npts = 1e2
mass = np.logspace(11, 14, npts)
input_hod_satelite = model.mean_occupation_satellites(prim_haloprop = mass)
input_hod_centrals = model.mean_occupation_centrals(prim_haloprop = mass)

index = np.random.choice(range(len(theta_cl[-1])), 1000, p = w_cl[-1]/np.sum(w_cl[-1]))
output_hod_all , output_hod_cents , output_hod_sats= [] , [] , []
for i in index:
    mod = Zheng07()
    mod.param_dict["logM0"] = theta_cl[-1][i][0]
    mod.param_dict["sigma_logM"] = theta_cl[-1][i][1]
    mod.param_dict["logMmin"] = theta_cl[-1][i][2]
    mod.param_dict["alpha"] = theta_cl[-1][i][3]
    mod.param_dict["logM1"] = theta_cl[-1][i][4]
    #mod.populate_mock()
    output_hod_all.append(mod.mean_occupation_centrals(prim_haloprop = mass)+ mod.mean_occupation_satellites(prim_haloprop = mass))
    output_hod_sats.append(mod.mean_occupation_satellites(prim_haloprop = mass))
    output_hod_cents.append(mod.mean_occupation_centrals(prim_haloprop = mass))
    #ax.plot(mass, output_hod, color='blue', linewidth=.05)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm tot}\rangle$', fontsize=25)
title = ax.set_title('All Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_all), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_centrals + input_hod_satelite, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig2.savefig('/home/mj/public_html/abcpaper/hod_allgals.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')



fig3 = plt.figure()
ax = fig3.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm sat}\rangle$', fontsize=25)
title = ax.set_title('Satellite Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_sats), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_satelite, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig3.savefig('/home/mj/public_html/abcpaper/hod_satellites.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')



fig4 = plt.figure()
ax = fig4.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm cen}\rangle$', fontsize=25)
title = ax.set_title('Central Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_cents), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_centrals, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig4.savefig('/home/mj/public_html/abcpaper/hod_centrals.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

"""
fig5 = plt.figure()
ax = fig5.add_subplot(111)
xlabel = ax.set_xlabel('R [Mpc]', fontsize=20)
ylabel = ax.set_ylabel(r'$\xi_{\rm gg}$', fontsize=25)
title = ax.set_title(r'tpcf of galaxies with $M_{r} > -20$', fontsize=25)
m = Zheng07()
m.populate_mock()
rr = m.compute_galaxy_clustering()[0]
xir = np.loadtxt("xir_Mr20.dat")
xir_cov = np.loadtxt("clustering_covariance_Mr20.dat")
ax.errorbar(rr , xir , yerr = np.diag(xir_cov)**.5 , fmt="ok", ms=1, capsize=2, alpha=1.)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(xmin = 0.1, xmax=15)
ax.set_yscale("log")
ax.set_xscale("log")
fig5.savefig('/home/mj/public_html/abcpaper/tpcf.pdf',
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

"""groups"""

theta_gr = []
for i in xrange(12):
    theta_gr.append(np.loadtxt("/home/mj/public_html/nbar_gmf5_Mr20_theta_t"+str(i)+".dat"))
for i in range(12, 19):
    theta_gr.append(np.loadtxt("/home/mj/public_html/nbar_gmf4_Mr20_theta_t"+str(i)+".dat"))
theta_gr = np.array(theta_gr)

w_gr = []
for i in xrange(12):
    w_gr.append(np.loadtxt("/home/mj/public_html/nbar_gmf5_Mr20_w_t"+str(i)+".dat"))
for i in range(12, 19):
    w_gr.append(np.loadtxt("/home/mj/public_html/nbar_gmf4_Mr20_w_t"+str(i)+".dat"))
w_gr = np.array(w_gr)

merrs_gr = []
for i in xrange(19):
    merrs_gr.append(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(theta_gr[i], [16, 50, 84],
                                                axis=0))))

iteration = np.arange(19)
merrs_gr = np.array(merrs_gr)

labels=['$\log \ M_{0}$',r'$\sigma_{\logM}$',r'$\log \ M_{min}$',r'$\alpha$',r'$\log \ M_{1}$']

for i in xrange(2,5):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabel = ax.set_xlabel('Iteration', fontsize=20)
    ylabel = ax.set_ylabel(labels[i], fontsize=25)
    #title = ax.set_title(r'$M_{\ast} > 10^{10.5} M_{\odot}$', fontsize=25)

    ax.errorbar(iteration, merrs_gr[:,i][:,0], yerr=[merrs_gr[:,i][:,1], merrs_gr[:,i][:,2]], 
                fmt='--o' , capsize = 2, color='blue', linewidth=2)


    ax.plot(np.linspace(-1, 20, 100), np.ones(100)*true_hod[i]  , color='gray')

    #plt.xscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin = -1, xmax=20)
    #plt.ylim(ymin = -0.5, ymax = 1)

    fig.savefig('/home/mj/public_html/abcpaper/group_convergence'+str(i)+'.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

model = Zheng07()
npts = 1e2
mass = np.logspace(11, 14, npts)
input_hod_satelite = model.mean_occupation_satellites(prim_haloprop = mass)
input_hod_centrals = model.mean_occupation_centrals(prim_haloprop = mass)

index = np.random.choice(range(len(theta_gr[-1])), 200, p = w_gr[-1]/np.sum(w_gr[-1]))
output_hod_all , output_hod_cents , output_hod_sats= [] , [] , []

output_gmf = []
sdss_mean = np.loadtxt("gmf_Mr20.dat")
sdss_bins = np.loadtxt("gmf_bins_Mr20_2.dat")
sdss_noise = np.loadtxt("gmf_noise_Mr20_2.dat")
for i in index:

    mod = Zheng07()
    mod.param_dict["logM0"] = theta_gr[-1][i][0]
    mod.param_dict["sigma_logM"] = np.exp(theta_gr[-1][i][1])
    mod.param_dict["logMmin"] = theta_gr[-1][i][2]
    mod.param_dict["alpha"] = theta_gr[-1][i][3]
    mod.param_dict["logM1"] = theta_gr[-1][i][4]
    mod.populate_mock()
    group_id = mod.mock.compute_fof_group_ids()
    group_richness = richness(group_id)
    y = plt.hist(group_richness , sdss_bins)[0] / 250.**3.
    plt.close()
    output_gmf.append(y)
    output_hod_all.append(mod.mean_occupation_centrals(prim_haloprop = mass)+ mod.mean_occupation_satellites(prim_haloprop = mass))
    output_hod_sats.append(mod.mean_occupation_satellites(prim_haloprop = mass))
    output_hod_cents.append(mod.mean_occupation_centrals(prim_haloprop = mass))

fig6 = plt.figure()
ax = fig6.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm tot}\rangle$', fontsize=25)
title = ax.set_title('All Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_all), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_centrals + input_hod_satelite, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig6.savefig('/home/mj/public_html/abcpaper/hod_allgals_gr.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')



fig7 = plt.figure()
ax = fig7.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm sat}\rangle$', fontsize=25)
title = ax.set_title('Satellite Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_sats), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_satelite, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig7.savefig('/home/mj/public_html/abcpaper/hod_satellites_gr.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

fig8 = plt.figure()
ax = fig8.add_subplot(111)
xlabel = ax.set_xlabel(r'$M_{\rm vir} [M_{\odot}]$', fontsize=25)
ylabel = ax.set_ylabel(r'$\langle N_{\rm cen}\rangle$', fontsize=25)
title = ax.set_title('Central Galaxies', fontsize=20)
a, b, c, d, e = np.percentile(np.array(output_hod_cents), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(mass, a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(mass, b, d, color="b", alpha=0.6, edgecolor="none")
ax.plot(mass, input_hod_centrals, color='red', linewidth = 1.)
plt.loglog()
plt.xlim(xmin=1e11, xmax=1e14)
plt.ylim(ymin=5e-3, ymax=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig8.savefig('/home/mj/public_html/abcpaper/hod_centrals_gr.pdf', 
                bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')


sdss_mean = np.loadtxt("gmf_Mr20.dat")
sdss_bins = np.loadtxt("gmf_bins_Mr20_2.dat")
sdss_noise = np.loadtxt("gmf_noise_Mr20_2.dat")
output_gmf = np.array(output_gmf)
fig9 = plt.figure()
ax = fig9.add_subplot(111)


ax.errorbar(0.5*(sdss_bins[:-1]+sdss_bins[1:]) , sdss_mean, yerr = sdss_noise
             ,fmt="ok", capsize=1.0)
a, b, c, d, e = np.percentile(np.array(output_gmf), [2.5, 16, 50, 84, 97.5], axis=0)
ax.fill_between(0.5*(sdss_bins[:-1]+sdss_bins[1:]), a, e, color="b", alpha=0.2, edgecolor="none")
ax.fill_between(0.5*(sdss_bins[:-1]+sdss_bins[1:]), b, d, color="b", alpha=0.6, edgecolor="none")

ylabel = ax.set_ylabel(r"Group Multiplicity Function [h$^{3}$ Mpc$^{-3}$]")
xlabel = ax.set_xlabel("Group Richness")
ax.set_yscale(log)
plt.xlim(xmin = 1 , xmax = 60)
title = ax.set_title("gmf of galaxies with Mr>-20")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig9.savefig('/home/mj/public_html/abcpaper/gmf.pdf', 
             bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
