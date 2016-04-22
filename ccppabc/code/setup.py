#!/usr/bin/env python
import os 
import sys
import subprocess

from data import build_observations

data_dir = raw_input('Please enter local directory to dump data : ')
u_sure = raw_input('Are you sure you want '+data_dir+' as your local data directory? [y/n]') 

if u_sure == 'y': 
    try: 
        os.symlink(data_dir, '../dat')
    except OSError: 
        os.remove('../dat')
        u_sure2 = raw_input('Overwrite symlink? [y/n]')
        if u_sure2 == 'y': 
            os.symlink(data_dir, '../dat')
else: 
    raise ValueError("Don't doubt yourself next time") 

for dir in ['observations', 'pmc_abc', 'mcmc', 'crash']: 
    if not os.path.exists('../dat/'+dir):
        print 'creating directory dat/'+dir+'/'
        os.makedirs('../dat/'+dir)

build_obvs = raw_input("Do you want to download the observations [0] or build them from scratch [1] (takes ~2 hours)?") 

if int(build_obvs) == 0:
    overwrite = raw_input("Overwrite the observations? [y/n]")
    if overwrite == 'y': 
        print 'Downloading observations ... '
	for file in [
                'ABC.nbar_xi_gmf_cov.no_poisson.Mr21.bnorm0.25.dat', 'MCMC.nbar_xi_gmf_cov.no_poisson.Mr21.bnorm0.25.dat', 
                'RR.md_sub.dat', 'RR.md_all.dat', 
                'randoms.md_sub.dat', 'randoms.md_all.dat', 
                'abc_nbar_var.Mr21.dat', 'data_vector.Mr21.bnorm0.25.dat', 'nbar_var.Mr21.dat', 'xir_rbin.Mr21.dat']: 
            subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/'+file, '-P', '../dat/observations/'])
    else: 
        print "Then do it yourself manually @ http://physics.nyu.edu/~chh327/data/ccppabc/"

elif int(build_obvs) == 1: 
    print 'Downloading randoms ... '
    if not os.path.exists('../dat/observations/randoms.dat'):
        subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/randoms.dat', '-P', '../dat/observations/'])
    if not os.path.exists('../dat/observations/RR.dat'):
        subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/RR.dat', '-P', '../dat/observations/'])
    print 'Building observations ... '
    build_observations(Mr=21, b_normal=0.25) 
