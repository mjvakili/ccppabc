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

print 'Downloading randoms ... '

build_obvs = raw_input("Do you want to download the observations [0] or build them from scratch [1]?") 

if not os.path.exists('../dat/observations/randoms.dat'):
    subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/randoms.dat', '-P', '../dat/observations/'])
if not os.path.exists('../dat/observations/RR.dat'):
    subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/RR.dat', '-P', '../dat/observations/'])

if build_obvs == 0:
    subprocess.call(['wget', 'http://physics.nyu.edu/~chh327/data/ccppabc/RR.dat', '-P', '../dat/observations/'])
elif build_obvs == 1: 
    build_observations(Mr=21, b_normal=0.25) 
