#!/usr/bin/env python
import os 
import sys
import subprocess

import data

data_dir = raw_input('Please enter local directory to dump data : ')
u_sure = raw_input('Are you sure you want '+data_dir+' as your local data directory? [y/n]') 

if u_sure == 'y': 
    try: 
        os.symlink(data_dir, 'dat')
    except OSError: 
        os.remove('dat')
        u_sure2 = raw_input('Overwrite symlink? [y/n]')
        if u_sure2 == 'y': 
            os.symlink(data_dir, 'dat')
else: 
    raise ValueError("Don't doubt yourself next time") 

for dir in ['observations', 'pmc_abc', 'mcmc', 'crash']: 
    if not os.path.exists('dat/'+dir):
        print 'creating directory dat/'+dir+'/'
        os.makedirs('dat/'+dir)

build_observations(Mr=21, b_normal=0.25) 
