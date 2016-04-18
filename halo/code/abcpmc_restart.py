'''

Module for ABC-PMC inference

Author(s): Chang, MJ

Commandline call sequence : 

python abc_pemcee.py Niter Npart ObsStr OutputDir

- Niter : int
    Number of iterations 
- Npart : int
    Number of particles 
- ObsStr : string
    String that specifies the observables. 
    'nbarxi' is ['nbar', 'xi']
    'nbargmf' is ['nbar', 'gmf']
    'xi' is ['xi']
- OutputDir : string
    String that specifies the output directory
'''
import sys 
import time
import pickle
import numpy as np
import abcpmc
from abcpmc import mpi_util
# --- Local --- 
import util
import data_multislice as Data
from hod_sim import ABC_HODsim
from prior import PriorRange
from group_richness import richness
# --- Plotting ---
from plotting import plot_thetas

def ABCpmc_HOD(T, eps_val, N_part=1000, prior_name='first_try', observables=['nbar', 'xi'], 
        data_dict={'Mr':21}, output_dir=None):
    '''
    ABC-PMC implementation. 

    Parameters
    ----------
    - T : Number of iterations 
    - eps_val : 
    - N_part : Number of particles
    - observables : list of observables. Options are 'nbar', 'gmf', 'xi'
    - data_dict : dictionary that specifies the observation keywords 
    '''
    if output_dir is None: 
        output_dir = util.dat_dir()
    else: 
        pass
    #Initializing the vector of observables and inverse covariance matrix
    if observables == ['xi']:
        fake_obs = Data.data_xi(**data_dict)
        fake_obs_cov = Data.data_abc_cov(**data_dict)[1:16 , 1:16]
        xi_Cii = np.diag(fake_obs_cov)
    elif observables == ['nbar','xi']:
        fake_obs = np.hstack([Data.data_nbar(**data_dict), Data.data_xi(**data_dict)])
        fake_obs_cov = Data.data_abc_cov(**data_dict)[:16 , :16]
        Cii = np.diag(fake_obs_cov)
        xi_Cii = Cii[1:]
        nbar_Cii = Cii[0]
    elif observables == ['nbar','gmf']:
        fake_obs = np.hstack([Data.data_nbar(**data_dict), Data.data_gmf(**data_dict)])
        fake_obs_cov = Data.data_abc_cov(**data_dict)
        Cii = np.diag(fake_obs_cov)
        gmf_Cii = Cii[16:]
        nbar_Cii = Cii[0]
    # True HOD parameters
    data_hod_dict = Data.data_hod_param(Mr=data_dict['Mr'])
    data_hod = np.array([
        data_hod_dict['logM0'],                 # log M0 
        np.log(data_hod_dict['sigma_logM']),    # log(sigma)
        data_hod_dict['logMmin'],               # log Mmin
        data_hod_dict['alpha'],                 # alpha
        data_hod_dict['logM1']                  # log M1
        ])
    # Priors
    prior_min, prior_max = PriorRange(prior_name)
    prior = abcpmc.TophatPrior(prior_min, prior_max)
    prior_range = np.zeros((len(prior_min),2))
    prior_range[:,0] = prior_min
    prior_range[:,1] = prior_max
    # simulator
    our_model = ABC_HODsim(Mr=data_dict['Mr'])    # initialize model
    kwargs = {'prior_range': prior_range, 'observables': observables}
    def simz(tt): 
        sim = our_model.sum_stat(tt, **kwargs)
        if sim is None: 
            pickle.dump(tt, open("simz_crash_theta.p", 'wb'))
            pickle.dump(kwargs, open('simz_crash_kwargs.p', 'wb'))
            raise ValueError('Simulator is giving NonetType')
        return sim

    def multivariate_rho(datum, model): 
        dists = [] 
        if observables == ['nbar','xi']: 
            dist_nbar = (datum[0] - model[0])**2. / nbar_Cii 
 	    dist_xi = np.sum((datum[1:] - model[1:])**2. / xi_Cii)
            dists = [dist_nbar , dist_xi]
        elif observables == ['nbar','gmf']:
            dist_nbar = (datum[0] - model[0])**2. / nbar_Cii 
            dist_gmf = np.sum((datum[1:] - model[1:])**2. / gmf_Cii)
            dists = [dist_nbar , dist_gmf]
        elif observables == ['xi']: 
            dist_xi = np.sum((datum- model)**2. / xi_Cii)
    	    dists = [dist_xi]
        #print np.array(dists)
        return np.array(dists)

    def launch(eps_start, init_pool=None):
        print eps_start 
        eps = abcpmc.ConstEps(T, eps_start)
        mpi_pool = mpi_util.MpiPool()
        pools = []
        abcpmc_sampler = abcpmc.Sampler(
                N=N_part,               #N_particles
                Y=fake_obs,             #data
                postfn=simz,            #simulator 
                dist=multivariate_rho,  #distance function  
                pool=mpi_pool)  
        abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal
        f = open("abc_tolerance.dat" , "w")
        f.close()
        eps_str = ''
        for pool in abcpmc_sampler.sample(prior, eps):
            #while pool.ratio > 0.01:
            new_eps_str = '\t'.join(np.array(pool.eps).astype('str'))+'\n'
            if eps_str != new_eps_str:  # if eps is different, open fiel and append 
                f = open("abc_tolerance.dat" , "a")
                eps_str = new_eps_str
                f.write(eps_str)
                f.close()
            print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
            print pool.eps
            # plot theta
            #plot_thetas(pool.thetas, pool.ws , pool.t, 
            #        Mr=data_dict["Mr"], truths=data_hod, plot_range=prior_range, 
            #        observables=observables, output_dir=output_dir)

            # write theta and w to file 
            theta_file = ''.join([output_dir, util.observable_id_flag(observables), 
                '_Mr', str(data_dict["Mr"]), '_theta_t', str(pool.t), '.mercer.dat'])
            w_file = ''.join([output_dir, util.observable_id_flag(observables), 
                '_Mr', str(data_dict["Mr"]), '_w_t', str(pool.t), '.mercer.dat'])
            dist_file = ''.join([output_dir, util.observable_id_flag(observables), 
                '_Mr', str(data_dict["Mr"]), '_dist_t', str(pool.t), '.mercer.dat'])
            np.savetxt(theta_file, pool.thetas)
            np.savetxt(w_file, pool.ws)
            np.savetxt(dist_file , pool.dists)
            if pool.t < 3: 
                eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
            elif (pool.t > 2) and (pool.t < 20):
                eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
            else:
                eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
            #if eps.eps < eps_min:
            #eps.eps = eps_min
            pools.append(pool)
        abcpmc_sampler.close()
        return pools
         
    print "Initial launch of the sampler"
    pools = launch(eps_val)
     
    #print "Restarting ABC-PMC"

    #last_thetas = np.loadtxt("/home/mj/abc/halo/dat/nbar_gmf_Mr21_theta_t11.mercer.dat")
    #last_ws = np.loadtxt("/home/mj/abc/halo/dat/nbar_gmf_Mr21_w_t11.mercer.dat")
    #last_dists = np.loadtxt("/home/mj/abc/halo/dat/nbar_gmf_Mr21_dist_t11.mercer.dat")
    #last_eps = [0.0244727,10.20701988]
    #last_time = 11

    #print("Restarting after iteration: %s"%last_time)
    #restart_pool = abcpmc.PoolSpec(last_time, None, None, last_thetas, last_dists, last_ws)
    #eps_start = last_eps
    #pools2= launch(eps_start, restart_pool)
     
if __name__=="__main__": 

    Niter = int(sys.argv[1])
    print 'N iterations = ', Niter
    Npart = int(sys.argv[2])
    print 'N particles = ', Npart
    obv_flag = sys.argv[3]
    if obv_flag == 'nbarxi': 
        obv_list = ['nbar', 'xi']
    elif obv_flag == 'nbargmf':  
        obv_list = ['nbar', 'gmf']
    elif obv_flag == 'xi':
        obv_list = ['xi']
    else: 
        raise ValueError
    print 'Observables: ', ', '.join(obv_list)
    eps_list = [1.e10 for i in range(len(obv_list))]

    if len(sys.argv) > 4: 
        out_dir = sys.argv[4]
        if out_dir[-1] != '/': 
            out_dir += '/'
        print 'Output to ', out_dir
        ABCpmc_HOD(Niter, eps_list, N_part=Npart, observables=obv_list, output_dir=out_dir)
    else: 
        ABCpmc_HOD(Niter, eps_list, N_part=Npart, observables=obv_list, output_dir=None)
