import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns
import scipy.stats
import sys
sys.path.insert(0, '../../ccppabc/')
from pmc_abc import ABC



class ABC_class(object):


    def __init__(self , ndata , mu , mean , sigma):
        self.ndata = ndata
        self.mu = mu 
        self.sigma = sigma
        self.mean = mean
        self.data=self.model(self.mu)  
    
    def model(self , m):
        stdev = 1.
        return np.random.normal(m , stdev , self.ndata)

    def true_post(self,size):
        
        post_stdev = np.sqrt(self.sigma**2. / (1. + self.ndata * self.sigma**2.))   #stdev of the true posterior distribution
        post_mean  = self.mean * 1. / (1. + self.sigma**2. * self.ndata) + \
                     self.ndata * np.mean(self.data) * self.sigma**2. /(1. + self.sigma**2. * self.ndata)
	
        return np.random.normal(post_mean , post_stdev , size=size)

    def dist(self,x,y):
        return np.array([np.abs((np.mean(x) - np.mean(y))/np.mean(x)) , np.abs((np.std(x) - np.std(y))/np.std(x))])
    

n = 500
mu = 0.
mean = 0.
std = .1

abc = ABC_class(n , mu , mean , std)

test_data = abc.data
test_dist = abc.dist
test_simulator = abc.model

prior_dict = {
              "mean":
               {
                "shape": 'gauss',
                'mean'  : 0.,
                'stddev'  :  .1,
               }
             }


test_abc = ABC(test_data, test_simulator, test_dist, prior_dict , eps0 = [10 ,10] , T = 1)

def run_serial():

    test_abc.basename = "test_abc_serial"
    test_abc.N_particles = 100
    print "testing ABC implementation in serial..."

    test_abc.run_abc()

run_serial()

