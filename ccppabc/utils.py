
def weighted_sampling(theta, w):
 
    """ Given array of thetas and their corresponding weights, sample
    """
    w_cdf = w.cumsum()/w.sum() # normalized CDF
    
    np.random.seed()
    rand1 = np.random.random(1)
    cdf_closest_index = np.argmin( np.abs(w_cdf - rand1) )
    closest_theta = theta[:, cdf_closest_index]
    
    return closest_theta

def better_multinorm(theta_stst, theta_before, cov): 
    n_par, n_part = theta_before.shape
    
    sig_inv = np.linalg.inv(cov)
    x_mu = theta_before.T - theta_stst

    nrmliz = 1.0 / np.sqrt( (2.0*np.pi)**n_par * np.linalg.det(cov))

    multinorm = nrmliz * np.exp(-0.5 * np.sum( (x_mu.dot(sig_inv[None,:])[:,0,:]) * x_mu, axis=1 ) )

    return multinorm

def covariance(theta , w , type = 'weighted'):

    if type == 'neutral':

      return np.cov(theta)

    if type == 'normalized neutral':

      return np.corrcoef(theta)

    if type == 'weighted':
      mean = np.sum(theta*w[None,:] , axis = 1)/ np.sum(w)
      tmm  = theta - mean.reshape(theta.shape[0] , 1)
      sigma2 = 1./(w.sum()) * (tmm*w[None,:]).dot(tmm.T)
      return sigma2 
