"""

PMC-ABC code 

"""
from prior import Prior

def pmc_abc(data, model, distance):
    """
    """

    # initial pool 
    pass


def initial_pool_sampling(i_particle, eps0, data, model, distance, prior_obj):
    """ Sample theta_star from prior distribution for the initial pool 
    returns [i_particle, theta_star, weights, rho]

    Parameters
    ----------
    i_particle : index of particle
    eps0       : initial threshold
    data       : observation or the summary statistcs of the observations
    model      : forward model (simulator function)  of the observation. It takes one particle "theta"
                 as an input and returns the value of the forward model for that particle.
    distance   : distance function between the data and the model evaluated at a given theta
 
    prior_obj  : Prior class object

    Note       : data, model, distance must be provided by the user.

    returns    : particle , weight, and its corresponding distance 

    """

    theta_star = prior_obj.sampler()
    n_param = len(theta_star)

    model_theta = model(theta_star)

    rho = distance(data, model_theta)

    while rho > eps0:
        theta_star = prior_obj.sampler()
        model_theta = model(theta_star)

        rho = distance(data, model_theta)

    pool_list = [np.int(i_particle)]
    for i_param in xrange(n_params):
        pool_list.append(theta_star[i_param])
    pool_list.append(1./np.float(N_particles))
    pool_list.append(rho)
