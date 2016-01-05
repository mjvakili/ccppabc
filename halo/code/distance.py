'''

Distance measurements for ABC-PMC


'''

def rho(data, model, sigma=None, type='multivariate'):
    '''
    Calculate the distance between the data and model.

    Notes
    -----
    * Currently, three different types (univariate, multivariate, chi-squared distance)
    * Things ard hardcoded
    '''

    if type == 'added distance': 

        dist_nz = np.abs(d_data[0] - d_model[0])/d_data[0]
        dist_xi = np.sum(np.abs(d_data[1] - d_model[1])/d_data[1])
        
        dist = dist_nz + dist_xi 

    elif type == 'multivariate':
        
        dist_nz = (data[0] - model[0])**2. / covar_nz
        dist_gr = np.sum((data[1] - model[1])**2. / sigma**2.)
        
        dist = np.array([dist_nz , dist_gr])

    elif type == 'chisq distance':

        dist = np.sum((data - model)**2. / sigma **2.)
        
    return dist



