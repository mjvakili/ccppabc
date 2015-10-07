"""

Plot the particles using DFM's Corner

"""
import corner
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_thetas(theta, w, prior_dict, iterno, basename="abc_run",
                truths=None, fig_name=None, **kwargs):
    """ Plot the parameter values of the particles
    """

    np.savetxt("{0}_{1}_thetas.dat".format(basename, str(iterno)), theta)
    np.savetxt("{0}_{1}_ws.dat".format(basename, str(iterno)), w)

    if 'range' in kwargs.keys():

        param_range = kwargs['range']

    else:
        ordered_key = prior_dict.keys()
        ordered_key.sort()

        param_range = []

        for key in ordered_key:
            if prior_dict[key]['shape'] == 'uniform':
                param_range.append(
                        (prior_dict[key]['min'], prior_dict[key]['max'])
                        )

            elif prior_dict[key]['shape'] == 'gaussian':
                param_range.append(
                        (prior_dict[key]['mean'] - 3.0 * prior_dict[key]['stddev'],
                            prior_dict[key]['mean'] + 3.0 * prior_dict[key]['stddev'])
                        )
            else:
                raise NotImplementedError()

    if 'labels' in kwargs.keys():
        plt_labels = kwargs['labels']
    else:
        plt_labels = ordered_key

    fig = corner.corner(
            theta.T,
            weights = w,
            smooth = True,
            truths = truths,
            truth_color = 'red',
            range = param_range,
            labels = plt_labels,
            levels = [0.68, 0.95],
            )

    if fig_name == None:
        fig_name = 'blah.png'

    fig.savefig(fig_name)
    plt.close()

