# import matplotlib.pyplot as plt
# import astropy.constants as c
# import numpy as np
# from astropy.time import Time
# from scipy.stats import loguniform, beta
#
# import radvel as rv
# import matplotlib.pyplot as plt
# import matplotlib.patches as ptch
import os
import sys
path = os.getcwd()
sys.path.insert(1, path+'/trends')
from log_likelihood import log_lik_tot, log_lik_gamma, log_lik_dmu, log_prior
import emcee
#import ptemcee

import numpy as np

pc_in_cm = 3.086e18
two_pi = 2*np.pi

# params_star = (m_star, distance(cm), gdot, gdot_err, gddot, gddot_err, 
#               rv_baseline(days), max_rv of residuals, rv_epoch, delta_mu, delta_mu_err)



def mcmc(initial_state, data, n_steps, args):
    """
    MCMC sampler that will eventually be replaced by a professional package.
    Initial state is the first set of model parameters, hopefully a good guess.
    true_gammas and true_dmu are the data values and errors we measured.
    args = (m_star, rv_epoch, d_star)
    """

    current_state = initial_state
    a, m, e, i, om, M_anom_0 = current_state
    gdot_data, gdot_error, gddot, gddot_error, dmu, dmu_error = data
    true_gammas = gdot_data, gdot_error, gddot, gddot_error
    true_dmu = dmu, dmu_error
    m_star, rv_epoch, d_star = args

    stored_steps = []
    acceptance_count = 0
    sig = 1
    for n in range(n_steps):

        _a = np.random.normal(a, sig)
        _m = np.random.normal(m, sig)
        _e = np.random.normal(e, sig)
        _i = i #np.random.normal(i, sig)%two_pi
        _om = om #np.random.normal(om, sig)%two_pi
        _M_anom_0 = M_anom_0 #np.random.normal(M_anom_0, sig)%two_pi

        proposed_state = _a, _m, _e, _i, _om, _M_anom_0
        # The kepler solver will crash if I use e<0, so I need to catch out-of-bounds parameters BEFORE calculating.
        log_pri = log_prior(proposed_state)
        if log_pri > -np.inf:

            proposed_log_prob = log_lik_gamma(proposed_state, true_gammas, (m_star, rv_epoch))\
                                + log_lik_dmu(proposed_state, true_dmu, (m_star, d_star))\
                                + log_pri

            current_log_prob = log_lik_gamma(current_state, true_gammas, (m_star, rv_epoch))\
                               + log_lik_dmu(current_state, true_dmu, (m_star, d_star))\
                               + log_prior(current_state)


            prob = np.exp(proposed_log_prob - current_log_prob)

        else:
            prob = -np.inf

        accept_limit = np.random.rand()
        if prob > accept_limit:
            acceptance_count += 1
            # Need them grouped for the function call but also ungrouped to propose new points
            current_state = _a, _m, _e, _i, _om, _M_anom_0
            a, m, e, i, om, M_anom_0 = current_state
        stored_steps.append([a, m])

    print('The acceptance rate is {}%.'.format(100*acceptance_count/n_steps))

    return stored_steps


if __name__ == "__main__":
    import time
    import corner
    import matplotlib.pyplot as plt
    
    from multiprocessing import Pool
    import os
    os.environ["OMP_NUM_THREADS"] = "12"
    
    
    # GL758, an example star in Tim Brandt's Orvara code. Using this to compare results.
    params_gl758 = (0.95, 15.5*pc_in_cm, -0.00633, 0.00025, -8.19e-7, 0.67e-7,
                    8413.010, 60, 2454995.123, 1.0397, 0.0261)

    # rv_epoch is the epoch where DATA values of g_dot and g_ddot are computed. Taken from radvel setup file.
    m_star, d_star, gammadot, gammadot_err, gammaddot, gammaddot_err,\
            rv_baseline, max_rv, rv_epoch, delta_mu, delta_mu_err = params_gl758
        
    data = (gammadot, gammadot_err, gammaddot, gammaddot_err, delta_mu, delta_mu_err)
    # true_gammas = (gammadot, gammadot_err, gammaddot, gammaddot_err)
    # true_dmu = (delta_mu, delta_mu_err)
    my_args = (m_star, rv_epoch, d_star)

    n_steps = int(1e4)
    
    # emcee implementation. Works better than homemade mcmc, but I can't get pool to work
    #################################################################
    means = [10, 10, 0.25, two_pi/8, two_pi/4, two_pi/6]
    sig   = [1, 1, 0.01, 0.1, 0.1, 0.1]
    cov = np.diag(sig)
    n_walkers = 13

    initial_state = np.random.multivariate_normal(means, cov, size = n_walkers)

    my_sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=6, log_prob_fn=log_lik_tot, args=[data, my_args])
    start = time.time()
    my_sampler.run_mcmc(initial_state, n_steps)
    end = time.time()
    run_time = end-start
    print("Serial took {0:.1f} seconds for {1} steps and {2} walkers".format(run_time, n_steps, n_walkers))

    # tau = my_sampler.get_autocorr_time()
    # print('TAU', tau)
    # chains has shape (n_steps, n_walkers, n_dimensions). Take only a and m.
    # chains = chains[:,:,:2]

    print('Acceptance frac for each walker is', my_sampler.acceptance_fraction)

    flat_samples = my_sampler.get_chain(discard=100, flat=True)[:,:2]


    fig = corner.corner(flat_samples, labels=['a','m','e','i','om','M'], show_titles=True)
    plt.show()

    
    # My homemade mcmc, superseded by emcee
    ##################################################################
    # a = 10
    # m = 10
    # e = 0.1
    # i = two_pi/12
    # om = two_pi/8
    # M_anom_0 = 0
    # initial_state = (a, m, e, i, om, M_anom_0)
    # args = my_args
    #
    # result = np.array(mcmc(initial_state, data, n_steps, args))
    #
    # print(result[-10:-1])
    #
    # a_list = result[:,0]
    # m_list = result[:,1]
    #
    # import corner
    # import matplotlib.pyplot as plt
    #
    # samples = np.vstack((a_list, m_list))
    # samples = np.transpose(samples)
    #
    # fig = corner.corner(samples, labels=['a', 'm'], show_titles=True)
    # plt.show()
    ##################################################################
    
    # Plots of likelihood versus a or m, to get a sense of which likelihood function dominates (it's RVs for GL758)
    #######################################################################################
    # import matplotlib.pyplot as plt
    #
    # a_list = np.linspace(1, 200, int(1e3))
    # tot_list = []
    #
    # for a in a_list:
    #
    #     rv_lik = -1*log_lik_gamma((a, m, e, i, om, M_anom_0), true_gammas, (m_star, rv_epoch))
    #     astro_lik = -1*log_lik_dmu((a, m, e, i, om, M_anom_0), true_dmu, (m_star, d_star))
    #
    #     tot_list.append(rv_lik + astro_lik)
    #     #print(rv_lik, astro_lik)
    #
    # plt.loglog(a_list, tot_list)
    # plt.show()
    # print(log_lik_gamma(initial_state, true_gammas, (m_star, rv_epoch)))
    # print(log_lik_dmu(initial_state, true_dmu, (m_star, d_star)))
    
    #######################################################################################
    













