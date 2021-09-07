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
from log_likelihood import log_lik_tot, log_lik_gamma, log_lik_dmu, log_prior, return_one
import emcee
import ptemcee

import numpy as np

pc_in_cm = 3.086e18
two_pi = 2*np.pi

# params_star = (m_star, distance(cm), gdot, gdot_err, gddot, gddot_err, 
#               rv_baseline(days), max_rv of residuals, rv_epoch, delta_mu, delta_mu_err)



if __name__ == "__main__":
    
    ##################################################
    from helper_functions_rv import gamma
    from helper_functions_general import P
    a = 10
    Mp = 10
    m_star = 0.8
    per = P(a, Mp, m_star)
    e = 0.5
    i = np.pi/4
    om = np.pi/3
    E = 5*np.pi/7

    print(gamma(a, Mp, per, e, i, om, E, m_star), per)
    dfdd
    ###################################################
    
    import time
    import corner
    import matplotlib.pyplot as plt
    
    from multiprocessing import Pool
    import os
    os.environ["OMP_NUM_THREADS"] = "12"
    
    
    # GL758, an example star in Tim Brandt's Orvara code. Using this to compare results.
    params_gl758 = (0.95, 15.5*pc_in_cm, -0.00633, 0.00025, -8.19e-7, 0.67e-7,
                    8413.010, 60, 2454995.123, 1.0397, 0.0261)
    params_191939 = (0.807, 58.3*pc_in_cm, 0.114, 0.006, -6e-5, 1.9e-5, 
                    430.2527364352718, 40.0581900021484, 2458847.780463, 0.12767382507786398, 0.034199052901953214)

    # rv_epoch is the epoch where DATA values of g_dot and g_ddot are computed. Taken from radvel setup file.
    m_star, d_star, gammadot, gammadot_err, gammaddot, gammaddot_err,\
            rv_baseline, max_rv, rv_epoch, delta_mu, delta_mu_err = params_191939
        
    data = (gammadot, gammadot_err, gammaddot, gammaddot_err, delta_mu, delta_mu_err)
    true_gammas = (gammadot, gammadot_err, gammaddot, gammaddot_err)
    true_dmu = (delta_mu, delta_mu_err)
    my_args = (m_star, rv_epoch, d_star)

    n_steps = int(1e3)
    
    # emcee implementation. Works better than homemade mcmc, but I can't get pool to work. Edit: Orvara uses emcee/ptemcee too, and says pool isn't reliable.
    #################################################################
    # means = [10, 10, 0.25, two_pi/8, two_pi/4, two_pi/6]
    # sig   = [1, 1, 0.01, 0.1, 0.1, 0.1]
    # cov = np.diag(sig)
    # n_walkers = 30
    #
    # initial_state = np.random.multivariate_normal(means, cov, size = n_walkers)
    #
    # my_sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=6, log_prob_fn=log_lik_tot, args=[data, my_args])
    # start = time.time()
    # my_sampler.run_mcmc(initial_state, n_steps)
    # end = time.time()
    # run_time = end-start
    # print("Serial took {0:.1f} seconds for {1} steps and {2} walkers".format(run_time, n_steps, n_walkers))
    #
    # # tau = my_sampler.get_autocorr_time()
    # # print('TAU', tau)
    # # chains has shape (n_steps, n_walkers, n_dimensions). Take only a and m.
    # # chains = chains[:,:,:2]
    #
    # print('Acceptance fraction for each walker is', my_sampler.acceptance_fraction)
    #
    # flat_samples = my_sampler.get_chain(discard=100, flat=True)[:,:2]
    #
    #
    # fig = corner.corner(flat_samples, labels=['a','m','e','i','om','M'], show_titles=True)
    # plt.show()
    ######################################################################################
    
    # ptemcee experimentation.
    #################################################################
    means = [4.5, 7, 0.7, two_pi/8, two_pi/4, two_pi/6]
    sig   = [1, 1, 0.01, 1, 1, 1]
    cov = np.diag(sig)
    n_temps = 10
    n_walkers = 32

    initial_state = np.random.multivariate_normal(means, cov, size = (n_temps, n_walkers))
    #print(initial_state[:,:,0])
    
    #my_sampler = ptemcee.Sampler(nwalkers=n_walkers, ndim=6, log_prob_fn=log_lik_tot, args=[data, my_args])
    my_sampler = ptemcee.Sampler(nwalkers=n_walkers, dim=6,
                                logl=log_lik_tot, ntemps=n_temps,
                                logp = return_one, loglargs=[data, my_args])
    start = time.time()
    my_sampler.run_mcmc(p0=initial_state, iterations=n_steps)
    end = time.time()
    run_time = end-start
    print("Serial took {0:.1f} seconds for {1} steps and {2} walkers".format(run_time, n_steps, n_walkers))

    # tau = my_sampler.get_autocorr_time()
    # print('TAU', tau)
    # chains has shape (n_steps, n_walkers, n_dimensions). Take only a and m.
    # chains = chains[:,:,:2]

    print('Acceptance fraction for each walker is', my_sampler.acceptance_fraction)

    flat_samples = my_sampler.flatchain[:,:2]
    flatter_samples = flat_samples.reshape((-1, 2))

    print(np.shape(flatter_samples))

    fig = corner.corner(flatter_samples, labels=['a','m','e','i','om','M'], show_titles=True)
    plt.show()
    #######################################################################################
    
    # Plots of likelihood versus a or m, to get a sense of which likelihood function dominates (it's RVs for GL758)
    #######################################################################################
    # import matplotlib.pyplot as plt
    #
    # a, m, e, i, om, M_anom_0 = [29, 38, 0.25, two_pi/8, two_pi/4, two_pi/6]
    # m_list = np.linspace(1, 200, int(1e3))
    # tot_list = []
    #
    # for m in m_list:
    #
    #     rv_lik = -1*log_lik_gamma((a, m, e, i, om, M_anom_0), true_gammas, (m_star, rv_epoch))
    #     astro_lik = -1*log_lik_dmu((a, m, e, i, om, M_anom_0), true_dmu, (m_star, d_star))
    #
    #     tot_list.append(rv_lik + astro_lik)
    #     #print(rv_lik, astro_lik)
    #
    # plt.loglog(m_list, tot_list)
    # plt.show()
    # print(log_lik_gamma(initial_state, true_gammas, (m_star, rv_epoch)))
    # print(log_lik_dmu(initial_state, true_dmu, (m_star, d_star)))
    
    #######################################################################################
    













