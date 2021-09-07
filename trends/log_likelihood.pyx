# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: binding=True
# The binding=True directive is useful for profiling code using kernprof. It also creates a lot of 'unused variable' warnings, so I will keep it commented out with two hashes for now

#import os
#import sys
#path = os.getcwd()
#sys.path.insert(1, path+'/trends')
#from trends.kern_profiler_dummy import *

import cython
from libc.math cimport log, sin, sqrt, abs

import numpy as np
cimport numpy
from numpy.math cimport INFINITY as inf

import scipy.stats as spst

import helper_functions_general as hlp
import helper_functions_rv as hlp_rv
import helper_functions_astro as hlp_astro


pi = np.pi
np.random.seed(0)

pc_in_cm = 3.086e18
two_pi = 2*np.pi

def mcmc(initial_state, data, n_steps, args):
    """
    MCMC sampler that will eventually be replaced by a professional package.
    Initial state is the first set of model parameters, hopefully a good guess.
    true_gammas and true_dmu are the data values and errors we measured.
    args = (m_star, rv_epoch, d_star)
    """
    cdef double a, m, e, i, om, M_anom_0, sig
    cdef int acceptance_count, n
    cdef double _a, _m, _e, _i, _om, _M_anom_0
    cdef double log_pri, proposed_log_prob, current_log_prob, prob
    cdef double accept_limit
    
    current_state = initial_state
    a, m, e, i, om, M_anom_0 = current_state
    
    stored_steps = []
    acceptance_count = 0
    sig = 1
    for n in range(n_steps):
        
        _a = np.random.normal(a, sig)
        _m = np.random.normal(m, sig)
        _e = e #np.random.normal(e, sig)
        _i = i #np.random.normal(i, sig)%two_pi
        _om = om #np.random.normal(om, sig)%two_pi
        _M_anom_0 = M_anom_0 #np.random.normal(M_anom_0, sig)%two_pi
        
        proposed_state = _a, _m, _e, _i, _om, _M_anom_0
        # The kepler solver will crash if I use e<0, so I need to catch out-of-bounds parameters BEFORE calculating.
        log_pri = log_prior(proposed_state)
        if log_pri > -inf:
            
            proposed_log_prob = log_lik_tot(proposed_state, data, args)
            current_log_prob = log_lik_tot(current_state, data, args)
        
            prob = np.exp(proposed_log_prob - current_log_prob)
        
        else:
            prob = -inf

        accept_limit = np.random.rand()
        if prob > accept_limit:
            acceptance_count += 1
            # Need them grouped for the function call but also ungrouped to propose new points
            current_state = _a, _m, _e, _i, _om, _M_anom_0
            a, m, e, i, om, M_anom_0 = current_state
        stored_steps.append([a, m])
    
    print('The acceptance rate is {}%.'.format(100*acceptance_count/n_steps))
    
    return stored_steps


def log_lik_tot(state, data, args):

    """
    Compute the log-likelihood of a state (set of a, Mp, e, i, om, and M_anom_0)
    given the RV data, astrometry data, and parameter priors.
    
    data = (gdot, gdot_error, gddot, gddot_error, dmu, dmu_error)
    args = m_star, rv_epoch, d_star
    """
    cdef double gdot_data, gdot_err, gddot_data, gddot_err, dmu_data, dmu_err
    cdef double m_star, rv_epoch, d_star
    cdef double log_lik_rv, log_lik_astro, log_pri
    
    gdot_data, gdot_err, gddot_data, gddot_err, dmu_data, dmu_err = data
    m_star, rv_epoch, d_star = args
    
    true_gammas = (gdot_data, gdot_err, gddot_data, gddot_err)
    true_dmu = (dmu_data, dmu_err)
    
    
    log_pri = log_prior(state)
    if log_pri > -inf:
        log_lik_rv = log_lik_gamma(state, true_gammas, (m_star, rv_epoch))
        log_lik_astro = log_lik_dmu(state, true_dmu, (m_star, d_star))

        log_lik_tot = log_lik_rv + log_lik_astro + log_pri

    else:
        log_lik_tot = -inf
    
    return log_lik_tot
    


def log_lik_gamma(state, true_gammas, args):
    """
    Compute the log-likelihood of a state (set of a, Mp, e, i, om, and M_anom_0)
    given the RV data (true gammas and their uncertainties).
    
    args = (m_star, rv_epoch), where m_star is in m_sun and rv_epoch is a bjd
    """
    
    cdef double a, m, e, i, om, M_anom_0 
    cdef double gdot_data, gdot_err, gddot_data, gddot_err, m_star, rv_epoch
    cdef double per, E, gdot_model, gddot_model
    cdef double log_likelihood_gdot, log_likelihood_gddot, log_likelihood_total
    
    a, m, e, i, om, M_anom_0 = state
    gdot_data, gdot_err, gddot_data, gddot_err = true_gammas
    m_star, rv_epoch = args
    
    per = hlp.P(a, m, m_star)
    E = hlp_rv.M_2_evolvedE(M_anom_0, per, e, rv_epoch)


    gdot_model, gddot_model = hlp_rv.gamma(a, m, per, e, i, om, E, m_star)

    # Log of the prefactor minus the log of the exponential
    log_likelihood_gdot  = log(1/(sqrt(two_pi)*gdot_err))\
                         - (gdot_data-gdot_model)**2/(2*gdot_err**2)
                         
    log_likelihood_gddot = log(1/(sqrt(two_pi)*gddot_err))\
                         - (gddot_data-gddot_model)**2/(2*gddot_err**2)
                 
    log_likelihood_total = log_likelihood_gdot + log_likelihood_gddot

    return log_likelihood_total


def log_lik_dmu(state, true_dmu, args):
    """
    Compute the log-likelihood of a given state given the astrometry data
    
    args = (m_star, d_star), where m_star is in m_sun and d_star is in cm
    """
    cdef double a, m, e, i, om, M_anom_0, dmu_data, dmu_data_err, m_star, d_star, dmu_model, log_likelihood

    a, m, e, i, om, M_anom_0 = state
    dmu_data, dmu_data_err = true_dmu
    m_star, d_star = args

    dmu_model = hlp_astro.dmu(a, m, e, i, om, M_anom_0, m_star, d_star)

    log_likelihood = log(1/(sqrt(two_pi)*dmu_data_err)) - (dmu_data - dmu_model)**2/(2*dmu_data_err**2)

    return log_likelihood


def log_prior(state, a_lim = (1, 1e2), m_lim = (1, 8e1)):

    cdef double a, m, e, i, om, M_anom_0
    cdef double a_min, a_max, m_min, m_max
    cdef bint a_pri, m_pri, e_pri, i_pri, om_pri, M_anom_0_pri
    cdef double log_prob_a, log_prob_m, log_prob_i, log_prob_e, log_prob

    a, m, e, i, om, M_anom_0 = state
    a_min, a_max = a_lim
    m_min, m_max = m_lim

    a_pri = a_min < a < a_max
    m_pri = m_min < m < m_max
    e_pri = 0 <= e < 0.99
    i_pri = -inf < i < inf
    om_pri = -inf < om < inf
    M_anom_0_pri = -inf < M_anom_0 < inf

    if a_pri and m_pri and e_pri and i_pri and om_pri and M_anom_0_pri: 
        log_prob_a = log(1/a/log(a_max/a_min))
        log_prob_m = log(1/m/log(m_max/m_min))
        log_prob_i = log(abs(sin(i)))
        log_prob_e = log(spst.beta.pdf(e, 0.867, 3.03))
        
        log_prob = log_prob_a + log_prob_m + log_prob_i + log_prob_e

    else:
        log_prob = -inf

    return log_prob
    
def return_one(x):
    """
    This function is just to pass to ptemcee.
    Since log_lik_tot() already accounts for priors,
    just set priors to 1 to obtain the posterior.
    """
    return 1
    











