# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
## cython: binding=True

#import os
#import sys
#path = os.getcwd()
#sys.path.append(path+'/trends') # To import c_kepler
from c_kepler._kepler import kepler_single

from astropy.time import Time
import cython
from libc.math cimport sin, cos, tan, atan, sqrt

##########################################
#### Kepler solver for one M and one e
# Wrapping kepler(M,e) a simple function that takes two doubles as
# arguments and returns a double
#cdef extern from "../c_kepler/kepler.c":
#    double kepler(double M, double e)
#    double rv_drive(double t, double per, double tp, double e, double cosom, double sinom, double k )
##########################################

cdef float pi, two_pi, math_e, G, M_sun, M_jup, au, pc_in_cm, hip_beginning

pi = 3.141592653589793
two_pi = 6.283185307179586
math_e = 2.718281828459045
G = 6.674299999999999e-08
M_sun = 1.988409870698051e+33
M_jup = 1.8981245973360504e+30
au = 14959787070000.0
pc_in_cm = 3.086e18

hip_beginning = Time(1989.85, format='decimalyear').jd

# It seems gamma() needs to be a cdef function, otherwise it returns nans
# Testing the above comment by making it a cpdef so I can use it in log_likelihood.py
#@profile
cpdef (double, double) gamma(double a, double Mp, double per, 
      double e, double i, double om, double E, double m_star):

    cdef double     a_units, sqrt_eterm, tan_E2, nu,\
                    cos_E, tan_nu2, cos_E2, sin_i, cos_nu,\
                    sin_nu, cos_nu_om, sin_nu_om, sin_E,\
                    E_dot, nu_dot, prefac, gd_t1, gd_t2,\
                    gamma_dot, gamma_ddot

    #per_sec = per*86400 # 24*3600 to convert days ==> seconds
    Mp_g = Mp*M_jup
    m_star_g = m_star*M_sun

    a_star = a * Mp_g/m_star_g
    a_tot = a + a_star

    a_tot_cm = a_tot*au

    e_term = (1+e)/(1-e)
    sqrt_eterm = sqrt(e_term)
    sqrt_e_sq_term = sqrt(1-e*e)

    cos_E = cos(E)
    sin_E = sin(E)
    tan_Eovr2 = sin_E/(1+cos_E)

    nu = 2*atan(sqrt_eterm*tan_Eovr2)

    # nu derivatives use days (not seconds) to give gdot/gddot correct units 
    nu_dot = two_pi*sqrt_e_sq_term/(per*(1-e*cos_E)**2) # Units of day^-1
    nu_ddot = -nu_dot**2 * 2*e*sin_E/sqrt_e_sq_term # # Units of day^-2

    cos_nu_om = cos(nu+om)
    sin_nu_om = sin(nu+om)
    sin_i = sin(i)

    # Fischer (analytic)
    pre_fac = sqrt(G)/sqrt_e_sq_term * Mp_g*sin_i/sqrt((Mp_g+m_star_g)*(a_tot_cm)) * 1/100 # cm/s ==> m/s

    gamma_dot = -pre_fac*nu_dot*sin_nu_om # m/s/d
    gamma_ddot = -pre_fac*(nu_dot**2*cos_nu_om + nu_ddot*sin_nu_om) # m/s/d/d

    return gamma_dot, gamma_ddot

    
cpdef M_2_evolvedE(double M0, double per, double e, double rv_epoch):
    """
    Takes a mean anomaly at the beginning of the Hip mission, 
    evolves it to the RV epoch, and converts it to eccentric anomaly.

    M0 is the mean anomaly in radians.
    per is the period in days
    e is the eccentricity
    rv_epoch is the bjd corresponding to the ~midpoint of the RV baseline. 
    It is where gdot and gddot are measured
    """
    
    M_evolved = ((two_pi/per)*(rv_epoch - hip_beginning) + M0)%two_pi

    E = kepler_single(M_evolved, e)

    return E
    