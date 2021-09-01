# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
## cython: binding=True

#import os
#import sys
#path = os.getcwd()
#sys.path.append(path+'/trends') # To import c_kepler
from c_kepler._kepler import kepler_single

from astropy.time import Time
import numpy as np
import cython
cimport numpy as np
from libc.math cimport sin, cos, tan, atan, sqrt


import helper_functions_general as hlp

cdef float pi, two_pi, math_e, G, M_sun, M_jup, au, pc_in_cm, baseline_yrs
cdef float hip_times[2]
cdef float gaia_times[2]

pi = 3.141592653589793
two_pi = 6.283185307179586
math_e = 2.718281828459045
G = 6.674299999999999e-08
M_sun = 1.988409870698051e+33
M_jup = 1.8981245973360504e+30
au = 14959787070000.0
pc_in_cm = 3.086e18

#https://www.cosmos.esa.int/web/hipparcos/catalogue-summary
hip_times  = [Time(1989.85, format='decimalyear').jd, Time(1993.21, format='decimalyear').jd] 
#https://www.cosmos.esa.int/web/gaia/earlydr3
gaia_times = [Time('2014-07-25', format='isot').jd, Time('2017-05-28', format='isot').jd] 

baseline_yrs = ((gaia_times[1] + gaia_times[0])/2 - (hip_times[1] + hip_times[0])/2)/365.25

def dmu(double a, double m, double e, double i, double om, double M_anom_0, 
        double m_star, double d_star):
    """
    Compute delta_mu for a set of model parameters
    """
    cdef double per, mean_motion, M1, M2, E1, E2
    cdef double x_pos_avg, y_pos_avg, x_vel_avg, y_vel_avg
    
    cdef double [:,::1] rot_mtrx # This makes rot_mtrx a memview
    rot_mtrx = np.zeros((3,3),dtype=np.float64)
    
    cdef double r_vec_list[3]
    cdef double [:] r_vec = r_vec_list

    cdef double rot_vec_list[3]
    cdef double [:] rot_vec = rot_vec_list

    cdef double ang_pos_list[2]
    cdef double [:] ang_pos = ang_pos_list
    
    cdef double time_endpoints[2][2]
    cdef double ang_pos_avg[2][2]
    cdef double mu_avg[2][2]
    cdef double mu_gaia[2]
    cdef double mu_hg[2]
    
    
    #######################################
    cdef double v_vec_star_list[3]
    cdef double [:] v_vec_star = v_vec_star_list

    cdef double rotated_v_vec_list[3]
    cdef double [:] rotated_v_vec = rotated_v_vec_list

    cdef double mu[2]
    #######################################

    mass_ratio_constant = M_jup/(m_star*M_sun)
    mass_ratio = m*mass_ratio_constant
    cm_2_mas = (206265*1e3)/d_star
    cmd_2_masyr = cm_2_mas * 365.25 # cm/day to milli-arcseconds/year

    time_endpoints = [[hip_times[0], gaia_times[0]], 
                      [hip_times[1], gaia_times[1]]]


    per = hlp.P(a, m, m_star)
    mean_motion = two_pi/per
    sqrt_eterm = sqrt((1+e)/(1-e))
    a_units = a*au
    a_star_units = a_units*mass_ratio
    e_sq = e**2
    rot_matrix(i, om, 0, rot_mtrx) # Omega = 0 arbitrarily
    r_star_num_fac = a_units*(1-e_sq)

    
    for l in range(2): # Hipparcos or Gaia
        start_time = time_endpoints[0][l] - time_endpoints[0][0] # The "start time" of Hip or Gaia relative to the start of Hip. For Hip, start_time is 0. For Gaia, it is the time between Hip_start and Gaia_start
        end_time = time_endpoints[1][l] - time_endpoints[0][0] # End time relative to the start of Hip.


        ## Mean anomaly is the elapsed time times the mean motion, plus a randomly-sampled starting mean anomaly
        M1 = (mean_motion*start_time + M_anom_0)%two_pi
        M2 = (mean_motion*end_time + M_anom_0)%two_pi

        E1 = kepler_single(M1, e)
        E2 = kepler_single(M2, e)

        # Get position of the STAR.
        x_pos_avg, y_pos_avg = pos_avg(a_star_units, mean_motion, e, 
                                       E1, E2, start_time, end_time)

        # r_vec points from barycenter to the *star* (note the - sign) in the orbital plane, and has magnitude r_star. Like r_star, it has units of cm.
        # Since we're using the E_anom for the planet, the star is located in the opposite direction
        r_vec[0] = -x_pos_avg
        r_vec[1] = -y_pos_avg
        r_vec[2] = 0

        # rot_vec points from barycenter to the star, but in coordinates where the xy-plane is the sky plane and the z-axis points toward Earth
        mat_mul(rot_mtrx, r_vec, rot_vec)

        # ang_pos is the angular position of the star relative to barycenter in milli-arcseconds.
        ang_pos[0] = rot_vec[0]*cm_2_mas
        ang_pos[1] = rot_vec[1]*cm_2_mas

        ang_pos_avg[l][0] = ang_pos[0]
        ang_pos_avg[l][1] = ang_pos[1]

        ################### Angular Velocities ########################

        # I only need Gaia velocities, not Hip velocities
        if l == 0:
            continue

        # Get velocity of the star
        x_vel_avg, y_vel_avg = vel_avg(a_star_units, mean_motion, e, 
                                       E1, E2, start_time, end_time)

        # Since we're using the E_anom for the planet, the star is moving in the opposite direction
        v_vec_star[0] = -x_vel_avg
        v_vec_star[1] = -y_vel_avg
        v_vec_star[2] = 0

        mat_mul(rot_mtrx, v_vec_star, rotated_v_vec)

        # mu is the proper motion of the star due to the planet's orbit in milli-arcseconds per year.
        # Since period is in days and a_star_units in cm, velocities are in cm/day.
        mu[0] = rotated_v_vec[0]*cmd_2_masyr
        mu[1] = rotated_v_vec[1]*cmd_2_masyr

        # mu_avg is a 2x2 array. The top row stays empty because we skip Hip. The bottom row is Gaia prop. motion
        mu_avg[l][0] = mu[0]
        mu_avg[l][1] = mu[1]

        ###############################################################
        ###############################################################

    mu_gaia = mu_avg[1]

    # To get the positional avg., subtract the epoch positions and divide by the time between in years.
    # First index tells Hip ([0]) or Gaia ([1]), second index tells x ([0]) or y ([1])
    # Units of mas/yr
    mu_hg[0] = (ang_pos_avg[1][0] - ang_pos_avg[0][0])/baseline_yrs # x-comp. = gaia_x - hip_x
    mu_hg[1] = (ang_pos_avg[1][1] - ang_pos_avg[0][1])/baseline_yrs # y-comp. = gaia_y - hip_y

    dmu_model = sqrt((mu_hg[0] - mu_gaia[0])**2 + (mu_hg[1] - mu_gaia[1])**2)

    return dmu_model


cdef pos_avg(double a, double n, double e, double E1, double E2, 
                  double t1, double t2):
    """
    Calculate the average x/y positions of an object on an elliptical orbit, 
    where (0,0) is the focus.

    a (cm): semi-major axis
    n (1/days): 2pi/per
    t1, t2 (days): beginning and ending time to calculate average

    returns: Average x and y positions (cm)
    """

    cdef double x_term_1, x_term_2, x_integral, x_avg,\
                y_term_1, y_term_2, y_integral, y_avg

    x_term_1 = a/n * ((1+e**2)*sin(E1) - e/4 * (6*E1+sin(2*E1)))
    x_term_2 = a/n * ((1+e**2)*sin(E2) - e/4 * (6*E2+sin(2*E2)))

    x_integral = x_term_2 - x_term_1

    x_avg = 1/(t2-t1) * x_integral

    y_term_1 = -a*sqrt(1-e**2)/n * (cos(E1) - e/2 * cos(E1)**2 )
    y_term_2 = -a*sqrt(1-e**2)/n * (cos(E2) - e/2 * cos(E2)**2 )
 
    y_integral = y_term_2 - y_term_1
 
    y_avg = 1/(t2-t1) * y_integral


    return x_avg, y_avg

cdef vel_avg(double a, double n, double e, double E1, double E2, 
                  double t1, double t2):
    """
    Calculate the average x/y positions of an object on an elliptical orbit, 
    where (0,0) is the focus.

    a (cm): semi-major axis
    n (1/days): 2pi/per
    t1, t2 (days): beginning and ending time to calculate average

    returns: Average x and y velocities (cm/day)
    """

    cdef double x_term_1, x_term_2, x_integral, x_avg,\
                y_term_1, y_term_2, y_integral, y_avg,\
                r1, r2

    r1 = a*(1-e*cos(E1))
    r2 = a*(1-e*cos(E2))

    x_term_1 = a**2/r1 * (cos(E1) + e/2 * sin(E1)**2)
    x_term_2 = a**2/r2 * (cos(E2) + e/2 * sin(E2)**2)

    x_integral = x_term_2 - x_term_1

    x_avg = 1/(t2-t1) * x_integral


    y_term_1 = a**2*sqrt(1-e**2)/r1 * (sin(E1) - e/2 * (E1 + 0.5*sin(2*E1)) )
    y_term_2 = a**2*sqrt(1-e**2)/r2 * (sin(E2) - e/2 * (E2 + 0.5*sin(2*E2)) )

    y_integral = y_term_2 - y_term_1

    y_avg = 1/(t2-t1) * y_integral


    return x_avg, y_avg

#@profile
cdef void rot_matrix(double i, double om, double Om, double [:,::1] rot_mtrx):
    """
    This is P3*P2*P1 from Murray & Dermott. It is not given explicitly in the text. 
    They multiply it immediately by r*[cos(f), sin(f), 0] because this gives the 
    projection of position onto the sky. However, we also need the projection of 
    velocity, so we need the matrix before multiplication by the position vector.

    This function doesn't return anything. Instead, declare a matrix in your 
    function and this will update it, saving lots of time by not allocating memory 
    to and returning a matrix.
    """
    cdef double sin_Om, sin_om, sin_i, cos_Om, cos_om, cos_i

    sin_Om = sin(Om)
    sin_om = sin(om)
    sin_i  = sin(i)
    cos_Om = cos(Om)
    cos_om = cos(om)
    cos_i  = cos(i)
    sin_Om_cos_i = sin_Om*cos_i
    cos_Om_cos_i = cos_Om*cos_i

    rot_mtrx[0][0] = cos_Om*cos_om - sin_Om_cos_i*sin_om
    rot_mtrx[0][1] = -sin_om*cos_Om - sin_Om_cos_i*cos_om
    rot_mtrx[0][2] = sin_Om*sin_i

    rot_mtrx[1][0] = sin_Om*cos_om + cos_Om_cos_i*sin_om
    rot_mtrx[1][1] = -sin_om*sin_Om + cos_Om_cos_i*cos_om
    rot_mtrx[1][2] = -cos_Om*sin_i

    rot_mtrx[2][0] = sin_i*sin_om
    rot_mtrx[2][1] = sin_i*cos_om
    rot_mtrx[2][2] = cos_i

    #return rot_mtrx


#@profile
cdef void mat_mul(double [:,:] mat, double [:] in_vec, double [:] out_vec):
    """
    This is written specifically to matrix multiply rot_matrix (3x3) with
    r_unit_vec (3x1) and later v_vec_star (3x1) in astro_post_dense_loop.
    """

    cdef int i, k

    for i in range(3):
        out_vec[i] = 0
        for k in range(3):
            out_vec[i] += mat[i][k]*in_vec[k]
            
            
            