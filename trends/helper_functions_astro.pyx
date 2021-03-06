# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: binding=True
# cython: profile=True
# cython: linetrace=True
from kern_profiler_dummy import *
from c_kepler._kepler import kepler_single

from astropy.time import Time
from tqdm import tqdm
import numpy as np
import cython
cimport numpy as np
from libc.math cimport sin, cos, tan, atan, sqrt, log


cdef double two_pi, math_e, G, M_sun, M_jup, au, pc_in_cm, baseline_yrs
cdef double hip_times[2]
cdef double gaia_times[2]

two_pi = 6.283185307179586
math_e = 2.718281828459045

old_G = 6.674299999999999e-08
# G in AU, M_Jup, day units.
G = 2.824760877012879e-07 # (c.G.cgs*(1/c.au.cgs)**3 * (c.M_jup.cgs) * (24*3600)**2).value

#https://www.cosmos.esa.int/web/hipparcos/catalogue-summary
hip_times  = [Time(1989.85, format='decimalyear').jd, Time(1993.21, format='decimalyear').jd] 
#https://www.cosmos.esa.int/web/gaia/earlydr3
gaia_times = [Time('2014-07-25', format='isot').jd, Time('2017-05-28', format='isot').jd]

baseline_yrs = ((gaia_times[1] + gaia_times[0])/2 - (hip_times[1] + hip_times[0])/2)/365.25

def astro_list(double [:] a_list, double [:] m_list, double [:] e_list, 
               double [:] i_list, double [:] om_list, double [:] M_anom_0_list, 
               double [:] per_list, 
               double m_star, double d_star, double delta_mu, double delta_mu_err):
      
    cdef int num_points, j
    cdef double a, m, e, i, om, M_anom_0, per, log_lik
    num_points = a_list.shape[0]     
        
    cdef np.ndarray[double, ndim=1] lik_list = np.ndarray(shape=(num_points,), dtype=np.float64)
    
    if delta_mu < 0:
       raise Exception()
    
    print('Running astrometry models')
    for j in tqdm(range(num_points)):
       
        a = a_list[j]
        m = m_list[j]
        e = e_list[j]
        i = i_list[j]
        om = om_list[j]
        M_anom_0 = M_anom_0_list[j]
        per = per_list[j]
        
        log_lik = log_lik_dmu(a, m, e, i, om, M_anom_0,
                              per, m_star, d_star,
                              delta_mu, delta_mu_err)
                              
        lik_list[j] = math_e**log_lik
    
    return lik_list


def log_lik_dmu(double a, double m, double e, double i, double om, double M_anom_0, 
                double per, double m_star, double d_star,
                double dmu_data, double dmu_data_err):
    """
    Compute the log-likelihood of a given state given the astrometry data
    """
    cdef double dmu_model, log_likelihood
    
    dmu_model = dmu(a, m, e, i, om, M_anom_0, per, m_star, d_star)
    
    log_likelihood = log(1/(sqrt(two_pi)*dmu_data_err)) - (dmu_data - dmu_model)**2/(2*dmu_data_err**2)
    
    return log_likelihood

#@profile
def dmu(double a, double m, double e, double i, double om, double M_anom_0, 
        double per, double m_star, double d_star):
    """
    Compute delta_mu for a set of model parameters
    
    Arguments:
        a (float, AU): Semi-major axis
        m (float, M_jup): Companion mass
        e (float): Orbital eccentricity
        i (float, radians): Orbital inclination
        om (float, radians): Argument of periastron
        M_anom_0 (float, radians): Mean anomaly at the beginning
                                   of the Hipparcos mission
        per (float, days): Orbital period
        m_star (float, M_jup): Host star mass
        d_star (float, AU): Distance of system from Earth
    
    Returns:
        dmu_model (float, mas/yr): Magnitude of difference between
                                   Gaia pm and positional average
                                   pm between Gaia and Hipparcos. 
        
    """
    cdef double mass_ratio, au_2_mas, aud_2_masyr
    cdef double mean_motion, sqrt_eterm, M1, M2, E1, E2
    cdef double x_pos_avg, y_pos_avg, x_vel_avg, y_vel_avg
    
    cdef double [:,:] rot_mtrx # This makes rot_mtrx a memview
    rot_mtrx = np.zeros((3,3),dtype=np.float64)
    
    # vec holds various values throughout dmu(). After each value
    # has served its purpose, it is overwritten so that only one
    # vector needs to be allocated. This saves time.
    cdef double vec_list[3]
    cdef double [:] vec = vec_list
    
    cdef double time_endpoints[2][2]
    cdef double ang_pos_avg[2][2]
    cdef double mu_avg[2][2]
    cdef double mu_gaia[2]
    cdef double mu_hg[2]
    
    mass_ratio = m/(m_star + m)
    d_star = d_star/206264.80624548031 # Divide by (c.pc.cgs/c.au.cgs).value to get units of pc
    au_2_mas = 1e3/d_star # Conversion factor btwn au and milli-arcseconds
    aud_2_masyr = au_2_mas * 365.25 # au/day to milli-arcseconds/year

    time_endpoints = [[hip_times[0], gaia_times[0]], 
                      [hip_times[1], gaia_times[1]]]


    mean_motion = two_pi/per
    sqrt_eterm = sqrt((1+e)/(1-e))
    a_star = a*mass_ratio
    e_sq = e**2
    rot_matrix(i, om, 0, rot_mtrx) # Omega = 0 arbitrarily
    r_star_num_fac = a*(1-e_sq)
    
    
    
    for l in range(2): # Hipparcos or Gaia
        start_time = time_endpoints[0][l] - time_endpoints[0][0] # The "start time" of Hip or Gaia relative to the start of Hip. For Hip, start_time is 0. For Gaia, it is the time between Hip_start and Gaia_start
        end_time = time_endpoints[1][l] - time_endpoints[0][0] # End time relative to the start of Hip.


        ## Mean anomaly is the elapsed time times the mean motion, plus a randomly-sampled starting mean anomaly
        # Mod 2?? should be the same, but it changes the answer. Staying with no mod in case mod introduces error.
        ##M1 = (mean_motion*start_time + M_anom_0)%two_pi
        ##M2 = (mean_motion*end_time + M_anom_0)%two_pi
        M1 = mean_motion*start_time + M_anom_0
        M2 = mean_motion*end_time + M_anom_0

        E1 = kepler_single(M1, e)
        E2 = kepler_single(M2, e)
        

        # Get position of the STAR (au).
        x_pos_avg, y_pos_avg = pos_avg(a_star, mean_motion, e, 
                                       E1, E2, start_time, end_time)

        # vec points from barycenter to the *star* (note the - sign) in the orbital plane, and has magnitude r_star. Like r_star, it has units of cm.
        # Since we're using the E_anom for the planet, the star is located in the opposite direction
        vec[0] = -x_pos_avg
        vec[1] = -y_pos_avg
        vec[2] = 0

        # vec is overwritten and replaced by the rotated version. The rotated version points from barycenter to the star, but in coordinates where the xy-plane is the sky plane and the z-axis points toward Earth.
        # Overwriting vec saves a lot of time vs. defining a new object
        mat_mul(rot_mtrx, vec, vec)

        # Angular position of the star relative to barycenter in milli-arcseconds.
        ang_pos_avg[l][0] = vec[0]*au_2_mas
        ang_pos_avg[l][1] = vec[1]*au_2_mas

        ################### Angular Velocities ########################

        # I only need Gaia velocities, not Hip velocities
        if l == 0:
            continue

        # Get velocity of the star (au/day)
        x_vel_avg, y_vel_avg = vel_avg(a_star, mean_motion, e, 
                                       E1, E2, start_time, end_time)
        
        # vec is overwritten again to store the stellar velocity instead of position.
        # Since we're using the E_anom for the planet, the star is moving in the opposite direction (neg. sign)
        vec[0] = -x_vel_avg
        vec[1] = -y_vel_avg
        vec[2] = 0

        # vec takes its final definition as the rotated stellar velocity.
        mat_mul(rot_mtrx, vec, vec)


        # mu_avg is a 2x2 array. The top row stays empty because we skip Hip. (The l==1 case is never executed
        # because we skip it in the if statement above.)
        # The bottom row is Gaia prop. motion.
        # The proper motion of the star due to the planet's orbit is in milli-arcseconds per year.
        # Since period is in days and a_star_units in au, velocities are in au/day.
        mu_avg[l][0] = vec[0]*aud_2_masyr
        mu_avg[l][1] = vec[1]*aud_2_masyr

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

    a (au): semi-major axis of the ellipse. Note that for an orbiting body,
            a must be the semi-major axis of that body's orbital path,
            not a_total of the full two-body orbit.
    n (1/days): 2pi/per
    t1, t2 (days): beginning and ending time to calculate average

    returns: Average x and y positions (au)
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

    a (au): semi-major axis of the ellipse. Note that for an orbiting body,
            a must be the semi-major axis of that body's orbital path,
            not a_total of the full two-body orbit.
    n (1/days): 2pi/per
    t1, t2 (days): beginning and ending time to calculate average

    returns: Average x and y velocities (au/day)
    """

    cdef double x_term_1, x_term_2, x_integral, x_avg,\
                y_term_1, y_term_2, y_integral, y_avg


    x_term_1 = a*cos(E1)
    x_term_2 = a*cos(E2)

    x_integral = x_term_2 - x_term_1

    x_avg = 1/(t2-t1) * x_integral


    y_term_1 = a*sqrt(1-e**2)*sin(E1)
    y_term_2 = a*sqrt(1-e**2)*sin(E2)

    y_integral = y_term_2 - y_term_1

    y_avg = 1/(t2-t1) * y_integral


    return x_avg, y_avg

#@profile
cdef void rot_matrix(double i, double om, double Om, double [:,:] rot_mtrx):
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


#cdef void mat_mul(double [:,:] mat, double [:] in_vec, double [:] out_vec):
#    """
#    This is written specifically to matrix multiply rot_matrix (3x3) with
#    r_unit_vec (3x1) and later v_vec_star (3x1) in astro_post_dense_loop.
#    """
#
#    cdef int i, k
#
#    for i in range(3):
#        out_vec[i] = 0
#        for k in range(3):
#            out_vec[i] += mat[i][k]*in_vec[k]
            

#THIS version of the function will let me input the same vector as in_vec and out_vec.
#I want to do this because it will allow me to allocate fewer arrays in the dmu function,
#hopefully giving significant speedups.
#
cpdef void mat_mul(double [:,:] mat, double [:] in_vec, double [:] out_vec):
    """
    This is written specifically to matrix multiply rot_matrix (3x3) with
    vec (3x1) in the dmu function. Saves time by receiving its "output" 
    (out_vec) as an argument and modifying it in place without returning 
    anything. I made this function even more idiosyncratic by removing
    the for-loop over vector elements. This lets me pass the same vector
    as both the input and the output, which saves time by only allocating
    one array.
    """

    cdef double m00, m01, m02,\
                m10, m11, m12,\
                m20, m21, m22,\
                iv0, iv1, iv2,\
                ov0, ov1, ov2
    
    m00 = mat[0][0]
    m01 = mat[0][1]
    m02 = mat[0][2]
    m10 = mat[1][0]
    m11 = mat[1][1]
    m12 = mat[1][2]
    m20 = mat[2][0]
    m21 = mat[2][1]
    m22 = mat[2][2]
    
    iv0 = in_vec[0]
    iv1 = in_vec[1]
    iv2 = in_vec[2]
    
    ov0 = m00*iv0 + m01*iv1 + m02*iv2
    ov1 = m10*iv0 + m11*iv1 + m12*iv2
    ov2 = m20*iv0 + m21*iv1 + m22*iv2
    
    out_vec[0] = ov0
    out_vec[1] = ov1
    out_vec[2] = ov2
            
             
            