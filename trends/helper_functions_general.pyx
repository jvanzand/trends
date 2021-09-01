# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
## cython: binding=True

import numpy as np
cimport numpy as np
import scipy as sp

import cython
from libc.math cimport sin, cos, tan, atan, sqrt

cdef float pi, two_pi, math_e, G, M_sun, M_jup, au, pc_in_cm

pi = 3.141592653589793
two_pi = 6.283185307179586
math_e = 2.718281828459045
G = 6.674299999999999e-08
M_sun = 1.988409870698051e+33
M_jup = 1.8981245973360504e+30
au = 14959787070000.0
pc_in_cm = 3.086e18


cpdef P(double a, double Mp, double Ms):
    """
    Uses Kepler's third law to find the period of a planet (in days) given its
    semimajor axis and the total mass of the system.

    a (au): semi-major axis
    Mp (M_Jup): companion mass
    Ms (M_sun): stellar mass
    """
    
    cdef double Mp_g, Ms_g, sec_2_days, P_days

    Mp_g = Mp*M_jup
    Ms_g = Ms*M_sun

    sec_2_days = 1./(24*3600) # Note the 1.; with 1, the result would be 0

    P_days = sqrt((2*pi)**2*(a*au)**3/(G*(Mp_g + Ms_g))) * sec_2_days

    return P_days
    

def contour_levels(prob_array, sig_list, t_num = 1e3):
    """
    Contour drawing method taken from 
    https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
    This function takes a 2-D array of probabilities and returns a 1-D array 
    of the probability values corresponding to 1-sigma and 2-sigma contours. 
    In this case, the 1-sigma contour encloses 68% of the total probability. 
    The array is expected to be normalized. sig_list is a list containing 
    any combination of the integers 1, 2, or 3 to indicate desired contours. 
    For example, [1,3] will return the 1 and 3 sigma contours.
    This function uses scipy.interpolate.interp1d.
    """


    # An array of probabilites from 0 to prob_max in rate_array
    t = np.linspace(0, np.array(prob_array).max(), int(t_num))

    # (prob_array >= t[:, None, None]) is a 3D array of shape (array_num, array_num, t_num). Each (array_num, array_num) layer is a 2D array of bool values indicating which values are greater than the value of the given t step.
    # Multiplying this 3D array of bools by prob_array replaces the bools with the array value if the bool is T and 0 if the bool is F.
    # Finally, sum along the array_num axes to get a single list of values, each with the total summed probability in its array.

    # integral is a 1D array of floats. The ith float is the sum of all probabilities in prob_array greater than the ith probability in t

    integral = ((prob_array >= t[:, None, None])*prob_array).sum(axis=(1,2))

    # Now create a function that takes integral as the x (not the y) and then returns the corresponding prob value from the t array. Interpolating between integral values allows me to choose any enclosed total prob. value (ie, integral value) and get the corresponding prob. value to use as my contour.
    f = sp.interpolate.interp1d(integral, t)

    contour_list = []
    prob_list = [0.68, 0.95, 0.997]

    for i in sig_list:
        contour_list.append(prob_list[i-1])

    # The plt.contourf function requires at least 2 levels. So if we want just one level, include a tiny contour that encompasses a small fraction of the total probability.
    if len(sig_list) == 1:
        contour_list.append(contour_list[0]-1e-4)
        # contour_list.append(1e-3)

    # Make sure list is in descending order
    t_contours = f(np.array(sorted(contour_list, reverse=True)))

    return t_contours

def contour_levels_1D(prob_list, sig_list, t_num = 1e3):
    """
    Same as contour_levels, but adapted for 1D arrays. 
    Hopefully I can condense these into 1 in the future.
    """


    # An array of probabilites from 0 to prob_max in rate_array
    t = np.linspace(0, prob_list.max(), int(t_num))

    # integral is a 1D array of floats. The ith float is the sum of all probabilities in prob_array greater than the ith probability in t

    integral = ((prob_list >= t[:, None])*prob_list).sum(axis=(1))

    # Now create a function that takes integral as the x (not the y) and then returns the corresponding prob value from the t array. Interpolating between integral values allows me to choose any enclosed total prob. value (ie, integral value) and get the corresponding prob. value to use as my contour.
    f = sp.interpolate.interp1d(integral, t)

    contour_list = []
    prob_list = [0.68, 0.95, 0.997]

    for i in sig_list:
        contour_list.append(prob_list[i-1])

    # The plt.contourf function requires at least 2 levels. So if we want just one level, include a tiny contour that encompasses a small fraction of the total probability. In this case, the contour we want will be at the 0th index.
    if len(sig_list) == 1:
        contour_list.append(contour_list[0]-1e-4)

    # Make sure the list of integrals is in descending order (eg, 99.7%, 95%, 68%). This will make the list of probabilities be in ascending order (eg, 0.05, 0.01, 0.007). These correspond to descending sigma levels (3, 2, 1).
    t_contours = f(np.array(sorted(contour_list, reverse=True)))


    return t_contours

def bounds_1D(prob_array, value_spaces, interp_num = 1e4):
    """
    Given a 2D probability array, this function collapses the array 
    along each axis to find the 68% confidence interval.

    value_spaces represents the parameter intervals covered by the 
    array along each axis. It is expected in the form 
    [(min_value1, max_value1), (min_value2, max_value2)], where 1 
    and 2 refer to the 0th and 1st axes. Note that the limits MUST 
    be in this order: if the array has shape (x_num, y_num), then 
    value_spaces must be [x_lims, y_lims].
    """
    bounds_list = []
    for i in range(2):

        array_1D = prob_array.sum(axis=i)
        grid_num = len(array_1D)


        # This gives only the 2-sigma, so that we get the 2-sigma limits at the end
        sig2 = contour_levels_1D(array_1D, [2])[0]

        # Interpolate between the points to get a finer spacing of points. This allows for more precise parameter estimation.
        func = sp.interpolate.interp1d(range(grid_num), array_1D)

        # Array over the same interval, but spaced (probably) more finely
        fine_array = np.linspace(0, grid_num-1, int(interp_num))

        # This is analogous to the original array_1D, but finer
        interp_vals = func(fine_array)

        #import matplotlib.pyplot as plt

        #plt.plot(range(len(fine_array)), interp_vals)
        #plt.show()


        # This is a shaky step. I'm just looking for places where the function value is really close to the probability corresponding to 2-sigma. But from what I can tell, this will fall apart for multimodal distributions, and maybe in other cases too. I use the 'take' method to pick out the first and last indices.


        inds_2sig = np.where(abs(interp_vals - sig2) < 1e-2*sig2)[0].take((0,-1))

        # value_bounds is a tuple of actual values, not indices
        value_bounds = index2value(inds_2sig, (0, interp_num-1), 
                                                value_spaces[::-1][i])

        bounds_list.append(value_bounds)

    return bounds_list


def value2index(value, index_space, value_space):
    """
    The inverse of index2value: take a value on a
    log scale and convert it to an index. index_space
    and value_space are expected as tuples of the form
    (min_value, max_value).
    """

    min_index, max_index = index_space[0],  index_space[1]
    min_value, max_value = value_space[0], value_space[1]

    index_range = max_index - min_index
    log_value_range = np.log10(max_value) - np.log10(min_value)

    value_arr = np.array(value)

    index = (np.log10(value_arr)-np.log10(min_value))\
                                    *(index_range/log_value_range) + min_index

    return index

def index2value(index, index_space, value_space):
    """
    The axis values for a plotted array are just the array indices.
    I want to convert these to Msini and a values, and on a log
    scale. This function takes a single index from a linear index range,
    and converts it to a parameter value in log space. index_space and
    value_space are expected as tuples of the form (min_value, max_value).
    index is in the range of index_space.
    """
    index = np.array(index)

    min_index, max_index = index_space[0],  index_space[1]
    min_value, max_value = value_space[0], value_space[1]

    index_range = max_index - min_index
    log_value_range = np.log10(max_value) - np.log10(min_value)

    # Convert from a linear space of indices to a linear space of log(values).
    log_value = (index-min_index)*(log_value_range/index_range) + np.log10(min_value)

    value = np.around(10**(log_value), 2) # Round to 2 decimal places

    return value

def period_lines(m, per, m_star):
    """
    Function to draw lines of constant period on the final plot.
    Rearranges Kepler's 3rd law to find how semi-major axis a 
    varies with period, companion mass, and stellar mass.

    Intended usage: Calculate an array of a values for a fixed per
                and m_star and an array of companion masses.
            
    Arguments:
            m (list of floats): companion masses (M_J)
            per (float): companion orbital period (days)
            m_star (float): stellar mass (M_sun)

    Returns:
            a (list of floats): Semi-major axis values (au)
    """
    m_grams = m*M_jup
    per_sec = per*24*3600
    m_star_grams = m_star*M_sun

    a_cm = ((per_sec/two_pi)**2*G*(m_grams+m_star_grams))**(0.3333333333)


    return a_cm / au
    
    