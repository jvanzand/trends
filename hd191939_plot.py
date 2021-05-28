# Get an import error for _kepler when I try to use these lines. Check out later
# import pyximport
# pyximport.install()

import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import trends.helper_functions as hlp


from trends.constants import *

import trends.giant_class as gc


# @profile
def main(save_data=False): 
    grid_num = 30
    plot_num = 30
    tick_num = 6
    
    a_lim = (1.9, 5e1)
    m_lim = (1.5, 2e2)
    
    linewidth = 4
    tick_size = 30
    label_size = 36
    title_size = 38
    region_label_size = 40
    restricted_region_label_size = 30

    

    a_list = np.logspace(np.log10(a_lim[0]), np.log10(a_lim[1]), plot_num)
    m_list = np.logspace(np.log10(m_lim[0]), np.log10(m_lim[1]), plot_num)
    
    ###########################################################
    min_index_m = hlp.value2index(min_m, (0, plot_num-1), m_lim)
    min_index_a = hlp.value2index(min_a, (0, plot_num-1), a_lim)


    prior_grid = np.zeros((plot_num, plot_num))
    prior_grid[min_index_m:, min_index_a:] = 1
    ###########################################################
    
    # HD191939 values: [Name, M_star, gdot, gdot_err, gddot, gddot_err, parallax, pm_anom_data, pm_anom_data_err]
    hd191939 = ['HD191939', 0.807, 0.114, 0.006, -6e-5, 1.9e-5, 18.62, pm_anom_data, pm_anom_data_err]
    my_planet = gc.Giant(*hd191939)
    my_planet.make_arrays(a_lim = a_lim, m_lim = m_lim, grid_num = grid_num, num_points = int(1e4), plot_num = plot_num)

    print('made the arrays')
    
    post_rv = my_planet.rv_post()
    post_astro = my_planet.astro_post()
    post_tot = my_planet.rv_astro_post()

    
    post_rv_plot    = my_planet.rv_plot_array
    post_astro_plot = my_planet.astro_plot_array
    post_tot_plot   = my_planet.tot_plot_array

    post_rv_plot    = (post_rv_plot*prior_grid)/((post_rv_plot*prior_grid).sum())
    post_astro_plot = (post_astro_plot*prior_grid)/((post_astro_plot*prior_grid).sum())
    post_tot_plot   = (post_tot_plot*prior_grid)/((post_tot_plot*prior_grid).sum())
    
    t_contours_astro = hlp.contour_levels(post_astro_plot, [1,2])
    t_contours_rv = hlp.contour_levels(post_rv_plot, [1,2])
    t_contours_tot = hlp.contour_levels(post_tot_plot, [1,2])
    
    if save_data:
        
        np.save('save_data/a_list', my_planet.a_list)
        np.save('save_data/m_list', my_planet.m_list)
        np.save('save_data/e_list', my_planet.e_list)
        np.save('save_data/i_list', my_planet.i_list)
        np.save('save_data/om_list', my_planet.om_list)
        np.save('save_data/M_anom_list', my_planet.M_anom_list)
    
        np.save('save_data/chi_sq_list_rv', my_planet.chi_sq_list_rv)
        np.save('save_data/chi_sq_list_astro', my_planet.chi_sq_list_astro)
        np.save('save_data/prob_list_rv', my_planet.prob_list_rv)
        np.save('save_data/prob_list_astro', my_planet.prob_list_astro)
    
    
        np.save('save_data/post_rv', post_rv)
        np.save('save_data/post_astro', post_astro)
        np.save('save_data/post_tot', post_tot)


    bounds = hlp.bounds_1D(post_tot, [m_lim, a_lim], interp_num = 1e4)
    print('a_lim, m_lim = ', bounds[0], bounds[1])
    ##################################################
    fig, ax = plt.subplots(figsize=(12,12))
    
    post_astro_cont = ax.contourf(post_astro_plot, t_contours_astro, cmap='Blues', extend='max', alpha=0.5)
    post_rv_cont = ax.contourf(post_rv_plot, t_contours_rv, cmap='Greens', extend='max', alpha=0.5)
    post_tot_cont = ax.contourf(post_tot_plot, t_contours_tot, cmap='Reds', extend='max', alpha=0.75)
    
    mass_rect = ptch.Rectangle((0, 0), plot_num-1, min_index_m, color='gray', alpha=1.0)
    a_rect = ptch.Rectangle((0, 0), min_index_a, plot_num-1, color='gray', alpha=1.0)

    ax.add_patch(mass_rect)
    ax.add_patch(a_rect)
    
    ########################################
    
    
    plt.text((19/32)*plot_num, (7/8)*plot_num, 'RV', size=region_label_size, rotation=50)
    plt.text((5/8)*plot_num, (1/4)*plot_num, 'Astrometry', size=region_label_size)
    
    plt.text((1/3)*plot_num, (1/3)*(min_index_m-1), 'Masses disallowed by RVs', size=restricted_region_label_size)
    plt.text((1/3)*(min_index_a-1), (1/3)*plot_num, 'Visible curvature', size=restricted_region_label_size, rotation=90)

    tick_array = np.linspace(0, plot_num-1, tick_num).astype(int)
    plt.xticks(tick_array, [np.round(a_list[i], 1) for i in tick_array], size=tick_size)
    plt.yticks(tick_array, [np.round(m_list[i], 1) for i in tick_array ], size=tick_size)
    
    ax.set_xlabel('Semi-major Axis (au)', size=label_size)
    ax.set_ylabel(r'$M_p$ ($M_{Jup}$)', size=label_size)
    # ax.set_title('Astrometry', size=title_size)
    
    
    fig.tight_layout()
    fig.savefig('5thCompConstraints_RV_astr.png')
    # plt.show()
    
    return




if __name__ == "__main__":
    main()








            
            
            
