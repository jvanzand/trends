import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import helper_functions as hlp

from constants import *

import giant_class as gc
     

if __name__ == "__main__":
    
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
    
    hd191939 = ['HD191939', 0.807, 0.114, 0.006, -6e-5, 1.9e-5, 18.62, pm_anom_data, pm_anom_data_err]
    my_planet = gc.Giant(*hd191939)
    my_planet.make_arrays(a_lim = a_lim, m_lim = m_lim, grid_num = grid_num, num_points = int(1e6), plot_num = plot_num)

    print('made the arrays')
    
    post_rv = my_planet.rv_post()
    post_astro = my_planet.astro_post()
    post_tot = my_planet.rv_astro_post()
    # post_tot_plot = np.load('post_tot_trimmed.npy')
    # post_tot_plot   = (post_tot_plot*prior_grid)/((post_tot_plot*prior_grid).sum())
    # print(hlp.bounds_1D(post_tot_plot, [m_lim, a_lim], interp_num = 1e4))
    # fdfd
    
    post_rv_plot    = my_planet.rv_plot_array
    post_astro_plot = my_planet.astro_plot_array
    post_tot_plot   = my_planet.tot_plot_array

    post_rv_plot    = (post_rv_plot*prior_grid)/((post_rv_plot*prior_grid).sum())
    post_astro_plot = (post_astro_plot*prior_grid)/((post_astro_plot*prior_grid).sum())
    post_tot_plot   = (post_tot_plot*prior_grid)/((post_tot_plot*prior_grid).sum())
    
    t_contours_astro = hlp.contour_levels(post_astro_plot, [3])
    t_contours_rv = hlp.contour_levels(post_rv_plot, [1,2])
    t_contours_tot = hlp.contour_levels(post_tot_plot, [1,2])
    
    
    # np.save('post_tot_trimmed', post_tot_plot)
    print(hlp.bounds_1D(post_tot_plot, [m_lim, a_lim], interp_num = 1e4))
    ##################################################
    fig, ax = plt.subplots(figsize=(12,12))
    
    post_astro_cont = ax.contourf(post_astro_plot, t_contours_astro, cmap='Blues', extend='max', alpha=0.5)
    post_rv_cont = ax.contourf(post_rv_plot, t_contours_rv, cmap='Greens', extend='max', alpha=0.5)
    post_tot_cont = ax.contourf(post_tot_plot, t_contours_tot, cmap='Reds', extend='max', alpha=0.75)
    
    # # seg_list_1sig = post_tot_plot.allsegs[1][0]
    # # List of segments corresponding to 1-sigma contour (actually to the lowest sigma value given in t_contours_tot, usually 1)
    # # Index explanation: -1 to get the tightest contour. Then if that contour is broken, I need to get the farthest-left point of the farthest-left chunk, and the farthest-right point of the farthest-right chunk. My assumption is that contour lists its contours as [left-most, middle, right-most], or even the reverse of that, they just need to not be jumbled. Under this assumption, I take the 0th block for the min value, and for the max I take the -1th block. Finally I transpose to go from [(x,y), (x,y), ...] to [(x,x, ...), (y,y, ...)], and take x for the a index and y for the mass.
    #
    # min_a_index_1sig = np.min(post_tot_plot.allsegs[-1][0].T[0])
    # max_a_index_1sig = np.max(post_tot_plot.allsegs[-1][-1].T[0])
    #
    # min_m_index_1sig = np.min(post_tot_plot.allsegs[-1][0].T[1])
    # max_m_index_1sig = np.max(post_tot_plot.allsegs[-1][-1].T[1])
    #
    # # seg_list_1sig = post_tot_plot.allsegs[-1][0]
    # # min_a_index_1sig = np.min(seg_list_1sig.transpose()[0])
    # # max_a_index_1sig = np.max(seg_list_1sig.transpose()[0])
    #
    # # min_m_index_1sig = np.min(seg_list_1sig.transpose()[1])
    # # max_m_index_1sig = np.max(seg_list_1sig.transpose()[1])
    #
    # min_a_1sig = hlp.index2value(min_a_index_1sig, (0, grid_num-1), a_lim)
    # max_a_1sig = hlp.index2value(max_a_index_1sig, (0, grid_num-1), a_lim)
    # min_m_1sig = hlp.index2value(min_m_index_1sig, (0, grid_num-1), m_lim)
    # max_m_1sig = hlp.index2value(max_m_index_1sig, (0, grid_num-1), m_lim)
    #
    # print(min_a_1sig, max_a_1sig)
    # print(min_m_1sig, max_m_1sig)
    
    mass_rect = ptch.Rectangle((0, 0), plot_num-1, min_index_m, color='gray', alpha=1.0)
    a_rect = ptch.Rectangle((0, 0), min_index_a, plot_num-1, color='gray', alpha=1.0)

    ax.add_patch(mass_rect)
    ax.add_patch(a_rect)
    
    ########################################
    
    
    # plt.text((19/32)*plot_num, (7/8)*plot_num, 'RV', size=region_label_size, rotation=50)
    plt.text((1/2)*plot_num, (1/8)*plot_num, 'Astrometry', size=region_label_size)
    
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
    plt.show()












            
            
            