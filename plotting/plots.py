from matplotlib import pyplot as plt
from typing import List
import numpy as np
from model.model import Gadget


def generic_timeseries_plot(gadget_list: List[Gadget],
                            state_to_plot: str,
                            log_xscale: bool = False,
                            log_yscale: bool = False):
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))

    for g in gadget_list:
        
        ax.plot(g.output_df.sim_time_us, g.output_df[state_to_plot], 
                label = 'ID: ' + str(g.id))
        
    ax.set_xlabel('Time (us)')
    ax.set_ylabel(state_to_plot)
    ax.set_title(state_to_plot)

    if log_yscale:
        ax.set_yscale('log')

    if log_xscale:
        ax.set_xscale('log')

    ax.legend()
    fig.tight_layout()


def plot_fission_rate_and_energy(gadget_list: List[Gadget], 
                                 log_xscale: bool = False,
                                 log_yscale: bool = False):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 8))
    
    for g in gadget_list:

        ax[0].plot(g.output_df.sim_time_us, g.output_df.fission_rate_pers)
        ax[1].plot(g.output_df.sim_time_us, g.output_df.total_energy_released_kt,
                   label = 'ID: ' + str(g.id))
       
    ax[1].set_xlabel('Time (us)')
    ax[0].set_ylabel('Fission Rate (1/s)')
    ax[1].set_ylabel('Energy Released (kt)')

    if log_yscale:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

    if log_xscale:
        ax[1].set_xscale('log')

    fig.suptitle('Fission Rate and Total Energy Released')
    ax[1].legend()
    fig.tight_layout()


def plot_conc_and_surf_flux(gadget: Gadget):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 8))

    ax[0].plot(gadget.output_df.sim_time_us, gadget.output_df.surf_conc_perm3, 
               color = 'black', label = 'Surface')
    ax[0].plot(gadget.output_df.sim_time_us, gadget.output_df['r-over-2_conc_perm3'],
               color = 'black', linestyle = '-.', label = 'R/2')
    ax[0].plot(gadget.output_df.sim_time_us, gadget.output_df.center_conc_perm3,
               color = 'black', linestyle = '--', label = 'Center')

    ax[1].plot(gadget.output_df.sim_time_us, gadget.output_df.surface_flux_perm2s,
               color = 'black')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(1E-16)

    ax[1].set_xscale('log')

    ax[1].set_xlabel('Time (us)')
    ax[0].set_ylabel('Neutron Concentration (1/m3)')
    ax[1].set_ylabel('Neutron Surface Flux (1/(m2 * s))')

    ax[0].legend()
    fig.suptitle(f'Neutron Concentration and Surface Flux \n ID: {gadget.id}')
    fig.tight_layout()


def plot_radial_concentration(gadget: Gadget, num_snapshots=8):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    steps_to_plot = np.linspace(0, gadget.out.neutron_conc_matrix.shape[0] - 1, 
                                num_snapshots, dtype=int)
    for s in steps_to_plot:
        ax.plot(gadget.out.radius_points_cm, gadget.out.neutron_conc_matrix[int(s), :], 
                label = round(s * gadget.time_step_s * 1E9, ndigits=3),
                marker='')

    ax.set_ylim(1E-12)    
    ax.set_yscale('log')
    #ax.set_ylim(0)

    ax.legend(title='Time (ns)')
    ax.set_ylabel('Neutron Concentration (1/m3)')
    ax.set_xlabel('Distance from center (cm)')
    ax.grid(visible=True, which='major')
    ax.set_xlim(0, gadget.initial_radius_cm)
    #ax.ticklabel_format(useOffset=False)
    ax.set_title(f'Neutron Concentrations At Timesteps \n ID: {gadget.id}')


def plot_neutron_counts(gadget: Gadget):

    fig, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 4))
    ax.plot(gadget.output_df.sim_time_us, gadget.output_df.neutrons_in_sphere, 
            color = 'black', label = 'In sphere')

    ax.plot(gadget.output_df.sim_time_us, gadget.output_df.cumulative_neutrons_left,
            color = 'black', linestyle = '--', label = 'Left surface')

    ax.plot(gadget.output_df.sim_time_us, gadget.output_df.cumulative_neutrons, color = 'black',
            linestyle = '-.', label = 'Total')

    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Neutron Count')
    ax.legend()
    ax.set_title(f'Total Neutron Counts \n ID: {gadget.id}')

    ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Neutron Count')
    ax.legend()


def plot_radius_and_pressure(gadget: Gadget):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 8))
    ax[0].plot(gadget.output_df.sim_time_us, gadget.output_df.radius_cm,
               color = 'black')

    ax[1].plot(gadget.output_df.sim_time_us, gadget.output_df.pressure_pa,
               color = 'black')

    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    #ax[1].set_ylim(1E-16)

    #ax[1].set_xscale('log')

    ax[1].set_xlabel('Time (us)')
    ax[0].set_ylabel('Radius (cm)')
    ax[1].set_ylabel('Pressure (Pa)')

    fig.suptitle(f'Radius and Pressure as a Function of Time \n ID: {gadget.id}')
    fig.tight_layout()