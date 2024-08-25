from matplotlib import pyplot as plt
from typing import List
from model import Gadget
import numpy as np


def plot_fissions_and_energy(gadget_list: List[Gadget]):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 8))
    
    for g in gadget_list:

        # Number of fissions
        ax[0].plot(g.sim_time_array_us, g.list_total_number_of_fissions)
        
        # Total energy (kt)
        ax[1].plot(g.sim_time_array_us, g.array_total_energy_released_kt,
                label = 'ID: ' + str(g.id) + ', ' + str(g.initial_radius_cm) + ' cm')

    ax[1].set_xlabel('Time (us)')
    ax[0].set_ylabel('Fission Count')
    ax[1].set_ylabel('Energy Released (kt)')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    fig.suptitle('Number of Fissions and Energy Released')
    fig.legend()
    fig.tight_layout()


def plot_conc_and_surf_flux(gadget: Gadget):

    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 8))
    ax[0].plot(gadget.sim_time_array_us, gadget.neutron_conc_matrix[:, -1], 
               color = 'black', label = 'Surface')
    ax[0].plot(gadget.sim_time_array_us, gadget.neutron_conc_matrix[:, 0], 
               color = 'black', linestyle = '--', label = 'Center')

    ax[1].plot(gadget.sim_time_array_us, gadget.list_surface_flux_perm2s, 
               color = 'black')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(1E-16)

    ax[1].set_xscale('log')

    ax[1].set_xlabel('Time (us)')
    ax[0].set_ylabel('Neutron Concentration (1/m3)')
    ax[1].set_ylabel('Neutron Surface Flux (1/(m2 * s))')

    ax[0].legend()
    fig.suptitle('Neutron Concentration and Surface Flux')
    fig.tight_layout()


def plot_radial_concentration(gadget: Gadget):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    snapshots = 8

    steps_to_plot = np.linspace(0, gadget.neutron_conc_matrix.shape[0] - 1, 
                                snapshots, dtype=int)
    for s in steps_to_plot:
        ax.plot(gadget.radius_points, gadget.neutron_conc_matrix[int(s), :], 
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
    ax.set_title('Neutron Concentrations At Timesteps')


def plot_neutron_counts(gadget: Gadget):

    fig, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 4))
    ax.plot(gadget.sim_time_array_us, gadget.list_total_neutrons_in_sphere, 
            color = 'black', label = 'In sphere')

    ax.plot(gadget.sim_time_array_us, gadget.list_cumulative_neutrons_left_surface, 
            color = 'black', linestyle = '--', label = 'Left surface')

    ax.plot(gadget.sim_time_array_us, gadget.list_total_neutrons, color = 'black',
            linestyle = '-.', label = 'Total')

    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Neutron Count')
    ax.legend()
    ax.set_title('Total Neutron Counts')

    ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Neutron Count')
    ax.legend()