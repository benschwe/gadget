from dataclasses import dataclass
from numpy import array
from pandas import DataFrame

@dataclass
class States:

    sim_time_s: float = 0.0

    # Volume, density, diffusivity
    dr_m: float = None
    radius_m: float = None
    surface_area_m2: float = None
    volume_m3: float = None
    density_kgperm3: float = None

    # Neutron concentration
    neutron_conc_array: array = None

    # Neutron count and concentration change per time step
    neutrons_generated_per_step: array = None
    neutron_conc_change_array: array = None

    # Nuclear parameters
    mean_free_path_fission_m: float = None
    mean_free_path_elastic_m: float = None
    mean_free_path_transport_m: float = None
    neutron_diffusivity_m2pers: float = None
    nuclear_number_density_perm3: float = None
    
    # Everything else
    neutron_gen_rate_pers: float = 0
    surface_flux_perm2s: float = 0
    neutrons_left: float = 0
    cumulative_neutrons_left: float = 0
    neutrons_in_sphere: float = None
    cumulative_neutrons: float = 0
    cumulative_number_of_fissions: float = 0
    fission_rate_pers: float = 0
    total_energy_released_j: float = 0
    heat_gen_w: float = 0
    pressure_pa: float = 101000
    expansion_vel_mpers: float = 0