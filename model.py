import numpy as np
import math
from params import PhysicalParamsU235, PhysicalParamsPu239
from data import States
from constants import *
from pandas import DataFrame

class Gadget(PhysicalParamsU235):

    @property
    def number_active_nuclei(self) -> float:

        active_material_moles = (self.parameters.fraction_active_material * 
                                (self.mass_kg / 
                                (self.parameters.atomic_mass_kgpermol)))
        
        num_active_nuclei = active_material_moles * AVOGADRO

        return num_active_nuclei
    

    @property
    def maximum_possible_energy_kt(self) -> float:

        return self.number_active_nuclei * self.parameters.energy_per_fission_j * J_TO_KILOTON
    
    
    def get_nuclear_number_density_perm3(self,
                                          rho_kgperm3,
                                          ) -> float:
        '''Given a physical density (kg/m3), return
        the nuclear number density (active nuclei/m3)
        '''
        
        nuclear_number_density_per_m3 = ((rho_kgperm3 * AVOGADRO) / 
                                         (self.parameters.atomic_mass_kgpermol))
        
        return nuclear_number_density_per_m3
    

    def get_mass_from_radius_m(self,
                               radius_m,
                               density_kgperm3):
        
        volume_m3 = self.get_volume_m3(radius_m=radius_m)
        mass_kg = volume_m3 * density_kgperm3

        return mass_kg
        

    def get_radius_m(self,
                     mass_kg,
                     density_kgperm3) -> float:

        volume_m3 = mass_kg / density_kgperm3
        radius_m = (volume_m3 / ((4 / 3) * math.pi)) ** (1 / 3)
        
        return radius_m
    

    def get_volume_m3(self,
                       radius_m) -> float:
        
        '''Returns the volumes (m3) given a radius (m)
        '''
        
        volume_m3 = (4 / 3) * math.pi * radius_m ** 3

        return volume_m3
    

    def get_surface_area_m2(self,
                             radius_m) -> float:
        
        surface_area_m2 = 4 * math.pi * radius_m ** 2

        return surface_area_m2
    

    def get_density_kgperm3(self,
                             radius_m) -> float:
        
        '''Returns the density (kg/m3) given a radius.
        We always assume no mass loss.
        '''
        
        density_kgperm3 = (self.mass_kg /
                           self.get_volume_m3(radius_m=radius_m))
                           
        return density_kgperm3
    

    def get_mean_free_path_fission_m(self, 
                                      nuclear_number_density_perm3=None) -> float:
        
        return (1 / (nuclear_number_density_perm3 * 
                     self.parameters.cross_section_fission_m2))
    

    def get_mean_free_path_elastic_m(self,
                                      nuclear_number_density_perm3=None) -> float:
        
        return (1 / (nuclear_number_density_perm3 * 
                     self.parameters.cross_section_elastic_scattering_m2))
    

    def get_mean_free_path_transport_m(self,
                                        nuclear_number_density_perm3=None) -> float:
        
        return (1 / ((1 / self.get_mean_free_path_fission_m(nuclear_number_density_perm3=
                                                             nuclear_number_density_perm3)) +
                    (1 / self.get_mean_free_path_elastic_m(nuclear_number_density_perm3=
                                                            nuclear_number_density_perm3))))
    

    def get_neutron_diffusivity_m2pers(self,
                                        nuclear_number_density_perm3=None) -> float:
        
        return ((self.get_mean_free_path_transport_m(nuclear_number_density_perm3=
                                                      nuclear_number_density_perm3) * 
                                                      self.parameters.vel_neutron_mpers) / 3)
    

    def get_neutron_gen_rate_pers(self,
                                   mean_free_path_fission_m=None) -> float:
        
        return ((self.parameters.vel_neutron_mpers / mean_free_path_fission_m) * 
                (self.parameters.neutrons_per_fission - 1))
        
       
    def get_shell_volumes_m3(self,
                             dr_m) -> np.array:

        '''Returns an array of volumes (m3) for each shell of the gadget
        '''
        radii = np.array([dr_m * i for i in range(0, self.num_points_radial)])

        shell_volumes_m3 = np.array([(4 / 3) * math.pi * (radii[i + 1] ** 3) - 
                                         (4 / 3) * math.pi * (radii[i] ** 3) 
                                         for i in range(0, len(radii) - 1)])
        
        return shell_volumes_m3
    

    def get_elapsed_time_s(self):

        return self.states[-1].sim_time_s
        

    def make_matrix_coeffs(self,
                           mean_free_path_transport_m,
                           neutron_diffusivity_m2pers,
                           dr_m):
        '''This makes the matrix coeffs for each time step
        '''

        # Core Surface BC - "dN/dr = Z * N(r=R,t)", 
        # where Z = -(3/2) * (1/lambda_transport)
        z = -(3 / 2) * (1 / mean_free_path_transport_m) 
        
        A = (neutron_diffusivity_m2pers * self.time_step_s) / (dr_m ** 2)
        B = (neutron_diffusivity_m2pers * self.time_step_s) / (dr_m) # MUST DIVIDE BY RADIUS IN MATRIX
        C = (3 * neutron_diffusivity_m2pers * self.time_step_s) / (dr_m ** 2)
        D = 2 * dr_m * z

        return A, B, C, D

    
    def make_G_matrix(self,
                      A,
                      B,
                      C,
                      D,
                      dr_m) -> np.array:
        
        # ***** G Matrix: Evolution matrix *****
        G = np.zeros((self.num_points_radial, self.num_points_radial))

        # Main Diagonal
        for i in range(1, self.num_points_radial - 1):

            G[i, i] = 1 + 2 * A

        # Upper Diagonal
        for i in range(1, self.num_points_radial - 1):

            G[i, i + 1] = -A - B / (dr_m * i)

        # Lower Diagonal
        for i in range(1, self.num_points_radial - 1):

            G[i, i - 1] = -A + B / (dr_m * i)

        # First row
        G[0, 0] = 1 + 2 * C
        G[0, 1] = -2 * C

        # Bottom row
        G[-1, -2] = -2 * A
        G[-1, -1] = (-A * D + (B / (dr_m * self.num_points_radial)) * D + 1 + 2 * A)

        return G
    

    def run_sim_step(self) -> None:

        prev_state = self.states[-1]
        new_state = States()

        '''
        '''

        # ***** (1) CONCENTRATION *****
        # Get the array row of concentrations from the previous timestep
        neutron_conc_prev = prev_state.neutron_conc_array.copy()

        # Neutron generation: rate * time = neutrons / m3
        # We make this controllable for testing the diffusion code
        # Grab the latest neutron generation rate to use
        if self.neutron_multiplication_on:
            self.H = neutron_conc_prev * prev_state.neutron_gen_rate_pers * self.time_step_s
            new_state.neutron_conc_change_array = self.H
            new_state.neutrons_generated_per_step = (new_state.neutron_conc_change_array[0:-1] * 
                                                     self.get_shell_volumes_m3(dr_m = 
                                                     prev_state.dr_m)).sum()

        # Update A, B, C, and D coefficients
        A, B, C, D = self.make_matrix_coeffs(
            mean_free_path_transport_m = prev_state.mean_free_path_transport_m,
            neutron_diffusivity_m2pers = prev_state.neutron_diffusivity_m2pers,
            dr_m = prev_state.dr_m)
        
        # Make new G and Ginv matrices
        self.G = self.make_G_matrix(A = A,
                                    B = B,
                                    C = C,
                                    D = D,
                                    dr_m = prev_state.dr_m)
        
        self.Ginv = np.linalg.inv(self.G)
        
        # Calcuate new neutron concentration values and add it to the new state
        neutron_conc_new = np.dot(self.Ginv, np.add(neutron_conc_prev, self.H))
        new_state.neutron_conc_array = neutron_conc_new

        # ***** (2) NEUTRON ACCOUNTING *****
        # Surface flux
        # Estimate neutron concentration surface gradient 
        # neutrons per m3 * 1 / m = neutrons / m4
        # Notations here: self.conc_list[-1] is the most recent concentration array.
        #                 self.conc_list[-1][-2] is the second to last element 
        #                 in the most recent
        #                 concentration array
        surface_conc_gradient_perm4 = (new_state.neutron_conc_array[-2] - 
                                       new_state.neutron_conc_array[-1]) / prev_state.dr_m

        # Surface flux update - surf concentration gradient * last diffusivity entry
        surface_flux_perm2s = (surface_conc_gradient_perm4 * 
                               prev_state.neutron_diffusivity_m2pers)
        new_state.surface_flux_perm2s = surface_flux_perm2s
        
        # Estimate neutrons leaving surface in the time step
        # Note - surface area changes with radius
        neutrons_left = (surface_flux_perm2s * 
                         prev_state.surface_area_m2 *
                         self.time_step_s)
        new_state.neutrons_left = neutrons_left
        
        # Tally total neutrons that left the surface
        new_state.cumulative_neutrons_left = (prev_state.cumulative_neutrons_left + 
                                              neutrons_left)
        
        # Estimate neutrons in sphere at this timestep
        new_state.neutrons_in_sphere = (new_state.neutron_conc_array[0:-1] * 
                                        self.get_shell_volumes_m3(dr_m = prev_state.dr_m)).sum()
        
        # Total neutron count - in sphere and cumlative that left surface
        new_state.cumulative_neutrons = (new_state.neutrons_in_sphere +
                                         new_state.cumulative_neutrons_left)
        
        # ***** (3) FISSIONS, ENERGY, PRESSURE, RADIUS, VOL, AREA, DR_M, EXPANSION SPEED *****
        # Total number of fissions
        new_state.cumulative_number_of_fissions = (new_state.cumulative_neutrons / 
                                                   self.parameters.neutrons_per_fission)
        
        # Incremental fissions and fission rate (1/s)
        incremental_fissions = (new_state.cumulative_number_of_fissions - 
                                prev_state.cumulative_number_of_fissions)
        new_state.fission_rate_pers = incremental_fissions / self.time_step_s
        
        # Total energy released due to fission of active nuclei
        new_state.total_energy_released_j = (new_state.cumulative_number_of_fissions * 
                                             self.parameters.energy_per_fission_j)
        
        # Energy released in this timestep
        energy_released_j = (new_state.total_energy_released_j - 
                             prev_state.total_energy_released_j)
        
        # Power (W = J/s)
        new_state.heat_gen_w = energy_released_j / self.time_step_s
        
        # Update pressure at this timestep
        # Uses gamma = 1/3 following B.C. Reed
        # Pressure = (gamma * total energy) / volume
        new_state.pressure_pa = ((1 / 3) * new_state.total_energy_released_j / 
                                self.get_volume_m3(radius_m=prev_state.dr_m)) + self.states[0].pressure_pa
        

        # Update delta expansion velocity (m/s) and expansion vel list (m/s)
        delta_expansion_vel_mpers = (4 * math.pi * (prev_state.radius_m ** 2) * 
                                      (1 / 3) * new_state.total_energy_released_j / 
                                      (self.get_volume_m3(radius_m=prev_state.radius_m) * 
                                      (self.mass_kg + self.tamper_mass_kg))) * self.time_step_s
     
        new_state.expansion_vel_mpers = (prev_state.expansion_vel_mpers + 
                                         delta_expansion_vel_mpers)

        # Update radius (m)
        delta_r_m = new_state.expansion_vel_mpers * self.time_step_s
        new_state.radius_m = prev_state.radius_m + delta_r_m

        # Update delta radius (m)
        new_state.dr_m = new_state.radius_m / (self.num_points_radial - 1)
        
        # Update surface area (m2)
        new_state.surface_area_m2 = self.get_surface_area_m2(radius_m = 
                                                             new_state.radius_m)

        # Update volume (m3)
        new_state.volume_m3 = self.get_volume_m3(radius_m =
                                                 new_state.radius_m)

        # Update density (kg/m3)
        new_state.density_kgperm3 = self.mass_kg / new_state.volume_m3

        # ***** (4) NUCLEAR PARAMS *****
        # Update nuclear number density
        new_state.nuclear_number_density_perm3 = self.get_nuclear_number_density_perm3(
                                                      rho_kgperm3=new_state.density_kgperm3)

        # Update fission mean free path (m)
        new_state.mean_free_path_fission_m = self.get_mean_free_path_fission_m(
                                                  nuclear_number_density_perm3=
                                                  new_state.nuclear_number_density_perm3)

        # Update elastic scattering mean free path (m)
        new_state.mean_free_path_elastic_m = self.get_mean_free_path_elastic_m(
                                                  nuclear_number_density_perm3=
                                                  new_state.nuclear_number_density_perm3)

        # Update transport mean free path (m)
        new_state.mean_free_path_transport_m = self.get_mean_free_path_transport_m(
                                                    nuclear_number_density_perm3=
                                                    new_state.nuclear_number_density_perm3)
        
        # Update neutron diffusivity (m2/s)
        new_state.neutron_diffusivity_m2pers = self.get_neutron_diffusivity_m2pers(
                                                    nuclear_number_density_perm3=
                                                    new_state.nuclear_number_density_perm3)
                                                    
        
         # Neutron generation rate
        new_state.neutron_gen_rate_pers = self.get_neutron_gen_rate_pers(
                                               mean_free_path_fission_m=
                                               new_state.mean_free_path_fission_m)
        
        # ***** (5) SIM TIME *****
        new_state.sim_time_s = prev_state.sim_time_s + self.time_step_s
        
        # ***** (6) UPDATE STATE LIST *****
        self.states.append(new_state)
      
        # Increment the time steps elapsed
        self.num_sim_steps += 1


    def post_process(self):
        '''
        This creates useful simulation data for plotting
        '''

        self.output_df = DataFrame(s.__dict__ for s in self.states)
        self.output_df['sim_time_us'] = self.output_df['sim_time_s'] * 1E6
        self.output_df['total_energy_released_kt'] = self.output_df['total_energy_released_j'] * J_TO_KILOTON
        self.output_df['center_conc_perm3'] = [self.output_df.neutron_conc_array.iloc[i][0] for i in self.output_df.index]

        r_over_2_radius_index = int(self.num_points_radial / 2)
        self.output_df['r-over-2_conc_perm3'] = [self.output_df.neutron_conc_array.iloc[i][r_over_2_radius_index] for i in self.output_df.index]
        self.output_df['surf_conc_perm3'] = [self.output_df.neutron_conc_array.iloc[i][-1] for i in self.output_df.index]
        self.output_df['radius_cm'] = self.output_df.radius_m * M_TO_CM
    
     
    def __init__(self,
                id: int=0,
                material: str='U235',
                mass_kg : float=45.0,
                initial_radius_m: float=None,
                initial_density_multiplier=None,
                initial_neutron_conc_perm3: float=0,
                initial_neutron_burst_conc_perm3: float=50000,
                time_step_s: float=1E-10,
                num_points_radial: int=100,
                neutron_multiplication_on: bool=True,
                tamper_mass_kg=0,
                ) -> None:
        
        # ***** Static attributes - constant or initial values *****
        self.id = id
        self.material = material
        self.mass_kg = mass_kg
        self.initial_radius_m = initial_radius_m
        self.initial_density_multiplier = initial_density_multiplier
        self.initial_neutron_conc_perm3 = initial_neutron_conc_perm3
        self.initial_neutron_burst_conc_perm3 = initial_neutron_burst_conc_perm3
        self.time_step_s = time_step_s
        self.num_points_radial = num_points_radial
        self.neutron_multiplication_on = neutron_multiplication_on
        self.tamper_mass_kg = tamper_mass_kg
        self.initial_density_multiplier = initial_density_multiplier

        if self.material == 'U235':
            self.parameters = PhysicalParamsU235()
        elif self.material == 'Pu239':
            self.parameters = PhysicalParamsPu239()
        else:
            raise ValueError('Incorrect material specified')
        
        if self.mass_kg is not None and self.initial_radius_m is not None:
            raise ValueError('Both radius and mass specified. Please specify either mass OR radius')

        # ***** States (Variables) *****
        self.num_sim_steps = 0

        self.states = []
        initial_state = States()

        # Initialize density (kg/m3)
        # Allow for overriding the initial density by using an optional arg
        # "initial_density_multiplier"
        if self.initial_density_multiplier is None:
            initial_state.density_kgperm3 = self.parameters.density_rho_kgperm3
        else:
            initial_state.density_kgperm3 = (self.parameters.density_rho_kgperm3 *
                                             self.initial_density_multiplier)

        # Radius (m)
        if self.initial_radius_m is not None:
            initial_state.radius_m = self.initial_radius_m
            self.mass_kg = self.get_mass_from_radius_m(radius_m=initial_state.radius_m,
                                                       density_kgperm3=initial_state.density_kgperm3)
            print(f'Establishing mass of {self.mass_kg:.2f} kg from radius of {(M_TO_CM * self.initial_radius_m):.2f} cm')

        else:
            initial_state.radius_m = self.get_radius_m(mass_kg=self.mass_kg,
                                                       density_kgperm3=initial_state.density_kgperm3)
            print(f'Establishing radius of {(M_TO_CM * initial_state.radius_m):.2f} cm from mass of {self.mass_kg} kg')

        # Surface area (m2)
        initial_state.surface_area_m2 = self.get_surface_area_m2(radius_m=
                                                                 initial_state.radius_m)

        # Initialize volume (m3)
        initial_state.volume_m3 = self.get_volume_m3(radius_m=
                                                     initial_state.radius_m)
        
        # Initialize the starting delta radius
        initial_state.dr_m = initial_state.radius_m / (self.num_points_radial - 1)

        # Initialize nuclear number density (1/m3)
        initial_state.nuclear_number_density_perm3 = self.get_nuclear_number_density_perm3(
                                                          rho_kgperm3=initial_state.density_kgperm3)
       
        # Initialize fission mean free path (m)
        initial_state.mean_free_path_fission_m = self.get_mean_free_path_fission_m(
                                                      initial_state.nuclear_number_density_perm3)

        # Initialize elastic scattering mean free path (m)
        initial_state.mean_free_path_elastic_m = self.get_mean_free_path_elastic_m(
                                                      initial_state.nuclear_number_density_perm3)
       

        # Initialize transport mean free path (m)
        initial_state.mean_free_path_transport_m = self.get_mean_free_path_transport_m(
                                                        initial_state.nuclear_number_density_perm3)
        
        # Initialize neutron diffusivity (m2/s)
        initial_state.neutron_diffusivity_m2pers = self.get_neutron_diffusivity_m2pers(
                                                        initial_state.nuclear_number_density_perm3)
        
         # Neutron generation
        initial_state.neutron_gen_rate_pers = self.get_neutron_gen_rate_pers(
                                                   mean_free_path_fission_m=
                                                   initial_state.mean_free_path_fission_m)
        
        # Neutron volumetric concentration matrix
        init_neutron_conc_array = (np.zeros(self.num_points_radial) + 
                                   self.initial_neutron_conc_perm3)
        
        # Initial neutron burst is in the first shell
        # TODO: Be able to specify a radius where the
        # burst happens, instead of fixed radii indicies
        init_neutron_conc_array[0:2] = self.initial_neutron_burst_conc_perm3
        initial_state.neutron_conc_array = init_neutron_conc_array

        # Ensure we start the neutron conc/change arrays at time 0 is 0 and recorder
        initial_state.neutron_conc_change_array = np.zeros(self.num_points_radial)
        initial_state.neutrons_generated_per_step = 0

        # Ensure the initial neutron count is recorded
        initial_state.neutrons_in_sphere = (initial_state.neutron_conc_array[0:-1] * 
                                            self.get_shell_volumes_m3(dr_m = initial_state.dr_m)).sum()

         # ***** G Matrix Initialization *****
        A_initial, B_initial, C_initial, D_initial = self.make_matrix_coeffs(
            mean_free_path_transport_m = initial_state.mean_free_path_transport_m,
            neutron_diffusivity_m2pers = initial_state.neutron_diffusivity_m2pers,
            dr_m = initial_state.dr_m)
        
        # Formualtion for 1D Spherical Diffusion Using the Implicit Method
        # **********************************
        # G x neutron conc(time + dt) = neutron conc(time)
        # **********************************
        self.G = self.make_G_matrix(A = A_initial,
                                    B = B_initial,
                                    C = C_initial,
                                    D = D_initial,
                                    dr_m = initial_state.dr_m)
        
        # ***** H Matrix: Neutron Generation *****
        self.H = np.zeros(self.num_points_radial)

        # Inverse evolution matrix
        self.Ginv = np.linalg.inv(self.G)

        # Add initial states to states list 
        self.states.append(initial_state)