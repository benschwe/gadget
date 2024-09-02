import numpy as np
import math
from params import PhysicalParamsU235, PhysicalParamsPu239
from constants import *

class Gadget(PhysicalParamsU235):

    @property
    def number_active_nuclei(self) -> float:

        active_material_moles = (self.parameters.fraction_u235 * 
                                (self.mass_kg / 
                                (self.parameters.atomic_mass_kgpermol)))
        
        num_active_nuclei = active_material_moles * AVOGADRO

        return num_active_nuclei
    

    @property
    def maximum_possible_energy_kt(self) -> float:

        return self.number_active_nuclei * self.parameters.energy_per_fission_j * J_TO_KILOTON
    
    
    def calc_nuclear_number_density_perm3(self,
                                          rho_kgperm3,
                                          ) -> float:
        '''Given a physical density (kg/m3), return
        the nuclear number density (active nuclei/m3)
        '''
        
        nuclear_number_density_per_m3 = ((rho_kgperm3 * AVOGADRO) / 
                                         (self.parameters.atomic_mass_kgpermol))
        
        return nuclear_number_density_per_m3
    

    def calc_initial_mass_kg(self,
                             initial_radius_m) -> float:
        
        '''Calculates the starting mass of the system
        given an initial radius. It uses the base density 
        from the parameters files.
        '''
        
        initial_volume_m3 = self.calc_volume_m3(radius_m=initial_radius_m)
        initial_mass_kg = initial_volume_m3 * self.parameters.density_rho_kgperm3

        return initial_mass_kg
    

    def calc_volume_m3(self,
                       radius_m) -> float:
        
        '''Returns the volumes (m3) given a radius (m)
        '''
        
        volume_m3 = (4 / 3) * math.pi * radius_m ** 3

        return volume_m3
    

    def calc_surface_area_m2(self,
                             radius_m) -> float:
        
        surface_area_m2 = 4 * math.pi * radius_m ** 2

        return surface_area_m2
    

    def calc_density_kgperm3(self,
                             radius_m) -> float:
        
        '''Returns the density (kg/m3) given a radius.
        We always assume no mass loss.
        '''
        
        density_kgperm3 = (self.mass_kg /
                           self.calc_volume_m3(radius_m=radius_m))
                           
        return density_kgperm3
    

    def calc_mean_free_path_fission_m(self, 
                                      nuclear_number_density_perm3=None) -> float:
        
        return (1 / (nuclear_number_density_perm3 * 
                     self.parameters.cross_section_fission_m2))
    

    def calc_mean_free_path_elastic_m(self,
                                      nuclear_number_density_perm3=None) -> float:
        
        return (1 / (nuclear_number_density_perm3 * 
                     self.parameters.cross_section_elastic_scattering_m2))
    

    def calc_mean_free_path_transport_m(self,
                                        nuclear_number_density_perm3=None) -> float:
        
        return (1 / ((1 / self.calc_mean_free_path_fission_m(nuclear_number_density_perm3=
                                                             nuclear_number_density_perm3)) +
                    (1 / self.calc_mean_free_path_elastic_m(nuclear_number_density_perm3=
                                                            nuclear_number_density_perm3))))
    

    def calc_neutron_diffusivity_m2pers(self,
                                        nuclear_number_density_perm3=None) -> float:
        
        return ((self.calc_mean_free_path_transport_m(nuclear_number_density_perm3=
                                                      nuclear_number_density_perm3) * 
                                                      self.parameters.vel_neutron_mpers) / 3)
    

    def calc_neutron_gen_rate_pers(self,
                                   mean_free_path_fission_m=None) -> float:
        
        return ((self.parameters.vel_neutron_mpers / mean_free_path_fission_m) * 
                (self.parameters.neutrons_per_fission - 1))
        
       
    def calc_shell_volumes_m3(self) -> np.array:

        '''Returns an array of volumes (m3) for each shell of the gadget
        '''

        radii = np.array([self.list_dr_m[-1] * i for i in range(0, self.num_points_radial)])

        shell_volumes_m3 = np.array([(4 / 3) * math.pi * (radii[i + 1] ** 3) - 
                                         (4 / 3) * math.pi * (radii[i] ** 3) 
                                         for i in range(0, len(radii) - 1)])
        
        return shell_volumes_m3
        

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

        # Get the array row of concentrations from the previous timestep
        conc_prev = self.conc_list[-1].copy()

        # Neutron generation: rate * time = neutrons / m3
        # We make this controllable for testing the diffusion code
        # Grab the latest neutron generation rate to use
        if self.neutron_multiplication_on:
            self.H = conc_prev * self.list_neutron_gen_rate_pers[-1] * self.time_step_s

        # Update A, B, C, and D coefficients
        A, B, C, D = self.make_matrix_coeffs(
            mean_free_path_transport_m = self.list_mean_free_path_transport_m[-1],
            neutron_diffusivity_m2pers = self.list_neutron_diffusivity_m2pers[-1],
            dr_m = self.list_dr_m[-1])
        
        # Make new G and Ginv matrices
        self.G = self.make_G_matrix(A = A,
                                    B = B,
                                    C = C,
                                    D = D,
                                    dr_m = self.list_dr_m[-1])
        
        self.Ginv = np.linalg.inv(self.G)
        
        # Calcuate new neutron concentration values and add it to the list
        self.conc_list.append(np.dot(self.Ginv, np.add(conc_prev, self.H)))

        # ***** Neutron accounting *****

        # Surface flux
        # Estimate neutron concentration surface gradient 
        # neutrons per m3 * 1 / m = neutrons / m4
        # Notations here: self.conc_list[-1] is the most recent concentration array.
        #                 self.conc_list[-1][-2] is the second to last element 
        #                 in the most recent
        #                 concentration array
        surface_conc_gradient_perm4 = (self.conc_list[-1][-2] - 
                                       self.conc_list[-1][-1]) / self.list_dr_m[-1]
        
        # Surface flux is the concentration gradient * last diffusivity entry
        surface_flux_perm2s = (surface_conc_gradient_perm4 * 
                               self.list_neutron_diffusivity_m2pers[-1])
        self.list_surface_flux_perm2s.append(surface_flux_perm2s)
        
        # Estimate neutrons leaving surface in the time step
        # Note - surface area changes with radius
        neutrons_left_surface = (surface_flux_perm2s * 
                                 self.calc_surface_area_m2(self.list_radius_m[-1]) *
                                 self.time_step_s)
        self.list_neutrons_left_surface.append(neutrons_left_surface)
        
        # Tally total neutrons that left the surface
        self.list_cumulative_neutrons_left_surface.append(
            self.list_cumulative_neutrons_left_surface[-1] + neutrons_left_surface)
        
        # Estimate neutrons in sphere at this timestep
        self.list_total_neutrons_in_sphere.append((self.conc_list[-1][0:-1] * 
                                                   self.calc_shell_volumes_m3()).sum())
        
        # Total neutron count - in sphere and total left surface
        self.list_total_neutrons.append(self.list_total_neutrons_in_sphere[-1] +
                                        self.list_cumulative_neutrons_left_surface[-1])
        
        # Total number of fissions
        self.list_total_number_of_fissions.append(self.list_total_neutrons[-1] / 
                                                  self.parameters.neutrons_per_fission)
        
        # Incremental fissions and fission rate (1/s)
        incremental_fissions = (self.list_total_number_of_fissions[-1] - 
                                self.list_total_number_of_fissions[-2])
        self.list_fission_rate_pers.append(incremental_fissions / self.time_step_s)
        
        # Total energy released due to fission of active nuclei
        self.list_total_energy_released_j.append(self.list_total_number_of_fissions[-1] * 
                                                 self.parameters.energy_per_fission_j)
        
        # Energy released in this timestep
        energy_released_j = (self.list_total_energy_released_j[-1] - 
                             self.list_total_energy_released_j[-2])
        
        # Power (W = J/s)
        self.list_heat_gen_w.append(energy_released_j / self.time_step_s)
        
        # Update pressure at this timestep
        # Uses gamma = 1/3 following B.C. Reed
        # Pressure = (gamma * total energy) / volume
        pressure_pa = (((1 / 3) * self.list_total_energy_released_j[-1]) / 
                      self.calc_volume_m3(radius_m=self.list_radius_m[-1]))
        self.list_pressure_pa.append(pressure_pa)

        # Update delta expansion velocity (m/s) and expansion vel list (m/s)
        delta_expansion_vel_mpers = ((4 * math.pi * ((self.list_radius_m[-1]) ** 2) * 
                                      (1 / 3) * self.list_total_energy_released_j[-1]) / 
                                      (self.calc_volume_m3(radius_m=self.list_radius_m[-1]) * 
                                      (self.mass_kg + self.tamper_mass_kg))) * self.time_step_s
        
        self.list_expansion_vel_mpers.append(self.list_expansion_vel_mpers[-1] + 
                                             delta_expansion_vel_mpers)

        # Update radius (m)
        delta_r_m = self.list_expansion_vel_mpers[-1] * self.time_step_s
        self.list_radius_m.append(self.list_radius_m[-1] + delta_r_m)

        # Update volume (m3)
        self.list_volume_m3.append(self.calc_volume_m3(self.list_radius_m[-1]))

        # Update density (kg/m3)
        self.list_density_kgperm3.append(self.mass_kg / self.list_volume_m3[-1])

        # ***** Update Nuclear Params *****

        # Update nuclear number density
        self.list_nuclear_number_density_perm3.append(self.calc_nuclear_number_density_perm3(
                                                      rho_kgperm3=self.list_density_kgperm3[-1]))

        # Update fission mean free path (m)
        self.list_mean_free_path_fission_m.append(self.calc_mean_free_path_fission_m(
                                                  self.list_nuclear_number_density_perm3[-1]))

        # Update elastic scattering mean free path (m)
        self.list_mean_free_path_elastic_m.append(self.calc_mean_free_path_elastic_m(
                                                  self.list_nuclear_number_density_perm3[-1]))

        # Update transport mean free path (m)
        self.list_mean_free_path_transport_m.append(self.calc_mean_free_path_transport_m(
                                                    self.list_nuclear_number_density_perm3[-1]))
        
        # Update neutron diffusivity (m2/s)
        self.list_neutron_diffusivity_m2pers.append(self.calc_neutron_diffusivity_m2pers(
                                                    self.list_nuclear_number_density_perm3[-1]))
        
         # Neutron generation rate
        self.list_neutron_gen_rate_pers.append(self.calc_neutron_gen_rate_pers(mean_free_path_fission_m=self.calc_mean_free_path_fission_m(nuclear_number_density_perm3
                                              =self.list_nuclear_number_density_perm3[-1])))
      
        # Increment the time steps elapsed
        self.num_sim_steps += 1


    def post_process(self):
        '''
        This creates useful simulation data for plotting
        '''
        # ***** Post processing for plotting and metrics *****
        self.neutron_conc_matrix = np.stack(self.conc_list, axis=0)

        # Create a matrix for the radius
        # Note - this is the initial radius!
        self.radius_points = np.linspace(0, self.initial_radius_cm, self.num_points_radial)

        # Create a simulation time array - one in sec, one in us
        self.sim_time_array_s = np.array([self.time_step_s * i for i in range(0, self.num_sim_steps + 1)])
        self.sim_time_array_us = self.sim_time_array_s * 1E6

        # Energy released array - kilotons
        self.array_total_energy_released_kt = np.array(self.list_total_energy_released_j) * J_TO_KILOTON

        # Radius array
        self.array_radius_m = np.array(self.list_radius_m)

        # Pressure array
        self.array_pressure_pa = np.array(self.list_pressure_pa)

        # Expansion velocity array
        self.array_expansion_vel_mpers = np.array(self.list_expansion_vel_mpers)

        # Fission rate array
        self.array_fission_rate_pers = np.array(self.list_fission_rate_pers)

        # Power array
        self.array_heat_gen_w = np.array(self.list_heat_gen_w)

    
    def get_elapsed_time_s(self):

        return self.time_step_s * self.num_sim_steps

     
    def __init__(self,
                id: int=0,
                material: str='U235',
                initial_radius_cm: float=7.0,
                initial_neutron_conc_perm3: float=0,
                initial_neutron_burst_conc_perm3: float=50000,
                time_step_s: float=1E-10,
                num_points_radial: int=100,
                neutron_multiplication_on: bool=True,
                tamper_mass_kg=0
                ) -> None:

        # ***** Static attributes - constant or initial values *****
        self.id = id
        self.material = material
        self.initial_radius_cm = initial_radius_cm
        self.initial_neutron_conc_perm3 = initial_neutron_conc_perm3
        self.initial_neutron_burst_conc_perm3 = initial_neutron_burst_conc_perm3
        self.time_step_s = time_step_s
        self.num_points_radial = num_points_radial
        self.neutron_multiplication_on = neutron_multiplication_on
        self.tamper_mass_kg = tamper_mass_kg

        # ***** Dynamic attributes/variables *****
        self.num_sim_steps = 0

        # Volume, density, diffusivity
        self.list_dr_m = []
        self.list_radius_m = []
        self.list_volume_m3 = []
        self.list_density_kgperm3 = []

        # Nuclear parameters
        self.list_mean_free_path_fission_m = []
        self.list_mean_free_path_elastic_m = []
        self.list_mean_free_path_transport_m = []
        self.list_neutron_diffusivity_m2pers = []
        self.list_nuclear_number_density_perm3 = []
        self.list_neutron_gen_rate_pers = []


        self.list_surface_flux_perm2s = [0]
        self.list_neutrons_left_surface = [0]
        self.list_cumulative_neutrons_left_surface = [0]
        self.list_total_neutrons_in_sphere = []
        self.list_total_neutrons = [0]
        self.list_total_number_of_fissions = [0]
        self.list_fission_rate_pers = [0]
        self.list_total_energy_released_j = [0]
        self.list_heat_gen_w = [0]
        self.list_pressure_pa = [100000]
        self.list_expansion_vel_mpers = [0]

        

        # **** Initilization ****
        if self.material == 'U235':
            self.parameters = PhysicalParamsU235()
        elif self.material == 'Pu239':
            self.parameters = PhysicalParamsPu239()
        else:
            raise ValueError('Incorrect material specified')
        
        # Radius (m)
        self.list_radius_m.append((initial_radius_cm / 100))
        
        # Mass (kg) - Using initial radius and initial density
        self.mass_kg = self.calc_initial_mass_kg(initial_radius_m = initial_radius_cm / 100)

        # Initialize density (kg/m3)
        self.list_density_kgperm3.append(self.parameters.density_rho_kgperm3)

        # Initialize volume (m3)
        self.list_volume_m3.append(self.calc_volume_m3(radius_m = initial_radius_cm / 100))
        
        # Initialize the starting delta radius
        self.list_dr_m.append((self.initial_radius_cm / 100) / (self.num_points_radial - 1))

        # Initialize nuclear number density (1/m3)
        self.list_nuclear_number_density_perm3.append(self.calc_nuclear_number_density_perm3(
                                                      rho_kgperm3=self.list_density_kgperm3[0]))

        # Initialize fission mean free path (m)
        self.list_mean_free_path_fission_m.append(self.calc_mean_free_path_fission_m(
                                                  self.list_nuclear_number_density_perm3[0]))

        # Initialize elastic scattering mean free path (m)
        self.list_mean_free_path_elastic_m.append(self.calc_mean_free_path_elastic_m(
                                                  self.list_nuclear_number_density_perm3[0]))

        # Initialize transport mean free path (m)
        self.list_mean_free_path_transport_m.append(self.calc_mean_free_path_transport_m(
                                                    self.list_nuclear_number_density_perm3[0]))
        
        # Initialize neutron diffusivity (m2/s)
        self.list_neutron_diffusivity_m2pers.append(self.calc_neutron_diffusivity_m2pers(
                                                    self.list_nuclear_number_density_perm3[0]))
        
         # Neutron generation
        self.list_neutron_gen_rate_pers.append(self.calc_neutron_gen_rate_pers(mean_free_path_fission_m=self.calc_mean_free_path_fission_m(nuclear_number_density_perm3
                                              =self.list_nuclear_number_density_perm3[0])))
        
        # Neutron volumetric concentration matrix
        # Each list entry is an array is a concentration across 
        # the radius at a simulated time
        self.conc_list = []
        init_conc_array = (np.zeros(self.num_points_radial) + 
                           self.initial_neutron_conc_perm3)
        
        # Initial neutron burst is in the first shell
        # TODO: Be able to specify a radius where the
        # burst happens, instead of fixed radii indicies
        init_conc_array[0:2] = self.initial_neutron_burst_conc_perm3
        self.conc_list.append(init_conc_array)

        # Ensure the initial neutron count is recorded
        self.list_total_neutrons_in_sphere.append((self.conc_list[-1][0:-1] * self.calc_shell_volumes_m3()).sum())

         # ***** G Matrix Initialization *****
        A_initial, B_initial, C_initial, D_initial = self.make_matrix_coeffs(
            mean_free_path_transport_m = self.list_mean_free_path_transport_m[0],
            neutron_diffusivity_m2pers = self.list_neutron_diffusivity_m2pers[0],
            dr_m = self.list_dr_m[0])
        
        # Formualtion for 1D Spherical Diffusion Using the Implicit Method
        # **********************************
        # G x neutron conc(time + dt) = neutron conc(time)
        # **********************************

        # Matrices are indexed by row, column
        self.G = self.make_G_matrix(A = A_initial,
                                    B = B_initial,
                                    C = C_initial,
                                    D = D_initial,
                                    dr_m = self.list_dr_m[0])
        

        # ***** H Matrix: Neutron Generation *****
        self.H = np.zeros(self.num_points_radial)

        # Inverse evolution matrix
        self.Ginv = np.linalg.inv(self.G)