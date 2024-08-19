import numpy as np
import math
from params import PhysicalParamsU235, PhysicalParamsPu239
import time
from constants import *

class Gadget(PhysicalParamsU235):
    
    @property
    def volume_m3(self):
        return (4 / 3) * math.pi * (self.initial_radius_cm / 100) ** 3
    
    
    @property
    def surface_area_m2(self):
        return 4 * math.pi * (self.initial_radius_cm / 100) ** 2
    

    @property
    def mass_kg(self):
        return self.volume_m3 * self.parameters.density_rho_kgperm3


    @property
    def number_active_nuclei(self) -> float:

        active_material_moles = (self.parameters.fraction_u235 * 
                                (self.mass_kg / 
                                (self.parameters.atomic_mass_kgpermol)))
        
        num_active_nuclei = active_material_moles * AVOGADRO

        return num_active_nuclei
    
    
    @property
    def num_fissions(self):

        # Assuming each fission event yields the number of neutrons in
        # parameter list
        return self.total_neutrons / self.parameters.neutrons_per_fission


    def set_initial_neutron_concentration(self,
                                          initial_neutron_conc_perm3=0):
        
        self.initial_neutron_conc_perm3 = initial_neutron_conc_perm3


    def initialize_matrix_coeffs(self):

        # Core Surface BC - "dN/dr = Z * N(r=R,t)", 
        # where Z = -(3/2) * (1/lambda_transport)
        
        Z = -(3 / 2) * (1 / self.parameters.mean_free_path_transport_m) 

        # Neutron Generation
        self.neutron_generation_rate = ((self.parameters.vel_neutron_mpers / 
                                         self.parameters.mean_free_path_fission_m) * 
                                        (self.parameters.neutrons_per_fission - 1))
        
        self.A = (self.parameters.neutron_diffusivity_m2pers * self.time_step_s) / (self.dr_m ** 2)
        self.B = (self.parameters.neutron_diffusivity_m2pers * self.time_step_s) / (self.dr_m) # MUST DIVIDE BY RADIUS IN MATRIX
        self.C = (3 * self.parameters.neutron_diffusivity_m2pers * self.time_step_s) / (self.dr_m ** 2)
        self.D = 2 * self.dr_m * Z


    def setup_matrices(self):

        # Formualtion for 1D Spherical Diffusion Using the Implicit Method
        # **********************************
        # G x neutron conc(time + dt) = neutron conc(time) + F + H
        # **********************************

        # Matrices are indexed by row, column

        # TODO: The below
        # Neutron volumetric concentration matrix
        # Each row is a concentration across the radius at a simulated time
        # self.neutron_conc_matrix = np.zeros((self.num_time_steps + 1, 
        #                                     self.num_points_radial))
        
        self.conc_list = []
        init_conc_array = (np.zeros(self.num_points_radial) + 
                           self.initial_neutron_conc_perm3)
        init_conc_array[0:2] = self.initial_neutron_burst_conc_perm3
        self.conc_list.append(init_conc_array)

        # Set initial concentration across radius
        #self.neutron_conc_matrix[0, :] = self.initial_neutron_conc_perm3
        
        # Neutron concetration in first shell - "initiator"
        #self.neutron_conc_matrix[0, 1:3] = self.initial_neutron_burst_conc_perm3 

        # ***** G Matrix: Evolution matrix *****
        self.G = np.zeros((self.num_points_radial, self.num_points_radial))

        # Main Diagonal
        for i in range(1, self.num_points_radial - 1):

            self.G[i, i] = 1 + 2 * self.A

        # Upper Diagonal
        for i in range(1, self.num_points_radial - 1):

            self.G[i, i + 1] = -self.A - self.B / (self.dr_m * i)

        # Lower Diagonal
        for i in range(1, self.num_points_radial - 1):

            self.G[i, i - 1] = -self.A + self.B / (self.dr_m * i)

        # First row
        self.G[0, 0] = 1 + 2 * self.C
        self.G[0, 1] = -2 * self.C

        # Bottom row
        self.G[-1, -2] = -2 * self.A
        self.G[-1, -1] = (-self.A * self.D + (self.B / (self.dr_m * self.num_points_radial)) * self.D + 1 + 2 * self.A)

        # ***** F Matrix: Surface BC *****
        self.F = np.zeros(self.num_points_radial)
        # Note - F is zeros in this formulation with dN/dr = -Z * N at the surface boundary

        # ***** H Matrix: Neutron Generation *****
        self.H = np.zeros(self.num_points_radial)

        # Inverse evolution matrix
        self.Ginv = np.linalg.inv(self.G)

    
    def post_process(self):
        '''
        This creates useful calculations
        '''
        # ***** Post processing for plotting and metrics *****
        self.neutron_conc_matrix = np.stack(self.conc_list, axis=0)

        # Create a matrix for the radius
        self.radius_points = np.linspace(0, self.initial_radius_cm, self.num_points_radial)

        # Create a simulation time array - one in sec, one in us
        self.sim_time_array_s = np.array([self.time_step_s * i for i in range(0, self.num_time_steps + 1)])
        self.sim_time_array_us = self.sim_time_array_s * 1E6

        # Estimate neutron concentration surface gradient | neutrons per m3 * 1 / m = neutrons / m4
        self.dNdr = (self.neutron_conc_matrix[:, -2] - 
                     self.neutron_conc_matrix[:, -1]) / self.dr_m

        # Estimate neutron surface flux | neutrons / (m2*s)
        self.surface_flux = self.parameters.neutron_diffusivity_m2pers * self.dNdr

        # Estimate neutrons leaving surface (both as per sim step and cumulative)
        self.neutrons_left = (self.surface_flux * 
                              self.surface_area_m2 * 
                              self.time_step_s)

        self.cumulative_neutrons_left = self.neutrons_left.cumsum()

        # Neutron counts in each shell - TODO: Check number of shells vs points
        radii = np.array([self.dr_m * i for i in range(0, self.num_points_radial)])

        shell_volumes_m3 = np.array([(4/3) * math.pi * (radii[i + 1] ** 3) - 
                                     (4/3) * math.pi * (radii[i] ** 3) 
                                     for i in range(0, len(radii) - 1)])
        
        # This multiplies the concentration at each time step times the shell volumes
        # It then sums all the neutron counts in the shells to get the total neturon 
        # counts in the sphere for each time step 
        self.neutrons_in_sphere = (self.neutron_conc_matrix[:, 0:-1] * shell_volumes_m3).sum(axis=1)

        # Total neutrons: starting, generated, and left
        self.total_neutrons = self.neutrons_in_sphere + self.cumulative_neutrons_left


    def run_sim_step(self) -> None:

        # Get the row of concentrations from the previous timestep
        #conc_prev = self.neutron_conc_matrix[step - 1, :].copy()
        conc_prev = self.conc_list[-1].copy()
        # Neutron generation: rate * time = neutrons / m3
        # We make this controllable for testing the diffusion code
        if self.neutron_multiplication_on:
            self.H = conc_prev * self.neutron_generation_rate * self.time_step_s

        # Add BC and neutron generation matrices
        temp = np.add(self.F, self.H)
        
        # Calcuate new neutron concentration values and overwrite conc matrix at
        # new step
        #self.neutron_conc_matrix[step, :] = np.dot(self.Ginv, np.add(conc_prev, temp))
        self.conc_list.append(np.dot(self.Ginv, np.add(conc_prev, temp)))

        self.num_time_steps += 1


    def __init__(self,
                 material: str='U235',
                 initial_radius_cm: float=5.0,
                 initial_neutron_conc=0,
                 initial_neutron_burst_conc_perm3=100,
                 time_step_s=1E-8,
                 num_time_steps=1000,
                 num_points_radial=100,
                 neutron_multiplication_on=True
                 ) -> None:

        self.material = material
        self.initial_radius_cm = initial_radius_cm
        self.initial_neutron_burst_conc_perm3 = initial_neutron_burst_conc_perm3
        self.time_step_s = time_step_s
        self.num_points_radial = num_points_radial
        self.neutron_multiplication_on = neutron_multiplication_on

        self.num_time_steps = 0

        # delta_radius, used in matrix math
        self.dr_m = (self.initial_radius_cm / 100) / (self.num_points_radial - 1)

        if self.material == 'U235':
            self.parameters = PhysicalParamsU235()
            print('Set up with U235')
        elif self.material == 'Pu239':
            self.params = PhysicalParamsPu239()
            print('Set up with Pu239')
        else:
            raise ValueError('Incorrect material specified')
        
         # TODO: simplify this or get rid of it if we dont need to set spatially varying neutron conc
        self.set_initial_neutron_concentration(initial_neutron_conc) 
        self.initialize_matrix_coeffs()
        self.setup_matrices()
        

        


    


        


