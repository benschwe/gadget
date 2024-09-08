from dataclasses import dataclass
from constants import *

@dataclass
class PhysicalParamsU235:

    # Nuclear Parameters and Constants (U235)
    density_rho_kgperm3: float = 18710
    fraction_u235 = 1.0
    atomic_mass_kgpermol = 0.23504 # kg/mol
    cross_section_fission_m2: float = 1.235E-28 # m2
    cross_section_elastic_scattering_m2: float = 4.566E-28 # m2
    neutrons_per_fission: float = 2.637
    nuclear_number_density_perm3: float = 4.794E22 * 1E6 # 1/ - Density of U235 nuclei
    tau_s: float = 8.635e-9
    vel_neutron_mpers = 1.9561E7 # m/sec (Carey Sublette lists 1.4E7 as more reasonable due to scattering)
    energy_per_fission_mev = 173.0

    @property
    def energy_per_fission_j(self) -> float:
        return self.energy_per_fission_mev * MEV_TO_EV * EV_TO_J

    @property
    def nuclear_number_density_effective_perm3(self) -> float:
        return self.nuclear_number_density_perm3 * self.fraction_u235

    @property
    def mean_free_path_fission_m(self) -> float:
        return 1 / (self.cross_section_fission_m2 * 
                    self.nuclear_number_density_effective_perm3)
    
    @property
    def mean_free_path_elastic_m(self) -> float:
        return 1 / (self.cross_section_elastic_scattering_m2 * 
                    self.nuclear_number_density_effective_perm3)
    
    @property 
    def mean_free_path_transport_m(self) -> float:
        return 1 / (1 / self.mean_free_path_fission_m + 
                    1 / self.mean_free_path_elastic_m)
    
    @property
    def neutron_diffusivity_m2pers(self) -> float:
        return (self.mean_free_path_transport_m * self.vel_neutron_mpers) / 3 # m2/s (BC Reed)


@dataclass
class PhysicalParamsPu239:
    pass