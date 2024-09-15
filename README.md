# gadget
Simulation of neutron multiplication in a gadget using the neutron diffusion equation  

## Usage
A gadget device is made by instantiating a gadget object. Initial arguments are passed to the constructor to define the initial parameters. Note that basic material and nuclear properties are automatically assigned by passing the material argument (e.g. "U235" or "Pu239").  

Once an object is created, a simulation step can be run by calling the run_sim_step() method. This method will execute a single simulation step lasting for a duration of time_step_s (defined in the gadget constructor).

## Basic Examples
* U235_Critical_Radius.ipynb
* U235_compression_ratio_example.ipynb


