# gadget
Simulation of neutron multiplication in a gadget using the neutron diffusion equation  

## Usage
A gadget device is made by instantiating a Gadget object. Initial arguments are passed to the constructor to define the initial parameters. Note that basic material and nuclear properties are automatically assigned by passing the material argument (e.g. "U235" or "Pu239").  

`gadget = Gadget(material = 'U235', ...)`

Once an object is created, a simulation step can be run by calling the run_sim_step() method. This method will execute a single simulation step lasting for a duration of time_step_s (defined in the Gadget constructor).

`gadget.run_sim_step()`

Calling the post_process() method on a gadget will create a dataframe of results that can be accessed from the object.

`sim_data_df = gadget.output_df`

## Basic Examples
* [U235 Critical Radius](U235_Critical_Radius.ipynb)
* [Pu239 Ciritcal Radius](Pu239_Critcal_Radius.ipynb)
* [Detonation simulation with vary compression ratios](U235_compression_ratio_example.ipynb)


