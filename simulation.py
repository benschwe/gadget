from typing import List
import math
from model import Gadget
import time

class Simulation():

    def __init__(self, 
                 gadget_list: List[Gadget],
                 sim_duration_us: float=5,
                 stop_when_nuclei_limit_met=True) -> None:
        
        self.gadget_list = gadget_list
        self.sim_duration_us = sim_duration_us
        self.stop_when_nuclei_limit_met = stop_when_nuclei_limit_met
        
        # For now, we use the time step for the first gadget in the list
        # This silently implies that each gadget should have the same time_step_s
        self.sim_steps = math.ceil((sim_duration_us / 1E6) / gadget_list[0].time_step_s)
        print(f'Number of sim steps in all simulations: {self.sim_steps}')
        print('')

    
    def run_gadgets(self):

        for gadget in self.gadget_list:
            print(f'Running gadget ID: {gadget.id}')
            start_time_s = time.time()
            for step in range(self.sim_steps):
                gadget.run_sim_step()
                if self.stop_when_nuclei_limit_met == True:
                    if gadget.list_total_number_of_fissions[-1] > (gadget.number_active_nuclei):
                        print('Stopped')
                        break
            end_time_s = time.time()
            print(f'Elapsed simulation time for gadget {gadget.id}: {1E3 * (end_time_s - start_time_s):.0f} ms')
            gadget.post_process()
            dN_sphere_last_time = gadget.list_total_neutrons_in_sphere[-1] - gadget.list_total_neutrons_in_sphere[-2]
            if dN_sphere_last_time <= 0:
                print('Not critical')
            else:
                print('Critical')
            print('---------------')