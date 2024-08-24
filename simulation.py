from typing import List
import math
from model import Gadget
import time

class Simulation():

    def __init__(self, 
                 gadget_list: List[Gadget],
                 sim_duration_us: float=5,
                 dt_us: float=0.001,
                 stop_when_nuclei_limit_met=True) -> None:
        
        self.gadget_list = gadget_list
        self.sim_duration_us = sim_duration_us
        self.dt_us = dt_us
        self.stop_when_nuclei_limit_met = stop_when_nuclei_limit_met
        
        self.sim_steps = math.ceil(sim_duration_us / dt_us)
        print(f'Number of sim steps: {self.sim_steps}')

    
    def run_gadgets(self):

        for gadget in self.gadget_list:
            start_time_s = time.time()
            for step in range(self.sim_steps):
                gadget.run_sim_step()
                if self.stop_when_nuclei_limit_met == True:
                    if gadget.num_fissions_occured > (gadget.number_active_nuclei):
                        print('Stopped')
                        break
            end_time_s = time.time()
            print(f'Elapsed simulation time for gadget {gadget.name}: {1E3 * (end_time_s - start_time_s):.0f} ms')
            gadget.post_process()