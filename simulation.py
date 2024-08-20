from typing import List
import math
from model import Gadget

class Simulation():

    def __init__(self, 
                 gadget_list: List[Gadget],
                 sim_duration_us: float=5,
                 dt_us: float=0.001) -> None:
        
        self.gadget_list = gadget_list
        self.sim_duration_us = sim_duration_us
        self.dt_us = dt_us
        
        self.sim_steps = math.ceil(sim_duration_us / dt_us)
        print(f'Number of sim steps: {self.sim_steps}')

    
    def run_gadgets(self):

        for gadget in self.gadget_list:
            for step in range(self.sim_steps):
                gadget.run_sim_step()
            gadget.post_process()