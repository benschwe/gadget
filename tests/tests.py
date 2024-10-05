import pytest
from model.model import Gadget
import math

class TestGadget():

    def test_get_volume_m3(self, radius_m=0.9):

         assert Gadget.get_volume_m3(self, radius_m=radius_m) == (4 / 3) * math.pi * radius_m ** 3


    def test_get_surface_area_m2(self, radius_m=0.9):
         
         assert Gadget.get_surface_area_m2(self, radius_m=radius_m) == 4 * math.pi * radius_m ** 2
         

