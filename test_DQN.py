from PAWS_Bot_Navigation.Simulation import Simulation
import pytest
import time

class Test_Simulation():

    def test_simulation(self):
        sim_port = 19999
        sim = Simulation()
        sim.connect(sim_port)
        
        for i in range(0, 5):
            sim.reset()
            time.sleep(2)
        