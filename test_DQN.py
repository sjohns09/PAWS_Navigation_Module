from DQN_Code.Simulation import Simulation
import pytest

class Test_Simulation():

    def test_simulation_connect(self):
        sim_port = 19999
        sim = Simulation(sim_port)