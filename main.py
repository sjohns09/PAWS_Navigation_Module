from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.DQN import DQN
import time

def main():
    
    sim_port = 19999
    sim = Simulation()
    is_connected = sim.connect(sim_port)
    
    if is_connected:
        sim.move(Actions.FORWARD)
        sim.move(Actions.BACKWARD)
        sim.move(Actions.LEFT)
        sim.move(Actions.RIGHT)
    
    


if __name__ == "__main__":
    main()