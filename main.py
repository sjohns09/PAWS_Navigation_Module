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
        for i in range(3):
            sim.move(Actions.FORWARD)
            sim.move(Actions.BACKWARD)
            sim.move(Actions.LEFT)
            sim.move(Actions.RIGHT)
    
    # Set a way to kill training and save off state of network before closing


if __name__ == "__main__":
    main()