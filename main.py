from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Simulation import Simulation
import time
from PAWS_Bot_Navigation.Actions import Actions

def main():
    sim_port = 19999
    sim = Simulation()
    is_connected = sim.connect(sim_port)
    
    if is_connected:
        for i in range(0, 2):
            state_bot = sim.get_postion(sim.paws_bot)
            state_human = sim.get_postion(sim.human)
            sim.move(Actions.FORWARD)
            time.sleep(4)
            print(f"State {i} Bot : {state_bot}")
            print(f"State {i} Human : {state_human}")
            sim.reset()
            time.sleep(4)


if __name__ == "__main__":
    main()