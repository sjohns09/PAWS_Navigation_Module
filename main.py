from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Config import (
    STATE_SIZE,
    ACTION_SIZE
)
import time

def main():
    
    dqn = DQN(STATE_SIZE, ACTION_SIZE, True)
    dqn.train()
    '''
    initial = [0,0,0,0,sim.human_coords[0], sim.human_coords[1]]
    new_state, reward, done = sim.step(initial, Actions.FORWARD)
    print(f"{new_state}, {reward}, {done}")
    new_state, reward, done = sim.step(new_state, Actions.BACKWARD)
    print(f"{new_state}, {reward}, {done}")
    new_state, reward, done = sim.step(new_state, Actions.LEFT)
    print(f"{new_state}, {reward}, {done}")
    new_state, reward, done = sim.step(new_state, Actions.RIGHT)
    print(f"{new_state}, {reward}, {done}")
    '''
    # Set a way to kill training and save off state of network before closing


if __name__ == "__main__":
    main()