from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.Config import (
    STATE_SIZE,
    ACTION_SIZE
)
import time

def main(train_mode: bool, model_filepath: str = ""):

    if train_mode:
        dqn = DQN(STATE_SIZE, ACTION_SIZE, train_mode)
        dqn.train()
    else:
        dqn = DQN(STATE_SIZE, ACTION_SIZE, train_mode, model_filepath=model_filepath)
    
    # Set a way to kill training and save off state of network before closing


if __name__ == "__main__":
    input_mode = input("Training Mode? [Y\N] :")
    if input_mode.upper() == 'Y':
        train_mode = True
    else:
        train_mode = False
        input_file = input("Network Filepath To Load :")
    main(mode, input_file)