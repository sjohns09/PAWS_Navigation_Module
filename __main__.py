import os
import time
from datetime import datetime
from PAWS_Bot_Navigation.DQN import DQN
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.Plot import Plot
from PAWS_Bot_Navigation.Config import (
    STATE_SIZE,
    ACTION_SIZE,
    NUM_TEST_RUNS,
    PLOT_SAVE_FOLDER,
    NETWORK_SAVE_FOLDER
)


def main(train_mode: bool, model_name: str = ""):
    this_folder = os.path.dirname(os.path.abspath(__file__))

    if train_mode:
        dqn = DQN(STATE_SIZE, ACTION_SIZE, train_mode)
        dqn.train()
        print("TRAINING COMPLETE - NETWORK SAVED")
    else:
        success_plot = Plot("Success per Trial")
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")

        model_filepath = os.path.join(this_folder, f"{NETWORK_SAVE_FOLDER}", f"{model_name}")
        dqn = DQN(STATE_SIZE, ACTION_SIZE, train_mode, model_filepath=model_filepath)
        
        for t in range(NUM_TEST_RUNS):
            success = dqn.test(now_str)
            success_plot.add_point(t, int(success == True))
    
        success_plot.plot(os.path.join(this_folder, f"{PLOT_SAVE_FOLDER}", f"TEST_plot_success_{now_str}"))
    
    print("SIMULATION COMPLETE")


if __name__ == "__main__":
    input_mode = input("Training Mode? [Y or N] :")
    #input_mode = 'Y'
    if input_mode.upper() == 'Y':
        train_mode = True
        main(train_mode)
    else:
        train_mode = False
        input_file = input("Network File To Load :")
        main(train_mode, input_file)