import os
import time
from datetime import datetime
from PAWS_Bot_Navigation.Utilities import Plot
from PAWS_Bot_Navigation.DQN_CoppeliaSim import (
    DQN as csim_dqn,
    Config as csim_config
)
from PAWS_Bot_Navigation.DQN_Custom_NN import (
    DQN as cnn_dqn, 
    Config as cnn_config
)
from PAWS_Bot_Navigation.DQN_Gym import (
    DQN as gym_dqn, 
    Config as gym_config
)


def main(train_mode: bool, run_mode: str, model_name: str = ""):
    this_folder = os.path.dirname(os.path.abspath(__file__))

    if train_mode:
        if run_mode == '1':
            dqn = csim_dqn.DQN(csim_config.STATE_SIZE, csim_config.ACTION_SIZE, train_mode)
        elif run_mode == '2':
            dqn = gym_dqn.DQN(gym_config.STATE_SIZE, gym_config.ACTION_SIZE, train_mode)
        else:
            dqn = cnn_dqn.DQN(cnn_config.STATE_SIZE, cnn_config.ACTION_SIZE, train_mode)
        
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
    
    print("TEST SIMULATION COMPLETE")


if __name__ == "__main__":
    run_mode = input("Which Mode? CoppeliaSim[1] Gym[2] CustomNN[3] :")
    #input_mode = input("Training Mode? [Y or N] :")
    input_mode = 'Y'
    if input_mode.upper() == 'Y':
        train_mode = True
        main(train_mode, run_mode)
    else:
        train_mode = False
        input_file = input("Network File To Load :")
        main(train_mode, run_mode, input_file)