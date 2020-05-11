import os
import time
from datetime import datetime
from PAWS_Bot_Navigation.Utilities.Plot import Plot
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


def main(train_mode: bool, run_mode: str, model_path: str = ""):
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
        success_plot = Plot("Success vs Trial")
        time_plot = Plot("Time per Success")
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")

        if run_mode == '1':
            dqn = csim_dqn.DQN(csim_config.STATE_SIZE, csim_config.ACTION_SIZE, train_mode, model_path)
            runs = csim_config.NUM_TEST_RUNS
            plot = csim_config.PLOT_SAVE_FOLDER
        elif run_mode == '2':
            dqn = gym_dqn.DQN(gym_config.STATE_SIZE, gym_config.ACTION_SIZE, train_mode, model_path)
            runs = gym_config.NUM_TEST_RUNS
            plot = os.path.join("DQN_Gym", gym_config.PLOT_SAVE_FOLDER)
        else:
            dqn = cnn_dqn.DQN(cnn_config.STATE_SIZE, cnn_config.ACTION_SIZE, train_mode, model_path)
            runs = cnn_config.NUM_TEST_RUNS
            plot = cnn_config.PLOT_SAVE_FOLDER
        
        success_count = 0
        for t in range(runs):
            success, final_time = dqn.test(now_str)
            success_plot.add_point(t, int(success == True))
            if success:
                success_count += 1
                time_plot.add_point(success_count, final_time)
        print(f"SUCCESS: {success_count}")
        time_plot.plot(os.path.join(this_folder, f"{plot}", f"TEST_plot_success_time_{now_str}"))
        success_plot.plot(os.path.join(this_folder, f"{plot}", f"TEST_plot_success_{now_str}"))
    
    print("TEST SIMULATION COMPLETE")


if __name__ == "__main__":
    #run_mode = input("Which Mode? CoppeliaSim[1] Gym[2] CustomNN[3] :")
    run_mode = '2'
    #input_mode = input("Training Mode? [Y or N] :")
    input_mode = 'N'
    if input_mode.upper() == 'Y':
        train_mode = True
        main(train_mode, run_mode)
    else:
        train_mode = False
        #input_file = input("Network File To Load :")
        input_file = "D:\Sammie\Documents\Grad School\Spring 2020 - RobotLearning\Project\PAWS_Bot_Navigation\DQN_Gym\saved_networks\\net_20200511_173028_e1000.h5"
        main(train_mode, run_mode, input_file)