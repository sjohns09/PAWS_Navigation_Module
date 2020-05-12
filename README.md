# PAWS_Navigation_Module

The navigation system of the PAWS (Psychological Assistance and Wellness System) Bot is complex due to its need to operate in an unknown environment, the human’s home. The majority of common path planning algorithms require a map of the environment in order to plan effectively, which would not be available to this robot. The use of machine learning to help plan in these unknown environments is on the rise, and one promising technique is training via reinforcement learning. A reinforcement learning algorithm called DDQN (Double Deep Q-Network) was implemented to act as the navigation system for the PAWS Bot. The goals of training this algorithm were to teach the robot to reach a changing goal in a reasonable amount of time. To acheive this the algorithm was implemented via Python using CoppeliaSim as the simulation environment. Due to issues with CoppeliaSim, OpenAI Gym was also implemented to verify the success of the DDQN algorithm and reward function. The algorithm's success rate in the OpenAI Gym shows the feasibility of DDQN and the custom reward function for path planning, assuming many more training iterations could be executed.

## Setup
* Windows 10 OS
* Python 3.6.10

### Dependencies
* Clone the repo
* Install the requirements.txt file into the virtual environment of your choice

`pip install requirements.txt`

### To Use the Gym Simulation
* Clone the gym_maze repo https://github.com/MattChanTK/gym-maze.git
* Run the setup file to install the needed dependencies in your environment
```
cd gym-maze
python setup.py install
```
### To Use the CoppeliaSim Simulation
* Install CoppeliaSim https://www.coppeliarobotics.com/downloads
* The needed CoppeliaSim libraries to use the remote API are already included in the CoppeliaSim folder in this repo

## Config

There are 3 config files which can be modified for each run mode to change various hyperparameters or training parameters. The majority of the values can be left as is.

* For the CoppeliaSim and CustomNN modes the following field needs to be modified in their respective config files:

`PAWS_BOT_MODEL_PATH = "[USER_PATH]\PAWS_Bot_Navigation\CoppeliaSim\paws_bot_model.ttm"`

## Running
* To run the program use

`python -m PAWS_Bot_Navigation`

* The console window will prompt for user input

`Which Mode? CoppeliaSim[1] Gym[2] CustomNN[3]`

`Training Mode? [Y or N]`

* If training mode is 'N', test mode is enabled which allows a saved network to be loaded via file path
    * Don't use quotes around the path or it will error

`Network File To Load` 

### Training Mode
Runs through the DDQN training and outputs plots and saved networks to use later for testing. 
* Plots by default go to the `"saved_data"` folder
* Networks by default go to the `"saved_networks`" folder

### Testing Mode
Loads a `.h5` network file into the DDQN algorithm which allows predictions to be calculated using an already trained network. There is no weight updating in this mode.

## Modes

### CoppeliaSim
This mode uses CoppeliaSim to attempt to train the robot to reach a dynamic goal
* Open the environment file in CoppeliaSim
    * Full Size with Obstacles - `"~\PAWS_Bot_Navigation\CoppeliaSim\paws_robotSim.ttt"`
    * Small Size no Obstacles -`"~\PAWS_Bot_Navigation\CoppeliaSim\paws_robotSim_noObs.ttt"`
* Start the CoppeliaSim simulation
* Execute `PAWS_Bot_Navigation` in the appropriate mode, it will connect to the simulation and begin execution

There is a test network saved that can be loaded for testing of this mode `net_20200510_165416_e40.h5`

### Gym
This mode uses OpenAI Gym to train the robot to traverse a maze to a static goal
* Execute `PAWS_Bot_Navigation` in the appropriate mode which will start up the maze simulation and begin execution

There is a test network using the custom implemented reward function that can be loaded for testing `net_20200511_173028_e1000.h5`. As well as one using the standar reward function from the simulation `net_20200511_123004_e500.h5`

### CustomNN
This mode uses CoppeliaSim to attempt to train the robot to reach a dynamic goal. It utilizes a custom neural net opposed to the Keras and TensorFlow libraries

NOTE: This mode is not fully tested since this method was abandoned partially through development. Should run, but can't guarantee results.

* Ran in the same way as the CoppeliaSim mode, just select CustomNN mode at start up

## Documentation
* The `reports` folder contains reports and presentations that were given on this project
* The `videos` folder contains videos of previous testing and training runs

